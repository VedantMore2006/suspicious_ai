import cv2
import numpy as np
import threading
import queue
import os
import time
from datetime import datetime

from detection.detector import Detector
from config import (CAMERA_SOURCE, SHOW_FPS, SAVE_CONFIDENCE, MOVEMENT_THRESHOLD,
                    WINDOW_NAME, WINDOW_MODE, WINDOW_WIDTH, WINDOW_HEIGHT,
                    DISPLAY_SCALE, BOX_THICKNESS, FONT_SCALE, FONT_THICKNESS,
                    FRAME_WIDTH, FRAME_HEIGHT)
from utils.fps_tracker import FPSTracker
from utils.drawing import setup_window
from utils.event_logger import EventLogger
from utils.audio import AudioManager
from behavior.loitering import LoiteringDetector
from behavior.conflict_detection import ConflictDetector
from behavior.scoring import ThreatScorer
from behavior.abandoned_object import AbandonedObjectDetector
import config

os.environ["QT_QPA_PLATFORM"] = "xcb"


def inference_worker(cap, detector, loiter_detector, abandon_detector,
                     conflict_detector, scorer, result_queue, stop_event):
    """
    Runs in a background thread.
    Captures frames, runs detection every DETECT_EVERY_N frames (reuses last
    result on skipped frames), runs behavior analysis on every frame, and
    pushes results into result_queue (maxsize=1, drop-on-full so display always
    gets the freshest result).
    """
    frame_count = 0
    last_tracked_objects = []

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        if frame_count % config.DETECT_EVERY_N == 0:
            results = detector.detect(frame)
            boxes = results.boxes
            tracked_objects = []

            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                for i in range(len(ids)):
                    x1, y1, x2, y2 = xyxy[i]
                    tracked_objects.append({
                        "id": ids[i],
                        "class": classes[i],
                        "bbox": (x1, y1, x2, y2),
                        "conf": float(confidences[i]),
                        "name": results.names[classes[i]],
                    })

            last_tracked_objects = tracked_objects
        else:
            tracked_objects = last_tracked_objects

        frame_count += 1

        suspicious_ids = loiter_detector.update(tracked_objects)
        suspicious_bags = abandon_detector.update(tracked_objects)
        conflict_alert = conflict_detector.update(tracked_objects)
        instant_scores, session_scores = scorer.update(
            tracked_objects, suspicious_ids, suspicious_bags, conflict_alert
        )

        result = {
            "frame": frame,
            "tracked_objects": tracked_objects,
            "suspicious_ids": suspicious_ids,
            "suspicious_bags": suspicious_bags,
            "conflict_alert": conflict_alert,
            "instant_scores": instant_scores,
            "session_scores": session_scores,
        }

        # Drop stale result if display hasn't consumed it yet; always keep freshest
        try:
            result_queue.get_nowait()
        except queue.Empty:
            pass
        result_queue.put(result)


def main():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    detector = Detector()
    loiter_detector = LoiteringDetector()
    abandon_detector = AbandonedObjectDetector()
    conflict_detector = ConflictDetector()
    scorer = ThreatScorer()
    event_logger = EventLogger()
    audio_manager = AudioManager()

    setup_window()
    save_dir = "saves"
    os.makedirs(save_dir, exist_ok=True)

    fps_tracker = FPSTracker(save_dir=save_dir)
    object_positions = {}
    last_save_time = {}

    active_alert = None
    alert_start_time = 0
    current_alert_state = "NONE"
    prev_time = time.time()

    result_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    worker = threading.Thread(
        target=inference_worker,
        args=(cap, detector, loiter_detector, abandon_detector,
              conflict_detector, scorer, result_queue, stop_event),
        daemon=True,
    )
    worker.start()

    last_result = None

    while True:
        try:
            last_result = result_queue.get(timeout=0.1)
        except queue.Empty:
            if last_result is None:
                continue  # Nothing to show yet

        result = last_result
        frame = result["frame"]
        tracked_objects = result["tracked_objects"]
        suspicious_ids = result["suspicious_ids"]
        suspicious_bags = result["suspicious_bags"]
        conflict_alert = result["conflict_alert"]
        instant_scores = result["instant_scores"]
        session_scores = result["session_scores"]

        frame_copy = frame.copy()
        current_time = time.time()

        # Alert state machine
        new_alert_state = "NONE"
        if conflict_alert:
            new_alert_state = "CONFLICT"
        elif suspicious_bags:
            new_alert_state = "ABANDONED"

        if new_alert_state != current_alert_state:
            if current_alert_state != "NONE":
                audio_manager.stop_alarm()

            if new_alert_state == "CONFLICT":
                active_alert = "POSSIBLE PHYSICAL CONFLICT"
                alert_start_time = current_time
                audio_manager.start_alarm()
                event_logger.log("conflict", "Possible physical conflict detected!", config.ALERT_COOLDOWN)
            elif new_alert_state == "ABANDONED":
                active_alert = "ABANDONED OBJECT DETECTED"
                alert_start_time = current_time
                audio_manager.start_alarm()
                event_logger.log("abandon", "Abandoned object detected!", config.ALERT_COOLDOWN)
            else:
                active_alert = None

            current_alert_state = new_alert_state

        # Draw bounding boxes
        check_time = time.time()
        save_frame = False
        detected_objects = []

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            obj_id = obj["id"]
            cls = obj["class"]
            class_name = obj["name"]
            conf = obj["conf"]

            color = (0, 255, 0)
            if obj_id in suspicious_ids or obj_id in suspicious_bags:
                color = (0, 0, 255)

            label = f"{class_name} ID:{obj_id}"
            if cls == config.PERSON and obj_id in instant_scores:
                level = scorer.get_level(instant_scores[obj_id])
                session_total = session_scores.get(obj_id, 0)
                label += f" | Threat: {level} | Session: {session_total}"

            cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, BOX_THICKNESS)
            cv2.putText(frame_copy, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS)

            if config.SAVE_FRAMES and conf > SAVE_CONFIDENCE:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                current_pos = (center_x, center_y)
                is_moving = False
                if obj_id in object_positions:
                    prev_pos = object_positions[obj_id]
                    dist = ((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2) ** 0.5
                    if dist > MOVEMENT_THRESHOLD:
                        is_moving = True
                object_positions[obj_id] = current_pos
                last_save_time.setdefault(obj_id, 0)
                if is_moving or check_time - last_save_time[obj_id] >= 1.0:
                    save_frame = True
                    last_save_time[obj_id] = check_time
                    if class_name not in detected_objects:
                        detected_objects.append(class_name)

        if config.SAVE_FRAMES and save_frame and detected_objects:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(save_dir, f"detected_{'_'.join(detected_objects)}_{timestamp}.jpg")
            cv2.imwrite(filename, frame_copy)

        # FPS counter
        if SHOW_FPS:
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time
            fps_tracker.update(fps)
            avg_fps = fps_tracker.get_average_fps()
            cv2.putText(frame_copy, f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)

        # Alert banner
        if config.SHOW_ALERT_BANNER and active_alert:
            if current_time - alert_start_time < config.ALERT_BANNER_DURATION:
                overlay = frame_copy.copy()
                cv2.rectangle(overlay, (0, 0), (frame_copy.shape[1], 80), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.6, frame_copy, 0.4, 0, frame_copy)
                cv2.putText(frame_copy, active_alert, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            else:
                active_alert = None

        # Side panel
        panel_width = 350
        h, w, _ = frame_copy.shape
        extended_frame = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        extended_frame[:, :w] = frame_copy
        extended_frame[:, w:] = (30, 30, 30)

        panel_x = w
        margin_left = 15
        y_offset = 35

        cv2.putText(extended_frame, "THREAT ANALYSIS", (panel_x + margin_left, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 10
        cv2.line(extended_frame,
                 (panel_x + margin_left, y_offset),
                 (panel_x + panel_width - margin_left, y_offset),
                 (100, 100, 100), 1)
        y_offset += 30

        cv2.putText(extended_frame, "ACTIVE THREATS:", (panel_x + margin_left, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_offset += 25

        for pid, score in instant_scores.items():
            level = scorer.get_level(score)
            session_total = session_scores.get(pid, 0)
            color = (0, 255, 0)
            if level == "SUSPICIOUS":
                color = (0, 165, 255)
            elif level == "HIGH":
                color = (0, 0, 255)

            cv2.putText(extended_frame, f"ID {pid}:", (panel_x + margin_left + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(extended_frame, level, (panel_x + margin_left + 70, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(extended_frame, f"({score})", (panel_x + margin_left + 185, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y_offset += 22
            cv2.putText(extended_frame, f"  Session Total: {session_total}",
                        (panel_x + margin_left + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            y_offset += 28

        if event_logger.timeline:
            y_offset = max(y_offset + 20, h - 150)
            cv2.putText(extended_frame, "EVENT TIMELINE:", (panel_x + margin_left, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y_offset += 25
            for _, msg in event_logger.timeline[-3:]:
                display_msg = msg[:30] + "..." if len(msg) > 30 else msg
                cv2.putText(extended_frame, f"• {display_msg}", (panel_x + margin_left + 5, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20

        display_frame = extended_frame
        if DISPLAY_SCALE != 1.0:
            display_frame = cv2.resize(extended_frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

        cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_event.set()
    worker.join(timeout=2)
    fps_tracker.finalize()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
