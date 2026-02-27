import cv2
import numpy as np
from detection.detector import Detector
from config import CAMERA_SOURCE, SHOW_FPS, SAVE_CONFIDENCE, MOVEMENT_THRESHOLD, WINDOW_NAME, WINDOW_MODE, WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_SCALE, BOX_THICKNESS, FONT_SCALE, FONT_THICKNESS, FRAME_WIDTH, FRAME_HEIGHT
from utils.fps_tracker import FPSTracker
from utils.drawing import setup_window
from utils.event_logger import EventLogger
from utils.audio import AudioManager
from behavior.loitering import LoiteringDetector
from behavior.conflict_detection import ConflictDetector
from behavior.scoring import ThreatScorer
import os
import time
from datetime import datetime
from behavior.abandoned_object import AbandonedObjectDetector
import config
os.environ["QT_QPA_PLATFORM"] = "xcb"

def main():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    detector = Detector()
    loiter_detector = LoiteringDetector()
    event_logger = EventLogger()
    audio_manager = AudioManager()
    alarm_active = False
    last_abandon_detect_time = 0
    ALARM_STABILITY_BUFFER = 0.5  # seconds
    active_alert = None
    alert_start_time = 0
    
    # Setup window based on configuration
    setup_window()

    # Create automation folder
    save_dir = "saves"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    prev_time = time.time()
    fps = 0
    fps_tracker = FPSTracker(save_dir=save_dir)
    
    # Track object positions and save times
    object_positions = {}
    last_save_time = {}

    abandon_detector = AbandonedObjectDetector()
    conflict_detector = ConflictDetector()
    scorer = ThreatScorer()
    
    current_alert_state = "NONE"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        results = detector.detect(frame)
        
        boxes = results.boxes
        frame_copy = frame.copy()  # Copy for saving with bounding boxes
        
        tracked_objects = []
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()

            for i in range(len(ids)):
                x1, y1, x2, y2 = xyxy[i]
                obj_id = ids[i]
                cls = classes[i]

                tracked_objects.append({
                    "id": obj_id,
                    "class": cls,
                    "bbox": (x1, y1, x2, y2)
                })

        suspicious_ids = loiter_detector.update(tracked_objects)
        suspicious_bags = abandon_detector.update(tracked_objects)
        conflict_alert = conflict_detector.update(tracked_objects)
        instant_scores, session_scores = scorer.update(
            tracked_objects,
            suspicious_ids,
            suspicious_bags,
            conflict_alert,
        )

        current_time = time.time()

        # Determine new alert state based on priority
        new_alert_state = "NONE"

        if conflict_alert:
            new_alert_state = "CONFLICT"
        elif suspicious_bags:
            new_alert_state = "ABANDONED"

        # Only react if state changes
        if new_alert_state != current_alert_state:
            # Stop previous alarm if needed
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

        if boxes.id is not None:
            check_time = time.time()
            save_frame = False
            detected_objects = []

            for i in range(len(ids)):
                x1, y1, x2, y2 = xyxy[i]
                obj_id = ids[i]
                cls = classes[i]
                class_name = results.names[cls]

                if config.SAVE_FRAMES and confidences[i] > SAVE_CONFIDENCE:
                    # Calculate center position
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    current_pos = (center_x, center_y)

                    # Check for movement
                    is_moving = False
                    if obj_id in object_positions:
                        prev_pos = object_positions[obj_id]
                        distance = ((current_pos[0] - prev_pos[0])**2 +
                                   (current_pos[1] - prev_pos[1])**2)**0.5
                        if distance > MOVEMENT_THRESHOLD:
                            is_moving = True

                    # Update position
                    object_positions[obj_id] = current_pos

                    # Determine if we should save
                    if obj_id not in last_save_time:
                        last_save_time[obj_id] = 0

                    time_since_last_save = check_time - last_save_time[obj_id]

                    # Save logic: moving objects save immediately, stationary save once per second
                    if is_moving or time_since_last_save >= 1.0:
                        save_frame = True
                        last_save_time[obj_id] = check_time
                        if class_name not in detected_objects:
                            detected_objects.append(class_name)

                # Draw bounding box with object name
                color = (0, 255, 0)
                if obj_id in suspicious_ids:
                    color = (0, 0, 255)
                if obj_id in suspicious_bags:
                    color = (0, 0, 255)
                label = f"{class_name} ID:{obj_id}"

                if cls == config.PERSON and obj_id in instant_scores:
                    level = scorer.get_level(instant_scores[obj_id])
                    session_total = session_scores.get(obj_id, 0)
                    label += f" | Threat: {level} | Session: {session_total}"

                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, BOX_THICKNESS)
                cv2.putText(frame_copy, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS)

            # Save frame with bounding boxes if needed
            if config.SAVE_FRAMES and save_frame and detected_objects:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                objects_str = "_".join(detected_objects)
                filename = os.path.join(save_dir, f"detected_{objects_str}_{timestamp}.jpg")
                cv2.imwrite(filename, frame_copy)
                print(f"Saved: {filename}")
        
        # Calculate and display FPS
        if SHOW_FPS:
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            fps_tracker.update(fps)
            avg_fps = fps_tracker.get_average_fps()
            cv2.putText(frame_copy, f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)

        # Draw alert banner when active
        if config.SHOW_ALERT_BANNER and active_alert:
            if time.time() - alert_start_time < config.ALERT_BANNER_DURATION:
                overlay = frame_copy.copy()
                cv2.rectangle(overlay, (0, 0), (frame_copy.shape[1], 80), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.6, frame_copy, 0.4, 0, frame_copy)

                cv2.putText(
                    frame_copy,
                    active_alert,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    3
                )
            else:
                active_alert = None

        # Create extended canvas for camera preview + side panel
        panel_width = 350
        h, w, _ = frame_copy.shape
        extended_frame = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        
        # Copy camera preview to left side
        extended_frame[:, :w] = frame_copy
        
        # Draw dark panel background on right side
        extended_frame[:, w:] = (30, 30, 30)

        # Draw panel content on the right side
        panel_x = w  # Start of panel area
        margin_left = 15
        
        # Panel title
        y_offset = 35
        cv2.putText(
            extended_frame,
            "THREAT ANALYSIS",
            (panel_x + margin_left, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        
        # Separator line
        y_offset += 10
        cv2.line(
            extended_frame,
            (panel_x + margin_left, y_offset),
            (panel_x + panel_width - margin_left, y_offset),
            (100, 100, 100),
            1
        )
        
        # Active Threats section
        y_offset += 30
        cv2.putText(
            extended_frame,
            "ACTIVE THREATS:",
            (panel_x + margin_left, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )
        
        y_offset += 25
        
        # Display each person's threat score
        for pid, score in instant_scores.items():
            level = scorer.get_level(score)
            session_total = session_scores.get(pid, 0)
            
            # Determine color based on threat level
            color = (0, 255, 0)
            if level == "SUSPICIOUS":
                color = (0, 165, 255)
            elif level == "HIGH":
                color = (0, 0, 255)
            
            # ID label
            cv2.putText(
                extended_frame,
                f"ID {pid}:",
                (panel_x + margin_left + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )
            
            # Threat level
            cv2.putText(
                extended_frame,
                level,
                (panel_x + margin_left + 70, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
            
            # Current score
            cv2.putText(
                extended_frame,
                f"({score})",
                (panel_x + margin_left + 185, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )
            
            y_offset += 22
            
            # Session total on next line
            cv2.putText(
                extended_frame,
                f"  Session Total: {session_total}",
                (panel_x + margin_left + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 150),
                1,
            )
            
            y_offset += 28
        
        # Event Timeline section
        if event_logger.timeline:
            y_offset = max(y_offset + 20, h - 150)
            
            cv2.putText(
                extended_frame,
                "EVENT TIMELINE:",
                (panel_x + margin_left, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 150, 150),
                1,
            )
            
            y_offset += 25
            
            recent = event_logger.timeline[-3:]
            for _, msg in recent:
                # Truncate message if too long
                display_msg = msg[:30] + "..." if len(msg) > 30 else msg
                cv2.putText(
                    extended_frame,
                    f"• {display_msg}",
                    (panel_x + margin_left + 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                y_offset += 20
        
        # Use extended frame as the display frame
        frame_copy = extended_frame

        display_frame = frame_copy
        if DISPLAY_SCALE != 1.0:
            display_frame = cv2.resize(
                frame_copy,
                None,
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE
            )

        cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save FPS data and plot at the end of session
    fps_tracker.finalize()
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()