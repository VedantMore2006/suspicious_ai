import cv2
from detection.detector import Detector
from config import CAMERA_SOURCE, SHOW_FPS, SAVE_CONFIDENCE, MOVEMENT_THRESHOLD, WINDOW_NAME, WINDOW_MODE, WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_SCALE, BOX_THICKNESS, FONT_SCALE, FONT_THICKNESS, FRAME_WIDTH, FRAME_HEIGHT
from utils.fps_tracker import FPSTracker
from utils.drawing import setup_window
from utils.event_logger import EventLogger
from utils.audio import AudioManager
from behavior.loitering import LoiteringDetector
from behavior.conflict_detection import ConflictDetector
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

        current_time = time.time()

        if suspicious_bags:
            last_abandon_detect_time = current_time

            if not alarm_active:
                if config.ENABLE_CONSOLE_LOG:
                    event_logger.log("abandon", "Abandoned object detected!", config.ALERT_COOLDOWN)
                audio_manager.start_alarm()
                alarm_active = True

            active_alert = "ABANDONED OBJECT DETECTED"
            alert_start_time = current_time

        else:
            if alarm_active:
                if (not conflict_alert) and current_time - last_abandon_detect_time > ALARM_STABILITY_BUFFER:
                    audio_manager.stop_alarm()
                    alarm_active = False
                    active_alert = None

        if conflict_alert:
            active_alert = "POSSIBLE PHYSICAL CONFLICT"
            alert_start_time = current_time

            if not alarm_active:
                audio_manager.start_alarm()
                alarm_active = True

            event_logger.log("conflict", "Possible physical conflict detected!", config.ALERT_COOLDOWN)

        if boxes.id is not None:
            check_time = time.time()
            save_frame = False
            detected_objects = []

            for i in range(len(ids)):
                x1, y1, x2, y2 = xyxy[i]
                obj_id = ids[i]
                cls = classes[i]
                class_name = results.names[cls]

                if confidences[i] > SAVE_CONFIDENCE:
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
                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, BOX_THICKNESS)
                label = f"{class_name} ID:{obj_id}"
                cv2.putText(frame_copy, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS)

            # Save frame with bounding boxes if needed
            if save_frame and detected_objects:
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