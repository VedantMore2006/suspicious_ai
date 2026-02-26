import cv2
from detection.detector import Detector
from config import CAMERA_SOURCE, SHOW_FPS, SAVE_CONFIDENCE, MOVEMENT_THRESHOLD
import os
import time
from datetime import datetime 

os.environ["QT_QPA_PLATFORM"] = "xcb"

def main():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    detector = Detector()
    
    # Create automation folder
    save_dir = "saves"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    prev_time = time.time()
    fps = 0
    
    # Track object positions and save times
    object_positions = {}
    last_save_time = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        results = detector.detect(frame)
        
        boxes = results.boxes
        frame_copy = frame.copy()  # Copy for saving with bounding boxes
        
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            
            check_time = time.time()
            save_frame = False
            detected_objects = []
            
            for i in range(len(ids)):
                if confidences[i] > SAVE_CONFIDENCE:
                    x1, y1, x2, y2 = xyxy[i]
                    obj_id = ids[i]
                    cls = classes[i]
                    class_name = results.names[cls]
                    
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
                    
                    # Draw bounding box
                    cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.putText(frame_copy, f"ID {obj_id}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
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
            cv2.putText(frame_copy, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Suspicious Behavior Detector", frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()