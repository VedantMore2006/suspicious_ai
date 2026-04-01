import cv2
import os

file_path = "/home/vedant/suspicious_ai/saves/processed_video_20260331_000236.mkv"
if os.path.exists(file_path):
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ MKV file is valid and readable")
        print(f"  Frames: {frame_count}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Resolution: {width}x{height}")
        
        # Try reading first frame
        ret, frame = cap.read()
        if ret:
            print(f"✓ First frame read successfully")
            print(f"  Frame shape: {frame.shape}")
        cap.release()
    else:
        print("✗ Failed to open MKV file with OpenCV")
else:
    print("✗ File not found")
