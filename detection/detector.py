from ultralytics import YOLO
import config


class Detector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.model.fuse()

    def detect(self, frame):
        # Track objects with configured thresholds and no console spam
        results = self.model.track(
            frame,
            imgsz=config.IMG_SIZE,
            conf=config.CONFIDENCE,
            iou=config.IOU_THRESHOLD,
            classes=config.DETECTION_CLASSES,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        return results[0]