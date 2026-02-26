from ultralytics import YOLO
import config


class Detector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.model.fuse()

    def detect(self, frame):
        results = self.model.track(
            frame,
            imgsz=config.IMG_SIZE,
            conf=config.CONFIDENCE,
            iou=config.IOU_THRESHOLD,
            classes=config.DETECTION_CLASSES,
            persist=True,
            tracker="bytetrack.yaml"
        )
        return results[0]