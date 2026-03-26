from ultralytics import YOLO
import numpy as np
import config


class Detector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = config.MODEL_PATH
        self.model_path = str(model_path)
        self.model = YOLO(model_path)
        self.is_pose = "pose" in self.model_path
        # fuse() only applies to PyTorch models
        if not self.model_path.endswith(".onnx"):
            self.model.fuse()
        # Pose model only detects persons; detection model uses full class list
        self._classes = [config.PERSON] if self.is_pose else config.DETECTION_CLASSES

    def detect(self, frame):
        results = self.model.track(
            frame,
            imgsz=config.IMG_SIZE,
            conf=config.CONFIDENCE,
            iou=config.IOU_THRESHOLD,
            classes=self._classes,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        return results[0]

    def parse_tracked_objects(self, results):
        """
        Extract tracked objects + keypoints (when pose model) from a results object.
        Returns a list of dicts ready for behavior detectors.
        """
        boxes = results.boxes
        if boxes.id is None:
            return []

        ids         = boxes.id.cpu().numpy().astype(int)
        xyxy        = boxes.xyxy.cpu().numpy()
        classes     = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        # Keypoints: shape (N, 17, 2) and (N, 17) — only present for pose model
        kp_xy   = None
        kp_conf = None
        if self.is_pose and results.keypoints is not None:
            kp_xy   = results.keypoints.xy.cpu().numpy()    # (N, 17, 2)
            kp_conf = results.keypoints.conf.cpu().numpy()  # (N, 17)

        objects = []
        for i in range(len(ids)):
            x1, y1, x2, y2 = xyxy[i]
            objects.append({
                "id":        ids[i],
                "class":     classes[i],
                "bbox":      (x1, y1, x2, y2),
                "conf":      float(confidences[i]),
                "name":      results.names[classes[i]],
                "keypoints": kp_xy[i]   if kp_xy   is not None else None,  # (17, 2)
                "kp_conf":   kp_conf[i] if kp_conf is not None else None,  # (17,)
            })
        return objects
