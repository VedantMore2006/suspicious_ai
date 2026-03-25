"""
One-time script to export yolov8n.pt → yolov8n.onnx at 320px input size.
Run once: python export_model.py
"""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=320, opset=12, simplify=True)
print("Export complete → yolov8n.onnx")
