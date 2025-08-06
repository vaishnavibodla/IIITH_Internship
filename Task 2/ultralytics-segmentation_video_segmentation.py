from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
results = model("sample.mp4", save=True)
