from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt")

# Run detection on image
results = model("bus.jpg", save=True)
