#Before this,need to takw a video and segment it into frames.Then go to roboflow and label all the objects and then using yolov8 we are going to train the model

from ultralytics import YOLO

# Load a YOLOv8 model (Nano version for speed — you can change to yolov8s.pt, yolov8m.pt, etc.)
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data=r"C:/Users/Vaishnavi/Downloads/Vehicle Object Detection Project/Yolov8/data.yaml",  # path to data.yaml
    epochs=50,       # number of training epochs
    imgsz=640        # image size
)
