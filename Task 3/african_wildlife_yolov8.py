from ultralytics import YOLO

# Load base model
model = YOLO("yolov8n.pt")  # or 'yolov8s.pt', 'yolov8m.pt' etc.

# Train the model
model.train(
    data="C:/Users/vyshu/african-wildlife.yaml",
    epochs=50,
    imgsz=640
)

# Load best weights after training
model = YOLO("runs/detect/train/weights/best.pt")

# Validate the model (on validation set)
metrics = model.val()

# Run predictions on test set
results = model.predict(
    source="C:/Users/vyshu/datasets/african-wildlife/images/test",
    save=True
)

# Display metrics
print("Validation Results:")
print(metrics)

print("Test predictions saved in folder:")
print(model.predictor.save_dir)
