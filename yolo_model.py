from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Train
model.train(
    data=r"C:\Users\chaka\Preethu\My_Git_Repo\Aerial_Project5\data\Detection_dataset\data.yaml",
    epochs=50,
    imgsz=640
)