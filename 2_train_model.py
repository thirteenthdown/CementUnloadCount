"""Fine-tune YOLO11n on custom cement bag dataset using Apple Silicon GPU."""
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    device="mps",
)

print("Training complete. Best weights saved to: runs/detect/train/weights/best.pt")
