from pathlib import Path
from ultralytics import YOLO

def main():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_YAML = BASE_DIR / "data" / "data.yaml"

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        project="runs/detect",
        name="helmet_yolov8n",
        patience=10,
        pretrained=True,
        cache=False,
    )

if __name__ == "__main__":
    main()