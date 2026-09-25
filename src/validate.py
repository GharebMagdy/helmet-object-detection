from pathlib import Path
from ultralytics import YOLO


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    MODEL_PATH = BASE_DIR / "runs" / "detect" / "runs" / "detect" / "helmet_yolov8n" / "weights" / "best.pt"
    DATA_YAML = BASE_DIR / "data" / "data.yaml"

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        plots=True,
        save_json=False,
    )

    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)


if __name__ == "__main__":
    main()