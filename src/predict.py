from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "runs" / "detect" / "helmet_yolov8n" / "weights" / "best.pt"


def load_model():
    model = YOLO(str(MODEL_PATH))
    model.names[0] = "No Helmet"   # كان "head"
    # model.names[1] فاضل "helmet" زي ما هو
    return model


def predict_image(image_path, conf=0.4):
    model = load_model()
    results = model.predict(
        source=image_path,
        conf=conf,
        classes=[0, 1],   # نتجاهل person (index=2) من الأساس
        save=True,
    )

    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            print(f"لقينا: {cls_name} بثقة {confidence:.2f}")

    return results


if __name__ == "__main__":
    test_folder = BASE_DIR / "test_images"
    images = list(test_folder.glob("*.jpg")) + list(test_folder.glob("*.png"))

    for img in images:
        print(f"\n--- {img.name} ---")
        predict_image(str(img))