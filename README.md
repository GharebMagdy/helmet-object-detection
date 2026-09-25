# 🪖 Helmet Detection System

A computer vision system that detects whether construction/industrial workers are wearing safety helmets, using a fine-tuned YOLOv8 model and an interactive Streamlit dashboard.

## 📋 Overview

This project fine-tunes a YOLOv8n (Nano) object detection model to identify helmet compliance from images. It was trained on a custom-labeled dataset and deployed through a simple web dashboard that lets users upload an image and instantly see detection results with safety statistics.

## ✨ Features

- **Real-time object detection** powered by a YOLOv8 model fine-tuned specifically for helmet detection
- **Interactive web dashboard** (Streamlit) — upload an image and get instant results
- **Adjustable confidence threshold** via slider
- **Safety statistics**: automatic count of helmets vs. non-compliant workers, with a clear SAFE / VIOLATION status
- **GPU-accelerated inference** (CUDA)

## 🗂 Dataset

- **Source:** [Hard Hat Workers Dataset](https://universe.roboflow.com/shdv1/hard-hat-workers-aibtb) (Roboflow Universe)
- **Size:** 16,328 images
- **Classes:** `head` (labeled as "No Helmet" in the app), `helmet`, `person`
- **Split:** Train / Validation / Test
- **Format:** YOLO (bounding boxes, normalized coordinates)

> **Note:** The `person` class was excluded from the final model's predictions due to severe class imbalance (only 1,344 instances vs. 41,044 for `helmet`), which caused near-zero detection accuracy (AP = 0.034) for that class. The model still detects `head` and `helmet` reliably.

## 🧠 Model & Training

| Detail | Value |
|---|---|
| Base model | YOLOv8n (pretrained, fine-tuned) |
| Framework | Ultralytics YOLO |
| Epochs | 50 |
| Image size | 640×640 |
| Batch size | 4 |
| Hardware | NVIDIA RTX 3050 Laptop GPU (4GB) |

### Results (final evaluation)

| Metric | head ("No Helmet") | helmet |
|---|---|---|
| AP (Average Precision) | 0.960 | 0.980 |
| Precision (validation, class-wise) | 95% | 96% |

Overall mAP50: **0.658** · mAP50-95: **0.447** (dragged down by the excluded `person` class; head/helmet performance individually is strong)

Best F1-confidence balance found at **confidence threshold ≈ 0.4**, used as the dashboard default.

## 🖥️ Project Structure

```
helmet-object-detection/
├── data/                   # Dataset (train/valid/test) — not tracked in git
├── dashboard/
│   └── app.py              # Streamlit web dashboard
├── src/
│   ├── train.py            # Model training script
│   ├── validate.py         # Model evaluation script
│   └── predict.py          # Standalone inference script
├── runs/                   # Training outputs & model weights — not tracked in git
├── requirements.txt
└── README.md
```

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd helmet-object-detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   ```

3. **Install PyTorch with CUDA support** (adjust the CUDA version for your GPU)
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

4. **Install remaining dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download the dataset** from [Roboflow](https://universe.roboflow.com/shdv1/hard-hat-workers-aibtb) (YOLOv8 format) and place it in `data/`.

## 🚀 Usage

### Train the model
```bash
python src\train.py
```

### Evaluate the model
```bash
python src\validate.py
```

### Run standalone inference on test images
```bash
python src\predict.py
```

### Launch the dashboard
```bash
streamlit run dashboard\app.py
```
Then open `http://localhost:8501` in your browser.

## ⚠️ Limitations

- The `person` class is not usable due to insufficient training examples
- Model trained on a relatively small GPU (4GB VRAM) using YOLOv8n (the smallest variant) — accuracy could improve with a larger model (YOLOv8s/m) or more training data
- Performance may vary on images with poor lighting, heavy occlusion, or unusual camera angles not well-represented in the training set

## 🔮 Future Improvements

- Retrain without the `person` class, or collect more `person` examples to fix the imbalance
- Add support for video/live camera feed detection
- Deploy as a standalone web service (FastAPI backend + frontend)
- Experiment with larger YOLOv8 variants or additional data augmentation

## 🛠️ Tech Stack

- **Model:** YOLOv8 (Ultralytics)
- **Deep Learning:** PyTorch (CUDA)
- **Dashboard:** Streamlit
- **Dataset annotation/hosting:** Roboflow

## 📄 License

This project is for educational purposes.
