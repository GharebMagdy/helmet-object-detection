import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)


# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "runs" / "detect" / "helmet_yolov8n" / "weights" / "best.pt"


# =========================
# Title
# =========================

st.title("🪖 Helmet Detection System")
st.write("YOLOv8-based Worker Helmet Detection")


# =========================
# Check Model
# =========================

if not MODEL_PATH.exists():

    st.warning(
        "The trained model (best.pt) is not available yet. "
        "Please wait until YOLOv8 training finishes."
    )

    st.info(
        f"Expected model location:\n{MODEL_PATH}"
    )

    st.stop()


# =========================
# Load Model
# =========================

@st.cache_resource
def load_model():
    model = YOLO(str(MODEL_PATH))
    model.names[0] = "No Helmet"   # كان "head"، خليناه أوضح
    return model


model = load_model()


# =========================
# Confidence Slider
# =========================

conf_threshold = st.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)


# =========================
# Upload Image
# =========================

uploaded_file = st.file_uploader(
    "Upload a worker image",
    type=["jpg", "jpeg", "png"]
)


# =========================
# Detection
# =========================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")

    st.image(
        image,
        width=700
    )

    if st.button("🔍 Detect Helmet"):

        results = model.predict(
            source=image,
            conf=conf_threshold,
            classes=[0, 1]   # نتجاهل كلاس person (index=2) من الأساس
        )

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            st.warning("No detections found in this image.")

        else:
            result_image = results[0].plot()

            st.subheader("Detection Result")

            st.image(
                result_image,
                channels="BGR",
                width=700
            )

            # =========================
            # Detection Statistics
            # =========================

            helmet_count = 0
            no_helmet_count = 0

            for cls in boxes.cls:

                class_id = int(cls)

                if class_id == 0:
                    no_helmet_count += 1

                elif class_id == 1:
                    helmet_count += 1

            # =========================
            # Dashboard Metrics
            # =========================

            st.subheader("Detection Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🪖 Helmets", helmet_count)

            with col2:
                st.metric("⚠️ No Helmet", no_helmet_count)

            with col3:
                if no_helmet_count > 0:
                    st.metric(
                        "🚨 Status",
                        "VIOLATION",
                        delta=f"{no_helmet_count} unsafe",
                        delta_color="inverse"
                    )
                else:
                    st.metric("✅ Status", "SAFE")