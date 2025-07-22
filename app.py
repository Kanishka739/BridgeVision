import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import io
import base64

# Set page config and title
st.set_page_config(page_title="BridgeGuard AI", page_icon="🌉", layout="wide")

# Inject custom CSS for fonts, dark mode, styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
    }

    .main-container {
        background-color: var(--background-color);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        transition: background-color 0.3s ease;
    }

    .title-text {
        color: var(--primary-color);
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }

    .footer {
        font-size: 0.9rem;
        color: gray;
        text-align: center;
        margin-top: 3rem;
    }

    /* Light mode variables */
    :root {
        --background-color: #f9fafb;
        --primary-color: #0b3d91;
    }

    /* Dark mode variables */
    [data-theme="dark"] {
        --background-color: #121212;
        --primary-color: #4ea8ff;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Dark mode toggle in sidebar
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

st.sidebar.title("⚙️ Settings")
st.sidebar.checkbox("Dark Mode", value=st.session_state.dark_mode, on_change=toggle_dark_mode)

if st.session_state.dark_mode:
    st.markdown('<body data-theme="dark">', unsafe_allow_html=True)
else:
    st.markdown('<body data-theme="light">', unsafe_allow_html=True)

# App Title & description
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<h1 class="title-text">🌉 BridgeVision</h1>', unsafe_allow_html=True)
st.markdown("Upload a bridge image and get real-time damage detection powered by YOLOv8.")

# Sidebar instructions
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Upload a clear image of a bridge.
    2. Adjust the confidence threshold.
    3. View detection results and confidence scores.
    4. Download annotated image.
    """)
    st.write("---")

# Load model once
model_path = "C:/Users/Kanishka/Desktop/vscode/DRDO/runs/detect/train2/weights/best.pt"
model = YOLO(model_path)

# History in session state (store list of dicts with image and result)
if "history" not in st.session_state:
    st.session_state.history = []

# Confidence threshold slider
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# File uploader
uploaded_file = st.file_uploader("Upload Bridge Image", type=["jpg", "jpeg", "png"])

def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def generate_download_link(img):
    img_bytes = image_to_bytes(img)
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="annotated.png">⬇️ Download Annotated Image</a>'
    return href

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Detecting damages..."):
        img_np = np.array(image)
        results = model.predict(img_np, conf=conf_threshold)
        annotated = results[0].plot()
        annotated_pil = Image.fromarray(annotated)

    # Show side-by-side images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(annotated_pil, caption="Detected Damage", use_container_width=True)

    # Extract detected classes and confidences
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.subheader("Detections")
        data = []
        for box in boxes:
            cls_id = int(box.cls.cpu())
            conf = float(box.conf.cpu())
            name = model.names[cls_id] if cls_id in model.names else str(cls_id)
            data.append({"Class": name, "Confidence": f"{conf:.2f}"})
        st.table(data)
    else:
        st.info("No damages detected with current confidence threshold.")

    # Download button for annotated image
    st.markdown(generate_download_link(annotated_pil), unsafe_allow_html=True)

    # Save history (store image bytes and detection summary)
    st.session_state.history.append(
        {
            "original": image,
            "annotated": annotated_pil,
            "detections": data if boxes is not None else [],
        }
    )

# Show history of last 3 uploads
if len(st.session_state.history) > 0:
    st.markdown("---")
    st.subheader("🕘 History (Last uploads)")
    for i, item in enumerate(st.session_state.history[-3:][::-1], 1):
        cols = st.columns(3)
        cols[0].image(item["original"], caption=f"Original #{i}", use_container_width=True)
        cols[1].image(item["annotated"], caption=f"Annotated #{i}", use_container_width=True)
        if item["detections"]:
            cols[2].write("Detections:")
            for det in item["detections"]:
                cols[2].write(f"{det['Class']} ({det['Confidence']})")
        else:
            cols[2].write("No detections")

st.markdown('<div class="footer">© 2025 Kanishka Jain | BridgeVision AI Project</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
