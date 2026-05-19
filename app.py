import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
import onnxruntime as ort

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# ─── Download Model from Google Drive ──────────────────────────
@st.cache_resource
def load_model():
    model_path = 'pneumonia_model.onnx'
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... please wait"):
            import urllib.request
            file_id = '1WmjS8YkklUdUyUdVzvbHWkQ6KjMbGh5Q'
            url = f'https://drive.google.com/uc?export=download&id={file_id}&confirm=t'
            urllib.request.urlretrieve(url, model_path)
    
    session = ort.InferenceSession(model_path)
    return session

session = load_model()

# ─── ImageNet Normalization ─────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def preprocess_image(img):
    img = img.resize((128, 128))
    img = img.convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array[:, :, 0] = (img_array[:, :, 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    img_array[:, :, 1] = (img_array[:, :, 1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    img_array[:, :, 2] = (img_array[:, :, 2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ─── UI ────────────────────────────────────────────────────────
st.title("🫁 Pneumonia Detection")
st.markdown("### ResNet50-Based Chest X-Ray Classifier")
st.markdown("Upload a chest X-ray image to detect whether it shows signs of **Pneumonia** or is **Normal**.")
st.divider()

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a chest X-ray image in JPG or PNG format"
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Uploaded X-Ray:**")
        st.image(img, use_container_width=True)
    
    with col2:
        st.markdown("**Prediction:**")
        
        with st.spinner("Analyzing X-Ray..."):
            processed = preprocess_image(img)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            prediction = session.run([output_name], {input_name: processed})[0][0][0]
        
        if prediction > 0.5:
            confidence = prediction * 100
            st.error(f"### 🔴 PNEUMONIA DETECTED")
            st.metric("Confidence", f"{confidence:.2f}%")
            st.warning("⚠️ Please consult a medical professional immediately.")
        else:
            confidence = (1 - prediction) * 100
            st.success(f"### 🟢 NORMAL")
            st.metric("Confidence", f"{confidence:.2f}%")
            st.info("✅ No signs of pneumonia detected.")
        
        st.divider()
        st.markdown("**Prediction Probability:**")
        st.markdown("NORMAL")
        st.progress(float(1 - prediction))
        st.markdown("PNEUMONIA")
        st.progress(float(prediction))

# ─── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    ⚠️ This tool is for research purposes only and should not replace professional medical diagnosis.
    </div>
""", unsafe_allow_html=True)