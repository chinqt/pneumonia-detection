import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# ─── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('resnet50_pneumonia_final.keras')
    return model

model = load_model()

# ─── ImageNet Normalization ─────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def preprocess_image(img):
    # Resize to 128x128
    img = img.resize((128, 128))
    # Convert to RGB (handles grayscale X-rays)
    img = img.convert('RGB')
    # Convert to numpy array
    img_array = np.array(img) / 255.0
    # Apply ImageNet channel-wise normalization
    img_array[:, :, 0] = (img_array[:, :, 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    img_array[:, :, 1] = (img_array[:, :, 1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    img_array[:, :, 2] = (img_array[:, :, 2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    # Add batch dimension
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
    # Display uploaded image
    img = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Uploaded X-Ray:**")
        st.image(img, use_container_width=True)
    
    with col2:
        st.markdown("**Prediction:**")
        
        with st.spinner("Analyzing X-Ray..."):
            # Preprocess and predict
            processed = preprocess_image(img)
            prediction = model.predict(processed)[0][0]
        
        # Display result
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
        
        # Probability bar
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