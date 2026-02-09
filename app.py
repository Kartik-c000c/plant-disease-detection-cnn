import os
import cv2
import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection System")
st.write("Upload a leaf image to predict the disease")

# ---------------------------
# Paths (VERY IMPORTANT)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_cnn.h5")
LABEL_PATH = os.path.join(BASE_DIR, "plant_disease_label_transform.pkl")

IMG_SIZE = 256

# ---------------------------
# Load CNN model
# ---------------------------
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

# ---------------------------
# Load Label Binarizer
# ---------------------------
@st.cache_resource
def load_label_binarizer():
    with open(LABEL_PATH, "rb") as f:
        return pickle.load(f)

# ---------------------------
# Prediction function
# ---------------------------
def predict_disease(image, model, label_binarizer):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    idx = np.argmax(preds)
    confidence = preds[0][idx] * 100
    disease = label_binarizer.classes_[idx]

    return disease, confidence

# ---------------------------
# Load models safely
# ---------------------------
try:
    model = load_cnn_model()
    label_binarizer = load_label_binarizer()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error("❌ Error loading model files")
    st.stop()

# ---------------------------
# File uploader
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing leaf..."):
            disease, confidence = predict_disease(
                image, model, label_binarizer
            )

        st.markdown("---")
        st.markdown(f"### 🦠 Disease: **{disease}**")
        st.markdown(f"### 🎯 Confidence: **{confidence:.2f}%**")
