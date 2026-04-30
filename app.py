import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
from src.utils import get_disposal_suggestion, preprocess_image

# Page Config
st.set_page_config(page_title="AI Waste Classifier", page_icon="♻️", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #28a745;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .suggestion-box {
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title & Sidebar
st.title("♻️ AI Waste Classification System")
st.markdown("---")

sidebar = st.sidebar
sidebar.header("Settings")
model_choice = sidebar.selectbox("Choose Model", ["Transfer Learning (MobileNetV2)", "Custom CNN"])
input_method = sidebar.radio("Input Method", ["Image Upload", "Webcam"])

# Load Model
@st.cache_resource
def load_model(choice):
    model_path = 'models/transfer_learning.h5' if choice == "Transfer Learning (MobileNetV2)" else 'models/custom_cnn.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_model(model_choice)
CLASSES = ['Biodegradable', 'Recyclable']

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Image")
    uploaded_file = None
    if input_method == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_container_width=True)
    else:
        cam_image = st.camera_input("Take a photo")
        if cam_image is not None:
            img = Image.open(cam_image)
            st.image(img, caption='Captured Image', use_container_width=True)
            uploaded_file = cam_image # Treat as uploaded for processing

with col2:
    st.subheader("Classification Result")
    if uploaded_file is not None:
        if model is None:
            st.warning("⚠️ Model file not found. Please run 'train.py' first to train the models.")
        else:
            # Prediction
            with st.spinner('Analyzing...'):
                processed_img = preprocess_image(img)
                preds = model.predict(processed_img)
                class_idx = np.argmax(preds[0])
                confidence = preds[0][class_idx] * 100
                label = CLASSES[class_idx]
                
                # Suggestion
                suggestion = get_disposal_suggestion(label)
                
                # UI Display
                st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Predicted Class</h3>
                        <h1 style="color: {suggestion['color']};">{label}</h1>
                        <h4>Confidence: {confidence:.2f}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class="suggestion-box" style="background-color: {suggestion['color']};">
                        <strong>Disposal Suggestion:</strong><br>
                        {suggestion['action']} - {suggestion['details']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Dashboard Stats (Mock / Session State)
                if 'stats' not in st.session_state:
                    st.session_state.stats = {'Biodegradable': 0, 'Recyclable': 0}
                
                st.session_state.stats[label] += 1

# Dashboard Statistics
st.markdown("---")
st.subheader("📊 Dashboard Statistics")
if 'stats' in st.session_state:
    stats_df = pd.DataFrame(list(st.session_state.stats.items()), columns=['Category', 'Count'])
    # st.bar_chart(stats_df.set_index('Category'))
else:
    st.info("Start classifying images to see statistics.")

# Instructions
with st.expander("How it works"):
    st.write("""
    1. **Upload or Capture**: Provide an image of a waste item.
    2. **AI Analysis**: Our CNN model processes the image to identify features.
    3. **Classification**: The model classifies the item into one of three categories.
    4. **Smart Suggestion**: Based on the class, we provide the best way to dispose of the item.
    """)
