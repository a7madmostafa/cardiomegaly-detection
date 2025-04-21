import streamlit as st
import requests
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# URL to download the model from Google Drive (replace with your file ID)
model_url = "https://drive.google.com/uc?id=1eWOmm28kKYHN2VFpijHv_YZZ2RFxWLsn"
model_path = "best_model.keras"

# Function to download the model from Google Drive
def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        st.write("Model downloaded successfully!")
    else:
        st.write("Model already exists locally.")

# Function to load the model
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.write("Model loaded successfully!")
        return model
    except ValueError as e:
        st.error(f"Error loading model: {e}")
        return None

# Download and load the model
download_model(model_url, model_path)
model = load_model(model_path)

# Function for X-ray image classification
def classify_image(uploaded_file):
    if model is None:
        return "Model is not loaded successfully!"

    # Open the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize the image

    # Expand dimensions to match model input shape
    image_array = np.expand_dims(image_array, axis=0)

    # Predict with the model
    prediction = model.predict(image_array)
    return prediction

# Streamlit UI
st.title("X-ray Classification")
st.write("Upload an X-ray image to classify.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Classify the image and show the result
    prediction = classify_image(uploaded_file)
    st.write(f"Prediction: {prediction}")
