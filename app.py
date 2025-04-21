import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ✅ Set config FIRST
st.set_page_config(page_title="X-ray Classifier", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./best_model.keras")

model = load_model()

st.title("X-ray Classification App")
st.write("Upload a chest X-ray image to predict Cardiomegaly.")

# Image upload
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_size = (224, 224)
    img_array = np.array(image.resize(img_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Positive (Cardiomegaly)" if prediction >= 0.5 else "Negative (Normal)"
    st.subheader("Prediction:")
    st.write(f"**{label}**")
    st.write(f"Confidence: `{prediction:.2f}`")
