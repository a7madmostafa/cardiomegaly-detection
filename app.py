import streamlit as st
import requests
import tensorflow as tf

# Function to download the model from Google Drive
def download_model(model_url, save_path):
    response = requests.get(model_url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# URL to the Google Drive file (replace with your actual file link)
model_url = 'https://drive.google.com/uc?id=1eWOmm28kKYHN2VFpijHv_YZZ2RFxWLsn'  # Use your modified link here

# Path to save the model
model_file = 'best_model.keras'

# Download the model if not already present
try:
    model = tf.keras.models.load_model(model_file)
    st.success("Model loaded successfully!")
except:
    st.warning("Model not found, downloading...")
    download_model(model_url, model_file)
    model = tf.keras.models.load_model(model_file)
    st.success("Model loaded successfully after download!")

# Streamlit app to classify X-ray images
st.title("X-ray Classification")
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image (resize, normalize, etc.)
    from PIL import Image
    import numpy as np

    img = Image.open(uploaded_file).resize((224, 224))
    img = np.array(img) / 255.0  # Normalize image

    # Make a prediction
    prediction = model.predict(np.expand_dims(img, axis=0))

    # Show result
    if prediction[0] > 0.5:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write("Prediction: **Positive (Disease Present)**")
    else:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write("Prediction: **Negative (No Disease)**")
