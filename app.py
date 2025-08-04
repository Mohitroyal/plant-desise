import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model_final.h5")
model = load_model()
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
st.title("🥔 Potato Leaf Disease Detector")
st.write("Upload a potato leaf image to detect Early Blight, Late Blight, or Healthy status.")
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img = image.resize((128, 128)) 
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2%}")
