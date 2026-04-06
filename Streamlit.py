import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras

#Navigaton bar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Classification", "Detection"]
)
if page == "Classification":
    st.set_page_config(page_title="🐦 Bird vs 🚁 Drone Classifier",layout="centered")
    st.divider()


    # Load model
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model(
            r"C:\Users\chaka\Preethu\My_Git_Repo\Aerial_Project5\saved_models\best_model.keras"
        )
    model = load_model()

    # -------------------------------------------------
    # 1. Page Configuration
    # -------------------------------------------------

    st.title("🛩️ Bird vs Drone Image Classifier")
    st.write("Upload an image and the model will classify it as **Bird** or **Drone**.")

    # -------------------------------------------------
    # 2. File Upload
    # -------------------------------------------------
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # -------------------------------------------------
        # 4. Preprocess
        # -------------------------------------------------
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # -------------------------------------------------
        # 4. Predict
        # -------------------------------------------------
        raw_pred = model.predict(img_array, verbose=0)[0][0]

        bird_prob = 1 - raw_pred
        drone_prob = raw_pred

        if raw_pred >= 0.5:
            predicted_label = "🚁 Drone"
            confidence = drone_prob
        else:
            predicted_label = "🐦 Bird"
            confidence = bird_prob

        # -------------------------------------------------
        # 5. Results
        # -------------------------------------------------
        st.markdown("### 📊 Prediction Results")
        st.success(f"**Predicted Class:** {predicted_label}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

        st.markdown("### 🔍 Class Probabilities")
        st.write(f"🐦 Bird:  {bird_prob * 100:.2f}%")
        st.write(f"🚁 Drone: {drone_prob * 100:.2f}%")

        st.progress(float(confidence))
    # -------------------------------------------------
    # Footer
    # -------------------------------------------------
    st.markdown("---")
    st.caption("🚀 Built with Streamlit & TensorFlow")

from ultralytics import YOLO
import cv2

if page == "Detection":
    st.set_page_config(page_title="🐦 YOLOv8 Bird and Drone Detection",layout="centered")
    st.divider()

    st.title("🐦 YOLOv8 Bird and Drone Detection")
    st.write("Upload an image and the model will classify it as **Bird** or **Drone**.")

    # Load model
    yolo_model = YOLO(r"C:\Users\chaka\runs\detect\train2\weights\best.pt")
# -------------------------------------------------
# File Upload
# -------------------------------------------------
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        # Convert to numpy
        img_array = np.array(image)
    
        # Run YOLO
        results = yolo_model(img_array)
    
        # Get result image
        result_img = results[0].plot()
    
        # Convert BGR to RGB
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
        # Show result
        st.image(result_img, caption="Detected Image", use_container_width=True)
     # -------------------------------------------------
    # Footer
    # -------------------------------------------------
    st.markdown("---")
    st.caption("🚀 Built with Streamlit & Ultralytics")
