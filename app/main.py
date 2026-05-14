import streamlit as st
from predict import predict_image
from gradcam import generate_gradcam

st.set_page_config(page_title="VisionSpec-QC", layout="centered")

st.title("VisionSpec-QC")
st.subheader("AI-Powered Industrial Quality Inspection")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    prediction, confidence, img = predict_image(uploaded_file)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write(f"### Prediction: {prediction}")
    st.write(f"### Confidence: {confidence:.2f}%")

    heatmap = generate_gradcam(uploaded_file)

    st.image(heatmap, caption="Grad-CAM Visualization", use_column_width=True)
