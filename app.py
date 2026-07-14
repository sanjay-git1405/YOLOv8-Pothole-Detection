import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Set up the webpage title and layout
st.set_page_config(page_title="AI Pothole Detector", layout="centered")
st.title("🚧 Real-Time Pothole Detection System")
st.write("Upload an image of a road, and the custom YOLOv8 model will automatically identify and locate potholes.")

# 2. Load your custom trained model weights safely
@st.cache_resource  # This prevents the app from reloading the model every time you click a button
def load_model():
    # Update this path to match your successful train folder if needed
    return YOLO("runs/detect/train-6/weights/best.pt")

model = load_model()

# 3. Create the file uploader widget
uploaded_file = st.file_uploader("Choose a road image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    
    # Create two side-by-side columns to show Before and After
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("AI Detection")
        
        # Run inference using the custom model
        # conf=0.4 means it will display detections with 40% confidence or higher
        results = model(image, conf=0.4)
        
        # Extract the image array with bounding boxes drawn on it
        annotated_frame = results[0].plot()
        
        # Convert BGR (OpenCV format) back to RGB (Streamlit/PIL format)
        annotated_image = Image.fromarray(annotated_frame[..., ::-1])
        
        # Display the result
        st.image(annotated_image, use_container_width=True)
        
    # Optional: Display a success message indicating how many potholes were found
    num_potholes = len(results[0].boxes)
    if num_potholes > 0:
        st.success(f"Detection complete! Found {num_potholes} pothole(s).")
    else:
        st.info("No potholes detected. Road appears clear!")