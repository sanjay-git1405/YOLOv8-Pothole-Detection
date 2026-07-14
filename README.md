# Real-Time Pothole Detection Using YOLOv8

This project features a custom-trained computer vision model capable of detecting road potholes in real-time. Built using the state-of-the-art **YOLOv8s (Small)** architecture, the model is optimized for a balance between computational efficiency and high detection accuracy, making it suitable for edge deployment and dashcam integration.

## 🚀 Project Overview

*   **Algorithm:** YOLOv8 (Ultralytics)
*   **Framework:** PyTorch
*   **Accuracy:** 79% True Positive Rate (mAP/Precision metrics detailed below)
*   **Hardware Used:** NVIDIA GeForce RTX 4050 (Windows 11)

## 📊 Dataset & Training

The model was trained on a custom dataset containing specialized bounding box annotations for road anomalies.

*   **Total Images:** 1,977 (1,581 Training / 396 Validation)
*   **Epochs:** 50
*   **Image Size:** 640x640
*   **Batch Size:** 16
*   **Workers:** 2 (Optimized to prevent Windows multi-threading data bottlenecks)

### Performance Results
After 50 epochs, the model's weights (`best.pt`) stabilized with strong feature extraction capabilities:
*   **True Positive Rate (Pothole -> Pothole):** 0.79 (79%)
*   The model successfully differentiates between standard asphalt textures, shadows, and actual road damage.
*(Check the `runs/detect/train-6` folder for the full Confusion Matrix and loss graphs).*

## 💻 Installation & Usage

### Prerequisites
Make sure you have Python installed, then install the required YOLOv8 library:
```bash
pip install ultralytics