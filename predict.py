from ultralytics import YOLO

def main():
    # 1. Load your brand new custom-trained model
    # Notice how it points to the train-6 folder!
    model = YOLO(r"C:\Users\SANJAY\Desktop\pothole_detection1\runs\detect\train-6\weights\best.pt")

    # 2. Run the prediction on your video
    # 'show=True' will pop up a window showing you the live results
    # 'save=True' will save a copy of the video with the boxes drawn on it
    results = model.predict(source="sample_video.mp4", show=True, save=True)

if __name__ == "__main__":
    main()