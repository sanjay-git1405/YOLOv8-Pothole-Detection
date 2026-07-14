import os
from ultralytics import YOLO

def main():
    # 1. Automatically get the exact folder where this Python script is located
    current_folder = os.path.dirname(os.path.abspath(__file__))

    # 2. Build the exact path to data.yaml automatically
    yaml_path = os.path.join(current_folder, "data.yaml")

    # 3. Verify the file actually exists before YOLO even tries to touch it
    if not os.path.exists(yaml_path):
        print(f"❌ ERROR: Python cannot find the file at: {yaml_path}")
        print("Please check if the file is named 'data.yaml.txt' by mistake!")
        return
    else:
        print(f"✅ SUCCESS: Found data.yaml at: {yaml_path}")

    # 4. YOLO prefers forward slashes, even on Windows, so we flip them
    yaml_path = yaml_path.replace("\\", "/")

    # 5. Load model and train
    model = YOLO("yolov8s.pt") 

    results = model.train(
        data=yaml_path,     # Passing the bulletproof path
        epochs=50,          
        imgsz=640,          
        batch=16,           
        device=0            
    )

if __name__ == "__main__":
    main()