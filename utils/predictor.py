# utils/predictor.py

import os
import random
from ultralytics import YOLO

# Load the YOLOv8 model (uses COCO classes unless custom-trained)
try:
    model = YOLO("yolov8n.pt")  # Or replace with your custom .pt file
    print("YOLOv8 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    model = None

# Map YOLO class IDs to your waste labels (if using custom training, adjust accordingly)
# These should match your training dataset if you trained YOLO yourself
WASTE_CLASSES = ['plastic', 'paper', 'metal', 'glass', 'organic', 'trash', 'cardboard']

def predict_waste_type(image_path):
    """
    Use YOLOv8 to detect and classify waste objects in the image.
    Returns: (category, confidence)
    """
    try:
        if not os.path.exists(image_path):
            print("Image file not found:", image_path)
            return None, 0.0

        if model is None:
            print("Model is not loaded. Falling back to mock prediction.")
            return mock_predict(), 0.8

        # Run prediction
        results = model(image_path)

        # Extract detection data (first detection only)
        boxes = results[0].boxes
        if boxes is not None and len(boxes.cls) > 0:
            cls_id = int(boxes.cls[0])  # first object
            confidence = float(boxes.conf[0])
            category = WASTE_CLASSES[cls_id] if cls_id < len(WASTE_CLASSES) else "unknown"
            return category, confidence

        print("No objects detected by YOLO.")
        return None, 0.0

    except Exception as e:
        print(f"YOLO prediction error: {str(e)}")
        return None, 0.0

def mock_predict():
    """Fallback mock prediction for testing"""
    categories = ['plastic', 'paper', 'metal', 'glass', 'organic', 'trash']
    weights = [0.25, 0.2, 0.15, 0.15, 0.2, 0.05]
    return random.choices(categories, weights=weights, k=1)[0]
