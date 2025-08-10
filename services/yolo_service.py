import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st

@st.cache_resource
def load_model():
    """Load YOLOv8 model with caching for better performance"""
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model, confidence_threshold=0.5):
    """Process image through YOLO model and return results"""
    try:
        # Convert PIL image to RGB if it has an alpha channel (RGBA)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Convert to RGB by removing alpha channel or converting palette
            image = image.convert('RGB')
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run inference
        results = model(img_array, conf=confidence_threshold)
        
        return results[0] if results else None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


def get_detection_summary(results):
    """Generate summary of detections"""
    if not results or not hasattr(results, 'boxes') or results.boxes is None:
        return "No objects detected"
    
    boxes = results.boxes
    detections = []
    
    for box in boxes:
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        class_name = results.names[class_id]
        detections.append({
            'class': class_name,
            'confidence': confidence
        })
    
    return detections
