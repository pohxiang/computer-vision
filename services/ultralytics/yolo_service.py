import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st


class yoloservices:
    @st.cache_resource
    def load_model(_self):
        try:
            model = YOLO('yolo11x.pt')
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    def process_image(self, image, model, confidence_threshold=0.5):
        """Process image through YOLO model and return results"""
        try:
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            img_array = np.array(image)
            results = model(img_array, conf=confidence_threshold)
            
            return results[0] if results else None
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None


    def get_detection_summary(self, results):
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
