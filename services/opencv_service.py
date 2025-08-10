import cv2
import numpy as np
from PIL import Image

def draw_predictions(image, results):
    """Draw bounding boxes and labels on image"""
    if not results or not hasattr(results, 'boxes') or results.boxes is None:
        return image
    
    # Create a copy of the image
    img_with_boxes = image.copy()
    
    # Convert PIL to CV2 format for drawing
    img_cv2 = cv2.cvtColor(np.array(img_with_boxes), cv2.COLOR_RGB2BGR)
    
    boxes = results.boxes
    
    for i, box in enumerate(boxes):
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # Get confidence and class
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        class_name = results.names[class_id]
        
        # Draw bounding box
        cv2.rectangle(img_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Draw label with confidence
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw label background
        cv2.rectangle(img_cv2, (int(x1), int(y1) - label_size[1] - 10), 
                     (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(img_cv2, label, (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Convert back to PIL format
    img_with_boxes = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    
    return img_with_boxes
