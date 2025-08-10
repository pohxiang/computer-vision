import streamlit as st
from PIL import Image
import numpy as np
import io
from services.yolo_service import (
    load_model, 
    process_image, 
    get_detection_summary
)
from services.opencv_service import draw_predictions

# Configure Streamlit page
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🔍 Object Detection with YOLOv8")
    st.markdown("Upload an image to detect objects using YOLOv8 model")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app uses YOLOv8 for real-time object detection.")
        st.markdown("Supported formats: JPG, JPEG, PNG")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load YOLO model. Please check if 'yolov8n.pt' exists in the project directory.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to detect objects"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display original image
            image = Image.open(uploaded_file)
            
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process image
            with st.spinner("Processing image..."):
                results = process_image(image, model, confidence_threshold)
            
            if results is not None:
                # Draw predictions on image
                processed_image = draw_predictions(image, results)
                
                with col2:
                    st.subheader("🎯 Detection Results")
                    st.image(processed_image, caption="Detected Objects", use_container_width=True)
                
                # Display detection summary
                st.subheader("📊 Detection Summary")
                detections = get_detection_summary(results)
                
                if detections == "No objects detected":
                    st.info("No objects detected with the current confidence threshold.")
                else:
                    # Create metrics columns
                    metric_cols = st.columns(min(len(detections), 4))
                    
                    # Group detections by class
                    class_counts = {}
                    class_confidences = {}
                    
                    for detection in detections:
                        class_name = detection['class']
                        confidence = detection['confidence']
                        
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                            class_confidences[class_name].append(confidence)
                        else:
                            class_counts[class_name] = 1
                            class_confidences[class_name] = [confidence]
                    
                    # Display metrics
                    for i, (class_name, count) in enumerate(class_counts.items()):
                        avg_confidence = np.mean(class_confidences[class_name])
                        max_confidence = np.max(class_confidences[class_name])
                        
                        with metric_cols[i % len(metric_cols)]:
                            st.metric(
                                label=f"{class_name.title()}",
                                value=f"{count} detected",
                                delta=f"{max_confidence:.2%} max confidence"
                            )
                    
                    # Detailed detection table
                    st.subheader("� Detailed Results")
                    detection_data = []
                    for i, detection in enumerate(detections, 1):
                        detection_data.append({
                            "Object #": i,
                            "Class": detection['class'].title(),
                            "Confidence": f"{detection['confidence']:.2%}"
                        })
                    
                    st.dataframe(detection_data, use_container_width=True)
                    
                    # Download processed image
                    st.subheader("💾 Download Results")
                    
                    # Convert processed image to bytes
                    img_buffer = io.BytesIO()
                    processed_image.save(img_buffer, format='PNG')
                    img_bytes = img_buffer.getvalue()
                    
                    st.download_button(
                        label="Download Processed Image",
                        data=img_bytes,
                        file_name=f"detected_{uploaded_file.name}",
                        mime="image/png"
                    )
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    else:
        # Show example or placeholder
        st.info("👆 Upload an image to get started with object detection!")
        
        # Example section
        st.subheader("🎨 Example Usage")
        st.markdown("""
        1. **Upload an image** using the file uploader above
        2. **Adjust confidence threshold** in the sidebar if needed
        3. **View results** with bounding boxes and detection confidence
        4. **Download** the processed image with detections
        
        The app will detect common objects like:
        - People, animals, vehicles
        - Household items, electronics
        - Sports equipment, food items
        - And many more (80+ classes total)
        """)

if __name__ == "__main__":
    main()
