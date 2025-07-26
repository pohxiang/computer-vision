import cv2
from ultralytics import YOLO
import torch


model = YOLO('yolov8n.pt') # Can be changed to liking

# Open default webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform object detection
    results = model(frame, imgsz=640, verbose=True)

    # Render results (bounding boxes + labels)
    annotated_frame = results[0].plot()

    # Show the output
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
