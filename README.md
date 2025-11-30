# Object Detection App using YOLO and Streamlit

This application provides an intuitive web interface for real-time object detection in images. Built with Streamlit, it leverages the power of YOLO (You Only Look Once) for accurate and fast detections, with OpenCV handling image processing and visualizations.

## Features

-   **Interactive Web UI**: A user-friendly interface built with Streamlit, allowing users to easily upload images and view results.
-   **Real-Time Object Detection**: Utilizes a pretrained YOLO model to identify multiple objects in an image quickly and accurately.
-   **Adjustable Confidence Threshold**: Users can fine-tune the detection sensitivity by adjusting the confidence threshold through a sidebar slider.
-   **Side-by-Side Comparison**: Displays the original uploaded image and the processed image with bounding boxes side-by-side for easy comparison.
-   **Detailed Detection Summary**: Provides a comprehensive summary of detected objects, including class names, confidence scores, and total counts.
-   **Downloadable Results**: Allows users to download the processed image with detection overlays for offline use.

## Technologies Used

-   **Backend**: Python
-   **Web Framework**: Streamlit
-   **Object Detection**: YOLO (You Only Look Once)
-   **Image Processing**: OpenCV, Pillow (PIL)
-   **Data Handling**: NumPy

## Setup and Installation

To run this application locally, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/pohxiang/computer-vision.git
    cd computer-vision
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install yolo11x.ptx**:
    Goto: https://drive.google.com/file/d/1Usqk_ORONDIfWBBD3y-XmO28NgCefYtd/view?usp=sharing
    Download it and place it in this folder

5.  **Run the Application**:
    ```bash
    streamlit run main.py
    ```

    The application will be accessible at `http://localhost:8501`.

## 📖 How to Use

1.  **Launch the App**: Follow the setup instructions to run the application.
2.  **Upload an Image**: Use the file uploader to select an image file (JPG, JPEG, or PNG).
3.  **Adjust Settings**: Use the sidebar to set the desired confidence threshold for detections.
4.  **View Results**: The app will display the original image and the processed image with bounding boxes.
5.  **Analyze Detections**: Review the detection summary and detailed results table to understand the detected objects.
6.  **Download Image**: Click the "Download Processed Image" button to save the results.

## 📂 File Structure

```
.
├── main.py                  # Main application script
├── requirements.txt         # Project dependencies
├── yolov11n.pt              # Pretrained YOLO model
├── services/                # Service modules
│   ├── ultralytics/
│   │   └── yolo_service.py  # Handles YOLO model loading and processing
│   └── opencv/
│       └── opencv_service.py# Handles image drawing and processing
└── README.md                # This file

Testting Testing

pohxiang PR