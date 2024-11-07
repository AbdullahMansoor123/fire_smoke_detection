import streamlit as st
import cv2
from datetime import timedelta
from ultralytics import YOLO
import tempfile
import numpy as np


# Load the YOLO model for smoke and fire detection
@st.cache_resource
def load_yolo_model(model_path=None):
    if model_path:
        model = YOLO(model_path)
    else:
        model = YOLO("model_weights\\best.pt")  # Default model
    return model


# Draw bounding boxes with labels and confidence scores
def draw_bounding_boxes(frame, detections):
    for detection in detections:
        x_min, y_min, x_max, y_max = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']

        color = (255, 0, 0) if label == 'fire' else (128, 128, 128)
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def main():
    st.title("Smoke and Fire Detection with YOLO")

    # Model upload option
    model_file = st.file_uploader("Upload a YOLO model (.pt file)", type=["pt"])
    
    # Load the YOLO model (uploaded or default)
    model_path = None
    if model_file:
        temp_model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pt").name
        with open(temp_model_path, 'wb') as f:
            f.write(model_file.read())
        model_path = temp_model_path
    
    model = load_yolo_model(model_path)

    video_file = st.file_uploader("Upload a video file for smoke/fire detection", type=["mp4", "avi", "mov"])

    if video_file:
        temp_video_path = tempfile.NamedTemporaryFile(delete=False).name
        with open(temp_video_path, 'wb') as f:
            f.write(video_file.read())
        
        cap = cv2.VideoCapture(temp_video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        
        output_frames = []
        previous_bbox = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            classes = ['fire', 'smoke']
            # Detect smoke or fire in a frame and get bounding boxes
            outputs = model.predict(frame)
            detections = []
            for detection in outputs:
                if len(detection) > 0:
                    bboxes = detection.boxes
                    previous_bbox = bboxes
                else:
                    bboxes = previous_bbox

                for box in bboxes:
                    bbox = box.xyxy.tolist()[0]
                    label = classes[int(box.cls[0].item())]
                    confidence = box.conf[0].item()

                    if label in ['smoke', 'fire']:
                        detections.append({'label': label, 'confidence': confidence, 'bbox': bbox})
                        
            frame_with_boxes = draw_bounding_boxes(frame, detections)
            output_frames.append(frame_with_boxes)
        cap.release()

        # Create output video for download
        temp_output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(temp_output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (output_frames[0].shape[1], output_frames[0].shape[0]))
        for frame in output_frames:
            out.write(frame)
        out.release()
            
        # Download processed video
        with open(temp_output_video_path, 'rb') as f:
            st.download_button("Download Video", f, "output_video.mp4")

if __name__ == '__main__':
    main()
