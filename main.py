import cv2
import datetime
from ultralytics import YOLO
import tempfile
import numpy as np
import os


def generate_video_name(output_video_path):
    # Get the current date and time
    now = datetime.datetime.now()
    # Format it into a compressed string: YYYYMMDD_HHMMSS
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    extension = output_video_path.split('.')[-1]
    new_video_name = f"video_{timestamp}.{extension}"
    return new_video_name


# Load the YOLO model for smoke and fire detection
def load_yolo_model(model_path=None):
    if model_path:
        model = YOLO(model_path)
    else:
        print('No Model weights added')
    return model

# Draw bounding boxes with labels and confidence scores
def draw_bounding_boxes(frame, detections):
    for detection in detections:
        x_min, y_min, x_max, y_max = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']
        color = (255, 0, 0) if label == 'fire' else (128, 128, 128)
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        text = f"{label}" #{confidence:.2f}
        cv2.putText(frame, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def main():
    print("Smoke and Fire Detection with YOLO")

    # Model upload option
    model_path = r"runs/detect/train2/weights/best.pt"

    # Load the YOLO model
    model = load_yolo_model(model_path)
    
    # Video file path
    video_path = "pp_fire.mp4"
    print(os.path.exists(video_path))
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(3)) 
    height = int(cap.get(4))

    output_video_path = generate_video_name(output_video_path = "output.mp4")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    previous_bbox = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        classes = ['fire', 'smoke']
        outputs = model.predict(frame)
        detections = []
        for detection in outputs:
            #### Comment this to remove flagging
            if len(detection) > 0:
                bboxes = detection.boxes
                previous_bbox = bboxes
            else:
                bboxes = previous_bbox
            ####
            
            # Commit this to not use flagging
            # bboxes = detection.boxes

            for box in bboxes:
                bbox = box.xyxy.tolist()[0]
                label = classes[int(box.cls[0].item())]
                confidence = box.conf[0].item()

                if label in ['smoke', 'fire']:
                    detections.append({'label': label, 'confidence': confidence, 'bbox': bbox})
                    
        frame_with_boxes = draw_bounding_boxes(frame, detections)
        out.write(frame_with_boxes)
    cap.release()
    out.release()
    
    print(f"Processing complete. Output video saved at: {output_video_path}")

if __name__ == '__main__':
    main()
