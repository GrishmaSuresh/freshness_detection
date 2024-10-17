import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import supervision as sv
from collections import deque

# Function to load and prepare frames for freshness detection model
def load_and_prepare_frame(frame):
    frame_resized = cv2.resize(frame, (150, 150))
    frame_array = np.expand_dims(frame_resized, axis=0)
    frame_array = frame_array / 255.0
    return frame_array

# Function to run both YOLO object detection and freshness prediction
def detect_and_classify_fruit(model_path, yolo_model_path="yolov8l.pt", smoothing_factor=5):
    # Load the freshness detection model
    freshness_model = tf.keras.models.load_model(model_path)
    freshness_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Load YOLOv8 model for object detection
    yolo_model = YOLO(yolo_model_path)

    # Initialize video capture and YOLO box annotator
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

    # Queue to store previous predictions for smoothing
    prediction_queue = deque(maxlen=smoothing_factor)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run YOLOv8 for object detection
        result = yolo_model(frame)[0]
        detections = sv.Detections.from_yolov8(result)

        # If YOLO detects a fruit/vegetable, classify it for freshness
        for detection in detections:
            class_id = detection[2]
            label = yolo_model.model.names[class_id]

            # Prepare the detected region for freshness prediction
            xmin, ymin, xmax, ymax = map(int, detection[-1])
            crop_img = frame[ymin:ymax, xmin:xmax]
            prepared_img = load_and_prepare_frame(crop_img)

            # Predict freshness using the pre-trained model
            prediction = freshness_model.predict(prepared_img)
            prediction_queue.append(prediction[0][0])

            # Smooth predictions over several frames
            avg_prediction = np.mean(prediction_queue)
            status = "Rotten" if avg_prediction > 0.5 else "Healthy"

            # Combine the label with the freshness status
            full_label = f"{label} - {status}"

            # Annotate the frame with the label and bounding box
            frame = box_annotator.annotate(scene=frame, detections=[detection], labels=[full_label])

        # Display the result
        cv2.imshow('Fruit/Veg Detection and Freshness', frame)

        # Break on 'q' key press
        if cv2.waitKey(30) == ord('q'):
            break

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    detect_and_classify_fruit('fresh_rotten_model.h5')
