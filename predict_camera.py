import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from collections import deque


def load_and_prepare_frame(frame):
    # Resize the frame to match the input size of the model (150x150)
    frame_resized = cv2.resize(frame, (150, 150))
    frame_array = np.expand_dims(frame_resized, axis=0)  # Add batch dimension
    frame_array = frame_array / 255.0  # Scale the pixel values as in training
    return frame_array


def predict_from_camera(model_path, smoothing_factor=5):
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Ensure the model is compiled

    # Open the camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Queue to store previous predictions (for smoothing)
    prediction_queue = deque(maxlen=smoothing_factor)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Prepare the frame for prediction
        frame_array = load_and_prepare_frame(frame)

        # Make prediction
        prediction = model.predict(frame_array)
        prediction_queue.append(prediction[0][0])

        # Average the predictions over the last few frames for smoothing
        avg_prediction = np.mean(prediction_queue)
        label = "Rotten" if avg_prediction > 0.5 else "Healthy"

        # Display the result on the frame
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Fruit/Veg Freshness Detection', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    predict_from_camera('fresh_rotten_model.h5')
