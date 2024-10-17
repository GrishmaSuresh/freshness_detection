# Fruit and Vegetable Freshness Detection

This project uses a pre-trained TensorFlow model to detect the freshness of fruits and vegetables using a live camera feed. The system classifies each frame as either "Healthy" or "Rotten" based on predictions from the model.

## Features

- **Real-time Detection**: Uses a live webcam feed to detect freshness in fruits and vegetables.
- **Model-based Prediction**: A TensorFlow model (`fresh_rotten_model.h5`) is used to classify each frame.
- **Prediction Smoothing**: Predictions are averaged over multiple frames to reduce noise and provide a smoother result.

## Requirements

- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- A pre-trained Keras model (`fresh_rotten_model.h5`)

## Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/yourusername/freshness-detection.git
  ```
2. Install the required dependencies:
  ```bash
  pip install tensorflow opencv-python numpy
  ```
3. Ensure you have a pre-trained model file (fresh_rotten_model.h5). You can train your own model or download one if available.

## Usage
1. Place your pre-trained model in the same directory as the script or provide the correct path in the code.

2. Run the script:

```bash
python freshness_detection.py
```
3. The live camera feed will open, and the freshness status ("Healthy" or "Rotten") will be displayed on the frame.

4. Press q to exit the application.

## How It Works
- **Frame Processing:** The live camera feed captures each frame, resizes it to 150x150, and normalizes the pixel values (dividing by 255) to match the input format used during model training.
- **Model Prediction:** The pre-trained model predicts the freshness of the produce in each frame. A prediction of over 0.5 is labeled as "Rotten", while lower values are labeled as "Healthy."
- **Smoothing:** A deque stores the last few predictions, and the average is calculated to avoid jittery predictions.
  
## Example Output
- **Healthy:** When the model predicts the fruit/vegetable is fresh.
- **Rotten:** When the model predicts the fruit/vegetable is no longer fresh.

## Arguments
- **model_path:** Path to the pre-trained model file (fresh_rotten_model.h5).
- **smoothing_factor:** The number of previous predictions to average for smoothing (default is 5).
  
## Troubleshooting
- **Camera Not Opening:** Ensure your camera is properly connected. If the camera index (0) does not work, try changing it to 1, 2, etc.
- **Low Prediction Accuracy:** Ensure the model was trained with relevant data. Retraining the model with better quality images may improve results.
  
## Future Enhancements
- Add support for more classes (e.g., multiple levels of freshness).
- Improve the model to handle more variations in lighting and angle.
- Add support for mobile or embedded systems.
  
## License
MIT License
