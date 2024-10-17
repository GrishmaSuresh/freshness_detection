import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


def load_and_prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Same scaling as during training
    return img_array


def predict_image(model_path, image_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Recompile the model

    # Prepare the image
    img_array = load_and_prepare_image(image_path)

    # Make prediction
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print("Rotten")
    else:
        print("Healthy")


# Example usage
if __name__ == "__main__":
    predict_image('fresh_rotten_model.h5', 'test_data/one.jpg')
