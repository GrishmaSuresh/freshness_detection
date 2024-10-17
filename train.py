import numpy as np
from data_loader import load_data
from model import build_model
import matplotlib.pyplot as plt
from sklearn.utils import class_weight


def train_model(data_dir, epochs=10):
    # Load data
    train_generator, validation_generator = load_data(data_dir)

    # Build model
    model = build_model()

    # Compute class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights))

    # Train the model with class weights
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        class_weight=class_weights  # Use class weights
    )

    # Save the model
    model.save('fresh_rotten_model.h5')

    return history


# Optional: Plot training history
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# Example usage
if __name__ == "__main__":
    history = train_model('dataset/Fruit And Vegetable Diseases Dataset')
    plot_history(history)
