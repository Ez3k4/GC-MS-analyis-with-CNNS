import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import os

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.config.experimental.enable_op_determinism()

# 0.) CUSTOM LAYERS
@tf.keras.utils.register_keras_serializable()
class SinusoidalAugmentation(tf.keras.layers.Layer):
    def __init__(self, amplitude_range=(-5, 5), frequency_range=(-1, 1), phase_shift_range=(0, 2 * np.pi), **kwargs):
        """
        Custom Keras layer for sinusoidal data augmentation with random parameters.

        Parameters:
        - amplitude_range: Tuple specifying the range for random amplitude (min, max).
        - frequency_range: Tuple specifying the range for random frequency (min, max).
        - phase_shift_range: Tuple specifying the range for random phase shift (min, max).
        """
        super(SinusoidalAugmentation, self).__init__(**kwargs)
        self.amplitude_range = amplitude_range
        self.frequency_range = frequency_range
        self.phase_shift_range = phase_shift_range

    def call(self, inputs, training=False):
        """
        Apply the sinusoidal augmentation to the input images.

        Parameters:
        - inputs: Input tensor (batch of images).
        - training: Whether the layer is in training mode.

        Returns:
        - Augmented images.
        """
        if training:
            # Generate random amplitude, frequency, and phase shift
            amplitude = tf.random.uniform([], self.amplitude_range[0], self.amplitude_range[1])
            frequency = tf.random.uniform([], self.frequency_range[0], self.frequency_range[1])
            phase_shift = tf.random.uniform([], self.phase_shift_range[0], self.phase_shift_range[1])

            # Print the random parameters (for debugging)
            # tf.print("Random amplitude:", amplitude, "Random frequency:", frequency, "Random phase shift:", phase_shift)

            # Get the shape of the input images
            batch_size, height, width, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]

            # Generate a sinusoidal wave along the height (y-axis)
            x = tf.range(0, height, dtype=tf.float32)  # Retention indices (y-axis)
            sin_wave = amplitude * tf.math.sin(2 * np.pi * frequency * x + phase_shift)

            # Expand the wave to match the image dimensions
            sin_wave = tf.reshape(sin_wave, (1, height, 1, 1))  # Shape: (1, height, 1, 1)
            sin_wave = tf.tile(sin_wave, [batch_size, 1, width, channels])  # Broadcast to match input shape

            # Add the sinusoidal wave to the input images
            augmented_images = inputs + sin_wave

            # Clip pixel values to ensure they remain in the valid range [0, 255]
            augmented_images = tf.clip_by_value(augmented_images, 0, 255)

            return augmented_images
        else:
            return inputs

# Load the saved model
model_load_path = r'c:\Studium\Bachelor_Arbeit\USB_stick_niehuis\03_dfferent_sample_sizes\zscore\results\polistes244x244_6\polistes244x244_6_1.keras'  # Update with the correct path
model = tf.keras.models.load_model(
    model_load_path,
    custom_objects={'SinusoidalAugmentation': SinusoidalAugmentation}
)

# Load test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r"c:\Studium\Bachelor_Arbeit\USB_stick_niehuis\03_dfferent_sample_sizes\zscore\test_zscore", image_size=(244, 244), batch_size=100, shuffle=False
)

# Evaluate model on test datas
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# Predict on a batch of test images
for images, labels in test_dataset.take(1):  # Take one batch
    predictions = model.predict(images)
    predictions = tf.where(predictions < 0.5, 0, 1)  # Convert to binary class
    print("Predictions:", predictions.numpy().flatten())
    print("Actual Labels:", labels.numpy().flatten())

