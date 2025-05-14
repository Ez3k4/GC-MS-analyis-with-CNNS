# https://www.tensorflow.org/tutorials/images/transfer_learning
import json
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import random

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# 0.) CUSTOM LAYERS
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
            # tf.print("SinusoidalAugmentation is in training mode.")
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
            # tf.print("SinusoidalAugmentation is in inference mode.")
            return inputs

def train_model(train_dir, validation_dir, BATCH_SIZE, IMG_SIZE, dropout, base_learning_rate, initial_epochs, fine_tune_at, fine_tune_epochs, base_path, base_filename):
    """ 
    This function takes samples from two different folders (Train, validate) to train 
    a modified version of the MobileNetV2 CNN.
    It also takes various model parameters to make automated training session easier.
    For each run it saves the model, accuracy and loss datapoints for later plotting, 
    the arguments and an image of the plot at base_path location.
    It can show you plots at certain points of the run if enabled.
    """
    
    
    # Save all arguments to a JSON file
    args = locals()  # Get all arguments as a dictionary
    args_save_path = os.path.join(base_path, f"{base_filename}_args.json")
    
    with open(args_save_path, "w") as json_file:
        json.dump(args, json_file, indent=4)
    
    print(f"Arguments saved to {args_save_path}")

    
    # image_dataset_from_directory creates a datset of images according to the folder structure at given location, folders are classes, foldernames are labels
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, 
                                                                batch_size=BATCH_SIZE, 
                                                                image_size=IMG_SIZE) 
    print('Number of train batches: %d' % tf.data.experimental.cardinality(train_dataset))
    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, 
                                                                    batch_size=BATCH_SIZE, 
                                                                    image_size=IMG_SIZE)

    class_names = train_dataset.class_names

    """ plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1): # take method takes one batch and shows the first 9 from them
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1) # creates a 3x3 grid and places image i+1
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show() """


    val_batches = tf.data.experimental.cardinality(validation_dataset)
    print('Number of validation batches before splitting: %d' % tf.data.experimental.cardinality(validation_dataset))

    # 1.2 Split the validation set to have some for training and some for validation
    test_dataset = validation_dataset.take(max(1, val_batches // 5)) # takes the first 1/5 of validation dataset
    validation_dataset = validation_dataset.skip(max(1, val_batches // 5)) # skips the first 1/5 of validation dataset => USED FOR TRAINING
    # => leaves 1/5 of the validation dataset untouched and uses 4/5 of it for validation during training

    print('Number of validation batches after splitting: %d' % tf.data.experimental.cardinality(validation_dataset))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

    # buffered prefetching for better performance of training, valdation and test dataset
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # 1.3 process data: data augmentation -> less overfitting 
    data_augmentation = tf.keras.Sequential([
        # tf.keras.layers.RandomFlip('horizontal'), # randomly flips the image on a horizontal axis
        # tf.keras.layers.RandomRotation(0.2), # randomly rotates the image about 20°
        tf.keras.layers.RandomTranslation(height_factor=5/244, width_factor=0.0), # shifts image randomly up and down in a range
        tf.keras.layers.RandomBrightness(factor=0.05),
        SinusoidalAugmentation(amplitude_range=(-1.5, 1.5), frequency_range=(0.4, 0.4), phase_shift_range=(0, 2* np.pi))
    ])


    """ for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0), training=True)
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
    plt.show() """

    # 2.) PRE-TRAINED MODEL
    # using the MobileNetV2 as base model: expects values between [-1, 1]
    # -> preprocessing: use preprocessing included in model or use tf.keras.layers.Rescaling
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input # rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

    # Create the base model from the pre-trained model MobileNet V2: established good weights and archictecture
    IMG_SHAPE = IMG_SIZE + (3,) # adds the thierd dimension to the image: (160x160) -> (160x160x3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet') # include_top=False removes the classification layer, just feature extractor

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    # 2.1 freeze convolutional base
    # freeze the layer that should not change (keep their weights) !!! -> freezing BatchNormalization leads to not updating its mean and variance
    # !!! during unfreezing keep the BatchNormalization in inference mode Training = false otherwise they will destroy the valuable weights
    base_model.trainable = False

    # Let's take a look at the base model architecture
    """ base_model.summary() # if everything is frozen there should be no trainable params """

    # add classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D() # averages over the 1280(5x5) feature maps returning in a single 1280-element vector
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape) # -> (32, 1280)

    # Dense layer converts it in our case to one prediction per image (pos. number=class1, neg. number=class0)
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape) # -> (32, 1)

    # assemble model: chain together data augnmentation, rescaling, base_model and feature extractor layers (classification)
    inputs = tf.keras.Input(shape=(244, 244, 3))
    x = data_augmentation(inputs, training=True)
    x = preprocess_input(x)
    x = base_model(x, training=False) # training has to be fale because model contains BatchNormalization !!! messes up the weights
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    """ model.summary() # trainable parameters are weights and biases """


    # the optimizer changes weights and biases according to minimize the result of the loss function -> higher loss stronger change in weights and biases
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), 
                loss=tf.keras.losses.BinaryCrossentropy(), 
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])


    # without learning
    loss0, accuracy0 = model.evaluate(validation_dataset)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    # train the model
    history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset)


    # Learning curves
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    """ plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show() """


    # 3.) FINE TUNING
    # the highest layers of the model are the most specialized ones (specialized to the new input data)
    # -> only fine tune (unfreeze) top layers progressively, else you loose your weights

    # 3.1 unfreeze model
    base_model.trainable = True

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # use an extremely low training rate to prevent overfitting base_learning_rate / 10
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/11), # different optimizer
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

    # model.summary()

    total_epochs =  initial_epochs + fine_tune_epochs

    # in history the frozen model trains, then it continues partly unfrozen where the old one left of
    history_fine = model.fit(train_dataset,
                            epochs=total_epochs,
                            initial_epoch=len(history.epoch),
                            validation_data=validation_dataset)

    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']


    # DATA COLLECTION
    extension = '.keras'  # change the extension if needed
    model_counter = 1
    plot_counter = 1
    json_counter = 1
    plot_save_path = os.path.join(base_path, f"training_validation_{base_filename}_{plot_counter}.png")  # Specify the file path and name
    json_save_path = os.path.join(base_path, f'{base_filename}_{json_counter}.json')
    model_save_path = os.path.join(base_path, f'{base_filename}_{model_counter}{extension}')

    # Loop to find a unique filename
    while os.path.exists(model_save_path):
        model_counter += 1
        model_save_path = os.path.join(base_path, f'{base_filename}_{model_counter}{extension}')

    while os.path.exists(json_save_path):
        json_counter += 1
        json_save_path = os.path.join(base_path, f'{base_filename}_{json_counter}.json')

    while os.path.exists(plot_save_path):
        plot_counter += 1
        plot_save_path = os.path.join(base_path, f'{base_filename}_{plot_counter}.png')


    # SAVE PLOT
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    # Save the plot to a file
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')  # Save the plot with high resolution
    print(f"Plot saved to {plot_save_path}")

    # 4.) EVALUATION
    eva_loss, eva_accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', eva_accuracy)
    print('Test loss :', eva_loss)

    # Retrieve a batch of images from the test set (it was trained with the train and validation set)
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    """ plt.figure(figsize=(10, 10))
    for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")
    plt.show()
    """
    # 5.) SAVE MODEL
    model.save(model_save_path)
    print(f'Model saved to {model_save_path}')

    # Save the plot data to a JSON file
    plot_data = {
        "accuracy": acc,
        "val_accuracy": val_acc,
        "loss": loss,
        "val_loss": val_loss,
        "evaluation_acc": eva_accuracy,
        "evaluation_loss": eva_loss
    }

    with open(json_save_path, "w") as json_file:
        json.dump(plot_data, json_file)

    print(f"Plot data saved to {json_save_path}")


def create_base_path(base_dir, file_name, iterable):
    """
    Creates a base path and ensures the directory exists.

    Parameters:
    - base_dir (str): The root directory where the folder should be created.
    - iterable: The iterable to include in the folder name.

    Returns:
    - str: The full path to the created directory.
    """
    base_filename = f"{file_name}{iterable}"
    base_path = os.path.join(base_dir, base_filename)
    
    # Create the folder if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    return base_path



if __name__ == "__main__":
    # 1.) LOAD DATA
    # 1.1 Create folders for training data if there are none 
    train_dir = r'c:\Users\emilk\.keras\datasets\diffsize_int\56\train' # creates a train subfolder in the cats_and_dogs_filtered folder
    validation_dir = r'c:\Users\emilk\.keras\datasets\diffsize_int\56\validation' # creates a validation subfolder in the cats_and_dogs_filtered folder

    # 1.2 parameters
    BATCH_SIZE = 24
    IMG_SIZE = (244, 244)
    dropout = 0.3
    base_learning_rate = 0.0001 # determines step size the optimizer takes, too fast -> overshoot, too slow -> takes ages
    initial_epochs = 50
    fine_tune_at = 30 # Fine-tune from this layer onwards
    fine_tune_epochs = 50

    # 1.3 filehandling


    # Define the base path and filename
    # base_path = r'c:\Studium\Bachelor_Arbeit\mymodels\bloated_int\Test'
    # base_filename = 'mobilenet_int_56_unfreeze_30'
    

    max_epochs = [4]
    for epoch in max_epochs:
        base_filename = f"epoch{epoch}"
        base_path = fr"c:\Studium\Bachelor_Arbeit\USB_stick_niehuis\{base_filename}"
        
        # Create the folder if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        train_model(train_dir, validation_dir, 24, IMG_SIZE, 0.3, 0.1, int(epoch/2), 100, int(epoch/2), base_path, base_filename)