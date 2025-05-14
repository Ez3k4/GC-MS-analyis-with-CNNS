#!/usr/bin/env python3
import os
from mobilenet_module import train_model
import tensorflow as tf
import numpy as np
import random

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# 1.) LOAD DATA
# 1.1 Create folders for training data if there are none 
train_dir = r'c:\Users\emilk\.keras\datasets\polistes244x244\train' # creates a train subfolder in the cats_and_dogs_filtered folder
validation_dir = r'c:\Users\emilk\.keras\datasets\polistes244x244\train' # creates a validation subfolder in the cats_and_dogs_filtered folder

# 1.2 parameters
BATCH_SIZE = 24
IMG_SIZE = (244, 244)
dropout = 0.3
base_learning_rate = 0.0001 # determines step size the optimizer takes, too fast -> overshoot, too slow -> takes ages
initial_epochs = 50
fine_tune_at = 100 # Fine-tune from this layer onwards
fine_tune_epochs = 50


""" # Epoch testing
max_epochs = [30, 50, 100, 300, 500]
for epoch in max_epochs:
    base_filename = f"epoch_{epoch}"
    base_path = fr"testing_folder\{base_filename}"
    
    # Create the folder if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    train_model(train_dir, validation_dir, 24, IMG_SIZE, 0.3, 0.0001, int(epoch/2), 100, int(epoch/2), base_path, base_filename)


# Dropout testing
dropouts = [0.1, 0.2, 0.4, 0.5]
for dropout in dropouts:
    base_filename = f"dropout_{dropout}"
    base_path = fr"testing_folder\{base_filename}"
    
    # Create the folder if it doesn't exist
    os.makedirs(base_path, exist_ok=True)

    train_model(train_dir, validation_dir, 24, IMG_SIZE, dropout, 0.0001, 150, 50, 150, base_path, base_filename) """


# Optimization 01
base_filename = f"optimization_01"
base_path = fr"C:\Studium\Bachelor_Arbeit\USB_stick_niehuis\{base_filename}"

# Create the folder if it doesn't exist
os.makedirs(base_path, exist_ok=True)

train_model(train_dir, validation_dir, 32, IMG_SIZE, 0.3, 0.01, 5, 135, 5, base_path, base_filename)


