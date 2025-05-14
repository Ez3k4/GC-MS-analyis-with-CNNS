#!/usr/bin/env python3
import shutil
import glob
import os

""" 
Script to distibute samples to train and validation folders. 
It creates subfolders with class names that contain the samples.
"""

# Folder containing the files
folder_path = fr'test'

file_extension = '*.png'

# Get all files in the folder with extension
file_paths = glob.glob(os.path.join(folder_path, file_extension))
print(file_paths)

a_count = b_count = 0

label_1 = '7105'
label_2 = '9495'

# Process each file
for file_path in file_paths:
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)

    if label_1 in file_path:
        b_count += 1
    elif label_2 in file_path:
        a_count += 1
print("A_count:", a_count, "B count:", b_count)

# split the dataset in train and validation
train_amount = [4 * (a_count // 5), 4 * (b_count // 5)]
print(train_amount)



# output folder
dir_train = fr'.\train'
dir_valid = fr'.\validation'
# Create target directories if they don't exist
os.makedirs(dir_train, exist_ok=True)# Create target directories if they don't exist
os.makedirs(dir_valid, exist_ok=True)

# Define target directories
label_1_train = os.path.join(dir_train, label_1)
label_1_valid = os.path.join(dir_valid, label_1)
label_2_train = os.path.join(dir_train, label_2)
label_2_valid = os.path.join(dir_valid, label_2)

# Create target directories if they don't exist
os.makedirs(label_1_train, exist_ok=True)
os.makedirs(label_1_valid, exist_ok=True)
os.makedirs(label_2_train, exist_ok=True)
os.makedirs(label_2_valid, exist_ok=True)


for file_path in file_paths:
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    
    if label_1 in file_path:
        if train_amount[1] > 0:
            target_path = os.path.join(label_1_train, base_name)
            print(f"Copying {base_name} to train ({label_1})")
            train_amount[1] -= 1
        else:
            target_path = os.path.join(label_1_valid, base_name)
            print(f"Copying {base_name} to valid ({label_2})")
        shutil.copy(file_path, target_path)
    elif label_2 in file_path:
        if train_amount[0] > 0:
            target_path = os.path.join(label_2_train, base_name)
            print(f"Copying {base_name} to train (9495)")
            train_amount[0] -= 1
        else:
            target_path = os.path.join(label_2_valid, base_name)
            print(f"Copying {base_name} to valid (9495)")
        shutil.copy(file_path, target_path)
print("A count:", a_count, "B_count:", b_count)