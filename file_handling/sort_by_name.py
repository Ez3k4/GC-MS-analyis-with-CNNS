#!/usr/bin/env python3
import shutil
import glob
import os

""" This script allows you to sort a specific number of samples in a folder """

# Folder containing the files
folder_path = r'/sample_folder'

file_extension = '*.png'

# Get all TSV files in the folder
file_paths = glob.glob(os.path.join(folder_path, file_extension))
print(file_paths)
print(type(file_paths))

amount = 100

target_path = r'output_folder'

label_1 = '7105'
label_2 = '9495'

a_count = b_count = int(amount/2)
# Process each file
file_paths.reverse()
for file_path in file_paths:
    if a_count <= 0 and b_count <= 0:
        break  # Stop processing when both counts reach zero

    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)

    if label_1 in file_path and b_count > 0:
        b_count -= 1
        print(f"Copying {base_name} to {target_path} (Label: {label_1})")
        shutil.copy(file_path, target_path)
    elif label_2 in file_path and a_count > 0:
        a_count -= 1
        print(f"Copying {base_name} to {target_path} (Label: {label_2})")
        shutil.copy(file_path, target_path)

    print("A_count:", a_count, "B_count:", b_count)

print("File processing complete.")

