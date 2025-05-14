import os
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance

from data_processing_module import load_csv_to_dataframe

""" takes all .tsv files from one folder and converts them to 244x244 images """

# Folder containing the files
folder_path = r'C:\Studium\Bachelor_Arbeit\data\females_integr\01_Area_of_Intrest'
out_folder_path = r'c:\Studium\Bachelor_Arbeit\data\females_integr\02_244x244'

# Get all TSV files in the folder
file_paths = glob.glob(os.path.join(folder_path, '*.tsv'))
for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
    df = load_csv_to_dataframe(file_path)
    df = df.set_index('Kovats')
    # Convert DataFrame to image
    data = df.to_numpy()
    data = (data - data.min()) / (data.max() - data.min()) # min-max normalized 
    data = data * 255 # multiplied to make it visible as image
    image = Image.fromarray(data).convert('RGB')  # Convert to RGB mode
    resized_image = image.resize((244, 244)) # correct size for inceptionV3

    # Save the resized array in the new location
    output_file_path = os.path.join(out_folder_path, os.path.basename(file_path).replace('.tsv', '_resized.png'))
    resized_image.save(output_file_path)
