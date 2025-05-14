#!/usr/bin/env python3
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from augmentation_module import sinusoidal_shift, shift_index, df_to_image, load_csv_to_dataframe

number = 100

folder_path = fr'C:\Studium\Bachelor_Arbeit\data\bloated_int\{number}\df'
output_folder = fr"c:\Studium\Bachelor_Arbeit\data\bloated_int\{number}\img"

# Define the desired index interval
start_index = 2650
end_index = 3449

# Get all TSV files in the folder
file_paths = glob.glob(os.path.join(folder_path, '*.tsv'))
for file_path in file_paths[0:1]:
    # Initialize a counter
    counter = 0

    df = load_csv_to_dataframe(file_path)
    print(df)
    # Set the 'Kovats' column as the index
    df.set_index('Kovats', inplace=True)

    for shift in range(-5, 6, 5):
        df_shifted = shift_index(df, shift)

        for set in [(0.00002, np.pi, 0.005),(0.00002, np.pi / 2, 0.05),(0.00002, np.pi / 0.75, 0.5)]:
            amp, shift, freq = set

            df_sin = sinusoidal_shift(df_shifted, amp, freq, shift)
            df_filtered = df_sin.loc[start_index:end_index]

            for factor in [0.9, 1, 1.1]:
                
                res = 244
                image = df_to_image(df_filtered, res, factor)

                output_file_path = os.path.join(output_folder, os.path.basename(file_path).replace('.tsv', f'_{counter}.png'))
                print(output_file_path)
                image.save(output_file_path)
                # image.show()

                # Increment the counter for specifc filenames
                counter += 1
