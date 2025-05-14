import pandas as pd
import os
import re

""" Renames samples that miss the barcode in their filename """

# Specify the path to your Excel file
excel_file_path = r'metadata.xlsx'
# Read the Excel file
df = pd.read_excel(excel_file_path)

# Extract the 'Sample tube' and 'barcoded' columns
extracted_df = df[['Sample tube', 'barcoded']]

# Create a dictionary mapping sample identifiers to barcodes
identifier_to_barcode = dict(zip(extracted_df['Sample tube'], extracted_df['barcoded']))

# Specify the folder containing the files
folder_path = r'sample_folder'

# Get all TSV files in the folder
file_paths = [f for f in os.listdir(folder_path) if f.endswith('.tsv')]
#print(file_paths)

x = 0
# Process each file
for file_name in file_paths:
    if "7105" in file_name or "9495" in file_name:
        continue
    else:
        # Extract the sample identifier from the file name
        parts = re.split(r'[_.]', file_name)
        print(parts)
        sample_identifier = parts[0] + '_' + parts[1]
        x += 1


        # print(file_name)
        barcode = identifier_to_barcode.get(sample_identifier, None)

        if barcode:
            # Construct the new file name
            new_file_name = f"{sample_identifier}_{barcode}_int.tsv"
            print(new_file_name)

            # Rename the file
            os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))
            print(f"Renamed {file_name} to {new_file_name}")
print(x)