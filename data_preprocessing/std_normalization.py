#!/usr/bin/env python3
import glob
import os
from tqdm import tqdm

from data_processing_module import load_csv_to_dataframe, bin_kovats_df, normalize_mz_values_std, save_df

""" This program normalizes (zscore) all tsv files in one folder and saves them in specified folder.
    The result should look like this.
     Kovats     m/z 29     m/z 30     m/z 31     m/z 32    m/z 33    m/z 34  ...   m/z 557   m/z 558   m/z 559   m/z 560   m/z 561   m/z 562   m/z 563
0      800.0  13.534379  17.485770  21.608210   6.947083  4.223339  1.169267  ...  1.961170  0.394086  0.974702  0.201974  0.791817  1.070208  0.000000
1      801.0  13.284482  13.126891  18.118557   6.317664  2.763806  1.140904  ...  2.444894  0.251929  0.306259  0.738431  0.337725  1.270047  0.000000
...      ...        ...        ...        ...        ...       ...       ...  ...       ...       ...       ...       ...       ...       ...       ...
2898  3698.0   0.153129   0.769936   0.074111  11.572682  1.839748  1.864619  ...  2.906755  3.458679  2.144597  2.396383  1.419776  0.755343  0.000000
2899  3699.0   0.162683   0.563338   0.000000  11.323170  3.679775  2.245420  ...  1.007997  0.580320  2.573870  2.005170  4.231314  1.941761  0.000000

[2900 rows x 536 columns]
 """

input_folder = r'./tsv_folder'
output_folder = r'./std_folder'
# Get the list of all .tsv files in the input folder
file_paths = glob.glob(os.path.join(input_folder, '*.tsv'))

# Iterate over the files with a progress bar
for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
    df = load_csv_to_dataframe(file_path)
    df = bin_kovats_df(df) # bins the Kovats to full integers, calculates the mean if multiple were binned to the same
    df = normalize_mz_values_std(df, 1, None) # normalizes by dividing all values by the std(n-0)
    
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)

    print(file_paths.index(file_path))
    if file_paths.index(file_path) == 0:
        print(df)
        answer = input(f'Do you want to save all .tsv files from {input_folder} normalized at {output_folder}? (y/n)')

    if answer.lower() == 'y':
        save_df(df, os.path.join(output_folder, f'{name}_std{ext}'))
        print(f'{name} was standarized, saved as {name}_std{ext} at {output_folder}')
    else:
        print("exit")
        break

print("Processing complete.")
