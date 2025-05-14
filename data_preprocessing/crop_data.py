#!/usr/bin/env python3
import os
import glob
import sys
from tqdm import tqdm

from data_processing_module import load_csv_to_dataframe, save_df

""" saves truncated data in specified folder, df has this format for area of intrest [2650:3449, None:None]

    Kovats    m/z 29    m/z 30    m/z 31    m/z 32    m/z 33    m/z 34    m/z 35    m/z 36    m/z 37    m/z 38  ...   m/z 553   m/z 554   m/z 555   m/z 556   m/z 557   m/z 558   m/z 559   m/z 560   m/z 561   m/z 562   m/z 563
0    2650.0 -0.263796 -0.361056 -0.332058  0.506795 -0.957439 -0.027638  0.554843 -0.471532  0.022599 -0.503844  ... -0.908225  0.759572 -0.928779 -0.402983  0.021767 -0.416357 -0.199056 -0.536583 -0.438629  2.494526 -0.309458
1    2651.0 -0.243725  0.102876 -0.523628  0.757993  2.733448  0.215802 -0.408298 -0.080300 -0.763798 -0.518484  ... -0.383611 -0.899631 -0.500946 -0.470444 -0.353549 -1.037472  1.246043 -0.538668 -0.663696 -0.225722 -0.309458
..      ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...  ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
798  3448.0 -0.195262 -0.172963 -0.311951  0.531059  2.095470  0.889123 -0.013783  3.428360 -0.082920 -0.231597  ...  2.308907 -0.108751  1.139334 -0.754562  0.674840 -0.368751 -0.524248 -0.588792  1.258867 -0.590019 -0.309458
799  3449.0 -0.159536  0.301000 -0.093524  1.336805  0.753749  1.339328 -0.792485  2.671805 -0.486839 -0.596050  ...  4.726499 -0.183609 -0.652732  1.730131 -0.902229  2.129686 -0.538536 -1.065255  3.274813  1.620260 -0.309458
"""

input_folder = r'./tsv_folder'
output_folder = r'./crop_folder'

# Get the list of all .tsv files in the input folder
file_paths = glob.glob(os.path.join(input_folder, '*.tsv'))
# print(file_paths)

# Iterate over the files with a progress bar
for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
    df = load_csv_to_dataframe(file_path)
    df.set_index('Kovats', inplace=True)
    # SET AREA OF INDEX HERE
    df = df.loc[2650:3449, None:None]
    df = df.reset_index()
    # print(df)

    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)

    print(file_paths.index(file_path))
    if file_paths.index(file_path) == 0:
        print(df)
        answer = input(f'Do you want to save all .tsv files from {input_folder} normalized at {output_folder}? (y/n)')

    if answer.lower() == 'y':
        save_df(df, os.path.join(output_folder, f'{name}_crop{ext}'))
        print(f'{name} was standarized, saved as {name}_crop{ext} at {output_folder}')
    else:
        print("exit")
        break

print("Processing complete.")

