#!/usr/bin/env python3
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance

def load_csv_to_dataframe(file_path):
    df = pd.read_csv(file_path, delimiter='\t')
    return df

def truncate_df(df, start_row, end_row, start_col, end_col):
    df_tru = df.iloc[start_row:end_row, start_col:end_col]
    return df_tru

def kovats_indices(df):
    """ function that sets kovats as indices and drops NaNs"""
    # Drop rows with NaN values in the 'Kovats' column
    df = df.dropna(subset=['Kovats'])

    # Set 'Kovats' as the index
    df_kov = df.set_index('Kovats')
    return df_kov

def normalize_mz_values_min_max(df, start_col=None, end_col=None):
    """
    Normalize the m/z values to be between 0 and 1 using Min-Max normalization.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the m/z values to be normalized.
    start_col (int): The starting column index for m/z values.
    end_col (int): The ending column index for m/z values.
    
    Returns:
    pd.DataFrame: DataFrame with normalized m/z values.
    """
    mz_columns = df.columns[start_col:end_col]
    df[mz_columns] = (df[mz_columns] - df[mz_columns].min()) / (df[mz_columns].max() - df[mz_columns].min())
    return df

def normalize_mz_values_std(df, start_col, end_col):
    """
    Normalize the m/z values by dividing them by the standard deviation after substracting the mean.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the m/z values to be normalized.
    
    Returns:
    pd.DataFrame: DataFrame with normalized m/z values.
    """
    mz_columns = df.columns[start_col:end_col]
    df[mz_columns] = (df[mz_columns] - df[mz_columns].mean()) / df[mz_columns].std(ddof=0) # <--- this is fully standarized
    # stand_scaled = StandardScaler().fit_transform(df)
    # df[mz_columns] = df[mz_columns].div(df[mz_columns].std(ddof=0)) # ddof=0 for std(n-0)
    return df

def normalize_mz_values_int(df):
    data = bin_kovats_df(df) # bins the Kovats to full integers, calculates the mean if multiple were binned to the same
    data = data.set_index("Kovats")
    print(data)
    # Calculate the sum of all rows and columns
    row_sums = data.sum(axis=1)  # Sum of each row
    integral = row_sums.sum(axis=0)

    data = data / integral
    data = data.reset_index()
    return data

def save_df(df, file_path):
    df.to_csv(file_path, sep='\t', index=False)

def bin_kovats_df(df):
    """ Rounds dataframe DOWN to whole Kovats and groups them by their Kovats Values.
     Then calculates the mean of grouped rows and drops the scan times column 
     (average scan time values make no sense).
      Returns a new dataset """
    # Round the 'Kovats' column down to the nearest integer
    df['Kovats'] = df['Kovats'].apply(np.floor)

    df = df.drop(columns=['scan times [s]'])

    df = df.dropna(subset=['Kovats'])

    # Group by the rounded 'Kovats' values and calculate the mean for each group
    df_grouped = df.groupby('Kovats').mean().reset_index()
    
    return df_grouped
    
def df_to_image(df, res, brightness=1):
    data = df.to_numpy()
    data = (data - data.min()) / (data.max() - data.min())  # Min-max normalization
    print(data)
    data = data * 255  # Scale to 0-255 for image representation
    print(data)
    image = Image.fromarray(data).convert('RGB').resize((res, res))  # Convert to RGB mode
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)  # Apply brightness adjustment
    return image

if __name__ == "__main__":
    # Example usage
    file_path1 = r''
    file_path2 = r''

    # read_n_lines(file_path, 2)
    df = load_csv_to_dataframe(file_path1)
    print(df)
 
    df2 = load_csv_to_dataframe(file_path2).dropna()
    # print(df.head(25))
    # print(df2.head(25))
    
    df_tru = truncate_df(df, 2450, 3550, 1, 300)
    print(df_tru)
