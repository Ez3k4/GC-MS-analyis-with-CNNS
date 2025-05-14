#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image, ImageEnhance


def load_csv_to_dataframe(file_path):
    df = pd.read_csv(file_path, delimiter='\t')
    return df

def sinusoidal_shift(df, amp, freq, phase_shift):
    """ adds sinusoidal changes in x axis direction """
    # Retention indices (y-axis)
    x = df.index.to_numpy()

    # Apply the sinusoidal function to each column (m/z channel)
    df_augmented = df.copy()
    for column in df.columns:
        sin_wave = amp * np.sin(2 * np.pi * freq * x + phase_shift)
        df_augmented[column] += sin_wave  # Add the sine wave to the column values

    return df_augmented

def shift_index(df, shift_amount):
    """
    Shifts the data in the index direction by a specified amount.

    Parameters:
    - df: The input DataFrame.
    - shift_amount: The number of steps to shift the data. Positive values shift down, negative values shift up.

    Returns:
    - A new DataFrame with the data shifted in the index direction.
    """
    # Shift the data along the index
    df_shifted = df.shift(periods=shift_amount, axis=0)
    return df_shifted

def multiply_values(df, factor):
    """
    Multiplies each value in the DataFrame (excluding the index) by a specified factor.

    Parameters:
    - df: The input DataFrame.
    - factor: The multiplication factor.

    Returns:
    - A new DataFrame with the values multiplied by the factor.
    """
    # Multiply all values in the DataFrame by the factor
    df_multiplied = df.copy()
    df_multiplied = df_multiplied * factor
    return df_multiplied

def df_to_image(df, res, brightness=1):
    data = df.to_numpy()
    # print("Before Normalization - Min:", data.min(), "Max:", data.max())
    
    data = (data - data.min()) / (data.max() - data.min())  # Min-max normalization
    # print("After Normalization - Min:", data.min(), "Max:", data.max())
    
    data = data * 255  # Scale to 0-255 for image representation
    # print("After Scaling - Min:", data.min(), "Max:", data.max())
    
    image = Image.fromarray(data).convert('RGB').resize((res, res))  # Convert to RGB mode
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)  # Apply brightness adjustment
    return image


if __name__ == "__main__":
    input_file = r'sample.tsv'

    df = load_csv_to_dataframe(input_file)
    print(df)

    # Set the 'Kovats' column as the index
    df.set_index('Kovats', inplace=True)

    # Define the desired index interval
    start_index = 2650
    end_index = 3449

    df_filtered = df.loc[start_index:end_index]

    # Apply a sinusoidal function for data augmentation
    amplitude = 0.00005 # Amplitude of the sine wave (10% of the max value across all columns)
    frequency = 0.005  # Frequency of the sine wave
    phase_shift = np.pi / 2  # Phase shift in radians (e.g., π/4 for 45°)

    df_sin = sinusoidal_shift(df, amplitude, frequency, phase_shift)
    df_sin_filtered = df_sin.loc[start_index:end_index]

    df_transl = shift_index(df, 100)
    df_transl_filtered = df_transl.loc[start_index:end_index]

    df_intens = multiply_values(df, 2)
    df_intens_filtered = df_intens.loc[start_index:end_index]

    # Plotting the heatmaps side by side
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    # Plot the first DataFrame (original data)
    sns.heatmap(df_filtered, cmap='viridis', ax=axes[0, 0], cbar=True)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlabel('m/z values')
    axes[0, 0].set_ylabel('Retention index (Kovats)')

    # Plot the second DataFrame (sinusoidal augmentation)
    sns.heatmap(df_sin_filtered, cmap='viridis', ax=axes[0, 1], cbar=True)
    axes[0, 1].set_title('Sinusoidal Data Augmentation')
    axes[0, 1].set_xlabel('m/z values')
    axes[0, 1].set_ylabel('Retention index (Kovats)')

    # Plot the third DataFrame (translational augmentation)
    sns.heatmap(df_transl_filtered, cmap='viridis', ax=axes[1, 0], cbar=True)
    axes[1, 0].set_title('Translational Data Augmentation')
    axes[1, 0].set_xlabel('m/z values')
    axes[1, 0].set_ylabel('Retention index (Kovats)')

    # Plot the fourth DataFrame (intensity augmentation)
    sns.heatmap(df_intens_filtered, cmap='viridis', ax=axes[1, 1], cbar=True)
    axes[1, 1].set_title('Intensity Data Augmentation')
    axes[1, 1].set_xlabel('m/z values')
    axes[1, 1].set_ylabel('Retention index (Kovats)')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

