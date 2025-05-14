# Gas-Chromatography Mass-Spectrometry (GC-MS) Data Analysis with CNN

This repository contains scripts for preprocessing GC-MS data and analyzing it using MobileNetV2.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Data Preprocessing](#data-preprocessing)
3. [Training the Model](#training-the-model)
4. [Augmentation](#augmentation)
5. [References](#references)

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Ez3k4/BA_clean
cd BA_clean
```

### 2. Set Up Virtual Environment
- **Using VS Code**:
  1. Open VS Code.
  2. Press `Ctrl+Shift+P` → Select `Create Environment` → Choose `Venv`.
  3. Press `Ctrl+Shift+P` → Select Interpreter → Choose `Python (3.10.11)`.

- **Manual Setup**:
  ```bash
  # macOS/Linux
  python3 -m venv .venv
  source .venv/bin/activate

  # Windows
  python -m venv .venv
  .venv\Scripts\activate
  ```


### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Data Preprocessing

### Input Format
Your data should be in `.tsv` format, structured as follows:
```
     Kovats     m/z 29     m/z 30     m/z 31     m/z 32    m/z 33    m/z 34  ...   m/z 557   m/z 558   m/z 559   m/z 560   m/z 561   m/z 562   m/z 563
0      800.0  13.534379  17.485770  21.608210   6.947083  4.223339  1.169267  ...  1.961170  0.394086  0.974702  0.201974  0.791817  1.070208  0.000000
1      801.0  13.284482  13.126891  18.118557   6.317664  2.763806  1.140904  ...  2.444894  0.251929  0.306259  0.738431  0.337725  1.270047  0.000000
...      ...        ...        ...        ...        ...       ...       ...  ...       ...       ...       ...       ...       ...       ...       ...
2898  3698.0   0.153129   0.769936   0.074111  11.572682  1.839748  1.864619  ...  2.906755  3.458679  2.144597  2.396383  1.419776  0.755343  0.000000
2899  3699.0   0.162683   0.563338   0.000000  11.323170  3.679775  2.245420  ...  1.007997  0.580320  2.573870  2.005170  4.231314  1.941761  0.000000

```

### Preprocessing Steps
1. **Crop Data**: Use `crop_data.py` to trim unnecessary parts.
2. **Standardize**: Use `std_normalization.py` for standardization.
3. **Normalize**: Use `integral_normalization.py` for normalization.

### Save as Images
Convert processed data into 244x244 images using `data_to_img.py`.

---

## Training the Model

1. Prepare training and validation datasets:
   - Use `add_barcode_to_name.py` to label files.
   - Use `sort_to_folder.py` to organize files into `train` and `validation` folders.

2. Train the model:
   ```bash
   python automated_training.py
   ```

---

## Augmentation

Generate augmented datasets using:
- `augmentation_pipeline.py`
- Individual functions from the `augmentation` module
- External tools (e.g., TensorFlow)

---

## References
- [MobileNetV2 Documentation](https://arxiv.org/abs/1801.04381)
- [TensorFlow Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)