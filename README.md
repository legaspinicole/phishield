# phishield

## Data Preprocessing

**Dataset loader** (`data/dataset_loader.py`)
- Loads the CIFAKE dataset from `data/train` and `data/test`
- Labels: `REAL = 0`, `FAKE = 1` (matches `models/baseline_cnn.py`)
- `load_dataset("data")` returns **three** datasets: `train_ds, val_ds, test_ds`
  - Train: 80,000 images / Validation: 20,000 (20% of train, seed 42) / Test: 20,000

**Preprocessing** (`preprocessing/preprocess.py`)
- `preprocess_dataset(ds)`: resize to 32×32 and normalize pixels to 0–1 (baseline CNN)
- `preprocess_dataset(ds, image_size=(224, 224), normalize=False)`: for Keras EfficientNet

**Setup**
- Python 3.10–3.13 (TensorFlow does not support 3.14 yet)
- `pip install tensorflow opencv-python`
- Run `python run_baseline.py` to check the pipeline