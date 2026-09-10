import tensorflow as tf

from data.dataset_loader import load_dataset
from models.baseline_cnn import build_baseline_cnn


# Path to your CIFAKE dataset
DATASET_DIR = "path/to/CIFAKE"

# Load the dataset
train_ds, test_ds = load_dataset(
    DATASET_DIR,
    image_size=(32, 32),
    batch_size=32
)

# Build the baseline CNN
model = build_baseline_cnn(
    input_shape=(32, 32, 3)
)

# Display the model architecture
model.summary()

