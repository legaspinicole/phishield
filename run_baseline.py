import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from data.dataset_loader import load_dataset, CLASS_NAMES
from preprocessing.preprocess import preprocess_dataset
from models.baseline_cnn import build_baseline_cnn

# The CIFAKE dataset is inside this repo
DATASET_DIR = "data"
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32

# 1. Load and label (0 = REAL, 1 = FAKE), with a validation split
train_ds, val_ds, test_ds = load_dataset(DATASET_DIR, batch_size=BATCH_SIZE)
print("Label mapping:", {name: i for i, name in enumerate(CLASS_NAMES)})

# 2. Preprocess: resize and normalize
train_ds = preprocess_dataset(train_ds, IMAGE_SIZE)
val_ds = preprocess_dataset(val_ds, IMAGE_SIZE)
test_ds = preprocess_dataset(test_ds, IMAGE_SIZE)

# 3. Build the baseline CNN
model = build_baseline_cnn(input_shape=(*IMAGE_SIZE, 3))
model.summary()

# 4. Quick check that the data fits the model (not real training)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds,
          epochs=1, steps_per_epoch=20, validation_steps=5)

print("Pipeline check finished.")