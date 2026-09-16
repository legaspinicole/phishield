import tensorflow as tf

# 0 = REAL, 1 = FAKE (AI-generated), to match models/baseline_cnn.py.
# Without this, Keras sorts folders alphabetically and gives FAKE = 0.
CLASS_NAMES = ["REAL", "FAKE"]


def load_dataset(dataset_dir, image_size=(32, 32), batch_size=32,
                 val_split=0.2, seed=42):
    """
    Load the CIFAKE dataset and split it into train, validation, and test.

    Expected directory structure:

        dataset_dir/
        ├── train/
        │   ├── FAKE/
        │   └── REAL/
        └── test/
            ├── FAKE/
            └── REAL/

    Args:
        dataset_dir: Path to the CIFAKE dataset (e.g. "data").
        image_size: Size the images are loaded at.
        batch_size: Number of images per batch.
        val_split: Fraction of train/ held out for validation (0.2 = 20%).
        seed: Fixed random seed so the split is the same on every run.

    Returns:
        train_ds, val_ds, test_ds
    """

    # Split train/ into training and validation sets in one call
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{dataset_dir}/train",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        class_names=CLASS_NAMES,
        validation_split=val_split,
        subset="both",
        seed=seed,
        shuffle=True
    )

    # The official test set is kept separate for final evaluation
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{dataset_dir}/test",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        class_names=CLASS_NAMES,
        shuffle=False
    )

    return train_ds, val_ds, test_ds