import tensorflow as tf


def load_dataset(dataset_dir, image_size=(32, 32), batch_size=32):
    """
    Load the CIFAKE dataset from a directory.

    Expected directory structure:

        dataset_dir/
        ├── train/
        │   ├── FAKE/
        │   └── REAL/
        │
        └── test/
            ├── FAKE/
            └── REAL/

    Args:
        dataset_dir: Path to the CIFAKE dataset.
        image_size: Image dimensions used by the CNN.
        batch_size: Number of images per batch.

    Returns:
        train_ds: Training dataset.
        test_ds: Testing dataset.
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{dataset_dir}/train",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{dataset_dir}/test",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False
    )

    return train_ds, test_ds

