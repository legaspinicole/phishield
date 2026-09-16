import tensorflow as tf


def resize_dataset(ds, image_size=(32, 32)):
    """
    Resize every image in a dataset to image_size.

    Args:
        ds: Dataset from data.dataset_loader.load_dataset.
        image_size: Target (height, width). Use (32, 32) for the
            baseline CNN, or (224, 224) for models like EfficientNet.

    Returns:
        The dataset with resized images. Labels are unchanged.
    """

    def resize(images, labels):
        images = tf.image.resize(images, image_size, method="bilinear")
        return images, labels

    return ds.map(resize, num_parallel_calls=tf.data.AUTOTUNE)




def normalize_dataset(ds):
    """
    Scale pixel values from 0-255 to 0-1. Labels are unchanged.
    """

    def normalize(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels

    return ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)


def preprocess_dataset(ds, image_size=(32, 32), normalize=True):
    """
    Full preprocessing: resize, then (optionally) normalize.

    Args:
        ds: Dataset from data.dataset_loader.load_dataset.
        image_size: Target (height, width).
        normalize: True for the baseline CNN (pixels 0-1).
            False for Keras EfficientNet (it expects pixels 0-255).

    Returns:
        A preprocessed dataset, ready for training.
    """

    ds = resize_dataset(ds, image_size)
    if normalize:
        ds = normalize_dataset(ds)

    # Prepare the next batch while the model is busy with the current one
    return ds.prefetch(tf.data.AUTOTUNE)