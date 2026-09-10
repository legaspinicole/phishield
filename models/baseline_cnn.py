import tensorflow as tf
from tensorflow.keras import layers


def build_baseline_cnn(input_shape=(32, 32, 3)):
    """
    Baseline CNN for detecting AI-generated images.

    Input:
        RGB image

    Output:
        Probability that the image is AI-generated.
    """

    inputs = tf.keras.Input(
        shape=input_shape,
        name="rgb_input"
    )

    # First convolution block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        name="conv1"
    )(inputs)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool1"
    )(x)

    # Second convolution block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        name="conv2"
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool2"
    )(x)

    # Convert feature maps into one feature vector
    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    # Classification head
    x = layers.Dense(
        64,
        activation="relu",
        name="dense_64"
    )(x)

    x = layers.Dropout(
        0.30,
        name="dropout"
    )(x)

    # Binary classification:
    # 0 = REAL
    # 1 = AI-generated
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="prediction"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="baseline_cnn"
    )

    return model
