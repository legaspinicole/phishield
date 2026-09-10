import cv2
import numpy as np


def create_sobel_edge_map(image):
    """
    Generate a Sobel edge map from an RGB image.

    Based on the Sobel edge extraction approach used in EGFNet.

    Args:
        image: RGB image as a NumPy array.

    Returns:
        Grayscale Sobel edge map as a NumPy array.
    """

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate horizontal and vertical Sobel gradients
    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)

    # Convert gradients to absolute 8-bit values
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)

    # Combine horizontal and vertical gradients
    edge_map = cv2.addWeighted(
        abs_x,
        0.5,
        abs_y,
        0.5,
        0
    )

    return edge_map

