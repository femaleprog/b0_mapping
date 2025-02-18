import numpy as np


def compute_mse(image1, image2):
    """
    Function to compute the Mean Squared Error (MSE) between two images.

    Parameters:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.

    Returns:
        float: The Mean Squared Error between the two images.
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    mse = np.mean((image1 - image2) ** 2)
    print(f"MSE is : {mse}")
