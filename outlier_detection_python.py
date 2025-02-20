import numpy as np
from scipy.ndimage import median_filter


def outlier_detection_fieldmap(B0map, neighsz, thresh):
    # Apply a median filter to the B0map directly
    filtered_B0map = median_filter(B0map, size=neighsz, mode='reflect')

    # Detect outliers by thresholding the absolute difference
    outliers = np.abs(B0map - filtered_B0map) > thresh

    return outliers, filtered_B0map
