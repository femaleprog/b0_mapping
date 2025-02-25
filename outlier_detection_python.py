import numpy as np
from scipy.ndimage import median_filter


def outlier_detection_fieldmap(B0map, neighsz, thresh):
    """Detect and filter outliers in a B0 field map using median filtering.

    Parameters
    ----------
    B0map : ndarray
        Input B0 field map array.
    neighsz : int or tuple of ints
        Size of the neighborhood for the median filter (e.g., 3 or (3, 3, 3)).
    thresh : float
        Threshold for detecting outliers based on absolute difference from filtered map.

    Returns
    -------
    outliers : ndarray
        Boolean array where True indicates an outlier.
    filtered_B0map : ndarray
        B0 map after applying the median filter.
    """
    filtered_B0map = median_filter(B0map, size=neighsz, mode='reflect')
    outliers = np.abs(B0map - filtered_B0map) > thresh
    return outliers, filtered_B0map
