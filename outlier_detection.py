import numpy as np
from scipy.ndimage import median_filter


def outlier_detection_fieldmap(Mask, Map, neighsz, thresh, MaskClean):
    # Create a 3D grid
    grid_shape = Mask.shape
    g = np.array(np.meshgrid(
        np.arange(grid_shape[0]),
        np.arange(grid_shape[1]),
        np.arange(grid_shape[2]),
        indexing='ij'
    ))
    r = g.reshape(3, -1).T  # Vectorize grid

    # Define parameters
    param = {
        'minNeighSize': 1,
        'thresh': thresh
    }

    w = np.round(np.array(neighsz)).astype(int).reshape(3, 1)
    if np.isscalar(w):
        w = np.tile(np.array(neighsz), (2, 1))

    else:
        w = np.hstack([w, w]).T

    assert w.shape == (2, 3), 'Bad input argument neighsz'

    # Outlier detection using median filtering
    Map_real = np.real(Map)
    median_filtered = median_filter(Map_real * MaskClean, size=tuple(w[1]))
    outlier = np.abs(Map_real - median_filtered) > thresh
    MapF = median_filtered

    if not np.isrealobj(Map):
        Map_imag = np.imag(Map)
        median_filtered_imag = median_filter(
            Map_imag * MaskClean, size=tuple(w[1]))
        outlier_imag = np.abs(Map_imag - median_filtered_imag) > thresh
        outlier |= outlier_imag
        MapF = MapF + 1j * median_filtered_imag

    return outlier, MapF
