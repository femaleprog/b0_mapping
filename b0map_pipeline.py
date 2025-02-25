import os
import numpy as np
import pydicom as pdcm
from scipy.ndimage import gaussian_filter
import scipy.io
from skimage import restoration as sr
from scipy import ndimage
import nibabel as nib
from scipy.ndimage import median_filter


def load_dicom_files(folder_path: str) -> list[str]:
    """Load and sort DICOM files from a folder by instance number."""
    dicom_files = [
        os.path.join(folder_path, filename)
        for filename in sorted(os.listdir(folder_path))
        if filename.lower().endswith((".ima", ".dcm"))
    ]
    sort_indices = np.argsort(
        [float(pdcm.dcmread(dcm).InstanceNumber) for dcm in dicom_files]
    )
    return [dicom_files[i] for i in sort_indices]


def load_pixel_array(dicom_files: list[str], nechos: int = 3) -> np.ndarray:
    """Load pixel arrays from DICOM files and reshape by echoes."""
    echo_data = np.array(
        [pdcm.dcmread(dcm).pixel_array for dcm in dicom_files])
    reshaped_data = echo_data.reshape(
        nechos, len(echo_data) // nechos, *echo_data[0].shape
    )
    return np.moveaxis(np.moveaxis(reshaped_data, 0, -1), 0, -2)


def extract_echo_times(dicom_file: str) -> tuple[float, float, float]:
    """Extract echo times (TE0, TE1, TE2) from a DICOM file."""
    TE = []
    with open(dicom_file, "rb") as f:
        for line in f.readlines():
            if line[:7] in [b"alTE[0]", b"alTE[1]", b"alTE[2]"]:
                TE.append(int(line.decode("ascii").split()[-1]))
    if len(TE) != 3:
        raise ValueError(
            "Could not find all three echo times in DICOM header.")
    return TE[0], TE[1], TE[2]


def load_magnitude_from_dicom(folder_path: str) -> np.ndarray:
    """Load magnitude data from DICOM files."""
    dicom_files = load_dicom_files(folder_path)
    return load_pixel_array(dicom_files)


def get_phantom_mask_from_snr(SNR, thres=None):
    if thres is None:
        option = int(input(
            "Determine threshold (option 1) or number of voxels in Mask (option 2) ?\n"))
        if option == 1:
            thres = float(
                input("Enter SNR threshold to determine mask (above 1):\n"))
            print("WARNING: generating mask with the specified SNR-threshold")
            while thres != 1:
                mask = np.zeros_like(SNR, dtype=bool)
                mask[SNR > thres] = True
                thres = float(
                    input("Is the mask OK ? \n1 = yes. \nEnter new threshold otherwise: \n"))
        else:  # Number of voxels in Mask
            nvox = int(input("Enter desired number of voxels in mask:\n"))
            print("WARNING: generating mask with the specified number of voxels")
            thres = 10
            mask = np.ones_like(SNR, dtype=bool)
            while np.count_nonzero(mask) > nvox and thres < 100000:
                thres += 1
                mask = SNR > thres

    else:
        mask = np.zeros_like(SNR, dtype=bool)
        mask[SNR > thres] = True

    return mask


def create_mask(magnitude: np.ndarray, mask_type: str) -> np.ndarray:
    """Create a mask based on the specified method."""
    if mask_type == "Matlab":
        mat_data = scipy.io.loadmat(
            '/volatile/home/st281428/field_map/B0/_1/Mask.mat')
        return mat_data['Mask']
    elif mask_type == "Python":
        mean_amplitude = np.mean(magnitude, axis=-1)
        smoothed_amplitude = gaussian_filter(mean_amplitude, sigma=1)
        threshold = np.percentile(mean_amplitude, 62)
        return smoothed_amplitude > threshold
    elif mask_type == "Phantom":
        mask = get_phantom_mask_from_snr(np.sum(magnitude, axis=-1))
        return ndimage.binary_fill_holes(mask)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


def unwrap_phase_3d(field_map: np.ndarray, dte1: float, dte2: float) -> np.ndarray:
    """Estimate off-resonance (B0 map) using three echoes instead of two.

    This function calculates the B0 field map by unwrapping the phase difference
    between the first two echoes and refining the estimate with a third echo.
    It uses a linear regression over the echo times to improve accuracy compared
    to a two-echo approach, correcting for phase wraps based on theoretical predictions.

    Parameters
    ----------
    field_map : np.ndarray
        4D array of phase data with shape (nx, ny, nz, num_echoes), where the last
        dimension contains phase values for three echoes.
    dte1 : float
        Time difference between the first and second echo (in seconds).
    dte2 : float
        Time difference between the second and third echo (in seconds).

    Returns
    -------
    b0_map : np.ndarray
        3D array representing the B0 field map in Hz, estimated from three-echo data.
    """
    tau = np.array([0.0, dte1, dte1 + dte2])
    sum_tau, sum_tau2 = np.sum(tau), np.sum(tau * tau)

    diff = field_map[:, :, :, 1] - field_map[:, :, :, 0]
    dunwrapped = sr.unwrap_phase(diff) / (2 * np.pi)

    phase_turns = [field_map[:, :, :, 0] / (2 * np.pi), None, None]
    phase_turns[1] = phase_turns[0] + dunwrapped
    phase_turns[2] = field_map[:, :, :, 2] / (2 * np.pi)

    d2_theoretical = dunwrapped * (dte1 + dte2) / dte1
    phase_turns[2] += np.round(d2_theoretical)
    eps = (phase_turns[2] - phase_turns[0]) - d2_theoretical
    D2neg = (eps < -0.5)
    D2pos = (eps >= 0.5)
    phase_turns[2] += D2neg.astype(np.int32) - D2pos.astype(np.int32)

    sxy = tau[1] * phase_turns[1] + tau[2] * phase_turns[2]
    sy = phase_turns[0] + phase_turns[1] + phase_turns[2]
    return np.round((3 * sxy - sum_tau * sy) / (3 * sum_tau2 - sum_tau ** 2))


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


def estimate_b0_from_phase(phase_data: np.ndarray, dte1: float, dte2: float) -> np.ndarray:
    """Estimate B0 map from pre-loaded phase data across three echoes.

    This function takes a 4D phase array and computes the B0 field map using three
    echo times, improving off-resonance estimation compared to a two-echo approach.

    Parameters
    ----------
    phase_data : np.ndarray
        4D array of phase data with shape (num_echoes, nx, ny, nz), where num_echoes
        is typically 3, containing phase values in radians for each echo.
    dte1 : float
        Time difference between the first and second echo (in seconds).
    dte2 : float
        Time difference between the second and third echo (in seconds).

    Returns
    -------
    b0_map : np.ndarray
        3D array representing the B0 field map in Hz, estimated from the phase data.

    Raises
    ------
    ValueError
        If phase_data does not have exactly 3 echoes.
    """
    if phase_data.shape[0] != 3:
        raise ValueError(
            "phase_data must have exactly 3 echoes in the first dimension.")

    # Reorder dimensions to match unwrap_phase_3d expectation (nx, ny, nz, num_echoes)
    field_map = np.moveaxis(phase_data, 0, -1)

    # Call existing 3D unwrapping function
    b0_map = unwrap_phase_3d(field_map, dte1, dte2)
    return b0_map


def load_field_map_from_dicom(
    folder_path: str, magnitude_path: str, nx: int, ny: int, nz: int, num_echoes: int = 3, mask_type: str = "Phantom", neighborhood: list = [3, 3, 3], threshold: int = 8
) -> np.ndarray:
    """Load field map from DICOM files and compute B0 map.

    This function processes DICOM files to generate a B0 field map, which estimates
    off-resonance effects using multiple echo times. The number of echoes influences
    the unwrapping method and accuracy of the off-resonance estimation.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing phase DICOM files.
    magnitude_path : str
        Path to the folder containing magnitude DICOM files.
    nx : int
        Number of pixels in the x-dimension (rows).
    ny : int
        Number of pixels in the y-dimension (columns).
    nz : int
        Number of slices in the z-dimension.
    num_echoes : int, optional
        Number of echo times used to estimate off-resonance (default is 3).
        Typically 2 or 3 echoes; 3 allows for more robust estimation, while 2 is
        simpler but may be less accurate.
    mask_type : str, optional
        Type of mask to apply ("Matlab", "Python", or "Phantom"; default is "Phantom").

    Returns
    -------
    b0_map_clean : ndarray
        Cleaned B0 field map array after outlier removal.

    Raises
    ------
    ValueError
        If the unwrap method is unsupported or if echo times cannot be extracted.
    """
    # Load and preprocess DICOM data
    dicom_files = load_dicom_files(folder_path)
    field_map = load_pixel_array(dicom_files)
    # Read the first DICOM file to get metadata
    dcm = pdcm.dcmread(dicom_files[0])
    # Use Bits Stored to determine the maximum value (e.g., 2^12 = 4096)
    bits_stored = dcm.BitsStored
    phase_max = 2 ** bits_stored  # e.g., 4096 for 12 bits
    field_map = (field_map / phase_max) * (2 * np.pi) - \
        np.pi  # Convert to radians

    # Extract echo times
    te0, te1, te2 = extract_echo_times(dicom_files[0])
    dte1, dte2 = (te1 - te0) * 1e-6, (te2 - te1) * 1e-6
    print(f"TE0 = {te0}, TE1 = {te1}, TE2 = {te2}")
    print(f"Delta TE1 = {dte1} s, Delta TE2 = {dte2} s")

    # Load magnitude and create mask
    magnitude = load_magnitude_from_dicom(
        "/volatile/home/st281428/Downloads/B0Map/_1/A")
    mask = create_mask(magnitude, mask_type)
    mask_expanded = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
    field_map *= mask_expanded

    # Calculate B0 map
    if num_echoes == 3:
        b0_map = estimate_b0_from_phase(
            np.moveaxis(field_map, -1, 0), dte1, dte2)
    elif num_echoes == 2:
        diff = field_map[:, :, :, 1] - field_map[:, :, :, 0]
        turns = diff / (2 * np.pi)
        b0_map = (turns + (turns < -0.5) - (turns >= 0.5)) / dte1
    else:
        raise ValueError(
            f"Unsupported off resonance estimation method with specified number of echoes: {num_echoes}")

    # Clean outliers
    _, b0_map_clean = outlier_detection_fieldmap(
        b0_map, neighborhood, threshold)
    return b0_map_clean
