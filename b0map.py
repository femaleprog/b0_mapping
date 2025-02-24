import os
import numpy as np
import pydicom as pdcm
from scipy.ndimage import gaussian_filter
import scipy.io
from skimage import restoration as sr
from visualize_b0map import visualize_map
from outlier_detection_python import outlier_detection_fieldmap
from scipy import ndimage
import get_phantom_mask_from_snr_map as gpm


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
        mask = gpm.get_phantom_mask_from_snr(np.sum(magnitude, axis=-1))
        return ndimage.binary_fill_holes(mask)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


def unwrap_phase_3d(field_map: np.ndarray, dte1: float, dte2: float) -> np.ndarray:
    """Perform 3D phase unwrapping and calculate B0 map."""
    tau = np.array([0.0, dte1, dte1 + dte2])
    sum_tau, sum_tau2 = np.sum(tau), np.sum(tau * tau)

    diff = field_map[:, :, :, 1] - field_map[:, :, :, 0]
    dunwrapped = sr.unwrap_phase(diff) / (2 * np.pi)

    D = [field_map[:, :, :, 0] / (2 * np.pi), None, None]
    D[1] = D[0] + dunwrapped
    D[2] = field_map[:, :, :, 2] / (2 * np.pi)

    d2_theoretical = dunwrapped * (dte1 + dte2) / dte1
    D[2] += np.round(d2_theoretical)
    eps = (D[2] - D[0]) - d2_theoretical
    D2neg = (eps < -0.5)
    D2pos = (eps >= 0.5)
    D[2] += D2neg.astype(np.int32) - D2pos.astype(np.int32)

    sxy = tau[1] * D[1] + tau[2] * D[2]
    sy = D[0] + D[1] + D[2]
    return np.round((3 * sxy - sum_tau * sy) / (3 * sum_tau2 - sum_tau ** 2))


def load_field_map_from_dicom(
    folder_path: str, nx: int, ny: int, nz: int, unwrap: str = "3D", mask_type: str = "Phantom"
) -> np.ndarray:
    """Load field map from DICOM files and compute B0 map."""
    # Load and preprocess DICOM data
    dicom_files = load_dicom_files(folder_path)
    field_map = load_pixel_array(dicom_files)
    field_map = (field_map / 4096) * (2 * np.pi) - np.pi  # Convert to radians

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
    if unwrap == "3D":
        b0_map = unwrap_phase_3d(field_map, dte1, dte2)
    elif unwrap == "2D":
        diff = field_map[:, :, :, 1] - field_map[:, :, :, 0]
        turns = diff / (2 * np.pi)
        b0_map = (turns + (turns < -0.5) - (turns >= 0.5)) / dte1
    else:
        raise ValueError(f"Unsupported unwrap method: {unwrap}")

    neighborhood = [3, 3, 3]
    threshold = 8
    # Clean outliers
    _, b0_map_clean = outlier_detection_fieldmap(
        b0_map, neighborhood, threshold)
    return b0_map_clean


if __name__ == "__main__":
    # Parameters
    NX, NY, NZ = 256, 256, 64
    FIELD_MAP_PATH = "/volatile/home/st281428/Downloads/B0Map/_1/P"

    # Compute B0 map
    b0_map = load_field_map_from_dicom(
        FIELD_MAP_PATH, NX, NY, NZ, unwrap="3D", mask_type="Phantom")
    visualize_map(b0_map, slice_index=22, vmin=-600,
                  vmax=600, subtitle="Cleaned B0map")
