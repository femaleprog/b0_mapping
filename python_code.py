import os
import numpy as np
import pydicom as pdcm
from skimage import transform as st
from skimage import restoration as sr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_opening
import scipy
import cv2
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from visualize_b0map import view_B0map, visualize_map
from calculate_mse import compute_mse
from outlier_detection_python import outlier_detection_fieldmap
from scipy import ndimage


def load_magnitude_from_dicom(folder_path):
    # Read and sort DICOM files
    fm_dcms = [os.path.join(folder_path, filename)
               for filename in sorted(os.listdir(folder_path))
               if (filename[-4:].lower() in [".ima", ".dcm"])]
    # sort files by instance number
    sort = np.argsort([float(pdcm.dcmread(dcm).InstanceNumber)
                      for dcm in fm_dcms])
    fm_dcms = [fm_dcms[i] for i in sort]

    # group files by echo
    nechos = 3
    echo_data = np.array([pdcm.dcmread(dcm).pixel_array for dcm in fm_dcms])
    magnitude = echo_data.reshape(nechos, len(
        echo_data)//nechos, *echo_data[0].shape)
    magnitude = np.moveaxis(magnitude, 0, -1)
    magnitude = np.moveaxis(magnitude, 0, -2)
    return magnitude


def load_field_map_from_dicom(folder_path, Nx, Ny, Nz, unwrap=False, Mask="Python", reference=None):
    # Read and sort DICOM files
    fm_dcms = [os.path.join(folder_path, filename)
               for filename in sorted(os.listdir(folder_path))
               if (filename[-4:].lower() in [".ima", ".dcm"])]

    # Sort files by instance number
    sort = np.argsort([float(pdcm.dcmread(dcm).InstanceNumber)
                      for dcm in fm_dcms])
    fm_dcms = [fm_dcms[i] for i in sort]

    # Group files by echo
    nechos = 3  # Assuming 3 echoes
    # Load and reshape the data
    echo_data = np.array([pdcm.dcmread(dcm).pixel_array for dcm in fm_dcms])
    field_map = echo_data.reshape(nechos, len(
        echo_data)//nechos, *echo_data[0].shape)
    field_map = np.moveaxis(field_map, 0, -1)
    field_map = np.moveaxis(field_map, 0, -2)

    # Get metadata from the first DICOM file
    fm_data = pdcm.dcmread(fm_dcms[0])

    # slope, intercept = fm_data.RescaleSlope, fm_data.RescaleIntercept
    # fm_range = 2 ** fm_data.BitsStored

    # Get the echo times (TE)
    TE = []
    with open(fm_dcms[0], "rb") as f:
        lines = f.readlines()
        for line in lines:
            if line[:7] == b"alTE[0]":
                TE.append(int(line.decode("ascii").split()[-1]))
            elif line[:7] == b"alTE[1]":
                TE.append(int(line.decode("ascii").split()[-1]))
            elif line[:7] == b"alTE[2]":
                TE.append(int(line.decode("ascii").split()[-1]))

    if len(TE) != 3:
        raise ValueError(
            "Could not find all three echo times (TE0, TE1, TE2) in the DICOM header.")

    TE0, TE1, TE2 = TE[0], TE[1], TE[2]  # Convert to seconds
    dTE1 = (TE1 - TE0) * 1e-6  # Delta TE1 in seconds
    dTE2 = (TE2 - TE1) * 1e-6  # Delta TE2 in seconds

    print(f"TE0 = {TE0}, TE1 = {TE1}, TE2 = {TE2}")
    print(f"Delta TE1 = {dTE1} s, Delta TE2 = {dTE2} s")

    DimR, DimP, DimS, nechos = field_map.shape  # Rows columns slices echoes

    # Visualize mid-slices of phase data for each echo phase : 0 - 4096
    mid_slice = DimS // 2  # Mid-slice index

    # Convert field map to phase data
    # field_map = (slope * field_map + intercept) / slope / fm_range

    # Convert the phase values from 0-4096 to -pi to pi
    field_map = (field_map / 4096) * (2 * np.pi) - np.pi

    # Visualize mid-slices of phase data for each echo phase : -pi - pi

    # Initialize phase calculation arrays
    DimR, DimP, DimS, _ = field_map.shape
    mask = []
    B0map = np.zeros((DimR, DimP, DimS))
    if Mask == "Matlab":
        mat_data = scipy.io.loadmat(
            '/volatile/home/st281428/field_map/B0/_1/Mask.mat')
        mask = mat_data['Mask']
    elif Mask == "Python":
        # Compute mean amplitude image and mask
        magnitude = load_magnitude_from_dicom(
            "/volatile/home/st281428/field_map/B0/_1/A")
        mean_amplitude = np.mean(magnitude, axis=-1)

        # Apply Gaussian smoothing to remove small noise
        smoothed_amplitude = gaussian_filter(mean_amplitude, sigma=1)

        # Adaptive thresholding
        threshold = 0.5 * np.mean(smoothed_amplitude)
        mask = smoothed_amplitude > threshold
        # mask = mean_amplitude > (0.5 * np.mean(mean_amplitude))

        # Adaptive threshold based on percentile
        # Adjust this percentile if necessary
        threshold = np.percentile(mean_amplitude, 62)
        mask = mean_amplitude > threshold
    mask0 = mask  # Will be used in outlier detection
    mask = np.expand_dims(mask, axis=-1)  # Ensure mask matches field_map
    mask_expanded = np.repeat(mask, nechos, axis=-1)
    field_map = field_map * mask_expanded
    # Calculate phase mapping using MATLAB-style approach
    Diff2 = sr.unwrap_phase(field_map[..., 1]) - \
        sr.unwrap_phase(field_map[..., 0])
    Diff = (field_map[:, :, :, 1] - field_map[:, :, :, 0])
    Dunwrapped = sr.unwrap_phase(Diff)
    Dunwrappeddd2 = Diff2

    Dunwrapped = sr.unwrap_phase(Diff)

    if unwrap == "2D":
        # Phase difference in radians
        D = (field_map[:, :, :, 1] - field_map[:, :, :, 0])
        # In matlab here : Unwrapped = SEGUE(Inputs);
        D_turns = D / (2 * np.pi)
        Dneg = D_turns < -0.5
        Dpos = D_turns >= 0.5
        unwrapped = D_turns + Dneg - Dpos
        B0map = unwrapped / dTE1

    elif unwrap == "3D":
        tau = np.array([0.0, dTE1, dTE1 + dTE2])
        sumtau = np.sum(tau)
        sumtau2 = np.sum(tau * tau)

        # Initialize D as a list of phase arrays
        D = [None] * nechos
        # Phase difference calculations
        D1 = Dunwrapped/2/np.pi
        D[0] = field_map[:, :, :, 0]/2/np.pi
        D[1] = D[0] + D1
        D[2] = field_map[:, :, :, 2]/2/np.pi

        D2th = D1 * (dTE1 + dTE2) / dTE1
        D[2] = D[2] + np.round(D2th)
        D2 = D[2] - D[0]
        eps = D2 - D2th
        D2neg = (eps < -0.5)
        D2pos = (eps >= 0.5)
        D2 = D2 + D2neg - D2pos
        D[2] = D[2] + D2neg - D2pos
        Sxy = tau[1] * D[1] + tau[2] * D[2]
        Sy = D[0] + D[1] + D[2]
        B0map = np.round((3 * Sxy - sumtau * Sy) / (3 * sumtau2 - sumtau ** 2))

    else:
        raise ValueError(f"Unexpected unwrap method: {unwrap}")

    B0_vect = B0map[mask0.astype(bool)].ravel()
    neighbourhood = [3, 3, 3]
    threshold = 8
    mask_clean = mask0.ravel()
    # outliers, B0_clean_vect = outlier_detection_fieldmap( \
    #    mask0, B0_vect, neighbourhood, threshold, mask_clean)
    outliers, B0_clean = outlier_detection_fieldmap(
        B0map, neighbourhood, threshold)
    view_B0map(B0map, slice_index=22)
    view_B0map(B0_clean, slice_index=22)
    return B0map


# Parameters
Nx = 256
Ny = 256
Nz = 64


# Load and process the field map
B0map = load_field_map_from_dicom(
    '/volatile/home/st281428/field_map/B0/_1/P', Nx, Ny, Nz, unwrap="3D", Mask="Matlab", reference=None)

view_B0map(B0map, slice_index=22)
