import glob
import os
import warnings
import numpy as np
import pydicom as pdcm
from skimage import transform as st
from skimage import restoration as sr
import poisson as ps
import matplotlib.pyplot as plt

# Give field map in hertzs
def load_field_map_from_dicom(folder_path, Nx, Ny, Nz, unwrap=False, reference=None):
    # Read and sort DICOM files
    fm_dcms = [os.path.join(folder_path, filename)
               for filename in sorted(os.listdir(folder_path))
               if (filename[-4:].lower() in [".ima", ".dcm"])]
    sort = np.argsort([float(pdcm.dcmread(os.path.join(folder_path, dcm)).InstanceNumber) for dcm in fm_dcms])
    fm_dcms = [fm_dcms[i] for i in sort]
    
    # Check if there are enough files for multiple echoes
    if len(fm_dcms) < 2:
        raise ValueError("At least 2 DICOM files are required for multi-echo processing.")
    
    # Get the scale
    fm_data = pdcm.dcmread(fm_dcms[0])
    slope, intercept = fm_data.RescaleSlope, fm_data.RescaleIntercept
    fm_range = 2 ** fm_data.BitsStored
    
    # Get the echo times (TE) manually
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
    
    # Ensure we have at least 2 echo times
    if len(TE) < 2:
        raise ValueError("Could not find echo times (TE0, TE1) in the DICOM header.")
    
    TE0, TE1 = TE[0], TE[1]
    dTE = (TE1 - TE0) * 1e-6  # Delta TE in seconds
    
    print(f"TE0 = {TE0}, TE1 = {TE1}")
    print(f"Delta TE = {dTE} s")
    
    # Load and reshape the data
    field_map = np.array([pdcm.dcmread(fm_dcm).pixel_array for fm_dcm in fm_dcms])
    field_map = [np.rot90(f.T, k=-1) for f in field_map][::-1]
    field_map = np.stack(field_map, axis=-1)
    
    # Debug: Print the shape of the loaded data
    print("Loaded Data Shape:", field_map.shape)
    
    # Reshape the data to [DimR, DimP, DimS, nechos]
    DimR, DimP, DimS = field_map.shape
    nechos = 1  # Default to 1 echo if not explicitly specified
    
    # If there are multiple echoes, reshape accordingly
    if len(fm_dcms) > 1:
        nechos = len(fm_dcms)
        field_map = field_map.reshape(DimR, DimP, DimS // nechos, nechos)
        DimS = DimS // nechos  # Update the number of slices
    
    # Debug: Print reshaped shapes
    print("Reshaped Data Shape:", field_map.shape)
    
    # Debug: Visualize mid-slices of phase data for each echo
    mid_slice = DimS // 2  # Mid-slice index
    for iecho in range(nechos):
        plt.figure()
        plt.imshow(field_map[:, :, mid_slice, iecho], cmap='gray')
        plt.colorbar()
        plt.title(f'Echo {iecho + 1} Phase (Mid-Slice)')
        plt.show()
    
    # Compute the field map
    field_map = (slope * field_map + intercept) / slope / fm_range / dTE
    
    # Phase unwrapping (optional)
    convert = (dTE * 2 * np.pi)
    if unwrap:
        print(field_map)
        reference = st.resize(reference, field_map.shape)
        field_map = ps.poisson_unwrap_gpu(field_map * convert, reference, kmax=10) / convert
    elif unwrap == "old":
        field_map = sr.unwrap_phase(field_map * convert) / convert
    
    # Resize field map
    field_map = st.resize(field_map.astype(np.float32), (Nx, Ny, Nz))
    
    return field_map, range_w

# Parameters
Nx = 256
Ny = 256
Nz = 64

# Load and process the field map
field_map, range_w = load_field_map_from_dicom('/volatile/home/st281428/field_map/B0/_1/P', Nx, Ny, Nz, unwrap="old", reference=None)
print("Field Map Shape:", field_map.shape)

# Visualizing a 2D slice of the field map
slice_index = field_map.shape[2] // 2
field_map_slice = field_map[:, :, slice_index]

# Plot the slice
plt.imshow(field_map_slice, cmap='viridis')  # You can change the colormap (e.g., 'gray', 'plasma')
plt.colorbar(label='Field Map Value')
plt.title(f"Field Map Slice (z = {slice_index})")
plt.show()