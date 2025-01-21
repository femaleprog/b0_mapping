import os
import numpy as np
import pydicom as pdcm
from skimage import transform as st
from skimage import restoration as sr
import poisson as ps
import matplotlib.pyplot as plt

def load_field_map_from_dicom(folder_path, Nx, Ny, Nz, unwrap=False, reference=None):

    
    # Read and sort DICOM files in the P folder
    fm_dcms = [os.path.join(folder_path, filename)
               for filename in sorted(os.listdir(folder_path))
               if (filename[-4:].lower() in [".ima", ".dcm"])]
    sort = np.argsort([float(pdcm.dcmread(os.path.join(folder_path, dcm)).InstanceNumber) for dcm in fm_dcms])
    fm_dcms = [fm_dcms[i] for i in sort]
    
    # Check if there are enough files for 3 echoes
    if len(fm_dcms) < 3:
        raise ValueError("At least 3 DICOM files are required for 3-echo processing.")
    
    # Get metadata from the first DICOM file
    fm_data = pdcm.dcmread(fm_dcms[0])
    slope, intercept = fm_data.RescaleSlope, fm_data.RescaleIntercept
    fm_range = 2 ** fm_data.BitsStored
    
    # Get the three TEs manually
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
    
    # Ensure we have 3 echo times
    if len(TE) != 3:
        raise ValueError("Could not find all three echo times (TE0, TE1, TE2) in the DICOM header.")
    
    TE0, TE1, TE2 = TE[0], TE[1], TE[2]
    dTE1 = (TE1 - TE0) * 1e-6  # Delta TE1 in seconds
    dTE2 = (TE2 - TE1) * 1e-6  # Delta TE2 in seconds
    
    print(f"TE0 = {TE0}, TE1 = {TE1}, TE2 = {TE2}")
    print(f"Delta TE1 = {dTE1} s, Delta TE2 = {dTE2} s")
    
    # Get the possible range
    range_w = np.array([(intercept) / slope / fm_range / dTE1,
                        (slope * fm_range + intercept) / slope / fm_range / dTE1]).astype(int)
    
    # Read phase data from the P folder
    phase_data = np.array([pdcm.dcmread(fm_dcm).pixel_array for fm_dcm in fm_dcms])
    phase_data = [np.rot90(f.T, k=-1) for f in phase_data][::-1]
    phase_data = np.stack(phase_data, axis=-1)
    
    # Convert phase data to radians
    phase_data = 2 * np.pi * (phase_data - 2047.) / 4095.  # Assuming phase is stored as 12-bit integers
    
    # Compute field map using phase differences for 3 echoes
    phase_diff1 = phase_data[:, :, 1] - phase_data[:, :, 0]  # Phase difference between Echo 1 and Echo 0
    phase_diff2 = phase_data[:, :, 2] - phase_data[:, :, 1]  # Phase difference between Echo 2 and Echo 1
    
    # Combine phase differences to compute the field map
    field_map = (phase_diff1 / (2 * np.pi * dTE1) + phase_diff2 / (2 * np.pi * dTE2)) / 2
    
    # Phase unwrapping (optional)
    convert = (dTE1 * 2 * np.pi)
    if unwrap:
        print(field_map)
        reference = st.resize(reference, field_map.shape)
        field_map = ps.poisson_unwrap_gpu(field_map * convert, reference, kmax=10) / convert
    elif unwrap == "old":
        field_map = sr.unwrap_phase(field_map * convert) / convert
    
    # Resize field map
    field_map = st.resize(field_map.astype(np.float32), (Nx, Ny, Nz))
    
    return field_map, range_w

Nx = 256
Ny = 256
Nz = 64

field_map, range_w = load_field_map_from_dicom('/volatile/home/st281428/field_map/B0/_1/P', Nx, Ny, Nz, unwrap="old", reference=None)
print(field_map.shape)

# visualizing a 2D slice 
# Select a slice (e.g., the middle slice along the z-axis)
slice_index = field_map.shape[2] // 2
field_map_slice = field_map[:, :, slice_index]

# Plot the slice
plt.imshow(field_map_slice, cmap='viridis')  # You can change the colormap (e.g., 'gray', 'plasma')
plt.colorbar(label='Field Map Value')
plt.title(f"Field Map Slice (z = {slice_index})")
plt.show()