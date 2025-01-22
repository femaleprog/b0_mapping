import os
import numpy as np
import pydicom as pdcm
from skimage import transform as st
from skimage import restoration as sr
import matplotlib.pyplot as plt

def load_field_map_from_dicom(folder_path, Nx, Ny, Nz, unwrap=False, reference=None):
    # Read and sort DICOM files
    fm_dcms = [os.path.join(folder_path, filename)
               for filename in sorted(os.listdir(folder_path))
               if (filename[-4:].lower() in [".ima", ".dcm"])]
    
    # Sort files by instance number
    sort = np.argsort([float(pdcm.dcmread(dcm).InstanceNumber) for dcm in fm_dcms])
    fm_dcms = [fm_dcms[i] for i in sort]
    
    # Group files by echo
    nechos = 3  # Assuming 3 echoes (adjust if different)
    echo_groups = [[] for _ in range(nechos)]
    for i, dcm in enumerate(fm_dcms):
        echo_groups[i % nechos].append(dcm)
    
    # Load and reshape the data
    field_map = []
    for iecho in range(nechos):
        echo_data = np.array([pdcm.dcmread(dcm).pixel_array for dcm in echo_groups[iecho]])
        echo_data = [np.rot90(f.T, k=-1) for f in echo_data][::-1]
        field_map.append(np.stack(echo_data, axis=-1))
    
    # Stack echoes along the last dimension
    field_map = np.stack(field_map, axis=-1)
    
    # Debug: Print the shape of the loaded data
    print("Loaded Data Shape:", field_map.shape)
    
    # Get metadata from the first DICOM file
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
    
    # Ensure we have 3 echo times
    if len(TE) != 3:
        raise ValueError("Could not find all three echo times (TE0, TE1, TE2) in the DICOM header.")
    
    TE0, TE1, TE2 = TE[0], TE[1], TE[2]
    dTE1 = (TE1 - TE0) * 1e-6  # Delta TE1 in seconds
    dTE2 = (TE2 - TE1) * 1e-6  # Delta TE2 in seconds
    
    print(f"TE0 = {TE0}, TE1 = {TE1}, TE2 = {TE2}")
    print(f"Delta TE1 = {dTE1} s, Delta TE2 = {dTE2} s")
    
    # Reshape the data to [DimR, DimP, DimS, nechos]
    DimR, DimP, DimS, nechos = field_map.shape
    
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
    
    # Compute the field map using 3 echoes
    # Phase differences between echoes
    phase_diff1 = np.angle(np.exp(1j * (field_map[:, :, :, 1] - field_map[:, :, :, 0])))
    phase_diff2 = np.angle(np.exp(1j * (field_map[:, :, :, 2] - field_map[:, :, :, 1])))
    
    # Field map calculation (weighted average of phase differences)
    field_map = (phase_diff1 / dTE1 + phase_diff2 / dTE2) / (1 / dTE1 + 1 / dTE2)
    
    # Phase unwrapping (optional)
    convert = (dTE1 * 2 * np.pi)
    if unwrap == "old":
        print(f"The shape of the field map is {field_map.shape}")
        
        # Extract the mid-slice along the third dimension (slices)
        mid_slice_index = field_map.shape[2] // 2  # Mid-slice index
        field_map_mid_slice = field_map[:, :, mid_slice_index]  # Shape: (64, 64)
        
        # Apply phase unwrapping to the mid-slice
        field_map_mid_slice = sr.unwrap_phase(field_map_mid_slice * convert) / convert
        
        # Resize the mid-slice to the final output shape (Nx, Ny, Nz)
        field_map_mid_slice = st.resize(field_map_mid_slice.astype(np.float32), (Nx, Ny, Nz))
        
        # Return the unwrapped mid-slice
        return field_map_mid_slice
    elif unwrap:
        print(field_map)
        reference = field_map[:, :, 0]
        reference = st.resize(reference, field_map.shape)
        field_map = ps.poisson_unwrap_gpu(field_map * convert, reference, kmax=10) / convert
    
    # Resize field map
    field_map_mid_slice = st.resize(field_map_mid_slice.astype(np.float32), (Nx, Ny, Nz))
    
    return field_map

# Parameters
Nx = 256
Ny = 256
Nz = 64

# Load and process the field map
field_map = load_field_map_from_dicom('/volatile/home/st281428/field_map/B0/_1/P', Nx, Ny, Nz, unwrap="old", reference=None)
print("Field Map Shape:", field_map.shape)

# Visualizing a 2D slice of the field map
slice_index = field_map.shape[2] // 2
field_map_slice = field_map[:, :, slice_index]

# Plot the slice
plt.imshow(field_map_slice, cmap='viridis')  # You can change the colormap (e.g., 'gray', 'plasma')
plt.colorbar(label='Field Map Value (Hz)')
plt.title(f"Field Map (Mid-Slice, z = {slice_index})")
plt.show()