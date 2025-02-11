import os
import numpy as np
import pydicom as pdcm
from skimage import transform as st
from skimage import restoration as sr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_opening 
import scipy
import cv2




def load_magnitude_from_dicom(folder_path):
    # Read and sort DICOM files 
    fm_dcms = [os.path.join(folder_path, filename)
               for filename in sorted(os.listdir(folder_path))
               if (filename[-4:].lower() in [".ima", ".dcm"])]
    # sort files by instance number
    sort = np.argsort([float(pdcm.dcmread(dcm).InstanceNumber) for dcm in fm_dcms])
    fm_dcms = [fm_dcms[i] for i in sort]
    
    # group files by echo 
    nechos = 3
    echo_data = np.array([pdcm.dcmread(dcm).pixel_array for dcm in fm_dcms]) 
    magnitude = echo_data.reshape(nechos,len(echo_data)//nechos,*echo_data[0].shape)
    magnitude = np.moveaxis(magnitude, 0, -1)
    magnitude = np.moveaxis(magnitude, 0, -2)
    return magnitude
    
def load_field_map_from_dicom(folder_path, Nx, Ny, Nz, unwrap=False, reference=None):
    # Read and sort DICOM files
    fm_dcms = [os.path.join(folder_path, filename)
               for filename in sorted(os.listdir(folder_path))
               if (filename[-4:].lower() in [".ima", ".dcm"])]
    
    # Sort files by instance number
    sort = np.argsort([float(pdcm.dcmread(dcm).InstanceNumber) for dcm in fm_dcms])
    fm_dcms = [fm_dcms[i] for i in sort]
    
    # Group files by echo
    nechos = 3  # Assuming 3 echoes 
    # Load and reshape the data
    echo_data = np.array([pdcm.dcmread(dcm).pixel_array for dcm in fm_dcms]) 
    field_map = echo_data.reshape(nechos,len(echo_data)//nechos,*echo_data[0].shape)
    field_map = np.moveaxis(field_map, 0, -1)
    field_map = np.moveaxis(field_map, 0, -2)

    # Get metadata from the first DICOM file
    fm_data = pdcm.dcmread(fm_dcms[0])

    #slope, intercept = fm_data.RescaleSlope, fm_data.RescaleIntercept
    #fm_range = 2 ** fm_data.BitsStored
 
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
        raise ValueError("Could not find all three echo times (TE0, TE1, TE2) in the DICOM header.")
    
    TE0, TE1, TE2 = TE[0], TE[1], TE[2]  # Convert to seconds
    dTE1 = (TE1 - TE0) * 1e-6  # Delta TE1 in seconds
    dTE2 = (TE2 - TE1) * 1e-6 # Delta TE2 in seconds
    
    print(f"TE0 = {TE0}, TE1 = {TE1}, TE2 = {TE2}")
    print(f"Delta TE1 = {dTE1} s, Delta TE2 = {dTE2} s")

    DimR, DimP, DimS, nechos = field_map.shape # Rows columns slices echoes

    # Visualize mid-slices of phase data for each echo phase : 0 - 4096
    mid_slice = DimS // 2  # Mid-slice index

    '''
    for iecho in range(nechos):
        plt.figure()
        plt.imshow(field_map[:, :, mid_slice, iecho], cmap='gray')
        plt.colorbar()
        plt.title(f'Echo {iecho + 1} Phase (Mid-Slice)')
        plt.show()
    '''
    # Convert field map to phase data
    #field_map = (slope * field_map + intercept) / slope / fm_range

    
    # Convert the phase values from 0-4096 to -pi to pi
    field_map = (field_map / 4096) * (2 * np.pi) - np.pi
    
    # Visualize mid-slices of phase data for each echo phase : -pi - pi
    '''
    for iecho in range(nechos):
        plt.figure()
        plt.imshow(field_map[:, :, mid_slice, iecho], cmap='gray')
        plt.colorbar()
        plt.title(f'Echo {iecho + 1} Phase (Mid-Slice)')
        plt.show()
    '''
    # Initialize phase calculation arrays
    DimR, DimP, DimS, _ = field_map.shape
    B0map = np.zeros((DimR, DimP, DimS))

    # Compute mean amplitude image and mask
    magnitude = load_magnitude_from_dicom("/volatile/home/st281428/field_map/B0/_1/A")
    mean_amplitude = np.mean(magnitude, axis=-1)
    
    # Apply Gaussian smoothing to remove small noise
    smoothed_amplitude = gaussian_filter(mean_amplitude, sigma=1)
    
    # Adaptive thresholding
    threshold = 0.5 * np.mean(smoothed_amplitude)  
    mask = smoothed_amplitude > threshold
    #mask = mean_amplitude > (0.5 * np.mean(mean_amplitude))
    
    # Adaptive threshold based on percentile
    threshold = np.percentile(mean_amplitude, 62)  # Adjust this percentile if necessary
    mask = mean_amplitude > threshold
    mask = np.expand_dims(mask, axis=-1)  # Ensure mask matches field_map 

    

    # Apply the mask to the field map
    mask_expanded = np.repeat(mask, nechos, axis=-1)
    field_map = field_map * mask_expanded  # in matlab we apply the mask later
    # Calculate phase mapping using MATLAB-style approach
    if unwrap=="2D":
        D = (field_map[:,:,:,1] - field_map[:,:,:,0])  # Phase difference in radians
        # In matlab here : Unwrapped = SEGUE(Inputs);
        D_turns = D / ( 2 * np.pi )
        Dneg = D_turns < -0.5
        Dpos = D_turns >= 0.5
        unwrapped = D_turns + Dneg - Dpos
        B0map = unwrapped / dTE1
    
    elif unwrap=="3D":
        tau = np.array([0.0, dTE1, dTE1 + dTE2])
        sumtau = np.sum(tau)
        sumtau2 = np.sum(tau * tau)
        
        # Initialize D as a list of phase arrays
        D = [field_map[:,:,:,i] / (2 * np.pi) for i in range(nechos)]
        
        # Phase difference calculations
        D1 = D[1] - D[0]
        D1neg = (D1 < -0.5)
        D1pos = (D1 >= 0.5)
        D1 = D1 + D1neg - D1pos
        D[1] = D[1] + D1neg - D1pos
        
        # Theoretical estimation and correction
        D2th = D1 * (dTE1 + dTE2) / dTE1
        D[2] = D[2] + np.round(D2th)
        D2 = D[2] - D[0]
        eps = D2 - D2th
        D2neg = (eps < -0.5)
        D2pos = (eps >= 0.5)
        D2 = D2 + D2neg - D2pos
        D[2] = D[2] + D2neg - D2pos
        
        # Calculate B0 map
        Sxy = tau[1] * D[1] + tau[2] * D[2]
        Sy = D[0] + D[1] + D[2]
        B0map = (3 * Sxy - sumtau * Sy) / (3 * sumtau2 - sumtau ** 2)
    
    else:
        raise ValueError(f"Unexpected unwrap method: {unwrap}")
    
    # Resize B0 map
    #B0map = st.resize(B0map.astype(np.float32), (Nx, Ny, Nz))
    print("B0 Map values:", B0map)

    slice_index = 22
    # Compute difference 

    # Load MATLAB B0map
    matlab_data = scipy.io.loadmat('/volatile/home/st281428/field_map/B0/_1/B0map.mat')  
    B0map_matlab = matlab_data['B0map']  # Extract variable

    # Axial slice comparison
    B0map_matlab_slice = B0map_matlab[:, :, 22]
    B0map_python_slice = B0map[:, :, 22]  # Select same slice
    B0map_diff_axial = B0map_matlab_slice - B0map_python_slice

    # compute the MSE
    mse_axial = np.mean((B0map_matlab_slice - B0map_python_slice) ** 2)
    print("MSE axial:", mse_axial)
    

    # Coronal slice comparison
    B0map_matlab_coronal = B0map_matlab[:, slice_index, :]  # MATLAB Coronal slice
    B0map_python_coronal = B0map[:, slice_index, :]  # Python Coronal slice
    B0map_diff_coronal = B0map_matlab_coronal - B0map_python_coronal
    
    # Compute MSE
    mse_coronal = np.mean(B0map_diff_coronal ** 2)
    print("Coronal MSE:", mse_coronal)
    
    # Sagittal slice comparison
    B0map_matlab_sagittal = B0map_matlab[slice_index, :, :]  # MATLAB Sagittal slice
    B0map_python_sagittal = B0map[slice_index, :, :]  # Python Sagittal slice
    B0map_diff_sagittal = B0map_matlab_sagittal - B0map_python_sagittal

    # Compute MSE
    mse_sagittal = np.mean(B0map_diff_sagittal ** 2)
    print("Sagittal MSE:", mse_sagittal)

    # Visualize the difference : Axial
    plt.figure(figsize=(6, 5))
    plt.imshow(B0map_diff_axial, cmap='bwr', vmin=-50, vmax=50)  # Blue-Red colormap for difference
    plt.colorbar(label='Difference (Hz)')
    plt.title("Difference- Axial: MATLAB vs Python B0map", fontsize=12, fontweight="bold")
    plt.axis('equal')
    plt.axis('tight')
    plt.show()
    
    # Visualize the difference : Coronal
    plt.figure(figsize=(6, 5))
    plt.imshow(B0map_diff_coronal, cmap='bwr', vmin=-50, vmax=50)  # Blue-Red colormap for difference
    plt.colorbar(label='Difference (Hz)')
    plt.title("Difference- Coronal: MATLAB vs Python B0map", fontsize=12, fontweight="bold")
    plt.axis('equal')
    plt.axis('tight')
    plt.show()

    # Visualize the difference : Sagittal
    plt.figure(figsize=(6, 5))
    plt.imshow(B0map_diff_sagittal, cmap='bwr', vmin=-50, vmax=50)  # Blue-Red colormap for difference
    plt.colorbar(label='Difference (Hz)')
    plt.title("Difference- Sagittal: MATLAB vs Python B0map", fontsize=12, fontweight="bold")
    plt.axis('equal')
    plt.axis('tight')
    plt.show()

    return B0map

# Parameters
Nx = 256
Ny = 256
Nz = 64

# Load and process the field map
B0map = load_field_map_from_dicom('/volatile/home/st281428/field_map/B0/_1/P', Nx, Ny, Nz, unwrap="3D", reference=None)
print("B0 Map Shape:", B0map.shape)

# Visualizing a 2D slice of the B0 map
slice_index = B0map.shape[2] // 2
B0map_slice = B0map[:, :, slice_index]

# Compute the min and max values across the entire B0map
vmin = np.min(B0map)
vmax = np.max(B0map)

fig, axes = plt.subplots(1, 3, figsize=(12, 5))

slice_index = B0map.shape[2] // 2  # Middle slice

# Axial View
ax = axes[0]
ax.imshow(B0map[:, :, slice_index], origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
ax.set_title('Axial')

# Coronal View
ax = axes[1]
ax.imshow(B0map[:, slice_index, :], origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
ax.set_title('Coronal')

# Sagittal View
ax = axes[2]
ax.imshow(B0map[slice_index, :, :], origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
ax.set_title('Sagittal')

plt.show()
