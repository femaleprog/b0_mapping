import os
import numpy as np
import pydicom
from scipy.ndimage import zoom

def read_triple_echo_gre(folder_path):
    # Define paths to A and P folders
    folder_A = os.path.join(folder_path, 'A')
    folder_P = os.path.join(folder_path, 'P')
    
    # Check if A and P folders exist
    if not os.path.exists(folder_A) or not os.path.exists(folder_P):
        raise ValueError("The folder must contain 'A' and 'P' subfolders.")
    
    # Read amplitude (A) and phase (P) DICOM files
    def read_dicom_files(subfolder):
        files = [os.path.join(subfolder, filename)
                 for filename in sorted(os.listdir(subfolder))
                 if (filename[-4:].lower() in [".ima", ".dcm"])]
        if not files:
            raise ValueError(f"No DICOM files found in {subfolder}.")
        return files
    
    amp_files = read_dicom_files(folder_A)
    phase_files = read_dicom_files(folder_P)
    
    # Sort files by instance number
    def sort_files(files):
        return sorted(files, key=lambda x: float(pydicom.dcmread(x).InstanceNumber))
    
    amp_files = sort_files(amp_files)
    phase_files = sort_files(phase_files)
    
    # Read amplitude and phase data
    amp_data = np.array([pydicom.dcmread(f).pixel_array for f in amp_files])
    phase_data = np.array([pydicom.dcmread(f).pixel_array for f in phase_files])
    
    # Get metadata from the first DICOM file
    header = pydicom.dcmread(amp_files[0])
    resolution = header.PixelSpacing  # in mm
    nechos = int(header[0x2001, 0x1018].value)  # Number of echoes
    print(f"Number of Contrasts = {nechos}")
    
    # Extract echo times (TE)
    TE = []
    for i in range(nechos):
        te_tag = f"alTE[{i}]"
        te_value = float(header[0x2001, 0x1018 + i].value) * 1e-6  # Convert to seconds
        TE.append(te_value)
        print(f"TE[{i + 1}] = {te_value} s")
    
    # Compute delta TE
    DTE1 = TE[1] - TE[0]
    print(f"Delta TE1 = {DTE1} s")
    if nechos == 3:
        DTE2 = TE[2] - TE[1]
        print(f"Delta TE2 = {DTE2} s")
    else:
        DTE2 = None
    
    # Reshape amplitude and phase data
    DimR, DimP = amp_data.shape[1], amp_data.shape[2]
    DimS = amp_data.shape[0] // nechos
    amp_data = amp_data.reshape(DimR, DimP, DimS, nechos)
    phase_data = phase_data.reshape(DimR, DimP, DimS, nechos)
    
    # Combine amplitude and phase to get complex data
    cechos = amp_data * np.exp(1j * (2 * np.pi * (phase_data - 2047.) / 4095.))
    
    # Compute FOV
    FOVf = DimR * resolution[0] / 1000  # in meters
    FOVp = DimP * resolution[1] / 1000  # in meters
    FOVs = DimS * header.SliceThickness / 1000  # in meters
    
    return cechos, nechos, DTE1, DTE2, phase_data, FOVs, FOVp, FOVf, resolution, header

# Test the function
try:
    folder_path = './B0/_1'
    cechos, nechos, DTE1, DTE2, Phase, FOVs, FOVp, FOVf, resolution, header = read_triple_echo_gre(folder_path)
    print("Complex Echo Data Shape:", cechos.shape)
    print("Resolution (mm):", resolution)
    print("FOV (m):", FOVs, FOVp, FOVf)
except ValueError as e:
    print(e)

cechos, nechos, DTE1, DTE2, Phase, FOVs, FOVp, FOVf, resolution, header = read_triple_echo_gre('./B0/_1')
print("Complex Echo Data Shape:", cechos.shape)
print("Resolution (mm):", resolution)
print("FOV (m):", FOVs, FOVp, FOVf)
