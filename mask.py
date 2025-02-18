import os
import numpy as np
import nibabel as nib
import subprocess
import scipy.ndimage as ndimage
from skimage.morphology import ball


def get_brain(AMP, FOVx, FOVy, FOVz, BETparam=0.5, BETdir="C:/Users/Public/BET"):
    """
    Extracts the brain mask from multi-echo MRI data using FSL's BET tool.

    Parameters:
    - AMP: 4D numpy array representing amplitude images (assumes at least 2 echos)
    - FOVx, FOVy, FOVz: Field-of-view in x, y, z directions in mm
    - BETparam: BET parameter (default = 0.5), should be >0 and <1
    - BETdir: Path to FSL's BET tool

    Returns:
    - Mask: Final brain mask as a 3D numpy array
    """

    if not os.path.isdir(BETdir):
        raise FileNotFoundError(f"BET directory not found: {BETdir}")

    wd = os.getcwd()  # Store current working directory
    Mask = None

    def save_nifti(image_data, filename, voxel_size):
        """Helper function to save NIfTI images"""
        affine = np.diag([voxel_size[0], voxel_size[1],
                         voxel_size[2], 1])  # Voxel size in mm
        nii = nib.Nifti1Image(image_data.astype(np.float32), affine)
        nib.save(nii, filename)

    def run_bet(input_nifti, output_prefix, bet_param):
        """Runs BET brain extraction"""
        bet_command = f'bet "{input_nifti}" "{output_prefix}" -m -n -f {bet_param}'
        subprocess.run(bet_command, shell=True, check=True)

    def load_nifti_mask(filename):
        """Loads a binary mask from a NIfTI file"""
        nii = nib.load(filename)
        return nii.get_fdata() > 0  # Convert to boolean mask

    # ** Step 1: Process First Echo **
    echo0_filename = os.path.join(wd, "Echo0_A.nii")
    voxel_size = [FOVx / AMP.shape[0], FOVy /
                  AMP.shape[1], FOVz / AMP.shape[2]]
    save_nifti(AMP[:, :, :, 0], echo0_filename, voxel_size)

    # ** Run BET on first echo **
    run_bet(echo0_filename, os.path.join(wd, "brain0"), BETparam)
    Mask0 = load_nifti_mask(os.path.join(wd, "brain0_mask.nii"))

    # ** User Interaction for BETparam Adjustment **
    while True:
        print("Displaying Mask0...")  # Add visualization if needed
        user_input = input(
            "Enter 1 if satisfied with the Brain Mask. Otherwise, enter a new BET parameter (>0 and <1): ")
        try:
            user_input = float(user_input)
            if user_input == 1:
                break  # Accept the mask
            elif 0 < user_input < 1:
                BETparam = user_input
                run_bet(echo0_filename, os.path.join(wd, "brain0"), BETparam)
                Mask0 = load_nifti_mask(os.path.join(wd, "brain0_mask.nii"))
            else:
                print(
                    "Invalid input. Please enter a value between 0 and 1, or 1 to accept the mask.")
        except ValueError:
            print("Invalid input. Please enter a numerical value.")

    # ** Step 2: Process Second Echo **
    echo1_filename = os.path.join(wd, "Echo1_A.nii")
    save_nifti(AMP[:, :, :, 1], echo1_filename, voxel_size)

    run_bet(echo1_filename, os.path.join(wd, "brain1"), BETparam)
    Mask1 = load_nifti_mask(os.path.join(wd, "brain1_mask.nii"))

    # ** Combine Masks for Multiple Echos **
    Mask = np.logical_and(Mask0, Mask1)

    # ** Morphological Erosion to Refine Mask **
    structuring_element = np.array([
        [[0, 1, 0], [1, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 0, 0]]
    ])  # Asymmetric erosion
    Mask = ndimage.binary_erosion(
        Mask, structure=structuring_element).astype(np.uint8)

    # ** Keep Only the Largest Connected Component **
    labeled_mask, num_features = ndimage.label(Mask)
    if num_features > 1:
        sizes = ndimage.sum(Mask, labeled_mask, range(num_features + 1))
        largest_region = (labeled_mask == np.argmax(
            sizes[1:]) + 1)  # Keep only largest
        Mask = largest_region.astype(np.uint8)

    print(f"Number of voxels in final Mask: {np.count_nonzero(Mask)}")

    # ** Save Final Mask **
    mask_filename = os.path.join(wd, "Mask.nii")
    save_nifti(Mask, mask_filename, voxel_size)

    return Mask
