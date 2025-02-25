# run_b0map.py
from b0map import load_field_map_from_dicom
from visualize_b0map import visualize_map
import nibabel as nib
import numpy as np
if __name__ == "__main__":
    # Parameters
    NX, NY, NZ = 256, 256, 64
    FIELD_MAP_PATH = "/volatile/home/st281428/Downloads/B0Map/_1/P"
    MAGNITUDE_PATH = "/volatile/home/st281428/Downloads/B0Map/_1/A"

    # Compute B0 map
    b0_map = load_field_map_from_dicom(
        FIELD_MAP_PATH, MAGNITUDE_PATH, NX, NY, NZ,
        num_echoes=3, mask_type="Phantom", unwrap="3D"
    )

    # Save as NIfTI
    nifti_img = nib.Nifti1Image(b0_map, affine=np.eye(4))
    nib.save(nifti_img, "b0_map.nii")

    # Visualize for debugging
    visualize_map(b0_map, slice_index=22, vmin=-600,
                  vmax=600, subtitle="Cleaned B0map")
