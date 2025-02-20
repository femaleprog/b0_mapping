import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def view_B0map(B0map, slice_index=None):
    """
    Function to visualize B0map in three views: Axial, Sagittal, and Coronal.

    Parameters:
        B0map (numpy.ndarray): 3D array representing the B0 field map.
        slice_index (int, optional): Index of the slice to display. Defaults to the middle slice.
    """
    if slice_index is None:
        slice_index = B0map.shape[2] // 2  # Default to middle slice

    # Compute min and max values for consistent colormap scaling
    vmin = np.min(B0map)
    vmax = np.max(B0map)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw={
                             'width_ratios': [1, 1, 1]})

    # Define colormap and normalization for the colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('viridis')

    # **Axial View**
    ax = axes[0]
    axial = B0map[:, :, slice_index]
    rot_axial = np.rot90(axial, 2, (1, 0))
    ax.imshow(rot_axial, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title('Axial')

    # **Sagittal View**
    ax = axes[1]
    sagittal = B0map[:, slice_index, :]
    rot_sagittal = np.rot90(sagittal, 1, (1, 0))
    rot_sagittal = np.fliplr(rot_sagittal)
    ax.imshow(rot_sagittal, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title('Sagittal')

    # **Coronal View**
    ax = axes[2]
    coronal = B0map[slice_index, :, :]
    rot_coronal = np.rot90(coronal, 1, (1, 0))
    rot_coronal = np.fliplr(rot_coronal)
    ax.imshow(rot_coronal, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title('Coronal')

    # Adjust layout to avoid overlap
    plt.subplots_adjust(right=0.85)

    # **Colorbar**
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cbar_ax).set_label("B0 Field (Hz)")

    plt.suptitle('Delta B0map in Hz')  # Title for the entire figure
    plt.show()


def visualize_map(B0map, slice_index=None, vmin=-100, vmax=100):
    """
    Function to visualize B0map in three views: Axial, Sagittal, and Coronal, with fixed scaling.

    Parameters:
        B0map (numpy.ndarray): 3D array representing the B0 field map.
        slice_index (int, optional): Index of the slice to display. Defaults to the middle slice.
        vmin (float): Minimum value for colormap normalization (fixed scale for comparison).
        vmax (float): Maximum value for colormap normalization (fixed scale for comparison).
    """
    if slice_index is None:
        slice_index = B0map.shape[2] // 2  # Default to middle slice

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw={
                             'width_ratios': [1, 1, 1]})

    # Define colormap and normalization for the colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('viridis')

    # **Axial View**
    ax = axes[0]
    axial = B0map[:, :, slice_index]
    ax.imshow(np.rot90(axial, 2, (1, 0)), origin='lower',
              vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title('Axial')

    # **Sagittal View**
    ax = axes[1]
    sagittal = B0map[:, slice_index, :]
    rotated_sagittal = np.fliplr(np.rot90(sagittal, 1, (1, 0)))
    ax.imshow(rotated_sagittal, origin='lower',
              vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title('Sagittal')

    # **Coronal View**
    ax = axes[2]
    coronal = B0map[slice_index, :, :]
    rotated_coronal = np.fliplr(np.rot90(coronal, 1, (1, 0)))
    ax.imshow(rotated_coronal, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title('Coronal')

    # Adjust layout to avoid overlap
    plt.subplots_adjust(right=0.85)

    # **Colorbar**
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cbar_ax).set_label("B0 Field (Hz)")

    # Title for the entire figure
    plt.suptitle('Delta B0map in Hz (Fixed Scale)')
    plt.show()
