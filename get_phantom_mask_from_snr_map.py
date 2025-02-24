import numpy as np
import scipy.ndimage as ndi


def plot_3d(image):
    # Placeholder function for 3D plotting
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.where(image > 0)
    ax.scatter(x, y, z, c='red', marker='o', alpha=0.5)
    plt.show()


def get_phantom_mask_from_snr(SNR, thres=None):
    if thres is None:
        option = int(input(
            "Determine threshold (option 1) or number of voxels in Mask (option 2) ?\n"))
        if option == 1:
            # plot_3d(SNR)
            thres = float(
                input("Enter SNR threshold to determine mask (above 1):\n"))
            print("WARNING: generating mask with the specified SNR-threshold")
            while thres != 1:
                Mask = np.zeros_like(SNR, dtype=bool)
                Mask[SNR > thres] = True
                # plot_3d(Mask)
                thres = float(
                    input("Is the mask OK ? \n1 = yes. \nEnter new threshold otherwise: \n"))
        else:  # Number of voxels in Mask
            nvox = int(input("Enter desired number of voxels in mask:\n"))
            print("WARNING: generating mask with the specified number of voxels")
            thres = 10
            Mask = np.ones_like(SNR, dtype=bool)
            while np.count_nonzero(Mask) > nvox and thres < 100000:
                thres += 1
                Mask = SNR > thres
            # plot_3d(Mask)
    else:
        Mask = np.zeros_like(SNR, dtype=bool)
        Mask[SNR > thres] = True
        # plot_3d(Mask)
    '''
    # Extract connected components
    labeled_array, num_features = ndi.label(Mask, structure=np.ones((3, 3, 3)))

    if num_features > 1:
        sizes = np.bincount(labeled_array.ravel())[1:]  # Ignore background
        max_label = np.argmax(sizes) + 1  # Find the largest component
        Mask = labeled_array == max_label  # Keep only the largest connected component

    print(
        f'SNR threshold = {thres}, nb of voxels in Mask = {np.count_nonzero(Mask)}')
    '''
    return Mask
