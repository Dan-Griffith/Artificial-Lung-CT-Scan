import numpy as np
import matplotlib.pyplot as plt

def plot_3d_npy(file_path, slice_axis=2, slice_index=None, cmap='gray'):
    """
    Plot slices of a 3D array stored in a .npy file.

    Parameters:
    - file_path (str): Path to the .npy file.
    - slice_axis (int): Axis along which to take slices (0, 1, or 2).
    - slice_index (int, optional): Index of the slice to plot. If None, the middle slice is chosen.
    - cmap (str): Colormap to use for plotting.
    """
    # Load the 3D array from the .npy file
    volume = np.load(file_path)
    print(volume.shape)
    
    # Determine the slice index if not provided
    if slice_index is None:
        slice_index = volume.shape[slice_axis] // 2
    
    # Select the slice along the specified axis
    if slice_axis == 0:
        slice_2d = volume[slice_index, :, :]
    elif slice_axis == 1:
        slice_2d = volume[:, slice_index, :]
    elif slice_axis == 2:
        slice_2d = volume[:, :, slice_index]
    else:
        raise ValueError("slice_axis must be 0, 1, or 2")
    
    # Plot the selected slice
    plt.imshow(slice_2d, cmap=cmap)
    plt.title(f'Slice {slice_index} along axis {slice_axis}')
    plt.axis('off')
    plt.show()

# Example usage
file_path = '/Users/dangriffith/Personal Projects/Igneium_3D_Reconstruction/segmented_lungs.npy'
plot_3d_npy(file_path, slice_axis=2)
