import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import ndimage

def voxels_inside_cube(point1, point2):
    # Determine the minimum and maximum coordinates along each axis
    min_x = min(point1[0], point2[0])
    max_x = max(point1[0], point2[0])
    min_y = min(point1[1], point2[1])
    max_y = max(point1[1], point2[1])
    min_z = min(point1[2], point2[2])
    max_z = max(point1[2], point2[2])

    # Iterate through each voxel within the cube
    voxels = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            for z in range(min_z, max_z + 1):
                voxels.append((x, y, z))

    return voxels

def set_voxel_to_zero(voxels):
    new_voxels = voxels.copy()
    one_points = np.argwhere(voxels == 1)
    shape = voxels.shape

    shape = voxels.shape
    positions_to_set_zero = []
    remove_points = []
    
    for point in one_points:
        i, j, k = point[0], point[1], point[2]
        # Check diagonal neighbors
        diagonal_neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if (di, dj, dk) != (0, 0, 0):
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2] and voxels[ni, nj, nk] and abs(i) + abs(j) + abs(k) >= 2:
                            diagonal_neighbors.append([ni, nj, nk])
                            
        for neighbor in diagonal_neighbors:
            inside_voxels = voxels_inside_cube(point, neighbor)
            count = 0
            for voxel in inside_voxels:
                if voxels[voxel[0]][voxel[1]][voxel[2]]:
                    count += 1
            
            if (count/len(inside_voxels)) < 0.5:
                remove_points.append(point)
    
    for voxel in remove_points:
        new_voxels[voxel[0]][voxel[1]][voxel[2]] = 0
               
    return new_voxels

def remove_noisy_regions(data):
    """
    The function `remove_noisy_regions` removes noisy regions from input data based on label counts.
    
    :param data: The `remove_noisy_regions` function takes an input `data` which is presumably a 2D
    array representing an image with labeled regions. The function aims to remove noisy regions from the
    input data based on the frequency of labels in the image
    :return: The function `remove_noisy_regions` returns the input `data` array with noisy regions
    removed.
    """
    labeled_array, num_labels = ndimage.label(data)
    labels, counts = np.unique(
        labeled_array[labeled_array != 0], return_counts=True)
    max_count_label = labels[np.argmax(counts)]
    data[(labeled_array != 0) & (labeled_array != max_count_label)] = 0

    return data


def remove_noisy_voxels(voxels, neighbor_threshold, is_remove):
    """
    The function `remove_noisy_voxels` takes a 3D array of voxels, counts the number of non-zero
    neighboring voxels for each voxel using a 3x3x3 kernel, and removes noisy voxels based on a
    condition, returning the updated voxel array.

    :param voxels: The code you provided is a function that removes noisy voxels from a 3D array based
    on the number of non-zero neighboring voxels. The function uses a 3x3x3 kernel to check the 26
    neighboring voxels around each voxel and removes the voxel if it has fewer than
    :return: The function `remove_noisy_voxels` returns the result after applying the condition to
    remove noisy voxels. It sets the value of a voxel to 0 if the voxel is noisy, which is defined as
    having a value of 1 and having less than 5 non-zero neighboring voxels.
    """
    # Define a 3x3x3 kernel to check the 26 neighbor_distances
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0

    # Count the number of non-zero neighbor_distances for each voxel
    neighbor_counts = np.zeros_like(voxels)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i == 0 and j == 0 and k == 0:
                    continue
                neighbor_counts += np.roll(voxels, (i, j, k), axis=(0, 1, 2))

    # Apply the condition to remove noisy voxels
    if is_remove:
        result = np.where((voxels == 1) & (neighbor_counts < neighbor_threshold), 0, voxels)
    else:
        result = np.where((voxels == 0) & (neighbor_counts >= neighbor_threshold), 1, voxels)

    return result


def find_intensity_threshold(preprocessed_data, change_interval=0):
    """
    The function `find_intensity_threshold` calculates an intensity threshold based on preprocessed data
    and a change interval.

    :param preprocessed_data: Preprocessed_data is the input data that has been processed or cleaned in
    some way before being passed to the `find_intensity_threshold` function. It seems like the function
    calculates a histogram of the preprocessed data and then computes an intensity threshold based on
    the median of the non-zero values in the data
    :param change_interval: The `change_interval` parameter in the `find_intensity_threshold` function
    represents the amount by which you want to adjust the intensity threshold calculated based on the
    median of the preprocessed data. It allows you to fine-tune the threshold value by adding or
    subtracting a specific amount, defaults to 0 (optional)
    :return: The function `find_intensity_threshold` returns the calculated value of
    `intensity_threshold_1`.
    """
    bins = np.arange(0.1, 1.1, 0.1)
    hist, _ = np.histogram(preprocessed_data, bins=bins)
    intensity_threshold = round(np.median(
        preprocessed_data[preprocessed_data != 0]) + change_interval, 1) + change_interval

    return intensity_threshold


def preprocess_data(
    original_data,
    segment_data,
    index=[],
    intensity_threshold_1=0.65,
    intensity_threshold_2=0.65,
    gaussian_sigma=0,
    neighbor_threshold_1=8,
    neighbor_threshold_2=15
):
    """
    The `preprocess_data` function preprocesses original and segmented data by applying Gaussian
    filtering, normalization, intensity thresholding, and noise removal.

    :param original_data: Original 3D image data before preprocessing
    :param segment_data: Segment_data is a numpy array representing the binary mask image that contains
    information about different segments or regions in the original data. In this context, it is used to
    select specific voxels corresponding to desired arteries based on the provided index values
    :param index: The `index` parameter in the `preprocess_data` function is used to specify a list of
    desired arteries to select from the binary mask image. Only voxels corresponding to the indices
    provided in the `index` list will be retained in the mask data, while all other voxels will be set
    :param intensity_threshold_1: The `intensity_threshold_1` parameter in the `preprocess_data`
    function is used to set the lower threshold for voxel intensity values in the normalized data.
    Voxels with intensity values below this threshold will be set to 0, while those above or equal to
    the threshold will be set to
    :param intensity_threshold_2: The `intensity_threshold_2` parameter in the `preprocess_data`
    function is used to set the threshold value for selecting voxels in the `surf_data` array based on
    their intensity levels. Voxels with intensity values below `intensity_threshold_2` are set to 0,
    :param gaussian_sigma: The `gaussian_sigma` parameter in the `preprocess_data` function represents
    the standard deviation of the Gaussian filter that is applied to the original data for noise
    reduction. A higher value for `gaussian_sigma` will result in a smoother output image by increasing
    the amount of blurring applied during the, defaults to 0 (optional)
    :return: The `preprocess_data` function returns three numpy arrays: `mask_data`, `cex_data`, and
    `surf_data`.
    """

    # Apply Gaussian filtering to remove noisy voxels in original image
    preprocessed_data = np.copy(original_data)
    preprocessed_data = gaussian_filter(
        preprocessed_data, sigma=gaussian_sigma)

    # Load binary mask image into numpy array + Select voxels with desired arteries
    mask_data = np.copy(segment_data)
    mask = np.isin(mask_data, index, invert=True)
    mask_data[mask] = 0
    mask_data[mask_data != 0] = 1

    # Normalize data
    min_value = np.min(preprocessed_data[mask_data != 1]) - 1
    preprocessed_data[mask_data != 1] = min_value
    normalized_data = (preprocessed_data - np.min(preprocessed_data)) / \
        (np.max(preprocessed_data) - np.min(preprocessed_data))
    cex_data = np.copy(normalized_data)
    surf_data = np.copy(normalized_data)

    # Choose intensity threshold in range [0->1] after normalization
    if intensity_threshold_1 == 0:
        intensity_threshold_1 = find_intensity_threshold(normalized_data, 0.05)

    if intensity_threshold_2 == 0:
        intensity_threshold_2 = intensity_threshold_1

    # Select voxels matching intensity threshold + remove noisy voxels
    cex_data[cex_data < intensity_threshold_1] = 0
    cex_data[cex_data >= intensity_threshold_1] = 1
    
    cex_data = remove_noisy_voxels(cex_data, neighbor_threshold_1, True)
    cex_data = remove_noisy_regions(cex_data)
    
    unchanged = False
    loop = 0
    while loop <= 200 and unchanged == False:
        loop += 1
        old_count = np.argwhere(cex_data == 1).shape[0]
        cex_data = remove_noisy_voxels(cex_data, neighbor_threshold_2, False)
        new_count = np.argwhere(cex_data == 1).shape[0]
        
        if old_count == new_count:
            unchanged = True
            
    cex_data = set_voxel_to_zero(cex_data)

    # Select voxels matching intensity threshold + remove noisy voxels
    surf_data[surf_data < intensity_threshold_2] = 0
    surf_data[surf_data >= intensity_threshold_2] = 1
    surf_data = remove_noisy_voxels(surf_data, neighbor_threshold_1, neighbor_threshold_2)
    surf_data = remove_noisy_regions(surf_data)
    
    return mask_data, cex_data, surf_data
