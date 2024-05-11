import nibabel as nib
import numpy as np

from artery_ica import *
from slice_selection import *

dataset_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/'

segment_file_path = dataset_dir + 'kissing_vessel.nii.gz'
original_file_path = dataset_dir + 'sub-9_run-1_mra_resampled.nii.gz'

segment_image = nib.load(segment_file_path)
original_image = nib.load(original_file_path)

original_data = original_image.get_fdata()
processed_mask = segment_image.get_fdata()

# intensity_threshold_1 = 0.1
# intensity_threshold_2 = 0.1
# gaussian_sigma=2
# distance_threshold=20
# laplacian_iter = 5
# neighbor_threshold_1 = 10
# neighbor_threshold_2 = neighbor_threshold_1 + 10
# resolution = 0.05

# min_value = np.min(original_data)
# processed_mask = find_skeleton_ica(segment_image, original_image, 2 , 0.5, intensity_threshold_2, gaussian_sigma, neighbor_threshold_1, neighbor_threshold_2)
# processed_mask = remove_noisy_voxels(processed_mask, neighbor_threshold_1, True)

skeleton = skeletonize(processed_mask)

# Get non-zero indices
nonzero_indices = np.nonzero(processed_mask)

# Find the minimum and maximum indices along each axis
min_x = np.min(nonzero_indices[0])
max_x = np.max(nonzero_indices[0])

min_y = np.min(nonzero_indices[1])
max_y = np.max(nonzero_indices[1])

min_z = np.min(nonzero_indices[2])
max_z = np.max(nonzero_indices[2])

# Iterate over each axis and extract slices
for axis in range(3):
    if axis == 0:
        axis_name = 'X'
        min_axis = min_x
        max_axis = max_x
    elif axis == 1:
        axis_name = 'Y'
        min_axis = min_y
        max_axis = max_y
    else:
        axis_name = 'Z'
        min_axis = min_z
        max_axis = max_z
    
    for i in range(min_axis, max_axis):
        slices = [slice(None)] * 3
        slices[axis] = i
        
        intensity_slice = original_data[tuple(slices)]
        segment_slice = processed_mask[tuple(slices)]
        skeleton_slice = skeleton[tuple(slices)]

        nonzero_indices_p = np.nonzero(segment_slice)
        min_x_p = np.min(nonzero_indices_p[0])
        max_x_p = np.max(nonzero_indices_p[0])
        min_y_p = np.min(nonzero_indices_p[1])
        max_y_p = np.max(nonzero_indices_p[1])

        intensity_slice_p = intensity_slice[min_x_p:max_x_p, min_y_p:max_y_p]
        segment_slice_p = segment_slice[min_x_p:max_x_p, min_y_p:max_y_p]
        skeleton_slice_p = skeleton_slice[min_x_p:max_x_p, min_y_p:max_y_p]
        skeleton_points = np.argwhere(skeleton_slice_p > 0)

        min_value = np.min(intensity_slice_p[segment_slice_p > 0]) - 1
        intensity_slice_p[segment_slice_p == 0] = min_value  
        min_val = np.min(intensity_slice_p)
        max_val = np.max(intensity_slice_p)
        intensity_slice_p = (intensity_slice_p - min_val) / (max_val - min_val)

        visualize_slice(intensity_slice_p, segment_slice_p, segment_slice_p, skeleton_points, 0, i, axis)
# for i in range(min_z, max_z):
#     intensity_slice = original_data[min_x:max_x, min_y:max_y, i]
#     segment_slice = processed_mask[min_x:max_x, min_y:max_y, i]
#     skeleton_slice = skeleton[min_x:max_x, min_y:max_y, i]

#     nonzero_indices_p = np.nonzero(segment_slice)
#     min_x_p = np.min(nonzero_indices_p[0])
#     max_x_p = np.max(nonzero_indices_p[0])
#     min_y_p = np.min(nonzero_indices_p[1])
#     max_y_p = np.max(nonzero_indices_p[1])


#     intensity_slice_p = intensity_slice[min_x_p:max_x_p, min_y_p:max_y_p]
#     segment_slice_p = segment_slice[min_x_p:max_x_p, min_y_p:max_y_p]
#     skeleton_slice_p = skeleton_slice[min_x_p:max_x_p, min_y_p:max_y_p]
#     skeleton_points = np.argwhere(skeleton_slice_p > 0)

#     min_value = np.min(intensity_slice_p[segment_slice_p > 0]) - 1
#     intensity_slice_p[segment_slice_p == 0] = min_value  
#     min_val = np.min(intensity_slice_p)
#     max_val = np.max(intensity_slice_p)
#     intensity_slice_p = (intensity_slice_p - min_val) / (max_val - min_val)

#     visualize_slice(intensity_slice_p, segment_slice_p, segment_slice_p, skeleton_points, 0, i, 2)


# min_value = np.min(original_data[processed_mask > 0]) - 1
# original_data[processed_mask == 0] = min_value

# min_val = np.min(original_data)
# max_val = np.max(original_data)
# normalized_array = (original_data - min_val) / (max_val - min_val)


# nifti_img = nib.Nifti1Image(normalized_array, original_image.affine)
# nib.save(nifti_img, dataset_dir + '/sub-9_run-1_ica.nii.gz')