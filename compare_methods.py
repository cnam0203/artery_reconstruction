import nibabel as nib
import numpy as np

from skimage import measure
from scipy.ndimage import distance_transform_edt
from visualize_graph import *

if __name__ == "__main__":

    # Calculate runtime - record start time
    dataset_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/'
    
    # Specify the path to your NIfTI file
    segment_file_path = dataset_dir + 'sub-4947_TOF_multiclass_segmentation.nii.gz'
    original_file_path = dataset_dir + 'sub-4947_run-1_mra_TOF.nii.gz'

    segment_image = nib.load(segment_file_path)
    original_image = nib.load(original_file_path)

    original_data = original_image.get_fdata()
    segment_data = segment_image.get_fdata()

    # segment_data = np.array([[
    #     [0, 0, 0], 
    #     [0, 0, 0],
    #     [0, 0, 0]
    # ], [
    #     [0, 0, 0], 
    #     [0, 1, 0],
    #     [0, 0, 0]
    # ], [
    #     [0, 0, 0], 
    #     [0, 0, 0],
    #     [0, 0, 0]
    # ]])
    total_points = segment_data.shape[0]*segment_data.shape[1]*segment_data.shape[2]
    segment_data[segment_data > 1] = 1
    voxel_sizes = segment_image.header.get_zooms()

    # Calculate the Euclidean distance transform
    distance_transform = distance_transform_edt(segment_data)
    boundary_points = np.argwhere(distance_transform >= 1)
    boundary_values = distance_transform[tuple(boundary_points.T)]
    print(boundary_values.shape)

    verts, faces, normals, values = measure.marching_cubes(segment_data, level=0.9, spacing=voxel_sizes)

    # line_traces = []
    # for face in faces:
    #     line_traces.append(generate_lines(np.array([verts[face[0]], verts[face[1]]]), 2))
    #     line_traces.append(generate_lines(np.array([verts[face[1]], verts[face[2]]]), 2))
    #     line_traces.append(generate_lines(np.array([verts[face[0]], verts[face[2]]]), 2))

    # for point in boundary_points:
    #     delta_1 = [0.5, 0, 0]
    #     delta_2 = [0, 0.5, 0]
    #     delta_3 = [0, 0, 0.5]

    #     new_points = [
    #         point + delta_1 + delta_2 + delta_3,
    #         point + delta_1 + delta_2 - delta_3,
    #         point + delta_1 - delta_2 - delta_3,
    #         point + delta_1 - delta_2 + delta_3,

    #         point - delta_1 + delta_2 + delta_3,
    #         point - delta_1 + delta_2 - delta_3,
    #         point - delta_1 - delta_2 - delta_3,
    #         point - delta_1 - delta_2 + delta_3,
    #     ]

    #     for i in range(len(new_points)):
    #         for j in range(len(new_points)):
    #             line_traces.append(generate_lines(np.array([new_points[i], new_points[j]]), 1, 'red'))

    # print("Distance transform:")
    # print(total_points)
    # print(boundary_points.shape[0])
    # print(verts.shape[0])
    # print(faces.shape[0])

    visualized_boundary_points = generate_points_viridis(boundary_points, 5, boundary_values)
    # visualized_vert_points = generate_points(verts, 2)
    show_figure([
                visualized_boundary_points, 
                # visualized_vert_points, 
            ] 
            # + line_traces
    )