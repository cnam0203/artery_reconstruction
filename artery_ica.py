import nibabel as nib
from skimage import measure
# import open3d as o3d
import numpy as np
# import plotly.graph_objs as go
from skimage.morphology import skeletonize, thin
from scipy.spatial import KDTree
from scipy.spatial import distance_matrix
import math
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from preprocess_data import *
from process_graph import *
from visualize_graph import *

# Nearest neighbor algorithm
def nearest_neighbor(points):
    distances = distance_matrix(points, points)
    n = len(points)
    unvisited = set(range(n))
    current_point = 0
    path = [current_point]
    unvisited.remove(current_point)

    while unvisited:
        nearest_point = min(unvisited, key=lambda x: distances[current_point, x])
        path.append(nearest_point)
        unvisited.remove(nearest_point)
        current_point = nearest_point

    return path

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def find_angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)

    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the cosine of the angle between the vectors
    cosine_angle = dot_product / (magnitude1 * magnitude2 + 0.1)

    # Calculate the angle in radians
    angle_radians = np.arccos(cosine_angle)
    
    return abs(np.degrees(angle_radians))

# def remove_noisy_voxels(voxels):
#     # Define a 3x3x3 kernel to check the 26 neighbor_distances
#     kernel = np.ones((3, 3, 3))
#     kernel[1, 1, 1] = 0

#     # Count the number of non-zero neighbor_distances for each voxel
#     neighbor_counts = np.zeros_like(voxels)
#     for i in range(-1, 2):
#         for j in range(-1, 2):
#             for k in range(-1, 2):
#                 if i == 0 and j == 0 and k == 0:
#                     continue
#                 neighbor_counts += np.roll(voxels, (i, j, k), axis=(0, 1, 2))

#     # Apply the condition to remove noisy voxels
#     result = np.where((voxels == 1) & (neighbor_counts < 5), 0, voxels)

#     return result

def dfs_with_vertex_count(graph, current_node, destination, visited, path):
    visited[current_node] = True
    path.append(current_node)

    if current_node == destination:
        return path

    max_path = path
    max_vertices = 0

    for neighbor in graph[current_node]:
        if graph[current_node][neighbor] > 0 and not visited[neighbor]:
            new_path = dfs_with_vertex_count(graph, neighbor, destination, visited.copy(), path.copy())
            if len(new_path) > len(max_path):
                max_path = new_path
                max_vertices = len(new_path)

    if max_vertices > len(path):
        return max_path
    else:
        return path

def longest_path_with_no_cycles(graph, source, destination):
    visited = {node: False for node in graph}
    path = []

    return dfs_with_vertex_count(graph, source, destination, visited, path)

def smooth_path(path, window_size):
    smoothed_path = []
    for i in range(len(path)):
        start = max(0, i - window_size // 2)
        end = min(len(path), i + window_size // 2)
        segment = path[start:end]
        avg_x = sum(point[0] for point in segment) / len(segment)
        avg_y = sum(point[1] for point in segment) / len(segment)
        smoothed_path.append((avg_x, avg_y))
    return smoothed_path

def find_touchpoints(mask_data, center_points, distance_threshold=20):
    new_points = 1000000
    loop = 0
    zero_positions = np.argwhere(mask_data != 0)

    artery_data = np.zeros_like(mask_data)
    for index, pos in enumerate(center_points):
        artery_data[pos[0]][pos[1]][pos[2]] = index

    while new_points != 0:
        loop += 1
        new_points = 0
        coordinates_to_update = []
        remove_idx = []

        for idx, pos in enumerate(zero_positions):
            i, j, k = pos[0], pos[1], pos[2]

            if artery_data[i][j][k] == 0:
                index = find_nearest_point(artery_data, i, j, k, distance_threshold)
                coordinates_to_update.append((i, j, k, index))

                if (index != 0):
                    remove_idx.append(idx)
                    new_points += 1

        for i, j, k, index in coordinates_to_update:
            artery_data[i][j][k] = index

        zero_positions = np.delete(zero_positions, remove_idx, axis=0)
    
    touch_points = []
    suspected_positions = np.argwhere(artery_data > 0)

    for pos in suspected_positions:
            i, j, k = pos[0], pos[1], pos[2]
            is_touch = find_distant_neighbors(artery_data, i, j, k, distance_threshold)
            if is_touch:
                touch_points.append([i, j, k])

    for pos in touch_points:
        artery_data[pos[0]][pos[1]][pos[2]] = -1 

    touch_points = np.argwhere(artery_data == -1)
    artery_points = np.argwhere(artery_data != 0)
    artery_values = artery_data[artery_points[:, 0], artery_points[:, 1], artery_points[:, 2]]

    # point_trace = go.Scatter3d(
    #     x=artery_points[:, 0],
    #     y=artery_points[:, 1],
    #     z=artery_points[:, 2],
    #     mode='markers',
    #     marker=dict(
    #         symbol='circle',  # Set marker symbol to 'cube'
    #         size=5,
    #         color=artery_values,  # Color based on the values
    #         colorscale='Viridis',  # Colormap
    #     ),
    #     text=[f'Value: {val}' for val in artery_values],
    #     hoverinfo='text'
    # )

    # # Create layout
    # layout = go.Layout(
    #     scene=dict(
    #         aspectmode='cube',
    #         camera=dict(
    #             eye=dict(x=1, y=1, z=1)
    #         )
    #     ),
    #     height=800,  # Set height to 800 pixels
    #     width=1200   # Set width to 1200 pixels
    # )
     
    # fig = go.Figure(data=[point_trace], layout=layout)
    # fig.show()
    
    # print('Number extension loops: ', loop)
    # print('Number kissing points:', touch_points.shape[0])

    return touch_points

def interpolate_path(path):
    interpolated_points = []

    interpolated_points.append(path[0])
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]

        # Compute the distance between the two points
        distance = np.linalg.norm(np.array(p2) - np.array(p1))
        # Determine the number of interpolated points needed
        num_interpolated_points = int(distance)  # You can adjust this if needed

        # Perform linear interpolation between p1 and p2
        for j in range(1, num_interpolated_points):
            alpha = j / num_interpolated_points
            interpolated_point = tuple((np.array(p1) * (1 - alpha) + np.array(p2) * alpha).astype(int))
            interpolated_points.append(interpolated_point)

        interpolated_points.append(p2)

    return interpolated_points

def find_skeleton_ica(segment_image, original_image=None, index=None, intensity_threshold_1=0.6, intensity_threshold_2=0.1, gaussian_sigma=0, neighbor_threshold_1=8, neighbor_threshold_2=15):
    original_data = original_image.get_fdata()
    segment_data = segment_image.get_fdata()
    voxel_sizes = segment_image.header.get_zooms()

    mask_data, cex_data, surf_data = preprocess_data(original_data, segment_data, index, intensity_threshold_1, intensity_threshold_2, gaussian_sigma, neighbor_threshold_1, neighbor_threshold_2 )
    
    # Find the voxel-based skeleton
    artery_points = np.argwhere(cex_data != 0)

    img = nib.Nifti1Image(cex_data, segment_image.affine)
    nib.save(img, 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/extract_art.nii.gz')

    skeleton = skeletonize(cex_data)
    skeleton_points, end_points, junction_points, connected_lines = find_graphs(skeleton)

    visualized_thin_points = generate_points(skeleton_points, 2, 'green')

    # show_figure([
    #             visualized_thin_points,
    #         ] 
    # )

    neighbor_distances = {}

    # Initialize neighbor distances for all points
    for i, point in enumerate(skeleton_points):
        neighbor_distances[i] = {}
        for j, other_point in enumerate(skeleton_points):
            neighbor_distances[i][j] = 0
    
    for line in connected_lines:
        for i in range(len(line)-1):
            distance = euclidean_distance(skeleton_points[line[i]], skeleton_points[line[i+1]])
            neighbor_distances[line[i]][line[i+1]] = distance
            neighbor_distances[line[i+1]][line[i]] = distance

    # # Create traces
    # # Visualize artery points 
    # point_trace_0 = go.Scatter3d(
    #     x=artery_points[:, 0],
    #     y=artery_points[:, 1],
    #     z=artery_points[:, 2],
    #     mode='markers',
    #     marker=dict(
    #         size=5,
    #         color='green',
    #     ),
    #     name='Points'
    # )

    # # Visualize skeleton points 
    # point_trace_1 = go.Scatter3d(
    #     x=skeleton_points[:, 0],
    #     y=skeleton_points[:, 1],
    #     z=skeleton_points[:, 2],
    #     mode='markers',
    #     marker=dict(
    #         size=5,
    #         color='green',
    #     ),
    #     name='Points'
    # )
    
    # # Visualize junction points
    # point_trace_2 = go.Scatter3d(
    #     x=skeleton_points[junction_points, 0],
    #     y=skeleton_points[junction_points, 1],
    #     z=skeleton_points[junction_points, 2],
    #     mode='markers',
    #     marker=dict(
    #         size=5,
    #         color='black',
    #     ),
    #     name='Points'
    # )
    
    # # Visualize end points
    # point_trace_3 = go.Scatter3d(
    #     x=skeleton_points[end_points, 0],
    #     y=skeleton_points[end_points, 1],
    #     z=skeleton_points[end_points, 2],
    #     mode='markers',
    #     marker=dict(
    #         size=5,
    #         color='red',
    #     ),
    #     name='Points'
    # )

    # # vector_trace = go.Cone(
    # #     x=points[:, 0],
    # #     y=points[:, 1],
    # #     z=points[:, 2],
    # #     u=normals[:, 0],
    # #     v=normals[:, 1],
    # #     w=normals[:, 2],
    # #     sizeref=1,
    # #     name='Vectors'
    # # )

    # # Create layout
    # layout = go.Layout(
    #     scene=dict(
    #         aspectmode='cube',
    #         camera=dict(
    #             eye=dict(x=1, y=1, z=1)
    #         )
    #     ),
    #     height=800,  # Set height to 800 pixels
    #     width=1200   # Set width to 1200 pixels
    # )



    # fig_1 = go.Figure(data=[point_trace_0], layout=layout)
    # fig_1.show()

    # fig_2 = go.Figure(data=[point_trace_1, point_trace_2, point_trace_3], layout=layout)
    # fig_2.show()

    # # Calculate marching cubes using scikit-image
    # verts, faces, normals, _ = measure.marching_cubes(cex_data, level=0.5, spacing=voxel_sizes)


    # # Convert marching cubes result to Open3D mesh
    # mesh_o3d = o3d.geometry.TriangleMesh()
    # mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
    # mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    # mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_o3d.triangles)[:, ::-1])
    # mesh_o3d = mesh_o3d.filter_smooth_laplacian(1, 0.5)
    # mesh_o3d.compute_vertex_normals()
    
    # points = np.asarray(mesh_o3d.vertices)
    # normals = np.asarray(mesh_o3d.vertex_normals)


    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points) 

    # # Create a LineSet object
    # line_set = o3d.geometry.LineSet()
    # line_set.points = o3d.utility.Vector3dVector(skeleton_points)
    # lines = []
    # colors = []

    # for i in range(skeleton_points.shape[0]):
    #     for j in range(skeleton_points.shape[0]):
    #         if neighbor_distances[i][j] > neighbor_distances[j][i]:
    #             neighbor_distances[i][j] = neighbor_distances[j][i]
    #         elif neighbor_distances[i][j] < neighbor_distances[j][i]:
    #             neighbor_distances[j][i] = neighbor_distances[i][j]

    # for i in range(skeleton_points.shape[0]):
    #     for j in range(skeleton_points.shape[0]):
    #         if i != j and neighbor_distances[i][j] > 0:
    #             lines.append([i, j])
    #             colors.append([0, 0, 0])

    # line_set.lines = o3d.utility.Vector2iVector(lines)
    # line_set.colors = o3d.utility.Vector3dVector(np.array(colors))

    # Find longest path
    longest_path = []
    max_points = 0
    for i in range (len(end_points)):
        for j in range (i+1, len(end_points)): 
            path = longest_path_with_no_cycles(neighbor_distances, end_points[0], end_points[-1])
            if (len(path) > max_points):
                max_points = len(path)
                longest_path = path

    center_points = skeleton_points[longest_path].astype(int)
    inter_points = interpolate_path(center_points)
    touch_points = find_touchpoints(surf_data, inter_points, 30)

    for point in touch_points:
        surf_data[point[0]][point[1]][point[2]] = 0

    # visualized_thin_points = generate_points(center_points, 2, 'green')

    # show_figure([
    #             visualized_thin_points,
    #         ] 
    # )

    return surf_data
    # # Prepare lines for centerline from our algorithm
    # centerline_set = o3d.geometry.LineSet()
    # centerline_set.points = o3d.utility.Vector3dVector(skeleton_points)
    # centerline_lines = []
    # centerline_colors = []
    # for i in range(len(longest_path)-1):
    #     centerline_lines.append([longest_path[i], longest_path[i+1]])
    #     centerline_colors.append([1, 0, 0])

    # centerline_set.lines = o3d.utility.Vector2iVector(centerline_lines)
    # centerline_set.colors = o3d.utility.Vector3dVector(np.array(centerline_colors))

    # # Visualize the LineSet
    # o3d.visualization.draw_geometries([line_set, centerline_set, point_cloud], window_name="Line Set Visualization")

# if __name__ == "__main__":
#     # Specify the path to your NIfTI file
#     # segment_file_path = '/Users/apple/Downloads/TOF_eICAB_CW.nii.gz'
#     # original_file_path = '/Users/apple/Downloads/TOF_resampled.nii.gz'

#     dataset_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/'
#     segment_file_path = dataset_dir + 'sub-25_run-1_mra_eICAB_CW.nii.gz'
#     original_file_path = dataset_dir + 'sub-25_run-1_mra_resampled.nii.gz'

#     # segment_file_path = dataset_dir + 'sub-9_run-1_mra_eICAB_CW.nii.gz'
#     # original_file_path = dataset_dir + 'sub-9_run-1_mra_resampled.nii.gz'

#     # Load the NIfTI image
#     segment_image = nib.load(segment_file_path)
#     original_image = nib.load(original_file_path)
#     intensity_threshold_1=0.65
#     intensity_threshold_2=0.1
#     gaussian_sigma=2
#     neighbor_threshold_1 = 5
#     neighbor_threshold_2 = neighbor_threshold_1 + 10

#     # Access the image data as a NumPy array

#     find_skeleton(segment_image, original_image, 2 , intensity_threshold_1, intensity_threshold_2, gaussian_sigma, neighbor_threshold_1, neighbor_threshold_2)