import skeletor as sk
import nibabel as nib
import numpy as np

from skimage.morphology import skeletonize, thin
from skimage import measure
from scipy.ndimage import binary_dilation

from scipy.spatial import KDTree, distance_matrix
from sklearn.cluster import DBSCAN, KMeans

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math
import time
import os
import json
import heapq
import copy
import trimesh as tm
import random

from preprocess_data import *
from process_graph import *
from visualize_graph import *
from slice_selection import *
from visualize_mesh import *
from marcube import *
from artery_ica import *


import plotly.graph_objs as go
from skimage import morphology
from scipy.ndimage import distance_transform_edt

from vmtk import pypes
from vmtk import vmtkscripts
import vtk

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

def find_cross_section(centerline, surface_points):
    centerline_tree = cKDTree(centerline)
    cross_sections = []

    for idx in range(len(centerline)-1):
        point = centerline[idx]

        # Calculate tangent at the point on the centerline
        tangent = centerline[idx+1] - centerline[idx]

        # Normalize the tangent vector
        tangent /= np.linalg.norm(tangent)

        # Create a transformation matrix to align the z-axis with the tangent
        z_axis = np.array([0, 0, 1])
        rotation_matrix = R.from_rotvec(np.cross(z_axis, tangent) * np.arccos(np.dot(z_axis, tangent))).as_matrix()

        # Transform surface points
        rotated_surface_points = np.dot(rotation_matrix, (surface_points - point).T).T

        # Find points in the plane z=0 (cross-section)
        cross_section = rotated_surface_points[np.isclose(rotated_surface_points[:, 2], 0)]

        # Rotate back to the original coordinate system
        cross_section = np.dot(rotation_matrix.T, cross_section.T).T + point

        cross_sections.append(cross_section)

    return cross_sections

def reconstruct_surface(segment_image,
                  original_image=None, 
                  index=[], 
                  intensity_threshold_1=0.65, 
                  intensity_threshold_2=0.65, 
                  gaussian_sigma=0, 
                  distance_threshold=20,
                  laplacian_iter=1,
                  folder_path='',
                  neighbor_threshold_1=8,
                  neighbor_threshold_2=15):
    
    # Load original image (TOF-MRA) into numpy array
    original_data = original_image.get_fdata()
    segment_data = segment_image.get_fdata()
    voxel_sizes = segment_image.header.get_zooms()
    mask_data, cex_data, surf_data = preprocess_data(original_data, segment_data, index, intensity_threshold_1, intensity_threshold_2, gaussian_sigma, neighbor_threshold_1, neighbor_threshold_2 )
    
    verts, faces, normals, values = measure.marching_cubes(cex_data, level=0.5, spacing=voxel_sizes)
    faces = np.flip(faces, axis=1)

    return cex_data,voxel_sizes,verts, faces

def angle_between_lines(point1, point2, point3):
    # Calculate direction vectors of the two lines
    vector1 = np.array(point2) - np.array(point1)
    vector2 = np.array(point2) - np.array(point3)
    
    # # Compute dot product of the direction vectors
    # dot_product = np.dot(vector1, vector2)
    
    # # Calculate magnitudes of the direction vectors
    # magnitude_vector1 = np.linalg.norm(vector1)
    # magnitude_vector2 = np.linalg.norm(vector2)
    
    # # Calculate angle between the lines using dot product formula
    # cos_theta = dot_product / (magnitude_vector1 * magnitude_vector2)
    # angle_radians = np.arccos(cos_theta)
    # angle_degrees = np.degrees(angle_radians)
    
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)

    dot_product = np.dot(unit_vector1, unit_vector2)

    angle = np.arccos(dot_product) 
    angle_degrees = np.degrees(angle)

    return angle_degrees

def find_projection_point(p1, p2, p3):
    l2 = np.sum((p1-p2)**2)
    if l2 == 0:
        return np.array([0, 0, 0])

    t = np.sum((p3 - p1) * (p2 - p1)) / l2

    if t > 1 or t < 0:
        return np.array([0, 0, 0])
    t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2))

    projection = p1 + t * (p2 - p1)
    return projection

def find_midpoint_perpendicular_3d(p1, p2, p3):
    # projection = find_projection_point(p1, p3, p2)
    # midpoint = (np.array(p2) + projection) / 2

    midpoint_1 = (p2 + p1) / 2
    midpoint_2 = (p2 + p3) / 2
    midpoint = (midpoint_1 + midpoint_2) / 2
    return midpoint

def interpolate_points(p1, p2, num_points):
    # Create array of linearly spaced values between 0 and 1
    t_values = np.linspace(0, 1, num_points + 2)[1:-1]  # Exclude endpoints
    
    # Compute intermediate points using linear interpolation
    interpolated_points = [p1 + t * (p2 - p1) for t in t_values]
    
    return interpolated_points

def fill_noisy_voxels(voxels, value=0):
    # Create a binary mask for the noisy voxels
    noisy_mask = voxels == value
    
    # Perform binary dilation to fill in noisy voxels
    filled_voxels = binary_dilation(noisy_mask, iterations=20)
    
    # Replace noisy voxels with original value in the original voxels array
    filled_voxels = np.where(filled_voxels, voxels, value)
    
    return filled_voxels

dataset_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/'

# segment_file_path = dataset_dir + 'sub-4947_TOF_multiclass_segmentation.nii.gz'
# original_file_path = dataset_dir + 'sub-4947_run-1_mra_TOF.nii.gz'

# segment_file_path = dataset_dir + 'sub-2983_TOF_multiclass_segmentation.nii.gz'
# original_file_path = dataset_dir + 'sub-2983_run-1_mra_TOF.nii.gz'

# segment_file_path = dataset_dir + 'sub-11_TOF_multiclass_segmentation.nii.gz'
# original_file_path = dataset_dir + 'sub-11_run-1_mra_TOF.nii.gz'

# segment_file_path = dataset_dir + 'sub-1057_TOF_multiclass_segmentation.nii.gz'
# original_file_path = dataset_dir + 'sub-1057_run-1_mra_TOF.nii.gz'

# segment_file_path = dataset_dir + 'sub-2849_TOF_multiclass_segmentation.nii.gz'
# original_file_path = dataset_dir + 'sub-2849_run-1_mra_TOF.nii.gz'

# segment_file_path = dataset_dir + 'sub-2049_TOF_multiclass_segmentation.nii.gz'
# original_file_path = dataset_dir + 'sub-2049_run-1_mra_TOF.nii.gz'

# segment_file_path = dataset_dir + 'sub-1425_TOF_multiclass_segmentation.nii.gz'
# original_file_path = dataset_dir + 'sub-1425_run-1_mra_TOF.nii.gz'

# segment_file_path = dataset_dir + 'TOF_multiclass_segmentation.nii.gz'
# original_file_path = dataset_dir + 'sub-1_run-1_mra_TOF.nii.gz'

segment_file_path = dataset_dir + 'sub-9_run-1_mra_eICAB_CW.nii.gz'
original_file_path = dataset_dir + 'sub-9_run-1_mra_resampled.nii.gz'
centerline_file_path = dataset_dir + 'sub-9_run-1_mra_CircleOfWillis_centerline.nii.gz'

segment_image = nib.load(segment_file_path)
original_image = nib.load(original_file_path)
centerline_image = nib.load(centerline_file_path)

intensity_threshold_1 = 0.1
intensity_threshold_2 = 0.1
gaussian_sigma=2
distance_threshold=20
laplacian_iter = 5
neighbor_threshold_1 = 10
neighbor_threshold_2 = neighbor_threshold_1 + 10
resolution = 0.05

original_data = original_image.get_fdata()
segment_data = segment_image.get_fdata()
centerline_data = centerline_image.get_fdata()
voxel_sizes = segment_image.header.get_zooms()

# processed_mask = segment_data
# processed_mask = find_skeleton_ica(segment_image, original_image, 6 , 0.5, intensity_threshold_2, gaussian_sigma, neighbor_threshold_1, neighbor_threshold_2)
# processed_mask = remove_noisy_voxels(processed_mask, neighbor_threshold_1, True)


mask_data, processed_mask, surf_data = preprocess_data(original_data, segment_data, 6, intensity_threshold_1, intensity_threshold_2, gaussian_sigma, neighbor_threshold_1, neighbor_threshold_2 )
# print(touch_points)

#Find skeleton
# cex_data, voxel_sizes, vertices, faces = reconstruct_surface(
#                 segment_image, 
#                 original_image, 
#                 index=[1], 
#                 intensity_threshold_1=intensity_threshold_1, 
#                 intensity_threshold_2=intensity_threshold_2, 
#                 gaussian_sigma=gaussian_sigma, 
#                 distance_threshold=distance_threshold,
#                 laplacian_iter=laplacian_iter,
#                 folder_path='',
#                 neighbor_threshold_1=neighbor_threshold_1,
#                 neighbor_threshold_2=neighbor_threshold_2
#             )

# vertices, faces, normals, values = measure.marching_cubes(processed_mask, level=0.5, spacing=voxel_sizes)
skeleton = skeletonize(processed_mask)
skeleton_points_1, end_points, junction_points, connected_lines = find_graphs(skeleton)

new_connected_lines = []
for line in connected_lines:
    if len(line) <= 7 and ((line[0] in end_points) or (line[-1] in end_points)):
        for point in line:
            if point not in junction_points:
                pos = skeleton_points_1[point].astype(int)
                skeleton[pos[0]][pos[1]][pos[2]] = 0
    else:
        new_connected_lines.append(line)


for index, line in enumerate(new_connected_lines):
    new_points = []

    if line[0] in end_points:
        if len(line) <= 7:
            consult_points = skeleton_points_1[line]
        else:
            consult_points = skeleton_points_1[line[:8]]

        point1 = consult_points[0]
        point2 = consult_points[-1]
        direction_vectors = np.abs(np.array(point2) - np.array(point1))
        max_value = np.max(direction_vectors)
        positions = np.where(direction_vectors == max_value)[0]
        axis = positions[0]

        increment = -1
        if point1[axis] > point2[axis]:
            increment = 1
        
        is_found = False
        new_centerpoint = np.copy(point1)

        while not is_found:
            cur_point = np.copy(new_centerpoint)
            cur_point[axis] = cur_point[axis] + increment
            i = cur_point[axis]
            
            intensity_slice_2d = original_data.take(i, axis=axis)
            segment_slice_2d = processed_mask.take(i, axis=axis)

            patch_size = 2
            x, y, z = cur_point[0], cur_point[1], cur_point[2]    

            if axis == 0:
                x_start = max(0, y - patch_size // 2)
                x_end = min(original_data.shape[1], y + patch_size // 2 + 1)
                y_start = max(0, z - patch_size // 2)
                y_end = min(original_data.shape[2], z + patch_size // 2 + 1)
            elif axis == 1:
                x_start = max(0, x - patch_size // 2)
                x_end = min(original_data.shape[0], x + patch_size // 2 + 1)
                y_start = max(0, z - patch_size // 2)
                y_end = min(original_data.shape[2], z + patch_size // 2 + 1)
            else:
                x_start = max(0, x - patch_size // 2)
                x_end = min(original_data.shape[0], x + patch_size // 2 + 1)
                y_start = max(0, y - patch_size // 2)
                y_end = min(original_data.shape[1], y + patch_size // 2 + 1)

            x_start, x_end, y_start, y_end = int(x_start), int(x_end), int(y_start), int(y_end)
            intensity_slice = intensity_slice_2d[x_start:x_end, y_start:y_end]
            segment_slice = segment_slice_2d[x_start:x_end, y_start:y_end]
            
            if not np.any(segment_slice > 0):
                is_found = True
            else:
                # Find the position in segment_slice where value = 1
                indices = np.argwhere(segment_slice > 0)

                # Initialize variables to store the highest intensity and its coordinates
                max_intensity = 0
                max_intensity_position = None

                # Loop through the found positions
                for pos in indices:
                    # Get the intensity at the current position
                    intensity = intensity_slice[pos[0], pos[1]]
                    
                    # Check if the intensity is higher than the current maximum
                    if intensity > max_intensity:
                        max_intensity = intensity
                        max_intensity_position = pos


                new_centerpoint = []
                if axis == 0:
                    new_centerpoint = [i, x_start+max_intensity_position[0], y_start+max_intensity_position[1]]
                elif axis == 1:
                    new_centerpoint = [x_start+max_intensity_position[0], i, y_start+max_intensity_position[1]]
                else:
                    new_centerpoint = [x_start+max_intensity_position[0], y_start+max_intensity_position[1], i]

                
                skeleton[int(cur_point[0])][int(cur_point[1])][int(cur_point[2])] = 1
                skeleton[int(new_centerpoint[0])][int(new_centerpoint[1])][int(new_centerpoint[2])] = 1

                new_points += [new_centerpoint, cur_point]
                # visualize_slice(intensity_slice, segment_slice, segment_slice, [], point1[axis], new_centerpoint[axis] - point1[axis], axis)
        
        indices = []
        for point in new_points:
            cur_index = skeleton_points_1.shape[0]
            indices.append(cur_index)
            skeleton_points_1 = np.vstack([skeleton_points_1, np.array([point])])

        new_connected_lines[index] = indices + new_connected_lines[index]


    if line[-1] in end_points:
        if len(line) <= 7:
            consult_points = skeleton_points_1[line]
        else:
            consult_points = skeleton_points_1[line[-7:]]

        point1 = consult_points[-1]
        point2 = consult_points[0]
        direction_vectors = np.abs(np.array(point2) - np.array(point1))
        max_value = np.max(direction_vectors)
        positions = np.where(direction_vectors == max_value)[0]
        axis = positions[0]

        increment = -1
        if point1[axis] > point2[axis]:
            increment = 1
        
        is_found = False
        new_centerpoint = np.copy(point1)

        while not is_found:
            cur_point = np.copy(new_centerpoint)
            cur_point[axis] = cur_point[axis] + increment
            i = cur_point[axis]
            half_edge = 3
            
            intensity_slice_2d = original_data.take(i, axis=axis)
            segment_slice_2d = processed_mask.take(i, axis=axis)

            patch_size = 2
            x, y, z = cur_point[0], cur_point[1], cur_point[2]    

            if axis == 0:
                x_start = max(0, y - patch_size // 2)
                x_end = min(original_data.shape[1], y + patch_size // 2 + 1)
                y_start = max(0, z - patch_size // 2)
                y_end = min(original_data.shape[2], z + patch_size // 2 + 1)
            elif axis == 1:
                x_start = max(0, x - patch_size // 2)
                x_end = min(original_data.shape[0], x + patch_size // 2 + 1)
                y_start = max(0, z - patch_size // 2)
                y_end = min(original_data.shape[2], z + patch_size // 2 + 1)
            else:
                x_start = max(0, x - patch_size // 2)
                x_end = min(original_data.shape[0], x + patch_size // 2 + 1)
                y_start = max(0, y - patch_size // 2)
                y_end = min(original_data.shape[1], y + patch_size // 2 + 1)

            x_start, x_end, y_start, y_end = int(x_start), int(x_end), int(y_start), int(y_end)
            intensity_slice = intensity_slice_2d[x_start:x_end, y_start:y_end]
            segment_slice = segment_slice_2d[x_start:x_end, y_start:y_end]

            if not np.any(segment_slice > 0):
                is_found = True
            else:
                # Find the position in segment_slice where value = 1
                indices = np.argwhere(segment_slice > 0)

                # Initialize variables to store the highest intensity and its coordinates
                max_intensity = 0
                max_intensity_position = None

                # Loop through the found positions
                for pos in indices:
                    # Get the intensity at the current position
                    intensity = intensity_slice[pos[0], pos[1]]
                    
                    # Check if the intensity is higher than the current maximum
                    if intensity > max_intensity:
                        max_intensity = intensity
                        max_intensity_position = pos


                new_centerpoint = []
                if axis == 0:
                    new_centerpoint = [i, x_start+max_intensity_position[0], y_start+max_intensity_position[1]]
                elif axis == 1:
                    new_centerpoint = [x_start+max_intensity_position[0], i, y_start+max_intensity_position[1]]
                else:
                    new_centerpoint = [x_start+max_intensity_position[0], y_start+max_intensity_position[1], i]

                skeleton[int(cur_point[0])][int(cur_point[1])][int(cur_point[2])] = 1
                skeleton[int(new_centerpoint[0])][int(new_centerpoint[1])][int(new_centerpoint[2])] = 1

                new_points += [cur_point, new_centerpoint]
                # visualize_slice(intensity_slice, segment_slice, segment_slice, [], point1[axis], new_centerpoint[axis] - point1[axis], axis)

        indices = []
        for point in new_points:
            cur_index = skeleton_points_1.shape[0]
            indices.append(cur_index)
            skeleton_points_1 = np.vstack([skeleton_points_1,  np.array([point])])

        new_connected_lines[index] = new_connected_lines[index] + indices


vertices, faces, normals, values = measure.marching_cubes(skeleton, level=0.5, spacing=voxel_sizes)
skeleton_points_1 = voxel_sizes*(skeleton_points_1 + 0.5)


# # old_skeleton_points_1 = np.copy(skeleton_points_1)
# # old_connected_lines = copy.deepcopy(connected_lines)

# Smoother:
# Create a VTK PolyData object representing the surface
surface_data = vtk.vtkPolyData()
points = vtk.vtkPoints()
for vertex in vertices:
    points.InsertNextPoint(vertex)
surface_data.SetPoints(points)

# Add polygons to the surface
polygons = vtk.vtkCellArray()
for face in faces:
    polygon = vtk.vtkPolygon()
    for index in face:
        polygon.GetPointIds().InsertNextId(index)
    polygons.InsertNextCell(polygon)
surface_data.SetPolys(polygons)

mySmoother = vmtkscripts.vmtkSurfaceSmoothing()
mySmoother.Surface = surface_data
mySmoother.PassBand = 0.1
mySmoother.NumberOfIterations = 2000
mySmoother.Execute()
smoothed_surface = mySmoother.Surface


myMeshGenerator = vmtkscripts.vmtkSurfaceSubdivision()
myMeshGenerator.Surface = smoothed_surface
myMeshGenerator.NumberOfSubdivisions = 2
myMeshGenerator.Execute()
smoothed_surface = myMeshGenerator.Surface	

# Get vertices and faces from the surface
vmtk_vertices = []
vmtk_faces = []

points = smoothed_surface.GetPoints()
for i in range(points.GetNumberOfPoints()):
    point = points.GetPoint(i)
    vmtk_vertices.append(point)

cells = smoothed_surface.GetPolys()
cells.InitTraversal()
while True:
    cell = vtk.vtkIdList()
    if cells.GetNextCell(cell) == 0:
        break
    face = [cell.GetId(j) for j in range(cell.GetNumberOfIds())]
    vmtk_faces.append(face)

# Convert vertices and faces to NumPy arrays
vmtk_vertices = np.array(vmtk_vertices)
vmtk_faces = np.array(vmtk_faces)

points = vmtk_vertices
skeleton_points = skeleton_points_1
connected_lines = new_connected_lines

tree = KDTree(skeleton_points)
distances, indices = tree.query(points, k=3)

clusters = {}
clusters_start_point = {}
clusters_min_dist = {}

for index, close_points in enumerate(indices):
    for point_index, point in enumerate(close_points):
        if point in junction_points:
            continue
        else:
            if point not in clusters:
                clusters[point] = []
                clusters_start_point[point] = None
                clusters_min_dist[point] = 0
            
            clusters[point].append(index)

            if clusters_start_point[point] is None:
                clusters_start_point[point] = index
                clusters_min_dist[point] = distances[index][point_index]
            else:
                if distances[index][point_index] < clusters_min_dist[point]:
                    clusters_start_point[point] = index
                    clusters_min_dist[point] = distances[index][point_index]
            break

arrange_lines = {}
for cluster_name, cluster_points in clusters.items():
    arrange_line = []
    arrange_line.append(clusters_start_point[cluster_name])
    reduced_points = cluster_points.copy()
    reduced_points.remove(clusters_start_point[cluster_name])

    while len(arrange_line) < len(cluster_points):
        start_point = points[arrange_line[0]]
        end_point = points[arrange_line[-1]]

        tree = KDTree(points[reduced_points])

        distances_1, indices_1 = tree.query(start_point)
        distances_2, indices_2 = tree.query(end_point)


        if distances_1 < distances_2:
            reduce_point = reduced_points[indices_1]
            arrange_line.insert(0, reduce_point)
        else:
            reduce_point = reduced_points[indices_2]
            arrange_line.append(reduce_point)

        reduced_points.remove(reduce_point)

    arrange_lines[cluster_name] = arrange_line

smooth_connected_lines = []

for line in connected_lines:
    smooth_line = []

    for point in line:
        if point not in junction_points:
            if point in arrange_lines and len(smooth_line) == 0:
                smooth_line += arrange_lines[point]
            elif point in arrange_lines:
                start_point = points[smooth_line[0]]
                end_point = points[smooth_line[-1]]
                tree = KDTree(points[[arrange_lines[point][-1], arrange_lines[point][0]]])

                distances_1, indices_1 = tree.query(start_point)
                distances_2, indices_2 = tree.query(end_point)

                # smooth_connected_lines.append(arrange_lines[point])

                if distances_1 < distances_2:
                    if indices_1 == 0:
                        smooth_line = arrange_lines[point] + smooth_line
                    else:
                        smooth_line = arrange_lines[point][::-1] + smooth_line
                else:
                    if indices_2 == 0:
                        smooth_line = smooth_line + arrange_lines[point][::-1]
                    else:
                        smooth_line = smooth_line + arrange_lines[point]

    smooth_connected_lines.append(smooth_line)

for point in junction_points:
    center_points = []
    pos = {}
    for index, line in enumerate(connected_lines):
        if line[0] == point or line[-1] == point:
            distance_1 = euclidean_distance(skeleton_points[point], points[smooth_connected_lines[index][0]])
            distance_2 = euclidean_distance(skeleton_points[point], points[smooth_connected_lines[index][-1]])
            if distance_1 < distance_2:
                center_points.append(smooth_connected_lines[index][0])
                pos[index] = 0
            else:
                center_points.append(smooth_connected_lines[index][-1])
                pos[index] = -1

    avg_point = np.mean(points[center_points], axis=0)
    new_index = points.shape[0]
    points = np.vstack([points, avg_point])

    for index, line in enumerate(connected_lines):
        if line[0] == point or line[-1] == point:
            if pos[index] == 0:
                smooth_connected_lines[index].insert(0, new_index)
            else:
                smooth_connected_lines[index].append(new_index)

num_loop = 100

while num_loop > 0:
    num_loop -= 1
    for line in smooth_connected_lines:
        for i in range(len(line) - 2):
            point_1 = points[line[i]]
            point_2 = points[line[i+1]]
            point_3 = points[line[i+2]]
            angle = angle_between_lines(point_1, point_2, point_3)
            if angle < 175:
                new_mid_point = (point_1 + point_3)/2
                points[line[i+1]] = new_mid_point

# num_loop = 50

# while num_loop > 0:
#     num_loop -= 1
#     new_connected_lines = []

#     for line in smooth_connected_lines:
#         new_line = []

#         new_line.append(line[0])
#         for i in range(len(line)-1):
#             point_1 = points[line[i]]
#             point_2 = points[line[i+1]]
#             distance = euclidean_distance(point_1, point_2)
#             new_num_points = int(distance/resolution) - 2
#             if new_num_points > 0:
#                 interpolated_points = interpolate_points(point_1, point_2, new_num_points)
#                 new_indices = []
#                 for point in interpolated_points:
#                     new_indices.append(points.shape[0])
#                     points = np.vstack([points, point])
                
#                 new_line += new_indices
#             new_line.append(line[i+1])
#         new_connected_lines.append(new_line)

#     smooth_connected_lines = new_connected_lines

line_traces = []

# skeleton_points_2 = skeleton_points_2 * voxel_sizes
# visualized_skeleton_points = generate_points(skeleton_points_2, 3, 'red')
visualized_vmtk_points = generate_points(points, 2, 'green')

for line in smooth_connected_lines:
    line_traces.append(generate_lines(points[line], 2, 'green'))

# for line in connected_lines_2:
#     line_traces.append(generate_lines(skeleton_points_2[line], 2, 'red'))



# # # # mesh_bound = tm.Trimesh(vertices=vertices, faces=faces, enable_post_processing=True, solid=True)
# # mesh_bound = tm.Trimesh(vertices=vmtk_vertices, faces=vmtk_faces, enable_post_processing=True, solid=True)
# # mesh_bound.export('C:/Users/nguc4116/Desktop/artery_reconstruction/mesh/smooth_mesh_11.stl')

# # # mesh_copy = tm.Trimesh(vertices=vmtk_vertices, faces=vmtk_faces, enable_post_processing=True, solid=True)
# # # mesh_sub = sk.pre.contract(mesh_copy, epsilon=0.2)
# # # mesh_sub = sk.pre.fix_mesh(mesh_sub, remove_disconnected=5, inplace=False)

# # # skeleton_points_1 = skeleton_points_1
# # # skeleton_points_2 = mesh_sub.vertices


# # # skeleton_3 = sk.skeletonize.by_wavefront(mesh_sub, waves=1)
# # # skeleton_3 = sk.post.clean_up(skeleton_3)
# # # skeleton_3 = sk.post.smooth(skeleton_3)
# # # edges_3 = skeleton_3.edges

# # # vertices_count = {}
# # # endpoint_idx = []
# # # for edge in edges_3:
# # #     vertice_1 = edge[0]
# # #     vertice_2 = edge[1]

# # #     if vertice_1 not in vertices_count:
# # #         vertices_count[vertice_1] = 0
# # #     if vertice_2 not in vertices_count:
# # #         vertices_count[vertice_2] = 0

# # #     vertices_count[vertice_1] += 1
# # #     vertices_count[vertice_2] += 1

# # # for key, value in vertices_count.items():
# # #     if value == 1:
# # #         endpoint_idx.append(key)

# # # end_points = skeleton_3.vertices[endpoint_idx]

# # # vertices_count = {}
# # # endpoint_idx = []
# # # for line in connected_lines:
# # #     vertice_1 = line[0]
# # #     vertice_2 = line[-1]

# # #     if vertice_1 not in vertices_count:
# # #         vertices_count[vertice_1] = 0
# # #     if vertice_2 not in vertices_count:
# # #         vertices_count[vertice_2] = 0

# # #     vertices_count[vertice_1] += 1
# # #     vertices_count[vertice_2] += 1

# # # for key, value in vertices_count.items():
# # #     if value == 1:
# # #         endpoint_idx.append(key)

# # # end_points_2 = skeleton_points_1[endpoint_idx]  
# # # tree = KDTree(end_points)
# # # distances, indices = tree.query(end_points_2, k=1)
# # # near_end_points_2 = end_points[indices]

# # # for line in connected_lines:
# # #     for idx, end_idx in enumerate(endpoint_idx):
# # #         mid_point = (near_end_points_2[idx] + skeleton_points_1[end_idx]) / 2
# # #         if mesh_bound.contains(np.array([near_end_points_2[idx]]))[0] == True and mesh_bound.contains(np.array([mid_point]))[0] == True:
# # #             if end_idx == line[0]:
# # #                 new_idx = skeleton_points_1.shape[0]
# # #                 skeleton_points_1 = np.vstack([skeleton_points_1, near_end_points_2[idx]])
# # #                 line.insert(0, new_idx)
# # #                 break
# # #             elif end_idx == line[-1]:
# # #                 new_idx = skeleton_points_1.shape[0]
# # #                 skeleton_points_1 = np.vstack([skeleton_points_1, near_end_points_2[idx]])
# # #                 line.append(new_idx)
# # #                 break

# # # for line in connected_lines:
# # #     for i in range(len(line) - 2):
# # #         point_1 = skeleton_points_1[line[i]]
# # #         point_2 = skeleton_points_1[line[i+1]]
# # #         point_3 = skeleton_points_1[line[i+2]]
# # #         angle = angle_between_lines(point_1, point_2, point_3)
# # #         if angle < 100:
# # #             new_mid_point = find_midpoint_perpendicular_3d(point_1, point_2, point_3)
# # #             skeleton_points_1[line[i+1]] = new_mid_point

# # # loop_count = 0
# # # outside_idx = []
# # # inside_idx = []

# # # while loop_count < 20:
# # #     loop_count += 1
# # #     tree = KDTree(skeleton_points_2)

# # #     distances, indices = tree.query(skeleton_points_1, k=1)
# # #     new_skeleton_points_1 = skeleton_points_2[indices]

# # #     for index, point in enumerate(skeleton_points_1):
# # #         if indices[index] in inside_idx:
# # #             skeleton_points_1[index] = new_skeleton_points_1[index]
# # #         else:
# # #             if indices[index] not in outside_idx:
# # #                 is_bound = mesh_bound.contains([new_skeleton_points_1[index]])[0]

# # #                 if is_bound:
# # #                     skeleton_points_1[index] = new_skeleton_points_1[index]
# # #                     inside_idx.append(indices[index])
# # #                 else:
# # #                     outside_idx.append(indices[index])

# # #     for line in connected_lines:
# # #         for i in range(len(line) - 2):
# # #             point_1 = skeleton_points_1[line[i]]
# # #             point_2 = skeleton_points_1[line[i+1]]
# # #             point_3 = skeleton_points_1[line[i+2]]
# # #             angle = angle_between_lines(point_1, point_2, point_3)
# # #             if angle < 100:
# # #                 new_mid_point = (point_1 + point_3)/2
# # #                 skeleton_points_1[line[i+1]] = new_mid_point

# # #     new_connected_lines = []

# # #     for line in connected_lines:
# # #         new_line = []

# # #         new_line.append(line[0])
# # #         for i in range(len(line)-1):
# # #             point_1 = skeleton_points_1[line[i]]
# # #             point_2 = skeleton_points_1[line[i+1]]
# # #             distance = euclidean_distance(point_1, point_2)
# # #             new_num_points = int(distance/resolution) - 2
            
# # #             if new_num_points > 0:
# # #                 interpolated_points = interpolate_points(point_1, point_2, new_num_points)
# # #                 new_indices = []
# # #                 for point in interpolated_points:
# # #                     new_indices.append(skeleton_points_1.shape[0])
# # #                     skeleton_points_1 = np.vstack([skeleton_points_1, point])
                
# # #                 new_line += new_indices
# # #             new_line.append(line[i+1])
# # #         new_connected_lines.append(new_line)

# # #     connected_lines = new_connected_lines


# Create a VTK PolyData object representing the surface
vertices, faces, normals, values = measure.marching_cubes(processed_mask, level=0.5, spacing=voxel_sizes)
surface_data = vtk.vtkPolyData()
vmtk_points = vtk.vtkPoints()
for vertex in vertices:
    vmtk_points.InsertNextPoint(vertex)
surface_data.SetPoints(vmtk_points)

# Add polygons to the surface
polygons = vtk.vtkCellArray()
for face in faces:
    polygon = vtk.vtkPolygon()
    for index in face:
        polygon.GetPointIds().InsertNextId(index)
    polygons.InsertNextCell(polygon)
surface_data.SetPolys(polygons)

mySmoother = vmtkscripts.vmtkSurfaceSmoothing()
mySmoother.Surface = surface_data
mySmoother.PassBand = 0.1
mySmoother.NumberOfIterations = 5000
mySmoother.Execute()
smoothed_surface = mySmoother.Surface

# Get vertices and faces from the surface
vertices = []
faces = []

vmtk_points = smoothed_surface.GetPoints()
for i in range(vmtk_points.GetNumberOfPoints()):
    vmtk_point = vmtk_points.GetPoint(i)
    vertices.append(vmtk_point)

cells = smoothed_surface.GetPolys()
cells.InitTraversal()
while True:
    cell = vtk.vtkIdList()
    if cells.GetNextCell(cell) == 0:
        break
    face = [cell.GetId(j) for j in range(cell.GetNumberOfIds())]
    faces.append(face)

vertices = np.array(vertices)
faces = np.array(faces)

new_splitted_line = []
point_clusters = {}

for line in smooth_connected_lines:
    line_points = points[line]

    total_length = 0
    point_indexes = []

    for i in range(len(line_points) - 1):
        cur_length = euclidean_distance(line_points[i], line_points[i+1])
        total_length += cur_length

        if (total_length > 2) or (i == len(line_points) - 2):
            point_indexes.append(line[i])
            point_indexes.append(line[i+1])

            point_clusters[line[i]] = len(new_splitted_line)
            point_clusters[line[i+1]] = len(new_splitted_line)

            new_splitted_line.append(point_indexes)
            total_length = 0
            point_indexes = []
        else:
            point_clusters[line[i]] = len(new_splitted_line)
            point_indexes.append(line[i])
            

colors = []
for _ in range(len(new_splitted_line)):
    # Generate random RGB values
    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    colors.append(color)


mesh_copy = tm.Trimesh(vertices=vertices, faces=faces, enable_post_processing=True, solid=True)
mesh_sub = sk.pre.contract(mesh_copy, epsilon=0.07)
mesh_sub = sk.pre.fix_mesh(mesh_sub, remove_disconnected=5, inplace=False)

# skeleton_points_1 = skeleton_points_1
skeleton_points_2 = mesh_sub.vertices

tree = KDTree(points)
distances, indices = tree.query(vertices, k=1)

vertices_colors = []
for index, vertex in enumerate(vertices):
    if indices[index] in point_clusters:
        color = colors[point_clusters[indices[index]]]
    else:
        color = [0, 0, 0]

    vertices_colors.append(color)


vertex_normals = mesh_copy.vertex_normals
print(vertex_normals)
# # # Find connection
# # tree = KDTree(skeleton_points)
# # distances, indices = tree.query(points, k=3)

# # clusters = {}
# # clusters_start_point = {}
# # clusters_min_dist = {}

# # for index, close_points in enumerate(indices):
# #     for point_index, point in enumerate(close_points):
# #         if point in junction_points:
# #             continue
# #         else:
# #             if point not in clusters:
# #                 clusters[point] = []
# #                 clusters_start_point[point] = None
# #                 clusters_min_dist[point] = 0
            
# #             clusters[point].append(index)

# #             if clusters_start_point[point] is None:
# #                 clusters_start_point[point] = index
# #                 clusters_min_dist[point] = distances[index][point_index]
# #             else:
# #                 if distances[index][point_index] < clusters_min_dist[point]:
# #                     clusters_start_point[point] = index
# #                     clusters_min_dist[point] = distances[index][point_index]
# #             break

# # arrange_lines = {}
# # for cluster_name, cluster_points in clusters.items():
# #     arrange_line = []
# #     arrange_line.append(clusters_start_point[cluster_name])
# #     reduced_points = cluster_points.copy()
# #     reduced_points.remove(clusters_start_point[cluster_name])

# #     while len(arrange_line) < len(cluster_points):
# #         start_point = points[arrange_line[0]]
# #         end_point = points[arrange_line[-1]]

# #         tree = KDTree(points[reduced_points])

# #         distances_1, indices_1 = tree.query(start_point)
# #         distances_2, indices_2 = tree.query(end_point)


# #         if distances_1 < distances_2:
# #             reduce_point = reduced_points[indices_1]
# #             arrange_line.insert(0, reduce_point)
# #         else:
# #             reduce_point = reduced_points[indices_2]
# #             arrange_line.append(reduce_point)

# #         reduced_points.remove(reduce_point)

# #     arrange_lines[cluster_name] = arrange_line

# # smooth_connected_lines = []

# # for line in connected_lines:
# #     smooth_line = []

# #     for point in line:
# #         if point not in junction_points:
# #             if len(smooth_line) == 0:
# #                 smooth_line += arrange_lines[point]
# #             else:
# #                 start_point = points[smooth_line[0]]
# #                 end_point = points[smooth_line[-1]]
# #                 tree = KDTree(points[[arrange_lines[point][-1], arrange_lines[point][0]]])

# #                 distances_1, indices_1 = tree.query(start_point)
# #                 distances_2, indices_2 = tree.query(end_point)

# #                 if distances_1 < distances_2:
# #                     if indices_1 == 0:
# #                         smooth_line = arrange_lines[point][::-1] + smooth_line
# #                     else:
# #                         smooth_line = arrange_lines[point] + smooth_line
# #                 else:
# #                     if indices_2 == 0:
# #                         smooth_line = smooth_line + arrange_lines[point][::-1]
# #                     else:
# #                         smooth_line = smooth_line + arrange_lines[point]

# #     smooth_connected_lines.append(smooth_line)

# # for point in junction_points:
# #     center_points = []
# #     pos = []
# #     for index, line in enumerate(connected_lines):
# #         if line[0] == point or line[-1] == point:
# #             distance_1 = euclidean_distance(skeleton_points[point], points[smooth_connected_lines[index][0]])
# #             distance_2 = euclidean_distance(skeleton_points[point], points[smooth_connected_lines[index][-1]])
# #             if distance_1 < distance_2:
# #                 center_points.append(smooth_connected_lines[index][0])
# #                 pos.append(0)
# #             else:
# #                 center_points.append(smooth_connected_lines[index][-1])
# #                 pos.append(-1)

# #     avg_point = np.mean(points[center_points], axis=0)
# #     new_index = points.shape[0]
# #     points = np.vstack([points, avg_point])

# #     for index, line in enumerate(connected_lines):
# #         if line[0] == point or line[-1] == point:
# #             if pos[index] == 0:
# #                 smooth_connected_lines[index].insert(0, new_index)
# #             else:
# #                 smooth_connected_lines[index].append(new_index)

# # # tree = KDTree(vmtk_vertices)
# # # distances, indices = tree.query(vertices, k=1)
# # # diameter = np.mean(distances)
# # # print("Avg distance to centerline (all): ", diameter)

# # # percentile_25 = np.percentile(distances, 25)
# # # percentile_50 = np.percentile(distances, 50)
# # # percentile_75 = np.percentile(distances, 75)
# # # selected_range = distances[(distances >= percentile_25) & (distances <= percentile_75)]
# # # diameter = np.mean(selected_range)

# # # print("0th: ", np.min(distances))
# # # print("25th: ", percentile_25)
# # # print("50th: ", percentile_50)
# # # print("75th: ", percentile_75)
# # # print("100th: ", np.max(distances))
# # # print("Avg distance to centerline (range 25th-75th): ", diameter)

# # # line_traces = []
# # # skeleton_points = np.argwhere(skeleton > 0)*voxel_sizes
# # # raw_skeleton_points = np.argwhere((centerline_data > 0) & (segment_data == 2))*voxel_sizes
# # # visualized_skeleton_points = generate_points(skeleton_points, 3, 'black')
# # # visualized_raw_skeleton_points = generate_points(raw_skeleton_points, 3, 'red')
# # # visualized_thin_points = generate_points(vmtk_vertices, 2, 'green')
# # # visualized_boundary_points = generate_points(vertices, 1, 'blue')
# # # show_figure([
# # #             visualized_boundary_points, 
# # #             visualized_thin_points,
# # #             visualized_skeleton_points,
# # #             visualized_raw_skeleton_points
# # #         ] 
# # # )

# # # colorscale = [
# # #     [0.0, 'rgb(0, 0, 128)'],    # Dark Blue
# # #     [0.2, 'rgb(0, 0, 255)'],    # Blue
# # #     [0.4, 'rgb(0, 128, 0)'],    # Green
# # #     [0.6, 'rgb(255, 255, 0)'],  # Yellow
# # #     [0.8, 'rgb(255, 0, 0)'],    # Red
# # #     [1.0, 'rgb(128, 0, 0)']     # Dark Red
# # # ]


# # # mesh = go.Mesh3d(
# # #     x=vertices[:, 0],
# # #     y=vertices[:, 1],
# # #     z=vertices[:, 2],
# # #     i=faces[:, 0],
# # #     j=faces[:, 1],
# # #     k=faces[:, 2],
# # #     intensity=distances,
# # #     colorscale='hot',
# # #     # intensity=distances,
# # #     # colorscale='plasma',
# # #     colorbar=dict(title='Distance to centerline (mm)', tickvals=[np.min(distances), np.mean(distances), np.max(distances)]),
# # #     hoverinfo='text',
# # #     text=distances
# # # )

# # # # Create the figure
# # # fig = go.Figure(data=[mesh])

# # # # Update layout
# # # fig.update_layout(scene=dict(
# # #                     xaxis_title='X',
# # #                     yaxis_title='Y',
# # #                     zaxis_title='Z',
# # #                     ),
# # #                     title='Mesh Surface Color Map'
# # #                 )

# # # # Show the plot
# # # fig.show()

# # # # Calculate histogram
# # # hist, bins = np.histogram(distances, bins=np.arange(min(distances), max(distances) + 2))

# # # # Plot histogram
# # # plt.bar(bins[:-1], hist, width=1, align='edge')
# # # plt.xlabel('Distance')
# # # plt.ylabel('Frequency')
# # # plt.title('Histogram of Distances')
# # # plt.grid(True)
# # # plt.show()


# Create a trace for the cone
vertex_normals = -1*vertex_normals
cone_trace = go.Cone(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        u=vertex_normals[:, 0],
        v=vertex_normals[:, 1],
        w=vertex_normals[:, 2],
        sizeref=2,
        name='Vectors'
)


visualized_boundary_points = generate_points(vertices, 1, vertices_colors)
visualized_skeleton_points = generate_points(skeleton_points_2, 1)
show_figure([
            visualized_skeleton_points,
            # visualized_vmtk_points,
            visualized_boundary_points,
            cone_trace
        ] + line_traces
)