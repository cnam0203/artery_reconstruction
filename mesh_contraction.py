import skeletor as sk
import nibabel as nib
import numpy as np

from skimage.morphology import skeletonize, thin
from skimage import measure

from scipy.spatial import KDTree, distance_matrix
from sklearn.cluster import DBSCAN, KMeans

import matplotlib.pyplot as plt

import math
import time
import os
import json
import heapq
import copy
import trimesh as tm

from preprocess_data import *
from process_graph import *
from visualize_graph import *
from slice_selection import *
from visualize_mesh import *

import plotly.graph_objs as go
from skimage import morphology
from scipy.ndimage import distance_transform_edt
import open3d as o3d

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
    
    # Compute dot product of the direction vectors
    dot_product = np.dot(vector1, vector2)
    
    # Calculate magnitudes of the direction vectors
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)
    
    # Calculate angle between the lines using dot product formula
    cos_theta = dot_product / (magnitude_vector1 * magnitude_vector2)
    angle_radians = np.arccos(cos_theta)
    
    # Convert angle from radians to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def find_projection_point(p1, p2, p3):
    l2 = np.sum((p1-p2)**2)
    if l2 == 0:
        return np.array([0, 0, 0])


    #The line extending the segment is parameterized as p1 + t (p2 - p1).
    #The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2

    #if you need the point to project on line extention connecting p1 and p2
    t = np.sum((p3 - p1) * (p2 - p1)) / l2

    #if you need to ignore if p3 does not project onto line segment
    if t > 1 or t < 0:
        return np.array([0, 0, 0])

    #if you need the point to project on line segment between p1 and p2 or closest point of the line segment
    t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2))

    projection = p1 + t * (p2 - p1)
    return projection

def find_midpoint_perpendicular_3d(p1, p2, p3):
    projection = find_projection_point(p1, p3, p2)
    midpoint = (np.array(p2) + projection) / 2
    print(p1, p2, p3, midpoint)
    return midpoint

def interpolate_points(p1, p2, num_points):
    # Create array of linearly spaced values between 0 and 1
    t_values = np.linspace(0, 1, num_points + 2)[1:-1]  # Exclude endpoints
    
    # Compute intermediate points using linear interpolation
    interpolated_points = [p1 + t * (p2 - p1) for t in t_values]
    
    return interpolated_points

dataset_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/'
segment_file_path = dataset_dir + 'sub-4947_TOF_multiclass_segmentation.nii.gz'
original_file_path = dataset_dir + 'sub-4947_run-1_mra_TOF.nii.gz'
segment_image = nib.load(segment_file_path)
original_image = nib.load(original_file_path)

intensity_threshold_1 = 0.1
intensity_threshold_2 = 0.1
gaussian_sigma=2
distance_threshold=20
laplacian_iter = 5
neighbor_threshold_1 = 5
neighbor_threshold_2 = neighbor_threshold_1 + 10

#Find skeleton
cex_data, voxel_sizes, vertices, faces = reconstruct_surface(
                segment_image, 
                original_image, 
                index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 
                intensity_threshold_1=intensity_threshold_1, 
                intensity_threshold_2=intensity_threshold_2, 
                gaussian_sigma=gaussian_sigma, 
                distance_threshold=distance_threshold,
                laplacian_iter=laplacian_iter,
                folder_path='',
                neighbor_threshold_1=neighbor_threshold_1,
                neighbor_threshold_2=neighbor_threshold_2
            )

skeleton = skeletonize(cex_data)
skeleton_points_1, end_points, junction_points, connected_lines = find_graphs(skeleton)
skeleton_points_1 = voxel_sizes*(skeleton_points_1 + 0.5)
old_skeleton_points_1 = np.copy(skeleton_points_1)
old_connected_lines = copy.deepcopy(connected_lines)

# for line in connected_lines:
#     for i in range(len(line) - 2):
#         point_1 = skeleton_points_1[line[i]]
#         point_2 = skeleton_points_1[line[i+1]]
#         point_3 = skeleton_points_1[line[i+2]]
#         angle = angle_between_lines(point_1, point_2, point_3)
#         if angle < 100:
#             new_mid_point = find_midpoint_perpendicular_3d(point_1, point_2, point_3)
#             skeleton_points_1[line[i+1]] = new_mid_point

resolution = 0.1
new_connected_lines = []

for line in connected_lines:
    new_line = []

    new_line.append(line[0])
    for i in range(len(line)-1):
        point_1 = skeleton_points_1[line[i]]
        point_2 = skeleton_points_1[line[i+1]]
        distance = euclidean_distance(point_1, point_2)
        new_num_points = int(distance/resolution) - 2
        
        if new_num_points > 0:
            print(new_num_points)
            interpolated_points = interpolate_points(point_1, point_2, new_num_points)
            new_indices = []
            for point in interpolated_points:
                new_indices.append(skeleton_points_1.shape[0])
                skeleton_points_1 = np.vstack([skeleton_points_1, point])
            
            new_line += new_indices
        new_line.append(line[i+1])
    new_connected_lines.append(new_line)

connected_lines = new_connected_lines

# mesh_copy = tm.load_mesh("C:/Users/nguc4116/Desktop/output_mesh_smooth.stl", enable_post_processing=True, solid=True) 
# mesh_sub = sk.pre.contract(mesh_copy, epsilon=0.9)
# mesh_sub = sk.pre.fix_mesh(mesh_sub, remove_disconnected=5, inplace=False)

# skeleton_points_1 = skeleton_points_1
# skeleton_points_2 = mesh_sub.vertices

# loop_count = 0
# while loop_count < 20:
#     loop_count += 1
#     tree = KDTree(skeleton_points_2)

#     distances, indices = tree.query(skeleton_points_1, k=1)
#     skeleton_points_1 = skeleton_points_2[indices]

#     new_connected_lines = []

#     for line in connected_lines:
#         new_line = []

#         new_line.append(line[0])
#         for i in range(len(line)-1):
#             point_1 = skeleton_points_1[line[i]]
#             point_2 = skeleton_points_1[line[i+1]]
#             distance = euclidean_distance(point_1, point_2)
#             new_num_points = int(distance/resolution) - 2
            
#             if new_num_points > 0:
#                 interpolated_points = interpolate_points(point_1, point_2, new_num_points)
#                 new_indices = []
#                 for point in interpolated_points:
#                     new_indices.append(skeleton_points_1.shape[0])
#                     skeleton_points_1 = np.vstack([skeleton_points_1, point])
                
#                 new_line += new_indices
#             new_line.append(line[i+1])
#         new_connected_lines.append(new_line)

#     connected_lines = new_connected_lines

# distances, indices = tree.query(skeleton_points_1, k=1)
# skeleton_points_1 = skeleton_points_2[indices]

    # for line in connected_lines:
    #     for i in range(len(line) - 2):
    #         point_1 = skeleton_points_1[line[i]]
    #         point_2 = skeleton_points_1[line[i+1]]
    #         point_3 = skeleton_points_1[line[i+2]]
    #         angle = angle_between_lines(point_1, point_2, point_3)
    #         if angle < 100:
    #             new_mid_point = find_midpoint_perpendicular_3d(point_1, point_2, point_3)
    #             skeleton_points_1[line[i+1]] = new_mid_point
    #             i += 1

line_traces = []
# visualized_skeleton_points = generate_points(mesh_sub.vertices, 1, 'red')
visualized_skeleton_points_1 = generate_points(old_skeleton_points_1, 3, 'red')
# visualized_thin_points = generate_points(skeleton_points_1, 3, 'green')
visualized_boundary_points = generate_points(vertices, 1, 'blue')

# for line in old_connected_lines:
#     for i in range(len(line) - 1):
#         line_traces.append(generate_lines(np.array([old_skeleton_points_1[line[i]], old_skeleton_points_1[line[i+1]]]), 2, 'red'))

show_figure([
            visualized_boundary_points, 
            # visualized_skeleton_points, 
            # visualized_thin_points,
            visualized_skeleton_points_1
            # visualized_end_points, 
            # visualized_junction_points,
            # visualized_artery_points,
        ] 
            + line_traces
)
# scene = tm.scene.scene.Scene()
# scene.add_geometry(mesh)
# scene.export('C:/Users/nguc4116/Desktop/' + 'sub_mesh_1.stl')# mesh = tm.load_mesh('mesh.obj')


# vmtksurfacereader -ifile "C:/Users/nguc4116/Desktop/output_mesh.stl" --pipe vmtksurfacesmoothing -iterations 2500 -passband 0.0001 --pipe vmtkrenderer --pipe vmtksurfaceviewer -display 1