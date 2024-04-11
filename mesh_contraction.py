import skeletor as sk
import nibabel as nib
import numpy as np

from skimage.morphology import skeletonize, thin
from skimage import measure

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

from preprocess_data import *
from process_graph import *
from visualize_graph import *
from slice_selection import *
from visualize_mesh import *

import plotly.graph_objs as go
from skimage import morphology
from scipy.ndimage import distance_transform_edt

from vmtk import pypes
from vmtk import vmtkscripts
import vtk

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
    angle_degrees = np.degrees(angle_radians)
    
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
    projection = find_projection_point(p1, p3, p2)
    midpoint = (np.array(p2) + projection) / 2
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

# segment_file_path = dataset_dir + 'sub-2983_TOF_multiclass_segmentation.nii.gz'
# original_file_path = dataset_dir + 'sub-2983_run-1_mra_TOF.nii.gz'

segment_image = nib.load(segment_file_path)
original_image = nib.load(original_file_path)

intensity_threshold_1 = 0.1
intensity_threshold_2 = 0.1
gaussian_sigma=2
distance_threshold=20
laplacian_iter = 5
neighbor_threshold_1 = 5
neighbor_threshold_2 = neighbor_threshold_1 + 10
resolution = 0.05

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
mySmoother.NumberOfIterations = 500
mySmoother.Execute()
smoothed_surface = mySmoother.Surface

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
# mesh_copy = tm.load_mesh("C:/Users/nguc4116/Desktop/output_mesh_smooth.stl", enable_post_processing=True, solid=True) 
mesh_copy = tm.Trimesh(vertices=vmtk_vertices, faces=vmtk_faces, enable_post_processing=True, solid=True)
mesh_sub = sk.pre.contract(mesh_copy, epsilon=0.1)
mesh_sub = sk.pre.fix_mesh(mesh_sub, remove_disconnected=5, inplace=False)

skeleton_points_1 = skeleton_points_1
skeleton_points_2 = mesh_sub.vertices

skeleton_3 = sk.skeletonize.by_wavefront(mesh_sub, waves=1)
skeleton_3 = sk.post.clean_up(skeleton_3)
skeleton_3 = sk.post.smooth(skeleton_3)
edges_3 = skeleton_3.edges

vertices_count = {}
endpoint_idx = []
for edge in edges_3:
    vertice_1 = edge[0]
    vertice_2 = edge[1]

    if vertice_1 not in vertices_count:
        vertices_count[vertice_1] = 0
    if vertice_2 not in vertices_count:
        vertices_count[vertice_2] = 0

    vertices_count[vertice_1] += 1
    vertices_count[vertice_2] += 1

for key, value in vertices_count.items():
    if value == 1:
        endpoint_idx.append(key)

end_points = skeleton_3.vertices[endpoint_idx]

vertices_count = {}
endpoint_idx = []
for line in connected_lines:
    vertice_1 = line[0]
    vertice_2 = line[-1]

    if vertice_1 not in vertices_count:
        vertices_count[vertice_1] = 0
    if vertice_2 not in vertices_count:
        vertices_count[vertice_2] = 0

    vertices_count[vertice_1] += 1
    vertices_count[vertice_2] += 1

for key, value in vertices_count.items():
    if value == 1:
        endpoint_idx.append(key)

end_points_2 = skeleton_points_1[endpoint_idx]  
tree = KDTree(end_points)
distances, indices = tree.query(end_points_2, k=1)
near_end_points_2 = end_points[indices]

for line in connected_lines:
    for idx, end_idx in enumerate(endpoint_idx):
        if end_idx == line[0]:
            new_idx = skeleton_points_1.shape[0]
            skeleton_points_1 = np.vstack([skeleton_points_1, near_end_points_2[idx]])
            line.insert(0, new_idx)
            break
        elif end_idx == line[-1]:
            new_idx = skeleton_points_1.shape[0]
            skeleton_points_1 = np.vstack([skeleton_points_1, near_end_points_2[idx]])
            line.append(new_idx)
            break

for line in connected_lines:
    for i in range(len(line) - 2):
        point_1 = skeleton_points_1[line[i]]
        point_2 = skeleton_points_1[line[i+1]]
        point_3 = skeleton_points_1[line[i+2]]
        angle = angle_between_lines(point_1, point_2, point_3)
        if angle < 100:
            new_mid_point = find_midpoint_perpendicular_3d(point_1, point_2, point_3)
            skeleton_points_1[line[i+1]] = new_mid_point

loop_count = 0
while loop_count < 20:
    loop_count += 1
    tree = KDTree(skeleton_points_2)

    distances, indices = tree.query(skeleton_points_1, k=1)
    skeleton_points_1 = skeleton_points_2[indices]

    for line in connected_lines:
        for i in range(len(line) - 2):
            point_1 = skeleton_points_1[line[i]]
            point_2 = skeleton_points_1[line[i+1]]
            point_3 = skeleton_points_1[line[i+2]]
            angle = angle_between_lines(point_1, point_2, point_3)
            if angle < 100:
                new_mid_point = (point_1 + point_3)/2
                skeleton_points_1[line[i+1]] = new_mid_point

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
                interpolated_points = interpolate_points(point_1, point_2, new_num_points)
                new_indices = []
                for point in interpolated_points:
                    new_indices.append(skeleton_points_1.shape[0])
                    skeleton_points_1 = np.vstack([skeleton_points_1, point])
                
                new_line += new_indices
            new_line.append(line[i+1])
        new_connected_lines.append(new_line)

    connected_lines = new_connected_lines

tree = KDTree(skeleton_points_1)
distances, indices = tree.query(vertices, k=1)

# line_traces = []
# # visualized_skeleton_points = generate_points(mesh_sub.vertices, 1, 'red')
# visualized_skeleton_points_1 = generate_points(old_skeleton_points_1, 3, 'black')
# visualized_thin_points = generate_points(skeleton_points_1, 3, 'green')
# visualized_boundary_points = generate_points(vertices, 1, 'blue')

# for line in connected_lines:
#     for i in range(len(line) - 1):
#         line_traces.append(generate_lines(np.array([skeleton_points_1[line[i]], skeleton_points_1[line[i+1]]]), 2, 'green'))

# show_figure([
#             visualized_boundary_points, 
#             # visualized_skeleton_points, 
#             # visualized_thin_points,
#             # visualized_skeleton_points_1,
#         ] 
#             + line_traces
# )

# Calculate colors based on values using the "plasma" colormap
# Define the trace for the mesh
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
mySmoother.NumberOfIterations = 5
mySmoother.Execute()
smoothed_surface = mySmoother.Surface

# Get vertices and faces from the surface
vertices = []
faces = []

points = smoothed_surface.GetPoints()
for i in range(points.GetNumberOfPoints()):
    point = points.GetPoint(i)
    vertices.append(point)

cells = smoothed_surface.GetPolys()
cells.InitTraversal()
while True:
    cell = vtk.vtkIdList()
    if cells.GetNextCell(cell) == 0:
        break
    face = [cell.GetId(j) for j in range(cell.GetNumberOfIds())]
    faces.append(face)

tree = KDTree(skeleton_points_1)
distances, indices = tree.query(vertices, k=1)

mesh = go.Mesh3d(
    x=vertices[:, 0],
    y=vertices[:, 1],
    z=vertices[:, 2],
    i=faces[:, 0],
    j=faces[:, 1],
    k=faces[:, 2],
    intensity=-1*distances,
    colorscale='plasma',
    colorbar=dict(title='Thickness (mm)', tickvals=[np.min(distances), np.max(distances)]),
)

# Create the figure
fig = go.Figure(data=[mesh])

# Update layout
fig.update_layout(scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    ),
                    title='Mesh Surface Color Map'
                )

# Show the plot
fig.show()