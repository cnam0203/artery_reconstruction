import skeletor as sk
import nibabel as nib
import numpy as np

from collections import deque
from skimage.morphology import skeletonize, thin
from skimage import measure
from scipy.ndimage import binary_dilation, center_of_mass

from scipy.spatial import KDTree, distance_matrix
from sklearn.cluster import DBSCAN, KMeans

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import pymeshlab

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
# from marcube import *
from artery_ica import *
from ray_intersection import *

# import plotly.graph_objs as go
from skimage import morphology
from scipy.ndimage import distance_transform_edt

from vmtk import pypes
from vmtk import vmtkscripts
import vtk

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

def check_and_create_directory(dir_path):
    # Check if the directory exists
    if not os.path.exists(dir_path):
        # Create the directory
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")

def split_smooth_lines(smooth_connected_lines, points, distance_threshold=2):
    new_splitted_lines = []
    splitted_branches = []
    point_clusters = {}

    for idx, line in enumerate(smooth_connected_lines):
        line_points = points[line]

        total_length = 0
        point_indexes = []

        for i in range(len(line_points) - 1):
            cur_length = euclidean_distance(line_points[i], line_points[i+1])
            total_length += cur_length

            if (total_length > distance_threshold) or (i == len(line_points) - 2):
                point_indexes.append(line[i])
                point_indexes.append(line[i+1])

                point_clusters[line[i]] = len(new_splitted_lines)
                point_clusters[line[i+1]] = len(new_splitted_lines)

                new_splitted_lines.append(point_indexes)
                splitted_branches.append(idx)
                total_length = 0
                point_indexes = []
            else:
                point_clusters[line[i]] = len(new_splitted_lines)
                point_indexes.append(line[i])
                

    colors = []
    for _ in range(len(new_splitted_lines)):
        # Generate random RGB values
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        colors.append(color)

    return new_splitted_lines, splitted_branches, point_clusters, colors
    
def smooth_centerline(vmtk_vertices, skeleton_points_1, new_connected_lines):
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

    return points, smooth_connected_lines

def vmtk_smooth_mesh(vertices, faces, num_iteration=2000, division_surface=2):
    print('Before: ')
    print(vertices.shape[0])
    print(faces.shape[0])

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
    mySmoother.NumberOfIterations = num_iteration
    mySmoother.Execute()
    smoothed_surface = mySmoother.Surface

    if division_surface > 1:
        print('Divide')
        myMeshGenerator = vmtkscripts.vmtkSurfaceSubdivision()
        myMeshGenerator.Surface = smoothed_surface
        myMeshGenerator.NumberOfSubdivisions = division_surface
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
    return vmtk_vertices, vmtk_faces

def vmtk_decimate_mesh(vertices, faces, target_faces=7000):
    print('Before: ')
    print(vertices.shape[0])
    print(faces.shape[0])

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

    # myDecimate = vmtkscripts.vmtkSurfaceDecimation()
    # myDecimate.Surface = surface_data
    # myDecimate.TargetReduction = 1 - target_faces/faces.shape[0]
    # myDecimate.Execute()
    # decimated_surface = myDecimate.Surface

    myMeshGenerator = vmtkscripts.vmtkSurfaceRemeshing()
    myMeshGenerator.Surface = surface_data
    myMeshGenerator.ElementSizeMode = "edgelength"
    myMeshGenerator.TargetEdgeLength = 0.8
    myMeshGenerator.Execute()
    decimated_surface = myMeshGenerator.Surface	

    # Get vertices and faces from the surface
    vmtk_vertices = []
    vmtk_faces = []

    points = decimated_surface.GetPoints()
    for i in range(points.GetNumberOfPoints()):
        point = points.GetPoint(i)
        vmtk_vertices.append(point)

    cells = decimated_surface.GetPolys()
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

    print('After: ')
    print(vmtk_vertices.shape[0])
    print(vmtk_faces.shape[0])
    return vmtk_vertices, vmtk_faces

    
def extend_skeleton(new_connected_lines, end_points, skeleton_points_1, original_data, processed_mask, skeleton, voxel_sizes):
    for index, line in enumerate(new_connected_lines):
        new_points = [[], []]
        if line[0] in end_points:
            if len(line) <= 7:
                consult_points = skeleton_points_1[line]
            else:
                consult_points = skeleton_points_1[line[:8]]

            point1 = consult_points[0]
            point2 = consult_points[-1]
            direction_vectors = np.abs(np.array(point2) - np.array(point1))*voxel_sizes
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
                        if intensity > max_intensity and segment_slice[pos[0], pos[1]] == 1:
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

                    new_points[0].append([int(cur_point[0]), int(cur_point[1]), int(cur_point[2])])
                    new_points[0].append([int(new_centerpoint[0]), int(new_centerpoint[1]), int(new_centerpoint[2])])
                    # visualize_slice(intensity_slice, segment_slice, segment_slice, [max_intensity_position], point1[axis], new_centerpoint[axis] - point1[axis], axis)


        if line[-1] in end_points:
            if len(line) <= 7:
                consult_points = skeleton_points_1[line]
            else:
                consult_points = skeleton_points_1[line[-7:]]

            point1 = consult_points[-1]
            point2 = consult_points[0]
            direction_vectors = np.abs(np.array(point2) - np.array(point1))*voxel_sizes
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
                        if intensity > max_intensity and segment_slice[pos[0], pos[1]] == 1:
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

                    new_points[1].append([int(cur_point[0]), int(cur_point[1]), int(cur_point[2])])
                    new_points[1].append([int(new_centerpoint[0]), int(new_centerpoint[1]), int(new_centerpoint[2])])
                    # visualize_slice(intensity_slice, segment_slice, segment_slice, [max_intensity_position], point1[axis], new_centerpoint[axis] - point1[axis], axis)

        
        indices = []
        for point in new_points[0]:
            cur_index = skeleton_points_1.shape[0]
            indices.append(cur_index)
            skeleton_points_1 = np.vstack([skeleton_points_1, np.array([point])])

        reversed_line = indices[::-1]
        new_connected_lines[index] = reversed_line + new_connected_lines[index]

        indices = []
        for point in new_points[1]:
            cur_index = skeleton_points_1.shape[0]
            indices.append(cur_index)
            skeleton_points_1 = np.vstack([skeleton_points_1,  np.array([point])])

        new_connected_lines[index] = new_connected_lines[index] + indices

    return new_connected_lines, skeleton, skeleton_points_1

def remove_short_branch(skeleton, skeleton_points, end_points, junction_points, connected_lines, len_threshold=10):
    
    is_first = 0
    count = 0

    while is_first == 0 or count > 0:
        is_first += 1
        count = 0
        new_connected_lines = []

        # print('1. ', connected_lines, end_points, junction_points)

        for line in connected_lines:
            if len(line) <= len_threshold and ((line[0] in end_points) or (line[-1] in end_points)):
                count += 1
                for point in line:
                    if point not in junction_points:
                        pos = skeleton_points[point].astype(int)
                        skeleton[pos[0]][pos[1]][pos[2]] = 0
            else:
                new_connected_lines.append(line)

        # print('2. ', new_connected_lines)
        end_points = []
        junction_points = []

        is_end = False
        
        while not is_end:
            is_end = True
            point_count = {}

            for idx, line in enumerate(new_connected_lines):
                if line[0] not in point_count:
                    point_count[line[0]] = []
                if line[-1] not in point_count:
                    point_count[line[-1]] = []
                
                point_count[line[0]].append(idx)
                point_count[line[-1]].append(idx)

            merge_point = None
            
            for idx in point_count:
                if len(point_count[idx]) == 2:
                    line_idx_1 = point_count[idx][0]
                    line_idx_2 = point_count[idx][1]
                    line_1 = new_connected_lines[line_idx_1]
                    line_2 = new_connected_lines[line_idx_2]

                    new_line = []
                    new_lines = []
                    reversed_line_2 = line_2[::-1]

                    if line_1[0] == line_2[0] and line_1[-1] != line_2[-1]:
                        new_line = reversed_line_2[:-1] + line_1
                        is_end = False
                    elif line_1[0] == line_2[-1] and line_1[-1] != line_2[0]:
                        new_line = line_2[:-1] + line_1
                        is_end = False
                    elif line_1[-1] == line_2[0] and line_1[0] != line_2[-1]:
                        new_line = line_1[:-1] + line_2
                        is_end = False
                    elif line_1[-1] == line_2[-1] and line_1[0] != line_2[0]:
                        new_line = line_1[:-1] + reversed_line_2
                        is_end = False

                    if not is_end:
                        for idx_p, line in enumerate(new_connected_lines):
                            if idx_p not in point_count[idx]:
                                new_lines.append(line)
                        
                        # print('Avant:', len(new_connected_lines))
                        # print(new_connected_lines)
                        new_lines.append(new_line)
                        new_connected_lines = new_lines
                        # print('Après:', len(new_connected_lines))
                        # print(new_connected_lines)
                        break
            
            end_points = []
            junction_points = []
            point_count = {}

            for idx, line in enumerate(new_connected_lines):
                if line[0] not in point_count:
                    point_count[line[0]] = []
                if line[-1] not in point_count:
                    point_count[line[-1]] = []
                
                point_count[line[0]].append(idx)
                point_count[line[-1]].append(idx)


            for idx in point_count:
                if len(point_count[idx]) == 1:
                    end_points.append(idx)
                else:
                    junction_points.append(idx)

        connected_lines = new_connected_lines
    
    return skeleton, new_connected_lines, end_points, junction_points

def extend_branch(branch_idx, point_idx, remove_list, add_list, connected_lines, points, voxel_sizes):
    considered_lines = []
    extended_line = connected_lines[branch_idx]

    for idx, line in enumerate(connected_lines):
        if idx in remove_list or idx in add_list:
            continue
        else:
            if line[0] == extended_line[point_idx] or line[-1] == extended_line[point_idx]:
                considered_lines.append(idx)

    if len(considered_lines) == 0:
        return remove_list, add_list
    else:
        max_angle = 0
        max_idx = None
        max_head = None
        max_weighted_len = None

        if point_idx == 0:
            vector_1 = (points[extended_line[1]] - points[extended_line[0]])*voxel_sizes
        else:
            vector_1 = (points[extended_line[-2]] - points[extended_line[-1]])*voxel_sizes

        for idx in considered_lines:
            considered_line = connected_lines[idx]

            if considered_line[0] == extended_line[point_idx]:
                vector_2 = points[considered_line[1]] - points[considered_line[0]]
                result_head = extend_list(idx, connected_lines, -1, points, voxel_sizes)
                head = -1
            else:
                vector_2 = points[considered_line[-2]] - points[considered_line[-1]]
                result_head = extend_list(idx, connected_lines, 0, points, voxel_sizes)
                head = 0

            cos_theta = np.dot(vector_1, vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
            # Compute the angle in radians
            theta_radians = np.arccos(cos_theta)
            angle = np.degrees(theta_radians)

            print('Line ', branch_idx, angle, idx, find_actual_length(result_head, points, voxel_sizes))

            if max_idx == None:
                max_angle = angle
                max_idx = idx
                max_head = head
                max_weighted_len = find_actual_length(result_head, points, voxel_sizes)
            elif (find_actual_length(result_head, points, voxel_sizes) > 1.2*max_weighted_len):
                max_angle = angle
                max_idx = idx
                max_head = head
                max_weighted_len = find_actual_length(result_head, points, voxel_sizes)
            elif (angle > max_angle + 2):
                max_angle = angle
                max_idx = idx
                max_head = head
                max_weighted_len = find_actual_length(result_head, points, voxel_sizes)
            elif angle >= max_angle - 20 and angle <= max_angle + 20: 
                if find_actual_length(result_head, points, voxel_sizes) > max_weighted_len+10:
                    max_angle = angle
                    max_idx = idx
                    max_head = head
                    max_weighted_len = find_actual_length(result_head, points, voxel_sizes)
                else:
                    if len(considered_line) > 1.5*len(connected_lines[max_idx]):
                        max_angle = angle
                        max_idx = idx
                        max_head = head
                        max_weighted_len = find_actual_length(result_head, points, voxel_sizes)

            print('Line ', branch_idx, max_idx)

        add_list.append(max_idx)
        for idx in considered_lines:
            if idx != max_idx:
                remove_list.append(idx)

        print(add_list, remove_list, len(connected_lines))
        return extend_branch(max_idx, max_head, remove_list, add_list, connected_lines, points, voxel_sizes)

def find_actual_length(list_points, points, voxel_sizes):
    length = 0
    for i in range(len(list(list_points))-1):
        length += euclidean_distance(points[list_points[i]]*voxel_sizes, points[list_points[i+1]]*voxel_sizes)

    return length

def extend_list(start_list_idx, list_of_lists, extend_at, points, voxel_sizes):
    def recursive_extend(current_list, current_used_indices, current_point, extend_at):
        longest_extension = list(current_list)
        
        for i, lst in enumerate(list_of_lists):
            if i in current_used_indices:
                continue
            
            if extend_at == 0:  # extend at head
                if lst[-1] == current_point:
                    new_list = deque(current_list)
                    new_list.extendleft(reversed(lst[:-1]))
                    new_used_indices = current_used_indices.copy()
                    new_used_indices.add(i)
                    extended_list = recursive_extend(new_list, new_used_indices, new_list[0], extend_at)
                    if find_actual_length(extended_list, points, voxel_sizes) > find_actual_length(longest_extension, points, voxel_sizes):
                        longest_extension = extended_list
                elif lst[0] == current_point:
                    new_list = deque(current_list)
                    new_list.extendleft(lst[1:])
                    new_used_indices = current_used_indices.copy()
                    new_used_indices.add(i)
                    extended_list = recursive_extend(new_list, new_used_indices, new_list[0], extend_at)
                    if find_actual_length(extended_list, points, voxel_sizes) > find_actual_length(longest_extension, points, voxel_sizes):
                        longest_extension = extended_list
            else:  # extend at tail
                if lst[0] == current_point:
                    new_list = deque(current_list)
                    new_list.extend(lst[1:])
                    new_used_indices = current_used_indices.copy()
                    new_used_indices.add(i)
                    extended_list = recursive_extend(new_list, new_used_indices, new_list[-1], extend_at)
                    if find_actual_length(extended_list, points, voxel_sizes) > find_actual_length(longest_extension, points, voxel_sizes):
                        longest_extension = extended_list
                elif lst[-1] == current_point:
                    new_list = deque(current_list)
                    new_list.extend(reversed(lst[:-1]))
                    new_used_indices = current_used_indices.copy()
                    new_used_indices.add(i)
                    extended_list = recursive_extend(new_list, new_used_indices, new_list[-1], extend_at)
                    if find_actual_length(extended_list, points, voxel_sizes) > find_actual_length(longest_extension, points, voxel_sizes):
                        longest_extension = extended_list
        
        return longest_extension

    # Initialize the starting list and the set of used indices
    start_list = list_of_lists[start_list_idx]
    extended_list = deque(start_list)
    used_indices = set()
    used_indices.add(start_list_idx)

    # Determine the current point to start extension
    if extend_at == 0:
        current_point = extended_list[0]
    else:
        current_point = extended_list[-1]

    # Find the longest extension
    longest_extended_list = recursive_extend(extended_list, used_indices, current_point, extend_at)
    return list(longest_extended_list)

def remove_redundant_branch(skeleton, skeleton_points, end_points, junction_points, connected_lines, voxel_sizes, len_threshold=10):
    remove_list = []
    add_list = []
    
    max_length = 0
    max_index = None

    for idx, line in enumerate(connected_lines):
        if len(line) > max_length:
            max_length = len(line)
            max_index = idx
    
    add_list.append(max_index)

    remove_list, add_list = extend_branch(max_index, 0, remove_list, add_list, connected_lines, skeleton_points, voxel_sizes)
    remove_list, add_list = extend_branch(max_index, -1, remove_list, add_list, connected_lines, skeleton_points, voxel_sizes)
    new_connected_lines = [line for idx, line in enumerate(connected_lines) if idx in add_list]

    points = []
    for line in new_connected_lines:
        points += line
    points = list(set(points))
    positions = skeleton_points[points]

    array_3d = np.zeros(skeleton.shape, dtype=int)
    array_3d[positions[:, 0], positions[:, 1], positions[:, 2]] = 1

    end_points = []
    junction_points = []

    is_end = False
    
    while not is_end:
        is_end = True
        point_count = {}

        for idx, line in enumerate(new_connected_lines):
            if line[0] not in point_count:
                point_count[line[0]] = []
            if line[-1] not in point_count:
                point_count[line[-1]] = []
            
            point_count[line[0]].append(idx)
            point_count[line[-1]].append(idx)

        merge_point = None
        
        for idx in point_count:
            if len(point_count[idx]) == 2:
                line_idx_1 = point_count[idx][0]
                line_idx_2 = point_count[idx][1]
                line_1 = new_connected_lines[line_idx_1]
                line_2 = new_connected_lines[line_idx_2]

                new_line = []
                new_lines = []
                reversed_line_2 = line_2[::-1]

                if line_1[0] == line_2[0] and line_1[-1] != line_2[-1]:
                    new_line = reversed_line_2[:-1] + line_1
                    is_end = False
                elif line_1[0] == line_2[-1] and line_1[-1] != line_2[0]:
                    new_line = line_2[:-1] + line_1
                    is_end = False
                elif line_1[-1] == line_2[0] and line_1[0] != line_2[-1]:
                    new_line = line_1[:-1] + line_2
                    is_end = False
                elif line_1[-1] == line_2[-1] and line_1[0] != line_2[0]:
                    new_line = line_1[:-1] + reversed_line_2
                    is_end = False

                if not is_end:
                    for idx_p, line in enumerate(new_connected_lines):
                        if idx_p not in point_count[idx]:
                            new_lines.append(line)
                    
                    # print('Avant:', len(new_connected_lines))
                    # print(new_connected_lines)
                    new_lines.append(new_line)
                    new_connected_lines = new_lines
                    # print('Après:', len(new_connected_lines))
                    # print(new_connected_lines)
                    break
        
        end_points = []
        junction_points = []
        point_count = {}

        for idx, line in enumerate(new_connected_lines):
            if line[0] not in point_count:
                point_count[line[0]] = []
            if line[-1] not in point_count:
                point_count[line[-1]] = []
            
            point_count[line[0]].append(idx)
            point_count[line[-1]].append(idx)


        for idx in point_count:
            if len(point_count[idx]) == 1:
                end_points.append(idx)
            else:
                junction_points.append(idx)

    return array_3d, new_connected_lines, end_points, junction_points

def angle_between_lines(point1, point2, point3):
    # Calculate direction vectors of the two lines
    vector1 = np.array(point2) - np.array(point1)
    vector2 = np.array(point2) - np.array(point3)
    
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)

    dot_product = np.dot(unit_vector1, unit_vector2)

    angle = np.arccos(dot_product) 
    angle_degrees = np.degrees(angle)

    return angle_degrees

def artery_analyse(vmtk_boundary_vertices, smooth_points, smooth_connected_lines, distance_threshold, metric=0):
    new_splitted_lines, splitted_branches, point_clusters, colors = split_smooth_lines(smooth_connected_lines, smooth_points, distance_threshold)

    tree = KDTree(smooth_points)
    distances, indices = tree.query(vmtk_boundary_vertices, k=1)

    sum_distances = [0] * len(new_splitted_lines)
    min_distances = [100000] * len(new_splitted_lines)
    point_counts = [0] * len(new_splitted_lines)
    mean_distances = [0] * len(new_splitted_lines)

    for i in range(vertices.shape[0]):
        if indices[i] in point_clusters:
            sum_distances[point_clusters[indices[i]]] += distances[i]
            point_counts[point_clusters[indices[i]]] += 1

            if distances[i] < min_distances[point_clusters[indices[i]]]:
                min_distances[point_clusters[indices[i]]] = distances[i]

    for i in range(len(sum_distances)):
        if metric == 0:
            mean_distances[i] = min_distances[i]
        else:
            mean_distances[i] = sum_distances[i]/(point_counts[i]+0.0001)

    #Show metric values along the centerline
    cur_branch = splitted_branches[0]
    cur_pos = 0
    # for i in range(len(new_splitted_lines)):
    #     if splitted_branches[i] != cur_branch:
    #         cur_pos = 0
    #         cur_branch = splitted_branches[i]
        
    #     if cur_pos == 0:
    #         print("Branch ", cur_branch)

    #     print(f"""At {cur_pos}: """, mean_distances[i]*2, ' mm')
    #     cur_pos += distance_threshold

    #Apply metric to points in centerline
    points_values = []
    for i in range(smooth_points.shape[0]):
        diameter = 0
        if i in point_clusters:
            diameter = mean_distances[point_clusters[i]]*2
        points_values.append(diameter)

    
    return new_splitted_lines, points_values, splitted_branches

def find_ring_vertices(new_splitted_lines, smooth_points, vmtk_boundary_vertices, vmtk_boundary_faces, radius_threshold=5):
    chosen_vertices = []
    unchosen_vertices = []
    intersection_points = []
    radiuses = []
    all_surfaces = []

    # print('Number of intervals: ', len(new_splitted_lines))
    for line in new_splitted_lines:
        ring_vertices = []
        removed_vertices = []
        intsecpoints = []
        distances = []
        min_distance = 10000
        radius = []

        point1, point2 = smooth_points[line[0]], smooth_points[line[-1]]
        plane1_normal, plane2_normal = perpendicular_planes(point1, point2)

        d1 = plane1_normal[3]
        d2 = plane2_normal[3]

        cur = d1
        if d1 > d2:
            d1 = d2
            d2 = cur
        
        for idx, vertex in enumerate(vmtk_boundary_vertices):
            d3 =  -(plane1_normal[0]*vertex[0] + plane1_normal[1]*vertex[1] + plane1_normal[2]*vertex[2])
            if d3 >= d1 and d3 <= d2:
                intersection_point = find_projection_point_on_line(point1, point2, vertex)
                distance = euclidean_distance(vertex, intersection_point)

                if distance <= radius_threshold:
                    intsecpoints.append(intersection_point)
                    distances.append(distance)
                    ring_vertices.append(idx)

                    if distance < min_distance:
                        min_distance = distance
                        radius = [vertex, intersection_point]

        surfaces = select_faces_with_chosen_vertices(vmtk_boundary_vertices, vmtk_boundary_faces, ring_vertices, 2)
        filter_vertices = []
        

        for idx, vertex_index in enumerate(ring_vertices):
            vertex = vmtk_boundary_vertices[vertex_index]
            intersection_point = intsecpoints[idx]
            distance = distances[idx]
            exist = False

            for surface in surfaces:
                is_intersect, triangular_point = ray_intersects_triangle(vertex, intersection_point, surface)
                
                if is_intersect:
                    removed_vertices.append([vertex_index, intersection_point, triangular_point])
                    exist = True
                    break
                           
            # if not exist:
            #     if distance > 1.5*min_distance:
            #         exist = True

            if not exist:                                            
                filter_vertices.append([vertex_index, intersection_point])
        
        chosen_vertices.append(filter_vertices)
        unchosen_vertices.append(removed_vertices)
        intersection_points.append(intsecpoints)
        radiuses.append(radius)
        all_surfaces.append(surfaces)

    return chosen_vertices, unchosen_vertices, intersection_points, radiuses, all_surfaces

def perpendicular_planes(point1, point2):
    # Get the direction vector of the line
    direction_vector = point2 - point1

    a, b, c = direction_vector
    d1 = -(a*point1[0] + b*point1[1] + c*point1[2])
    d2 = -(a*point2[0] + b*point2[1] + c*point2[2])

    v1 = [a, b, c, d1]
    v2 = [a, b, c, d2]

    return v1, v2

# Initialize
sub_nums = [9, 129, 167, 269, 581, 619, 2285, 2463, 2799, 3857]
dataset_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/'
# sub_nums = ['BCW-1205-RES']

for sub_num in sub_nums:
    segment_file_path = dataset_dir + f'sub-{str(sub_num)}_run-1_mra_eICAB_CW.nii.gz'
    original_file_path = dataset_dir + f'sub-{str(sub_num)}_run-1_mra_resampled.nii.gz'

    # segment_file_path = dataset_dir + f'{str(sub_num)}.nii.gz'
    # original_file_path = dataset_dir + f'{str(sub_num)}_0000.nii.gz'

    # segment_file_path = dataset_dir + 'sub-581_run-1_mra_eICAB_CW.nii.gz'
    # original_file_path = dataset_dir + 'sub-581_run-1_mra_resampled.nii.gz'
    # centerline_file_path = dataset_dir + 'sub-9_run-1_mra_CircleOfWillis_centerline.nii.gz'

    segment_image = nib.load(segment_file_path)
    original_image = nib.load(original_file_path)
    # centerline_image = nib.load(centerline_file_path)

    intensity_threshold_1 = 0.1
    intensity_threshold_2 = 0.1
    gaussian_sigma=2
    distance_threshold=20
    laplacian_iter = 5
    neighbor_threshold_1 = 10
    neighbor_threshold_2 = neighbor_threshold_1 + 10
    resolution = 0.05
    len_threshold = 15

    original_data = original_image.get_fdata()
    segment_data = segment_image.get_fdata()
    # centerline_data = centerline_image.get_fdata()
    voxel_sizes = segment_image.header.get_zooms()
    info_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/info_files/' + str(sub_num) + '/'
    check_and_create_directory(info_dir)
    print('Voxel size:', voxel_sizes)
    print('Image shape:', segment_data.shape)

    for artery_index in [1, 2, 3]:
        print('Artery ', artery_index)
        ## For treated kising vessels
        # processed_mask = segment_data

        ## For untreated kissing vessels
        if artery_index in [1, 2]:
            processed_mask = find_skeleton_ica(segment_image, original_image, artery_index , 0.5, intensity_threshold_2, gaussian_sigma, neighbor_threshold_1, neighbor_threshold_2)
            processed_mask = remove_noisy_voxels(processed_mask, neighbor_threshold_1, True)
        else:
            # For normal artery
            processed_mask, cf_mask, surf_data = preprocess_data(original_data, segment_data, [artery_index], intensity_threshold_1, intensity_threshold_2, gaussian_sigma, neighbor_threshold_1, neighbor_threshold_2 )

        # Extract smooth centerline
        skeleton = skeletonize(processed_mask)
        skeleton_points, end_points, junction_points, connected_lines = find_graphs(skeleton)
        skeleton, connected_lines, end_points, junction_points = remove_short_branch(skeleton, skeleton_points, end_points, junction_points, connected_lines, len_threshold)
        skeleton, connected_lines, end_points, junction_points = remove_redundant_branch(skeleton, skeleton_points, end_points, junction_points, connected_lines, voxel_sizes, len_threshold)
        connected_lines, skeleton, skeleton_points = extend_skeleton(connected_lines, end_points, skeleton_points, original_data, processed_mask, skeleton, voxel_sizes)
        
        vertices, faces, normals, values = measure.marching_cubes(skeleton, level=0.5, spacing=voxel_sizes)
        skeleton_points = voxel_sizes*(skeleton_points+0.5)
        visualized_boundary_points = generate_points(vertices, 1, 'blue')

        line_traces = []
        for line in connected_lines:
            for idx in range(len(line)-1):
                line_traces.append(generate_lines(np.array([skeleton_points[line[idx]], skeleton_points[line[idx+1]]]), 2))

        vmtk_skeleton_vertices, vmtk_skeleton_faces = vmtk_smooth_mesh(vertices, faces, 2000)
        smooth_points, smooth_connected_lines = smooth_centerline(vmtk_skeleton_vertices, skeleton_points, connected_lines)

        # Extract boundary
        vertices, faces, normals, values = measure.marching_cubes(processed_mask, level=0.1, spacing=voxel_sizes)

        # Create an Open3D triangle mesh
        target_faces = 15000
        if (faces.shape[0] > target_faces):
            vertices, faces = vmtk_decimate_mesh(vertices, faces, target_faces)
        #     vertices = np.array(vertices, dtype=np.float64)
        #     faces = np.array(faces, dtype=np.int32)
        #     normals = np.array(normals, dtype=np.float64)
        #     mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces, v_normals_matrix=normals)
            
        #     # Create a MeshSet
        #     ms = pymeshlab.MeshSet()
        #     ms.add_mesh(mesh)
        #     ms.apply_filter("simplification_quadric_edge_collapse_decimation", targetfacenum=target_faces, preservenormal=True)

        #     # Get the resulting mesh
        #     resulting_mesh = ms.current_mesh()

        #     # Extract vertices and faces as numpy arrays
        #     vertices = np.array(resulting_mesh.vertex_matrix())
        #     faces = np.array(resulting_mesh.face_matrix())

        vmtk_boundary_vertices, vmtk_boundary_faces = vmtk_smooth_mesh(vertices, faces, 20, 1)

        
        print('After: ')
        print(vmtk_boundary_vertices.shape[0])
        print(vmtk_boundary_faces.shape[0])

        np.savetxt(info_dir + f'smooth_points_{artery_index}.txt', smooth_points, delimiter=',', fmt='%.2f')
        np.savetxt(info_dir + f'vmtk_boundary_vertices_{artery_index}.txt', vmtk_boundary_vertices, delimiter=',', fmt='%.2f')
        np.savetxt(info_dir + f'vmtk_boundary_faces_{artery_index}.txt', vmtk_boundary_faces, delimiter=',', fmt='%.2f')
        with open(info_dir + f'smooth_connected_lines_{artery_index}.json', 'w') as file:
            json.dump(smooth_connected_lines, file)  # indent=4 makes the file more readable


        
        visualized_skeleton_points = generate_points(skeleton_points, 3, 'red')
        visualized_boundary_vertices = generate_points(vmtk_boundary_vertices, 1, 'blue')
        show_figure([visualized_boundary_vertices, visualized_boundary_points, visualized_skeleton_points]+line_traces)