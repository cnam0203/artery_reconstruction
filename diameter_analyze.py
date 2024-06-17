import nibabel as nib
import numpy as np
from scipy.spatial import KDTree, distance_matrix
from sklearn.cluster import DBSCAN, KMeans
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Delaunay
import pandas as pd

import alphashape
import matplotlib.pyplot as plt
import math
import time
import os
import json
import heapq
import copy
import trimesh as tm
import random
from collections import Counter

from visualize_graph import *
from ray_intersection import *
from descartes import PolygonPatch
import plotly.graph_objects as go

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
    
def artery_analyse(vmtk_boundary_vertices, smooth_points, smooth_connected_lines, distance_threshold, metric=0):
    new_splitted_lines, splitted_branches, point_clusters, colors = split_smooth_lines(smooth_connected_lines, smooth_points, distance_threshold)
    
    return new_splitted_lines, None, splitted_branches

def is_point_in_hull(point, delaunay):
    """
    Check if a point is inside a convex hull.

    Parameters:
    point (array-like): The point to check.
    hull (ConvexHull): The convex hull.

    Returns:
    bool: True if the point is inside the convex hull, False otherwise.
    """
    return delaunay.find_simplex(point) >= 0

def find_ring_vertices(new_splitted_lines, smooth_points, vmtk_boundary_vertices, vmtk_boundary_faces, radius_threshold=5):
    chosen_vertices = []
    centerpoints = []
    vertex_ring = {}
    min_distances = []

    for idx in range(vmtk_boundary_vertices.shape[0]):
        vertex_ring[idx] = []

    for line_idx, line in enumerate(new_splitted_lines):
        ring_vertices = []
        intsecpoints = []
        min_distance = 10000

        point1, point2 = smooth_points[line[0]], smooth_points[line[-1]]
        centerpoint = (point1+point2)/2
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
                    ring_vertices.append(idx)

                    if distance < min_distance:
                        min_distance = distance

        surfaces = select_faces_with_chosen_vertices(vmtk_boundary_vertices, vmtk_boundary_faces, ring_vertices, 2)
        info_vertices = []
        form_vertice_index = []
        
        for idx, vertex_index in enumerate(ring_vertices):
            vertex = vmtk_boundary_vertices[vertex_index]
            intersection_point = intsecpoints[idx]
            distance = euclidean_distance(vertex, intersection_point)
            exist = False

            for surface in surfaces:
                is_intersect, triangular_point = ray_intersects_triangle(vertex, intersection_point, surface)
                
                if is_intersect:
                    exist = True
                    break
                           
            if not exist:
                if distance > 8*min_distance:
                    exist = True

            if not exist:                      
                form_vertice_index.append(idx)         

        if len(form_vertice_index) > 2:
            ring_vertices_pos = vmtk_boundary_vertices[ring_vertices]
            pca = PCA(n_components=2)
            points_2d = pca.fit_transform(ring_vertices_pos)
            # Calculate Convex Hull
            hull = ConvexHull(points_2d[form_vertice_index])
            filter_vertices = []
            removed_vertices = []
            delaunay = Delaunay(hull.points[hull.vertices])

            for idx, point in enumerate(points_2d):
                if is_point_in_hull(point, delaunay):
                    vertex_idx = ring_vertices[idx]
                    vertex_ring[vertex_idx].append(line_idx)

        else:
            for idx in form_vertice_index:
                vertex_idx = ring_vertices[idx]
                vertex_ring[vertex_idx].append(line_idx)

        centerpoints.append(centerpoint)
        min_distances.append(min_distance)
        chosen_vertices.append([])

    for vertex in vertex_ring:
        ring_indices = vertex_ring[vertex]
        min_distance = 10000
        chosen_ring_idx = None

        for ring_idx in ring_indices:
            distance = euclidean_distance(vmtk_boundary_vertices[vertex], centerpoints[ring_idx])
            if distance < min_distance:
                min_distance = distance
                chosen_ring_idx = ring_idx

        vertex_ring[vertex] = chosen_ring_idx    

        if chosen_ring_idx:
            chosen_vertices[chosen_ring_idx].append(vertex)
        
    defined_vertices = [vertex for vertex in vertex_ring if vertex_ring[vertex] != None]
    undefined_vertices = [vertex for vertex in vertex_ring if vertex_ring[vertex] == None]

    defined_points = vmtk_boundary_vertices[defined_vertices]
    undefined_points = vmtk_boundary_vertices[undefined_vertices]

    kdtree = KDTree(defined_points)
    distances, indices = kdtree.query(undefined_points)

    for idx, _ in enumerate(undefined_points):
        vertex_ring[undefined_vertices[idx]] = vertex_ring[defined_vertices[indices[idx]]]

    return chosen_vertices, centerpoints, vertex_ring

def perpendicular_planes(point1, point2):
    # Get the direction vector of the line
    direction_vector = point2 - point1

    a, b, c = direction_vector
    d1 = -(a*point1[0] + b*point1[1] + c*point1[2])
    d2 = -(a*point2[0] + b*point2[1] + c*point2[2])

    v1 = [a, b, c, d1]
    v2 = [a, b, c, d2]

    return v1, v2

def point_to_line_distance(point, line_start, line_end):
    """
    Calculate the shortest distance from a point to a line segment.
    """
    px, py = point[0], point[1]
    x1, y1 = line_start[0], line_start[1]
    x2, y2 = line_end[0], line_end[1]
    
    # Calculate the line segment length squared
    line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
    
    if line_length_sq == 0:
        # The line segment is a point
        return np.linalg.norm(np.array(point) - np.array(line_start))
    
    # Projection formula to find the closest point on the line segment
    t = max(0, min(1, np.dot(np.array([px - x1, py - y1]), np.array([x2 - x1, y2 - y1])) / line_length_sq))
    projection = np.array([x1, y1]) + t * np.array([x2 - x1, y2 - y1])
    
    return np.linalg.norm(np.array(point) - projection)

def find_closest_edges(points, edges):
    """
    Find the closest edge and its distance for each point.
    
    points: List of tuples (x, y)
    edges: List of tuples ((x1, y1), (x2, y2))
    
    Returns a list of tuples (closest_edge, shortest_distance)
    """
    results = []
    
    for point in points:
        min_distance = float('inf')
        closest_edge = None
        
        for edge in edges:
            distance = point_to_line_distance(point, edge[0], edge[1])
            if distance < min_distance:
                min_distance = distance
                closest_edge = edge
        
        results.append(round(min_distance, 2))
    
    return results

def sort_edges(edges):
    sorted_edges = [edges[0]]  # Start with the first edge
    count = 0

    while count < len(edges):
        for edge in edges:
            if edge[0] == sorted_edges[-1][1] and edge[1] != sorted_edges[-1][0]:
                sorted_edges.append(edge)
                count += 1
                break
            elif edge[1] == sorted_edges[-1][1] and edge[0] != sorted_edges[-1][0]:
                sorted_edges.append([edge[1], edge[0]])
                count += 1
                break
                
    return sorted_edges

def area_of_polygon_from_edges(vertices, point_2d):
    vertices = point_2d[np.array(vertices)]
    
    # Calculate the areas of triangles formed by consecutive vertices
    areas = 0.5 * np.abs(np.dot(vertices[:-1, 0], np.roll(vertices[:-1, 1], 1)) -
                         np.dot(vertices[:-1, 1], np.roll(vertices[:-1, 0], 1)))
    
    return round(np.sum(areas), 2)

def normalize_array(array):
    """
    Normalize a NumPy array to the range [0, 1].
    
    Parameters:
    - array: NumPy array to be normalized.
    
    Returns:
    - Normalized array.
    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def point_on_line(a, b, p):
    ap = p - a
    ab = b - a
    result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result

def point_on_line(a, b, p):
    ap = p - a
    ab = b - a
    result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result

def find_middle_point(point_1_idx, point_2_idx, rest_points, points_2d, distance):
    point_1 = points_2d[point_1_idx]
    point_2 = points_2d[point_2_idx]
    # Vector from P1 to P2
    # line_vec = point_2 - point_1
    # line_len = np.linalg.norm(line_vec)
    # line_unitvec = line_vec / line_len
    
    min_dist = 100000
    nearest_point = None
    
    for idx in rest_points:
        point = points_2d[idx]
        projection_point = point_on_line(point_1, point_2, point)

        if point_1[0] < point_2[0]:
            min_x = point_1[0]
            max_x = point_2[0]
        else:
            min_x = point_2[0]
            max_x = point_1[0]

        if point_1[1] < point_2[1]:
            min_y = point_1[1]
            max_y = point_2[1]
        else:
            min_y = point_2[1]
            max_y = point_1[1]

        
        if min_x <= projection_point[0] <= max_x and min_y <= projection_point[1] <= max_y:
            dist = np.linalg.norm(point - projection_point)
            if dist < min_dist and dist < distance:
                min_dist = dist
                nearest_point = idx

        # vec_to_point = point - point_1
        # projection_length = np.dot(vec_to_point, line_unitvec)
        
        # if projection_length < 0:
        #     projection = point_1
        # elif projection_length > line_len:
        #     projection = point_2
        # else:
        #     projection = point_1 + projection_length * line_unitvec
        
        # dist = np.linalg.norm(point - projection)
        
        # if dist < min_dist and dist < distance:
        #     min_dist = dist
        #     nearest_point = idx
    
    return nearest_point

def find_interior_hull(points_2d):
    # x = np.concatenate((x[0,:], x[1,:]))
    # y = np.concatenate((y[0,:], y[1,:]))
    x = points_2d[:, 0]
    y = points_2d[:, 1]

    rSquared = x**2 + y**2
    q = rSquared / max(rSquared)**2
    xx = x / q
    yy = y / q

    hull = ConvexHull(np.column_stack((xx, yy)))

    concave_points = hull.vertices.tolist()
    rest_points = [idx for idx, i in enumerate(points_2d) if idx not in concave_points]
    is_found = True

    while is_found:
        is_found = False
        new_concave_points = []
        
        for i in range(len(concave_points)):
            point_1 = concave_points[i-1]
            point_2 = concave_points[i]

            distance = np.linalg.norm(points_2d[point_1] - points_2d[point_2])

            if distance > 0.5:
                new_point = find_middle_point(point_1, point_2, rest_points, points_2d, distance)
                if new_point != None:
                    is_found = True
                    new_concave_points.append(new_point)
                    rest_points = [element for element in rest_points if element != new_point]

            new_concave_points.append(point_2)
        
        concave_points = new_concave_points

    return hull, np.array(new_concave_points)
    # plt.plot(points_2d[:, 0], points_2d[:, 1], 'o')
    # for simplex in hull.simplices:
    #     plt.plot(points_2d[simplex, 0], points_2d[simplex, 1], 'k-')
    # plt.fill(points_2d[hull.vertices, 0], points_2d[hull.vertices, 1], 'b', alpha=0.2)
    # plt.show()

def side_stable_mask(arr, ratio_threshold, dif_thresh):
    a = np.copy(arr)
    is_end = False
    loop_count = 0
    while not is_end and loop_count <= 50:
        loop_count += 1
        max_value = round(a.max(), 1)
        mask = np.r_[ False, np.abs(a - max_value) < dif_thresh, False]
        idx = np.flatnonzero(mask[1:] != mask[:-1])
        s0 = (idx[1::2] - idx[::2]).argmax()
        valid_mask = np.zeros(a.size, dtype=int) #Use dtype=bool for mask o/p
        valid_mask[idx[2*s0]:idx[2*s0+1]] = 1

        if np.argwhere(valid_mask == 1).shape[0] >= ratio_threshold*a.shape[0]:
            is_end = True
        else:
            second_max_value = round(np.max(a[a != max_value]), 1)
            
            if second_max_value >= max_value - dif_thresh:
                a[a == max_value] = second_max_value
            else:
                a[a == max_value] = 0
            
    return a, valid_mask

def max_stable_mask(a, pos, ratio_threshold, distance, interval_size, dif_thresh): # thresh controls noise
    pass_steps = int(distance/interval_size)
    
    left_mask = None
    right_mask = None
    left_radius = None
    right_radius = None
    mean_radius = None
    #Left
    if pos - 2*pass_steps >= 0:
        left_arr = a[pos - 2*pass_steps : pos - pass_steps]
        left_values, left_mask = side_stable_mask(left_arr, ratio_threshold, dif_thresh)
        left_radius = np.mean(left_values[left_mask == 1])

    if pos + 2*pass_steps < a.shape[0]:
        right_arr = a[pos + pass_steps : pos + 2*pass_steps]
        right_values, right_mask = side_stable_mask(right_arr, ratio_threshold, dif_thresh)
        right_radius = np.mean(right_values[right_mask == 1])
    
    if left_radius == None and right_radius != None:
        mean_radius = right_radius
    elif left_radius != None and right_radius == None:
        mean_radius = left_radius
    elif left_radius != None and right_radius != None:
        mean_radius = (left_radius + right_radius)/2
    
    return mean_radius

def find_longest_branch(splitted_branches):
    # Use Counter to count the occurrences of each number
    counter = Counter(splitted_branches)
    most_common_number, highest_frequency = counter.most_common(1)[0]

    return most_common_number

sub_nums = [9, 129, 167, 269, 581, 619, 2285, 2463, 2799, 3857]

for sub_num in sub_nums:
    sub_num = str(sub_num)
    dataset_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/'
    segment_file_path = dataset_dir + f'sub-{str(sub_num)}_run-1_mra_eICAB_CW.nii.gz'
    original_file_path = dataset_dir + f'sub-{str(sub_num)}_run-1_mra_resampled.nii.gz'

    segment_image = nib.load(segment_file_path)
    original_image = nib.load(original_file_path)

    original_data = original_image.get_fdata()
    segment_data = segment_image.get_fdata()
    voxel_sizes = segment_image.header.get_zooms()
    info = {}
    info_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/info_files/' + str(sub_num) + '/'
    showed_data = []

    vmtk_boundary_vertices_all = []
    vmtk_boundary_faces_all = []
    stenosis_ratios_all = []
    vert_num = 0
    line_traces = []
    end_points = []
    start_points = [] 
    middle_points = []
    cons_points = []
    chosen_arteries = [1]
    cen_points = []
    cen_values = []
    chosen_arteries = [1, 2, 3]

    for artery_index in chosen_arteries:
        artery_key = "Artery_" + str(artery_index)
        info[artery_key] = []
        
        print(artery_key)
        smooth_points = np.genfromtxt(info_dir + f'smooth_points_{artery_index}.txt', delimiter=',')
        vmtk_boundary_vertices = np.genfromtxt(info_dir + f'vmtk_boundary_vertices_{artery_index}.txt', delimiter=',')
        vmtk_boundary_vertices = np.round(vmtk_boundary_vertices, 2)
        vmtk_boundary_faces = np.genfromtxt(info_dir + f'vmtk_boundary_faces_{artery_index}.txt', delimiter=',', dtype=int)
        
        with open(info_dir + f'smooth_connected_lines_{artery_index}.json', 'r') as file:
            smooth_connected_lines = json.load(file)
        
        # Calculate distance
        distance_threshold = 0.5
        new_splitted_lines, points_values, splitted_branches = artery_analyse(vmtk_boundary_vertices, smooth_points, smooth_connected_lines, distance_threshold, metric=1)
        ring_vertices, centerpoints, vertex_ring = find_ring_vertices(new_splitted_lines, smooth_points, vmtk_boundary_vertices, vmtk_boundary_faces)
        longest_branch_idx = find_longest_branch(splitted_branches)
        chosen_ring_vertices = [ring for idx, ring in enumerate(ring_vertices) if splitted_branches[idx] == longest_branch_idx]
        chosen_centerpoints = [centerpoint for idx, centerpoint in enumerate(centerpoints) if splitted_branches[idx] == longest_branch_idx]
        start_points.append(chosen_centerpoints[0])
        end_points.append(chosen_centerpoints[-1])
        middle_points.append(chosen_centerpoints[int(len(chosen_centerpoints)/2)])
        cons_points.append(chosen_centerpoints[int(len(chosen_centerpoints)/4)])
        cons_points.append(chosen_centerpoints[int(3*len(chosen_centerpoints)/4)])
        cen_points += chosen_centerpoints

        min_distances = []
        avg_distances = []

        for idx, ring in enumerate(chosen_ring_vertices):
            min_distance = 10000
            distances = []

            for vertex in ring:
                distance = euclidean_distance(vmtk_boundary_vertices[vertex], chosen_centerpoints[idx])
                distances.append(distance)

                if distance < min_distance:
                    min_distance = distance

            if min_distance == 10000:
                min_distances.append(0)
                avg_distances.append(0)
            else:
                min_distances.append(min_distance)
                avg_distances.append(np.mean(np.array(distances)))

        is_stop = False
        while not is_stop:
            is_stop = True
            for idx, ring in enumerate(chosen_ring_vertices):
                neighbor_min_distances = []
                neighbor_avg_distances = []

                if avg_distances[idx] == 0:
                    if idx > 0 and avg_distances[idx-1]:
                        neighbor_min_distances.append(min_distances[idx-1])
                        neighbor_avg_distances.append(avg_distances[idx-1])
                    if idx < (len(chosen_ring_vertices) - 1) and avg_distances[idx+1]:
                        neighbor_min_distances.append(min_distances[idx+1])
                        neighbor_avg_distances.append(avg_distances[idx+1])

                    avg_distances[idx] = np.mean(np.array(neighbor_avg_distances))
                    min_distances[idx] = np.mean(np.array(neighbor_min_distances))
            
            undefined_ranges = [distance for distance in avg_distances if distance == 0 or distance is None]
            if len(undefined_ranges):
                is_stop = False

        ref_min_distances = []
        ref_avg_distances = []

        for i in range(len(chosen_ring_vertices)):
            ratio_threshold = 0.1
            distance = (1/10)*len(chosen_ring_vertices)*distance_threshold
            interval_size = distance_threshold
            dif_thresh = 0.5*avg_distances[i]

            avg_distance = max_stable_mask(np.array(avg_distances), i, ratio_threshold, distance, interval_size, dif_thresh)
            min_distance = max_stable_mask(np.array(min_distances), i, ratio_threshold, distance, interval_size, dif_thresh)
            
            ref_min_distances.append(min_distance)
            ref_avg_distances.append(avg_distance)

        is_stop = False
        while not is_stop:
            is_stop = True
            for idx, ring in enumerate(chosen_ring_vertices):
                neighbor_min_distances = []
                neighbor_avg_distances = []

                if ref_avg_distances[idx] == None:
                    if idx > 0 and ref_avg_distances[idx-1] != None:
                        neighbor_min_distances.append(ref_min_distances[idx-1])
                        neighbor_avg_distances.append(ref_avg_distances[idx-1])
                    if idx < (len(chosen_ring_vertices) - 1) and ref_avg_distances[idx+1] != None:
                        neighbor_min_distances.append(ref_min_distances[idx+1])
                        neighbor_avg_distances.append(ref_avg_distances[idx+1])

                    ref_avg_distances[idx] = np.mean(np.array(neighbor_avg_distances))
                    ref_min_distances[idx] = np.mean(np.array(neighbor_min_distances))

            undefined_ranges = [distance for distance in ref_avg_distances if distance == 0 or distance is None]
            if len(undefined_ranges):
                is_stop = False

        ratios = []
        for idx, point in enumerate(vmtk_boundary_vertices):
            ring_idx = vertex_ring[idx]
            centerpoint = centerpoints[ring_idx]
            distance = euclidean_distance(point, centerpoint)
            ratio = distance/ref_min_distances[ring_idx]
            
            if ratio > 1:
                ratio = 1
            ratios.append(ratio*100)

        stenosis_ratio_min = np.array(min_distances)/np.array(ref_min_distances)
        stenosis_ratio_min[stenosis_ratio_min > 1] = 1
        stenosis_ratio_min = 1 - stenosis_ratio_min
        stenosis_ratio_min_squared = stenosis_ratio_min**2
        stenosis_ratio_min_squared_fd = np.gradient(stenosis_ratio_min_squared)
        stenosis_ratio_min_squared_sd = np.gradient(stenosis_ratio_min_squared_fd)
        stenosis_min_squared_fd = np.gradient(np.array(min_distances))
        stenosis_min_squared_sd = np.gradient(stenosis_min_squared_fd)
        stenosis_ratio_avg = np.array(avg_distances)/np.array(ref_avg_distances)
        stenosis_ratio_avg[stenosis_ratio_avg > 1] = 1
        stenosis_ratio_avg = 1 - stenosis_ratio_avg
        stenosis_ratio_avg_squared = stenosis_ratio_avg**2
        stenosis_ratio_avg_squared_fd = np.gradient(stenosis_ratio_avg_squared)
        stenosis_ratio_avg_squared_sd = np.gradient(stenosis_ratio_avg_squared_fd)
        stenosis_ratio_avg_squared_fd = np.gradient(stenosis_ratio_avg_squared)
        
        circle_positions = [0, 0.25, 0.5, 0.75, 1]  # Example circle positions
        circle_colors = ['blue', 'green', 'yellow', 'green', 'red']  # Colors for the circles

        x_values = [i*distance_threshold for i in range(len(chosen_ring_vertices))]
        cen_values += x_values
        # plt.figure(figsize=(np.max(np.array(x_values)), np.max(np.array(avg_distances))))
        plt.figure()
        plt.subplot(4, 1, 1)  # Creating subplot 1 (top)
        plt.plot(x_values, np.array(min_distances), label='Min distances(MD)')
        plt.plot(x_values, np.array(ref_min_distances), label='(MV=(MD_BE+MD_AF)/2')
        for pos, color in zip(circle_positions, circle_colors):
            if pos < 1:
                plt.scatter(x_values[int(pos*len(x_values))], 0, color=color)
            else:
                plt.scatter(x_values[-1], 0, color=color)
        plt.xlabel('Length (mm)')
        plt.ylabel('Distance to centerline (mm)')
        plt.title('Change in distance')
        plt.legend(bbox_to_anchor = (1.5, 0.6), loc='upper right')  # Show legend with labels
        # plt.show()

        # plt.figure(figsize=(np.max(np.array(x_values)), np.max(np.array(stenosis_ratio_min))))
        plt.subplot(4, 1, 2)  # Creating subplot 2 (bottom)
        plt.plot(x_values, np.array(stenosis_min_squared_fd), label="MD'")
        plt.plot(x_values, np.array(stenosis_min_squared_sd ), label="MD''")
        for pos, color in zip(circle_positions, circle_colors):
            if pos < 1:
                plt.scatter(x_values[int(pos*len(x_values))], 0, color=color)
            else:
                plt.scatter(x_values[-1], 0, color=color)
        plt.xlabel('Length (mm)')
        plt.title('Change in distance gradient')
        plt.legend(bbox_to_anchor = (1.5, 0.6), loc='upper right')  # Show legend with labels


        # plt.figure(figsize=(np.max(np.array(x_values)), np.max(np.array(stenosis_ratio_min))))
        plt.subplot(4, 1, 3)  # Creating subplot 2 (bottom)
        plt.plot(x_values, np.array(stenosis_ratio_min), label='SR=MD/MV')
        plt.plot(x_values, np.array(stenosis_ratio_min_squared), label='SSR=SR^2')
        for pos, color in zip(circle_positions, circle_colors):
            if pos < 1:
                plt.scatter(x_values[int(pos*len(x_values))], 0, color=color)
            else:
                plt.scatter(x_values[-1], 0, color=color)
        plt.xlabel('Length (mm)')
        plt.ylabel('Stenosis ratio (%)')
        plt.title('Change in stenosis ratio')
        plt.legend(bbox_to_anchor = (1.5, 0.6), loc='upper right')  # Show legend with labels

        # plt.figure(figsize=(np.max(np.array(x_values)), np.max(np.array(stenosis_ratio_min))))
        plt.subplot(4, 1, 4)  # Creating subplot 2 (bottom)
        plt.plot(x_values, np.array(stenosis_ratio_min_squared_fd), label="SSR'")
        plt.plot(x_values, np.array(stenosis_ratio_min_squared_sd ), label="SSR''")
        for pos, color in zip(circle_positions, circle_colors):
            if pos < 1:
                plt.scatter(x_values[int(pos*len(x_values))], 0, color=color)
            else:
                plt.scatter(x_values[-1], 0, color=color)
        plt.xlabel('Length (mm)')
        plt.title('Change in stenosis ratio gradient')
        plt.legend(bbox_to_anchor = (1.5, 0.6), loc='upper right')  # Show legend with labels

        # plt.show()
        # Saving the combined plot as a PNG image
        plt.subplots_adjust(hspace=0.5)
        plt.tight_layout()  # Adjust the layout to prevent overlap
        plt.savefig('C:/Users/nguc4116/Desktop/artery_reconstruction/info_files/' + sub_num + '/measure_' + str(artery_index) + '.png')
        # plt.show()

        
        x_values = [i*distance_threshold for i in range(len(chosen_ring_vertices))]
        plt.figure(figsize=(np.max(np.array(x_values)), np.max(np.array(avg_distances))))
        plt.plot(x_values, np.array(avg_distances), label='Avg distances')
        plt.plot(x_values, np.array(ref_avg_distances), label='Referred distances')
        plt.plot(x_values, np.array(stenosis_ratio_avg*2), label='Stenosis ratio (Avg distances)')
        plt.xlabel('Length (mm)')
        plt.title('Change in vascular geometry')
        plt.legend()  # Show legend with labels
        # plt.show()


        x_values, np.array(min_distances)
        x_values, np.array(ref_min_distances)
        x_values, np.array(stenosis_min_squared_fd)
        x_values, np.array(stenosis_min_squared_fd)
        x_values, np.array(stenosis_ratio_min)
        x_values, np.array(stenosis_ratio_min_squared)
        x_values, np.array(stenosis_ratio_min_squared_fd)
        x_values, np.array(stenosis_ratio_min_squared_sd )

        # Creating the DataFrame
        df = pd.DataFrame({
            'ID': x_values,
            'min_distances': np.array(min_distances),
            'ref_min_distances': np.array(ref_min_distances),
            'stenosis_min_squared_fd': np.array(stenosis_min_squared_fd),
            'stenosis_min_squared_sd': np.array(stenosis_min_squared_fd),
            'stenosis_ratio_min': np.array(stenosis_ratio_min),
            'stenosis_ratio_min_squared': np.array(stenosis_ratio_min_squared),
            'stenosis_ratio_min_squared_fd': np.array(stenosis_ratio_min_squared_fd),
            'stenosis_ratio_min_squared_sd': np.array(stenosis_ratio_min_squared_sd),
            'avg_radius': np.array(avg_distances),
            'ref_avg_radius': np.array(ref_avg_distances),
            'stenosis_ratio_avg': np.array(stenosis_ratio_avg)
        })

        # Exporting the DataFrame to a CSV file
        df.to_csv(info_dir + f'measure_output_{artery_index}.csv', index=False)



        # x_values = [i*distance_threshold for i in range(len(radius))]
    #     # Find the longest branch
    #     branches, longest_branch_idx = find_longest_branch(ring_vertices, splitted_branches, radiuses, distance_threshold)
    #     chosen_ring = rings[longest_branch_idx]
    #     longest_branch = branches[longest_branch_idx]
    #     radius = [round(euclidean_distance(item[0], item[1]), 2) for item in longest_branch]

    #     # Stenosis evaluation
    #     # local_min = argrelextrema(np.array(radius), np.less)[0]
    #     # stenose_points = [item[1] for index, item in enumerate(longest_branch) if index in local_min.tolist()]
    #     # stenose_rings = [item for index, item in enumerate(chosen_ring) if index in local_min.tolist()]
    #     # stenose_radius = [item for index, item in enumerate(radius) if index in local_min.tolist()]
    #     stenose_rings += [item for index, item in enumerate(chosen_ring)]
    #     stenose_radius = [item for index, item in enumerate(radius)]

        
    # #     new_stenose_radius, valid_mask = side_stable_mask(np.array(stenose_radius), ratio_threshold=0.2, dif_thresh=0.1)
    # #     x_values = [i*distance_threshold for i in range(len(radius))]


    # #     stable_ring_points = []
    # #     refer_radius = np.min(new_stenose_radius[valid_mask == 1])

    # #     stenosis_grades = {
    # #         '3': [],
    # #         '2': [],
    # #         '1': [],
    # #         '0': []
    # #     }
    # #     stenosis_colors = {
    # #         '0': 'red',
    # #         '1': 'orange',
    # #         '2': 'yellow',
    # #         '3': 'green'
    # #     }

    # #     for idx, value in enumerate(valid_mask):
    # #         if value == 1:
    # #             for item in stenose_rings[idx]:
    # #                 stable_ring_points.append(item[0])

    # #         ratio = radius[idx]/refer_radius

    # #         if ratio < 0.2:
    # #             for item in stenose_rings[idx]:
    # #                 stenosis_grades['0'].append(item[0])
    # #         elif ratio < 0.5:
    # #             for item in stenose_rings[idx]:
    # #                 stenosis_grades['1'].append(item[0])
    # #         elif ratio < 0.8:
    # #             for item in stenose_rings[idx]:
    # #                 stenosis_grades['2'].append(item[0])
    # #         else:
    # #             for item in stenose_rings[idx]:
    # #                 stenosis_grades['3'].append(item[0])

        
    # #     stenosis_ratios = [10000]*vmtk_boundary_vertices.shape[0]

    # #     for stenose_ring in stenose_rings:
    # #         for point in stenose_ring:
    # #             idx = point[0]
    # #             distance = euclidean_distance(vmtk_boundary_vertices[idx], point[1])
    # #             if distance < stenosis_ratios[idx]:
    # #                 stenosis_ratios[idx] = distance
        
    # #     # Build KDTree from array1
    # #     kdtree = KDTree(smooth_points)
    # #     distances, indices = kdtree.query(vmtk_boundary_vertices)

    # #     for idx, value in enumerate(stenosis_ratios):
    # #         if value == 10000:
    # #             stenosis_ratios[idx] = distances[idx]

    # #     stenosis_ratios = np.array(stenosis_ratios)/refer_radius
    # #     stenosis_ratios[stenosis_ratios >= 1] = 1

    # #     # Initialize the color array with the same size as stenosis_ratios
    # #     color_array = np.empty(stenosis_ratios.shape, dtype='<U6')

    # #     # Assign colors based on the ratios
    # #     color_array[stenosis_ratios < 0.2] = 1
    # #     color_array[(stenosis_ratios >= 0.2) & (stenosis_ratios < 0.5)] = 2
    # #     color_array[(stenosis_ratios >= 0.5) & (stenosis_ratios < 0.7)] = 3
    # #     color_array[stenosis_ratios >= 0.7] = 4

        vmtk_boundary_vertices_all.append(vmtk_boundary_vertices)
        vmtk_boundary_faces_all.append(vmtk_boundary_faces + vert_num)
        stenosis_ratios_all.append(ratios)
        vert_num += vmtk_boundary_vertices.shape[0]

    # #     # stenose_ring_points = []
    # #     # stenosis_indices = []
    # #     # ex_surface_areas = []
    # #     # in_surface_areas = []
    # #     # is_stenoses = []
    # #     # max_distances = []

    # #     # for idx, ring in enumerate(stenose_rings):
    # #     #     ring_vertex = [item[0] for item in ring]

    # #     #     if len(ring_vertex) > 2:
    # #     #         ring_vertices = vmtk_boundary_vertices[ring_vertex]
    # #     #         pca = PCA(n_components=2)
    # #     #         points_2d = pca.fit_transform(ring_vertices)
    # #     #         # Calculate convex Hull
    # #     #         ex_hull = ConvexHull(points_2d)
    # #     #         ex_surface_area = area_of_polygon_from_edges(ex_hull.vertices, points_2d)
    # #     #         ex_surface_areas.append(ex_surface_area)
    # #     #         plt.plot(points_2d[:, 0], points_2d[:, 1], 'o')
    # #     #         for simplex in ex_hull.simplices:
    # #     #             plt.plot(points_2d[simplex, 0], points_2d[simplex, 1], 'k-')
    # #     #         plt.fill(points_2d[ex_hull.vertices, 0], points_2d[ex_hull.vertices, 1], 'b', alpha=0.2)

    # #     #         # Find maxmimum distance from interior points to convex hull
    # #     #         interior_points = [idx for idx, point in enumerate(points_2d) if idx not in ex_hull.vertices]
    # #     #         interior_pos = points_2d[interior_points]
    # #     #         edges = [points_2d[item] for item in ex_hull.simplices]
    # #     #         results = find_closest_edges(interior_pos, edges)
    # #     #         max_distances.append(np.max(np.array(results)))
    # #     #         max_pos = None
    # #     #         is_stenose = False
    # #     #         if np.max(np.array(results)) >= 0.4*stenose_radius[idx]:
    # #     #             is_stenose = True
    # #     #             max_idx = np.argmax(np.array(results))
    # #     #             max_pos = interior_pos[max_idx]

    # #     #             for item in stenose_rings[idx]:
    # #     #                 stenose_ring_points.append(item[0])
    # #     #         is_stenoses.append(is_stenose)

    # #     #         # Find the interior hull
    # #     #         in_hull, new_concave_points = find_interior_hull(points_2d)
    # #     #         in_surface_area = area_of_polygon_from_edges(new_concave_points, points_2d)
    # #     #         in_surface_areas.append(in_surface_area)

    # #     #         for idx in range(new_concave_points.shape[0]):
    # #     #             plt.plot(points_2d[new_concave_points[[idx-1, idx]], 0], points_2d[new_concave_points[[idx-1, idx]], 1], 'k-')
    # #     #         plt.fill(points_2d[new_concave_points, 0], points_2d[new_concave_points, 1], 'r', alpha=0.2)

    # #     #         # Plotting
    # #     #         if is_stenose:
    # #     #             print('Length (mm):', idx*distance_threshold, ', surface area:', ex_surface_area, ('mm2'), ', min distance to centerline:', radius[idx], 'mm', ', max distance to surface:', np.max(np.array(results)), 'mm', ', concave')
    # #     #             plt.suptitle('Stenosis')
    # #     #             stenosis_indices.append(idx)
    # #     #             plt.scatter(max_pos[0], max_pos[1], color='red', s=50)
    # #     #         else:
    # #     #             print('Length (mm):', idx*distance_threshold, ', surface area:', ex_surface_area, ('mm2'), ', min distance to centerline:', radius[idx], 'mm', ', max distance to surface:', np.max(np.array(results)), 'mm', ', convex')
    # #     #             plt.suptitle('Without stenosis')

    # #     #         plt.show()

    # #     #     else:
    # #     #         ex_surface_areas.append(0)
    # #     #         in_surface_areas.append(0)
    # #     #         max_distances.append(0)
    # #     #         is_stenoses.append(False)
    # #     # # Plotting the first line
    # #     # x_values = [i*distance_threshold for i in range(len(radius))]

    # #     # plt.figure(figsize=(np.max(np.array(x_values)), np.max(radius)))
    # #     # # plt.plot(x_values, normalize_array(np.array(radius)), label='Min distance to centerline')
    # #     # # plt.plot(x_values, normalize_array(np.array(max_distances)), label='Max distance to surface')
    # #     # # plt.plot(x_values, normalize_array(np.array(ex_surface_areas)), label='Ex Surface area')
    # #     # # plt.plot(x_values, normalize_array(np.array(in_surface_areas)), label='In Surface area')
    # #     # # plt.plot(x_values, np.array(is_stenoses).astype(int), marker='o', linestyle='-', label='Is stenose')

    # #     # plt.plot(x_values, np.array(ex_surface_areas), label='Ex Surface area')
    # #     # plt.plot(x_values, np.array(in_surface_areas), label='In Surface area')
    # #     # plt.plot(x_values, np.array(is_stenoses).astype(int), marker='o', linestyle='-', label='Is stenose')

    # #     # # Adding labels and title
    # #     # plt.xlabel('Length (mm)')
    # #     # plt.title('Change in vascular geometry')
    # #     # plt.legend()  # Show legend with labels

    # #     # # Displaying the plot
    # #     # plt.show()

    # #     # # stenosis_min = np.array([item for idx, item in enumerate(local_min.tolist()) if idx in stenosis_indices])
    # #     # # radius_array = np.array(radius)
    # #     # # x_values = [i*distance_threshold for i in range(len(radius))]
    # #     # # # Plot the line graph
    # #     # # plt.figure(figsize=(np.max(np.array(x_values)), np.max(radius_array)))
    # #     # # plt.plot(x_values, radius, label='Radius')
    # #     # # plt.scatter(local_min*distance_threshold, radius_array[local_min], color='red', zorder=5, label='Local Minima')
    # #     # # plt.scatter(stenosis_min*distance_threshold, radius_array[stenosis_min], color='green', zorder=5, label='Stenosis point')
    # #     # # plt.xlabel('Position')
    # #     # # plt.ylabel('Radius')
    # #     # # plt.title('Radius Values with Local Minima Highlighted')
    # #     # # plt.legend()
    # #     # # plt.show()

    # #     #     # alpha_shape = alphashape.alphashape(points_2d,0.01)

    # #     #     # # Plotting
    # #     #     # fig, ax = plt.subplots()
    # #     #     # ax.scatter(points_2d[:, 0], points_2d[:, 1], c='blue', marker='o')
    # #     #     # ax.add_patch(PolygonPatch(alpha_shape, alpha=0.5))
    # #     #     # ax.set_title('2D Projection of 3D Points using PCA')
    # #     #     # ax.set_xlabel('Principal Component 1')
    # #     #     # ax.set_ylabel('Principal Component 2')
    # #     #     # ax.axis('equal')  # Ensure equal scaling for both axes

    # #     #     # plt.show()

    # #     # # print("Min radius at half the length: ", radius[int(len(longest_branch)/2)])
        
        
    # #     # # max_branch_pos = [idx for idx, value in enumerate(splitted_branches) if value == longest_branch]
    # #     # # ring_vertices = [ring for idx, ring in enumerate(ring_vertices) if idx in max_branch_pos]
    # #     # # middle_ring = ring_vertices[int(len(ring_vertices)/2)]

    # #     # # middle_surface_points = vmtk_boundary_vertices[[sublist[0] for sublist in middle_ring]]
    # #     # # middle_intersection_points = intersection_points[int(len(intersection_points)/2)]

    # #     # # sum_distance = []
    # #     # # for idx, point in enumerate(middle_surface_points):
    # #     # #     sum_distance.append(euclidean_distance(point, middle_intersection_points[idx]))

    # #     # # mean_radius = sum(sum_distance)/len(sum_distance)
    # #     # # print("Average radius at half the length: ", round(mean_radius, 2))
        
    # #     visualized_boundary_points = generate_points(vmtk_boundary_vertices, 1, 'blue')
    # #     visualized_smooth_points = generate_points(smooth_points, 3, 'red')
    # #     visualized_stable_points = generate_points(vmtk_boundary_vertices[stable_ring_points], 3, 'red')
    # #     # visualized_stenose_points = generate_points(np.array(stenose_points), 5, 'red')
    # #     # visualized_stenose_ring_points = generate_points(vmtk_boundary_vertices[stenose_ring_points], 3, 'red')
        
        
        for line in smooth_connected_lines:
            line_traces.append(generate_lines(smooth_points[line], 2))


    # #     # visualize_stenose_grades = []
    # #     # for key in stenosis_grades:
    # #     #     visualize_stenose_grades.append(generate_points(vmtk_boundary_vertices[stenosis_grades[key]], 3, stenosis_colors[key]))

    # #     # showed_data.append(mesh)
    # #     # showed_data.append(visualized_smooth_points)
    # #     show_figure([visualized_boundary_points, visualized_stable_points, visualized_smooth_points], 'Stenosis grade along the extracted centerline of ICA'
    # #         # + line_traces + visualize_stenose_grades
    # #         )

    vmtk_boundary_vertices_all = np.concatenate(vmtk_boundary_vertices_all, axis=0)
    vmtk_boundary_faces_all = np.concatenate(vmtk_boundary_faces_all, axis=0)
    stenosis_ratios_all = np.concatenate(stenosis_ratios_all, axis=0)
    visualized_start_points = generate_points(np.array(start_points), 10, 'blue')
    visualized_end_points = generate_points(np.array(end_points), 10, 'red')
    visualized_middle_points = generate_points(np.array(middle_points), 10, 'yellow')
    visualized_cons_points = generate_points(np.array(cons_points), 10, 'green')
    visualized_cen_points = generate_points_values(np.array(cen_points),5, 'black', np.array(cen_values))

    mesh = generate_mesh(vmtk_boundary_vertices_all, vmtk_boundary_faces_all)
    mesh = generate_mesh_color(vmtk_boundary_vertices_all, vmtk_boundary_faces_all, stenosis_ratios_all, 'Stenosis ratio')
    showed_data.append(mesh)
    showed_data.append(visualized_start_points)
    showed_data.append(visualized_middle_points)
    showed_data.append(visualized_end_points)
    showed_data.append(visualized_cons_points)
    showed_data.append(visualized_cen_points) 
    show_figure(showed_data + line_traces, 'Stenosis grade along the extracted centerline of ICA'
    )
    # plt.show()

