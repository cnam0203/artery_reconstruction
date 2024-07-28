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
import re
from collections import Counter

from visualize_graph import *
from ray_intersection import *
from descartes import PolygonPatch
import plotly.graph_objects as go
from ref_measurement import *

def split_smooth_lines(artery_index, smooth_connected_lines, points, distance_threshold=2):
    new_splitted_lines = []
    splitted_branches = []
    point_clusters = {}

    longest_branch = max(smooth_connected_lines, key=len)
    longest_branch_idx = smooth_connected_lines.index(longest_branch)

    point_1 = points[longest_branch[0]]
    point_2 = points[longest_branch[-1]]
    if artery_index in [1, 2, 3]:
        if point_1[2] > point_2[2]:
            longest_branch.reverse()
            smooth_connected_lines[longest_branch_idx] = longest_branch
    elif artery_index == 5:
        if point_1[0] > point_2[0]:
            longest_branch.reverse()
            smooth_connected_lines[longest_branch_idx] = longest_branch
    elif artery_index == 6:
        if point_1[0] < point_2[0]:
            longest_branch.reverse()
            smooth_connected_lines[longest_branch_idx] = longest_branch
    elif artery_index == 7:
        if point_1[0] < point_2[0]:
            longest_branch.reverse()
            smooth_connected_lines[longest_branch_idx] = longest_branch
    elif artery_index == 8:
        if point_1[0] > point_2[0]:
            longest_branch.reverse()
            smooth_connected_lines[longest_branch_idx] = longest_branch
    

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

    return new_splitted_lines, splitted_branches, point_clusters, colors, longest_branch_idx
    
def artery_analyse(artery_index, vmtk_boundary_vertices, smooth_points, smooth_connected_lines, distance_threshold, metric=0):
    new_splitted_lines, splitted_branches, point_clusters, colors, longest_branch_idx = split_smooth_lines(artery_index, smooth_connected_lines, smooth_points, distance_threshold)
    
    return new_splitted_lines, None, splitted_branches, longest_branch_idx

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
    removed_vertices = []
    centerpoints = []
    vertex_ring = {}
    min_distances = []

    for idx in range(vmtk_boundary_vertices.shape[0]):
        vertex_ring[idx] = []

    for line_idx, line in enumerate(new_splitted_lines):
        ring_vertices = []
        intsecpoints = []
        min_distance = 10000
        min_vertex_idx = None

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
                        min_vertex_idx = idx

        surfaces = select_faces_with_chosen_vertices(vmtk_boundary_vertices, vmtk_boundary_faces, ring_vertices, 2)
        form_vertice_index = select_form_vertices_by_ray(ring_vertices, vmtk_boundary_vertices, surfaces, intsecpoints, min_distance, 10)
        # form_vertice_index = select_form_vertices_by_connection(vmtk_boundary_vertices, surfaces, ring_vertices, min_vertex_idx)
        
        if len(form_vertice_index) > 2:
            ring_vertices_pos = vmtk_boundary_vertices[ring_vertices]
            pca = PCA(n_components=2)
            points_2d = pca.fit_transform(ring_vertices_pos)
            # Calculate Convex Hull
            hull = ConvexHull(points_2d[form_vertice_index])
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
        removed_vertices.append([])

        for idx in form_vertice_index:
            vertex_idx = ring_vertices[idx]
            chosen_vertices[-1].append(vertex_idx)

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

        if chosen_ring_idx != None:
            for ring_idx in ring_indices:
                distance = euclidean_distance(vmtk_boundary_vertices[vertex], centerpoints[ring_idx])
                if distance > 1.3*min_distance:
                    removed_vertices[ring_idx].append(vertex)


    for idx, _ in enumerate(chosen_vertices):
        chosen_vertices[idx] = [vertex for vertex in chosen_vertices[idx] if vertex not in removed_vertices[idx]]

    defined_vertices = [vertex for vertex in vertex_ring if vertex_ring[vertex] != None]
    undefined_vertices = [vertex for vertex in vertex_ring if vertex_ring[vertex] == None]

    defined_points = vmtk_boundary_vertices[defined_vertices]
    undefined_points = vmtk_boundary_vertices[undefined_vertices]

    kdtree = KDTree(defined_points)
    distances, indices = kdtree.query(undefined_points)

    for idx, _ in enumerate(undefined_points):
        vertex_ring[undefined_vertices[idx]] = vertex_ring[defined_vertices[indices[idx]]]

    undefined_vertices = [vertex for vertex in vertex_ring if vertex_ring[vertex] == None]
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

def find_longest_branch(splitted_branches):
    # Use Counter to count the occurrences of each number
    counter = Counter(splitted_branches)
    most_common_number, highest_frequency = counter.most_common(1)[0]

    return most_common_number

def append_to_file(file_path, data, is_replace):
    # Open the file in append mode and save the data
    if data.shape[0]:
        if not os.path.isfile(file_path):
            np.savetxt(file_path, data, delimiter=',', fmt='%.2f')

        if is_replace:
            with open(file_path, 'w') as f:
                np.savetxt(f, data, delimiter=',', fmt='%.2f')
        else:
            np.savetxt(file_path, data, delimiter=',', fmt='%.2f')

def compute_ring_diameters(chosen_ring_vertices, vmtk_boundary_vertices, chosen_centerpoints):
    diameter_segments = []
    
    for idx, ring_vertices in enumerate(chosen_ring_vertices):
        lines = []
        ring_diameters = []

        if len(ring_vertices) > 2 and len(ring_vertices) < 200:
            ring_pos = vmtk_boundary_vertices[ring_vertices]
            pca = PCA(n_components=2)
            points_2d = pca.fit_transform(ring_pos)
            cen_point = pca.transform(np.array([chosen_centerpoints[idx]]))[0]

            center_mass = np.mean(points_2d, axis=0)
            cen_point = (center_mass + cen_point)/2
            # plt.plot(points_2d[:, 0], points_2d[:, 1], 'o')
            # plt.plot(cen_point[0], cen_point[1], 'ro')
            # plt.show()
                
            for point_idx, point in enumerate(points_2d):
                chosen_rest_point_idx = None
                max_angle = -1

                for rest_point_idx, rest_point in enumerate(points_2d):
                    vector_1 = point - cen_point
                    vector_2 = rest_point - cen_point
                    angle = find_angle(vector_1, vector_2)

                    if 160 <= angle <= 180:
                        if angle > max_angle:
                            max_angle = angle
                            chosen_rest_point_idx = rest_point_idx

                if max_angle != -1:
                    lines.append([point_idx, chosen_rest_point_idx])
                    ring_diameters.append(euclidean_distance(ring_pos[point_idx], ring_pos[chosen_rest_point_idx]))
                else:
                    ring_diameters.append(euclidean_distance(ring_pos[point_idx], chosen_centerpoints[idx]) * 2)

        diameter_segments.append(ring_diameters)
    
    return diameter_segments

def process_diameters(diameter_segments, chosen_ring_vertices, vmtk_boundary_vertices, chosen_centerpoints):
    min_distances = []
    avg_distances = []

    for idx, ring_segments in enumerate(diameter_segments):
        diameters = diameter_segments[idx]

        if len(diameters):
            diameters = np.array(diameters)
            min_distances.append(np.min(diameters))
            avg_distances.append(np.mean(diameters))
        else:
            min_distances.append(0)
            avg_distances.append(0)

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

                if len(neighbor_avg_distances):
                    avg_distances[idx] = np.mean(np.array(neighbor_avg_distances))
                    min_distances[idx] = np.mean(np.array(neighbor_min_distances))

        undefined_ranges = [distance for distance in avg_distances if distance == 0 or distance is None]
        if len(undefined_ranges):
            is_stop = False

    return min_distances, avg_distances

mapping_names = {
    1: 'LICA',
    2: 'RICA',
    3: 'BA',
    5: 'LACA',
    6: 'RACA',
    7: 'LMCA',
    8: 'RMCA',
    17: 'LAchA',
    18: 'RAchA',
}

options = [
    {
        'dataset_dir': 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/tof_mra_julia/',
        'pattern': re.compile(r'sub-(\d+)_run-1_mra_eICAB_CW.nii.gz'),
        'arteries': [1, 2, 3],
        'is_replace': True,
        'org_pre_str': 'sub-',
        'org_post_str': '_run-1_mra_eICAB_CW.nii.gz',
        'seg_pre_str': 'sub-',
        'seg_post_str': '_run-1_mra_resampled.nii.gz',
    }, {
        'dataset_dir': 'E:/pascal/',
        'pattern': re.compile(r'^PT_(.*?)_ToF_eICAB_CW\.nii\.gz$'),
        'arteries': [17, 18],
        'is_replace': True,
        'org_pre_str': 'PT_',
        'org_post_str': '_ToF_eICAB_CW.nii.gz',
        'seg_pre_str': 'PT_',
        'seg_post_str': '_ToF_resampled.nii.gz',
    }, {
        'dataset_dir': 'E:/stenosis/',
        'pattern': re.compile(r'sub-(\d+)_run-1_mra_eICAB_CW.nii.gz'),
        'arteries': [1],
        'is_replace': True,
        'org_pre_str': 'sub-',
        'org_post_str': '_run-1_mra_eICAB_CW.nii.gz',
        'seg_pre_str': 'sub-',
        'seg_post_str': '_run-1_mra_resampled.nii.gz',
    }
]

option_idx = 2

option = options[option_idx]
dataset_dir = option['dataset_dir']
pattern = option['pattern']
chosen_arteries = option['arteries']
is_replace = option['is_replace']
org_pre_str = option['org_pre_str']
org_post_str = option['org_post_str']
seg_pre_str = option['seg_pre_str']
seg_post_str = option['seg_post_str']

if option_idx == 1:
    stenosis_df = pd.read_csv('C:/Users/nguc4116/Desktop/filtered_stenosis_all.csv', sep=',')

result_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/info_files/'
sub_nums = []

# Iterate over the files in the directory
for filename in os.listdir(dataset_dir):
    match = pattern.match(filename)
    if match:
        index = match.group(1)
        sub_nums.append(str(index))


for sub_num in ['1509']:
    print(result_dir + str(sub_num))
    
    segment_file_path = dataset_dir + f'{org_pre_str}{str(sub_num)}{org_post_str}'
    original_file_path = dataset_dir + f'{seg_pre_str}{str(sub_num)}{seg_post_str}'

    segment_image = nib.load(segment_file_path)
    original_image = nib.load(original_file_path)

    original_data = original_image.get_fdata()
    segment_data = segment_image.get_fdata()
    voxel_sizes = segment_image.header.get_zooms()
    info = {}
    info_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/info_files/' + str(sub_num) + '/'

    if not os.path.exists(info_dir):
        os.makedirs(info_dir)

    showed_data = []

    vmtk_boundary_vertices_all = []
    vmtk_boundary_faces_all = []
    stenosis_ratios_all = []
    min_diam_rings = []
    vert_num = 0
    line_traces = []
    end_points = []
    start_points = [] 
    middle_points = []
    cons_points = []
    cen_points = []
    cen_values = []

    for artery_index in chosen_arteries:
        artery_key = "Artery_" + str(artery_index)
        print(artery_key)
        info[artery_key] = []
        min_vertices = []

        if not os.path.isfile(info_dir + f'smooth_points_{artery_index}.txt'): 
            continue
        if os.path.isfile(info_dir + f'measure_{artery_index}.png') and not is_replace:
            continue
        
        smooth_points = np.genfromtxt(info_dir + f'smooth_points_{artery_index}.txt', delimiter=',')
        vmtk_boundary_vertices = np.genfromtxt(info_dir + f'vmtk_boundary_vertices_{artery_index}.txt', delimiter=',')
        vmtk_boundary_vertices = np.round(vmtk_boundary_vertices, 2)
        vmtk_boundary_faces = np.genfromtxt(info_dir + f'vmtk_boundary_faces_{artery_index}.txt', delimiter=',', dtype=int)
        vmtk_boundary_vertices, inverse_indices = np.unique(vmtk_boundary_vertices, axis=0, return_inverse=True)
        vmtk_boundary_faces = np.array([inverse_indices[face] for face in vmtk_boundary_faces])

        with open(info_dir + f'smooth_connected_lines_{artery_index}.json', 'r') as file:
            smooth_connected_lines = json.load(file)
        
        # Calculate distance
        distance_threshold = 0.5
        new_splitted_lines, points_values, splitted_branches, longest_branch_idx = artery_analyse(artery_index, vmtk_boundary_vertices, smooth_points, smooth_connected_lines, distance_threshold, metric=1)
        ring_vertices, centerpoints, vertex_ring = find_ring_vertices(new_splitted_lines, smooth_points, vmtk_boundary_vertices, vmtk_boundary_faces)
        chosen_ring_vertices = [ring for idx, ring in enumerate(ring_vertices) if splitted_branches[idx] == longest_branch_idx]
        with open(info_dir + f'chosen_ring_{artery_index}.json', 'w') as file:
            json.dump(chosen_ring_vertices, file)  # indent=4 makes the file more readable

        chosen_centerpoints = [centerpoint for idx, centerpoint in enumerate(centerpoints) if splitted_branches[idx] == longest_branch_idx]
        start_points.append(chosen_centerpoints[0])
        end_points.append(chosen_centerpoints[-1])
        middle_points.append(chosen_centerpoints[int(len(chosen_centerpoints)/2)])
        cons_points.append(chosen_centerpoints[int(len(chosen_centerpoints)/4)])
        cons_points.append(chosen_centerpoints[int(3*len(chosen_centerpoints)/4)])
        cen_points += chosen_centerpoints

        min_distances = []
        avg_distances = []
        diameter_segments = compute_ring_diameters(chosen_ring_vertices, vmtk_boundary_vertices, chosen_centerpoints)
        min_distances, avg_distances = process_diameters(diameter_segments, chosen_ring_vertices, vmtk_boundary_vertices, chosen_centerpoints)
        
        # Creating the DataFrame
        x_values = [i*distance_threshold for i in range(len(chosen_ring_vertices))]
        df = pd.DataFrame({
            'ID': x_values,
            'min_distances': np.array(min_distances),
            'avg_radius': np.array(avg_distances),
        })

        # Exporting the DataFrame to a CSV file
        print('Finish')
        df.to_csv(info_dir + f'measure_output_{artery_index}.csv', index=False)

    append_to_file(os.path.join(info_dir, 'start_points.txt'), np.array(start_points), is_replace)
    append_to_file(os.path.join(info_dir, 'end_points.txt'), np.array(end_points), is_replace)
    append_to_file(os.path.join(info_dir, 'middle_points.txt'), np.array(middle_points), is_replace)
    append_to_file(os.path.join(info_dir, 'cons_points.txt'), np.array(cons_points), is_replace)