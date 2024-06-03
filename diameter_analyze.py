import nibabel as nib
import numpy as np
from scipy.spatial import KDTree, distance_matrix
from sklearn.cluster import DBSCAN, KMeans
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Delaunay

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

from visualize_graph import *
from ray_intersection import *
from descartes import PolygonPatch

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
        info_vertices = []
        form_vertice_index = []
        
        for idx, vertex_index in enumerate(ring_vertices):
            vertex = vmtk_boundary_vertices[vertex_index]
            intersection_point = intsecpoints[idx]
            distance = distances[idx]
            exist = False

            for surface in surfaces:
                is_intersect, triangular_point = ray_intersects_triangle(vertex, intersection_point, surface)
                
                if is_intersect:
                    info_vertices.append([vertex_index, intersection_point, triangular_point])
                    exist = True
                    break
                           
            if not exist:
                if distance > 8*min_distance:
                    info_vertices.append([vertex_index, intersection_point, None])
                    exist = True

            if not exist:                      
                form_vertice_index.append(idx)                      
                info_vertices.append([vertex_index, intersection_point, None])

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
                    filter_vertices.append(info_vertices[idx])
                else:
                    removed_vertices.append(info_vertices[idx])
        else:
            for idx in range(len(info_vertices)):
                if idx in form_vertice_index:
                    filter_vertices.append(info_vertices[idx])
                else:
                    removed_vertices.append(info_vertices[idx])
                    
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
    
    print(point_1, point_2, distance)
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

dataset_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/'
segment_file_path = dataset_dir + 'BCW-1205-RES.nii.gz'
original_file_path = dataset_dir + 'BCW-1205-RES_0000.nii.gz'

segment_image = nib.load(segment_file_path)
original_image = nib.load(original_file_path)

original_data = original_image.get_fdata()
segment_data = segment_image.get_fdata()
voxel_sizes = segment_image.header.get_zooms()
info = {}
sub_num = 'BCW-1205-RES'
info_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/info_files/' + sub_num + '/'
showed_data = []

vmtk_boundary_vertices_all = []
vmtk_boundary_faces_all = []
stenosis_ratios_all = []
vert_num = 0

for artery_index in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
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
    ring_vertices, removed_vertices, intersection_points, radiuses, all_surfaces = find_ring_vertices(new_splitted_lines, smooth_points, vmtk_boundary_vertices, vmtk_boundary_faces)

    cur_branch = splitted_branches[0]
    cur_pos = 0
    branches = [[]]
    rings = [[]]
    line_traces = []

    for idx, ring in enumerate(ring_vertices):
        interval_radius = radiuses[idx]

        if splitted_branches[idx] != cur_branch:
            cur_pos = 0
            cur_branch = splitted_branches[idx]
            branches.append([])
            rings.append([])

        # if cur_pos == 0:
        #     print("Branch ", cur_branch)
        
        if len(interval_radius):
            # print(f"""Length = {round(cur_pos, 2)} mm, at the position""", interval_radius[1], ", min radius =""", round(euclidean_distance(interval_radius[0], interval_radius[1]), 2), '(mm)')
            branches[-1].append(interval_radius)
            rings[-1].append(ring)

        cur_pos += distance_threshold

    longest_branch = None
    longest_length = 0
    stenose_points = []
    stenose_rings = []

    for idx, branch in enumerate(branches):
        # radius = [euclidean_distance(item[0], item[1]) for item in branch]
        # np_branch = np.array(radius)

        if len(branch) > longest_length:
            longest_length = len(branch)
            longest_branch = idx

        # percentile_25 = np.percentile(np_branch, 25)
        # percentile_45 = np.percentile(np_branch, 45)
        # percentile_50 = np.percentile(np_branch, 50)
        # percentile_55 = np.percentile(np_branch, 55)
        # percentile_75 = np.percentile(np_branch, 75)
        # selected_range = np_branch[(np_branch >= percentile_25) & (np_branch <= percentile_75)]
        
        # local_min = argrelextrema(np.array(radius), np.less)[0]
        # stenose_points += [item[1] for index, item in enumerate(branch) if index in local_min.tolist()]
        
        # # Distance between each point (in mm)
        # distance_between_points = distance_threshold
        # x_values = [i * distance_between_points for i in range(len(radius))]

        # # Create the plot
        # plt.figure(figsize=(10, 6))
        # plt.plot(x_values, radius, marker='o', linestyle='-', color='b')

        # # Add title and labels
        # plt.title('Min Radius (mm)')
        # plt.xlabel('Length (mm)')
        # plt.ylabel('Radius (mm)')
        # plt.show()
        
        # # print('Branch ', idx, ':')
        # # print("0th: ", np.min(np_branch))
        # # print("25th: ", percentile_25)
        # # print("45th: ", percentile_45)
        # # print("50th: ", percentile_50)
        # # print("55th: ", percentile_55)
        # # print("75th: ", percentile_75)
        # # print("100th: ", np.max(np_branch))
        # print("Min radius at half the length: ", round(radius[int(len(branch)/2)], 2))
        # # print("Avg min radius (range 25th-75th): ", np.mean(selected_range))
        # # print("Avg min radius (range 0th-100th): ", np.mean(np_branch))

    chosen_ring = rings[longest_branch]
    longest_branch = branches[longest_branch]

    radius = [round(euclidean_distance(item[0], item[1]), 2) for item in longest_branch]
    # for i in range(len(longest_branch)):
    #     info[artery_key].append({
    #         'Length (mm)': i*distance_threshold,
    #         'Position at': longest_branch[i][1].tolist(),
    #         'Min radius (mm)': radius[i]
    #     })
        
    #     print(f"""Length = {i*distance_threshold} mm, at the position""", longest_branch[i][1], ", min radius =", radius[i], '(mm)')
    
    local_min = argrelextrema(np.array(radius), np.less)[0]
    stenose_points = [item[1] for index, item in enumerate(longest_branch) if index in local_min.tolist()]
    # stenose_rings = [item for index, item in enumerate(chosen_ring) if index in local_min.tolist()]
    # stenose_radius = [item for index, item in enumerate(radius) if index in local_min.tolist()]
    stenose_rings += [item for index, item in enumerate(chosen_ring)]
    stenose_radius = [item for index, item in enumerate(radius)]
    new_stenose_radius, valid_mask = side_stable_mask(np.array(stenose_radius), ratio_threshold=0.2, dif_thresh=0.1)
    x_values = [i*distance_threshold for i in range(len(radius))]

    plt.figure(figsize=(np.max(np.array(x_values)), np.max(radius)))
    plt.plot(x_values, np.array(radius), color="blue", label='Min distance to centerline')
    plt.plot(x_values, valid_mask, color="red", label='Stable interval')
    # plt.plot(x_values, normalize_array(np.array(max_distances)), label='Max distance to surface')
    # plt.plot(x_values, normalize_array(np.array(ex_surface_areas)), label='Ex Surface area')
    # plt.plot(x_values, normalize_array(np.array(in_surface_areas)), label='In Surface area')
    # plt.plot(x_values, np.array(is_stenoses).astype(int), marker='o', linestyle='-', label='Is stenose')

    # Adding labels and title
    plt.xlabel('Length (mm)')
    plt.title('Change in vascular geometry')
    plt.legend()  # Show legend with labels

    # Displaying the plot
    # plt.show()

    stable_ring_points = []
    refer_radius = np.min(new_stenose_radius[valid_mask == 1])

    print(refer_radius)
    stenosis_grades = {
        '3': [],
        '2': [],
        '1': [],
        '0': []
    }
    stenosis_colors = {
        '0': 'red',
        '1': 'orange',
        '2': 'yellow',
        '3': 'green'
    }

    for idx, value in enumerate(valid_mask):
        if value == 1:
            for item in stenose_rings[idx]:
                stable_ring_points.append(item[0])

        ratio = radius[idx]/refer_radius

        if ratio < 0.2:
            for item in stenose_rings[idx]:
                stenosis_grades['0'].append(item[0])
        elif ratio < 0.5:
            for item in stenose_rings[idx]:
                stenosis_grades['1'].append(item[0])
        elif ratio < 0.8:
            for item in stenose_rings[idx]:
                stenosis_grades['2'].append(item[0])
        else:
            for item in stenose_rings[idx]:
                stenosis_grades['3'].append(item[0])

    
    stenosis_ratios = [10000]*vmtk_boundary_vertices.shape[0]

    for stenose_ring in stenose_rings:
        for point in stenose_ring:
            idx = point[0]
            distance = euclidean_distance(vmtk_boundary_vertices[idx], point[1])
            if distance < stenosis_ratios[idx]:
                stenosis_ratios[idx] = distance
    
    # Build KDTree from array1
    kdtree = KDTree(smooth_points)
    distances, indices = kdtree.query(vmtk_boundary_vertices)

    for idx, value in enumerate(stenosis_ratios):
        if value == 10000:
            stenosis_ratios[idx] = distances[idx]

    stenosis_ratios = np.array(stenosis_ratios)/refer_radius
    stenosis_ratios[stenosis_ratios > 0.6] = 1

    # Initialize the color array with the same size as stenosis_ratios
    color_array = np.empty(stenosis_ratios.shape, dtype='<U6')

    # Assign colors based on the ratios
    color_array[stenosis_ratios < 0.2] = stenosis_colors['0']
    color_array[(stenosis_ratios >= 0.2) & (stenosis_ratios < 0.5)] = stenosis_colors['1']
    color_array[(stenosis_ratios >= 0.5) & (stenosis_ratios < 0.7)] = stenosis_colors['2']
    color_array[stenosis_ratios >= 0.7] = stenosis_colors['3']

    vmtk_boundary_vertices_all.append(vmtk_boundary_vertices)
    vmtk_boundary_faces_all.append(vmtk_boundary_faces + vert_num)
    stenosis_ratios_all.append(stenosis_ratios)
    vert_num += vmtk_boundary_vertices.shape[0]
    # stenose_ring_points = []
    # stenosis_indices = []
    # ex_surface_areas = []
    # in_surface_areas = []
    # is_stenoses = []
    # max_distances = []

    # for idx, ring in enumerate(stenose_rings):
    #     ring_vertex = [item[0] for item in ring]

    #     if len(ring_vertex) > 2:
    #         ring_vertices = vmtk_boundary_vertices[ring_vertex]
    #         pca = PCA(n_components=2)
    #         points_2d = pca.fit_transform(ring_vertices)
    #         # Calculate convex Hull
    #         ex_hull = ConvexHull(points_2d)
    #         ex_surface_area = area_of_polygon_from_edges(ex_hull.vertices, points_2d)
    #         ex_surface_areas.append(ex_surface_area)
    #         plt.plot(points_2d[:, 0], points_2d[:, 1], 'o')
    #         for simplex in ex_hull.simplices:
    #             plt.plot(points_2d[simplex, 0], points_2d[simplex, 1], 'k-')
    #         plt.fill(points_2d[ex_hull.vertices, 0], points_2d[ex_hull.vertices, 1], 'b', alpha=0.2)

    #         # Find maxmimum distance from interior points to convex hull
    #         interior_points = [idx for idx, point in enumerate(points_2d) if idx not in ex_hull.vertices]
    #         interior_pos = points_2d[interior_points]
    #         edges = [points_2d[item] for item in ex_hull.simplices]
    #         results = find_closest_edges(interior_pos, edges)
    #         max_distances.append(np.max(np.array(results)))
    #         max_pos = None
    #         is_stenose = False
    #         if np.max(np.array(results)) >= 0.4*stenose_radius[idx]:
    #             is_stenose = True
    #             max_idx = np.argmax(np.array(results))
    #             max_pos = interior_pos[max_idx]

    #             for item in stenose_rings[idx]:
    #                 stenose_ring_points.append(item[0])
    #         is_stenoses.append(is_stenose)

    #         # Find the interior hull
    #         in_hull, new_concave_points = find_interior_hull(points_2d)
    #         in_surface_area = area_of_polygon_from_edges(new_concave_points, points_2d)
    #         in_surface_areas.append(in_surface_area)

    #         for idx in range(new_concave_points.shape[0]):
    #             plt.plot(points_2d[new_concave_points[[idx-1, idx]], 0], points_2d[new_concave_points[[idx-1, idx]], 1], 'k-')
    #         plt.fill(points_2d[new_concave_points, 0], points_2d[new_concave_points, 1], 'r', alpha=0.2)

    #         # Plotting
    #         if is_stenose:
    #             print('Length (mm):', idx*distance_threshold, ', surface area:', ex_surface_area, ('mm2'), ', min distance to centerline:', radius[idx], 'mm', ', max distance to surface:', np.max(np.array(results)), 'mm', ', concave')
    #             plt.suptitle('Stenosis')
    #             stenosis_indices.append(idx)
    #             plt.scatter(max_pos[0], max_pos[1], color='red', s=50)
    #         else:
    #             print('Length (mm):', idx*distance_threshold, ', surface area:', ex_surface_area, ('mm2'), ', min distance to centerline:', radius[idx], 'mm', ', max distance to surface:', np.max(np.array(results)), 'mm', ', convex')
    #             plt.suptitle('Without stenosis')

    #         plt.show()

    #     else:
    #         ex_surface_areas.append(0)
    #         in_surface_areas.append(0)
    #         max_distances.append(0)
    #         is_stenoses.append(False)
    # # Plotting the first line
    # x_values = [i*distance_threshold for i in range(len(radius))]

    # plt.figure(figsize=(np.max(np.array(x_values)), np.max(radius)))
    # # plt.plot(x_values, normalize_array(np.array(radius)), label='Min distance to centerline')
    # # plt.plot(x_values, normalize_array(np.array(max_distances)), label='Max distance to surface')
    # # plt.plot(x_values, normalize_array(np.array(ex_surface_areas)), label='Ex Surface area')
    # # plt.plot(x_values, normalize_array(np.array(in_surface_areas)), label='In Surface area')
    # # plt.plot(x_values, np.array(is_stenoses).astype(int), marker='o', linestyle='-', label='Is stenose')

    # plt.plot(x_values, np.array(ex_surface_areas), label='Ex Surface area')
    # plt.plot(x_values, np.array(in_surface_areas), label='In Surface area')
    # plt.plot(x_values, np.array(is_stenoses).astype(int), marker='o', linestyle='-', label='Is stenose')

    # # Adding labels and title
    # plt.xlabel('Length (mm)')
    # plt.title('Change in vascular geometry')
    # plt.legend()  # Show legend with labels

    # # Displaying the plot
    # plt.show()

    # # stenosis_min = np.array([item for idx, item in enumerate(local_min.tolist()) if idx in stenosis_indices])
    # # radius_array = np.array(radius)
    # # x_values = [i*distance_threshold for i in range(len(radius))]
    # # # Plot the line graph
    # # plt.figure(figsize=(np.max(np.array(x_values)), np.max(radius_array)))
    # # plt.plot(x_values, radius, label='Radius')
    # # plt.scatter(local_min*distance_threshold, radius_array[local_min], color='red', zorder=5, label='Local Minima')
    # # plt.scatter(stenosis_min*distance_threshold, radius_array[stenosis_min], color='green', zorder=5, label='Stenosis point')
    # # plt.xlabel('Position')
    # # plt.ylabel('Radius')
    # # plt.title('Radius Values with Local Minima Highlighted')
    # # plt.legend()
    # # plt.show()

    #     # alpha_shape = alphashape.alphashape(points_2d,0.01)

    #     # # Plotting
    #     # fig, ax = plt.subplots()
    #     # ax.scatter(points_2d[:, 0], points_2d[:, 1], c='blue', marker='o')
    #     # ax.add_patch(PolygonPatch(alpha_shape, alpha=0.5))
    #     # ax.set_title('2D Projection of 3D Points using PCA')
    #     # ax.set_xlabel('Principal Component 1')
    #     # ax.set_ylabel('Principal Component 2')
    #     # ax.axis('equal')  # Ensure equal scaling for both axes

    #     # plt.show()

    # # print("Min radius at half the length: ", radius[int(len(longest_branch)/2)])
    
    
    # # max_branch_pos = [idx for idx, value in enumerate(splitted_branches) if value == longest_branch]
    # # ring_vertices = [ring for idx, ring in enumerate(ring_vertices) if idx in max_branch_pos]
    # # middle_ring = ring_vertices[int(len(ring_vertices)/2)]

    # # middle_surface_points = vmtk_boundary_vertices[[sublist[0] for sublist in middle_ring]]
    # # middle_intersection_points = intersection_points[int(len(intersection_points)/2)]

    # # sum_distance = []
    # # for idx, point in enumerate(middle_surface_points):
    # #     sum_distance.append(euclidean_distance(point, middle_intersection_points[idx]))

    # # mean_radius = sum(sum_distance)/len(sum_distance)
    # # print("Average radius at half the length: ", round(mean_radius, 2))
    
    visualized_boundary_points = generate_points(vmtk_boundary_vertices, 1, 'blue')
    visualized_smooth_points = generate_points(smooth_points, 3, 'red')
    visualized_stable_points = generate_points(vmtk_boundary_vertices[stable_ring_points], 3, 'red')
    # visualized_stenose_points = generate_points(np.array(stenose_points), 5, 'red')
    # visualized_stenose_ring_points = generate_points(vmtk_boundary_vertices[stenose_ring_points], 3, 'red')
    
    
    # line_traces = []
    # for line in smooth_connected_lines:
    #     print(len(line))
    #     line_traces.append(generate_lines(smooth_points[line], 2))


    # visualize_stenose_grades = []
    # for key in stenosis_grades:
    #     visualize_stenose_grades.append(generate_points(vmtk_boundary_vertices[stenosis_grades[key]], 3, stenosis_colors[key]))

    # showed_data.append(mesh)
    # showed_data.append(visualized_smooth_points)
    show_figure([visualized_boundary_points, visualized_stable_points, visualized_smooth_points], 'Stenosis grade along the extracted centerline of ICA'
        # + line_traces + visualize_stenose_grades
        )

vmtk_boundary_vertices_all = np.concatenate(vmtk_boundary_vertices_all, axis=0)
vmtk_boundary_faces_all = np.concatenate(vmtk_boundary_faces_all, axis=0)
stenosis_ratios_all = np.concatenate(stenosis_ratios_all, axis=0)

print()

mesh = generate_mesh_color(vmtk_boundary_vertices_all, vmtk_boundary_faces_all, stenosis_ratios_all)
showed_data.append(mesh)
show_figure(showed_data, 'Stenosis grade along the extracted centerline of ICA'
# + line_traces + visualize_stenose_grades
)