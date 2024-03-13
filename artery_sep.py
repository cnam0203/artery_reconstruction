import nibabel as nib
from skimage import measure
import open3d as o3d
import numpy as np
import plotly.graph_objs as go
from skimage.morphology import skeletonize, thin
from scipy.spatial import KDTree
from scipy.spatial import distance_matrix
import math
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
import time
import os
from scipy import ndimage
import json
import heapq
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2


def find_intersection_point(point1, point2, slice_coord, axis):
    # Convert points to arrays
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # Calculate direction vector of the line connecting the two points
    direction_vector = point2 - point1
    
    # Calculate the parameter t for the intersection point
    t = (slice_coord[axis] - point1[axis]) / direction_vector[axis]
    
    # Calculate the intersection point
    intersection_point = point1 + t * direction_vector
    intersection_point = np.round(intersection_point).astype(int)
    
    return intersection_point.tolist()

def find_square_region(cube_size, intersection_point, square_edge, axis, i):
    # Calculate the boundaries of the square region
    half_edge = square_edge // 2
    min_boundaries = np.maximum(intersection_point - half_edge, [0, 0, 0])
    max_boundaries = np.minimum(intersection_point + half_edge, cube_size)
    
    # Create a mask to select the square region
    mask = np.zeros(cube_size, dtype=bool)

    if axis == 0:
        mask[i, min_boundaries[1]:max_boundaries[1], min_boundaries[2]:max_boundaries[2]] = True
    elif axis == 1:
        mask[min_boundaries[0]:max_boundaries[0], i, min_boundaries[2]:max_boundaries[2]] = True
    else:
        mask[min_boundaries[0]:max_boundaries[0], min_boundaries[1]:max_boundaries[1], i] = True

    return mask

def select_slices(cube_size, point1, point2, square_edge):
    # Convert points to arrays
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate direction vector of the line connecting the two points
    direction_vectors = point2 - point1
    axe_slices = []
    increment = 1

    for index, vect in enumerate(direction_vectors):
        if vect != 0:
            # Determine the slices along the axis of the line
            start_point = min(point1[index], point2[index])
            end_point = max(point1[index], point2[index])
            perpendicular_slices = []

            for i in range(start_point, end_point + increment, increment):
                intersection_point = find_intersection_point(point1, point2, [i, i, i], index)
                print(index, intersection_point)
                # Find the square region centered at the intersection point
                square_region = find_square_region(cube_size, np.array(intersection_point), square_edge, index, i)
                # Append slice and square region to the list
                perpendicular_slices.append(square_region)
            
            axe_slices.append({
                'axis': index,
                'slices': perpendicular_slices,
                'head_points': [start_point, end_point]
            })
    
    return axe_slices


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

def remove_noisy_voxels(voxels):
    # Define a 3x3x3 kernel to check the 26 neighbor_distances
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0

    # Count the number of non-zero neighbor_distances for each voxel
    neighbor_counts = np.zeros_like(voxels)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i == 0 and j == 0 and k == 0:
                    continue
                neighbor_counts += np.roll(voxels, (i, j, k), axis=(0, 1, 2))

    # Apply the condition to remove noisy voxels
    result = np.where((voxels == 1) & (neighbor_counts < 5), 0, voxels)

    return result

def dfs_with_vertex_count(graph, current_node, destination, visited, path):
    visited[current_node] = True
    path.append(current_node)

    if current_node == destination:
        return path

    max_path = path
    max_vertices = 0

    for neighbor in graph[current_node]:
        if (graph[current_node][neighbor] > 0 or graph[neighbor][current_node] > 0) and not visited[neighbor]:
            new_path = dfs_with_vertex_count(graph, neighbor, destination, visited.copy(), path.copy())
            if len(new_path) > len(max_path) and new_path[-1] == destination:
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

def visualize_original_points(original_datas=[], auto_colors=[], sizes=[]):
    data = []

    for index, original_data in enumerate(original_datas):
        if original_data.ndim == 2:
            points = original_data
            color_values = auto_colors[index]
            point_size = sizes[index]
        else:
            selected_data = np.copy(original_data)
            points = np.argwhere(selected_data != 0)
            color_values = selected_data[selected_data != 0]
            sizes = 1

        # Create traces
        # Visualize artery points 
        point_trace = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                colorscale='Viridis',
                color=color_values
            ),
            name='Points'
        )

        data.append(point_trace)

    # Create layout
    layout = go.Layout(
        scene=dict(
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1, y=1, z=1)
            )
        ),
        height=1200,  # Set height to 800 pixels
        width=2000   # Set width to 1200 pixels
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

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

def are_symmetric(point_1, point_2, centerpoint):
    # Calculate the distances from each point to the centerpoint
    distance_1 = np.linalg.norm(np.array(point_1) - np.array(centerpoint))
    distance_2 = np.linalg.norm(np.array(point_2) - np.array(centerpoint))
    
    # Check if the distances are equal
    if np.isclose(distance_1, distance_2):
        # Calculate the vector from the centerpoint to each point
        vector_1 = np.array(point_1) - np.array(centerpoint)
        vector_2 = np.array(point_2) - np.array(centerpoint)

        if np.all(np.isclose(np.cross(vector_1, vector_2), 0)):
            return True
    
    return False

def sum_vectors(vectors):
    if not vectors:
        return None  # Handle empty list case
    sum_vector = [0] * len(vectors[0])  # Initialize sum_vector with zeros
    for vector in vectors:
        sum_vector = [sum(x) for x in zip(sum_vector, vector)]  # Sum corresponding components
    return sum_vector

def find_choosen_points(artery_data, i, j, k):
    list_points = []

    for x in [-1, 1]:
        ni = i + x
        if 0 <= ni < artery_data.shape[0] and artery_data[ni][j][k] == 1:
            list_points.append((-x, 0, 0))

    for y in [-1, 1]:
        nj = j + y
        if 0 <= nj < artery_data.shape[1] and artery_data[i][nj][k] == 1:
            list_points.append((0, -y, 0))

    for z in [-1, 1]:
        nk = k + z
        if 0 <= nk < artery_data.shape[2] and artery_data[i][j][nk] == 1:
            list_points.append((0, 0, -z))

    return sum_vectors(list_points)

def can_create_zero_vector(vector1s, vector2s):
    for v1 in vector1s:
        for v2 in vector2s:
            if [v1[i] + v2[i] for i in range(3)] == [0, 0, 0]:
                return True
    return False

def find_nearest_point(artery_data, i, j, k, distance_threshold):
    count_x = 0
    index_x = -2
    count_y = 0
    index_y = -2
    count_z = 0
    index_z = -2

    for x in [-1, 1]:
        ni = i + x
        if 0 <= ni < artery_data.shape[0] and artery_data[ni][j][k] > 0:
            count_x += 1

            if count_x == 2:
                if abs(artery_data[ni][j][k] - index_x) >= distance_threshold:
                    index_x = -1
                else:
                    index_x = artery_data[ni][j][k]
            else:
                index_x = artery_data[ni][j][k]

    for y in [-1, 1]:
        nj = j + y
        if 0 <= nj < artery_data.shape[1] and artery_data[i][nj][k] > 0:
            count_y += 1

            if count_y == 2:
                if abs(artery_data[i][nj][k] - index_y) >= distance_threshold:
                    index_y = -1
                else:
                    index_y = artery_data[i][nj][k]
            else:
                index_y = artery_data[i][nj][k]
    
    for z in [-1, 1]:
        nk = k + z
        if 0 <= nk < artery_data.shape[2] and artery_data[i][j][nk] > 0:
            count_z += 1

            if count_z == 2:
                if abs(artery_data[i][j][nk] - index_z) >= distance_threshold:
                    index_z = -1
                else:
                    index_z = artery_data[i][j][nk]
            else:
                index_z = artery_data[i][j][nk]

    if (index_x == -1 or index_y == -1 or index_z == -1):
        return -1
    
    if (index_x != -2): return index_x
    if (index_y != -2): return index_y
    if (index_z != -2): return index_z

    return 0

# Define a function to find neighbors
def find_distant_neighbors(data, i, j, k, threshold):
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == dj == dk == 0:
                    continue  # Skip the current point
                ni, nj, nk = i + di, j + dj, k + dk
                if 0 <= ni < data.shape[0] and 0 <= nj < data.shape[1] and 0 <= nk < data.shape[2]:
                    if (data[ni][nj][nk] > 0):
                        value_1 = data[i][j][k]
                        value_2 = data[ni][nj][nk]
                        if abs(value_1 - value_2) >= threshold:
                            return True

    return False

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

    point_trace = go.Scatter3d(
        x=artery_points[:, 0],
        y=artery_points[:, 1],
        z=artery_points[:, 2],
        mode='markers',
        marker=dict(
            symbol='circle',  # Set marker symbol to 'cube'
            size=5,
            color=artery_values,  # Color based on the values
            colorscale='Viridis',  # Colormap
        ),
        text=[f'Value: {val}' for val in artery_values],
        hoverinfo='text'
    )

    # Create layout
    layout = go.Layout(
        scene=dict(
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1, y=1, z=1)
            )
        ),
        height=800,  # Set height to 800 pixels
        width=1200   # Set width to 1200 pixels
    )
     
    fig = go.Figure(data=[point_trace], layout=layout)
    fig.show()
    
    print('Number extension loops: ', loop)
    print('Number kissing points:', touch_points.shape[0])

    return touch_points 

def find_intensity_threshold(preprocessed_data, change_interval=0):
    bins = np.arange(0.1, 1.1, 0.1)
    hist, _ = np.histogram(preprocessed_data, bins=bins)
    intensity_threshold_1 = round(np.median(preprocessed_data[preprocessed_data!=0]) + change_interval, 1) + change_interval
    return intensity_threshold_1

def remove_cycle_edges(skeleton_points, neighbor_distances, v2):
    m = len(neighbor_distances)
    remove_edges = []
    
    # Iterate through each point
    for i in range(m):
        for j in range(i+1, m):
            for k in range(j+1, m):
                # Check if i, j, k form a cycle
                if (neighbor_distances[i][j] > 0 or neighbor_distances[j][i] > 0) and (neighbor_distances[j][k] > 0 or neighbor_distances[k][j] > 0) and (neighbor_distances[k][i] > 0 or neighbor_distances[i][k] > 0):
                    vt_1 = skeleton_points[i] - skeleton_points[j]
                    vt_2 = skeleton_points[j] - skeleton_points[k]
                    vt_3 = skeleton_points[i] - skeleton_points[k]

                    angle_1 = find_angle(vt_1, v2)
                    angle_1 = min(angle_1, 180 - angle_1)

                    angle_2 = find_angle(vt_2, v2)
                    angle_2 = min(angle_2, 180 - angle_2)

                    angle_3 = find_angle(vt_3, v2)
                    angle_3 = find_angle(angle_3, 180 - angle_3)

                    min_angle = min(angle_1, angle_2, angle_3)
                    
                    if min_angle == angle_1:
                        neighbor_distances[i][j] = 0
                        neighbor_distances[j][i] = 0
                        remove_edges.append([i, j])
                    elif min_angle == angle_2:
                        neighbor_distances[j][k] = 0
                        neighbor_distances[k][j] = 0
                        remove_edges.append([j, k])
                    else:
                        neighbor_distances[i][k] = 0
                        neighbor_distances[k][i] = 0
                        remove_edges.append([i, k])

    junction_points = refine_junction_points(skeleton_points, neighbor_distances)
    return junction_points, neighbor_distances, remove_edges

def refine_junction_points(skeleton_points, neighbor_distances):
    junction_points = []

    for i in range(skeleton_points.shape[0]):
        count = 0
        for j in range(skeleton_points.shape[0]):
            if (i != j):
                if (neighbor_distances[i][j] > 0 or neighbor_distances[j][i] > 0):
                    count += 1
        
        if count >= 3:
            junction_points.append(i)

    return junction_points

def remove_junction_points(neighbor_distances, junction_points, skeleton_points):
    removed_edges = []
    for i in range (len(junction_points)):
        for j in range (i+1, len(junction_points)): 
            if i != j:
                pidx_1 = junction_points[i]
                pidx_2 = junction_points[j]
                if neighbor_distances[pidx_1][pidx_2] or neighbor_distances[pidx_2][pidx_1]:
                    count_1 = 0
                    count_2 = 0

                    for m in range(len(junction_points)):
                        pidx_3 = junction_points[m]

                        if m != i and (neighbor_distances[pidx_1][pidx_3] or neighbor_distances[pidx_3][pidx_1]):
                            count_1 += 1
                        if m != j and (neighbor_distances[pidx_2][pidx_3] or neighbor_distances[pidx_3][pidx_2]):
                            count_2 += 1
                        
                    if count_1 == 1 or count_2 == 1:
                        removed_edges.append([pidx_1, pidx_2])

    for edge in removed_edges:
        point_1 = edge[0]
        point_2 = edge[1]
        neighbor_distances[point_1][point_2] = 0
        neighbor_distances[point_2][point_1] = 0

    junction_points = refine_junction_points(skeleton_points, neighbor_distances)

    return junction_points, neighbor_distances

def merge_lines(index, point_1, point_2, lines, junction_points, end_points, skeleton_points):
    connected_lines = copy.deepcopy(lines)
    list_1s = []
    list_1_ids = []
    list_1 = []
    pos_1 = -1

    list_2s = []
    list_2_ids = []
    list_2 = []
    pos_2 = -1

    list_1_id = -1
    list_2_id = -1


    for idx, line in enumerate(connected_lines):
        if (index in line) and (point_1 in line) and (point_2 in line):
            return connected_lines
        if (line[0] in junction_points or line[0] in end_points) and (line[-1] in junction_points or line[-1] in end_points) and (index in line):
            return connected_lines
        if (line[0] in junction_points or line[0] in end_points) and (line[-1] in junction_points or line[-1] in end_points):
            continue
        if point_1 == line[0] or point_1 == line[-1] or ((index == line[0] or index == line[-1]) and point_1 in line):
            if not ((point_1 == line[0]) and (point_1 in junction_points) and (index != line[-1])) and not ((point_1 == line[-1]) and (point_1 in junction_points) and (index != line[0])):
                list_1s.append(line)
                list_1_ids.append(idx)
        if point_2 == line[0] or point_2 == line[-1]  or ((index == line[0] or index == line[-1]) and point_2 in line):
            if not ((point_2 == line[0]) and (point_2 in junction_points) and (index != line[-1])) and not ((point_2 == line[-1]) and (point_2 in junction_points) and (index != line[0])):
                list_2s.append(line)
                list_2_ids.append(idx)
    
    if len(list_1s) == 0:
        list_1 = [point_1]
    else:
        exist_1 = False
        count = 0
        while(not exist_1) and count < len(list_1s):
            list_1 = copy.deepcopy(list_1s[count])
            list_1_id = list_1_ids[count]
            
            if index == list_1[0] or index == list_1[-1]:
                exist_1 = True

            if point_1 == list_1[-1]:
                list_1 = list_1[::-1]

            if exist_1:
                if index == list_1[0]:
                    list_1 = list_1[::-1]
                list_1.remove(index)
            else:
                list_1 = list_1[::-1]

            count += 1

    if len(list_2s) == 0:
        list_2 = [point_2]
    else:
        exist_2 = False
        count = 0
        while (not exist_2) and count < len(list_2s):
            list_2 = copy.deepcopy(list_2s[count])
            list_2_id = list_2_ids[count]
            
            if index == list_2[0] or index == list_2[-1]:
                exist_2 = True

            if point_2 == list_2[0]:
                list_2 = list_2[::-1]

            if exist_2:
                if index == list_2[-1]:
                    list_2 = list_2[::-1]
                list_2.remove(index)
            else:
                list_2 = list_2[::-1]
                
            count += 1

    indices_to_remove = []

    if list_1_id != -1:
        indices_to_remove.append(list_1_id)
    if list_2_id != -1:
        indices_to_remove.append(list_2_id)

    for i in sorted(indices_to_remove, reverse=True):
        del connected_lines[i]

    new_line = list_1 + [index] + list_2
    connected_lines.append(new_line)
        
    # print('[', index, point_1 , point_2, ']', list_1, list_2, new_line)

    return connected_lines

def reduce_skeleton_points(neighbor_distances, junction_points, end_points, skeleton_points):
    is_removed = True
    head_points = {}
    new_neighbor_distances = copy.deepcopy(neighbor_distances)
    connected_lines = []

    for i in range(len(neighbor_distances)):
        for j in range(i+1, len(neighbor_distances)):
            if (neighbor_distances[i][j] or neighbor_distances[j][i]) and (i in junction_points) and (j in junction_points):
                connected_lines.append([i, j])
            
    while is_removed:
        is_removed = False

        for index, point in enumerate(skeleton_points):
            if (index not in junction_points) and (index not in end_points):
                list_neighbors = []

                for j in range(len(new_neighbor_distances)):
                    if new_neighbor_distances[index][j] > 0 or new_neighbor_distances[j][index] > 0:
                        list_neighbors.append(j)

                for idx1 in range(len(list_neighbors)):
                    for idx2 in range(idx1+1, len(list_neighbors)):
                        point_1 = list_neighbors[idx1]
                        point_2 = list_neighbors[idx2]
                        # if point_1 != point_2:
                        if point_1 != point_2 and (point_1 in junction_points and point_2 in junction_points):
                            if point_1 not in head_points:
                                head_points[point_1] = {}
                            if point_2 not in head_points[point_1]:
                                head_points[point_1][point_2] = []
                            
                            head_points[point_1][point_2].append(index)
                        elif point_1 != point_2 and (point_1 not in junction_points or point_2 not in junction_points):
                            is_removed = True
                            new_neighbor_distances[index][point_1] = 0
                            new_neighbor_distances[index][point_2] = 0
                            new_neighbor_distances[point_1][index] = 0
                            new_neighbor_distances[point_2][index] = 0

                            pos_1 = skeleton_points[point_1]
                            pos_2 = skeleton_points[point_2]
                            distance = (pos_1[0] - pos_2[0])**2 + (pos_1[1] - pos_2[1])**2 + (pos_1[2] - pos_2[2])**2
                            new_neighbor_distances[point_1][point_2] = distance
                            new_neighbor_distances[point_2][point_1] = distance

                            connected_lines = merge_lines(index, point_1, point_2, connected_lines, junction_points, end_points, skeleton_points)
               
    for point_1 in head_points:
        for point_2 in head_points[point_1]:
            list_points = head_points[point_1][point_2]
            unique_points = list(set(list_points))
            if len(unique_points) == 1:
                index = unique_points[0]
                new_neighbor_distances[index][point_1] = 0
                new_neighbor_distances[index][point_2] = 0
                new_neighbor_distances[point_1][index] = 0
                new_neighbor_distances[point_2][index] = 0
                pos_1 = skeleton_points[point_1]
                pos_2 = skeleton_points[point_2]
                distance = (pos_1[0] - pos_2[0])**2 + (pos_1[1] - pos_2[1])**2 + (pos_1[2] - pos_2[2])**2
                new_neighbor_distances[point_1][point_2] = distance
                new_neighbor_distances[point_2][point_1] = distance

                connected_lines = merge_lines(index, point_1, point_2, connected_lines, junction_points, end_points, skeleton_points)

    # print(junction_points)
    # print(end_points)

    junction_points = refine_junction_points(skeleton_points, new_neighbor_distances)
    return junction_points, new_neighbor_distances, connected_lines

def find_graph(skeleton_points):
    # Check which points lie below the plane defined by each point and its closest point
    end_points = []
    junction_points = []
    neighbor_distances = {}

    # Initialize neighbor distances for all points
    for i, point in enumerate(skeleton_points):
        neighbor_distances[i] = {}
        for j, other_point in enumerate(skeleton_points):
            neighbor_distances[i][j] = 0

    # Finding 3 neighbor points for each point based on Euclidean distance
    for i, point in enumerate(skeleton_points):
        closest_point_1 = None
        min_dist_1 = 1000000
        closest_point_2 = None
        min_dist_2 = 1000000
        closest_point_3 = None
        min_dist_3 = 1000000
        
        for j, other_point in enumerate(skeleton_points):
            if i != j:
                dist = (point[0]-other_point[0])**2 + (point[1]-other_point[1])**2 + (point[2]-other_point[2])**2
                if (dist < min_dist_1):
                    min_dist_3 = min_dist_2
                    closest_point_3 = closest_point_2
                    min_dist_2 = min_dist_1
                    closest_point_2 = closest_point_1
                    min_dist_1 = dist
                    closest_point_1 = j
                elif (dist < min_dist_2):
                    min_dist_3 = min_dist_2
                    closest_point_3 = closest_point_2
                    min_dist_2 = dist
                    closest_point_2 = j
                elif (dist < min_dist_3):
                    min_dist_3 = dist
                    closest_point_3 = j
                    
        if closest_point_2 == None:
            min_dist_2 = min_dist_1
            closest_point_2 = closest_point_1
        
        if closest_point_3 == None:
            min_dist_3 = min_dist_2
            closest_point_3 = closest_point_2

        # Intialize normal vectors for each point with its 3 neighbors
        vector11 = skeleton_points[closest_point_1] - point
        vector12 = skeleton_points[closest_point_2] - point
        vector13 = skeleton_points[closest_point_3] - point
        
        is_exist_1 = False
        is_exist_2 = False
        
        # Check end point
        for j, other_point in enumerate(skeleton_points):
            # Calculate the dot product of the two vectors
            if (i != j):
                vector2 = other_point - point
                # Calculate the angle in radians
                
                angle_degrees = find_angle(vector11, vector2)
                dist_2 = (point[0]-other_point[0])**2 + (point[1]-other_point[1])**2 + (point[2]-other_point[2])**2
                
                if angle_degrees > 90 and math.sqrt(dist_2) < math.sqrt(min_dist_1)*2.5:
                    is_exist_1 = True
                    break
                    
        for j, other_point in enumerate(skeleton_points):
            # Calculate the dot product of the two vectors
            if (i != j):
                vector2 = other_point - point
    
                # Convert radians to degrees and take the absolute value
                angle_degrees = find_angle(vector12, vector2)
                dist_2 = (point[0]-other_point[0])**2 + (point[1]-other_point[1])**2 + (point[2]-other_point[2])**2
                
                if angle_degrees > 90 and math.sqrt(dist_2) < math.sqrt(min_dist_2)*2.5:
                    is_exist_2 = True
                    break
        
        if not is_exist_1 and not is_exist_2: # A point is considered an endpoint when there is no points below the normal planes
            end_points.append(i)
            neighbor_distances[i][closest_point_1] = min_dist_1
        else:   # If not, check whether this points is in the junction
            angle_1 = find_angle(vector11, vector12)
            angle_2 = find_angle(vector12, vector13)
            angle_3 = find_angle(vector11, vector13)
            
            angle_threshold = 70
            
            # Check junction point
            if (angle_1 >= angle_threshold and angle_2 > angle_threshold and angle_3 > angle_threshold):
                junction_points.append(i)
                neighbor_distances[i][closest_point_1] = min_dist_1
                neighbor_distances[i][closest_point_2] = min_dist_2
                neighbor_distances[i][closest_point_3] = min_dist_3
            else:
                neighbor_distances[i][closest_point_1] = min_dist_1
                neighbor_distances[i][closest_point_2] = min_dist_2

    return end_points, junction_points, neighbor_distances

def dijkstra(edges, start, end, skeleton_points, direction):
    graph = {}
    for edge in edges:
        src = edge[0]
        dest = edge[1]

        if src not in graph:
            graph[src] = []
        if dest not in graph:
            graph[dest] = []
        graph[src].append(dest)
        graph[dest].append(src)

    pq = [(0, start, [], None)]  # Added None as the current edge for the initial node
    visited = set()

    while pq:
        (cost, node, path, current_edge) = heapq.heappop(pq)
        if node not in visited:
            visited.add(node)
            path = path + [node]
            if node == end:
                return cost, path
            for neighbor in graph[node]:
                if neighbor not in visited:
                    new_edge = (node, neighbor)
                    new_cost = cost + calculate_edge_cost(current_edge, new_edge, skeleton_points, direction)
                    heapq.heappush(pq, (new_cost, neighbor, path, new_edge))
    return float('inf'), []

def calculate_edge_cost(current_edge, new_edge, nodes_positions, direction):
    if current_edge is None:
        return 0  # Initial edge cost
    else:
        # Calculate the angle between the current edge and the new edge
        current_vector = calculate_vector(nodes_positions[current_edge[0]], nodes_positions[current_edge[1]])
        new_vector = calculate_vector(nodes_positions[current_edge[0]], nodes_positions[new_edge[1]])
        angle = calculate_angle(current_vector, new_vector)

        if direction == 'left':
            return angle
        elif direction == 'right':
            return -angle

def calculate_vector(point1, point2):
    return (point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2])

def calculate_angle(vector1, vector2):
    dot_product = sum(x * y for x, y in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(x**2 for x in vector1))
    magnitude2 = math.sqrt(sum(x**2 for x in vector2))
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    return math.acos(cosine_angle)  # Angle in radians

def list_not_in_lists(tuple_to_check, list_of_tuples):
    for tup in list_of_tuples:
        if tup == tuple_to_check:
            return False
    return True

def find_rest_paths(edges, points, trunk):
    visited_points = copy.deepcopy(points)
    count = 1
    new_paths = []

    while count:
        count = 0
        for edge in edges:
            point_1 = edge[0]
            point_2 = edge[1]

            if ((point_1 in visited_points) and (point_2 not in visited_points) and list_not_in_lists(edge, trunk)) or ((point_1 not in visited_points) and list_not_in_lists(edge, trunk) and (point_2 in visited_points)):
                count += 1
                new_paths.append(edge)
                visited_points.append(point_1)
                visited_points.append(point_2)

    return new_paths

def split_direction_2(principal_vectors, other_vectors):
    left_vectors = []
    right_vectors = []

    a11 = find_angle(principal_vectors[0], other_vectors[0])
    a12 = find_angle(principal_vectors[0], other_vectors[1])
    a21 = find_angle(principal_vectors[1], other_vectors[0])
    a22 = find_angle(principal_vectors[1], other_vectors[1])
    min_angle = min(a11, a12, a21, a22)

    if min_angle == a11 or min_angle == a22:
        return 1, 0
    else:
        return 0, 1
    
def split_direction_1(principal_vector, other_vectors):
    left_vectors = []
    right_vectors = []

    a11 = find_angle(principal_vector, other_vectors[0])
    a12 = find_angle(principal_vector, other_vectors[1])
    min_angle = min(a11, a12)

    if min_angle == a11:
        return 1, 0
    else:
        return 0, 1

def find_neighbors(point, left_paths, right_paths, connected_lines, stop_left_paths, stop_right_paths):
    left_point = None
    right_point = None

    left_path_index = None
    left_connected_index = None
    right_path_index = None
    right_connected_index = None

    for i, path in enumerate(left_paths):
        if (path[0] == point or path[-1] == point) and (i not in stop_left_paths):
            for j, connected_line in enumerate(connected_lines):
                if (path[0] in connected_line and path[-1] in connected_line):
                    if point == connected_line[0]:
                        left_point = connected_line[1]
                        left_path_index = i
                        left_connected_index = j
                        break
                    elif point == connected_line[-1]:
                        left_point = connected_line[-2]
                        left_path_index = i
                        left_connected_index = j
                        break

    for i, path in enumerate(right_paths):
        if (path[0] == point or path[-1] == point) and (i not in stop_right_paths):
            for j, connected_line in enumerate(connected_lines):
                if (path[0] in connected_line and path[-1] in connected_line):
                    if point == connected_line[0]:
                        right_point = connected_line[1]
                        right_path_index = i
                        right_connected_index = j
                        break
                    elif point == connected_line[-1]:
                        right_point = connected_line[-2]
                        right_path_index = i
                        right_connected_index = j
                        break

    if left_point and right_point:
        return [{
            'point': left_point,
            'path': left_path_index,
            'connected_line': left_connected_index
            }, {
            'point': right_point,
            'path': right_path_index,
            'connected_line': right_connected_index
            }]
    else:
        return None

def find_p4_p5(p1, p2, p3):
    # Compute direction vectors
    p1p2 = np.array(p2) - np.array(p1)
    p2p1 = -p1p2
    p3p4 = p1p2
    p3p5 = p2p1
    # Normalize direction vectors
    p3p4_normalized = p3p4 / np.linalg.norm(p3p4)
    p3p5_normalized = p3p5 / np.linalg.norm(p3p5)

    # Compute positions of p4 and p5
    p5 = np.array(p3) + p3p4_normalized
    p4 = np.array(p3) + p3p5_normalized

    return p4, p5

def interpolate_center_path(line1, line2, num_points):
    # Get the shorter and longer lines
    shorter_line = line1 if len(line1) < len(line2) else line2
    longer_line = line1 if len(line1) >= len(line2) else line2

    # Interpolate points between corresponding points on the shorter line and their corresponding points on the longer line
    center_path = []
    for i in range(len(shorter_line)):
        # Linear interpolation between points
        interpolated_points = np.linspace(shorter_line[i], longer_line[i], num_points + 2)
        interpolated_points = np.around(interpolated_points).astype(int)[1:-1]
        center_path.extend(interpolated_points)

    # If the longer line has more points, interpolate remaining points
    if len(longer_line) > len(shorter_line):
        remaining_points = longer_line[len(shorter_line):]
        interpolated_points = np.linspace(remaining_points[0], remaining_points[-1], num_points + 2)
        interpolated_points = np.around(interpolated_points).astype(int)[1:-1]
        center_path.extend(interpolated_points)

    return center_path

def select_slice(preprocessed_data, index, axis, min_coords, max_coords):
    # Select the slice along the specified axis
    if axis == 0:
        slice_data = preprocessed_data[index, min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
    elif axis == 1:
        slice_data = preprocessed_data[ min_coords[0]:max_coords[0], index, min_coords[2]:max_coords[2] ]
    else:
        slice_data = preprocessed_data[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], index]
    
    return slice_data

def find_max_count_positions(max_count):
    max_count_positions = []
    max_count_value = np.max(max_count)

    # # Iterate over each position in the max_count array
    # for x1 in range(max_count.shape[0]):
    #     for y1 in range(max_count.shape[1]):
    #         # Check if the count at this position is equal to the maximum count
    #         if max_count[x1][y1] == max_count_value:
    #             # Iterate over neighboring positions
    #             for x2 in range(max(0, x1-1), min(max_count.shape[0], x1+2)):
    #                 for y2 in range(max(0, y1-1), min(max_count.shape[1], y1+2)):
    #                     # Check if the distance between the positions is greater than 1
    #                     if np.sqrt((x1 - x2)**2 + (y1 - y2)**2) > 1:
    #                         max_count_positions.append(((x1, y1), (x2, y2)))

    print(max_count_value)
    return max_count_positions

def auto_select_slices(cube_size, point1, point2, square_edge):
    # Convert points to arrays
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate direction vector of the line connecting the two points
    direction_vectors = np.abs(point2 - point1)
    axis = np.argmax(direction_vectors)

    increment = 1

    # Determine the slices along the axis of the line
    start_point = min(point1[axis], point2[axis])
    end_point = max(point1[axis], point2[axis])
    perpendicular_slices = []

    for i in range(start_point, end_point + increment, increment):
        intersection_point = find_intersection_point(point1, point2, [i, i, i], axis)
        print(axis, intersection_point)
        # Find the square region centered at the intersection point
        square_region = find_square_region(cube_size, np.array(intersection_point), square_edge, axis, i)
        # Append slice and square region to the list
        perpendicular_slices.append(square_region)
    
    return {
        'axis': axis,
        'slices': perpendicular_slices,
        'head_points': [start_point, end_point]
    }

def find_skeleton(segment_image,
                  original_image=None, 
                  index=[], 
                  intensity_threshold_1=0.65, 
                  intensity_threshold_2=0.65, 
                  gaussian_sigma=0, 
                  distance_threshold=20,
                  laplacian_iter=1,
                  folder_path=''):
    
    original_data = original_image.get_fdata()
    voxel_sizes = segment_image.header.get_zooms()
    print('Voxel size: ', voxel_sizes)
    print("Image size:", original_data.shape)

    preprocessed_data = np.copy(original_data)
    preprocessed_data = gaussian_filter(preprocessed_data, sigma=gaussian_sigma)
    
    segment_data = segment_image.get_fdata()
    mask_data = np.copy(segment_data)

    mask = np.isin(mask_data, index, invert=True) 
    mask_data[mask] = 0 
    mask_data[mask_data != 0] = 1
    
    min_value = np.min(preprocessed_data[mask_data != 1]) - 1

    preprocessed_data[mask_data != 1] = min_value
    preprocessed_data = (preprocessed_data - np.min(preprocessed_data)) / (np.max(preprocessed_data) - np.min(preprocessed_data))
    selected_data = np.copy(preprocessed_data)

    if intensity_threshold_1 == 0:
        intensity_threshold_1 = find_intensity_threshold(preprocessed_data, 0.05)

    if intensity_threshold_2 == 0:
        intensity_threshold_2 = intensity_threshold_1

    selected_data[selected_data < intensity_threshold_1] = 0
    selected_data[selected_data >= intensity_threshold_1] = 1
    
    selected_data = remove_noisy_voxels(selected_data)
    labeled_array, num_labels = ndimage.label(selected_data)
    labels, counts = np.unique(labeled_array[labeled_array != 0], return_counts=True)
    max_count_label = labels[np.argmax(counts)]
    selected_data[(labeled_array != 0) & (labeled_array != max_count_label)] = 0
    verts, faces, normals, _ = measure.marching_cubes(selected_data, level=0.5, spacing=[1, 1, 1])

    artery_points_old = np.argwhere(mask_data != 0)
    artery_points = np.argwhere(mask_data != 0)*voxel_sizes

    skeleton = skeletonize(selected_data)
    skeleton_points = np.argwhere(skeleton != 0)

    end_points, junction_points, neighbor_distances = find_graph(skeleton_points)
    lines = []
    line_values = []
    line_colors = []

    if 5 in index or 6 in index:
        point_1 = [141, 201, 287]
        point_2 = [138, 216, 315]
        differences = np.abs(np.array(point_1) - np.array(point_2))
        cube_size = selected_data.shape

        result = auto_select_slices(cube_size, point_1, point_2, 14)
        axis = result['axis']

        if axis == 0:
            axis_name = 'X'
        elif axis == 1:
            axis_name = 'Y'
        else:
            axis_name = 'Z'

        start_point, end_point = result['head_points']
        slices = result['slices']
        num_cols = 3
        num_rows = math.ceil(len(slices)/3)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 15))


        for index, slice in enumerate(slices):
            center_points = []
            boundaries = np.argwhere(slice==True)
            if boundaries.shape[0]:
                min_coords = np.min(np.argwhere(slice==True), axis=0)
                max_coords = np.max(np.argwhere(slice==True), axis=0)
                segment_slice = select_slice(mask_data, start_point + index, axis, min_coords, max_coords)
                intensity_slice = select_slice(original_data, start_point + index, axis, min_coords, max_coords)
                fixed_slice = select_slice(selected_data, start_point + index, axis, min_coords, max_coords)

                normalized_data = (intensity_slice - intensity_slice.min()) / (intensity_slice.max() - intensity_slice.min())
                offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]

                # Iterate over each pixel
                max_count = np.zeros((normalized_data.shape[0], normalized_data.shape[1])) 
                padded_data = np.pad(normalized_data, 1, mode='constant')

                for i in range(1, normalized_data.shape[0] + 1):
                    for j in range(1, normalized_data.shape[1] + 1):
                        # Get the values of the pixel and its neighbors from the padded array
                        neighbors_values = [padded_data[i + ni, j + nj] for ni, nj in offsets]
                        # Find the index of the neighbor with the highest value
                        max_neighbor_index = np.argmax(neighbors_values)
                        max_neighbor_position = (i + offsets[max_neighbor_index][0], j + offsets[max_neighbor_index][1])
                        
                        x_pos = max_neighbor_position[0] - 1  # Adjust for padding
                        y_pos = max_neighbor_position[1] - 1  # Adjust for padding
                        
                        if normalized_data[x_pos][y_pos] > 0.4:
                            max_count[x_pos][y_pos] += 1
                    
                    max_indices = np.argsort(max_count, axis=None)[-2:]
                    max_indices_2d = np.unravel_index(max_indices, max_count.shape)
                    # max_positions_pairs = list(zip(max_indices_2d[0], max_indices_2d[1]))
                
                # for pair in max_positions_pairs:
                #     if axis == 0:
                #         # voxel_point = [start_point + index, min_coords[1] + pair[0], min_coords[2] + pair[1]]
                #     elif axis == 1:
                #         point = [min_coords[0] + pair[0], min_coords[2] + pair[1]]
                #         # voxel_point = [min_coords[0] + pair[0], start_point +  index, min_coords[2] +  pair[1]]
                #     else:
                #         point = [min_coords[0] + pair[0], min_coords[1] + pair[1]]
                #         # voxel_point = [min_coords[0] + pair[0], min_coords[1] + pair[1], start_point +  index]
                
                #     center_points.append(point)

                # Plot the gradient vectors
                segment_indices = np.argwhere(segment_slice == 1)
                changed_indices = np.argwhere(fixed_slice == 1)

                row = math.ceil((index+1)/3) - 1
                col = (index+1)%3 - 1

                axs[row, col].imshow(intensity_slice, cmap='viridis')
                axs[row, col].scatter(max_indices_2d[1], max_indices_2d[0], color='red', marker='o')
                
                for idx in segment_indices:
                    rect = plt.Rectangle((idx[1] - 0.5, idx[0] - 0.5), 1, 1, linewidth=2, edgecolor='blue', facecolor='none')
                    axs[row, col].add_patch(rect)

                for idx in changed_indices:
                    rect = plt.Rectangle((idx[1] - 0.5, idx[0] - 0.5), 1, 1, linewidth=0.5, edgecolor='red', facecolor='none')
                    axs[row, col].add_patch(rect)

                axs[row, col].axis('off')  # Hide axes
                axs[row, col].set_title(f"{axis_name} - {start_point + index}")

        # axe_slices = select_slices(cube_size, point_1, point_2, 14)
        # fig, axs = plt.subplots(3, np.max(differences)+1, figsize=(20, 15))

        # for axe_index, direction in enumerate(axe_slices):
        #     axis = direction['axis']

        #     if axis == 0:
        #         axis_name = 'X'
        #     elif axis == 1:
        #         axis_name = 'Y'
        #     else:
        #         axis_name = 'Z'

        #     start_point, end_point = direction['head_points']
        #     slices = direction['slices']

        #     for index, slice in enumerate(slices):
        #         center_points = []
        #         boundaries = np.argwhere(slice==True)
        #         if boundaries.shape[0]:
        #             min_coords = np.min(np.argwhere(slice==True), axis=0)
        #             max_coords = np.max(np.argwhere(slice==True), axis=0)
        #             segment_slice = select_slice(mask_data, start_point + index, axis, min_coords, max_coords)
        #             intensity_slice = select_slice(original_data, start_point + index, axis, min_coords, max_coords)

        #             normalized_data = (intensity_slice - intensity_slice.min()) / (intensity_slice.max() - intensity_slice.min())
        #             offsets = [(-1, -1), (-1, 0), (-1, 1),
        #                     (0, -1),          (0, 1),
        #                     (1, -1),  (1, 0),  (1, 1)]

        #             # Iterate over each pixel
        #             max_count = np.zeros((normalized_data.shape[0], normalized_data.shape[1])) 
        #             padded_data = np.pad(normalized_data, 1, mode='constant')

        #             for i in range(1, normalized_data.shape[0] + 1):
        #                 for j in range(1, normalized_data.shape[1] + 1):
        #                     # Get the values of the pixel and its neighbors from the padded array
        #                     neighbors_values = [padded_data[i + ni, j + nj] for ni, nj in offsets]
        #                     # Find the index of the neighbor with the highest value
        #                     max_neighbor_index = np.argmax(neighbors_values)
        #                     max_neighbor_position = (i + offsets[max_neighbor_index][0], j + offsets[max_neighbor_index][1])
                            
        #                     x_pos = max_neighbor_position[0] - 1  # Adjust for padding
        #                     y_pos = max_neighbor_position[1] - 1  # Adjust for padding
                            
        #                     if normalized_data[x_pos][y_pos] > 0.4:
        #                         max_count[x_pos][y_pos] += 1
                        
        #                 max_indices = np.argsort(max_count, axis=None)[-2:]
        #                 max_indices_2d = np.unravel_index(max_indices, max_count.shape)
        #                 # max_positions_pairs = list(zip(max_indices_2d[0], max_indices_2d[1]))
                    
        #             # for pair in max_positions_pairs:
        #             #     if axis == 0:
        #             #         # voxel_point = [start_point + index, min_coords[1] + pair[0], min_coords[2] + pair[1]]
        #             #     elif axis == 1:
        #             #         point = [min_coords[0] + pair[0], min_coords[2] + pair[1]]
        #             #         # voxel_point = [min_coords[0] + pair[0], start_point +  index, min_coords[2] +  pair[1]]
        #             #     else:
        #             #         point = [min_coords[0] + pair[0], min_coords[1] + pair[1]]
        #             #         # voxel_point = [min_coords[0] + pair[0], min_coords[1] + pair[1], start_point +  index]
                    
        #             #     center_points.append(point)

        #             # Plot the gradient vectors
        #             segment_indices = np.argwhere(segment_slice == 1)

        #             axs[axe_index, index].imshow(intensity_slice, cmap='viridis')
        #             axs[axe_index, index].scatter(max_indices_2d[1], max_indices_2d[0], color='red', marker='o')
                    
        #             for idx in segment_indices:
        #                 rect = plt.Rectangle((idx[1] - 0.5, idx[0] - 0.5), 1, 1, linewidth=1, edgecolor='blue', facecolor='none')
        #                 axs[axe_index, index].add_patch(rect)

        #             axs[axe_index, index].axis('off')  # Hide axes
        #             axs[axe_index, index].set_title(f"{axis_name} - {start_point + index}")

        plt.show()
        return 

            # print(slice_value[0])

        # center_points = np.array(center_points)
        # cluster_points = go.Scatter3d(
        #         x=center_points[:, 0],
        #         y=center_points[:, 1],
        #         z=center_points[:, 2],
        #         mode='markers',
        #         marker=dict(
        #             size=3,
        #             color='black',
        #         ),
        #         name='Points',
        #     )

    #     aca_endpoints = [15, 701]
    #     directions = ['left', 'right']
    #     end_point_1, end_point_2 = skeleton_points[aca_endpoints[0]], skeleton_points[aca_endpoints[1]]

    #     v2 = end_point_1 - end_point_2
    #     junction_points, neighbor_distances = remove_junction_points(neighbor_distances, junction_points, skeleton_points)
    #     junction_points, new_neighbor_distances, connected_lines = reduce_skeleton_points(neighbor_distances, junction_points, end_points, skeleton_points)
    #     junction_points, new_neighbor_distances, remove_edges = remove_cycle_edges(skeleton_points, new_neighbor_distances, v2)

    #     for edge in remove_edges:
    #         point_1 = edge[0]
    #         point_2 = edge[1]
    #         for line in connected_lines:
    #             if (point_1 == line[0] or point_1 == line[-1]) and (point_2 == line[0] or point_2 == line[-1]):
    #                 for i in range(len(line) - 1):
    #                     neighbor_distances[line[i]][line[i+1]] = 0
    #                     neighbor_distances[line[i+1]][line[i]] = 0

    #     junction_points, new_neighbor_distances, connected_lines = reduce_skeleton_points(neighbor_distances, junction_points, end_points, skeleton_points)
        
    #     edges = []
    #     for i in range(skeleton_points.shape[0]):
    #         for j in range(i+1, skeleton_points.shape[0]):
    #             if i != j and (new_neighbor_distances[i][j] > 0 or new_neighbor_distances[j][i] > 0):
    #                 lines.append([skeleton_points[i], skeleton_points[j]])
    #                 edges.append([i, j])

    #     duplicate_lines = {}
    #     for line in connected_lines:
    #         if line[0] != line[-1]:
    #             point_1 = line[0]
    #             point_2 = line[-1]

    #             if point_1 > point_2:
    #                 tmp = point_1
    #                 point_1 = point_2
    #                 point_2 = tmp
    #                 line.reverse()

    #             if point_1 not in duplicate_lines:
    #                 duplicate_lines[point_1] = {}
                
    #             if point_2 not in duplicate_lines[point_1]:
    #                 duplicate_lines[point_1][point_2] = []


    #             duplicate_lines[point_1][point_2].append(line)

    #     new_connected_lines = []

    #     for key, value in duplicate_lines.items():
    #         for sub_key, sub_value in value.items():
    #             if len(sub_value) == 2:
    #                 line_1 = sub_value[0]
    #                 line_2 = sub_value[1]
    #                 interpolate_line = interpolate_center_path(skeleton_points[line_1], skeleton_points[line_2], 1)

    #                 removed_idx = []
    #                 for i, line in enumerate(connected_lines):
    #                     if (line[0] == key and line[-1] == sub_key) or (line[-1] == key and line[0] == sub_key):
    #                         removed_idx.append(i)
                        
    #                 removed_idx.sort(reverse=True)
    #                 for index in removed_idx:
    #                     connected_lines.pop(index)

    #                 new_line = [key]
    #                 for i in range(1, len(interpolate_line)-1):
    #                     new_point = interpolate_line[i]
    #                     new_index = skeleton_points.shape[0]
    #                     skeleton_points = np.vstack([skeleton_points, new_point])
    #                     new_line.append(new_index)

    #                 new_line.append(sub_key)
    #                 connected_lines.append(new_line)

    #                 removed_edges = []
    #                 for i, edge in enumerate(edges):
    #                     edge_1 = edge[0]
    #                     edge_2 = edge[-1]

    #                     if (edge_1 in line_1 and edge_2 in line_1) or (edge_1 in line_2 and edge_2 in line_2):
    #                         removed_edges.append(i)

    #                 removed_edges.sort(reverse=True)
    #                 for index in removed_edges:
    #                     edges.pop(index)
                    
    #                 edges.append([key, sub_key])

        
    #     main_trunks = []
    #     line_weights = []
    #     for i, endpoint in enumerate(aca_endpoints):
    #         list_paths = []
    #         for point in end_points:
    #             if point not in aca_endpoints:
    #                 shortest_cost, shortest_path = dijkstra(edges, endpoint, point, skeleton_points, directions[i])
    #                 list_paths.append(shortest_path)


    #         path_weights = {}

    #         for path in list_paths:
    #             for i in range(len(path)-1):
    #                 point_1 = path[i]
    #                 point_2 = path[i+1]

    #                 if point_1 < point_2:
    #                     key = (point_1, point_2)
    #                 else:
    #                     key = (point_2, point_1)
                    
    #                 if key not in path_weights:
    #                     path_weights[key] = 0
                    
    #                 path_weights[key] += 1
            
    #         start_point = endpoint
    #         visited_edges = []
    #         main_points = [start_point]

    #         while start_point != -1:
    #             connected_edges = []

    #             for key, value in path_weights.items():
    #                 if start_point in key and list_not_in_lists(key, visited_edges):
    #                     connected_edges.append(key)

    #             if len(connected_edges) == 0:
    #                 start_point = -1
    #             else:
    #                 weights = [path_weights[key] for key in connected_edges]
    #                 max_value = max(weights)
    #                 max_positions = [i for i, weight in enumerate(weights) if weight == max_value]
                    
    #                 if len(max_positions) > 1:
    #                     start_point = -1
    #                 else:
    #                     edge = connected_edges[max_positions[0]]
    #                     if start_point == edge[0]:
    #                         start_point = edge[1]
    #                     else:
    #                         start_point = edge[0]
    #                     visited_edges.append(edge)
    #                     main_points.append(start_point)
            
    #         main_trunk = []
    #         for idx in range(len(main_points) - 1):
    #             if main_points[idx] < main_points[idx+1]:
    #                 main_trunk.append([main_points[idx], main_points[idx+1]])
    #             else:
    #                 main_trunk.append([main_points[idx+1], main_points[idx]])
            
    #         main_trunks.append(main_trunk)
    #         line_weights.append(path_weights)

    #     common_paths = []

    #     # Convert each sublist to sets for easier comparison
    #     set1 = {tuple(sublist) for sublist in main_trunks[0]}
    #     set2 = {tuple(sublist) for sublist in main_trunks[1]}

    #     # Find the common 2-element lists
    #     common_paths = set1.intersection(set2)
    #     common_paths = [list(sublist) for sublist in common_paths]
    #     left_paths = set1 - set2
    #     left_paths = [list(sublist) for sublist in left_paths]
    #     right_paths = set2 - set1
    #     right_paths = [list(sublist) for sublist in right_paths]

    #     loop = 0
    #     while (len(left_paths) + len(right_paths) + len(common_paths) < len(edges) and loop <= 10):
    #         loop += 1
    #         common_points = list(set([item for sublist in common_paths for item in sublist]))
    #         left_points = list(set([item for sublist in left_paths for item in sublist]))
    #         right_points = list(set([item for sublist in right_paths for item in sublist]))
    #         diff_left_points = set(left_points) - set(right_points) - set(common_points)
    #         diff_left_points = list(diff_left_points)
    #         diff_right_points = set(right_points) - set(left_points) - set(common_points)
    #         diff_right_points = list(diff_right_points)
            
    #         new_left_paths = find_rest_paths(edges, diff_left_points, left_paths)
    #         new_right_paths = find_rest_paths(edges, diff_right_points, right_paths)   

    #         left_paths = set(map(tuple, left_paths + new_left_paths))
    #         left_paths = [list(sublist) for sublist in left_paths]
    #         right_paths = set(map(tuple, right_paths + new_right_paths))
    #         right_paths = [list(sublist) for sublist in right_paths]

    #         diff_left_points = list(set([item for sublist in left_paths for item in sublist]))
    #         diff_right_points = list(set([item for sublist in right_paths for item in sublist]))
         
    #         undefined_paths = []
    #         for edge in edges:
    #             if list_not_in_lists(edge, common_paths) and list_not_in_lists(edge, left_paths) and list_not_in_lists(edge, right_paths):
    #                 undefined_paths.append(edge)


    #         undefined_points = list(set([item for sublist in undefined_paths for item in sublist]))
    #         suspected_points = {}

    #         for point in junction_points:
    #             if (point in undefined_points) and ((point in diff_left_points) or (point in diff_right_points) or (point in common_points)):
    #                 suspected_points[point] = {}

    #                 for edge in edges:
    #                     if point == edge[0]:
    #                         point_1 = edge[0]
    #                         point_2 = edge[1]
    #                     elif point == edge[1]:
    #                         point_1 = edge[1]
    #                         point_2 = edge[0]
    #                     else:
    #                         continue

    #                     w = 0
    #                     if edge in common_paths:
    #                         w = 0
    #                     elif edge in left_paths:
    #                         w = 1
    #                     elif edge in right_paths:
    #                         w = 2
    #                     else:
    #                         max_w = 0
    #                         w1 = 0
    #                         w2 = 0

    #                         edge_key = tuple(edge)
    #                         if edge_key in line_weights[0]:
    #                             w1 = line_weights[0][edge_key]
    #                         if edge_key in line_weights[1]:
    #                             w2 = line_weights[1][edge_key]
                            
    #                         max_w = max(w1, w2)

    #                         if max_w == 1:
    #                             w = -1
    #                         else:
    #                             w = 3

    #                     suspected_points[point][point_2] = w

    #         for key, sub_dict in suspected_points.items():
    #             found_one = False
    #             found_two = False
    #             found_zero = False
    #             zero_count = 0
                
    #             check_edges = []
                
    #             for sub_key, value in sub_dict.items():
    #                 if value == 1:
    #                     found_one = True
    #                     one_key = sub_key
    #                 elif value == 2:
    #                     found_two = True
    #                     two_key = sub_key
    #                 elif value == 0:
    #                     found_zero = True
    #                     zero_count += 1
    #                     zero_key = sub_key
    #                 else:
    #                     check_edges.append(sub_key)

    #             if found_one and found_two:
    #                 principal_vector_1 = skeleton_points[key] - skeleton_points[one_key]
    #                 principal_vector_2 = skeleton_points[key] - skeleton_points[two_key]
    #                 other_vectors = [skeleton_points[key] - skeleton_points[point] for point in check_edges]
    #                 left_index, right_index = split_direction_2([principal_vector_1, principal_vector_2], other_vectors)
    #                 left_point = check_edges[left_index]
    #                 right_point = check_edges[right_index]

    #                 if key < left_point:
    #                     path1 = [key, left_point]
    #                 else:
    #                     path1 = [left_point, key]

    #                 if key < right_point:
    #                     path2 = [key, right_point]
    #                 else:
    #                     path2 = [right_point, key]

    #                 left_paths = set(map(tuple, left_paths + [path1]))
    #                 left_paths = [list(sublist) for sublist in left_paths]
    #                 right_paths = set(map(tuple, right_paths + [path2]))
    #                 right_paths = [list(sublist) for sublist in right_paths]

    #             # elif found_zero and zero_count == 1:
    #             #     principal_vector = skeleton_points[zero_key] - skeleton_points[key]
    #             #     other_vectors = [skeleton_points[key] - skeleton_points[point] for point in check_edges]
    #             #     left_index, right_index = split_direction_1(principal_vector, other_vectors)

    #             #     left_point = check_edges[left_index]
    #             #     right_point = check_edges[right_index]

    #             #     if key < left_point:
    #             #         path1 = [key, left_point]
    #             #     else:
    #             #         path1 = [left_point, key]

    #             #     if key < right_point:
    #             #         path2 = [key, right_point]
    #             #     else:
    #             #         path2 = [right_point, key]

    #             #     left_paths = set(map(tuple, left_paths + [path1]))
    #             #     left_paths = [list(sublist) for sublist in left_paths]
    #             #     right_paths = set(map(tuple, right_paths + [path2]))
    #             #     right_paths = [list(sublist) for sublist in right_paths]
    #             # else:
    #             #     for point in check_edges:
    #             #         if sub_dict[point] == -1:
    #             #             distance_1 = np.linalg.norm(np.array(skeleton_points[point]) - np.array(skeleton_points[aca_endpoints[0]]))
    #             #             distance_2 = np.linalg.norm(np.array(skeleton_points[point]) - np.array(skeleton_points[aca_endpoints[1]]))
    #             #             if distance_1 < distance_2:
    #             #                 if key < point:
    #             #                     path = [key, point]
    #             #                 else:
    #             #                     path = [point, key]
    #             #                 left_paths = set(map(tuple, left_paths + [path]))
    #             #                 left_paths = [list(sublist) for sublist in left_paths]
    #             #             else:
    #             #                 if key < point:
    #             #                     path = [key, point]
    #             #                 else:
    #             #                     path = [point, key]
    #             #                 right_paths = set(map(tuple, right_paths + [path]))
    #             #                 right_paths = [list(sublist) for sublist in right_paths]
    #             #         else:
    #             #             if key < point:
    #             #                 path = [key, point]
    #             #             else:
    #             #                 path = [point, key]
    #             #             common_paths.append(path)
    #             #             common_paths = [list(sublist) for sublist in common_paths]
        
    #     # loop = 0
    #     # stop_left_paths = []
    #     # stop_right_paths = []

    #     # while len(common_paths) and loop < 150:
    #     #     loop += 1
    #     #     remove_idx = []

    #     #     for idx, path in enumerate(common_paths):
    #     #         point_1 = path[0]
    #     #         point_2 = path[1]

    #     #         touch_point = point_1
    #     #         neighbors = find_neighbors(point_1, left_paths, right_paths, connected_lines, stop_left_paths, stop_right_paths)
                
    #     #         if neighbors is None:
    #     #             touch_point = point_2
    #     #             neighbors = find_neighbors(point_2, left_paths, right_paths, connected_lines, stop_left_paths, stop_right_paths)
                    
    #     #             if neighbors is None:
    #     #                 continue

    #     #         left_point, right_point = find_p4_p5(
    #     #             skeleton_points[neighbors[0]['point']], 
    #     #             skeleton_points[neighbors[1]['point']], 
    #     #             skeleton_points[touch_point])

    #     #         # print(skeleton_points[touch_point], skeleton_points[neighbors[0]['point']], skeleton_points[neighbors[1]['point']], left_point, right_point )
                
    #     #         p1_index = skeleton_points.shape[0]
    #     #         p2_index = p1_index + 1

    #     #         skeleton_points = np.vstack([skeleton_points, left_point])
    #     #         skeleton_points = np.vstack([skeleton_points, right_point])

    #     #         left_path = left_paths[neighbors[0]['path']]
    #     #         right_path = right_paths[neighbors[1]['path']]
    #     #         left_connected_line = connected_lines[neighbors[0]['connected_line']]
    #     #         right_connected_line = connected_lines[neighbors[1]['connected_line']]

    #     #         next_point = None
    #     #         common_connected_index = None
    #     #         for k, connected_line in enumerate(connected_lines):
    #     #             if (connected_line[0] == point_1 and connected_line[-1] == point_2) or (connected_line[0] == point_2 and connected_line[-1] == point_1):
    #     #                 if connected_line[0] == touch_point:
    #     #                     next_point = connected_line[1]
    #     #                     common_connected_index = k
    #     #                     break
    #     #                 elif connected_line[-1] == touch_point:
    #     #                     next_point = connected_line[-2]
    #     #                     common_connected_index = k
    #     #                     break

    #     #         if next_point and common_connected_index:
    #     #             common_connected_line = connected_lines[common_connected_index]

    #     #             if left_path[0] == touch_point:
    #     #                 left_path[0] = next_point
    #     #             elif left_path[-1] == touch_point:
    #     #                 left_path[-1] = next_point
                    

    #     #             if right_path[0] == touch_point:
    #     #                 right_path[0] = next_point
    #     #             elif right_path[-1] == touch_point:
    #     #                 right_path[-1] = next_point

    #     #             if left_connected_line[0] == touch_point:
    #     #                 left_connected_line[0] = p1_index
    #     #                 left_connected_line.insert(0, next_point)
    #     #             elif left_connected_line[-1] == touch_point:
    #     #                 left_connected_line[-1] = p1_index
    #     #                 left_connected_line.append(next_point)

    #     #             if right_connected_line[0] == touch_point:
    #     #                 right_connected_line[0] = p2_index
    #     #                 right_connected_line.insert(0, next_point)
    #     #             elif right_connected_line[-1] == touch_point:
    #     #                 right_connected_line[-1] = p2_index
    #     #                 right_connected_line.append(next_point)

    #     #             # print('After left: ', left_path, left_connected_line)

    #     #             if touch_point == point_1:
    #     #                 if next_point == point_2:
    #     #                     stop_left_paths.append(neighbors[0]['path'])
    #     #                     stop_right_paths.append(neighbors[1]['path'])
    #     #                     del common_paths[idx]
    #     #                 else:
    #     #                     path[0] = next_point
    #     #             elif touch_point == point_2:
    #     #                 if next_point == point_1:
    #     #                     stop_left_paths.append(neighbors[0]['path'])
    #     #                     stop_right_paths.append(neighbors[1]['path'])
    #     #                     del common_paths[idx]
    #     #                 else:
    #     #                     path[1] = next_point

    #     #             if common_connected_line[0] == touch_point:
    #     #                 common_connected_line.pop(0)

    #     #             elif common_connected_line[-1] == touch_point:
    #     #                 common_connected_line.pop(-1)

    #     #             if len(common_connected_line) == 1 or len(common_connected_line) == 0:
    #     #                 del connected_lines[common_connected_index]

    #     # print(new_connected_lines)
    #     line_groups = [common_paths, right_paths, left_paths, undefined_paths]
    #     line_colors = ['black', 'blue', 'green', 'orange']
    #     line_traces = []

    #     for i, line_group in enumerate(line_groups):
    #         for line in line_group:
    #             for connected_line in connected_lines:
    #                 if (line[0] in connected_line and line[-1] in connected_line):
    #                     color = line_colors[i]
    #                     show_points = [connected_line[0], connected_line[-1]]
    #                     x_vals = [skeleton_points[point][0] for point in show_points]
    #                     y_vals = [skeleton_points[point][1] for point in show_points]
    #                     z_vals = [skeleton_points[point][2] for point in show_points]
    #                     trace = go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, 
    #                                             mode='lines', 
    #                                             line=dict(
    #                                                 color=color,
    #                                                 width=5
    #                                             ), 
    #                                         )
    #                     line_traces.append(trace)


    # # lines_2 = []
    # # for line in connected_lines:
    # #     for i in range(len(line)-1):
    # #         lines_2.append([skeleton_points[line[i]], skeleton_points[line[i+1]]])

    # # # Create traces for each line
    # # line_traces_2 = []
    # # for i, line in enumerate(lines_2):
    # #     color = 'black'
    # #     x_vals = [point[0] for point in line]
    # #     y_vals = [point[1] for point in line]
    # #     z_vals = [point[2] for point in line]
    # #     trace = go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='lines', line=dict(color='green'))
    # #     line_traces_2.append(trace)
    # # # # Create traces
    # # # Visualize artery points 
    # # point_trace_0 = go.Scatter3d(
    # #     x=artery_points[:, 0],
    # #     y=artery_points[:, 1],
    # #     z=artery_points[:, 2],
    # #     mode='markers',
    # #     marker=dict(
    # #         size=5,
    # #         color='green',
    # #     ),
    # #     name='Points'
    # # )

    # # border_point = go.Scatter3d(
    # #     x=verts[:, 0],
    # #     y=verts[:, 1],
    # #     z=verts[:, 2],
    # #     mode='markers',
    # #     marker=dict(
    # #         size=1,
    # #         color='black',
    # #     ),
    # #     name='Points'
    # # )

    # # Visualize skeleton points 
    # point_trace_1 = go.Scatter3d(
    #     x=skeleton_points[:, 0],
    #     y=skeleton_points[:, 1],
    #     z=skeleton_points[:, 2],
    #     mode='markers',
    #     marker=dict(
    #         size=3,
    #         color='green',
    #     ),
    #     name='Points',
    #     text=[f'Point {i}' for i, point in enumerate(skeleton_points)]
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
    #     name='Points',
    #     text=[f'Point {i}' for i in junction_points]
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
    #     name='Points',
    #     text=[f'Point {i}' for i in end_points]
    # )

    # # Create layout
    # layout = go.Layout(
    #     scene=dict(
    #         aspectmode='cube',
    #         camera=dict(
    #             eye=dict(x=voxel_sizes[0], y=voxel_sizes[1], z=voxel_sizes[2])
    #         )
    #     ),
    #     height=800,  # Set height to 800 pixels
    #     width=1600   # Set width to 1200 pixels
    # )

    # # fig_1 = go.Figure(data=[point_trace_0], layout=layout)
    # # fig_1.show()
    # fig_2 = go.Figure(data=[point_trace_2, point_trace_3] + line_traces, layout=layout)
    # fig_2.show()

    # return
    # # # Find longest path
    # # longest_path = []
    # # max_points = 0
    # # for i in range (len(end_points)):
    # #     for j in range (len(end_points)): 
    # #         if i != j:
    # #             path = longest_path_with_no_cycles(neighbor_distances, end_points[i], end_points[j])
    # #             if (len(path) > max_points):
    # #                 max_points = len(path)
    # #                 longest_path = path

    # # center_points = skeleton_points[longest_path]

    # # if not center_points.shape[0]:
    # #     return selected_data, preprocessed_data, None, None, None
    
    # # inter_points = interpolate_path(center_points)
    # # print('Number of centerpoints: ', len(inter_points))

    # # #Final results
    # # selected_data = np.copy(preprocessed_data)
    # # selected_data[selected_data < intensity_threshold_2] = 0
    # # selected_data[selected_data >= intensity_threshold_2] = 1
    # # touch_points = find_touchpoints(selected_data, inter_points, distance_threshold)

    # # if (touch_points.shape[0]):
    # #     selected_data[touch_points[:, 0], touch_points[:, 1], touch_points[:, 2]] = 0
        
    # # labeled_array, num_labels = ndimage.label(selected_data)
    # # labels, counts = np.unique(labeled_array[labeled_array != 0], return_counts=True)
    # # max_count_label = labels[np.argmax(counts)]
    # # selected_data[(labeled_array != 0) & (labeled_array != max_count_label)] = 0

    # # # Calculate marching cubes using scikit-image
    # # verts, faces, normals, _ = measure.marching_cubes(selected_data, level=0.5, spacing=voxel_sizes)
    # # skeleton = skeletonize(selected_data)
    # # skeleton_points = np.argwhere(skeleton != 0)
    # # visualized_center_points = skeleton_points*voxel_sizes
    # # print(skeleton_points.shape)

    # # vtmk_points = []
    # # json_path = '/Users/apple/Desktop/neuroscience/artery_surfaces/sub-25_run-1_mra_eICAB_CW/Scene/centerline.json'
    # # with open(json_path, 'r') as file:
    # #     json_data = json.load(file)
    
    # #     control_points = json_data['markups'][0]['controlPoints']
    # #     for point in control_points:
    # #         vtmk_points.append(point['position'])
    # # vtmk_points = np.array(vtmk_points)

    # # visualize_original_points([verts, visualized_center_points, vtmk_points], ['black', 'red', 'blue'], [1, 4, 5])

    # # # Find indices where values are greater than 0
    # # indices = np.nonzero(selected_data > 0)

    # # # Get the minimum and maximum positions for each dimension
    # # min_positions = np.min(indices, axis=1)
    # # max_positions = np.max(indices, axis=1)

    # # print("Minimum positions:", min_positions)
    # # print("Maximum positions:", max_positions)
    # # # point_cloud = o3d.geometry.PointCloud()
    # # # point_cloud.points = o3d.utility.Vector3dVector(artery_points) 

    # # # # Create a LineSet object
    # # # line_set = o3d.geometry.LineSet()
    # # # line_set.points = o3d.utility.Vector3dVector(skeleton_points)
    # # # lines = []
    # # # colors = []

    # # # for i in range(skeleton_points.shape[0]):
    # # #     for j in range(skeleton_points.shape[0]):
    # # #         if i != j and neighbor_distances[i][j] > 0:
    # # #             lines.append([i, j])
    # # #             colors.append([0, 0, 0])
    # # # line_set.lines = o3d.utility.Vector2iVector(lines)

    # # # # Convert marching cubes result to Open3D mesh
    # # # mesh_o3d = o3d.geometry.TriangleMesh()
    # # # mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
    # # # mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    # # # mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_o3d.triangles)[:, ::-1])
    # # # mesh_o3d = mesh_o3d.filter_smooth_laplacian(laplacian_iter, 0.5)
    # # # mesh_o3d.compute_vertex_normals()
    # # # mesh_o3d.compute_triangle_normals()

    # # # # Prepare lines for centerline from our algorithm
    # # # centerline_set = o3d.geometry.LineSet()
    # # # centerline_set.points = o3d.utility.Vector3dVector(skeleton_points)
    # # # centerline_lines = []
    # # # centerline_colors = []
    # # # for i in range(len(longest_path)-1):
    # # #     centerline_lines.append([longest_path[i], longest_path[i+1]])
    # # #     centerline_colors.append([1, 0, 0])

    # # # centerline_set.lines = o3d.utility.Vector2iVector(centerline_lines)
    # # # # centerline_set.colors = o3d.utility.Vector3dVector(np.array(centerline_colors))

    # # # # Visualize the LineSet
    # # # # o3d.visualization.draw_geometries([line_set, centerline_set], window_name="Line Set Visualization")
    # # # o3d.io.write_triangle_mesh(f"{folder_path}/artery_{index}_it1-{intensity_threshold_1}_it2-{intensity_threshold_2}_laplacian-{laplacian_iter}.stl", mesh_o3d)

    # # return selected_data, preprocessed_data, skeleton_points, skeleton_points[end_points], touch_points

def create_go_points(points, color, size=5):
    # Visualize skeleton points 
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
        ),
        name='Points'
    )

if __name__ == "__main__":
    # Specify the path to your NIfTI file
    start_time = time.time()

    # segment_file_path = '/Users/apple/Downloads/output-1/TOF_eICAB_CW.nii.gz'
    # original_file_path = '/Users/apple/Downloads/output-1/TOF_resampled.nii.gz'

    # segment_file_path = 'sub-25_run-1_mra_eICAB_CW (1).nii'
    # original_file_path = 'sub-25_run-1_mra_resampled.nii'

    segment_file_path = '/Users/apple/Downloads/sub61_harvard_watershed.nii.gz'
    original_file_path = '/Users/apple/Downloads/sub-61_acq-tof_angio_resampled.nii.gz'

    filename = os.path.basename(segment_file_path)  # Extracts the filename from the path
    filename_without_extension = os.path.splitext(filename)[0]  # Removes the extension
    folder_path = f'/Users/apple/Desktop/neuroscience/artery_surfaces/{filename_without_extension}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Load the NIfTI image
    segment_image = nib.load(segment_file_path)
    original_image = nib.load(original_file_path)
    intensity_threshold_1 = 0.2
    intensity_threshold_2 = 0.1
    gaussian_sigma=2
    distance_threshold=20
    laplacian_iter = 5
    bins = np.arange(0, 1.1, 0.1)

    #Find skeleton
    find_skeleton(
                    segment_image, 
                    original_image, 
                    index=[5, 6], 
                    intensity_threshold_1=intensity_threshold_1, 
                    intensity_threshold_2=intensity_threshold_2, 
                    gaussian_sigma=gaussian_sigma, 
                    distance_threshold=distance_threshold,
                    laplacian_iter=laplacian_iter,
                    folder_path=folder_path
                )# Record end time

    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Execution time:", elapsed_time, "seconds")