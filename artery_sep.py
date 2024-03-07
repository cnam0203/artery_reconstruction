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

    artery_points_old = np.argwhere(mask_data != 0)
    artery_points = np.argwhere(mask_data != 0)*voxel_sizes

    skeleton = skeletonize(selected_data)
    skeleton_points = np.argwhere(skeleton != 0)

    end_points, junction_points, neighbor_distances = find_graph(skeleton_points)

    lines = []
    line_values = []
    line_colors = []

    if 5 in index or 6 in index:
        aca_endpoints = [15, 701]
        directions = ['right', 'left']
        end_point_1, end_point_2 = skeleton_points[aca_endpoints[0]], skeleton_points[aca_endpoints[1]]

        v2 = end_point_1 - end_point_2
        junction_points, neighbor_distances = remove_junction_points(neighbor_distances, junction_points, skeleton_points)
        junction_points, new_neighbor_distances, connected_lines = reduce_skeleton_points(neighbor_distances, junction_points, end_points, skeleton_points)
        junction_points, new_neighbor_distances, remove_edges = remove_cycle_edges(skeleton_points, new_neighbor_distances, v2)

        for edge in remove_edges:
            point_1 = edge[0]
            point_2 = edge[1]
            for line in connected_lines:
                if (point_1 == line[0] or point_1 == line[-1]) and (point_2 == line[0] or point_2 == line[-1]):
                    for i in range(len(line) - 1):
                        neighbor_distances[line[i]][line[i+1]] = 0
                        neighbor_distances[line[i+1]][line[i]] = 0

        junction_points, new_neighbor_distances, connected_lines = reduce_skeleton_points(neighbor_distances, junction_points, end_points, skeleton_points)

        edges = []

        for i in range(skeleton_points.shape[0]):
            for j in range(i+1, skeleton_points.shape[0]):
                if i != j and (new_neighbor_distances[i][j] > 0 or new_neighbor_distances[j][i] > 0):
                    lines.append([skeleton_points[i], skeleton_points[j]])
                    edges.append([i, j])

        list_paths = []
        for i, endpoint in enumerate(aca_endpoints):
            for point in end_points:
                if point not in aca_endpoints and i!= 0:
                    shortest_cost, shortest_path = dijkstra(edges, endpoint, point, skeleton_points, directions[i])
                    list_paths.append(shortest_path)

        path_weights = {}

        for path in list_paths:
            for i in range(len(path)-1):
                point_1 = path[i]
                point_2 = path[i+1]

                if point_1 < point_2:
                    key = (point_1, point_2)
                else:
                    key = (point_2, point_1)
                
                if key not in path_weights:
                    path_weights[key] = 0
                
                path_weights[key] += 1
        
        for edge in edges:
            point_1 = edge[0]
            point_2 = edge[1]

            if point_1 < point_2:
                key = (point_1, point_2)
            else:
                key = (point_2, point_1)
            
            if key in path_weights:
                line_value = path_weights[key]
            else:
                line_value = 0
            line_values.append(line_value)
    
    # Create traces for each line
    line_traces = []
    color_scale = [
        [0, 'blue'],    # Color for low weight values
        [0.5, 'green'], # Color for medium weight values
        [1, 'red']      # Color for high weight values
    ]

    for i, line in enumerate(lines):
        color = 'blue'
        x_vals = [point[0] for point in line]
        y_vals = [point[1] for point in line]
        z_vals = [point[2] for point in line]
        trace = go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, 
                                mode='lines', 
                                line=dict(
                                    color=line_values[i],
                                    colorscale=color_scale,
                                    cmin=min(line_values),
                                    cmax=max(line_values)
                                ), 
                                text=f'Weight {line_values[i]}'
                            )
        line_traces.append(trace)

    # lines_2 = []
    # for line in connected_lines:
    #     for i in range(len(line)-1):
    #         lines_2.append([skeleton_points[line[i]], skeleton_points[line[i+1]]])

    # # Create traces for each line
    # line_traces_2 = []
    # for i, line in enumerate(lines_2):
    #     color = 'blue'
    #     x_vals = [point[0] for point in line]
    #     y_vals = [point[1] for point in line]
    #     z_vals = [point[2] for point in line]
    #     trace = go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='lines', line=dict(color='green'))
    #     line_traces_2.append(trace)
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

    # Visualize skeleton points 
    point_trace_1 = go.Scatter3d(
        x=skeleton_points[:, 0],
        y=skeleton_points[:, 1],
        z=skeleton_points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='green',
        ),
        name='Points',
        text=[f'Point {i}' for i, point in enumerate(skeleton_points)]
    )
    
    # Visualize junction points
    point_trace_2 = go.Scatter3d(
        x=skeleton_points[junction_points, 0],
        y=skeleton_points[junction_points, 1],
        z=skeleton_points[junction_points, 2],
        mode='markers',
        marker=dict(
            size=5,
            color='black',
        ),
        name='Points',
        text=[f'Point {i}' for i in junction_points]
    )
    
    # Visualize end points
    point_trace_3 = go.Scatter3d(
        x=skeleton_points[end_points, 0],
        y=skeleton_points[end_points, 1],
        z=skeleton_points[end_points, 2],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
        ),
        name='Points',
        text=[f'Point {i}' for i in end_points]
    )

    # Create layout
    layout = go.Layout(
        scene=dict(
            aspectmode='cube',
            camera=dict(
                eye=dict(x=voxel_sizes[0], y=voxel_sizes[1], z=voxel_sizes[2])
            )
        ),
        height=800,  # Set height to 800 pixels
        width=1600   # Set width to 1200 pixels
    )

    # fig_1 = go.Figure(data=[point_trace_0], layout=layout)
    # fig_1.show()
    fig_2 = go.Figure(data=[point_trace_2, point_trace_3]+line_traces, layout=layout)
    fig_2.show()

    return
    # # Find longest path
    # longest_path = []
    # max_points = 0
    # for i in range (len(end_points)):
    #     for j in range (len(end_points)): 
    #         if i != j:
    #             path = longest_path_with_no_cycles(neighbor_distances, end_points[i], end_points[j])
    #             if (len(path) > max_points):
    #                 max_points = len(path)
    #                 longest_path = path

    # center_points = skeleton_points[longest_path]

    # if not center_points.shape[0]:
    #     return selected_data, preprocessed_data, None, None, None
    
    # inter_points = interpolate_path(center_points)
    # print('Number of centerpoints: ', len(inter_points))

    # #Final results
    # selected_data = np.copy(preprocessed_data)
    # selected_data[selected_data < intensity_threshold_2] = 0
    # selected_data[selected_data >= intensity_threshold_2] = 1
    # touch_points = find_touchpoints(selected_data, inter_points, distance_threshold)

    # if (touch_points.shape[0]):
    #     selected_data[touch_points[:, 0], touch_points[:, 1], touch_points[:, 2]] = 0
        
    # labeled_array, num_labels = ndimage.label(selected_data)
    # labels, counts = np.unique(labeled_array[labeled_array != 0], return_counts=True)
    # max_count_label = labels[np.argmax(counts)]
    # selected_data[(labeled_array != 0) & (labeled_array != max_count_label)] = 0

    # # Calculate marching cubes using scikit-image
    # verts, faces, normals, _ = measure.marching_cubes(selected_data, level=0.5, spacing=voxel_sizes)
    # skeleton = skeletonize(selected_data)
    # skeleton_points = np.argwhere(skeleton != 0)
    # visualized_center_points = skeleton_points*voxel_sizes
    # print(skeleton_points.shape)

    # vtmk_points = []
    # json_path = '/Users/apple/Desktop/neuroscience/artery_surfaces/sub-25_run-1_mra_eICAB_CW/Scene/centerline.json'
    # with open(json_path, 'r') as file:
    #     json_data = json.load(file)
    
    #     control_points = json_data['markups'][0]['controlPoints']
    #     for point in control_points:
    #         vtmk_points.append(point['position'])
    # vtmk_points = np.array(vtmk_points)

    # visualize_original_points([verts, visualized_center_points, vtmk_points], ['black', 'red', 'blue'], [1, 4, 5])

    # # Find indices where values are greater than 0
    # indices = np.nonzero(selected_data > 0)

    # # Get the minimum and maximum positions for each dimension
    # min_positions = np.min(indices, axis=1)
    # max_positions = np.max(indices, axis=1)

    # print("Minimum positions:", min_positions)
    # print("Maximum positions:", max_positions)
    # # point_cloud = o3d.geometry.PointCloud()
    # # point_cloud.points = o3d.utility.Vector3dVector(artery_points) 

    # # # Create a LineSet object
    # # line_set = o3d.geometry.LineSet()
    # # line_set.points = o3d.utility.Vector3dVector(skeleton_points)
    # # lines = []
    # # colors = []

    # # for i in range(skeleton_points.shape[0]):
    # #     for j in range(skeleton_points.shape[0]):
    # #         if i != j and neighbor_distances[i][j] > 0:
    # #             lines.append([i, j])
    # #             colors.append([0, 0, 0])
    # # line_set.lines = o3d.utility.Vector2iVector(lines)

    # # # Convert marching cubes result to Open3D mesh
    # # mesh_o3d = o3d.geometry.TriangleMesh()
    # # mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
    # # mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    # # mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_o3d.triangles)[:, ::-1])
    # # mesh_o3d = mesh_o3d.filter_smooth_laplacian(laplacian_iter, 0.5)
    # # mesh_o3d.compute_vertex_normals()
    # # mesh_o3d.compute_triangle_normals()

    # # # Prepare lines for centerline from our algorithm
    # # centerline_set = o3d.geometry.LineSet()
    # # centerline_set.points = o3d.utility.Vector3dVector(skeleton_points)
    # # centerline_lines = []
    # # centerline_colors = []
    # # for i in range(len(longest_path)-1):
    # #     centerline_lines.append([longest_path[i], longest_path[i+1]])
    # #     centerline_colors.append([1, 0, 0])

    # # centerline_set.lines = o3d.utility.Vector2iVector(centerline_lines)
    # # # centerline_set.colors = o3d.utility.Vector3dVector(np.array(centerline_colors))

    # # # Visualize the LineSet
    # # # o3d.visualization.draw_geometries([line_set, centerline_set], window_name="Line Set Visualization")
    # # o3d.io.write_triangle_mesh(f"{folder_path}/artery_{index}_it1-{intensity_threshold_1}_it2-{intensity_threshold_2}_laplacian-{laplacian_iter}.stl", mesh_o3d)

    # return selected_data, preprocessed_data, skeleton_points, skeleton_points[end_points], touch_points

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
    intensity_threshold_1 = 0.000001
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