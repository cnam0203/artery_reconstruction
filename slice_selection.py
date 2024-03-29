import numpy as np
import math
import matplotlib.pyplot as plt
from process_graph import *
from collections import deque

def visualize_slice(intensity_slice, segment_slice, fixed_slice, split_points, start_point, index, axis):
    fig, axs = plt.subplots()  # Create figure and axes objects
    
    if axis == 0:
        axis_name = 'X'
    elif axis == 1:
        axis_name = 'Y'
    else:
        axis_name = 'Z'
        
    segment_indices = np.argwhere(segment_slice == 1)
    changed_indices = np.argwhere(fixed_slice == 1)
        
    axs.imshow(intensity_slice, cmap='viridis')
    
    y_values = [row[1] for row in split_points]
    x_values = [row[0] for row in split_points]
    axs.scatter(y_values, x_values, color='red', marker='o')
    
    for idx in segment_indices:
        rect = plt.Rectangle((idx[1] - 0.5, idx[0] - 0.5), 1, 1, linewidth=2, edgecolor='blue', facecolor='none')
        axs.add_patch(rect)
    
    for idx in changed_indices:
        rect = plt.Rectangle((idx[1] - 0.5, idx[0] - 0.5), 1, 1, linewidth=0.5, edgecolor='red', facecolor='none')
        axs.add_patch(rect)
        
    axs.axis('off')  # Hide axes
    axs.set_title(f"{axis_name} - {start_point + index}")
    plt.show()
    
    return

def select_slice(preprocessed_data, index, axis, min_coords, max_coords):
    # Select the slice along the specified axis
    if axis == 0:
        slice_data = preprocessed_data[index, min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
    elif axis == 1:
        slice_data = preprocessed_data[ min_coords[0]:max_coords[0], index, min_coords[2]:max_coords[2] ]
    else:
        slice_data = preprocessed_data[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], index]
    
    return slice_data

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

def auto_select_slices(cube_size, path_index, skeleton_points, square_edge, common_paths, connected_lines):
    # Convert points to arrays
    path = common_paths[path_index]
    point1 = skeleton_points[path[0]]
    point2 = skeleton_points[path[1]]
    
    connected_line = []
    for line in connected_lines:
        if (line[0] == path[0] and line[-1] == path[-1]) or (line[0] == path[-1] and line[-1] == path[0]):
            connected_line = line
            break

    # Calculate direction vector of the line connecting the two points
    direction_vectors = np.abs(np.array(point2) - np.array(point1))
    max_value = np.max(direction_vectors)
    positions = np.where(direction_vectors == max_value)[0]
    
    if len(positions) == 1:
        axis = positions[0]
    else:
        extend_path = find_extend_paths(path_index, common_paths)
        extend_point_1 = skeleton_points[extend_path[0]]
        extend_point_2 = skeleton_points[extend_path[1]]
        
        direction_vectors = np.abs(np.array(extend_point_2) - np.array(extend_point_1))
        max_value = np.max(direction_vectors)
        positions = np.where(direction_vectors == max_value)[0]
        axis = positions[0]

    increment = 1

    # Determine the slices along the axis of the line
    if point1[axis] > point2[axis]:
        increment = -1
    
    # start_point = min(point1[axis], point2[axis])
    # end_point = max(point1[axis], point2[axis])
    perpendicular_slices = []
    axis_points = skeleton_points[connected_line][:, axis]
    latest_point = point1

    for i in range(point1[axis], point2[axis] + increment, increment):
        pos = np.argwhere(axis_points == i)[:, 0].tolist()

        if len(pos) == 0:
            intersection_point = latest_point
            intersection_point[axis] = intersection_point[axis] + increment
        elif len(pos) == 1:
            intersection_point = skeleton_points[connected_line[pos[0]]]
        else:
            selected_index = [connected_line[i] for i in pos]
            intersection_point = np.mean(skeleton_points[selected_index], axis=0).astype(int)
        
        latest_point = intersection_point
        square_region = find_square_region(cube_size, np.array(intersection_point), square_edge, axis, i)
        perpendicular_slices.append(square_region)
    
    return {
        'axis': axis,
        'slices': perpendicular_slices,
        'head_points': [point1[axis], point2[axis]]
    }
    
def find_extend_paths(path_index, common_paths):
    path = common_paths[path_index]
    point_1 = path[0]
    point_2 = path[1]
    extend_point_1 = point_1
    extend_point_2 = point_2
    
    for i, path in enumerate(common_paths):
        if i != path_index:
            if point_1 == path[0]:
                extend_point_1 = path[1]
                break
            elif point_1 == path[1]:
                extend_point_1 = path[0]
                break
            
    for i, path in enumerate(common_paths):
        if i != path_index:
            if point_2 == path[0]:
                extend_point_2 = path[1]
                break
            elif point_2 == path[1]:
                extend_point_2 = path[0]
                break
            
    return [extend_point_1, extend_point_2]
            
def are_neighbors(point1, point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    
    return abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1 and (x1 != x2 or y1 != y2) 

def is_connectable(mask, start_point, end_point):
    """
    Determine if two points can be connected within a binary mask.
    
    Args:
    - mask: numpy array representing the binary mask
    - start_point: tuple of (row, column) representing the starting point
    - end_point: tuple of (row, column) representing the ending point
    
    Returns:
    - True if the points can be connected, False otherwise
    """
    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    
    # Check if start and end points are valid
    if not (0 <= start_point[0] < rows and 0 <= start_point[1] < cols):
        return False
    if not (0 <= end_point[0] < rows and 0 <= end_point[1] < cols):
        return False
    
    queue = deque([start_point])
    
    while queue:
        current_point = queue.popleft()
        
        # Check if current point is the end point
        if current_point[0] == end_point[0] and current_point[1] == end_point[1]:
            return True
        
        # Explore neighbors
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, -1), (-1, 1)]:
            new_row, new_col = current_point[0] + dr, current_point[1] + dc
            if (0 <= new_row < rows and 0 <= new_col < cols and
                mask[new_row, new_col] == 1 and not visited[new_row, new_col]):
                queue.append([new_row, new_col])
                visited[new_row, new_col] = True
                
    return False

def find_split_points(common_paths, original_data, mask_data, selected_data, skeleton_points, connected_lines):
    split_groups = []
    for path_index, path in enumerate(common_paths):
        split_group = []
        cube_size = original_data.shape

        result = auto_select_slices(cube_size, path_index, skeleton_points, 8, common_paths, connected_lines)
        axis = result['axis']
        start_point, end_point = result['head_points']
        
        slices = result['slices']
        split_points = []
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
        
        num_slit_slices = 0
        num_slices = abs(start_point - end_point) + 1
        
        for index, slice in enumerate(slices):
            fig_points = []
            
            if start_point > end_point:
                increment = -1*index
            else:
                increment = index
                
            boundaries = np.argwhere(slice==True)
            
            if boundaries.shape[0]:
                min_coords = np.min(np.argwhere(slice==True), axis=0)
                max_coords = np.max(np.argwhere(slice==True), axis=0)
                intensity_slice = select_slice(original_data, start_point + increment, axis, min_coords, max_coords)
                segment_slice = select_slice(mask_data, start_point + increment, axis, min_coords, max_coords)
                fixed_slice = select_slice(selected_data, start_point + increment, axis, min_coords, max_coords)
        
                normalized_data = (intensity_slice - intensity_slice.min()) / (intensity_slice.max() - intensity_slice.min() + 0.000001)
                
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
                        
                        if normalized_data[x_pos][y_pos] > 0.2 and segment_slice[x_pos][y_pos] == 1:
                            max_count[x_pos][y_pos] += 1
                
                for i in range(0, normalized_data.shape[0]):
                    for j in range(0, normalized_data.shape[1]):
                        if ((max_count[i][j] == 5) and (normalized_data[i][j] >= 0.5)) and ((i == 0) or (i == normalized_data.shape[0]-1) or (j == 0) or (j == normalized_data.shape[1]-1)):
                            max_count[i][j] = 8
                            
                max_indices = np.argsort(max_count, axis=None)[-2:]
                max_indices_2d = np.unravel_index(max_indices, max_count.shape)
                max_positions_pairs = list(zip(max_indices_2d[0], max_indices_2d[1]))
                
                for pair in max_positions_pairs:
                    if axis == 0:
                        point = [pair[0], pair[1]]
                        voxel_point = [start_point + increment, min_coords[1] + pair[0], min_coords[2] + pair[1]]
                    elif axis == 1:
                        point = [pair[0], pair[1]]
                        voxel_point = [min_coords[0] + pair[0], start_point + increment, min_coords[2] +  pair[1]]
                    else:
                        point = [pair[0], pair[1]]
                        voxel_point = [min_coords[0] + pair[0], min_coords[1] + pair[1], start_point + increment]

                    split_points.append(voxel_point)
                    fig_points.append(point)
                
                if euclidean_distance(max_positions_pairs[0], max_positions_pairs[1]) == 1 or not is_connectable(segment_slice, fig_points[0], fig_points[1]):
                    num_slit_slices += 1

                # visualize_slice(intensity_slice, segment_slice, fixed_slice, fig_points, start_point, increment, axis)
        
        slice_percent = num_slit_slices/num_slices
        if (slice_percent >= 0.7):
            print("Not split line")
            split_points = []
        else:
            print("Split line")

        for new_point in split_points:
            point_index = skeleton_points.shape[0]
            skeleton_points = np.vstack([skeleton_points, new_point])
            split_group.append(point_index)
        
        split_groups.append(split_group)
        
    return skeleton_points, split_groups
