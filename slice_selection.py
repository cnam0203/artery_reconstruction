import numpy as np
import math

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

def auto_select_slices(cube_size, path_index, skeleton_points, square_edge, common_paths):
    # Convert points to arrays
    path = common_paths[path_index]
    point1 = skeleton_points[path[0]]
    point2 = skeleton_points[path[1]]

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
    start_point = min(point1[axis], point2[axis])
    end_point = max(point1[axis], point2[axis])
    perpendicular_slices = []

    for i in range(start_point, end_point + increment, increment):
        intersection_point = find_intersection_point(point1, point2, [i, i, i], axis)
        # Find the square region centered at the intersection point
        square_region = find_square_region(cube_size, np.array(intersection_point), square_edge, axis, i)
        # Append slice and square region to the list
        perpendicular_slices.append(square_region)
    
    return {
        'axis': axis,
        'slices': perpendicular_slices,
        'head_points': [start_point, end_point]
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
            
    
def find_split_points(common_paths, original_data, skeleton_points):
    split_groups = []
    for path_index, path in enumerate(common_paths):
        split_group = []
        point_1 = skeleton_points[path[0]]
        point_2 = skeleton_points[path[1]]
        
        differences = np.abs(np.array(point_1) - np.array(point_2))
        cube_size = original_data.shape

        result = auto_select_slices(cube_size, path_index, skeleton_points, 16, common_paths)
        axis = result['axis']
        start_point, end_point = result['head_points']
        slices = result['slices']
        split_points = []
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
        
        for index, slice in enumerate(slices):
            boundaries = np.argwhere(slice==True)
            if boundaries.shape[0]:
                min_coords = np.min(np.argwhere(slice==True), axis=0)
                max_coords = np.max(np.argwhere(slice==True), axis=0)
                # segment_slice = select_slice(mask_data, start_point + index, axis, min_coords, max_coords)
                # fixed_slice = select_slice(selected_data, start_point + index, axis, min_coords, max_coords)
                intensity_slice = select_slice(original_data, start_point + index, axis, min_coords, max_coords)
                normalized_data = (intensity_slice - intensity_slice.min()) / (intensity_slice.max() - intensity_slice.min())
                
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
                    max_positions_pairs = list(zip(max_indices_2d[0], max_indices_2d[1]))
                
                for pair in max_positions_pairs:
                    if axis == 0:
                        point = [min_coords[1] + pair[0], min_coords[2] + pair[1]]
                        voxel_point = [start_point + index, min_coords[1] + pair[0], min_coords[2] + pair[1]]
                    elif axis == 1:
                        point = [min_coords[0] + pair[0], min_coords[2] + pair[1]]
                        voxel_point = [min_coords[0] + pair[0], start_point +  index, min_coords[2] +  pair[1]]
                    else:
                        point = [min_coords[0] + pair[0], min_coords[1] + pair[1]]
                        voxel_point = [min_coords[0] + pair[0], min_coords[1] + pair[1], start_point +  index]
                
                    split_points.append(voxel_point)
                
        for new_point in split_points:
            point_index = skeleton_points.shape[0]
            skeleton_points = np.vstack([skeleton_points, new_point])
            split_group.append(point_index)
        
        split_groups.append(split_group)
        
    return skeleton_points, split_groups
