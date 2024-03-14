import numpy as np
import math
import copy
import heapq

def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def list_not_in_lists(tuple_to_check, list_of_tuples):
    for tup in list_of_tuples:
        if tup == tuple_to_check:
            return False
    return True

def find_angle(vector1, vector2):
    """
    The function calculates the angle in degrees between two vectors using their dot product and
    magnitudes.
    
    :param vector1: It looks like you were about to provide the details of `vector1` but the input is
    missing. Could you please provide the values for `vector1` so that I can assist you further with
    calculating the angle between `vector1` and `vector2`?
    :param vector2: It seems like you have provided the code snippet for finding the angle between two
    vectors, but you haven't provided the actual values for the `vector2` parameter. In order to
    calculate the angle between two vectors, you need to provide the values for both `vector1` and
    `vector2`
    :return: The function `find_angle` calculates the angle in degrees between two vectors `vector1` and
    `vector2` using the dot product and vector magnitudes. The function returns the absolute value of
    the angle in degrees between the two vectors.
    """

    dot_product = np.dot(vector1, vector2)

    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the cosine of the angle between the vectors
    cosine_angle = dot_product / (magnitude1 * magnitude2 + 0.1)

    # Calculate the angle in radians
    angle_radians = np.arccos(cosine_angle)
    
    return abs(np.degrees(angle_radians))

def find_graph(skeleton_points):
    """
    The function `find_graph` takes a list of skeleton points, identifies endpoints and junction points
    based on Euclidean distances and angles, and returns the end points, junction points, and neighbor
    distances for each point.
    
    :param skeleton_points: It seems like the code snippet you provided is a function named `find_graph`
    that processes a list of `skeleton_points` to identify end points, junction points, and neighbor
    distances based on certain criteria involving Euclidean distances and angles between vectors
    :return: The function `find_graph` returns three values: `end_points`, `junction_points`, and
    `neighbor_distances`.
    """
    # Check which points lie below the plane defined by each point_a and its closest point_a
    end_points = []
    junction_points = []
    neighbor_distances = {}

    # Initialize neighbor distances for all points
    for i, point_a in enumerate(skeleton_points):
        neighbor_distances[i] = {}
        for j, point_b in enumerate(skeleton_points):
            neighbor_distances[i][j] = 0

    # Finding 3 neighbor points for each point_a based on Euclidean distance
    for i, point_a in enumerate(skeleton_points):
        closest_point_1 = None
        min_dist_1 = 1000000
        closest_point_2 = None
        min_dist_2 = 1000000
        closest_point_3 = None
        min_dist_3 = 1000000
        
        for j, point_b in enumerate(skeleton_points):
            if i != j:
                dist = (point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2 + (point_a[2]-point_b[2])**2
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

        # Intialize normal vectors for each point_a with its 3 neighbors
        vector_a1 = skeleton_points[closest_point_1] - point_a
        vector_a2 = skeleton_points[closest_point_2] - point_a
        vector_a3 = skeleton_points[closest_point_3] - point_a
        
        is_exist_1 = False
        is_exist_2 = False
        
        '''
        Check endpoint if there is no point which is situated in the opposite space 
        which is separated by the plane which is perpendicular to the vector created 
        by currently considered points and its two nearest points
        '''
        for j, point_b in enumerate(skeleton_points):
            # Calculate the dot product of the two vectors
            if (i != j):
                vector_ba = point_b - point_a
                # Calculate the angle in radians
                
                angle_degrees = find_angle(vector_a1, vector_ba)
                dist_2 = (point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2 + (point_a[2]-point_b[2])**2
                
                if angle_degrees > 90 and math.sqrt(dist_2) < math.sqrt(min_dist_1)*2.5:
                    is_exist_1 = True
                    break
                    
        for j, point_b in enumerate(skeleton_points):
            # Calculate the dot product of the two vectors
            if (i != j):
                vector_ba = point_b - point_a
    
                # Convert radians to degrees and take the absolute value
                angle_degrees = find_angle(vector_a2, vector_ba)
                dist_2 = (point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2 + (point_a[2]-point_b[2])**2
                
                if angle_degrees > 90 and math.sqrt(dist_2) < math.sqrt(min_dist_2)*2.5:
                    is_exist_2 = True
                    break
        
        # A point_a is considered an endpoint when there is no points below the normal planes
        if not is_exist_1 and not is_exist_2: 
            end_points.append(i)
            neighbor_distances[i][closest_point_1] = min_dist_1
        else:   # If not, check whether this points is in the junction
            angle_1 = find_angle(vector_a1, vector_a2)
            angle_2 = find_angle(vector_a2, vector_a3)
            angle_3 = find_angle(vector_a1, vector_a3)
            angle_threshold = 70
            
            # Check junction point_a
            if (angle_1 >= angle_threshold and angle_2 > angle_threshold and angle_3 > angle_threshold):
                junction_points.append(i)
                neighbor_distances[i][closest_point_1] = min_dist_1
                neighbor_distances[i][closest_point_2] = min_dist_2
                neighbor_distances[i][closest_point_3] = min_dist_3
            else:
                neighbor_distances[i][closest_point_1] = min_dist_1
                neighbor_distances[i][closest_point_2] = min_dist_2

    return end_points, junction_points, neighbor_distances

def remove_junction_points(neighbor_distances, junction_points, skeleton_points):
    """
    Remove redundant junction points from a skeleton representation based on neighbor distances.

    This function identifies redundant junction points in a skeleton represented by a neighbor distances matrix.
    Junction points are considered redundant if they have only one neighbor or are directly connected to another junction point.
    Redundant junction points are removed by resetting their connections in the neighbor distances matrix to zero.

    Args:
        neighbor_distances (list of lists): A matrix representing the distances between neighboring points.
            It should be symmetric, with distances from point i to point j being stored at neighbor_distances[i][j].
        junction_points (list): A list of indices representing the junction points in the skeleton.
        skeleton_points (list): A list of points forming the skeleton representation.

    Returns:
        tuple: A tuple containing two elements:
            - The updated list of junction points after removing redundant points.
            - The modified neighbor_distances matrix with connections to redundant junction points removed.

    Example:
        neighbor_distances = [
            [0, 3, 0, 2],
            [3, 0, 5, 0],
            [0, 5, 0, 4],
            [2, 0, 4, 0]
        ]
        junction_points = [0, 2, 3]
        skeleton_points = [(0, 0), (1, 1), (2, 2), (3, 3)]
        new_junction_points, reduced_distances = remove_junction_points(neighbor_distances, junction_points, skeleton_points)
    """
    removed_edges = []

    # Iterate through each pair of junction points
    for i in range(len(junction_points)):
        for j in range(i + 1, len(junction_points)):
            if i != j:
                junction_1 = junction_points[i]
                junction_2 = junction_points[j]
                
                # Check if there's a direct connection between the junctions
                if neighbor_distances[junction_1][junction_2] or neighbor_distances[junction_2][junction_1]:
                    count_junction_1 = 0
                    count_junction_2 = 0

                    # Count how many other junction points each junction is connected to
                    for k in range(len(junction_points)):
                        junction_3 = junction_points[k]

                        if k != i and (neighbor_distances[junction_1][junction_3] or neighbor_distances[junction_3][junction_1]):
                            count_junction_1 += 1
                        if k != j and (neighbor_distances[junction_2][junction_3] or neighbor_distances[junction_3][junction_2]):
                            count_junction_2 += 1
                        
                    # If either junction has only one connection, consider it for removal
                    if count_junction_1 == 1 or count_junction_2 == 1:
                        removed_edges.append([junction_1, junction_2])

    # Remove the edges from the neighbor distances matrix
    for edge in removed_edges:
        point_1 = edge[0]
        point_2 = edge[1]
        neighbor_distances[point_1][point_2] = 0
        neighbor_distances[point_2][point_1] = 0

    # Refine the junction points based on the updated neighbor distances
    junction_points = refine_junction_points(skeleton_points, neighbor_distances)

    return junction_points, neighbor_distances

def refine_junction_points(skeleton_points, neighbor_distances):
    """
    Identify junction points in a skeleton based on neighbor distances.

    This function identifies junction points in a skeleton represented by skeleton_points based on the neighbor distances matrix.
    A junction point is a point in the skeleton that is connected to three or more other points.

    Args:
        skeleton_points (list of tuples): A list of points forming the skeleton representation.
        neighbor_distances (list of lists): A matrix representing the distances between neighboring points.
            It should be symmetric, with distances from point i to point j being stored at neighbor_distances[i][j].

    Returns:
        list: A list containing the indices of identified junction points in the skeleton.

    Example:
        skeleton_points = [(0, 0), (1, 1), (2, 2), (3, 3)]
        neighbor_distances = [
            [0, 3, 0, 2],
            [3, 0, 5, 0],
            [0, 5, 0, 4],
            [2, 0, 4, 0]
        ]
        junction_points = refine_junction_points(skeleton_points, neighbor_distances)
    """
    junction_points = []

    # Iterate through each point in the skeleton
    for i in range(len(skeleton_points)):
        count = 0
        # Count how many other points each point is connected to
        for j in range(len(skeleton_points)):
            if i != j and (neighbor_distances[i][j] > 0 or neighbor_distances[j][i] > 0):
                count += 1
        
        # If a point is connected to three or more other points, consider it a junction point
        if count >= 3:
            junction_points.append(i)

    return junction_points

def merge_lines(index, point_1, point_2, lines, junction_points, end_points, skeleton_points):
    """
    Merge lines in the skeleton representation based on given points.

    This function merges lines in the skeleton representation based on the given points and returns the updated list of lines.

    Args:
        index (int): The index of the point to be merged.
        point_1 (int): The index of the first point to merge.
        point_2 (int): The index of the second point to merge.
        lines (list): A list of lines in the skeleton representation.
        junction_points (list): A list of indices representing the junction points in the skeleton.
        end_points (list): A list of indices representing the end points in the skeleton.
        skeleton_points (list): A list of points forming the skeleton representation.

    Returns:
        list: The updated list of lines after merging.

    Example:
        lines = [[0, 1], [1, 2], [2, 3]]
        junction_points = [1, 2]
        end_points = [0, 3]
        skeleton_points = [(0, 0), (1, 1), (2, 2), (3, 3)]
        merged_lines = merge_lines(1, 0, 2, lines, junction_points, end_points, skeleton_points)
    """
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

    return connected_lines

def reduce_skeleton_points(neighbor_distances, junction_points, end_points, skeleton_points):
    """
    Reduce redundant points in the skeleton representation based on neighbor distances.

    This function reduces redundant points in the skeleton representation by removing points
    that are not junction points or end points and have fewer than three connections to other points.

    Args:
        neighbor_distances (list of lists): A matrix representing the distances between neighboring points.
            It should be symmetric, with distances from point i to point j being stored at neighbor_distances[i][j].
        junction_points (list): A list of indices representing the junction points in the skeleton.
        end_points (list): A list of indices representing the end points in the skeleton.
        skeleton_points (list): A list of points forming the skeleton representation.

    Returns:
        tuple: A tuple containing three elements:
            - The updated list of junction points after removing redundant points.
            - The modified neighbor_distances matrix with connections to redundant points removed.
            - The list of connected lines after merging redundant points.

    Example:
        neighbor_distances = [
            [0, 3, 0, 2],
            [3, 0, 5, 0],
            [0, 5, 0, 4],
            [2, 0, 4, 0]
        ]
        junction_points = [0, 2, 3]
        end_points = [1]
        skeleton_points = [(0, 0), (1, 1), (2, 2), (3, 3)]
        new_junction_points, reduced_distances, connected_lines = reduce_skeleton_points(
            neighbor_distances, junction_points, end_points, skeleton_points)
    """
    is_removed = True
    head_points = {}
    reduced_distances = copy.deepcopy(neighbor_distances)
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

                for j in range(len(reduced_distances)):
                    if reduced_distances[index][j] > 0 or reduced_distances[j][index] > 0:
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
                            reduced_distances[index][point_1] = 0
                            reduced_distances[index][point_2] = 0
                            reduced_distances[point_1][index] = 0
                            reduced_distances[point_2][index] = 0

                            pos_1 = skeleton_points[point_1]
                            pos_2 = skeleton_points[point_2]
                            distance = (pos_1[0] - pos_2[0])**2 + (pos_1[1] - pos_2[1])**2 + (pos_1[2] - pos_2[2])**2
                            reduced_distances[point_1][point_2] = distance
                            reduced_distances[point_2][point_1] = distance

                            connected_lines = merge_lines(index, point_1, point_2, connected_lines, junction_points, end_points, skeleton_points)
               
    for point_1 in head_points:
        for point_2 in head_points[point_1]:
            list_points = head_points[point_1][point_2]
            unique_points = list(set(list_points))
            if len(unique_points) == 1:
                index = unique_points[0]
                reduced_distances[index][point_1] = 0
                reduced_distances[index][point_2] = 0
                reduced_distances[point_1][index] = 0
                reduced_distances[point_2][index] = 0
                pos_1 = skeleton_points[point_1]
                pos_2 = skeleton_points[point_2]
                distance = (pos_1[0] - pos_2[0])**2 + (pos_1[1] - pos_2[1])**2 + (pos_1[2] - pos_2[2])**2
                reduced_distances[point_1][point_2] = distance
                reduced_distances[point_2][point_1] = distance

                connected_lines = merge_lines(index, point_1, point_2, connected_lines, junction_points, end_points, skeleton_points)

    # print(junction_points)
    # print(end_points)

    junction_points = refine_junction_points(skeleton_points, reduced_distances)
    return junction_points, reduced_distances, connected_lines

def remove_cycle_edges(skeleton_points, reduced_distances, neighbor_distances, v2, connected_lines):
    """
    Remove cycle edges from the skeleton representation based on a reference vector.

    This function removes cycle edges from the skeleton representation based on the given reference vector and returns the updated data.

    Args:
        skeleton_points (list): A list of points forming the skeleton representation.
        neighbor_distances (list of lists): A matrix representing the distances between neighboring points.
            It should be symmetric, with distances from point i to point j being stored at neighbor_distances[i][j].
        reference_vector (tuple): A reference vector used for determining the angles.

    Returns:
        tuple: A tuple containing three elements:
            - The updated list of junction points after removing cycle edges.
            - The modified neighbor_distances matrix with cycle edges removed.
            - A list of edges that were removed.

    Example:
        skeleton_points = [(0, 0), (1, 1), (2, 2), (3, 3)]
        neighbor_distances = [
            [0, 3, 0, 2],
            [3, 0, 5, 0],
            [0, 5, 0, 4],
            [2, 0, 4, 0]
        ]
        reference_vector = (1, 0)
        new_junction_points, reduced_distances, removed_edges = remove_cycle_edges(
            skeleton_points, neighbor_distances, reference_vector)
    """
    m = len(reduced_distances)
    remove_edges = []
    
    # Iterate through each point
    for i in range(m):
        for j in range(i+1, m):
            for k in range(j+1, m):
                # Check if i, j, k form a cycle
                if (reduced_distances[i][j] > 0 or reduced_distances[j][i] > 0) and (reduced_distances[j][k] > 0 or reduced_distances[k][j] > 0) and (reduced_distances[k][i] > 0 or reduced_distances[i][k] > 0):
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
                        reduced_distances[i][j] = 0
                        reduced_distances[j][i] = 0
                        remove_edges.append([i, j])
                    elif min_angle == angle_2:
                        reduced_distances[j][k] = 0
                        reduced_distances[k][j] = 0
                        remove_edges.append([j, k])
                    else:
                        reduced_distances[i][k] = 0
                        reduced_distances[k][i] = 0
                        remove_edges.append([i, k])

    junction_points = refine_junction_points(skeleton_points, reduced_distances)
    
    for edge in remove_edges:
        point_1 = edge[0]
        point_2 = edge[1]
        for line in connected_lines:
            if (point_1 == line[0] or point_1 == line[-1]) and (point_2 == line[0] or point_2 == line[-1]):
                for i in range(len(line) - 1):
                    neighbor_distances[line[i]][line[i+1]] = 0
                    neighbor_distances[line[i+1]][line[i]] = 0
                        
    return junction_points, neighbor_distances

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

def remove_duplicate_paths(skeleton_points, reduced_distances, connected_lines):
    edges = []
    for i in range(skeleton_points.shape[0]):
        for j in range(i+1, skeleton_points.shape[0]):
            if i != j and (reduced_distances[i][j] > 0 or reduced_distances[j][i] > 0):
                edges.append([i, j])

    duplicate_lines = {}
    for line in connected_lines:
        if line[0] != line[-1]:
            point_1 = line[0]
            point_2 = line[-1]

            if point_1 > point_2:
                tmp = point_1
                point_1 = point_2
                point_2 = tmp
                line.reverse()

            if point_1 not in duplicate_lines:
                duplicate_lines[point_1] = {}
            
            if point_2 not in duplicate_lines[point_1]:
                duplicate_lines[point_1][point_2] = []

            duplicate_lines[point_1][point_2].append(line)

    for key, value in duplicate_lines.items():
        for sub_key, sub_value in value.items():
            if len(sub_value) == 2:
                line_1 = sub_value[0]
                line_2 = sub_value[1]
                interpolate_line = interpolate_center_path(skeleton_points[line_1], skeleton_points[line_2], 1)

                removed_idx = []
                for i, line in enumerate(connected_lines):
                    if (line[0] == key and line[-1] == sub_key) or (line[-1] == key and line[0] == sub_key):
                        removed_idx.append(i)
                    
                removed_idx.sort(reverse=True)
                for index in removed_idx:
                    connected_lines.pop(index)

                new_line = [key]
                for i in range(1, len(interpolate_line)-1):
                    new_point = interpolate_line[i]
                    new_index = skeleton_points.shape[0]
                    skeleton_points = np.vstack([skeleton_points, new_point])
                    new_line.append(new_index)

                new_line.append(sub_key)
                connected_lines.append(new_line)

                removed_edges = []
                for i, edge in enumerate(edges):
                    edge_1 = edge[0]
                    edge_2 = edge[-1]

                    if (edge_1 in line_1 and edge_2 in line_1) or (edge_1 in line_2 and edge_2 in line_2):
                        removed_edges.append(i)

                removed_edges.sort(reverse=True)
                for index in removed_edges:
                    edges.pop(index)
                
                edges.append([key, sub_key])
                
    return skeleton_points, connected_lines, edges

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
    return math.acos(cosine_angle)  # Angle in radian
                                        
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

def split_direction(principal_vectors, other_vectors):
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

def find_branches(aca_endpoints, end_points, directions, skeleton_points, junction_points, edges):
    main_trunks = []
    line_weights = []
    for i, endpoint in enumerate(aca_endpoints):
        list_paths = []
        for point in end_points:
            if point not in aca_endpoints:
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
        
        start_point = endpoint
        visited_edges = []
        main_points = [start_point]

        while start_point != -1:
            connected_edges = []

            for key, value in path_weights.items():
                if start_point in key and list_not_in_lists(key, visited_edges):
                    connected_edges.append(key)

            if len(connected_edges) == 0:
                start_point = -1
            else:
                weights = [path_weights[key] for key in connected_edges]
                max_value = max(weights)
                max_positions = [i for i, weight in enumerate(weights) if weight == max_value]
                
                if len(max_positions) > 1:
                    start_point = -1
                else:
                    edge = connected_edges[max_positions[0]]
                    if start_point == edge[0]:
                        start_point = edge[1]
                    else:
                        start_point = edge[0]
                    visited_edges.append(edge)
                    main_points.append(start_point)
        
        main_trunk = []
        for idx in range(len(main_points) - 1):
            if main_points[idx] < main_points[idx+1]:
                main_trunk.append([main_points[idx], main_points[idx+1]])
            else:
                main_trunk.append([main_points[idx+1], main_points[idx]])
        
        main_trunks.append(main_trunk)
        line_weights.append(path_weights)

    common_paths = []

    # Convert each sublist to sets for easier comparison
    set1 = {tuple(sublist) for sublist in main_trunks[0]}
    set2 = {tuple(sublist) for sublist in main_trunks[1]}

    # Find the common 2-element lists
    common_paths = set1.intersection(set2)
    common_paths = [list(sublist) for sublist in common_paths]
    left_paths = set1 - set2
    left_paths = [list(sublist) for sublist in left_paths]
    right_paths = set2 - set1
    right_paths = [list(sublist) for sublist in right_paths]

    num_undefined_paths_old = None
    num_undefined_paths_new = None
    
    while num_undefined_paths_new == None or num_undefined_paths_new != num_undefined_paths_old:
        num_undefined_paths_old = num_undefined_paths_new
        
        common_points = list(set([item for sublist in common_paths for item in sublist]))
        left_points = list(set([item for sublist in left_paths for item in sublist]))
        right_points = list(set([item for sublist in right_paths for item in sublist]))
        diff_left_points = set(left_points) - set(right_points) - set(common_points)
        diff_left_points = list(diff_left_points)
        diff_right_points = set(right_points) - set(left_points) - set(common_points)
        diff_right_points = list(diff_right_points)
        
        new_left_paths = find_rest_paths(edges, diff_left_points, left_paths)
        new_right_paths = find_rest_paths(edges, diff_right_points, right_paths)   

        left_paths = set(map(tuple, left_paths + new_left_paths))
        left_paths = [list(sublist) for sublist in left_paths]
        right_paths = set(map(tuple, right_paths + new_right_paths))
        right_paths = [list(sublist) for sublist in right_paths]

        diff_left_points = list(set([item for sublist in left_paths for item in sublist]))
        diff_right_points = list(set([item for sublist in right_paths for item in sublist]))
        
        undefined_paths = []
        for edge in edges:
            if list_not_in_lists(edge, common_paths) and list_not_in_lists(edge, left_paths) and list_not_in_lists(edge, right_paths):
                undefined_paths.append(edge)

        num_undefined_paths_new = len(undefined_paths)
        undefined_points = list(set([item for sublist in undefined_paths for item in sublist]))
        suspected_points = {}

        for point in junction_points:
            if (point in undefined_points) and ((point in diff_left_points) or (point in diff_right_points) or (point in common_points)):
                suspected_points[point] = {}

                for edge in edges:
                    if point == edge[0]:
                        point_1 = edge[0]
                        point_2 = edge[1]
                    elif point == edge[1]:
                        point_1 = edge[1]
                        point_2 = edge[0]
                    else:
                        continue

                    w = 0
                    if edge in common_paths:
                        w = 0
                    elif edge in left_paths:
                        w = 1
                    elif edge in right_paths:
                        w = 2
                    else:
                        max_w = 0
                        w1 = 0
                        w2 = 0

                        edge_key = tuple(edge)
                        if edge_key in line_weights[0]:
                            w1 = line_weights[0][edge_key]
                        if edge_key in line_weights[1]:
                            w2 = line_weights[1][edge_key]
                        
                        max_w = max(w1, w2)

                        if max_w == 1:
                            w = -1
                        else:
                            w = 3

                    suspected_points[point][point_2] = w

        for key, sub_dict in suspected_points.items():
            found_one = False
            found_two = False
            found_zero = False
            zero_count = 0
            
            check_edges = []
            
            for sub_key, value in sub_dict.items():
                if value == 1:
                    found_one = True
                    one_key = sub_key
                elif value == 2:
                    found_two = True
                    two_key = sub_key
                elif value == 0:
                    found_zero = True
                    zero_count += 1
                    zero_key = sub_key
                else:
                    check_edges.append(sub_key)

            if found_one and found_two:
                principal_vector_1 = skeleton_points[key] - skeleton_points[one_key]
                principal_vector_2 = skeleton_points[key] - skeleton_points[two_key]
                other_vectors = [skeleton_points[key] - skeleton_points[point] for point in check_edges]
                left_index, right_index = split_direction([principal_vector_1, principal_vector_2], other_vectors)
                left_point = check_edges[left_index]
                right_point = check_edges[right_index]

                if key < left_point:
                    path1 = [key, left_point]
                else:
                    path1 = [left_point, key]

                if key < right_point:
                    path2 = [key, right_point]
                else:
                    path2 = [right_point, key]

                left_paths = set(map(tuple, left_paths + [path1]))
                left_paths = [list(sublist) for sublist in left_paths]
                right_paths = set(map(tuple, right_paths + [path2]))
                right_paths = [list(sublist) for sublist in right_paths]

        undefined_paths = []
        for edge in edges:
            if list_not_in_lists(edge, common_paths) and list_not_in_lists(edge, left_paths) and list_not_in_lists(edge, right_paths):
                undefined_paths.append(edge)

        num_undefined_paths_new = len(undefined_paths)
        
    return common_paths, left_paths, right_paths, undefined_paths

def connect_split_points(split_groups, skeleton_points):
    split_paths = []
    
    for group in split_groups:
        # visualized_split_points.append(generate_points(skeleton_points[group]))
        left_paths = []
        right_paths = []
        
        for i in range(0, len(group) - 1, 2):
            point_1 = skeleton_points[group[i]]
            point_2 = skeleton_points[group[i+1]]
            
            if i == 0:
                left_paths.append(group[i])
                right_paths.append(group[i+1])
            
            else:
                left_point = skeleton_points[left_paths[-1]]
                right_point = skeleton_points[right_paths[-1]]
                
                distance_1 = euclidean_distance(point_1, left_point)
                distance_2 = euclidean_distance(point_1, right_point)
                
                if distance_1 < distance_2:
                    left_paths.append(group[i])
                    right_paths.append(group[i+1])
                else:
                    left_paths.append(group[i+1])
                    right_paths.append(group[i])
                    
        split_paths.append[[left_paths, right_paths]]
    
    return split_paths