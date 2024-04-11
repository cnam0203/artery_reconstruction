import numpy as np
import math
import copy
import heapq
import plotly.graph_objs as go
from scipy.ndimage import convolve
from skimage.draw import line_nd
import skan
from scipy.spatial import KDTree, distance_matrix

def find_graphs(skeleton):
    # skeleton_points = np.argwhere(skeleton != 0)
    skeleton_skan = skan.Skeleton(skeleton)
    skeleton_points = skeleton_skan.coordinates

    connected_lines = []
    for i in range(skeleton_skan.n_paths):
        connected_lines.append(skeleton_skan.path(i).tolist())
    
    count_points = {}
    end_points = []
    junction_points = []
    
    for line in connected_lines:
        head_point_1, head_point_2 = line[0], line[-1]
        
        if head_point_1 not in count_points:
            count_points[head_point_1] = 0
        count_points[head_point_1] += 1
        
        if head_point_2 not in count_points:
            count_points[head_point_2] = 0
        count_points[head_point_2] += 1
                    
    for key, value in count_points.items():
        if value == 1:
            end_points.append(key)
        elif value > 2:
            junction_points.append(key)
                
    return skeleton_points, end_points, junction_points, connected_lines


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

def connected_neighbors_prob(point_a, point_b, segment_data):
    line_indices = line_nd(point_a, point_b)
    voxel_positions = np.transpose(line_indices)
    num_points = voxel_positions.shape[0]
    valid_points = 0
    
    for point in voxel_positions:
        x, y, z = point[0], point[1], point[2]
        if segment_data[x][y][z]:
            valid_points += 1
            
    return valid_points/num_points
    
def find_graph(skeleton_points, cex_data):
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
        
        distance_gap = 2
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
                
                if angle_degrees > 90 and math.sqrt(dist_2) < math.sqrt(min_dist_1)*distance_gap:
                    if connected_neighbors_prob(point_a, point_b, cex_data) > 0.5:
                        is_exist_1 = True
                        
                        break
                    
        for j, point_b in enumerate(skeleton_points):
            # Calculate the dot product of the two vectors
            if (i != j):
                vector_ba = point_b - point_a
    
                # Convert radians to degrees and take the absolute value
                angle_degrees = find_angle(vector_a2, vector_ba)
                dist_2 = (point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2 + (point_a[2]-point_b[2])**2
                
                if angle_degrees > 90 and math.sqrt(dist_2) < math.sqrt(min_dist_2)*distance_gap:
                    if connected_neighbors_prob(point_a, point_b, cex_data) > 0.5:
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
                
    for i in range(0, len(neighbor_distances)):
        for j in range(i, len(neighbor_distances)):
            if neighbor_distances[i][j] > 0 and neighbor_distances[j][i] == 0:
                neighbor_distances[j][i] = neighbor_distances[i][j]
            elif neighbor_distances[i][j] == 0 and neighbor_distances[j][i] > 0:
                neighbor_distances[i][j] = neighbor_distances[j][i]
    
    junction_points = refine_junction_points(skeleton_points, neighbor_distances)

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
                            
    # print(connected_lines)           
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
            else:
                for index in unique_points:
                    remove_idx = []
                    new_lines = []
                    found_1 = False
                    found_2 = False
                    
                    for idx, line in enumerate(connected_lines):
                        if index in line and point_1 in line and point_2 in line:
                            split_pos = line.index(index)
                            new_line_1, new_line_2 = line[:split_pos + 1], line[split_pos:]
                            new_lines.append(new_line_1)
                            new_lines.append(new_line_2)
                            remove_idx.append(idx)
                        elif index in line and point_1 in line:
                            found_1 = True
                        elif index in line and point_2 in line:
                            found_2 = True
                     
                    if not found_1:
                        new_lines.append([index, point_1])
                    if not found_2:
                        new_lines.append([index, point_2])  
                             
                    for i in sorted(remove_idx, reverse=True):
                        del connected_lines[i]
                    connected_lines += new_lines
                    
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

def calculate_edge_cost(current_edge, new_edge, nodes_positions, direction, connected_lines):
    if current_edge is None:
        return 0  # Initial edge cost
    else:
        # Calculate the angle between the current edge and the new edge
        selected_line = None
        for line in connected_lines:
            if (line[0] == current_edge[1] and line[-1] == new_edge[1]) or (line[-1] == current_edge[1] and line[0] == new_edge[1]):
                selected_line = line
                break
            
        current_vector = calculate_vector(nodes_positions[current_edge[0]], nodes_positions[current_edge[1]])
        max_angle = 0
    
        for point in selected_line:
            if point != current_edge[1]:
                new_vector = calculate_vector(nodes_positions[current_edge[1]], nodes_positions[point])
                angle = calculate_angle(current_vector, new_vector)
                if angle > max_angle:
                    max_angle = angle

        if direction == 'left':
            return max_angle
        elif direction == 'right':
            return -max_angle

def calculate_vector(point1, point2):
    return (point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2])

def calculate_angle(vector1, vector2):
    dot_product = sum(x * y for x, y in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(x**2 for x in vector1))
    magnitude2 = math.sqrt(sum(x**2 for x in vector2))
    cosine_angle = dot_product / (magnitude1 * magnitude2 + 0.001)
    return math.acos(cosine_angle)  # Angle in radian
                                        
def dijkstra(edges, start, end, skeleton_points, direction, connected_lines):
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
                    new_cost = cost + calculate_edge_cost(current_edge, new_edge, skeleton_points, direction, connected_lines)
                    heapq.heappush(pq, (new_cost, neighbor, path, new_edge))
    return float('inf'), []

def find_min_distance(line_idx_1, line_idx_2, connected_lines, skeleton_points, key):
    line_1 = connected_lines[line_idx_1]
    line_2 = connected_lines[line_idx_2]
    
    if line_1[0] == line_2[0] == key:
        return euclidean_distance(skeleton_points[line_1[1]], skeleton_points[line_2[1]])
    elif line_1[0] == line_2[-1] == key:
        return euclidean_distance(skeleton_points[line_1[1]], skeleton_points[line_2[-2]])
    elif line_1[-1] == line_2[0] == key:
        return euclidean_distance(skeleton_points[line_1[-2]], skeleton_points[line_2[1]])
    elif line_1[-1] == line_2[-1] == key:
        return euclidean_distance(skeleton_points[line_1[-2]], skeleton_points[line_2[-2]])
    else:
        return 0
    
def split_direction(principal_vectors, other_vectors, connected_lines, skeleton_points, key):
    left_vectors = []
    right_vectors = []

    a11 = find_min_distance(principal_vectors[0], other_vectors[0], connected_lines, skeleton_points, key)
    a12 = find_min_distance(principal_vectors[0], other_vectors[1], connected_lines, skeleton_points, key)
    a21 = find_min_distance(principal_vectors[1], other_vectors[0], connected_lines, skeleton_points, key)
    a22 = find_min_distance(principal_vectors[1], other_vectors[1], connected_lines, skeleton_points, key)
    min_distance = min(a11, a12, a21, a22)

    if min_distance == a11 or min_distance == a22:
        return 0, 1
    else:
        return 1, 0
    
def find_rest_paths(edges, points, trunk):
    visited_points = copy.deepcopy(points)
    new_paths = []


    for edge in edges:
        point_1 = edge[0]
        point_2 = edge[1]

        if ((point_1 in visited_points) and (point_2 not in visited_points)) or ((point_1 not in visited_points) and (point_2 in visited_points)):
            new_paths.append(edge)
    return new_paths

class Node:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.adjacent = []  # List of adjacent nodes

def dfs_path_priority(start, end, connected_lines, skeleton_points, direction):
    # Priority queue to store nodes based on decreasing z-axis values and greatest x-axis
    if direction == 'left':
        sign = 1
    else:
        sign = -1
        
    priority_queue = [(sign*skeleton_points[start][0], start, [start])]
    visited = []
    
    while priority_queue:
        ignore_paths = []
        is_found = False

        while len(priority_queue):
            neg_x, node_id, path = heapq.heappop(priority_queue)

            if len(visited) == 0 or path[-2] == visited[-1]:
                is_found = True
                break
            else:
                ignore_paths.append((neg_x, node_id, path))
        
        for item in ignore_paths:
            heapq.heappush(priority_queue, item)
        
        if not is_found:
            neg_x, node_id, path = heapq.heappop(priority_queue)

        current_node = path[-1]

        if current_node == end:
            return path  # Found the end node

        visited.append(current_node)

        # Explore adjacent nodes
        adjacent = []
        for idx, line in enumerate(connected_lines):
            if current_node == line[0]:
                avg_x = np.mean(skeleton_points[line[1:3]][:, 0])
                adjacent.append([line[-1], idx, avg_x])
            elif current_node == line[-1]:
                avg_x = np.mean(skeleton_points[line[-3:-1]][:, 0])
                adjacent.append([line[0], idx, avg_x])
        
        for adj_node in adjacent:
            if adj_node[0] not in visited:
                new_path = list(path)
                new_path.append(adj_node[0])
                # Priority is based on decreasing z-axis and number of nodes explored
                priority = (sign*adj_node[2], -len(new_path))
                heapq.heappush(priority_queue, (priority, adj_node[0], new_path))

    return None  # No path found

# def dfs_path_priority(start, end, connected_lines, skeleton_points, direction):
#     # Priority queue to store nodes based on decreasing z-axis values and greatest x-axis
#     if direction == 'left':
#         sign = 1
#     else:
#         sign = -1
        
#     priority_queue = [(sign*skeleton_points[start][0], start, [start])]
#     visited = set()

#     while priority_queue:
#         neg_x, node_id, path = heapq.heappop(priority_queue)
#         current_node = path[-1]

#         if current_node == end:
#             return path  # Found the end node

#         visited.add(current_node)

#         # Explore adjacent nodes
#         adjacent = []
#         for idx, line in enumerate(connected_lines):
#             if current_node == line[0]:
#                 adjacent.append([line[-1], idx])
#             elif current_node == line[-1]:
#                 adjacent.append([line[0], idx])
        
#         for adj_node in adjacent:
#             if adj_node[0] not in visited:
#                 new_path = list(path)
#                 new_path.append(adj_node[0])
#                 # Priority is based on decreasing z-axis and greatest x-axis among decreasing z-axis nodes
#                 position_with_max_abs = np.argmax(np.abs(skeleton_points[connected_lines[adj_node[1]]][:, 0]))
#                 priority = (sign*skeleton_points[connected_lines[adj_node[1]][position_with_max_abs]][0])
#                 heapq.heappush(priority_queue, (priority, adj_node[0], new_path))

#     return None  # No path found
    
# def dfs_path_priority(start, end, connected_lines, skeleton_points, direction):
#     # Priority queue to store nodes based on decreasing z-axis values and greatest x-axis
#     if direction == 'left':
#         sign = 1
#     else:
#         sign = -1
        
#     priority_queue = [(sign*skeleton_points[start][0], start, [start])]
#     visited = set()

#     while priority_queue:
#         neg_x, node_id, path = heapq.heappop(priority_queue)
#         current_node = path[-1]

#         if current_node == end:
#             return path  # Found the end node

#         visited.add(current_node)

#         # Explore adjacent nodes
#         adjacent = []
#         for idx, line in enumerate(connected_lines):
#             if current_node == line[0]:
#                 adjacent.append([line[-1], idx])
#             elif current_node == line[-1]:
#                 adjacent.append([line[0], idx])
        
#         for adj_node in adjacent:
#             if adj_node[0] not in visited:
#                 new_path = list(path)
#                 new_path.append(adj_node[0])
#                 # Priority is based on decreasing z-axis and greatest x-axis among decreasing z-axis nodes
#                 position_with_max_abs = np.argmax(np.abs(skeleton_points[connected_lines[adj_node[1]]][:, 0]))
#                 priority = (sign*skeleton_points[connected_lines[adj_node[1]][position_with_max_abs]][0])
#                 heapq.heappush(priority_queue, (priority, adj_node[0], new_path))

#     return None  # No path found
    
def find_branches(aca_endpoints, end_points, directions, skeleton_points, junction_points, edges, connected_lines, connected_points):
    main_trunks = []
    line_weights = []
    for i, endpoint in enumerate(aca_endpoints):
        list_paths = []
        for point in end_points:
            if point not in aca_endpoints:
                # shortest_cost, shortest_path = dijkstra(edges, endpoint, point, skeleton_points, directions[i], connected_lines)
                shortest_path = dfs_path_priority(point, endpoint, connected_lines, skeleton_points, directions[i])
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
    undefined_paths = []
    for edge in edges:
        if list_not_in_lists(edge, common_paths) and list_not_in_lists(edge, left_paths) and list_not_in_lists(edge, right_paths):
            undefined_paths.append(edge)
            
    num_undefined_paths_old = 0
    num_undefined_paths_new = len(undefined_paths)
    
    while num_undefined_paths_old != num_undefined_paths_new:
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
        
        left_set = set(map(tuple, new_left_paths))
        right_set = set(map(tuple, new_right_paths))
        common_sets = left_set.intersection(right_set)
        new_left_paths = [path for path in new_left_paths if tuple(path) not in common_sets]
        new_right_paths = [path for path in new_right_paths if tuple(path) not in common_sets]
        
        left_set = set(map(tuple, left_paths))
        right_set = set(map(tuple, right_paths))
        common_sets = left_set.intersection(right_set)

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
        
        undefined_points = list(set([item for sublist in undefined_paths for item in sublist]))
        suspected_points = {}

        for point in junction_points:
            if (point in undefined_points) and ((point in diff_left_points) or (point in diff_right_points) or (point in common_points)):
                suspected_points[point] = {}

                for idx, edge in enumerate(edges):
                    if point == edge[0]:
                        point_1 = edge[0]
                    elif point == edge[1]:
                        point_1 = edge[1]
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

                    suspected_points[point][idx] = w

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
                principal_vector_1 = one_key
                principal_vector_2 = two_key
                other_vectors = check_edges
                
                if len(other_vectors) == 2:
                    left_index, right_index = split_direction([principal_vector_1, principal_vector_2], other_vectors, connected_lines, skeleton_points, key)
                    path1 = edges[check_edges[left_index]]
                    path2 = edges[check_edges[right_index]]

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

def direction_vector(p1, p2):
    return np.array(p2) - np.array(p1)

def angle_between(v1, v2):
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms == 0:
        return 0
    return np.arccos(np.clip(dot_product / norms, -1.0, 1.0))

def find_parallel_lines_top_view(p1, p2, p3, p4, skeleton_points):
    direction_vector1 = direction_vector(skeleton_points[p1], skeleton_points[p3])
    direction_vector2 = direction_vector(skeleton_points[p2], skeleton_points[p4])

    # Project vectors onto the xy-plane (ignoring z-axis)
    direction_vector1[2] = 0
    direction_vector2[2] = 0

    angle1 = angle_between(direction_vector1, direction_vector2)

    direction_vector1 = direction_vector(skeleton_points[p1], skeleton_points[p4])
    direction_vector2 = direction_vector(skeleton_points[p2], skeleton_points[p3])

    # Project vectors onto the xy-plane (ignoring z-axis)
    direction_vector1[2] = 0
    direction_vector2[2] = 0

    angle2 = angle_between(direction_vector1, direction_vector2)

    if angle1 < angle2:
        return (p1, p3), (p2, p4)
    else:
        return (p1, p4), (p2, p3)

def connect_split_points(split_groups, skeleton_points, undefined_paths, connected_lines):
    split_paths = []
    
    for group in split_groups:
        # visualized_split_points.append(generate_points(skeleton_points[group]))
        left_paths = []
        right_paths = []
        
        for i in range(0, len(group) - 1, 2):
            
            if i == 0:
                left_paths.append(group[i])
                right_paths.append(group[i+1])
            
            else:
                point_1 = group[i]
                point_2 = group[i+1]

                left_point = left_paths[-1]
                right_point = right_paths[-1]

                distance_11 = euclidean_distance(skeleton_points[point_1], skeleton_points[left_point])
                distance_12 = euclidean_distance(skeleton_points[point_1], skeleton_points[right_point])
                distance_21 = euclidean_distance(skeleton_points[point_2], skeleton_points[left_point])
                distance_22 = euclidean_distance(skeleton_points[point_2], skeleton_points[right_point])

                if distance_11 + distance_22 < distance_12 + distance_21:
                    left_paths.append(point_1)
                    right_paths.append(point_2)
                else:
                    left_paths.append(point_2)
                    right_paths.append(point_1)

        split_paths.append([left_paths, right_paths])
    
    # print(undefined_paths, connected_lines)
    return split_paths, undefined_paths, connected_lines


def find_adjacent_paths(common_point, left_paths, right_paths, connected_lines):
    left_path_idx, right_path_idx, left_connected_line_idx, right_connected_line_idx = None, None, None, None
    
    for i, path in enumerate(left_paths):
        if path[0] == common_point or path[1] == common_point:
            left_path_idx = i
            
            for j, line in enumerate(connected_lines):
                if (path[0] == line[0] and path[1] == line[-1]) or (path[0] == line[-1] and path[1] == line[0]):
                    left_connected_line_idx = j
                    break
            
            break
        
    for i, path in enumerate(right_paths):
        if path[0] == common_point or path[1] == common_point:
            right_path_idx = i
            
            for j, line in enumerate(connected_lines):
                if (path[0] == line[0] and path[1] == line[-1]) or (path[0] == line[-1] and path[1] == line[0]):
                    right_connected_line_idx = j
                    break
            
            break
    
    return left_path_idx, right_path_idx, left_connected_line_idx, right_connected_line_idx
    
def find_shortest_distance(start_point, list_points):
    min_distance = 1000000
    min_index = None
    
    for i, end_point in enumerate(list_points):
        distance = euclidean_distance(start_point, end_point)
        
        if distance < min_distance and distance != 0:
            min_distance = distance
            min_index = i
            
    return min_distance, min_index


def connect_common_paths(common_paths, left_paths, right_paths, connected_lines, split_paths, skeleton_points, undefined_paths):
    defined_paths = [False] * len(common_paths)
    is_found = True
    expanded_points = {}
    
    if len(left_paths) == 0 and len(right_paths) == 0 and len(common_paths) > 0:
        common_path = common_paths[0]
        head_pos = 0
        defined_paths[0] = [0, 1]
        
        for path in undefined_paths:
            common_point = None
            
            if common_path[0] == path[0] or common_path[0] == path[1]:
                common_point = common_path[0]
            
            if common_point is None:
                if common_path[1] == path[0] or common_path[1] == path[1]:
                    common_point = common_path[1]
                    head_pos = 1
            
            if common_point is not None:
                left_split_path = copy.deepcopy(split_paths[0][0])
                right_split_path = copy.deepcopy(split_paths[0][1])
                
                if head_pos == 0:
                    left_paths.append([split_paths[0][0][-1], split_paths[0][0][0]])
                    right_paths.append([split_paths[0][1][-1], split_paths[0][1][0]])
                    connected_lines.append(left_split_path)
                    connected_lines.append(right_split_path)
                else:
                    left_paths.append([split_paths[0][0][-1], split_paths[0][0][0]])
                    right_paths.append([split_paths[0][1][-1], split_paths[0][1][0]])
                    connected_lines.append(left_split_path)
                    connected_lines.append(right_split_path)
                    
                expanded_points[common_point] = [connected_lines[-2], connected_lines[-1]]
                break
    
    while is_found:
        is_found = False
        
        for idx, path in enumerate(common_paths):
            if defined_paths[idx] == False:
                common_point_1 = path[0]
                common_point_2 = path[1]
                split_path = split_paths[idx]
                
                selected_point = 0
                left_path_idx, right_path_idx, left_connected_line_idx, right_connected_line_idx = find_adjacent_paths(common_point_1, left_paths, right_paths, connected_lines)
                
                if left_path_idx == None:
                    selected_point = 1
                    left_path_idx, right_path_idx, left_connected_line_idx, right_connected_line_idx = find_adjacent_paths(common_point_2, left_paths, right_paths, connected_lines)
                
                    if left_path_idx == None:
                        continue
                
                is_found = True
                
                left_path = left_paths[left_path_idx]
                right_path = right_paths[right_path_idx]

                left_connected_line = connected_lines[left_connected_line_idx]
                right_connected_line = connected_lines[right_connected_line_idx]
                
                if selected_point == 0:
                    adjacent_point = common_point_1
                    split_point_1 = split_path[0][0]
                    split_point_2 = split_path[1][0]
                    next_split_point_1 = split_path[0][1]
                    next_split_point_2 = split_path[1][1]
                else:
                    adjacent_point = common_point_2
                    split_point_1 = split_path[0][-1]
                    split_point_2 = split_path[1][-1]
                    next_split_point_1 = split_path[0][-2]
                    next_split_point_2 = split_path[1][-2]
                    
                if left_connected_line[0] == adjacent_point:
                    left_points = left_connected_line[2:]
                else:
                    left_points = left_connected_line[:-2]
                
                if right_connected_line[0] == adjacent_point:
                    right_points = right_connected_line[2:]
                else:
                    right_points = right_connected_line[:-2]
                    
                shortest_distance_11, point_11 = find_shortest_distance(skeleton_points[split_point_1], skeleton_points[left_points])
                shortest_distance_12, point_12 = find_shortest_distance(skeleton_points[split_point_1], skeleton_points[right_points])
                shortest_distance_21, point_21 = find_shortest_distance(skeleton_points[split_point_2], skeleton_points[left_points])
                shortest_distance_22, point_22 = find_shortest_distance(skeleton_points[split_point_2], skeleton_points[right_points])
                min_distance = min(shortest_distance_11, shortest_distance_12, shortest_distance_21, shortest_distance_22)
                
                if min_distance == shortest_distance_11 or min_distance == shortest_distance_22:
                    defined_paths[idx] = [0, 1]
                    
                    if selected_point == 0:
                        left_paths.append([next_split_point_1, common_point_2])
                        right_paths.append([next_split_point_2, common_point_2])
                    else:
                        left_paths.append([next_split_point_1, common_point_1])
                        right_paths.append([next_split_point_2, common_point_1])
                    
                    if left_path[0] == adjacent_point:
                        left_paths[left_path_idx][0] = next_split_point_1
                    else:
                        left_paths[left_path_idx][-1] = next_split_point_1
                        
                    if right_path[0] == adjacent_point:
                        right_paths[right_path_idx][0] = next_split_point_2
                    else:
                        right_paths[right_path_idx][-1] = next_split_point_2
                        
                    if left_connected_line[0] == adjacent_point:
                        new_left_line = copy.deepcopy(left_connected_line)
                        new_left_line[0] = next_split_point_1
                    else:
                        new_left_line = copy.deepcopy(left_connected_line)
                        new_left_line[-1] = next_split_point_1
                    
                        
                    if right_connected_line[0] == adjacent_point:
                        new_right_line = copy.deepcopy(right_connected_line)
                        new_right_line[0] = next_split_point_2
                    else:    
                        new_right_line = copy.deepcopy(right_connected_line)
                        new_right_line[-1] = next_split_point_2
                    
                else:
                    defined_paths[idx] = [1, 0]
                    if selected_point == 0:
                        left_paths.append([next_split_point_2, common_point_2])
                        right_paths.append([next_split_point_1, common_point_2])
                    else:
                        left_paths.append([next_split_point_2, common_point_1])
                        right_paths.append([next_split_point_1, common_point_1])
                    
                    if left_path[0] == adjacent_point:
                        left_paths[left_path_idx][0] = next_split_point_2
                    else:
                        left_paths[left_path_idx][-1] = next_split_point_2
                        
                    if right_path[0] == adjacent_point:
                        right_paths[right_path_idx][0] = next_split_point_1
                    else:
                        right_paths[right_path_idx][-1] = next_split_point_1
                        
                    if left_connected_line[0] == adjacent_point:
                        new_left_line = copy.deepcopy(left_connected_line)
                        new_left_line[0] = next_split_point_2
                    else:
                        new_left_line = copy.deepcopy(left_connected_line)
                        new_left_line[-1] = next_split_point_2
                    
                    if right_connected_line[0] == adjacent_point:
                        new_right_line = copy.deepcopy(right_connected_line)
                        new_right_line[0] = next_split_point_1
                    else:    
                        new_right_line = copy.deepcopy(right_connected_line)
                        new_right_line[-1] = next_split_point_1
                
                for i in sorted([left_connected_line_idx, right_connected_line_idx], reverse=True):
                    del connected_lines[i]
            
                connected_lines.append(new_left_line)
                connected_lines.append(new_right_line)
                
                expanded_points[adjacent_point] = [connected_lines[-2], connected_lines[-1]]
                
                split_path_1 = copy.deepcopy(split_path[0])
                split_path_2 = copy.deepcopy(split_path[1])
                
                if selected_point == 0:
                    del split_path_1[0]
                    del split_path_2[0]
                    split_path_1.append(common_point_2)
                    split_path_2.append(common_point_2)
                else:
                    del split_path_1[-1]
                    del split_path_2[-1]
                    split_path_1.insert(0, common_point_1)
                    split_path_2.insert(0, common_point_1)
                
                connected_lines.append(split_path_1)
                connected_lines.append(split_path_2)
                
    
    is_found = True

    while is_found:
        is_found = False
        ajacent_left_points = {}
        ajacent_right_points = {}
        
        for i, path in enumerate(left_paths):
            point_1 = path[0]
            point_2 = path[1]
            
            if point_1 not in ajacent_left_points:
                ajacent_left_points[point_1] = []
            if point_2 not in ajacent_left_points:
                ajacent_left_points[point_2] = []
            
            if len(path) > 1:
                ajacent_left_points[point_1].append(i)
                ajacent_left_points[point_2].append(i)
        
        for i, path in enumerate(right_paths):
            point_1 = path[0]
            point_2 = path[1]
            
            if point_1 not in ajacent_right_points:
                ajacent_right_points[point_1] = []
            if point_2 not in ajacent_right_points:
                ajacent_right_points[point_2] = []
            
            if len(path) > 1:
                ajacent_right_points[point_1].append(i)
                ajacent_right_points[point_2].append(i)
        
        for key in ajacent_left_points:
            if len(ajacent_left_points[key]) == 2:
                if key in ajacent_right_points:
                    if len(ajacent_right_points[key]) == 2:
                        is_found = True
                        new_lines = []
                        list_idx = []

                        two_left_paths = ajacent_left_points[key]
                        two_right_paths = ajacent_right_points[key]
                        
                        left_path_1 = left_paths[two_left_paths[0]]
                        left_path_2 = left_paths[two_left_paths[1]]
                        
                        connected_point_11, connected_point_12 = None, None
                        path_11, path_12 = None, None
                        
                        for idx, line in enumerate(connected_lines):
                            if (line[0] == left_path_1[0] and line[-1] == left_path_1[-1]) or (line[0] == left_path_1[-1] and line[-1] == left_path_1[0]):
                                list_idx.append(idx)
                                path_11 = line
                            elif (line[0] == left_path_2[0] and line[-1] == left_path_2[-1]) or (line[0] == left_path_2[-1] and line[-1] == left_path_2[0]):
                                list_idx.append(idx)
                                path_12 = line
                                
                        new_line_11 = []
                        new_line_12 = []
                        
                        if path_11[0] == key:
                            connected_point_11 = path_11[1]
                            new_line_11 = copy.deepcopy(path_11)[1:]
                            left_paths[two_left_paths[0]][0] = connected_point_11
                        else:
                            connected_point_11 = path_11[-2]
                            new_line_11 = copy.deepcopy(path_11)[0:-1]
                            left_paths[two_left_paths[0]][-1] = connected_point_11

                        if  left_paths[two_left_paths[0]][0] == key: 
                            left_paths[two_left_paths[0]][0] = connected_point_11
                        else:
                            left_paths[two_left_paths[0]][-1] = connected_point_11
                        
                        if path_12[0] == key:
                            connected_point_12 = path_12[1]
                            new_line_12 = copy.deepcopy(path_12)[1:]
                            left_paths[two_left_paths[1]][0] = connected_point_12
                        else:
                            connected_point_12 = path_12[-2]
                            new_line_12 = copy.deepcopy(path_12)[0:-1]
                            left_paths[two_left_paths[1]][-1] = connected_point_12

                        if  left_paths[two_left_paths[1]][0] == key: 
                            left_paths[two_left_paths[1]][0] = connected_point_12
                        else:
                            left_paths[two_left_paths[1]][-1] = connected_point_12
                
                        left_paths.append([connected_point_11, connected_point_12])
                        new_lines = new_lines + [new_line_11, new_line_12, [connected_point_11, connected_point_12]]
        
                        right_path_1 = right_paths[two_right_paths[0]]
                        right_path_2 = right_paths[two_right_paths[1]]
                        
                        connected_point_21, connected_point_22 = None, None
                        path_21, path_22 = None, None
                        
                        for idx, line in enumerate(connected_lines):
                            if (line[0] == right_path_1[0] and line[-1] == right_path_1[-1]) or (line[0] == right_path_1[-1] and line[-1] == right_path_1[0]):
                                list_idx.append(idx)
                                path_21 = line
                            elif (line[0] == right_path_2[0] and line[-1] == right_path_2[-1]) or (line[0] == right_path_2[-1] and line[-1] == right_path_2[0]):
                                list_idx.append(idx)
                                path_22 = line
                                
                        new_line_21 = []
                        new_line_22 = []

                        if path_21[0] == key:
                            connected_point_21 = path_21[1]
                            new_line_21 = copy.deepcopy(path_21)[1:]
                            right_paths[two_right_paths[0]][0] = connected_point_21
                        else:
                            connected_point_21 = path_21[-2]
                            new_line_21 = copy.deepcopy(path_21)[0:-1]
                            right_paths[two_right_paths[0]][-1] = connected_point_21

                        if  right_paths[two_right_paths[0]][0] == key: 
                            right_paths[two_right_paths[0]][0] = connected_point_21
                        else:
                            right_paths[two_right_paths[0]][-1] = connected_point_21

                        if path_22[0] == key:
                            connected_point_22 = path_22[1]
                            new_line_22 = copy.deepcopy(path_22)[1:]
                            right_paths[two_right_paths[1]][0] = connected_point_22
                        else:
                            connected_point_22 = path_22[-2]
                            new_line_22 = copy.deepcopy(path_22)[0:-1]
                            right_paths[two_right_paths[1]][-1] = connected_point_22
                        
                        if  right_paths[two_right_paths[1]][0] == key: 
                            right_paths[two_right_paths[1]][0] = connected_point_22
                        else:
                            right_paths[two_right_paths[1]][-1] = connected_point_22

                        right_paths.append([connected_point_21, connected_point_22])
                        new_lines = new_lines + [new_line_21, new_line_22, [connected_point_21, connected_point_22]]

                        for i in sorted(list_idx, reverse=True):
                            del connected_lines[i]

                        for line in new_lines:
                            connected_lines.append(line)
                        
                        expanded_points[key] = [connected_lines[-4], connected_lines[-1]]
                        break
    
    return common_paths, left_paths, right_paths, connected_lines, defined_paths, expanded_points
                
def find_neighbor_point(path, connected_lines, pos):
    for idx, line in enumerate(connected_lines):
        if (line[0] == path[0]) and (line[-1] == path[-1]):
            if pos == 0:
                return line[1], idx, 0
            else:
                return line[-2], idx, -1
        elif (line[-1] == path[0]) and (line[0] == path[-1]):
            if pos == 0:
                return line[-2], idx, -1
            else:
                return line[1], idx, 0
    
    return 0, None

def swap_values(x):
    return -1 if x == 0 else 0 if x == -1 else x


# def connect_undefined_paths(left_paths, right_paths, connected_lines, defined_paths, undefined_paths, common_paths, split_paths, skeleton_points):
#     remove_idx = []
    
#     for idx, common_path in enumerate(common_paths):
#         if defined_paths[idx]:
#             common_path_idx_0 = []
#             common_path_idx_1 = []
#             common_path_pos_0 = []
#             common_path_pos_1 = []
#             remove_idx = []
            
#             for path_idx, path in enumerate(undefined_paths):
#                 if (path[0] == common_path[0]):
#                     common_path_idx_0.append(path_idx)
#                     common_path_pos_0.append(0)
#                 elif (path[0] == common_path[-1]):
#                     common_path_idx_1.append(path_idx)
#                     common_path_pos_1.append(0)
#                 elif (path[-1] == common_path[0]):
#                     common_path_idx_0.append(path_idx)
#                     common_path_pos_0.append(-1)
#                 elif (path[-1] == common_path[-1]):
#                     common_path_idx_1.append(path_idx)
#                     common_path_pos_1.append(-1)
                    
#             if len(common_path_idx_0) == 2:
#                 remove_idx = common_path_idx_0
#                 path_1 = undefined_paths[common_path_idx_0[0]]
#                 path_2 = undefined_paths[common_path_idx_0[1]]
#                 touch_point_1 = common_path_pos_0[0]
#                 touch_point_2 = common_path_pos_0[1]
#                 split_point_1, con_line_idx_1, head_point_1 = find_neighbor_point(path_1, connected_lines, touch_point_1)
#                 split_point_2, con_line_idx_2, head_point_2 = find_neighbor_point(path_2, connected_lines,  touch_point_2)
#                 left_connected_line = split_paths[idx][0]
#                 right_connected_line = split_paths[idx][1]
                
#                 shortest_distance_11, point_11 = find_shortest_distance(skeleton_points[split_point_1], skeleton_points[left_connected_line])
#                 shortest_distance_12, point_12 = find_shortest_distance(skeleton_points[split_point_1], skeleton_points[right_connected_line])
#                 shortest_distance_21, point_21 = find_shortest_distance(skeleton_points[split_point_2], skeleton_points[left_connected_line])
#                 shortest_distance_22, point_22 = find_shortest_distance(skeleton_points[split_point_2], skeleton_points[right_connected_line])
#                 min_distance = min(shortest_distance_11, shortest_distance_12, shortest_distance_21, shortest_distance_22)
                
#                 if (min_distance == shortest_distance_11) or (min_distance == shortest_distance_22):
#                     path_dir_1 = defined_paths[idx][0]
#                     add_point_1 = left_connected_line[point_11]
#                     add_point_2 = right_connected_line[point_22]
#                 else:
#                     path_dir_1 = defined_paths[idx][1]
#                     add_point_1 = right_connected_line[point_12]
#                     add_point_2 = left_connected_line[point_21]
                    
#                 if head_point_1 == 0:
#                     new_path_1 = [connected_lines[con_line_idx_1][-1], add_point_1]
#                     connected_lines[con_line_idx_1][0] = add_point_1
#                 else:
#                     new_path_1 = [connected_lines[con_line_idx_1][0], add_point_1]
#                     connected_lines[con_line_idx_1][-1] = add_point_1
                
#                 if head_point_2 == 0:
#                     new_path_2 = [connected_lines[con_line_idx_2][-1], add_point_2]
#                     connected_lines[con_line_idx_2][0] = add_point_2
#                 else:
#                     new_path_2 = [connected_lines[con_line_idx_2][0], add_point_2]
#                     connected_lines[con_line_idx_2][-1] = add_point_2
                    
#                 if path_dir_1 == 0:  
#                     left_paths.append(new_path_1)
#                     right_paths.append(new_path_2)
#                 else:
#                     left_paths.append(new_path_2)
#                     right_paths.append(new_path_1)
                        
#             if len(common_path_idx_1) == 2:
#                 remove_idx = remove_idx + common_path_idx_1
#                 path_1 = undefined_paths[common_path_idx_1[0]]
#                 path_2 = undefined_paths[common_path_idx_1[1]]
#                 touch_point_1 = common_path_pos_1[0]
#                 touch_point_2 = common_path_pos_1[1]
#                 split_point_1, con_line_idx_1, head_point_1 = find_neighbor_point(path_1, connected_lines, touch_point_1)
#                 split_point_2, con_line_idx_2, head_point_2 = find_neighbor_point(path_2, connected_lines,  touch_point_2)
#                 left_connected_line = split_paths[idx][0]
#                 right_connected_line = split_paths[idx][1]
                
#                 shortest_distance_11, point_11 = find_shortest_distance(skeleton_points[split_point_1], skeleton_points[left_connected_line])
#                 shortest_distance_12, point_12 = find_shortest_distance(skeleton_points[split_point_1], skeleton_points[right_connected_line])
#                 shortest_distance_21, point_21 = find_shortest_distance(skeleton_points[split_point_2], skeleton_points[left_connected_line])
#                 shortest_distance_22, point_22 = find_shortest_distance(skeleton_points[split_point_2], skeleton_points[right_connected_line])
#                 min_distance = min(shortest_distance_11, shortest_distance_12, shortest_distance_21, shortest_distance_22)
                
#                 if (min_distance == shortest_distance_11) or (min_distance == shortest_distance_22):
#                     path_dir_1 = defined_paths[idx][0]
#                     add_point_1 = left_connected_line[point_11]
#                     add_point_2 = right_connected_line[point_22]
#                 else:
#                     path_dir_1 = defined_paths[idx][1]
#                     add_point_1 = right_connected_line[point_12]
#                     add_point_2 = left_connected_line[point_21]
                
#                 if head_point_1 == 0:
#                     new_path_1 = [connected_lines[con_line_idx_1][-1], add_point_1]
#                     connected_lines[con_line_idx_1][0] = add_point_1
#                 else:
#                     new_path_1 = [connected_lines[con_line_idx_1][0], add_point_1]
#                     connected_lines[con_line_idx_1][-1] = add_point_1
                
#                 if head_point_2 == 0:
#                     new_path_2 = [connected_lines[con_line_idx_2][-1], add_point_2]
#                     connected_lines[con_line_idx_2][0] = add_point_2
#                 else:
#                     new_path_2 = [connected_lines[con_line_idx_2][0], add_point_2]
#                     connected_lines[con_line_idx_2][-1] = add_point_2
                
#                 if path_dir_1 == 0:  
#                     left_paths.append(new_path_1)
#                     right_paths.append(new_path_2)
#                 else:
#                     left_paths.append(new_path_2)
#                     right_paths.append(new_path_1)
        
#             for i in sorted(remove_idx, reverse=True):
#                 del undefined_paths[i]
    
#     remove_idx = []   
#     for path_idx, path in enumerate(undefined_paths):
#         for idx, common_path in enumerate(common_paths):
#             split_path = split_paths[idx]
#             path_1 = split_path[0]
#             path_2 = split_path[1]
            
#             distance_1, distance_2 = None, None
#             path_dir = None
#             touch_point = None
#             new_connect_point = None
            
#             if path[0] == common_path[0]:
#                 touch_point = 0
#                 new_connect_point = 0
#                 distance_1 = euclidean_distance(skeleton_points[path[0]], skeleton_points[path_1[0]])
#                 distance_2 = euclidean_distance(skeleton_points[path[0]], skeleton_points[path_2[0]])
#             elif path[0] == common_path[-1]:
#                 touch_point = 0
#                 new_connect_point = -1
#                 distance_1 = euclidean_distance(skeleton_points[path[0]], skeleton_points[path_1[-1]])
#                 distance_2 = euclidean_distance(skeleton_points[path[0]], skeleton_points[path_2[-1]])
#             elif path[-1] == common_path[0]:
#                 touch_point = -1
#                 new_connect_point = 0
#                 distance_1 = euclidean_distance(skeleton_points[path[-1]], skeleton_points[path_1[0]])
#                 distance_2 = euclidean_distance(skeleton_points[path[-1]], skeleton_points[path_2[0]])
#             elif path[-1] == common_path[-1]:
#                 touch_point = -1
#                 new_connect_point = -1
#                 distance_1 = euclidean_distance(skeleton_points[path[-1]], skeleton_points[path_1[-1]])
#                 distance_2 = euclidean_distance(skeleton_points[path[-1]], skeleton_points[path_2[-1]])
            
#             if distance_1 is not None and distance_2 is not None:
#                 _, con_line_idx, head_point = find_neighbor_point(path, connected_lines, touch_point)
                
#                 if distance_1 < distance_2: 
#                     new_path = [path[swap_values(touch_point)], path_1[new_connect_point]]
#                     path_dir = defined_paths[idx][0]
#                     connected_lines[con_line_idx][swap_values(head_point)] = path_1[new_connect_point]
#                 else:
#                     new_path = [path[swap_values(touch_point)], path_2[new_connect_point]]
#                     path_dir = defined_paths[idx][1]
#                     connected_lines[con_line_idx][swap_values(head_point)] = path_2[new_connect_point]
                
#                 if path_dir == 0:
#                     left_paths.append(new_path)
#                 else:
#                     right_paths.append(new_path)
                    
#                 remove_idx.append(path_idx)
#                 break
    
#     for i in sorted(remove_idx, reverse=True):
#         del undefined_paths[i]
        
#     is_found = True
    
#     while is_found:
#         is_found = False
        
#         remove_idx = []
#         for idx, path in enumerate(undefined_paths):
#             for left_path in left_paths:
#                 if path[0] == left_path[0] or path[0] == left_path[-1] or path[-1] == left_path[-1] or path[-1] == left_path[0]:
#                     left_paths.append(path)
#                     remove_idx.append(idx)
#                     is_found = True
#                     break
                
#             if is_found == False:
#                 for right_path in right_paths:
#                     if path[0] == right_path[0] or path[0] == right_path[-1] or path[-1] == right_path[-1] or path[-1] == right_path[0]:
#                         right_paths.append(path)
#                         remove_idx.append(idx)
#                         is_found = True
#                         break
                
#         for i in sorted(remove_idx, reverse=True):
#             del undefined_paths[i]
    
        
#     return left_paths, right_paths, undefined_paths, connected_lines

def distance_point_to_line(p1, idx_2, idx_3, skeleton_points):
    # p1 = skeleton_points[idx_1]
    p2 = skeleton_points[idx_2]
    p3 = skeleton_points[idx_3]
    
    # Vector representing the line segment (p2, p3)
    line_vector = p3 - p2

    # Vector from p2 to p1
    point_vector = p1 - p2

    # Calculate the cross product
    cross_product = np.cross(line_vector, point_vector)

    # Calculate the distance
    distance = np.linalg.norm(cross_product) / np.linalg.norm(line_vector)
    if np.linalg.norm(p1 - p2) < np.linalg.norm(p1 - p3):
        nearest_point = idx_2
    else:
        nearest_point = idx_3

    return distance, nearest_point

def connect_undefined_paths(left_paths, right_paths, connected_lines, defined_paths, undefined_paths, common_paths, split_paths, skeleton_points, expanded_points, reserved_points):
    remove_idx = []
    
    for idx, path in enumerate(undefined_paths):
        for adjacent_point in expanded_points:
            if adjacent_point == path[0] or adjacent_point == path[1]:
                remove_idx.append(idx)
                pos = None
                select_line = []
                
                for line in connected_lines:
                    if line[0] == path[0] and line[-1] == path[1]:
                        select_line = line
                        break
                    elif line[-1] == path[0] and line[0] == path[1]:
                        select_line = line[::-1]
                        break
                
                if adjacent_point == path[0]:
                    pos = 0
                else:
                    pos = 1
                
                if path[pos] in reserved_points and path[pos-1] in reserved_points[path[pos]]:
                    if pos == 0:
                        original_point = (skeleton_points[reserved_points[path[pos]][path[pos-1]]] + skeleton_points[select_line[1]])/2
                    else:
                        original_point = (skeleton_points[reserved_points[path[pos]][path[pos-1]]] + skeleton_points[select_line[-1]])/2
                else:
                    if pos == 0:
                        original_point = (skeleton_points[select_line[0]] + skeleton_points[select_line[1]])/2
                    else:
                        original_point = (skeleton_points[select_line[-2]] + skeleton_points[select_line[-1]])/2
                    
                path_1 = expanded_points[adjacent_point][0]
                path_2 = expanded_points[adjacent_point][1]
                distance_1, nearest_point_1 = distance_point_to_line(original_point, path_1[0], path_1[-1], skeleton_points)
                distance_2, nearest_point_2 = distance_point_to_line(original_point, path_2[0], path_2[-1], skeleton_points)
                
                print(original_point, distance_1, distance_2, skeleton_points[nearest_point_1], skeleton_points[nearest_point_2])
                
                if distance_1 < distance_2:
                    if pos == 0:
                        new_line = [nearest_point_1] + select_line
                    else:
                        new_line = select_line + [nearest_point_1]
                    left_paths.append([new_line[0], new_line[-1]])
                    
                else:
                    if pos == 0:
                        new_line = [nearest_point_2] + select_line
                    else:
                        new_line = select_line + [nearest_point_2]
                    right_paths.append([new_line[0], new_line[-1]])
                
                connected_lines.append(new_line)
                
                break
            
    for i in sorted(remove_idx, reverse=True):
        del undefined_paths[i]

    
    is_found = True
    while is_found:
        is_found = False
        remove_idx = []
        new_left_paths = []
        new_right_paths = []
    
        for idx, path in enumerate(undefined_paths):
            info = [{
                    'left': [],
                    'right': []
                },{
                    'left': [],
                    'right': []
                }
            ]
            
            for left_path in left_paths:
                if path[0] == left_path[0] or path[0] == left_path[1]:
                    info[0]['left'].append(left_path)
                if path[1] == left_path[0] or path[1] == left_path[1]:
                    info[1]['left'].append(left_path)
                    
            for right_path in right_paths:
                if path[0] == right_path[0] or path[0] == right_path[1]:
                    info[0]['right'].append(right_path)
                if path[1] == right_path[0] or path[1] == right_path[1]:
                    info[1]['right'].append(right_path)
                
            if (len(info[0]['left']) and len(info[1]['right'])) or (len(info[1]['left']) and len(info[0]['right'])):
                continue
            elif not len(info[0]['left']) and not len(info[0]['right']) and not len(info[1]['left']) and not len(info[1]['right']):
                continue
            elif not len(info[0]['left']) and not len(info[0]['right']):
                is_found = True
                remove_idx.append(idx)
                
                if len(info[1]['left']) and not len(info[1]['right']):
                    new_left_paths.append(path)
                elif not len(info[1]['left']) and len(info[1]['right']):
                    new_right_paths.append(path)
                else:
                    min_left_distance = 10000
                    min_right_distance = 10000
                    
                    for line in connected_lines:
                        if line[0] == path[0] and line[-1] == path[-1]:
                            next_point = line[-2]
                            break
                        elif line[-1] == path[0] and line[0] == path[-1]:
                            next_point = line[1]
                            break
                                        
                    for left_path in info[1]['left']:
                        select_line = []
                        for line in connected_lines:
                            if line[0] == left_path[0] and line[-1] == left_path[-1]:
                                if line[0] == path[1]:
                                    select_line = line[2:]
                                else:
                                    select_line = line[:-2]
                                break
                            elif line[-1] == left_path[0] and line[0] == left_path[-1]:
                                if line[0] == path[1]:
                                    select_line = line[2:]
                                else:
                                    select_line = line[:-2]
                                break
                            
                        distance, _ = find_shortest_distance(skeleton_points[next_point], skeleton_points[select_line])
                        if distance < min_left_distance:
                            min_left_distance = distance
                    
                    for right_path in info[1]['right']:
                        select_line = []
                        for line in connected_lines:
                            if line[0] == right_path[0] and line[-1] == right_path[-1]:
                                if line[0] == path[1]:
                                    select_line = line[2:]
                                else:
                                    select_line = line[:-2]
                                break
                            elif line[-1] == right_path[0] and line[0] == right_path[-1]:
                                if line[0] == path[1]:
                                    select_line = line[2:]
                                else:
                                    select_line = line[:-2]
                                break
                            
                        distance, _ = find_shortest_distance(skeleton_points[next_point], skeleton_points[select_line])
                        
                        if distance < min_right_distance:
                            min_right_distance = distance
                    
                    if min_left_distance < min_right_distance:
                        new_left_paths.append(path)
                    else:
                        new_right_paths.append(path)
            elif not len(info[1]['left']) and not len(info[1]['right']):
                is_found = True
                remove_idx.append(idx)
                
                if len(info[0]['left']) and not len(info[0]['right']):
                    new_left_paths.append(path)
                elif not len(info[0]['left']) and len(info[0]['right']):
                    new_right_paths.append(path)
                else:
                    for line in connected_lines:
                        if line[0] == path[0] and line[-1] == path[-1]:
                            next_point = line[1]
                            break
                        elif line[-1] == path[0] and line[0] == path[-1]:
                            next_point = line[-2]
                            break
                    
                    min_left_distance = 10000
                    min_right_distance = 10000
                    
                    for left_path in info[0]['left']:
                        select_line = []
                        for line in connected_lines:
                            if line[0] == left_path[0] and line[-1] == left_path[-1]:
                                if line[0] == path[0]:
                                    select_line = line[2:]
                                else:
                                    select_line = line[:-2]
                                break
                            elif line[-1] == left_path[0] and line[0] == left_path[-1]:
                                if line[0] == path[0]:
                                    select_line = line[2:]
                                else:
                                    select_line = line[:-2]
                                break
                            
                        distance, _ = find_shortest_distance(skeleton_points[next_point], skeleton_points[select_line])
                        if distance < min_left_distance:
                            min_left_distance = distance
                    
                    for right_path in info[0]['right']:
                        select_line = []
                        for line in connected_lines:
                            if line[0] == right_path[0] and line[-1] == right_path[-1]:
                                if line[0] == path[0]:
                                    select_line = line[2:]
                                else:
                                    select_line = line[:-2]
                                break
                            elif line[-1] == right_path[0] and line[0] == right_path[-1]:
                                if line[0] == path[0]:
                                    select_line = line[2:]
                                else:
                                    select_line = line[:-2]
                                break
                            
                        distance, _ = find_shortest_distance(skeleton_points[next_point], skeleton_points[select_line])
                        if distance < min_right_distance:
                            min_right_distance = distance
                    
                    if min_left_distance < min_right_distance:
                        new_left_paths.append(path)
                    else:
                        new_right_paths.append(path)   
        
        left_paths += new_left_paths
        right_paths += new_right_paths
        for i in sorted(remove_idx, reverse=True):
            del undefined_paths[i]
        
    return left_paths, right_paths, undefined_paths, connected_lines
                
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

def find_touchpoints(mask_data, left_points, right_points, distance_threshold=1):
    new_points = 1000000
    loop = 0
    zero_positions = np.argwhere(mask_data != 0)

    artery_data = np.zeros_like(mask_data)
    for pos in left_points:
        artery_data[pos[0]][pos[1]][pos[2]] = 1
    
    for pos in right_points:
        artery_data[pos[0]][pos[1]][pos[2]] = 2

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
    
    zero_positions = np.argwhere(mask_data != 0)
    coordinates_to_update = []
    kernel = np.ones((3, 3, 3), dtype=np.int64)
    kernel[1, 1, 1] = 0  # Exclude the center voxel

    for idx, pos in enumerate(zero_positions):
        x, y, z = pos[0], pos[1], pos[2]
        # Extract the neighborhood around the current voxel
        neighborhood = artery_data[max(0, x-1):min(artery_data.shape[0], x+2),
                                max(0, y-1):min(artery_data.shape[1], y+2),
                                max(0, z-1):min(artery_data.shape[2], z+2)]

        # Perform convolution to count neighbor occurrences of each value
        count_1 = np.argwhere(neighborhood == 1).shape[0]
        count_2 = np.argwhere(neighborhood == 2).shape[0]
        current_value = artery_data[x][y][z]
        
        if count_1 > count_2:
            if count_1 != current_value and count_1 > 0.5*(count_1 + count_2):
                coordinates_to_update.append((x, y, z, 1))
        
        elif count_2 > count_1:
            if count_2 != current_value and count_2 > 0.5*(count_1 + count_2):
                coordinates_to_update.append((x, y, z, 2))
        
    for i, j, k, updated_data in coordinates_to_update:
        artery_data[i][j][k] = updated_data
        
    # touch_points = []
    # suspected_positions = np.argwhere(artery_data > 0)

    # for pos in suspected_positions:
    #         i, j, k = pos[0], pos[1], pos[2]
    #         is_touch = find_distant_neighbors(artery_data, i, j, k, distance_threshold)
    #         if is_touch:
    #             touch_points.append([i, j, k])
    
    # for pos in touch_points:
    #     artery_data[pos[0]][pos[1]][pos[2]] = -1 

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

    return touch_points, artery_data

def extract_point_position(skeleton_points, connected_lines, left_paths, right_paths):
    left_voxels = []
    right_voxels = []
    
    for path in left_paths:
        for line in connected_lines:
            if (path[0] == line[0] and path[1] == line[-1]) or (path[0] == line[-1] and path[1] == line[0]):
                left_voxels = left_voxels + line
                break
            
    for path in right_paths:
        for line in connected_lines:
            if (path[0] == line[0] and path[1] == line[-1]) or (path[0] == line[-1] and path[1] == line[0]):
                right_voxels = right_voxels + line
                break
            
    left_voxels = list(set(left_voxels))
    right_voxels = list(set(right_voxels))
    
    return skeleton_points[left_voxels], skeleton_points[right_voxels]

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

def reinterpolate_connected_lines(skeleton_points, connected_lines):
    for i, line in enumerate(connected_lines):
        inter_points = interpolate_path(skeleton_points[line])
        
        if len(inter_points) > 2:
            new_line = [line[0]]
            
            for new_point in inter_points[1:-1]:
                new_index = skeleton_points.shape[0]
                skeleton_points = np.vstack([skeleton_points, new_point])
                new_line.append(new_index)
            
            new_line.append(line[-1])
            connected_lines[i] = new_line
        
    return skeleton_points, connected_lines

def correct_undefined_paths(common_paths, undefined_paths, split_groups):
    remove_idx = []
    
    for idx, path in enumerate(common_paths):
        if len(split_groups[idx]) == 0:
            remove_idx.append(idx)
            undefined_paths.append(path)

    for i in sorted(remove_idx, reverse=True):
        del common_paths[i]
        del split_groups[i]
        
    return common_paths, undefined_paths, split_groups
    