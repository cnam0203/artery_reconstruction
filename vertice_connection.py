from scipy.spatial import KDTree, distance_matrix
from sklearn.cluster import DBSCAN, KMeans
from process_graph import *
import numpy as np
import matplotlib.pyplot as plt

points = np.array([
    [5, 2], [5, 1], [5, 0], [4, 0], [3, 0], [2, 0],
    [1.5, 1.5], [1, 2], [0, 3], [0, 4], [0, 5], [1, 5],
    [2, 5], [3, 5], [3, 4], [3, 3], [3, 6], [3, 7],
    [4, 5], [5, 5], [6, 5], [7, 5], [7, 6], [7, 7], [6, 8], [5, 9],
    [4, 9], [3, 9], [2, 9], [1, 8]
])

skeleton_points = np.array([
    [5.5, 2],   #0
    [5.5, 0],   #1
    [3.5, 0],   #2
    [2.1, 0],   #3
    [1.4, 1.2], #4
    [0.9, 2],   #5
    [0, 3],   #6
    [0, 5],   #7
    [1.5, 5],   #8
    [3, 5],   #9
    [3, 3],   #10
    [3, 7],   #11
    [4.5, 5],   #12
    [7, 5],   #13
    [7, 7],   #14
    [5, 9],   #15
    [3.5, 9],   #16
    [2, 9],   #17
    [1, 8],   #18
])

junction_points = [9]
connected_lines = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [9, 10],
    [9, 11],
    [9, 12, 13, 14, 15, 16, 17, 18]
]

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
            if len(smooth_line) == 0:
                smooth_line += arrange_lines[point]
            else:
                start_point = points[smooth_line[0]]
                end_point = points[smooth_line[-1]]
                tree = KDTree(points[[arrange_lines[point][-1], arrange_lines[point][0]]])

                distances_1, indices_1 = tree.query(start_point)
                distances_2, indices_2 = tree.query(end_point)

                if distances_1 < distances_2:
                    if indices_1 == 0:
                        smooth_line = arrange_lines[point][::-1] + smooth_line
                    else:
                        smooth_line = arrange_lines[point] + smooth_line
                else:
                    if indices_2 == 0:
                        smooth_line = smooth_line + arrange_lines[point][::-1]
                    else:
                        smooth_line = smooth_line + arrange_lines[point]

    smooth_connected_lines.append(smooth_line)

for point in junction_points:
    center_points = []
    pos = []
    for index, line in enumerate(connected_lines):
        if line[0] == point or line[-1] == point:
            distance_1 = euclidean_distance(skeleton_points[point], points[smooth_connected_lines[index][0]])
            distance_2 = euclidean_distance(skeleton_points[point], points[smooth_connected_lines[index][-1]])
            if distance_1 < distance_2:
                center_points.append(smooth_connected_lines[index][0])
                pos.append(0)
            else:
                center_points.append(smooth_connected_lines[index][-1])
                pos.append(-1)

    avg_point = np.mean(points[center_points], axis=0)
    new_index = points.shape[0]
    points = np.vstack([points, avg_point])

    for index, line in enumerate(connected_lines):
        if line[0] == point or line[-1] == point:
            if pos[index] == 0:
                smooth_connected_lines[index].insert(0, new_index)
            else:
                smooth_connected_lines[index].append(new_index)


# Plot points
plt.scatter(points[:, 0], points[:, 1], color='blue', label='Points')
plt.scatter(skeleton_points[:, 0], skeleton_points[:, 1], color='black', label='Points')

# Plot lines
for line in smooth_connected_lines:
    line_points = [points[i] for i in line]
    line_xs, line_ys = zip(*line_points)
    plt.plot(line_xs, line_ys, color='red')

plt.gca().invert_yaxis() 
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Visualization of Lines')
plt.legend()
plt.grid(True)
plt.show()

# print(smooth_connected_lines)


