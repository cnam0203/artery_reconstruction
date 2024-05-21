import nibabel as nib
import numpy as np
from scipy.spatial import KDTree, distance_matrix
from sklearn.cluster import DBSCAN, KMeans
from scipy.signal import argrelextrema

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
        filter_vertices = []
        

        for idx, vertex_index in enumerate(ring_vertices):
            vertex = vmtk_boundary_vertices[vertex_index]
            intersection_point = intsecpoints[idx]
            distance = distances[idx]
            exist = False

            for surface in surfaces:
                is_intersect, triangular_point = ray_intersects_triangle(vertex, intersection_point, surface)
                
                if is_intersect:
                    removed_vertices.append([vertex_index, intersection_point, triangular_point])
                    exist = True
                    break
                           
            if not exist:
                if distance > 1.8*min_distance:
                    exist = True

            if not exist:                                            
                filter_vertices.append([vertex_index, intersection_point])
        
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


dataset_dir = '/Users/apple/Desktop/neuroscience/artery_separate/dataset/'
segment_file_path = dataset_dir + 'BCW-1205-RES.nii.gz'
original_file_path = dataset_dir + 'BCW-1205-RES_0000.nii.gz'

segment_image = nib.load(segment_file_path)
original_image = nib.load(original_file_path)

original_data = original_image.get_fdata()
segment_data = segment_image.get_fdata()
voxel_sizes = segment_image.header.get_zooms()
info = {}
sub_num = 'BCW-1222-RES'
info_dir = '/Users/apple/Desktop/neuroscience/artery_separate/info_files/' + sub_num + '/'

for artery_index in [1, 2, 3, 5, 6, 7, 8]:
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
    line_traces = []

    for idx, ring in enumerate(ring_vertices):
        interval_radius = radiuses[idx]

        if splitted_branches[idx] != cur_branch:
            cur_pos = 0
            cur_branch = splitted_branches[idx]
            branches.append([])

        # if cur_pos == 0:
        #     print("Branch ", cur_branch)
        
        if len(interval_radius):
            # print(f"""Length = {round(cur_pos, 2)} mm, at the position""", interval_radius[1], ", min radius =""", round(euclidean_distance(interval_radius[0], interval_radius[1]), 2), '(mm)')
            branches[-1].append(interval_radius)

        cur_pos += distance_threshold

    longest_branch = None
    longest_length = 0
    stenose_points = []
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

    longest_branch = branches[longest_branch]
    radius = [round(euclidean_distance(item[0], item[1]), 2) for item in longest_branch]
    for i in range(len(longest_branch)):
        info[artery_key].append({
            'Length (mm)': i*distance_threshold,
            'Position at': longest_branch[i][1].tolist(),
            'Min radius (mm)': radius[i]
        })
        
        print(f"""Length = {i*distance_threshold} mm, at the position""", longest_branch[i][1], ", min radius =", radius[i], '(mm)')
    
    local_min = argrelextrema(np.array(radius), np.less)[0]
    stenose_points += [item[1] for index, item in enumerate(longest_branch) if index in local_min.tolist()]
        
    # print("Min radius at half the length: ", radius[int(len(longest_branch)/2)])
    
    
    # max_branch_pos = [idx for idx, value in enumerate(splitted_branches) if value == longest_branch]
    # ring_vertices = [ring for idx, ring in enumerate(ring_vertices) if idx in max_branch_pos]
    # middle_ring = ring_vertices[int(len(ring_vertices)/2)]

    # middle_surface_points = vmtk_boundary_vertices[[sublist[0] for sublist in middle_ring]]
    # middle_intersection_points = intersection_points[int(len(intersection_points)/2)]

    # sum_distance = []
    # for idx, point in enumerate(middle_surface_points):
    #     sum_distance.append(euclidean_distance(point, middle_intersection_points[idx]))

    # mean_radius = sum(sum_distance)/len(sum_distance)
    # print("Average radius at half the length: ", round(mean_radius, 2))
    
    # visualized_boundary_points = generate_points(vmtk_boundary_vertices, 1, 'blue')
    visualized_smooth_points = generate_points(smooth_points, 1, 'green')
    visualized_stenose_points = generate_points(np.array(stenose_points), 3, 'red')
    
    mesh = go.Mesh3d(
        x=vmtk_boundary_vertices[:, 0],
        y=vmtk_boundary_vertices[:, 1],
        z=vmtk_boundary_vertices[:, 2],
        i=vmtk_boundary_faces[:, 0],
        j=vmtk_boundary_faces[:, 1],
        k=vmtk_boundary_faces[:, 2],
        opacity=0.7,
        color="#000000"
    )

    # print(mesh)
    
    show_figure([mesh])

with open(info_dir + f'min_radius_BCW-1205-RES.json', 'w') as file:
    json.dump(info, file)  # indent=4 makes the file more readable
