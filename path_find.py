import nibabel as nib
import numpy as np

from skimage.morphology import skeletonize, thin
from skimage import measure

from scipy.spatial import KDTree, distance_matrix
from sklearn.cluster import DBSCAN, KMeans

import matplotlib.pyplot as plt

import math
import time
import os
import json
import heapq
import copy

from preprocess_data import *
from process_graph import *
from visualize_graph import *
from slice_selection import *
from visualize_mesh import *


def reconstruct_surface(segment_image,
                  original_image=None, 
                  index=[], 
                  intensity_threshold_1=0.65, 
                  intensity_threshold_2=0.65, 
                  gaussian_sigma=0, 
                  distance_threshold=20,
                  laplacian_iter=1,
                  folder_path='',
                  neighbor_threshold_1=8,
                  neighbor_threshold_2=15):
    
    # Load original image (TOF-MRA) into numpy array
    original_data = original_image.get_fdata()
    segment_data = segment_image.get_fdata()
    voxel_sizes = segment_image.header.get_zooms()
    mask_data, cex_data, surf_data = preprocess_data(original_data, segment_data, index, intensity_threshold_1, intensity_threshold_2, gaussian_sigma, neighbor_threshold_1, neighbor_threshold_2 )
    
    # Find the voxel-based skeleton
    artery_points = np.argwhere(cex_data != 0)
    skeleton = skeletonize(cex_data)
    skeleton_points, end_points, junction_points, connected_lines = find_graphs(skeleton)
    
    max_length = 2
    reserved_points = {}
    
    while max_length <= 5:
        short_idx = []
        reduced_idx = []
        new_lines = []
        
        for idx, line in enumerate(connected_lines):
            if len(line) == max_length:
                if line[0] in end_points or line[-1] in end_points:
                    continue
                short_idx.append(idx)
        
        
        for idx in short_idx:
            reduced_idx.append(idx)
            cur_line = connected_lines[idx]
            connected_points = []
            
            for sub_idx, line in enumerate(connected_lines):
                if (sub_idx != idx) and sub_idx not in reduced_idx:
                    if line[0] == cur_line[0] or line[0] == cur_line[-1]:
                        connected_points.append(line[1])
                    elif line[-1] == cur_line[0] or line[-1] == cur_line[-1]:
                        connected_points.append(line[-1])
            
            mid_point = np.mean(skeleton_points[connected_points], axis=0, dtype=np.int64)
            new_index = skeleton_points.shape[0]
            skeleton_points = np.vstack([skeleton_points, mid_point])
            
            for sub_idx, line in enumerate(connected_lines):
                if (sub_idx != idx) and sub_idx not in reduced_idx:
                    point_1 = line[0]
                    point_2 = line[-1]
                    
                    if line[0] == cur_line[0] or line[0] == cur_line[-1]:
                        connected_lines[sub_idx][0] = new_index
                        
                        if new_index not in reserved_points:
                            reserved_points[new_index] = {}
                        
                        if point_1 in reserved_points and point_2 in reserved_points[point_1]:
                            reserved_points[new_index][point_2] = reserved_points[point_1][point_2]
                        else:
                            reserved_points[new_index][point_2] = point_1
                            
                    elif line[-1] == cur_line[0] or line[-1] == cur_line[-1]:
                        connected_lines[sub_idx][-1] = new_index
                        
                        if new_index not in reserved_points:
                            reserved_points[new_index] = {}
                        
                        if point_2 in reserved_points and point_1 in reserved_points[point_2]:
                            reserved_points[new_index][point_1] = reserved_points[point_2][point_1]
                        else:
                            reserved_points[new_index][point_1] = point_2
            
        for i in sorted(short_idx, reverse=True):
            del connected_lines[i]
            
        max_length += 1
    
    num_points = skeleton_points.shape[0]
    neighbor_distances = np.zeros((num_points, num_points), dtype=int)
    reduced_distances = np.zeros((num_points, num_points), dtype=int)
            
    for line in connected_lines:
        head_point_1, head_point_2 = line[0], line[-1]
        n = len(line)
        sum_distance = 0
        
        for i in range(n-1):
            point_1 = line[i]
            point_2 = line[i+1]
            
            distance = euclidean_distance(skeleton_points[point_1], skeleton_points[point_2])
            neighbor_distances[point_1][point_2] = distance
            neighbor_distances[point_2][point_1] = distance
    
    junction_points = refine_junction_points(skeleton_points, neighbor_distances) 
    
    remove_idx = []
    for idx, line in enumerate(connected_lines):
        true_1 = False
        true_2 = False
        
        if (line[0] in end_points or line[0] in junction_points):
            true_1 = True
        if (line[-1] in end_points or line[-1] in junction_points):
            true_2 = True
            
        if (true_1 and true_2) == False:
            remove_idx.append(idx)
            
    for i in sorted(remove_idx, reverse=True):
        del connected_lines[i]
        
    distinct_edges = {}
    for idx, line in enumerate(connected_lines):
        point_1 = line[0]
        point_2 = line[-1]
        
        if point_1 > point_2:
            tmp = point_1
            point_1 = point_2
            point_2 = tmp
        
        if point_1 not in distinct_edges:
            distinct_edges[point_1] = {}
        if point_2 not in distinct_edges[point_1]:
            distinct_edges[point_1][point_2] = []
            
        distinct_edges[point_1][point_2].append(idx)
    
    
    remove_idx = []
    for point_1 in distinct_edges:
        for point_2 in distinct_edges[point_1]:
            list_idx = distinct_edges[point_1][point_2]
            
            if len(list_idx) > 1:
                for idx in list_idx:
                    line = connected_lines[idx]
                    length_of_line = len(line)
                    
                    if length_of_line > 2:
                        remove_idx.append(idx)
                        middle_positions = ((length_of_line - 1) // 2, (length_of_line // 2) if length_of_line % 2 == 0 else None)
                    
                        if middle_positions[1] is not None:  # If the length of the list is even
                            first_half = line[:middle_positions[0] + 1]
                            second_half = line[middle_positions[0]:]
                        else:  # If the length of the list is odd
                            first_half = line[:middle_positions[0] + 1]
                            second_half = line[middle_positions[0]:]

                        connected_lines.append(first_half)
                        connected_lines.append(second_half)
    
    for i in sorted(remove_idx, reverse=True):
        del connected_lines[i]
        
    edges = []
    line_traces = []
    left_paths, right_paths = [], []
    
    connected_points = {}
    for idx, line in enumerate(connected_lines):
        point_1 = line[0]
        point_2 = line[-1]
        
        if point_1 not in connected_points:
            connected_points[point_1] = []
        if point_2 not in connected_points:
            connected_points[point_2] = []
        
        connected_points[point_1].append(idx)
        connected_points[point_2].append(idx)
        
        if point_1 > point_2:
            point_2 = point_1
            point_1 = line[-1]
            
        edges.append([point_1, point_2])
    
    left_paths = []
    right_paths = []  
    print(reserved_points)
    
    if 5 in index or 6 in index:
        endpoint_pos = skeleton_points[end_points]
        sorted_indices = np.argsort(endpoint_pos[:, 2])
        smallest_indices = sorted_indices[:3]
        
        x_values = endpoint_pos[smallest_indices, 0]
        sorted_indices = np.argsort(x_values)
        filtered_indices = [smallest_indices[sorted_indices[0]], smallest_indices[sorted_indices[-1]]]
        
        aca_endpoints = [end_points[filtered_indices[0]], end_points[filtered_indices[1]]]
        aca_endpoints.sort()
        directions = ['left', 'right']
        endpoint_vector = skeleton_points[aca_endpoints[0]] - skeleton_points[aca_endpoints[1]]
        common_paths, left_paths, right_paths, undefined_paths = find_branches(aca_endpoints, end_points, directions, skeleton_points, junction_points, edges, connected_lines, connected_points)
        skeleton_points, split_groups = find_split_points(common_paths, original_data, mask_data, cex_data, skeleton_points, connected_lines)
        common_paths, undefined_paths, split_groups = correct_undefined_paths(common_paths, undefined_paths, split_groups)
        split_paths, undefined_paths, connected_lines = connect_split_points(split_groups, skeleton_points, undefined_paths, connected_lines)
        common_paths, left_paths, right_paths, connected_lines, defined_paths, expanded_points = connect_common_paths(common_paths, left_paths, right_paths, connected_lines, split_paths, skeleton_points, undefined_paths) 
        left_paths, right_paths, undefined_paths, connected_lines = connect_undefined_paths(left_paths, right_paths, connected_lines, defined_paths, undefined_paths, common_paths, split_paths, skeleton_points, expanded_points, reserved_points)
        common_paths, left_paths, right_paths, connected_lines, defined_paths, expanded_points = connect_common_paths(common_paths, left_paths, right_paths, connected_lines, split_paths, skeleton_points, undefined_paths) 
        skeleton_points, connected_lines = reinterpolate_connected_lines(skeleton_points, connected_lines)
        left_points, right_points = extract_point_position(skeleton_points, connected_lines, left_paths, right_paths)
        touch_points, artery_data = find_touchpoints(mask_data, left_points, right_points, 1)
        visualize_artery_mesh(artery_data, voxel_sizes, [1, 2], folder_path + '_split')
        visualize_artery_mesh(mask_data, voxel_sizes, [1], folder_path + '_orginal')
        
    line_groups = [common_paths, left_paths, right_paths, undefined_paths]
    line_colors = ['red', 'blue', 'green', 'orange', ]
    for i, line_group in enumerate(line_groups):
        for line in line_group:
            for connected_line in connected_lines:
                if (line[0] in connected_line and line[-1] in connected_line):
                    color = line_colors[i]
                    line_traces.append(generate_lines(skeleton_points[connected_line], 4, color))
    
    visualized_skeleton_points = generate_points(skeleton_points, 3)
    visualized_end_points = generate_points(skeleton_points[end_points], 5, 'red')
    visualized_junction_points = generate_points(skeleton_points[junction_points], 5, 'green')
    visualized_artery_points = generate_points(artery_points, 1, 'blue')
        
    show_figure([
                visualized_skeleton_points, 
                visualized_end_points, 
                visualized_junction_points,
                # visualized_artery_points,
            ] 
                + line_traces
    )
    
    return

if __name__ == "__main__":

    # Calculate runtime - record start time
    start_time = time.time()
    dataset_dir = '/Users/apple/Desktop/neuroscience/artery_separate/dataset/'
    
    # Specify the path to your NIfTI file
    
    # 5-15
    # segment_file_path = dataset_dir + 'TOF_multiclass_segmentation.nii.gz'
    # original_file_path = dataset_dir + 'sub-1_run-1_mra_TOF.nii.gz'
    
    segment_file_path = dataset_dir + 'sub-4947_TOF_multiclass_segmentation.nii.gz'
    original_file_path = dataset_dir + 'sub-4947_run-1_mra_TOF.nii.gz'
    
    # segment_file_path = dataset_dir + 'sub-2983_TOF_multiclass_segmentation.nii.gz'
    # original_file_path = dataset_dir + 'sub-2983_run-1_mra_TOF.nii.gz'
    
    # segment_file_path = dataset_dir + 'sub-11_TOF_multiclass_segmentation.nii.gz'
    # original_file_path = dataset_dir + 'sub-11_run-1_mra_TOF.nii.gz'
    
    # segment_file_path = dataset_dir + 'sub-1057_TOF_multiclass_segmentation.nii.gz'
    # original_file_path = dataset_dir + 'sub-1057_run-1_mra_TOF.nii.gz'
    
    # segment_file_path = dataset_dir + 'sub-2849_TOF_multiclass_segmentation.nii.gz'
    # original_file_path = dataset_dir + 'sub-2849_run-1_mra_TOF.nii.gz'
    
    # 10-20
    # segment_file_path = dataset_dir + 'sub-2049_TOF_multiclass_segmentation.nii.gz'
    # original_file_path = dataset_dir + 'sub-2049_run-1_mra_TOF.nii.gz'
    
    # segment_file_path = dataset_dir + 'sub-1425_TOF_multiclass_segmentation.nii.gz'
    # original_file_path = dataset_dir + 'sub-1425_run-1_mra_TOF.nii.gz'
    
    
    folder_path = '/Users/apple/Desktop/neuroscience/artery_separate/mesh/' + segment_file_path.split('/')[-1].split('.')[-3]
    # Load the NIfTI image
    segment_image = nib.load(segment_file_path)
    original_image = nib.load(original_file_path)
    intensity_threshold_1 = 0.1
    intensity_threshold_2 = 0.1
    gaussian_sigma=2
    distance_threshold=20
    laplacian_iter = 5
    neighbor_threshold_1 = 5
    neighbor_threshold_2 = neighbor_threshold_1 + 10
 
    #Find skeleton
    reconstruct_surface(
                    segment_image, 
                    original_image, 
                    index=[5, 6], 
                    intensity_threshold_1=intensity_threshold_1, 
                    intensity_threshold_2=intensity_threshold_2, 
                    gaussian_sigma=gaussian_sigma, 
                    distance_threshold=distance_threshold,
                    laplacian_iter=laplacian_iter,
                    folder_path=folder_path,
                    neighbor_threshold_1=neighbor_threshold_1,
                    neighbor_threshold_2=neighbor_threshold_2
                )
    
    # Record end time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Execution time:", elapsed_time, "seconds")