import nibabel as nib
import open3d as o3d
import numpy as np

from skimage.morphology import skeletonize, thin
from skimage import measure

from scipy.spatial import KDTree, distance_matrix
from sklearn.cluster import DBSCAN, KMeans

from PIL import Image
import matplotlib.pyplot as plt

from preprocess_data import *
from process_graph import *
from visualize_graph import *
from slice_selection import *

import cv2
import math
import time
import os
import json
import heapq
import copy

def reconstruct_surface(segment_image,
                  original_image=None, 
                  index=[], 
                  intensity_threshold_1=0.65, 
                  intensity_threshold_2=0.65, 
                  gaussian_sigma=0, 
                  distance_threshold=20,
                  laplacian_iter=1,
                  folder_path=''):
    
    # Load original image (TOF-MRA) into numpy array
    original_data = original_image.get_fdata()
    segment_data = segment_image.get_fdata()
    voxel_sizes = segment_image.header.get_zooms()
    print('Voxel size: ', voxel_sizes)
    print("Image size:", original_data.shape)

    # Preprocess data: Select data for desired arteries, remove noise voxels
    '''
        mask_data is a binary mask for desired arteries
        cex_data is used for centerline extraction with higher intensity threshold
        surf_data is udes for surface reconstruction with lower intensity threshold
    '''
    mask_data, cex_data, surf_data = preprocess_data(original_data, segment_data, index, intensity_threshold_1, intensity_threshold_2, gaussian_sigma)
    
    # Find the voxel-based skeleton
    skeleton = skeletonize(cex_data)
    skeleton_points = np.argwhere(skeleton != 0)
    
    # Find endpoints, junctions from skeletons
    end_points, junction_points, neighbor_distances = find_graph(skeleton_points)

    if 5 in index or 6 in index:
        aca_endpoints = [15, 701]
        directions = ['left', 'right']
        endpoint_vector = skeleton_points[aca_endpoints[0]] - skeleton_points[aca_endpoints[1]]
        
        # Remove redundant junction points
        junction_points, neighbor_distances = remove_junction_points(neighbor_distances, junction_points, skeleton_points)
        junction_points, reduced_distances, connected_lines = reduce_skeleton_points(neighbor_distances, junction_points, end_points, skeleton_points)
        junction_points, neighbor_distances = remove_cycle_edges(skeleton_points, reduced_distances, neighbor_distances, endpoint_vector, connected_lines)
        junction_points, reduced_distances, connected_lines = reduce_skeleton_points(neighbor_distances, junction_points, end_points, skeleton_points)
        skeleton_points, connected_lines, edges = remove_duplicate_paths(skeleton_points, reduced_distances, connected_lines)
        common_paths, left_paths, right_paths, undefined_paths = find_branches(aca_endpoints, end_points, directions, skeleton_points, junction_points, edges)
        skeleton_points, split_groups = find_split_points(common_paths, original_data, skeleton_points)
        split_paths = connect_split_points(split_groups, skeleton_points)
        common_paths, left_paths, right_paths, connected_lines, defined_paths = connect_paths(common_paths, left_paths, right_paths, connected_lines, split_paths, skeleton_points) 
        print(defined_paths)
        left_paths, right_paths, undefined_paths, connected_lines = connect_undefined_paths(left_paths, right_paths, connected_lines, defined_paths, undefined_paths, common_paths, split_paths, skeleton_points)
        
    visualized_skeleton_points = generate_points(skeleton_points)
    visualized_end_points = generate_points(skeleton_points[end_points], 5, 'red')
    visualized_junction_points = generate_points(skeleton_points[junction_points], 5, 'green')
    visualized_direct_lines = []
    visualized_connected_lines = []
    
    # for line in connected_lines:
    #     visualized_direct_lines.append(generate_lines(skeleton_points[[line[0], line[-1]]], 5))
    #     for k in range(len(line)-1):
    #         visualized_connected_lines.append(generate_lines(skeleton_points[[line[k], line[k+1]]]))
    
    line_groups = [right_paths, left_paths, undefined_paths]
    line_colors = ['blue', 'green', 'orange']
    line_traces = []

    for i, line_group in enumerate(line_groups):
        for line in line_group:
            for connected_line in connected_lines:
                if (line[0] in connected_line and line[-1] in connected_line):
                    color = line_colors[i]
                    line_traces.append(generate_lines(skeleton_points[connected_line], 2, color))
    
    # visualized_split_points = []
    # visualized_split_paths = []
    # for group in split_groups:
    #     visualized_split_points.append(generate_points(skeleton_points[group]))
        
    show_figure([visualized_end_points, visualized_junction_points] + line_traces)
    
    return

if __name__ == "__main__":

    # Calculate runtime - record start time
    start_time = time.time()

    # Specify the path to your NIfTI file
    segment_file_path = '/Users/apple/Downloads/sub61_harvard_watershed.nii.gz'
    original_file_path = '/Users/apple/Downloads/sub-61_acq-tof_angio_resampled.nii.gz'

    # Create a folder holding all outpout files
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
                    folder_path=folder_path
                )
                
    # Record end time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Execution time:", elapsed_time, "seconds")