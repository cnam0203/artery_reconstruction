from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import nibabel as nib
import re
import os
import json
from visualize_graph import *
from process_graph import *
from ref_measurement import *
import itertools
import trimesh
import open3d as o3d
from collections import defaultdict, deque

def sort_tuples(tuples):
    # Create a directed graph
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    for u, v in tuples:
        graph[u].append(v)
        out_degree[u] += 1
        in_degree[v] += 1

    # Find the starting node
    start_node = None
    for node in out_degree:
        if out_degree[node] - in_degree[node] == 1:
            start_node = node
            break
    if start_node is None:
        start_node = tuples[0][0]

    # Hierholzer's algorithm to find Eulerian path
    def find_eulerian_path(graph, start):
        stack = [start]
        path = []
        while stack:
            node = stack[-1]
            if graph[node]:
                next_node = graph[node].pop()
                stack.append(next_node)
            else:
                path.append(stack.pop())
        return path[::-1]

    # Get the Eulerian path
    eulerian_path = find_eulerian_path(graph, start_node)

    # Convert path to list of tuples
    sorted_tuples = [(eulerian_path[i], eulerian_path[i+1]) for i in range(len(eulerian_path)-1)]

    return sorted_tuples

# Function to get unique edges
def get_unique_edges(faces):
    edge_count = {}
    
    # Iterate over each face and extract edges
    for face in faces:
        # Create edges, ensuring each edge is represented in sorted order (to avoid directional differences)
        edges = [(min(face[i], face[(i + 1) % 3]), max(face[i], face[(i + 1) % 3])) for i in range(3)]
        
        for edge in edges:
            if edge in edge_count:
                edge_count[edge] += 1
            else:
                edge_count[edge] = 1
    
    # Select edges that occur only once
    unique_edges = [edge for edge, count in edge_count.items() if count == 1]
    
    return unique_edges

def is_point_in_mesh(mesh, point):
    return mesh.contains([point])[0]

def find_closest_points(point, arr2):
    
    # Initialize the minimum distance and indices
    min_distance = 10000
    min_idx = None

    # Calculate pairwise distances and find the minimum
    for j in range(len(arr2)):
        distance = np.linalg.norm(point - arr2[j])
        if distance < min_distance:
            min_distance = distance
            min_idx = j
    
    return min_idx, min_distance

# Create a mesh from vertices and faces
info_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/info_files/PL002_V1/'
line_traces = []
visualized_all_points = []
meshes = {}
visualized_meshes = []

for artery_index in ['1', '7']:
    artery_key = "Artery_" + str(artery_index)

    smooth_points = np.genfromtxt(info_dir + f'smooth_points_{artery_index}.txt', delimiter=',')
    vmtk_boundary_vertices = np.genfromtxt(info_dir + f'vmtk_boundary_vertices_{artery_index}.txt', delimiter=',')
    vmtk_boundary_vertices = np.round(vmtk_boundary_vertices, 2)
    vmtk_boundary_faces = np.genfromtxt(info_dir + f'vmtk_boundary_faces_{artery_index}.txt', delimiter=',', dtype=int)

    vmtk_boundary_vertices, inverse_indices = np.unique(vmtk_boundary_vertices, axis=0, return_inverse=True)
    vmtk_boundary_faces = np.array([inverse_indices[face] for face in vmtk_boundary_faces])

    with open(info_dir + f'smooth_connected_lines_{artery_index}.json', 'r') as file:
        smooth_connected_lines = json.load(file)

    line = smooth_connected_lines[0]
    line_traces.append(generate_lines(smooth_points[line], 1))
    
    meshes[artery_index] = trimesh.Trimesh(vertices=vmtk_boundary_vertices, faces=vmtk_boundary_faces)

vmtk_boundary_vertices_all = []
vmtk_boundary_faces_all = []
border_point_lists = []
count = 0

for artery_index in ['1', '7']:
    artery_key = "Artery_" + str(artery_index)

    smooth_points = np.genfromtxt(info_dir + f'smooth_points_{artery_index}.txt', delimiter=',')
    vmtk_boundary_vertices = np.genfromtxt(info_dir + f'vmtk_boundary_vertices_{artery_index}.txt', delimiter=',')
    vmtk_boundary_vertices = np.round(vmtk_boundary_vertices, 2)
    vmtk_boundary_faces = np.genfromtxt(info_dir + f'vmtk_boundary_faces_{artery_index}.txt', delimiter=',', dtype=int)

    vmtk_boundary_vertices, inverse_indices = np.unique(vmtk_boundary_vertices, axis=0, return_inverse=True)
    vmtk_boundary_faces = np.array([inverse_indices[face] for face in vmtk_boundary_faces])

    removed_point = []
    for idx, point in enumerate(vmtk_boundary_vertices):
        for key in meshes:
            if key != artery_index:
                if is_point_in_mesh(meshes[key], point):
                    removed_point.append(idx)

    removed_face = []
    for idx, face in enumerate(vmtk_boundary_faces):
        if face[0] in removed_point or face[1] in removed_point or face[2] in removed_point:
            removed_face.append(idx)

    filtered_faces = np.delete(vmtk_boundary_faces, removed_face, axis=0)
    mesh = generate_mesh(vmtk_boundary_vertices, filtered_faces, '', 'black')
    visualized_meshes.append(mesh)
    
    unique_edges = get_unique_edges(filtered_faces)
    list_points = []
    if len(unique_edges):
        list_points.append(unique_edges[0][0])
        list_points.append(unique_edges[0][1])
    is_end = False
    while (not is_end):
        is_end = True
        for edge in unique_edges:
            if edge[0] == list_points[-1] and edge[1] != list_points[-2]:
                list_points.append(edge[1])
                is_end = False
                break
            elif edge[1] == list_points[-1] and edge[0] != list_points[-2]:
                list_points.append(edge[0])
                is_end = False
                break

        if list_points[0] == list_points[-1]:
            list_points.pop()
            is_end = True

    list_points = [point + count for point in list_points]
    filtered_faces += count
    border_point_lists.append(list_points)
    vmtk_boundary_vertices_all.append(vmtk_boundary_vertices)
    vmtk_boundary_faces_all.append(filtered_faces)
    count += vmtk_boundary_vertices.shape[0]

vmtk_boundary_vertices_all = np.concatenate(vmtk_boundary_vertices_all, axis=0)

list_1 = border_point_lists[0]
list_2 = border_point_lists[1]


points_1 = np.array(vmtk_boundary_vertices_all[list_1])
points_2 = np.array(vmtk_boundary_vertices_all[list_2])
min_idx, _ = find_closest_points(points_1[0], points_2)
list_2 = list_2[min_idx:] + list_2[0:min_idx]
points_2 = np.array(vmtk_boundary_vertices_all[list_2])

distance_1 = np.linalg.norm(points_1[1] - points_2[1])
distance_2 = np.linalg.norm(points_1[1] - points_2[-1])
if distance_1 > distance_2:
    list_2 = list_2[::-1]
    list_2 = [list_2[-1]] + list_2[:-1]

new_faces = []
if len(list_1) > len(list_2):
    tmp_list = list_1
    list_1 = list_2
    list_2 = tmp_list

end_idx = len(list_1) - 1

for i in range(0, end_idx):
    new_faces.append([list_1[i], list_1[i+1], list_2[i]])
    new_faces.append([list_2[i], list_2[i+1], list_1[i+1]])

if len(list_2) > len(list_1):
    for i in range(end_idx, len(list_2)):
        new_faces.append([list_1[end_idx], list_2[i+1], list_2[i]])

new_faces.append([list_1[-1], list_2[-1], list_1[0]])
new_faces.append([list_1[0], list_2[-1], list_2[0]])

new_faces = np.array(new_faces)
vmtk_boundary_faces_all.append(new_faces)
vmtk_boundary_faces_all = np.concatenate(vmtk_boundary_faces_all, axis=0)

mesh = generate_mesh(vmtk_boundary_vertices_all, vmtk_boundary_faces_all, '', 'black')
show_points = generate_points(np.array([vmtk_boundary_vertices_all[list_1[-1]],
vmtk_boundary_vertices_all[list_1[0]], 
vmtk_boundary_vertices_all[list_2[0]], 
vmtk_boundary_vertices_all[list_2[-1]]]), 5, 'red')

# Create edges
edges = set()
for face in new_faces:
    for i in range(3):
        edge = tuple(sorted((face[i], face[(i + 1) % 3])))
        edges.add(edge)

edge_x = []
edge_y = []
edge_z = []
for edge in edges:
    edge_x.extend([vmtk_boundary_vertices_all[edge[0], 0], vmtk_boundary_vertices_all[edge[1], 0], None])
    edge_y.extend([vmtk_boundary_vertices_all[edge[0], 1], vmtk_boundary_vertices_all[edge[1], 1], None])
    edge_z.extend([vmtk_boundary_vertices_all[edge[0], 2], vmtk_boundary_vertices_all[edge[1], 2], None])

lines = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='black', width=2)
)


show_figure([mesh])
        


