from dash import Dash, html, dcc, callback, Output, Input, dash_table, State
import dash
import plotly.express as px
import pandas as pd
import numpy as np
import nibabel as nib
import re
import os
import json
import csv
from skimage import measure
from visualize_graph import *
from process_graph import *
from ref_measurement import *
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize

def compute_curvature_derivative(curvatures):
    """
    Computes the derivative of the curvature along a curve.

    Parameters:
    curvatures (np.array): Array of curvature values along the curve.

    Returns:
    np.array: Derivative of curvature at each point.
    """
    # Compute the derivative of curvature
    curvature_derivative = np.gradient(curvatures)
    return curvature_derivative

def compute_curvature(points, percentage=0.05):
    """
    Computes the curvature at each point of a 3D line by comparing with two points
    that are a fixed percentage of the total length away.

    Parameters:
    points (np.array): N*3 numpy array of points.
    percentage (float): Percentage of the total length to determine comparison points.

    Returns:
    np.array: Curvature at each point.
    """
    N = len(points)
    curvatures = np.zeros(N)

    # Calculate the distances between consecutive points
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = np.sum(distances)

    # Determine the number of points to skip based on the percentage
    num_points_to_skip = int(percentage * N)

    for i in range(N):
        # Find the indices of the points to compare with
        idx1 = max(0, i - num_points_to_skip)
        idx2 = min(N - 1, i + num_points_to_skip)

        # If the indices are the same as the current point, skip curvature calculation
        if idx1 == i or idx2 == i:
            continue

        # Compute the vectors
        vec1 = points[idx1] - points[i]
        vec2 = points[idx2] - points[i]

        # Compute the angle between the vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            continue

        dot_product = np.dot(vec1, vec2)
        angle = np.arccos(dot_product / (norm1 * norm2))

        # Compute the curvature
        curvature = 2 * np.sin(angle / 2) / norm1
        curvatures[i] = curvature

    return curvatures

result_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/info_files/'
neurologist_df = pd.read_csv('C:/Users/nguc4116/Desktop/artery_reconstruction/stenosis.csv', sep=',')

mapping_names = {
    1: 'LICA',
    2: 'RICA',
    3: 'BA',
    5: 'LACA',
    6: 'RACA',
    7: 'LMCA',
    8: 'RMCA',
    17: 'LAchA',
    18: 'RAchA',
}
mapping_colors = {
    1: 'orange',
    2: 'green',
    3: 'blue',
    5: 'red',
    6: 'black',
    7: 'yellow',
    8: 'pink',
    17: 'purple',
    18: 'gray',
}

options = {
    'pascal': {
        'dataset_dir': 'E:/pascal',
        'pattern': re.compile(r'^PT_(.*?)_ToF_eICAB_CW\.nii\.gz$'),
        'sub_names': [],
        'arteries': [1, 2, 3, 17, 18],
        'org_pre_str': 'PT_',
        'org_post_str': '_ToF_eICAB_CW.nii.gz',
        'seg_pre_str': 'PT_',
        'seg_post_str': '_ToF_resampled.nii.gz',
    },
    'stenosis': {
        'dataset_dir': 'E:/stenosis/',
        'pattern': re.compile(r'sub-(\d+)_run-1_mra_eICAB_CW.nii.gz'),
        'arteries': [2],
        'is_replace': True,
        'org_pre_str': 'sub-',
        'org_post_str': '_run-1_mra_eICAB_CW.nii.gz',
        'seg_pre_str': 'sub-',
        'seg_post_str': '_run-1_mra_resampled.nii.gz',
    },
    'tof_mra_julia': {
        'dataset_dir': 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/tof_mra_julia',
        'pattern': re.compile(r'sub-(\d+)_run-1_mra_eICAB_CW.nii.gz'),
        'sub_names': [],
        'arteries': [1, 2, 3, 5, 6, 7, 8],
        'org_pre_str': 'sub-',
        'org_post_str': '_run-1_mra_eICAB_CW.nii.gz',
        'seg_pre_str': 'sub-',
        'seg_post_str': '_run-1_mra_resampled.nii.gz',
    },
}

stenosis_methods = {
    1: 'entire',
    # 2: '2. local_min/avg_global_min',
    # 3: '3. local_min/avg_disprox_min',
    4: 'disprox',
    # 5: '5. local_avg/avg_global_avg',
    # 6: '6. local_avg/avg_disprox_avg',
    # 7: '7. local_min/avg_distal_min',
    # 8: '8. local_min/avg_proximal_min',
    9: 'distal',
    10: 'proximal',
}

for key in options:
    sub_names = []
    for filename in os.listdir(options[key]['dataset_dir']):
        match = options[key]['pattern'].match(filename)
        if match:
            index = match.group(1)
            sub_names.append(index)

    # with open(f'C:/Users/nguc4116/Desktop/{key}.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for line in sub_names:
    #         writer.writerow([line])

    options[key]['sub_names'] = sub_names

app = Dash()

app.layout =  html.Div([
    html.H1(children='Artery stenosis map', style={'textAlign':'center'}),
    html.Div([
        html.Div([dcc.Graph(id='graph-content-2', style={'width': '100%'})], style={'width': '50%'}),
        html.Div([dcc.Graph(id='graph-content', style={'width': '100%'})], style={'width': '50%'}),
        html.Div([
            html.Div([
                html.Label('Choose a dataset:', style={'width': '100px'}),
                dcc.Dropdown(options=[{'label': key, 'value': key} for key in options.keys()],
                    value='stenosis', id='dropdown-selection-dataset', style={'width': '300px', 'display': 'inline-block'})
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div([    
                html.Label('Choose a subject:', style={'width': '100px'}),
                dcc.Dropdown(id='dropdown-selection-subject', style={'width': '300px', 'display': 'inline-block'})
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div([
                html.Label('Neighboring percentage:', style={'width': '200px', 'textAlign': 'right', 'marginRight': '10px'}),
                html.Div([dcc.Slider(
                    id='slider',
                    min=0,
                    max=20,
                    step=1,
                    marks={i: str(i) for i in range(0, 21, 2)},  # Optional: Show marks at every 10th step
                    value=10,  # Default value
                )], style={"width": "400px"})
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div([
                html.Label('Percentage of artery length:', style={'width': '200px', 'textAlign': 'right', 'marginRight': '10px'}),
                html.Div([dcc.RangeSlider(
                    id='interval-slider',
                    min=0,
                    max=100,
                    step=1,
                    marks={i: str(i) for i in range(0, 101, 10)},  # Optional: Show marks at every 10th step
                    value=[25, 75],  # Default value
                )], style={"width": "400px"})
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div([dash_table.DataTable(id='tbl',
            style_cell={
                'textAlign': 'center',
                'whiteSpace': 'normal',
                'height': 'auto',
            },)]),
        ], style={'width': '40%'})
    ], style={'display': 'flex'}),
])

@app.callback(
    Output('dropdown-selection-subject', 'options'),
    Output('dropdown-selection-subject', 'value'),
    Input('dropdown-selection-dataset', 'value')
)
def update_subject_dropdown(selected_dataset):
    subjects = options[selected_dataset]['sub_names']
    sub_options = [{'label': sub, 'value': sub} for sub in subjects]
    value = subjects[0] if subjects else None
    return sub_options, value

def update_graph(original_data, segment_data, voxel_sizes, value_0, value, value_2):
    dataset_name = value_0
    sub_num = value
    info = {}
    info_dir = result_dir + str(sub_num) + '/'
    showed_data = []

    vmtk_boundary_vertices_all = []
    vmtk_boundary_faces_all = []
    vert_num = 0
    line_traces = []
    meshes = []
    min_diam_rings = []
    middle_points_all = []

    start_points = []
    end_points = []
    middle_points = []
    cons_points = []

    for artery_index in options[value_0]['arteries']:
        artery_key = "Artery_" + str(artery_index)
        info[artery_key] = []
        min_vertices = []

        if not os.path.isfile(info_dir + f'smooth_points_{artery_index}.txt'):
            continue

        smooth_points = np.genfromtxt(info_dir + f'smooth_points_{artery_index}.txt', delimiter=',')
        vmtk_boundary_vertices = np.genfromtxt(info_dir + f'vmtk_boundary_vertices_{artery_index}.txt', delimiter=',')
        vmtk_boundary_vertices = np.round(vmtk_boundary_vertices, 2)
        vmtk_boundary_faces = np.genfromtxt(info_dir + f'vmtk_boundary_faces_{artery_index}.txt', delimiter=',', dtype=int)

        vmtk_boundary_vertices, inverse_indices = np.unique(vmtk_boundary_vertices, axis=0, return_inverse=True)
        vmtk_boundary_faces = np.array([inverse_indices[face] for face in vmtk_boundary_faces])

        with open(info_dir + f'smooth_connected_lines_{artery_index}.json', 'r') as file:
            smooth_connected_lines = json.load(file)

        for idx, line in enumerate(smooth_connected_lines):
            line_traces.append(generate_lines(smooth_points[line], 2))

        mesh = generate_mesh(vmtk_boundary_vertices, vmtk_boundary_faces, mapping_names[artery_index], mapping_colors[artery_index])
        meshes.append(mesh)

    layout = go.Layout(
        scene=dict(
            aspectmode='manual',
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis =dict(visible=False)
        ),
    )



    fig = go.Figure(data=meshes+line_traces+[go.Scatter3d()], layout=layout)
    fig.update_layout(height=500,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(
        x=0,       # position on the x-axis (0 is left)
        y=1,       # position on the y-axis (1 is top)
        xanchor='left',  # anchor the legend at the left
        yanchor='top'    # anchor the legend at the top
    ) )
    return fig

def update_graph_2(original_data, segment_data, voxel_sizes, value_0, value, value_2):
    # dataset_name = value_0
    # sub_num = value
    # info = {}
    # info_dir = result_dir + str(sub_num) + '/'
    # showed_data = []

    # vmtk_boundary_vertices_all = []
    # vmtk_boundary_faces_all = []
    # vert_num = 0
    # line_traces = []
    # meshes = []
    # min_diam_rings = []
    # middle_points_all = []

    # start_points = []
    # end_points = []
    # middle_points = []
    # cons_points = []
    # line_traces = []

    # for artery_index in options[value_0]['arteries']:
    #     mask_data = np.copy(segment_data)
    #     mask = np.isin(mask_data, artery_index, invert=True)
    #     mask_data[mask] = 0
    #     mask_data[mask_data != 0] = 1

    #     skeleton = skeletonize(mask_data)
    #     skeleton_points, end_points, junction_points, connected_lines = find_graphs(skeleton)
    #     skeleton_points = skeleton_points*voxel_sizes

        
    #     for idx, line in enumerate(connected_lines):
    #         line_traces.append(generate_lines(skeleton_points[line], 2))

    #     vertices, faces, normals, values = measure.marching_cubes(mask_data, level=0.5, spacing=voxel_sizes)

    #     mesh = generate_mesh(vertices, faces, mapping_names[artery_index], mapping_colors[artery_index])
    #     meshes.append(mesh)


    layout = go.Layout(
        scene=dict(
            aspectmode='manual',
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis =dict(visible=False)
        ),
    )

    fig =  go.Figure(layout=layout)
    # fig = go.Figure(data=meshes+line_traces, layout=layout)
    fig.update_layout(height=500,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(
        x=0,       # position on the x-axis (0 is left)
        y=1,       # position on the y-axis (1 is top)
        xanchor='left',  # anchor the legend at the left
        yanchor='top'    # anchor the legend at the top
    ) )
    return fig

@callback(
    [
        Output('graph-content-2', 'figure'),
        Output('graph-content', 'figure'),
    ],
    [
        Input('dropdown-selection-dataset', 'value'),
        Input('dropdown-selection-subject', 'value'),
        Input('interval-slider', 'value'),
        Input('slider', 'value'),
    ]
)
def update_data(value_0, value_1, value_2, value_3):
    dataset_name = value_0
    sub_num = value_1

    segment_file_path = options[dataset_name]['dataset_dir'] + f"""/{options[dataset_name]['org_pre_str']}{str(sub_num)}{options[dataset_name]['org_post_str']}"""
    original_file_path = options[dataset_name]['dataset_dir'] + f"""/{options[dataset_name]['seg_pre_str']}{str(sub_num)}{options[dataset_name]['seg_post_str']}"""

    segment_image = nib.load(segment_file_path)
    original_image = nib.load(original_file_path)

    original_data = original_image.get_fdata()
    segment_data = segment_image.get_fdata()
    voxel_sizes = segment_image.header.get_zooms()

    output_0 = update_graph_2(original_data, segment_data, voxel_sizes, value_0, value_1, value_2)
    output_1 = update_graph(original_data, segment_data, voxel_sizes, value_0, value_1, value_2)
   
    return output_0, output_1

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)