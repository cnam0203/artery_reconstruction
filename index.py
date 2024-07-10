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

datasets_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/'
result_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/info_files/'

select_options = {
    'tof_mra_julia': {
        'pattern': re.compile(r'sub-(\d+)_run-1_mra_eICAB_CW.nii.gz'),
        'sub_names': [],
        'arteries': [1, 2, 3],
        'org_pre_str': 'sub-',
        'org_post_str': '_run-1_mra_eICAB_CW.nii.gz',
        'seg_pre_str': 'sub-',
        'seg_post_str': '_run-1_mra_resampled.nii.gz',
    },
    'pascal': {
        'pattern': re.compile(r'^PT_(.*?)_ToF_eICAB_CW\.nii\.gz$'),
        'sub_names': [],
        'arteries': [1, 2, 3, 5, 6, 7, 8],
        'org_pre_str': 'PT_',
        'org_post_str': '_ToF_eICAB_CW.nii.gz',
        'seg_pre_str': 'PT_',
        'seg_post_str': '_ToF_resampled.nii.gz',
    }
}

for key in select_options:
    sub_names = []
    for filename in os.listdir(datasets_dir+key):
        match = select_options[key]['pattern'].match(filename)
        if match:
            index = match.group(1)
            sub_names.append(index)

    select_options[key]['sub_names'] = sub_names


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')
app = Dash()

app.layout =  html.Div([
    html.H1(children='Artery stenosis map', style={'textAlign':'center'}),
    html.Div([
        html.Label('Choose a dataset:', style={'width': '100px'}),
        dcc.Dropdown(options=[{'label': key, 'value': key} for key in select_options.keys()],
            value='tof_mra_julia', id='dropdown-selection-dataset', style={'width': '25%', 'display': 'inline-block'})
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.Div([    
        html.Label('Choose a subject:', style={'width': '100px'}),
        dcc.Dropdown(id='dropdown-selection-subject', style={'width': '25%', 'display': 'inline-block'})
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.Div([
        html.Label('Percentage of artery length:', style={'width': '200px', 'textAlign': 'right', 'marginRight': '10px'}),
        html.Div([dcc.RangeSlider(
            id='interval-slider',
            min=0,
            max=100,
            step=1,
            marks={i: str(i) for i in range(0, 101, 10)},  # Optional: Show marks at every 10th step
            value=[20, 80],  # Default value
        )], style={"width": "400px"})
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'textAlign': 'center', 'marginBottom': '20px'}),
    html.Div([
        dcc.Graph(id='graph-content', style={'width': '60%'}),
        html.Div([
            html.H2(children='Comparison of measurement', style={'textAlign':'center'}),
            dash_table.DataTable(id='tbl',
                style_cell={
                    'textAlign': 'center',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                },
                style_table={'overflowX': 'auto'},),
            dcc.Graph(id='multiline-graph-1', style={'marginTop': '20px'}),
            dcc.Graph(id='multiline-graph-2', style={'marginTop': '20px'}),
            dcc.Graph(id='multiline-graph-3', style={'marginTop': '20px'}),
            dcc.Graph(id='multiline-graph-4', style={'marginTop': '20px'}),
            dcc.Graph(id='multiline-graph-5', style={'marginTop': '20px'}),
            dcc.Graph(id='multiline-graph-6', style={'marginTop': '20px'}),
            dcc.Graph(id='multiline-graph-7', style={'marginTop': '20px'}),
            dcc.Graph(id='multiline-graph-8', style={'marginTop': '20px'})
        ], style={'width': '40%'})
    ], style={'display': 'flex', 'float': 'left', 'width': '100%'}),
])

def ratio_to_level(percent):
    if percent < 0.15:
        return 0
    elif percent < 0.26:
        return '0-1'
    elif percent < 0.45:
        return 1
    elif percent < 0.56:
        return '1-2'
    elif percent < 0.65:
        return 2
    elif percent < 0.76:
        return '2-3'
    elif percent < 0.95:
        return '3'
    else:
        return '3-4'

neurologist_df = pd.read_csv('C:/Users/nguc4116/Desktop/artery_reconstruction/stenosis.csv', sep=',')
mapping_names = {
    1: 'LICA',
    2: 'RICA',
    3: 'BA',
    5: 'LACA',
    6: 'RACA',
    7: 'LMCA',
    8: 'RMCA'
}
mapping_colors = {
    1: 'orange',
    2: 'green',
    3: 'blue',
    5: 'red',
    6: 'black',
    7: 'yellow',
    8: 'pink'
}


def get_figure(value_0, value, value_2, chart_type):
    results = []
    title = ''
    y_title = ''

    for i in select_options[value_0]['arteries']:
        info_dir = result_dir + f"""{str(value)}/measure_output_{str(i)}.csv"""
        if os.path.isfile(info_dir):
            df = pd.read_csv(info_dir)
            df = df.round(2)

            if chart_type == 0:
                results.append({'idx': i, 'line': df['min_distances']})
                title = 'Min diameter by Length'
                y_title = 'Diameter (mm)'
            elif chart_type == 1:
                results.append({'idx': i, 'line': df['avg_radius']})
                title = 'Avg diameter by Length'
                y_title = 'Diameter (mm)'
            elif chart_type == 2:
                avg_diameter = df['avg_radius'].mean()
                results.append({'idx': i, 'line': np.maximum(1-df['min_distances']/avg_diameter,0)})
                title = 'Stenosis ratio by length (min/avg_all_avg)'
                y_title = 'Percentage (%)'
            elif chart_type == 3:
                avg_diameter = df['avg_radius'].mean()
                results.append({'idx': i, 'line': np.gradient(np.maximum(1-df['min_distances']/avg_diameter,0))})
                title = '1st derivative stenosis ratio by length (min/avg_all_avg)'
                y_title = 'Derivative value'
            elif chart_type == 4:
                results.append({'idx': i, 'line': np.maximum(df['stenosis_ratio_min'],0)})
                title = 'Stenosis ratio by length (min/distal)'
                y_title = 'Percentage (%)'
            elif chart_type == 5:
                avg_diameter = df['avg_radius'].mean()
                results.append({'idx': i, 'line': np.gradient(np.maximum(df['stenosis_ratio_min'],0))})
                title = '1st derivative stenosis ratio by length (min/distal)'
                y_title = 'Derivative value'
            elif chart_type == 6:
                avg_diameter = df['min_distances'].mean()
                results.append({'idx': i, 'line': np.maximum(1-df['min_distances']/avg_diameter,0)})
                title = 'Stenosis ratio by length (min/avg_all_min)'
                y_title = 'Percentage (%)'
            elif chart_type == 7:
                avg_diameter = df['min_distances'].mean()
                results.append({'idx': i, 'line': np.gradient(np.maximum(df['stenosis_ratio_min'],0))})
                title = '1st derivative stenosis ratio by length (min/avg_all_min)'
                y_title = 'Derivative value'
            else:
                results.append(None)
        else:
            results.append(None)
    
    x_values = [i * 0.5 for i in range(max(len(line) for line in results))]
    # Colors for the specific marker positions
    marker_colors = ['blue', 'green', 'orange', 'green', 'red']

    # Create the combined line and marker traces
    traces = []
    for result in results:
        if result is None:
            continue

        idx = int(result['idx'])
        line = result['line']

        line_length = len(line)
        x_line_values = [i * 0.5 for i in range(line_length)]
        
        marker_indices = [
            0,
            line_length // 4,
            line_length // 2,
            3 * line_length // 4,
            line_length - 1
        ]
        
        marker_x = [x_line_values[mi] for mi in marker_indices]
        marker_y = [line[mi] for mi in marker_indices]
        marker_colors_for_line = [marker_colors[i] for i in range(len(marker_indices))]
        
        # Prepare the line segments based on the interval
        interval_start_index = int(value_2[0]*len(line)/100)
        interval_end_index = int(value_2[1]*len(line)/100)

        # Create trace with full line as dashed
        traces.append(
            go.Scatter(
                x=x_line_values,
                y=line,
                mode='lines',
                name=mapping_names[idx],
                line={'color': mapping_colors[idx], 'dash': 'dash'},
                showlegend=False,
                legendgroup=mapping_names[idx],
            )
        )
        
        # Modify the interval to solid line
        solid_x_values = x_line_values[interval_start_index:interval_end_index+1]
        solid_y_values = line[interval_start_index:interval_end_index+1]
        
        # Create trace with solid line within the interval
        traces.append(
            go.Scatter(
                x=solid_x_values,
                y=solid_y_values,
                mode='lines',
                name=mapping_names[idx],
                line={'color': mapping_colors[idx], 'dash': 'solid'},
                legendgroup=mapping_names[idx],
            )
        )

        # # Line trace
        # traces.append(
        #     go.Scatter(
        #         x=x_line_values,
        #         y=line,
        #         mode='lines',
        #         name=mapping_names[idx+1],
        #         line={'color': mapping_colors[idx+1]},
        #         legendgroup=mapping_names[idx+1],
        #     )
        # )
        
        # Marker trace
        traces.append(
            go.Scatter(
                x=marker_x,
                y=marker_y,
                mode='markers',
                name=f"{mapping_names[idx]} markers",
                showlegend=False,
                marker=dict(
                    symbol='circle',
                    size=10,
                    color=marker_colors_for_line,
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                legendgroup=mapping_names[idx],
            )
        )

    # Create the figure
    figure = {
        'data': traces,
        'layout': go.Layout(
            title=title,
            xaxis={'title': 'Length (mm)'},
            yaxis={'title': y_title},
            showlegend=True,
            margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
            hovermode='closest'
        )
    }
    return figure

# @callback(
#     Output('multiline-graph-1', 'figure' ),
#     Input('dropdown-selection-subject', 'value')
# )
# def update_line_1(value):
#     return get_figure(value, 0)

# @callback(
#     Output('multiline-graph-2', 'figure' ),
#     Input('dropdown-selection-subject', 'value')
# )
# def update_line_2(value):
#     return get_figure(value, 1)


# @callback(
#     Output('multiline-graph-3', 'figure' ),
#     Input('dropdown-selection-subject', 'value')
# )
# def update_line_3(value):
#     return get_figure(value, 2)

# @callback(
#     Output('multiline-graph-4', 'figure' ),
#     Input('dropdown-selection-subject', 'value')
# )
# def update_line_4(value):
#     return get_figure(value, 3)

# @callback(
#     Output('multiline-graph-5', 'figure' ),
#     Input('dropdown-selection-subject', 'value')
# )
# def update_line_5(value):
#     return get_figure(value, 4)

# @callback(
#     Output('multiline-graph-6', 'figure' ),
#     Input('dropdown-selection-subject', 'value')
# )
# def update_line_5(value):
#     return get_figure(value, 5)

# @callback(
#     Output('tbl', 'data' ),
#     Input('dropdown-selection-subject', 'value')
# )
def update_table(value_0, value, value_2):
    results = []
    neurologist_df['ID'] = neurologist_df['ID'].astype(str)
    neuro_rows = neurologist_df[neurologist_df['ID'] == str(value)]
    
    if len(neuro_rows) > 0:
        neuro_row = neuro_rows.iloc[0]
    else:
        neuro_row = pd.Series({col: 0 for col in neurologist_df.columns})

    for i in select_options[value_0]['arteries']:
        info_dir = result_dir + f"""{str(value)}/measure_output_{str(i)}.csv"""
        if os.path.isfile(info_dir):
            diam_col_name = mapping_names[i] + '_diam'
            sten_col_name = mapping_names[i] + '_stenosis'

            diam = float(neuro_row.get(diam_col_name.lower(), 0))
            sten = float(neuro_row.get(sten_col_name.lower(), 0))

            metrics = {}
            df = pd.read_csv(info_dir)
            df = df.round(2)

            start_idx = int(value_2[0]*len(df['min_distances'])/100)
            end_idx = int(value_2[1]*len(df['min_distances'])/100)

            avg_diameter = df['avg_radius'].mean()
            avg_min_diameter = df['min_distances'].mean()
            metrics['Artery'] = mapping_names[i]
            metrics[f'Neuro_diam (mm)'] = diam
            metrics[f'Actual_min_diam'] = df['min_distances'][start_idx:end_idx].min()
            metrics[f'Actual_avg_diam'] = round(df['avg_radius'][start_idx:end_idx].mean(), 2)
            metrics[f'Neuro_stenosis_level'] = sten
            metrics[f'Act_stenosis_ratio (min/avg_all_avg)'] = round((1-df['min_distances'][start_idx:end_idx]/avg_diameter).max(), 2)
            metrics[f'Act_stenosis_level (min/avg_all_avg)'] = ratio_to_level((1-df['min_distances'][start_idx:end_idx]/avg_diameter).max())
            metrics[f'Act_stenosis_ratio (min/avg_all_min)'] = max(round((1-df['min_distances'][start_idx:end_idx]/avg_min_diameter).max(), 2), 0)
            metrics[f'Act_stenosis_level (min/avg_all_min)'] = ratio_to_level((1-df['min_distances'][start_idx:end_idx]/avg_min_diameter).max())
            metrics[f'Act_stenosis_ratio (min/distal)'] = round((df['stenosis_ratio_min'][start_idx:end_idx]).max(), 2)
            metrics[f'Act_stenosis_level (min/distal)'] = ratio_to_level((df['stenosis_ratio_min'][start_idx:end_idx]).max())

            results.append(metrics)
    return results

def generate_circular_plane_plotly(x1, x2, radius, num_points=100):
    # Normalize the normal vector
    normal_vector = x2 - x1
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Create a random vector
    random_vector = np.random.rand(3)
    
    # Ensure the random vector is not parallel to the normal vector
    if np.dot(normal_vector, random_vector) == 1:
        random_vector = np.random.rand(3)
    
    # Use the Gram-Schmidt process to find a vector orthogonal to the normal vector
    orthogonal_vector1 = random_vector - np.dot(random_vector, normal_vector) * normal_vector
    orthogonal_vector1 /= np.linalg.norm(orthogonal_vector1)
    
    # Find a second vector orthogonal to both the normal vector and the first orthogonal vector
    orthogonal_vector2 = np.cross(normal_vector, orthogonal_vector1)
    orthogonal_vector2 /= np.linalg.norm(orthogonal_vector2)
    
    # Generate the circular grid points
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi = np.linspace(0, radius, num_points)
    theta, phi = np.meshgrid(theta, phi)
    
    x_circle = x1[0] + np.outer(np.cos(theta), phi) * orthogonal_vector1[0] + np.outer(np.sin(theta), phi) * orthogonal_vector2[0]
    y_circle = x1[1] + np.outer(np.cos(theta), phi) * orthogonal_vector1[1] + np.outer(np.sin(theta), phi) * orthogonal_vector2[1]
    z_circle = x1[2] + np.outer(np.cos(theta), phi) * orthogonal_vector1[2] + np.outer(np.sin(theta), phi) * orthogonal_vector2[2]
    
    # Create the Plotly surface
    surface = go.Surface(
        x=x_circle,
        y=y_circle,
        z=z_circle,
        colorscale='Viridis',
        showscale=False,
        opacity=0.75
    )
    
    return surface


# @callback(
#     Output('graph-content', 'figure'),
#     Input('dropdown-selection-subject', 'value')
# )
def update_graph(value_0, value, value_2):
    dataset_name = value_0
    sub_num = value


    # segment_file_path = dataset_dir + f'sub-{str(sub_num)}_run-1_mra_eICAB_CW.nii.gz'
    # original_file_path = dataset_dir + f'sub-{str(sub_num)}_run-1_mra_resampled.nii.gz'
    # segment_file_path = dataset_dir + f'PT_{str(sub_num)}_ToF_eICAB_CW.nii.gz'
    # original_file_path = dataset_dir + f'PT_{str(sub_num)}_ToF_resampled.nii.gz'

    segment_file_path = datasets_dir + dataset_name + f"""/{select_options[dataset_name]['org_pre_str']}{str(sub_num)}{select_options[dataset_name]['org_post_str']}"""
    original_file_path = datasets_dir + dataset_name + f"""/{select_options[dataset_name]['seg_pre_str']}{str(sub_num)}{select_options[dataset_name]['seg_post_str']}"""

    segment_image = nib.load(segment_file_path)
    original_image = nib.load(original_file_path)

    original_data = original_image.get_fdata()
    segment_data = segment_image.get_fdata()
    voxel_sizes = segment_image.header.get_zooms()
    info = {}
    info_dir = result_dir + str(sub_num) + '/'
    showed_data = []

    vmtk_boundary_vertices_all = []
    vmtk_boundary_faces_all = []
    vert_num = 0
    line_traces = []
    meshes = []
    min_diam_rings = []

    for artery_index in select_options[value_0]['arteries']:
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

        vmtk_boundary_vertices_all.append(vmtk_boundary_vertices)
        vmtk_boundary_faces_all.append(vmtk_boundary_faces + vert_num)
        vert_num += vmtk_boundary_vertices.shape[0]

        # Initialize variables to track the maximum length and its corresponding index
        max_length = 0
        max_index = -1

        # Iterate through each sublist and compare lengths
        for idx, sublist in enumerate(smooth_connected_lines):
            current_length = len(sublist)
            if current_length > max_length:
                max_length = current_length
                max_index = idx
        
        print('Max index:', max_index)
        interval_start_index = int(value_2[0]*len(smooth_connected_lines[max_index])/100)
        interval_end_index = int(value_2[1]*len(smooth_connected_lines[max_index])/100)

        for idx, line in enumerate(smooth_connected_lines):
            # line_traces.append(generate_lines(smooth_points[line], 2))
            if idx != max_index:
                line_traces.append(generate_lines(smooth_points[line], 1))
            else:
                line_traces.append(generate_lines(smooth_points[line[0:interval_start_index+1]], 1))
                line_traces.append(generate_lines(smooth_points[line[interval_start_index:interval_end_index+1]], 5))
                line_traces.append(generate_lines(smooth_points[line[interval_end_index:-1]], 1))

                # measure_dir = f"""C:/Users/nguc4116/Desktop/artery_reconstruction/info_files/{str(value)}/measure_output_{str(artery_index)}.csv"""
                # if os.path.isfile(measure_dir):
                #     df = pd.read_csv(measure_dir)
                #     min_pos = np.argmin(df['min_distances'][interval_start_index:interval_end_index])
                #     length = min_pos * 0.5
                #     distance = 0

                #     for sub_idx, point in enumerate(line[:-1]):
                #         if distance < length:
                #             distance += euclidean_distance(vmtk_boundary_vertices[line[sub_idx]], vmtk_boundary_vertices[line[sub_idx+1]])
                #         else:
                #             squares.append(generate_circular_plane_plotly(vmtk_boundary_vertices[line[sub_idx]], vmtk_boundary_vertices[line[sub_idx+1]], 5))
                #             break

        mesh = generate_mesh(vmtk_boundary_vertices, vmtk_boundary_faces, mapping_names[artery_index], mapping_colors[artery_index])
        meshes.append(mesh)

        with open(info_dir + f'chosen_ring_{artery_index}.json', 'r') as file:
            chosen_ring_vertices = json.load(file)
        measure_dir = result_dir + f"""{str(value)}/measure_output_{str(artery_index)}.csv"""
        
        interval_start_index = int(value_2[0]*len(chosen_ring_vertices)/100)
        interval_end_index = int(value_2[1]*len(chosen_ring_vertices)/100)

        if os.path.isfile(measure_dir):
            df = pd.read_csv(measure_dir)
            min_pos = np.argmin(df['min_distances'][interval_start_index:interval_end_index])
            min_diam_rings.append(vmtk_boundary_vertices[chosen_ring_vertices[min_pos+interval_start_index]])

    vmtk_boundary_vertices_all = np.concatenate(vmtk_boundary_vertices_all, axis=0)
    vmtk_boundary_faces_all = np.concatenate(vmtk_boundary_faces_all, axis=0)

    layout = go.Layout(
        scene=dict(
            aspectmode='manual',
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis =dict(visible=False)
        ),
    )

    # min_diam_rings = np.genfromtxt(info_dir + f'min_vertices.txt', delimiter=',')
    start_points = np.genfromtxt(info_dir + f'start_points.txt', delimiter=',')
    end_points = np.genfromtxt(info_dir + f'end_points.txt', delimiter=',')
    middle_points = np.genfromtxt(info_dir + f'middle_points.txt', delimiter=',')
    cons_points = np.genfromtxt(info_dir + f'cons_points.txt', delimiter=',')

    visualized_start_points = generate_points_name(np.array(start_points), 10, 'blue', 'start point')
    visualized_end_points = generate_points_name(np.array(end_points), 10, 'red', 'end point')
    visualized_middle_points = generate_points_name(np.array(middle_points), 10, 'orange', 'middle point')
    visualized_cons_points = generate_points_name(np.array(cons_points), 10, 'green', '25th/75th point')
    visualized_min_diam_rings = generate_points_name(np.concatenate(min_diam_rings, axis=0), 3, 'red', 'ring with min diam')
    
    show_points = []
    show_points.append(visualized_start_points)
    show_points.append(visualized_end_points)
    show_points.append(visualized_middle_points)
    show_points.append(visualized_cons_points)
    show_points.append(visualized_min_diam_rings)

    fig = go.Figure(data=show_points+meshes+line_traces, layout=layout)
    fig.update_layout(height=1200,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', )
    return fig


@callback(
    [
        Output('graph-content', 'figure'),
        Output('tbl', 'data' ),
        Output('multiline-graph-1', 'figure' ),
        Output('multiline-graph-2', 'figure' ),
        Output('multiline-graph-3', 'figure' ),
        Output('multiline-graph-4', 'figure' ),
        Output('multiline-graph-5', 'figure' ),
        Output('multiline-graph-6', 'figure' ),
        Output('multiline-graph-7', 'figure' ),
        Output('multiline-graph-8', 'figure' ),
    ],
    [
        Input('dropdown-selection-dataset', 'value'),
        Input('dropdown-selection-subject', 'value'),
        Input('interval-slider', 'value')
    ]
)
def update_data(value_0, value_1, value_2):
    output_1 = update_graph(value_0, value_1, value_2)
    output_2 = update_table(value_0, value_1, value_2)
    output_3 = get_figure(value_0, value_1, value_2, 0)
    output_4 = get_figure(value_0, value_1, value_2, 1)
    output_5 = get_figure(value_0, value_1, value_2, 2)
    output_6 = get_figure(value_0, value_1, value_2, 3)
    output_7 = get_figure(value_0, value_1, value_2, 4)
    output_8 = get_figure(value_0, value_1, value_2, 5)
    output_9 = get_figure(value_0, value_1, value_2, 6)
    output_10 = get_figure(value_0, value_1, value_2, 7)

    return output_1, output_2, output_3, output_4, output_5, output_6, output_9, output_10, output_7, output_8


@app.callback(
    Output('dropdown-selection-subject', 'options'),
    Output('dropdown-selection-subject', 'value'),
    Input('dropdown-selection-dataset', 'value')
)
def update_subject_dropdown(selected_dataset):
    subjects = select_options[selected_dataset]['sub_names']
    options = [{'label': sub, 'value': sub} for sub in subjects]
    value = subjects[0] if subjects else None
    return options, value

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)