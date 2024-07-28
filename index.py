from dash import Dash, html, dcc, callback, Output, Input, dash_table, State
import dash
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
from scipy.ndimage import gaussian_filter1d

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
    'stenosis': {
        'dataset_dir': 'E:/stenosis/',
        'pattern': re.compile(r'sub-(\d+)_run-1_mra_eICAB_CW.nii.gz'),
        'arteries': [1, 2, 3],
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
        'arteries': [1, 2, 3],
        'org_pre_str': 'sub-',
        'org_post_str': '_run-1_mra_eICAB_CW.nii.gz',
        'seg_pre_str': 'sub-',
        'seg_post_str': '_run-1_mra_resampled.nii.gz',
    },
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
}

stenosis_methods = {
    1: 'global',
    # 2: '2. local_min/avg_global_min',
    # 3: '3. local_min/avg_disprox_min',
    4: 'disprox',
    # 5: '5. local_avg/avg_global_avg',
    # 6: '6. local_avg/avg_disprox_avg',
    # 7: '7. local_min/avg_distal_min',
    # 8: '8. local_min/avg_proximal_min',
    # 9: 'distal',
    # 10: 'proximal',
}

for key in options:
    sub_names = []
    for filename in os.listdir(options[key]['dataset_dir']):
        match = options[key]['pattern'].match(filename)
        if match:
            index = match.group(1)
            sub_names.append(index)

    options[key]['sub_names'] = sub_names

app = Dash()

app.layout =  html.Div([
    html.H1(children='Artery stenosis map', style={'textAlign':'center'}),
    html.Div([
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
                    value=[20, 80],  # Default value
                )], style={"width": "400px"})
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div([dash_table.DataTable(id='tbl',
            style_cell={
                'textAlign': 'center',
                'whiteSpace': 'normal',
                'height': 'auto',
            },)]),
        ], style={'width': '40%'}),
        html.Div([dcc.Graph(id='graph-content', style={'width': '100%'})], style={'width': '60%'})
    ], style={'display': 'flex'}),
    html.Div([
        # html.H2(children='Comparison of measurement', style={'textAlign':'center'}),
        html.Div([
            dcc.Graph(id='multiline-graph-7', style={'marginTop': '20px', 'width': '50%'}),
            dcc.Graph(id='multiline-graph-8', style={'marginTop': '20px', 'width': '50%'}),
            dcc.Graph(id='multiline-graph-10', style={'marginTop': '20px', 'width': '50%'}),
            dcc.Graph(id='multiline-graph-9', style={'marginTop': '20px', 'width': '50%'}),
            dcc.Graph(id='multiline-graph-5', style={'marginTop': '20px', 'width': '50%'}),
            dcc.Graph(id='multiline-graph-3', style={'marginTop': '20px', 'width': '50%'}),
            dcc.Graph(id='multiline-graph-6', style={'marginTop': '20px', 'width': '50%'}),
            dcc.Graph(id='multiline-graph-1', style={'marginTop': '20px', 'width': '50%'}),
            dcc.Graph(id='multiline-graph-2', style={'marginTop': '20px', 'width': '50%'}),
            dcc.Graph(id='multiline-graph-4', style={'marginTop': '20px', 'width': '50%'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    ], style={'width':'100%'}),
])

@app.callback(
    [Output('graph-content', 'figure', allow_duplicate=True),
        Output('multiline-graph-1', 'figure', allow_duplicate=True),
        Output('multiline-graph-2', 'figure', allow_duplicate=True),
        Output('multiline-graph-3', 'figure', allow_duplicate=True),
        Output('multiline-graph-4', 'figure', allow_duplicate=True),
        Output('multiline-graph-5', 'figure', allow_duplicate=True),
        Output('multiline-graph-6', 'figure', allow_duplicate=True),
        Output('multiline-graph-7', 'figure', allow_duplicate=True),
        Output('multiline-graph-8', 'figure', allow_duplicate=True),
        Output('multiline-graph-9', 'figure', allow_duplicate=True),
        Output('multiline-graph-10', 'figure', allow_duplicate=True)
    ],
    [Input('multiline-graph-1', 'clickData'), 
    Input('multiline-graph-2', 'clickData'), 
    Input('multiline-graph-3', 'clickData'), 
    Input('multiline-graph-4', 'clickData'), 
    Input('multiline-graph-5', 'clickData'), 
    Input('multiline-graph-6', 'clickData'),
    Input('multiline-graph-7', 'clickData'),
    Input('multiline-graph-8', 'clickData'),
    Input('multiline-graph-9', 'clickData'),
    Input('multiline-graph-10', 'clickData'),],
    [Input('slider', 'value'), 
        State('multiline-graph-1', 'figure'), 
        State('multiline-graph-2', 'figure'), 
        State('multiline-graph-3', 'figure'), 
        State('multiline-graph-4', 'figure'), 
        State('multiline-graph-5', 'figure'), 
        State('multiline-graph-6', 'figure'), 
        State('multiline-graph-7', 'figure'), 
        State('multiline-graph-8', 'figure'), 
        State('multiline-graph-9', 'figure'), 
        State('multiline-graph-10', 'figure'), 
        State('graph-content', 'figure'), 
        State('dropdown-selection-dataset', 'value'),
        State('dropdown-selection-subject', 'value'),
    ],
    prevent_initial_call=True, 
)
def display_hover_data(clickData_1, clickData_2, clickData_3, clickData_4, clickData_5, clickData_6, clickData_7, clickData_8, clickData_9, clickData_10, length_percentage, multiline_figure_1, multiline_figure_2, multiline_figure_3, multiline_figure_4, multiline_figure_5, multiline_figure_6, multiline_figure_7, multiline_figure_8, multiline_figure_9, multiline_figure_10, graph_figure, dataset_value, subject_value):
    ctx = dash.callback_context
    if ctx.triggered:
        clickData = None
        if clickData_1:
            clickData = clickData_1
            multiline_figure = multiline_figure_1
        if clickData_2:
            clickData = clickData_2
            multiline_figure = multiline_figure_2
        if clickData_3:
            clickData = clickData_3
            multiline_figure = multiline_figure_3
        if clickData_4:
            clickData = clickData_4
            multiline_figure = multiline_figure_4
        if clickData_5:
            clickData = clickData_5
            multiline_figure = multiline_figure_5
        if clickData_6:
            clickData = clickData_6
            multiline_figure = multiline_figure_6
        if clickData_7:
            clickData = clickData_7
            multiline_figure = multiline_figure_7
        if clickData_9:
            clickData = clickData_9
            multiline_figure = multiline_figure_9
        if clickData_8:
            clickData = clickData_8
            multiline_figure = multiline_figure_8
        if clickData_10:
            clickData = clickData_10
            multiline_figure = multiline_figure_10

        if clickData:
            point_index = clickData['points'][0]['pointIndex']
            trace_index = clickData['points'][0]['curveNumber']
            custom_data = multiline_figure['data'][trace_index]['legendgroup']
            artery_index = int(custom_data.split('.')[0])
            ring_filename = result_dir + subject_value + f'/chosen_ring_{artery_index}.json'
            point_filename = result_dir + subject_value + f'/vmtk_boundary_vertices_{artery_index}.txt'
            if os.path.isfile(ring_filename):
                with open(ring_filename, 'r') as file:
                    chosen_ring_vertices = json.load(file)
                    points = np.genfromtxt(point_filename, delimiter=',')
                    points, inverse_indices = np.unique(points, axis=0, return_inverse=True)
                    point_index = int(clickData['points'][0]['x']/0.5)

                    show_points = []
                    chosen_points = points[chosen_ring_vertices[point_index]]
                    show_points.append(chosen_points)

                    step_num = int(length_percentage/100*len(chosen_ring_vertices))
                    x0 = point_index*0.5
                    x1 = point_index*0.5
                    x2 = point_index*0.5

                    if point_index - step_num >= 0:
                        x0 = (point_index - step_num)*0.5
                        # chosen_points = points[chosen_ring_vertices[point_index- step_num]]
                        # show_points.append(chosen_points)

                    if point_index + step_num < len(chosen_ring_vertices):
                        x1 = (point_index + step_num)*0.5
                        # chosen_points = points[chosen_ring_vertices[point_index + step_num]]
                        # show_points.append(chosen_points)

                    visualized_rings = generate_points_name(np.concatenate(show_points, axis=0), 3, 'red', 'ring with min diam')
                    graph_figure['data'][-1] = visualized_rings

                    shape_1 = {
                            'type': 'rect',
                            'xref': 'x',
                            'yref': 'paper',
                            'x0': x0,
                            'x1': x1,
                            'y0': 0,
                            'y1': 1,
                            'fillcolor': 'rgba(128, 128, 128, 0.2)',
                            'line': {'width': 0}
                        }
                    shape_2 = {
                            'type': 'rect',
                            'xref': 'x',
                            'yref': 'paper',
                            'x0': x2,
                            'x1': x2,
                            'y0': 0,
                            'y1': 1,
                            'fillcolor': 'red',
                            'line': {'width': 2}
                        }
                    multiline_figure_1['layout']['shapes'] = [shape_1, shape_2]
                    multiline_figure_2['layout']['shapes'] = [shape_1, shape_2]
                    multiline_figure_3['layout']['shapes'] = [shape_1, shape_2]
                    multiline_figure_4['layout']['shapes'] = [shape_1, shape_2]
                    multiline_figure_5['layout']['shapes'] = [shape_1, shape_2]
                    multiline_figure_6['layout']['shapes'] = [shape_1, shape_2]
                    multiline_figure_7['layout']['shapes'] = [shape_1, shape_2]
                    multiline_figure_8['layout']['shapes'] = [shape_1, shape_2]
                    multiline_figure_9['layout']['shapes'] = [shape_1, shape_2]
                    multiline_figure_10['layout']['shapes'] = [shape_1, shape_2]
    
    return graph_figure, multiline_figure_1, multiline_figure_2, multiline_figure_3, multiline_figure_4, multiline_figure_5, multiline_figure_6, multiline_figure_7, multiline_figure_8, multiline_figure_9, multiline_figure_10

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

def get_figure(value_0, value, value_2, value_3, chart_type):
    if value_0 == 'pascal':
        overestimate = 0
    else:
        overestimate = 5*0.3125

    results = []
    title = ''
    y_title = ''

    figure = go.FigureWidget()

    for i in options[value_0]['arteries']:
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
                stenosis = find_stenosis_ratios(df['min_distances'], df['avg_radius'], side=0, length_percentage=value_3/100)
                results.append({'idx': i, 'line': stenosis})
                title = 'Stenosis ratio by length (min/disprox)'
                y_title = 'Percentage (%)'
            elif chart_type == 3:
                smooth_min = gaussian_filter1d(df['min_distances'], sigma=2)
                smooth_avg = gaussian_filter1d(df['avg_radius'], sigma=2)
                stenosis = find_stenosis_ratios(smooth_min, smooth_avg, side=0, length_percentage=value_3/100)
                results.append({'idx': i, 'line': np.gradient(stenosis)})
                title = '1st derivative stenosis ratio by length (min/disprox)'
                y_title = 'Derivative value'
            elif chart_type == 5:
                results.append({'idx': i, 'line': np.gradient(df['min_distances'])})
                title = '1st derivative of min diameter'
                y_title = 'Derivative value'
            elif chart_type == 6:
                smooth_data = gaussian_filter1d(df['min_distances'], sigma=2)
                results.append({'idx': i, 'line': smooth_data})
                title = 'Min diameter'
                y_title = 'Diameter (mm)'
            elif chart_type == 7:
                smooth_data = gaussian_filter1d(df['avg_radius'], sigma=2)
                results.append({'idx': i, 'line': smooth_data})
                title = 'Avg diameter'
                y_title = 'Diameter (mm)'
            elif chart_type == 8:
                smooth_data = gaussian_filter1d(df['min_distances'], sigma=2)
                results.append({'idx': i, 'line': np.gradient(smooth_data)})
                title = '1st derivative of min diameter'
                y_title = 'Diameter (mm)'
            elif chart_type == 9:
                smooth_min = gaussian_filter1d(df['min_distances'], sigma=2)
                smooth_avg = gaussian_filter1d(df['avg_radius'], sigma=2)
                stenosis = find_stenosis_ratios(smooth_min, smooth_avg, side=0, length_percentage=value_3/100)
                results.append({'idx': i, 'line': stenosis})
                title = 'Stenosis ratio by length (min/disprox) after smoothing'
            elif chart_type == 4:
                if not os.path.isfile(result_dir + f'{str(value)}/smooth_points_{i}.txt'):
                    results.append(None)
                else:
                    smooth_points = np.genfromtxt(result_dir + f'{str(value)}/smooth_points_{i}.txt', delimiter=',')

                    with open(result_dir + f'{str(value)}/smooth_connected_lines_{i}.json', 'r') as file:
                        smooth_connected_lines = json.load(file)

                    points = smooth_points[smooth_connected_lines[0]]
                    num_segments = len(df['avg_radius'])
                    middle_points = find_middle_points(points, num_segments)
        
                    curvature =  gaussian_filter1d(compute_curvature(np.array(middle_points)), sigma=2)
                    results.append({'idx': i, 'line': curvature})
                    title = 'Curvature along the centerline'
                    y_title = 'Curvature'
            else:
                results.append(None)
        else:
            results.append(None)
    
    x_values = [i * 0.5 for i in range(max(len(line) for line in results if line is not None))]
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
        trace = go.Scatter(
                x=x_line_values,
                y=line,
                mode='lines',
                name=f'{str(idx)}.{mapping_names[idx]}',
                line={'color': mapping_colors[idx], 'dash': 'dash'},
                showlegend=False,
                legendgroup=f'{str(idx)}.{mapping_names[idx]}',
            )
        
        traces.append(trace)
        figure.add_trace(trace)
        
        # Modify the interval to solid line
        solid_x_values = x_line_values[interval_start_index:interval_end_index+1]
        solid_y_values = line[interval_start_index:interval_end_index+1]
        
        # Create trace with solid line within the interval
        trace = go.Scatter(
                x=solid_x_values,
                y=solid_y_values,
                mode='lines',
                name=f'{str(idx)}.{mapping_names[idx]}',
                line={'color': mapping_colors[idx], 'dash': 'solid'},
                legendgroup=f'{str(idx)}.{mapping_names[idx]}',
            )

        traces.append(trace)
        figure.add_trace(trace)

        # Marker trace
        trace = go.Scatter(
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
                legendgroup=f'{str(idx)}.{mapping_names[idx]}',
            )

        traces.append(trace)
        figure.add_trace(trace)

    # Create the figure
    figure.layout = go.Layout(
            title=title,
            xaxis={'title': 'Length (mm)'},
            yaxis={'title': y_title},
            showlegend=True,
            margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
            hovermode='closest'
        )
    figure.update_layout(title_x=0.5)

    return figure


def update_table(value_0, value, value_2, value_3):
    if value_0 == 'pascal':
        overestimate = 0
    else:
        overestimate = 5*0.3125

    results = []
    neurologist_df['ID'] = neurologist_df['ID'].astype(str)
    neuro_rows = neurologist_df[neurologist_df['ID'] == str(value)]
    
    if len(neuro_rows) > 0:
        neuro_row = neuro_rows.iloc[0]
    else:
        neuro_row = pd.Series({col: 0 for col in neurologist_df.columns})

    for i in options[value_0]['arteries']:
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

            avg_avg_diameter = df['avg_radius'].mean()
            avg_min_diameter = df['min_distances'].mean()
            metrics['Artery'] = mapping_names[i]
            
            if value_0 != 'pascal':
                metrics[f'Neuro_SL'] = sten
                metrics[f'Neuro_diam'] = diam

            metrics[f'Min_diam'] = round(df['min_distances'][start_idx:end_idx].min()-overestimate, 2)
            metrics[f'Avg_diam'] = round(df['avg_radius'][start_idx:end_idx].mean()-overestimate, 2)

            if value_0 != 'pascal':
                metrics[f'Neuro_SL'] = sten
            
            for key in stenosis_methods:
                method_name = stenosis_methods[key]

                if value_0 == 'pascal':
                    is_peaks = [True]
                else:
                    is_peaks = [True, False]

                for is_peak in is_peaks:
                    is_check = '' if is_peak else 'non-checked'
                    if key == 1: #local_min/avg_global_avg
                        metrics[f'SR-{method_name}'] = round((1-df['min_distances'][start_idx:end_idx]/avg_avg_diameter).max(), 2)
                    elif key == 2: #local_min/avg_global_min
                        metrics[f'SR-{method_name}'] = max(round((1-df['min_distances'][start_idx:end_idx]/avg_min_diameter).max(), 2), 0)
                    elif key == 5: #local_avg/avg_global_avg
                        metrics[f'SR-{method_name}'] = max(round((1-df['avg_radius'][start_idx:end_idx]/avg_avg_diameter).max(), 2), 0)
                    elif key == 3: #local_min/avg_disprox_min
                        metrics[f'SR-{method_name}-({is_check})'] = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['min_distances'], side=0, length_percentage=value_3/100, is_peak=is_peak)
                    elif key == 4: #local_min/avg_disprox_avg
                        metrics[f'SR-{method_name}-({is_check})'] = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['avg_radius'], side=0, length_percentage=value_3/100, is_peak=is_peak)
                    elif key == 6: #local_avg/avg_disprox_avg                        
                        metrics[f'SR-{method_name}-({is_check})'] = find_stenosis_ratio(start_idx, end_idx, df['avg_radius'], df['avg_radius'], side=0, length_percentage=value_3/100, is_peak=is_peak)
                    elif key == 7: #local_min/avg_distal_min                        
                        metrics[f'SR-{method_name}-({is_check})'] = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['min_distances'], side=2, length_percentage=value_3/100, is_peak=is_peak)
                    elif key == 8: #local_min/avg_proximal_min                        
                        metrics[f'SR-{method_name}-({is_check})'] = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['min_distances'], side=1, length_percentage=value_3/100, is_peak=is_peak)
                    elif key == 9: #local_min/avg_distal_avg                        
                        metrics[f'SR-{method_name}-({is_check})'] = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['avg_radius'], side=2, length_percentage=value_3/100, is_peak=is_peak)
                    elif key == 10: #local_min/avg_proximal_avg                        
                        metrics[f'SR-{method_name}-({is_check})'] = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['avg_radius'], side=1, length_percentage=value_3/10, is_peak=is_peak)
                    else:
                        metrics[f'SR-{method_name}'] = 'N/A'
                    
                    if key in [1, 2, 5]:
                        metrics[f'SL-{method_name}'] = ratio_to_level(metrics[f'SR-{method_name}'])
                    else:
                        metrics[f'SL-{method_name}-({is_check})'] = ratio_to_level(metrics[f'SR-{method_name}-({is_check})'])

            results.append(metrics)
    return results

def find_middle_points(points, num_segments):
    num_points_per_segment = int(len(points)/num_segments)
    middle_points = []

    for seg_idx in range(num_segments):
        start_idx = seg_idx * num_points_per_segment
        end_idx = (seg_idx + 1) * num_points_per_segment

        # Ensure the last segment includes any remaining points
        if seg_idx == num_segments - 1:
            end_idx = len(points)

        segment = points[start_idx:end_idx]
        middle_idx = len(segment) // 2

        if len(segment):
            middle_points.append(segment[middle_idx])

    return middle_points

def update_graph(value_0, value, value_2):
    dataset_name = value_0
    sub_num = value

    segment_file_path = options[dataset_name]['dataset_dir'] + f"""/{options[dataset_name]['org_pre_str']}{str(sub_num)}{options[dataset_name]['org_post_str']}"""
    original_file_path = options[dataset_name]['dataset_dir'] + f"""/{options[dataset_name]['seg_pre_str']}{str(sub_num)}{options[dataset_name]['seg_post_str']}"""

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

        main_line = smooth_connected_lines[0]
        line_length = len(main_line)
        start_points.append(smooth_points[main_line[0]])
        end_points.append(smooth_points[main_line[line_length-1]])
        middle_points.append(smooth_points[main_line[int(line_length/2)]])
        cons_points.append(smooth_points[main_line[int(0.25*line_length)]])
        cons_points.append(smooth_points[main_line[int(0.75*line_length)]])

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

        mesh = generate_mesh(vmtk_boundary_vertices, vmtk_boundary_faces, mapping_names[artery_index], mapping_colors[artery_index])
        meshes.append(mesh)

        # if os.path.isfile(info_dir + f'chosen_ring_{artery_index}.json'):
        #     with open(info_dir + f'chosen_ring_{artery_index}.json', 'r') as file:
        #         chosen_ring_vertices = json.load(file)

        #     points = smooth_points[smooth_connected_lines[0]]
        #     num_segments = len(chosen_ring_vertices)
        #     middle_points = find_middle_points(points, num_segments)

        #     middle_points_all.append(points)

    vmtk_boundary_vertices_all = np.concatenate(vmtk_boundary_vertices_all, axis=0)
    vmtk_boundary_faces_all = np.concatenate(vmtk_boundary_faces_all, axis=0)
    # middle_points_all = np.concatenate(middle_points_all, axis=0)

    layout = go.Layout(
        scene=dict(
            aspectmode='manual',
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis =dict(visible=False)
        ),
    )


    visualized_start_points = generate_points_name(np.array(start_points), 10, 'blue', 'start point')
    visualized_end_points = generate_points_name(np.array(end_points), 10, 'red', 'end point')
    visualized_middle_points = generate_points_name(np.array(middle_points), 10, 'orange', 'middle point')
    visualized_cons_points = generate_points_name(np.array(cons_points), 10, 'green', '25th/75th point')
    # visualized_segment_points = generate_points_name(np.array(middle_points_all), 5, 'blue', 'segment point')

    show_points = []
    show_points.append(visualized_start_points)
    show_points.append(visualized_end_points)
    show_points.append(visualized_middle_points)
    show_points.append(visualized_cons_points)
    # show_points.append(visualized_segment_points)

    fig = go.Figure(data=show_points+meshes+line_traces+[go.Scatter3d()], layout=layout)
    fig.update_layout(height=500,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0) )
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
        Output('multiline-graph-9', 'figure' ),
        Output('multiline-graph-10', 'figure' ),
    ],
    [
        Input('dropdown-selection-dataset', 'value'),
        Input('dropdown-selection-subject', 'value'),
        Input('interval-slider', 'value'),
        Input('slider', 'value'),
    ]
)
def update_data(value_0, value_1, value_2, value_3):
    output_1 = update_graph(value_0, value_1, value_2)
    output_2 = update_table(value_0, value_1, value_2, value_3)
    output_3 = get_figure(value_0, value_1, value_2, value_3, 0)
    output_4 = get_figure(value_0, value_1, value_2, value_3, 1)
    output_5 = get_figure(value_0, value_1, value_2, value_3, 2)
    output_6 = get_figure(value_0, value_1, value_2, value_3, 3)
    output_7 = get_figure(value_0, value_1, value_2, value_3, 4)
    output_8 = get_figure(value_0, value_1, value_2, value_3, 5)
    output_9 = get_figure(value_0, value_1, value_2, value_3, 6)
    output_10 = get_figure(value_0, value_1, value_2, value_3, 7)
    output_11 = get_figure(value_0, value_1, value_2, value_3, 8)
    output_12 = get_figure(value_0, value_1, value_2, value_3, 9)

    return output_1, output_2, output_3, output_4, output_7, output_5, output_6, output_8, output_9, output_10, output_11, output_12

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)