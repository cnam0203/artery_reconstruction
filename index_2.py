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
import itertools

def max_stable_mask(a, pos, ratio_threshold, distance, interval_size, dif_thresh): # thresh controls noise
    pass_steps = int(distance/interval_size)
    
    left_mask = None
    right_mask = None
    left_radius = None
    right_radius = None
    mean_radius = None
    #Left
    if pos - 2*pass_steps >= 0:
        left_arr = a[pos - 2*pass_steps : pos - pass_steps]
        left_values, left_mask = side_stable_mask(left_arr, ratio_threshold, dif_thresh)
        left_radius = np.mean(left_values[left_mask == 1])

    if pos + 2*pass_steps < a.shape[0]:
        right_arr = a[pos + pass_steps : pos + 2*pass_steps]
        right_values, right_mask = side_stable_mask(right_arr, ratio_threshold, dif_thresh)
        right_radius = np.mean(right_values[right_mask == 1])
    
    if left_radius == None and right_radius != None:
        mean_radius = right_radius
    elif left_radius != None and right_radius == None:
        mean_radius = left_radius
    elif left_radius != None and right_radius != None:
        mean_radius = (left_radius + right_radius)/2
    
    return mean_radius

def side_stable_mask(arr, ratio_threshold, dif_thresh):
    a = np.copy(arr)
    is_end = False
    loop_count = 0
    while not is_end and loop_count <= 50:
        loop_count += 1
        max_value = round(a.max(), 1)
        mask = np.r_[ False, np.abs(a - max_value) < dif_thresh, False]
        idx = np.flatnonzero(mask[1:] != mask[:-1])
        s0 = (idx[1::2] - idx[::2]).argmax()
        valid_mask = np.zeros(a.size, dtype=int) #Use dtype=bool for mask o/p
        valid_mask[idx[2*s0]:idx[2*s0+1]] = 1

        if np.argwhere(valid_mask == 1).shape[0] >= ratio_threshold*a.shape[0]:
            is_end = True
        else:
            second_max_value = round(np.max(a[a != max_value]), 1)
            
            if second_max_value >= max_value - dif_thresh:
                a[a == max_value] = second_max_value
            else:
                a[a == max_value] = 0
            
    return a, valid_mask

def check_truth(level, ratio, case):
    if case == -3 and level == 0:
        return -1
    elif case == -3 and level != 0:
        if ratio >= 0.15:
            return 1
        else:
            return 0
    elif case == -2 and level != 0:
        return -1
    else:
        is_correct = 0
        
        if level == 0:
            if ratio <= 0.25:
                is_correct = 1
            else:
                is_correct = 0
        elif level == 1:
            if ratio >= 0.15 and ratio <= 0.55:
                is_correct = 1
            else:
                is_correct = 0
        elif level == 2:
            if ratio >= 0.45 and ratio <= 0.75:
                is_correct = 1
            else:
                is_correct = 0
        elif level == 3:
            if ratio >= 0.65:
                is_correct = 1
            else:
                is_correct = 0
        else:
            if ratio >= 0.95:
                is_correct = 1
            else:
                is_correct = 0
        
        if case in [-3, -1, -2] or level == case:
            return is_correct
        else:
            return -1

dataset_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/tof_mra_julia/'
result_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/info_files/'

pattern = re.compile(r'sub-(\d+)_run-1_mra_eICAB_CW.nii.gz')
sub_nums = []
chosen_arteries = [1, 2, 3]
neurologist_df = pd.read_csv('C:/Users/nguc4116/Desktop/artery_reconstruction/stenosis.csv', sep=',')
neurologist_df['ID'] = neurologist_df['ID'].astype(str)

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

stenosis_methods = {
    1: '1. local_min/avg_global_avg',
    2: '2. local_min/avg_global_min',
    3: '3. local_min/avg_distal_min',
    4: '4. local_min/avg_distal_avg',
    5: '5. local_avg/avg_global_avg',
    6: '6. local_avg/avg_distal_avg',
}

metrics = {
    '-3': 'stenosis',
    '-2': 'non-stenosis',
    '-1': 'stenosis and non-stenosis',
    '0': 'stenosis level-0',
    '1': 'stenosis level-1',
    '2': 'stenosis level-2',
    '3': 'stenosis level-3',
    '4': 'stenosis level-4'
}


# Iterate over the files in the directory
for filename in os.listdir(dataset_dir):
    match = pattern.match(filename)
    if match:
        index = match.group(1)
        sub_nums.append(index)

combined_info = {}
metrics_values = [metrics[key] for key in metrics] 
arteries_values = [mapping_names[key] for key in chosen_arteries] 
methods_values = [stenosis_methods[key] for key in stenosis_methods] 
combinations = list(itertools.product(metrics_values, arteries_values, methods_values))

for sub_num in sub_nums:
    combined_info[sub_num] = {}

    #Read Julia info
    neuro_rows = neurologist_df[neurologist_df['ID'] == str(sub_num)]
    
    if len(neuro_rows) > 0:
        neuro_row = neuro_rows.iloc[0]
    else:
        neuro_row = pd.Series({col: 0 for col in neurologist_df.columns})

    for art_idx in chosen_arteries:
        diam_col_name = mapping_names[art_idx] + '_diam'
        sten_col_name = mapping_names[art_idx] + '_stenosis'

        neuro_diam = float(neuro_row.get(diam_col_name.lower(), 0))
        neuro_sten = int(neuro_row.get(sten_col_name.lower(), 0))

        combined_info[sub_num][art_idx] = {
            'neuro_diam': neuro_diam,
            'neuro_sten': neuro_sten,
            'min_diam': 0,
            'avg_diam': 0,
            'sten_results': {}
        }

app = Dash()

app.layout =  html.Div([
    html.H1(children='Artery stenosis map', style={'textAlign':'center'}),
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
    html.Div(id='graphs'),
])


@callback(
    Output('graphs', 'children'),
    [Input('interval-slider', 'value')]
)
def update_charts(len_percents):
    combined_df = pd.DataFrame(combinations, columns=['Metrics', 'Artery', 'Methods'])
    combined_df['True_Prediction'] = 0
    combined_df['Total_Samples'] = 0
    combined_df['Wrong_Points'] = [[] for _ in range(len(combined_df))]
    
    error_df = {}
    for key in chosen_arteries:
        error_df[mapping_names[key]] = {
            'min_diam': [],
            'avg_diam': []
        }

    

    for sub_num in sub_nums:
        for art_idx in chosen_arteries:     
            #Read our measurements
            info_dir = result_dir + f"""{str(sub_num)}/measure_output_{str(art_idx)}.csv"""
            if os.path.isfile(info_dir):
                df = pd.read_csv(info_dir)
                start_idx = int(len_percents[0]*len(df['min_distances'])/100)
                end_idx = int(len_percents[1]*len(df['min_distances'])/100)
                avg_avg_diameter = df['avg_radius'].mean()
                avg_min_diameter = df['min_distances'].mean()

                combined_info[sub_num][art_idx]['min_diam'] = df['min_distances'][start_idx:end_idx].min()
                combined_info[sub_num][art_idx]['avg_diam'] = round(df['avg_radius'][start_idx:end_idx].mean(), 2)
                
                error_df[mapping_names[art_idx]]['min_diam'].append(abs(combined_info[sub_num][art_idx]['neuro_diam'] - df['min_distances'][start_idx:end_idx].min()))
                error_df[mapping_names[art_idx]]['avg_diam'].append(abs(combined_info[sub_num][art_idx]['neuro_diam'] - round(df['avg_radius'][start_idx:end_idx].mean(), 2)))

                results = {}
                for key in stenosis_methods:
                    method_name = stenosis_methods[key]
                    result = {}

                    if key == 1:
                        result['ratio'] = round((1-df['min_distances'][start_idx:end_idx]/avg_avg_diameter).max(), 2)
                    elif key == 2:
                        result['ratio'] = max(round((1-df['min_distances'][start_idx:end_idx]/avg_min_diameter).max(), 2), 0)
                    elif key == 3:
                        result['ratio'] = round((df['stenosis_ratio_min'][start_idx:end_idx]).max(), 2)
                    elif key == 4:
                        ref_min_distances = []
                        ref_avg_distances = []
                        avg_distances = df['avg_radius']
                        ratio_threshold = 0.1
                        distance_threshold = 0.5
                        distance = (1/10)*len(df['avg_radius'])*distance_threshold
                        interval_size = distance_threshold

                        for i in range(len(df['min_distances'])):
                            dif_thresh = 0.5*avg_distances[i]

                            avg_distance = max_stable_mask(np.array(avg_distances), i, ratio_threshold, distance, interval_size, dif_thresh)                        
                            ref_avg_distances.append(avg_distance)
                        
                        is_stop = False
                        while not is_stop:
                            is_stop = True
                            for idx, ring in enumerate(avg_distances):
                                neighbor_avg_distances = []

                                if ref_avg_distances[idx] == None:
                                    if idx > 0 and ref_avg_distances[idx-1] != None:
                                        neighbor_avg_distances.append(ref_avg_distances[idx-1])
                                    if idx < (len(avg_distances) - 1) and ref_avg_distances[idx+1] != None:
                                        neighbor_avg_distances.append(ref_avg_distances[idx+1])

                                    if len(neighbor_avg_distances):
                                        ref_avg_distances[idx] = np.mean(np.array(neighbor_avg_distances))

                            undefined_ranges = [distance for distance in ref_avg_distances if distance == 0 or distance is None]
                            if len(undefined_ranges):
                                is_stop = False

                        stenosis_ratio_avg = np.array(df['min_distances'])/np.array(ref_avg_distances)
                        result['ratio'] = max(round((1-stenosis_ratio_avg[start_idx:end_idx]).max(), 2), 0)
                    elif key == 5:
                        result['ratio'] = max(round((1-df['avg_radius'][start_idx:end_idx]/avg_avg_diameter).max(), 2), 0)
                    elif key == 6:
                        result['ratio'] = round((df['stenosis_ratio_avg'][start_idx:end_idx]).max(), 2)
                    else:
                        result['ratio'] = 'N/A'

                    for i in metrics:
                        if result['ratio'] != 'N/A':
                            predict_result = check_truth(combined_info[sub_num][art_idx]['neuro_sten'], result['ratio'], int(i))

                            if predict_result == 1:
                                combined_df.loc[(combined_df['Metrics'] == metrics[i]) & (combined_df['Artery'] == mapping_names[art_idx]) & (combined_df['Methods'] == stenosis_methods[key]), 'True_Prediction'] += 1
                                combined_df.loc[(combined_df['Metrics'] == metrics[i]) & (combined_df['Artery'] == mapping_names[art_idx]) & (combined_df['Methods'] == stenosis_methods[key]), 'Total_Samples'] += 1
                            elif predict_result == 0:
                                combined_df.loc[(combined_df['Metrics'] == metrics[i]) & (combined_df['Artery'] == mapping_names[art_idx]) & (combined_df['Methods'] == stenosis_methods[key]), 'Total_Samples'] += 1
                                # Append a new element to the 'Wrong_Points' column where the 'Methods' column matches the key
                                matching_indices = combined_df.index[(combined_df['Metrics'] == metrics[i]) & (combined_df['Artery'] == mapping_names[art_idx]) & (combined_df['Methods'] == stenosis_methods[key])].tolist()

                                for index in matching_indices:
                                    combined_df.at[index, 'Wrong_Points'].append(result['ratio']*100)

                    results[key] = result

                combined_info[sub_num][art_idx]['sten_results'] = results

    figures = []


    # for art_idx in chosen_arteries:     
    #     error_df[mapping_names[art_idx]]['min_diam'] = np.array(error_df[mapping_names[art_idx]]['min_diam']).mean()
    #     error_df[mapping_names[art_idx]]['avg_diam'] = np.array(error_df[mapping_names[art_idx]]['avg_diam']).mean()

    # Extract keys and values from the data structure
    artery_names = list(error_df.keys())
    min_distances_values = [np.array(error_df[artery]['min_diam']).mean() for artery in artery_names]
    avg_distances_values = [np.array(error_df[artery]['avg_diam']).mean() for artery in artery_names]
    artery_names_qty = [f"{artery}_({str(len(error_df[artery]['min_diam']))})" for artery in artery_names]

    # Create the figure
    fig = go.Figure()

    # Add traces for min_distances and avg_distances
    fig.add_trace(go.Bar(
        x=artery_names_qty,
        y=min_distances_values,
        name='min_distances'
    ))

    fig.add_trace(go.Bar(
        x=artery_names_qty,
        y=avg_distances_values,
        name='avg_distances'
    ))

    # Update layout
    fig.update_layout(
        barmode='group',
        xaxis=dict(title='Arteries'),
        yaxis=dict(title='Diameter error (mm)'),
        title='Comparison of Diameter error by Arteries'
    )

    figures.append(dcc.Graph(figure=fig))
    # Group by Metrics and calculate accuracy for each Method
    for metric, group in combined_df.groupby('Metrics'):
        metric_accuracy = []
        total_samples = 0
        for method in group['Methods'].unique():
            df_method = group[group['Methods'] == method]
            concatenated_wrong_points = sum(df_method['Wrong_Points'], [])
            accuracy = 100 * df_method['True_Prediction'].sum() / df_method['Total_Samples'].sum()
            total_samples = df_method['Total_Samples'].sum()
            metric_accuracy.append({'Method': method, 'Accuracy': accuracy, 'Wrong_Points': concatenated_wrong_points})

        print(metric_accuracy)
        # Create a bar chart for each metric
        df_metric_accuracy = pd.DataFrame(metric_accuracy)
        fig = px.bar(
            df_metric_accuracy,
            x='Method',
            y='Accuracy',
            text='Accuracy',
            title=f'Accuracy of {metric} prediction by methods ({total_samples} arteries)',
            labels={'Method': 'Methods', 'Accuracy': 'Accuracy (%)'}
        )
        fig.update_layout(title_x=0.5)
        figures.append(dcc.Graph(figure=fig))


        # Prepare data for Plotly
        fig = go.Figure()
        for method_data in metric_accuracy:
            method = method_data['Method']
            wrong_points = method_data['Wrong_Points']
            
            fig.add_trace(go.Scatter(
                x=[method] * len(wrong_points),
                y=wrong_points,
                mode='markers',
                name=method,
                marker=dict(symbol='circle', size=10)
            ))

        if metric == 'stenosis':
            y0 = 20
            y1 = 100
        elif metric == 'non-stenosis':
            y0 = 0
            y1 = 20
        elif metric == 'stenosis level-0':
            y0 = 0
            y1 = 20
        elif metric == 'stenosis level-1':
            y0 = 20
            y1 = 50
        elif metric == 'stenosis level-2':
            y0 = 50
            y1 = 70
        elif metric == 'stenosis level-3':
            y0 = 70
            y1 = 99
        elif metric == 'stenosis level-4':
            y0 = 99
            y1 = 100
        else:
            y0 = 0
            y1 = 0

        fig.add_shape(type="rect",
                  x0=-0.5, y0=y0, x1=len(group['Methods'].unique())-0.5, y1=y1,
                  line=dict(color="RoyalBlue", width=0),
                  fillcolor="LightSkyBlue", opacity=0.3)
        
        # Update layout for grouped bar chart
        fig.update_layout(
            barmode='group',
            title=f'Wrong points in {metric} prediction by methods ({total_samples} arteries)',
            xaxis_title='Methods',
            yaxis_title='Stenosis percentage (%)',
            yaxis=dict(range=[0, 100])
        )
        figures.append(dcc.Graph(figure=fig))



    # Group by Metrics, Methods, and Arteries to calculate accuracy
    figures_data = []
    for metric, group in combined_df.groupby(['Metrics', 'Methods', 'Artery']):
        accuracy = round(100 * group['True_Prediction'].sum() / group['Total_Samples'].sum(), 2)
        figures_data.append({
            'metric': metric[0],
            'method': metric[1],
            'artery': metric[2],
            'accuracy': accuracy,
            'total_samples': group['Total_Samples'].sum(),  # Total samples for the current group
            'artery_num': f"""{metric[2]}_({group['Total_Samples'].sum()})"""
        })

    # Create bar charts for each metric
    for metric, metric_data in pd.DataFrame(figures_data).groupby('metric'):
        fig = px.bar(
            metric_data,
            x='method',
            y='accuracy',
            color='artery_num',
            barmode='group',
            text='accuracy',
            title=f'Accuracy of {metric} prediction by Methods and Arteries',
            labels={'method': 'Methods', 'accuracy': 'Accuracy (%)'}
        )
        fig.update_layout(title_x=0.5)
        figures.append(dcc.Graph(figure=fig))

    # Create bar charts for each metric
    for metric, metric_data in pd.DataFrame(figures_data).groupby('metric'):
        fig = px.bar(
            metric_data,
            x='artery_num',
            y='accuracy',
            color='method',
            barmode='group',
            text='accuracy',
            title=f'Accuracy of {metric} prediction by Methods and Arteries',
            labels={'artery_num': 'Arteries', 'accuracy': 'Accuracy (%)'}
        )
    
        fig.update_layout(title_x=0.5)
        figures.append(dcc.Graph(figure=fig))

    return figures

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8051)