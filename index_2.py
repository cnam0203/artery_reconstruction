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

options = [
    {
        'dataset_dir': 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/tof_mra_julia/',
        'pattern': re.compile(r'sub-(\d+)_run-1_mra_eICAB_CW.nii.gz'),
        'arteries': [1, 2, 3],
        'is_replace': True,
        'org_pre_str': 'sub-',
        'org_post_str': '_run-1_mra_eICAB_CW.nii.gz',
        'seg_pre_str': 'sub-',
        'seg_post_str': '_run-1_mra_resampled.nii.gz',
    }, {
        'dataset_dir': 'E:/pascal/',
        'pattern': re.compile(r'^PT_(.*?)_ToF_eICAB_CW\.nii\.gz$'),
        'arteries': [17, 18],
        'is_replace': False,
        'org_pre_str': 'PT_',
        'org_post_str': '_ToF_eICAB_CW.nii.gz',
        'seg_pre_str': 'PT_',
        'seg_post_str': '_ToF_resampled.nii.gz',
    }, {
        'dataset_dir': 'E:/stenosis/',
        'pattern': re.compile(r'sub-(\d+)_run-1_mra_eICAB_CW.nii.gz'),
        'arteries': [1, 2, 3],
        'is_replace': True,
        'org_pre_str': 'sub-',
        'org_post_str': '_run-1_mra_eICAB_CW.nii.gz',
        'seg_pre_str': 'sub-',
        'seg_post_str': '_run-1_mra_resampled.nii.gz',
    }
]

option_idx = 2
option = options[option_idx]
dataset_dir = option['dataset_dir']
pattern = option['pattern']
chosen_arteries = option['arteries']
is_replace = option['is_replace']
org_pre_str = option['org_pre_str']
org_post_str = option['org_post_str']
seg_pre_str = option['seg_pre_str']
seg_post_str = option['seg_post_str']
sub_nums = []
result_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/info_files/'

neurologist_df = pd.read_csv('C:/Users/nguc4116/Desktop/artery_reconstruction/stenosis.csv', sep=',')
neurologist_df['ID'] = neurologist_df['ID'].astype(str)
checked_df = pd.read_csv('C:/Users/nguc4116/Desktop/checked_stenosis.csv', sep=',')
checked_df['ID'] = checked_df['ID'].astype(str)

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

stenosis_methods = {
    # 6: '1. local_avg/disprox_avg',
    # 11: '2. local_avg/distal_avg',
    # 12: '3. local_avg/proximal_avg',
    # 5: '4. local_avg/global_avg',
    1: 'A. local_min/global_avg',
    # 2: '6. local_min/global_min',
    # 3: '7. local_min/disprox_min',
    # 7: '8. local_min/distal_min',
    # 8: '9. local_min/proximal_min',
    4: 'B. local_min/disprox_avg',
    9: 'C. local_min/distal_avg',
    10: 'D. local_min/proximal_avg',
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

is_peak = True
# Iterate over the files in the directory
for filename in os.listdir(dataset_dir):
    match = pattern.match(filename)
    if match:
        index = match.group(1)

        if option_idx == 2:
            checked_rows = checked_df[checked_df['ID'] == str(index)]
        
            if len(checked_rows) > 0:
                checked_row = checked_rows.iloc[0]
                if checked_row['valid'] in [1, 4, '1', '4']:
                    sub_nums.append(index)
                else:
                    continue

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
    html.H1(children='Artery stenosis report', style={'textAlign':'center'}),
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
    html.Div(id='graphs', style={'display': 'flex', 'flexWrap': 'wrap'}),
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
            'avg_diam': [],
            'min_diam_5vx': [],
            'avg_diam_5vx': []
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
                
                error_df[mapping_names[art_idx]]['min_diam'].append(combined_info[sub_num][art_idx]['neuro_diam'] - df['min_distances'][start_idx:end_idx].min())
                error_df[mapping_names[art_idx]]['avg_diam'].append(combined_info[sub_num][art_idx]['neuro_diam'] - round(df['avg_radius'][start_idx:end_idx].mean(), 2))
                error_df[mapping_names[art_idx]]['min_diam_5vx'].append(combined_info[sub_num][art_idx]['neuro_diam'] - round(df['min_distances'][start_idx:end_idx].min() - 5*0.375))
                error_df[mapping_names[art_idx]]['avg_diam_5vx'].append(combined_info[sub_num][art_idx]['neuro_diam'] - round(df['avg_radius'][start_idx:end_idx].mean() - 5*0.375, 2))

                results = {}
                for key in stenosis_methods:
                    method_name = stenosis_methods[key]
                    result = {}


                    if key == 1: #local_min/avg_global_avg
                        result['ratio'] = round((1-df['min_distances'][start_idx:end_idx]/avg_avg_diameter).max(), 2)
                    elif key == 2: #local_min/avg_global_min
                        result['ratio'] = max(round((1-df['min_distances'][start_idx:end_idx]/avg_min_diameter).max(), 2), 0)
                    elif key == 3: #local_min/avg_disprox_min
                        result['ratio'] = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['min_distances'], side=0, length_percentage=0.1, is_peak=is_peak)
                    elif key == 4: #local_min/avg_disprox_avg
                        result['ratio'] = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['avg_radius'], side=0, length_percentage=0.1, is_peak=is_peak)
                    elif key == 5: #local_avg/avg_global_avg
                        result['ratio'] = max(round((1-df['avg_radius'][start_idx:end_idx]/avg_avg_diameter).max(), 2), 0)
                    elif key == 6: #local_avg/avg_disprox_avg                        
                        result['ratio'] = find_stenosis_ratio(start_idx, end_idx, df['avg_radius'], df['avg_radius'], side=0, length_percentage=0.1, is_peak=is_peak)
                    elif key == 7: #local_min/avg_distal_min                        
                        result['ratio'] = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['min_distances'], side=2, length_percentage=0.1, is_peak=is_peak)
                    elif key == 8: #local_min/avg_proximal_min                        
                        result['ratio'] = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['min_distances'], side=1, length_percentage=0.1, is_peak=is_peak)
                    elif key == 9: #local_min/avg_distal_avg                        
                        result['ratio'] = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['avg_radius'], side=2, length_percentage=0.1, is_peak=is_peak)
                    elif key == 10: #local_min/avg_proximal_avg                        
                        result['ratio'] = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['avg_radius'], side=1, length_percentage=0.1, is_peak=is_peak)
                    elif key == 11: #local_avg/avg_distal_avg                        
                        result['ratio'] = find_stenosis_ratio(start_idx, end_idx, df['avg_radius'], df['avg_radius'], side=2, length_percentage=0.1, is_peak=is_peak)
                    elif key == 12: #local_avg/avg_proximal_avg                        
                        result['ratio'] = find_stenosis_ratio(start_idx, end_idx, df['avg_radius'], df['avg_radius'], side=1, length_percentage=0.1, is_peak=is_peak)
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

    # Extract keys and values from the data structure
    artery_names = list(error_df.keys())
    min_distances_values = [error_df[artery]['min_diam'] for artery in artery_names]
    avg_distances_values = [error_df[artery]['avg_diam'] for artery in artery_names]
    min_distances_5vx_values = [error_df[artery]['min_diam_5vx'] for artery in artery_names]
    avg_distances_5vx_values = [error_df[artery]['avg_diam_5vx'] for artery in artery_names]
    artery_names_qty = [f"{artery}_({str(len(error_df[artery]['min_diam']))})" for artery in artery_names]


    # Extract keys and values from the data structure
    artery_names = list(error_df.keys())

    # Prepare data for the violin plots
    min_distances = []
    avg_distances = []
    arteries_min = []
    arteries_avg = []
    min_5vx_distances = []
    avg_5vx_distances = []
    arteries_min_5vx = []
    arteries_avg_5vx = []

    for artery in artery_names:
        min_distances.extend(error_df[artery]['min_diam'])
        avg_distances.extend(error_df[artery]['avg_diam'])
        min_5vx_distances.extend(error_df[artery]['min_diam_5vx'])
        avg_5vx_distances.extend(error_df[artery]['avg_diam_5vx'])
        arteries_min.extend([f"{artery}_({len(error_df[artery]['min_diam'])})"] * len(error_df[artery]['min_diam']))
        arteries_avg.extend([f"{artery}_({len(error_df[artery]['avg_diam'])})"] * len(error_df[artery]['avg_diam']))
        arteries_min_5vx.extend([f"{artery}_({len(error_df[artery]['min_diam_5vx'])})"] * len(error_df[artery]['min_diam_5vx']))
        arteries_avg_5vx.extend([f"{artery}_({len(error_df[artery]['avg_diam_5vx'])})"] * len(error_df[artery]['avg_diam_5vx']))

    # Create the figure
    fig = go.Figure()

    # Add traces for min_distances and avg_distances
    fig.add_trace(go.Violin(
        x=arteries_min,
        y=min_distances,
        name='min_distances',
        box_visible=True,
        meanline_visible=True
    ))

    fig.add_trace(go.Violin(
        x=arteries_avg,
        y=avg_distances,
        name='avg_distances',
        box_visible=True,
        meanline_visible=True
    ))

    # Add traces for min_distances and avg_distances
    fig.add_trace(go.Violin(
        x=arteries_min_5vx,
        y=min_5vx_distances,
        name='min_distances_5vx',
        box_visible=True,
        meanline_visible=True
    ))

    fig.add_trace(go.Violin(
        x=arteries_avg_5vx,
        y=avg_5vx_distances,
        name='avg_distances_5vx',
        box_visible=True,
        meanline_visible=True
    ))
    

    # Update layout
    fig.update_layout(
        title_x=0.5,
        xaxis=dict(title='Arteries'),
        yaxis=dict(title='Diameter error (mm)'),
        title="Comparison of Diameter error by Arteries (Neuro's diameter - Our diameter)",
        violinmode='group'
    )

    figures.append(dcc.Graph(figure=fig, style={'width': '100%'}))
    # Group by Metrics and calculate accuracy for each Method
    for metric, group in combined_df.groupby('Metrics'):
        metric_accuracy = []
        total_samples = 0
        for method in group['Methods'].unique():
            df_method = group[group['Methods'] == method]
            concatenated_wrong_points = sum(df_method['Wrong_Points'], [])
            accuracy = 100 * df_method['True_Prediction'].sum() / df_method['Total_Samples'].sum()
            total_samples = df_method['Total_Samples'].sum()
            metric_accuracy.append({'Method': method, 'Accuracy': round(accuracy, 2), 'Wrong_Points': concatenated_wrong_points})

        # Create a bar chart for each metric
        df_metric_accuracy = pd.DataFrame(metric_accuracy)
        fig = px.bar(
            df_metric_accuracy,
            x='Method',
            y='Accuracy',
            text='Accuracy',
            title=f'Accuracy of {metric} prediction by methods ({total_samples} samples)',
            labels={'Method': 'Methods', 'Accuracy': 'Accuracy (%)'}
        )
        fig.update_layout(title_x=0.5)
        fig.update_yaxes(range=[0, 100])
        figures.append(dcc.Graph(figure=fig, style={'width': '50%'}))


        # Prepare data for Plotly
        fig = go.Figure()
        for method_data in metric_accuracy:
            method = method_data['Method']
            wrong_points = method_data['Wrong_Points']
            
            fig.add_trace(go.Violin(
                x=[method] * len(wrong_points),
                y=wrong_points,
                name=method,
                box_visible=True,  # Show box plot inside the violin
                meanline_visible=True,
                showlegend=False,
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
            xaxis_title='Methods',
            yaxis_title='Stenosis percentage (%)',
            yaxis=dict(range=[-10, 140]), 
            title=dict(
                text=f'Distribution of incorrect prediction ({metric}) by methods ({total_samples} samples)',
                x=0.5,  # Centers the title
                xanchor='center'
            )
        )
        figures.append(dcc.Graph(figure=fig, style={'width': '50%'}))



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
        fig.update_yaxes(range=[0, 100])
        figures.append(dcc.Graph(figure=fig, style={'width': '50%'}))

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
        fig.update_yaxes(range=[0, 100])
        figures.append(dcc.Graph(figure=fig, style={'width': '50%'}))

    return figures

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8052)