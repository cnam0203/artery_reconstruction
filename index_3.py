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
import matplotlib.pyplot as plt
import seaborn as sns

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

stenosis_methods = {
    1: 'local_min/avg_global_avg',
    # 2: '2. local_min/avg_global_min',
    # 3: '3. local_min/avg_disprox_min',
    4: 'local_min/avg_disprox_avg',
    # 5: '5. local_avg/avg_global_avg',
    # 6: '6. local_avg/avg_disprox_avg',
    # 7: '7. local_min/avg_distal_min',
    # 8: '8. local_min/avg_proximal_min',
    9: 'local_min/avg_distal_avg',
    10: 'local_min/avg_proximal_avg',
}


app = Dash()

app.layout =  html.Div([
    html.H1(children='Chronic Pain vs Artery measurement', style={'textAlign':'center'}),
    html.Div([
        html.Label('Choose an artery:', style={'width': '100px'}),
        dcc.Dropdown(options=[{'label': mapping_names[key], 'value': mapping_names[key]} for key in mapping_names],
            value=mapping_names[1], id='dropdown-selection-artery', style={'width': '25%', 'display': 'inline-block'})
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.Div(id='graphs'),
])

csv_filename = f'C:/Users/nguc4116/Desktop/pascal/combined_CNPB_VASC_20_80.csv'
joined_df = pd.read_csv(csv_filename, sep=',')
session_names = joined_df['session'].unique()
groups = joined_df['Groupe'].unique()
para_cols = ['diameter']
for key in stenosis_methods:
    method_name = stenosis_methods[key]
    para_cols.append(f'stenosis_ratio_({method_name})')
    para_cols.append(f'stenosis_level_({method_name})')

color_discrete_map = {
    0: '#ADD8E6',  # Light Blue
    1: '#87CEEB',  # Sky Blue
    2: '#4682B4',  # Steel Blue
    3: '#0000FF',  # Blue
    4: '#00008B'   # Dark Blue
}

@callback(
    Output('graphs', 'children'),
    [Input('dropdown-selection-artery', 'value')]
)
def update_charts(value):
    figures = []
    filtered_df = joined_df[(joined_df['artery'] == value)]
    
    for col in para_cols:
        y_title = col.replace('_', ' ')
        for session in session_names:
            info_df = filtered_df[(filtered_df['session'] == session)]
            info_df = info_df.sort_values(by=['Groupe', col])

            if 'stenosis_level' in col:
                count_df = info_df.groupby(['Groupe', col]).size().reset_index(name='Count')
                count_df[col] = count_df[col].astype(str)

                # Create the bar chart
                fig = px.bar(
                    count_df,
                    x='Groupe',
                    y='Count',
                    color=col,
                    barmode='stack',
                    text='Count',
                )
                fig.update_layout(legend_title_text='stenosis level')
            else:

                fig = px.violin(info_df, y=col, x='Groupe', color='Groupe', box=True)
            fig.update_layout(title_text=f'Artery {y_title} in visit {session[1]}', title=dict(
                x=0.5,  # Centers the title
                xanchor='center'
            ),),
            figures.append(fig)
    
    for col in para_cols:
        y_title = col.replace('_', ' ')
        for session in session_names:
            info_df = filtered_df[(filtered_df['session'] == session)]
            info_df = info_df.sort_values(by='Groupe')

            if 'stenosis_level' in col:
                count_df = info_df.groupby(['Groupe', 'Sexe', col]).size().reset_index(name='Count')
                count_df[col] = count_df[col].astype(str)
                print(count_df)
                # Create the bar chart
                fig = px.bar(
                    count_df,
                    x='Groupe',
                    y='Count',
                    color=col,
                    barmode='stack',
                    text='Count',
                    facet_col='Sexe',  # Facet by Level
                )
                fig.update_layout(legend_title_text='stenosis level')
            else:
                fig = px.violin(info_df, y=col, x='Groupe', color='Sexe', box=True)
            fig.update_layout(title_text=f'Artery {y_title} by sex in visit {session[1]}', title=dict(
                x=0.5,  # Centers the title
                xanchor='center'
            ),),
            figures.append(fig)

    color_map = {
        'Healthy': 'blue',
        'Chronic pain': 'red',
    }
    for col in para_cols:
        y_title = col.replace('_', ' ')
        for group in groups:
            for session in session_names:
                info_df = filtered_df[(filtered_df['session'] == session) & (filtered_df['Groupe'] == group)]
                if 'stenosis_level' in col:
                    info_df[col] = info_df[col].astype(str)  
                    info_df = info_df.sort_values(by=col)
                    fig = px.violin(info_df, y='Age', x=col, color=col, box=True)
                    fig.update_layout(legend_title_text='stenosis level', xaxis_title='stenosis level')
                else:
                    x_val = 'Age'
                    y_val = col
                    range_y=[0, info_df[col].max() * 1.1]
                    
                    fig = px.scatter(
                        info_df,
                        x=x_val,
                        y=y_val,
                        color_discrete_sequence=[color_map[group]],  # Color points by 'Groupe'
                        # labels={'Age': 'Age', col: y_title},  # Set axis labels
                        range_y=range_y  # Adjust y-axis range if needed
                    )
                fig.update_layout(title_text=f'Artery {y_title} by age of {group} group in visit {session[1]}', title=dict(
                x=0.5,  # Centers the title
                xanchor='center'
            ),)  # Center the title
                figures.append(fig)

    # Arrange figures into rows of 3
    rows = []
    for i in range(0, len(figures), 3):
        row = html.Div(
            children=[
                dcc.Graph(figure=figures[j]) for j in range(i, min(i + 3, len(figures)))
            ],
            style={'display': 'flex', 'justify-content': 'space-around'}
        )
        rows.append(row)

    return rows

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8051)