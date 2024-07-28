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
import matplotlib.pyplot as plt
import seaborn as sns

dataset_dir = 'E:/pascal/'
result_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/info_files/'
pattern = re.compile(r'^PT_(.*?)_ToF_eICAB_CW\.nii\.gz$')
sub_nums = []
chosen_arteries = [1, 2, 3, 5, 6, 7, 8, 17, 18]

mapping_names = {
    1: 'LCAR',
    2: 'RCAR',
    3: 'BAS',
    5: 'LACA1',
    6: 'RACA1',
    7: 'LMCA1',
    8: 'RMCA1',
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
    # 1: 'local_min/avg_global_avg',
    # 2: '2. local_min/avg_global_min',
    # 3: '3. local_min/avg_disprox_min',
    4: 'local_min/avg_disprox_avg',
    # 5: '5. local_avg/avg_global_avg',
    # 6: '6. local_avg/avg_disprox_avg',
    # 7: '7. local_min/avg_distal_min',
    # 8: '8. local_min/avg_proximal_min',
    # 9: 'local_min/avg_distal_avg',
    # 10: 'local_min/avg_proximal_avg',
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

length_percents = [20, 80]
sub_infos = []

for sub_num in sub_nums:
    sub_id = 'PT_' + sub_num.split('_')[0]
    sub_ses = sub_num.split('_')[1][:2]
    sub_info = {
                'ID': sub_id,
                'session': sub_ses
            }
    for artery_idx in chosen_arteries:
        info_dir = result_dir + f"""{str(sub_num)}/measure_output_{str(artery_idx)}.csv"""
        if os.path.isfile(info_dir):
            df = pd.read_csv(info_dir)
            df = df.round(2)
            start_idx = int(length_percents[0]*len(df['min_distances'])/100)
            end_idx = int(length_percents[1]*len(df['min_distances'])/100)
            avg_avg_diameter = round(df['avg_radius'].mean(), 2)
            avg_min_diameter = df['min_distances'].mean()

            sub_info[f'{mapping_names[artery_idx]}_avg_diameter'] = avg_avg_diameter

            for key in stenosis_methods:
                method_name = stenosis_methods[key]

                ratio = 'N/A'
                if key == 1: #local_min/avg_global_avg
                    ratio = round((1-df['min_distances'][start_idx:end_idx]/avg_avg_diameter).max(), 2)
                elif key == 2: #local_min/avg_global_min
                    ratio = max(round((1-df['min_distances'][start_idx:end_idx]/avg_min_diameter).max(), 2), 0)
                elif key == 3: #local_min/avg_disprox_min
                    ratio = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['min_distances'], side=0, length_percentage=0.1)
                elif key == 4: #local_min/avg_disprox_avg
                    ratio = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['avg_radius'], side=0, length_percentage=0.1)
                elif key == 5: #local_avg/avg_global_avg
                    ratio = max(round((1-df['avg_radius'][start_idx:end_idx]/avg_avg_diameter).max(), 2), 0)
                elif key == 6: #local_avg/avg_disprox_avg                        
                    ratio = find_stenosis_ratio(start_idx, end_idx, df['avg_radius'], df['avg_radius'], side=0, length_percentage=0.1)
                elif key == 7: #local_min/avg_distal_min                        
                    ratio = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['min_distances'], side=2, length_percentage=0.1)
                elif key == 8: #local_min/avg_proximal_min                        
                    ratio = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['min_distances'], side=1, length_percentage=0.1)
                elif key == 9: #local_min/avg_distal_avg                        
                    ratio = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['avg_radius'], side=2, length_percentage=0.1)
                elif key == 10: #local_min/avg_proximal_avg                        
                    ratio = find_stenosis_ratio(start_idx, end_idx, df['min_distances'], df['avg_radius'], side=1, length_percentage=0.1)
                else:
                    ratio = 'N/A'

                if ratio != 'N/A':
                    level = ratio_to_level_1(ratio)
                    sub_info[f'{mapping_names[artery_idx]}_stenosis_ratio'] = ratio
                    sub_info[f'{mapping_names[artery_idx]}_stenosis_level']= level
            
    sub_infos.append(sub_info)

sub_df = pd.DataFrame(sub_infos)

#Read pascal info
pascal_dir = 'C:/Users/nguc4116/Downloads/Analyses_psycho_pilot_PhD_VF.xlsx'
cols = ['ID', 'Groupe', 'Age', 'Sexe']
sheets = ['Donnes_psycho_V1', 'Donnes_psycho_V2', 'Donnes_psycho_V3']
# Read each sheet into a separate DataFrame
dfs = []
for sheet in sheets:
    df = pd.read_excel(pascal_dir, sheet_name=sheet, usecols=cols, engine='openpyxl')
    df['session'] = sheet[-2:]  # Extract session identifier from sheet name
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)[:-4]
joined_df = pd.merge(merged_df, sub_df, on=['ID', 'session'])
# Convert specific columns to integer
cols_to_convert = ['Groupe', 'Age', 'Sexe']
joined_df[cols_to_convert] = joined_df[cols_to_convert].astype(int)

csv_filename = f'C:/Users/nguc4116/Desktop/pascal/combined_CNPB_VASC_{str(length_percents[0])}_{str(length_percents[1])}.csv'
joined_df.to_csv(csv_filename, index=False)

print(joined_df)
