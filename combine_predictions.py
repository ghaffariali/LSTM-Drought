# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 08:23:59 2024

@author: ghaff

this temporary script combines results for different cells into one result. I
had to split the runs for different reasons.
"""
# wd = r'C:\Projects\Drought\Code\output\lstm\backup - moderate complexity'
wd = r'C:\Projects\Drought\Code\output\cnn'
output_dir = r'C:\Projects\Drought\Code\output\lstm'
output_dir = wd
import os
import numpy as np
import pandas as pd
import re
import shutil

# Function to extract the starting range number after "metrics_"
def extract_starting_number(filename, keyword):
    match = re.search(rf"{keyword}_(\d+)", filename)
    return int(match.group(1)) if match else float('inf')

def copy_files(source_dir, target_dir):
    """
    Copies all files from source_dir to target_dir.
    
    Parameters:
    source_dir (str): The path to the source directory.
    target_dir (str): The path to the target directory.
    """
    # Ensure source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")
    
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Iterate through the files in the source directory
    for file_name in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)
        
        # Only copy files (skip directories)
        if os.path.isfile(source_file):
            shutil.copy(source_file, target_file)
            # print(f"Copied: {file_name} -> {target_dir}")


# for var in ['P','NDVI','SM','T']:
for var in ['NDVI']:
    # for n_steps in [12,24,36,48]:
    for n_steps in [12,24]:
        
        save_dir = os.path.join(output_dir, var)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print(f'Created directory for saving {var} values.')

        var_dir = os.path.join(wd, var, str(n_steps))
        if os.path.exists(var_dir):
            files = [f.path for f in os.scandir(var_dir)]
            if len(files)>3:
                files1, files2, files3 = [], [], []
                # Sort files into lists
                for file in files:
                    if file.endswith('.csv'):
                        files1.append(file)
                    if 'metrics' in file:
                        files2.append(file)
                    if 'predictions' in file:
                        files3.append(file)
                
                ### reorder files
                files1 = sorted(files1, key=lambda x: extract_starting_number(x, keyword=var))
                files2 = sorted(files2, key=lambda x: extract_starting_number(x, keyword="metrics"))
                files3 = sorted(files3, key=lambda x: extract_starting_number(x, keyword="predictions"))
                
                ### merge csv files
                frame1 = []
                for file in files1:
                    df = pd.read_csv(file)
                    frame1.append(df)
                merged_results = pd.concat(frame1, axis=0).reset_index(drop=True)
                
                ### merge metrics
                frame2 = []
                for file in files2:
                    arr = np.load(file)
                    frame2.append(arr)
                merged_metrics = np.concatenate(frame2, axis=0)
                
                ### merge predictions
                frame3 = []
                for file in files3:
                    arr = np.load(file)
                    frame3.append(arr)
                merged_predictions = np.concatenate(frame3, axis=0)
                if len(merged_results)==2068:
                    print('Output is ok.')
                else:
                    print('Predictions are not complete.')
                
                ### save files
                merged_results.to_csv(os.path.join(save_dir, f'{n_steps}_{var}_0_2067_moderate.csv'), index=None)
                np.save(os.path.join(save_dir, f'{n_steps}_metrics_0_2067_moderate.npy'), merged_metrics)
                np.save(os.path.join(save_dir, f'{n_steps}_predictions_0_2067_moderate.npy'), merged_predictions)
                
                print(f'\nMerged predictions for {var} for {n_steps}.')
            
            else:
                print('\nNo need to merge predictions. Copied files instead.')
                copy_files(var_dir, save_dir)
