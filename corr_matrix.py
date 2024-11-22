# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:50:25 2024

@author: ghaff

this script does the following:
    1- calculate correlation between inputs (P, T, NDVI, SM, ET)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### read data
data_dir = r'C:\Projects\Drought\Code\data\Texas'
p_data = pd.read_csv(os.path.join(data_dir,'P//1_resampled//1.csv'))
t_data = pd.read_csv(os.path.join(data_dir,'LST//1_resampled//1.csv'))
n_data = pd.read_csv(os.path.join(data_dir,'NDVI//1_resampled//1.csv'))
e_data = pd.read_csv(os.path.join(data_dir,'ET//1_resampled//1.csv'))
s_data = pd.read_csv(os.path.join(data_dir,'SM//1_resampled//1.csv'))

frame_cell = []
for cell in p_data.index:                                                               # loop over cells
    p_cell = p_data.iloc[cell, 2:]
    t_cell = t_data.iloc[cell, 2:]
    n_cell = n_data.iloc[cell, 2:]
    e_cell = e_data.iloc[cell, 2:]
    s_cell = s_data.iloc[cell, 2:] 
    
    arrays = [p_cell, t_cell, n_cell, e_cell, s_cell]                                   # Create a list of arrays
    correlation_matrix = np.zeros((5, 5))                                               # Initialize a 5x5 matrix to store correlation coefficients
    for i in range(5):                                                          
        for j in range(5):
            correlation_matrix[i, j] = np.corrcoef(arrays[i], arrays[j])[0, 1]          # Calculate correlation coefficients
    frame_cell.append(correlation_matrix)

corr_avg = np.nanmean(frame_cell, axis=0)
corr_max = np.nanmax(frame_cell, axis=0)
max_indices = np.nanargmax(frame_cell, axis=0)

    

