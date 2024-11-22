# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:45:52 2023

@author: alg721

this code contains codes for predicting future values using SARIMA method.
"""
# import libraries and packages
wd = r'C:\Projects\Drought\Code'
import os
import sys
import warnings
import itertools
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed
from tqdm import tqdm
import random

sys.path.append(wd)                                                                         # append working directory to path
random.seed(42)
np.random.seed(42)

# from functions import download_data, regrid

# =============================================================================
# functions
# =============================================================================
# Create a set of SARIMA configs to try
from scipy import stats
def remove_outliers(df, z_threshold):
    # Function to replace outliers with interpolated values for each row and return a DataFrame
    def replace_outliers(row):
        z_scores = np.abs(stats.zscore(row))
        is_outlier = z_scores > z_threshold
        if any(is_outlier):
            # Interpolate outliers with adjacent values
            interpolated_row = row.copy()
            interpolated_row[is_outlier] = np.interp(
                np.where(is_outlier)[0], 
                np.where(~is_outlier)[0], 
                row[~is_outlier]
            )
            return interpolated_row
        else:
            return row
    
    data = df.iloc[:,2:]                                                                        # get only precipitation values
    coords = df.iloc[:,0:2]                                                                     # get coordinations
    filtered_df = data.apply(replace_outliers, axis=1)                                          # Apply the replace_outliers function to each row in the DataFrame
    filtered_df.reset_index(drop=True, inplace=True)                                            # Reset the index of the resulting DataFrame if needed    
    filtered_df.index = data.index
    smoothed_df = pd.concat([coords,filtered_df], axis=1)
    return smoothed_df

def sarima_configs(seasonal=[0]):
    models = list()
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n', 'c', 't', 'ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    
    # Create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p, d, q), (P, D, Q, m), t]
                                    models.append(cfg)
    return models

# SARIMA grid search function (now parallelized)
def sarima_grid_search_parallel(time_series, config):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(time_series[:-n_test], order=config[0][:3], seasonal_order=config[1], trend=config[2])
            fitted_model = model.fit(disp=-1)
            predictions = fitted_model.get_forecast(steps=n_test)  # Forecast the next 12 months
            mse = mean_squared_error(time_series[-n_test:], predictions.predicted_mean)
            nse = r2_score(time_series[-n_test:], predictions.predicted_mean)
            return predictions.predicted_mean, config, mse, nse
    except:
        return None, config, np.inf, -np.inf  # Return a large value for configurations that result in errors

# Parallelized SARIMA grid search using joblib
def parallel_sarima_grid_search(time_series, config_list, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)\
        (delayed(sarima_grid_search_parallel)(time_series, cfg) 
         for cfg in tqdm(config_list))
    results.sort(key=lambda x: x[3])                                                    # Sort the results by mean squared error (lower is better)
    return results

#%% 1- download monthly data at scale=27830m and store in 0_original
data_file = r'C:\Projects\Drought\RS_data.csv'                                       # data_file contains details for downloading data
data_dir = r'C:\Projects\Drought\Code\data\Texas'                                  # downloaded data will be saved to data_dir
bounding_box = [-106.6468, 25.8371, -93.5083, 36.5007]                                  # bounding box for the region of interest
var_ids = [14]                                                                          # variables to download
download_data(data_file, data_dir, bounding_box, var_ids)                               # download data

#%% 2- read data and regrid to remove mismatches between grids
input_files = [os.path.join(data_dir,'P//0_original//1.csv'),
               os.path.join(data_dir,'LST//0_original//1.csv'),
               os.path.join(data_dir,'NDVI//0_original//1.csv'),
               os.path.join(data_dir,'SM//0_original//1.csv'),
               os.path.join(data_dir,'ET//0_original//1.csv')]
output_files = [os.path.join(data_dir,'P//1_resampled//1.csv'),
                os.path.join(data_dir,'LST//1_resampled//1.csv'),
                os.path.join(data_dir,'NDVI//1_resampled//1.csv'),
                os.path.join(data_dir,'SM//1_resampled//1.csv'),
                os.path.join(data_dir,'ET//1_resampled//1.csv')]

regrid(input_files, output_files)

#%% run grid search for all cells
data_dir = r'C:\Projects\Drought\Code\data\Texas'
p_data = pd.read_csv(os.path.join(data_dir,'NDVI//1_resampled//1.csv'))                   # read original data
p_smoothed = remove_outliers(p_data, z_threshold=5)                                     # smooth data
# p_normalized = normalize(p_smoothed)                                                  # normalize data

configurations_to_try = sarima_configs(seasonal=[0, 12])                                # Adjust seasonal parameter as needed
n_test=12
frame = []
m, n = [1000,1100]
for cell in p_smoothed.index[m:n]:
    print(f'\nStarted predicting future values for cell {cell}:')
    cell_data = p_smoothed.iloc[cell, 2:]
    result_list = parallel_sarima_grid_search(cell_data, configurations_to_try)
    # print('Completed grid search for best config!\n Here are top 3 configs:')
    # for prediction, cfg, mse, nse in result_list[-3:][::-1]:                            # list top 3 configs
    #     print(cfg, f'MSE={np.round(mse,4)}', f'NSE={np.round(nse,4)}')
    frame.append(result_list)

### save top configs for all cells in a csv file
top_cfg = []
for f in frame:
    top_cfg.append(f[-1])
df_out = pd.DataFrame(top_cfg, columns=['predictions', 'config', 'mse', 'nse'])
df_out.to_csv(os.path.join(r'C:\Projects\Drought\Code\output\sarima\NDVI', f'{m}-{n}.csv'))

#%% single SARIMA run given the config
n_test=12; cell=56
cell_data = p_smoothed.iloc[cell, 2:]
config = df_out.loc[cell,'config']
random.seed(42)
np.random.seed(42)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = SARIMAX(cell_data[:-n_test], order=config[0][:3], seasonal_order=config[1], trend=config[2])
    fitted_model = model.fit(disp=-1)
    predictions = fitted_model.get_forecast(steps=n_test)  # Forecast the next 12 months
    mse = mean_squared_error(cell_data[-n_test:], predictions.predicted_mean)
    nse = r2_score(cell_data[-n_test:], predictions.predicted_mean)
    print(f'MSE={np.round(mse,4)}', f'NSE={np.round(nse,4)}')
    