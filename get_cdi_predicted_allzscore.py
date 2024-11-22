# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:29:57 2023

@author: ghaff
"""
wd = r'C:\Projects\Drought\Code'
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(wd)
from functions import zscore_batch, spi_batch, stack_monthly_batch, get_weights, \
    cdi_batch, save_tiff_batch
from sklearn.preprocessing import MinMaxScaler

data_dir = os.path.join(wd, 'data\\Texas_pred')                                         # this is a copy of original data from Texas.zip in data folder
pred_dir = os.path.join(wd, 'output\\lstm\\predictions')                                # predicted values from different n_steps
cdi_dir = os.path.join(wd, 'output\\cdi_predicted_allzscore')
if os.path.exists(cdi_dir):
    print('Directory for predicted CDI values exists.')
else:
    os.mkdir(cdi_dir)

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

from sklearn.preprocessing import MinMaxScaler
def normalize(df):
    scaler = MinMaxScaler()
    data = df.iloc[:,2:]                                                                        # get only precipitation values
    coords = df.iloc[:,0:2]                                                                     # get coordination
    normalized_data = scaler.fit_transform(data.T)                                              # Normalize each row separately and store in a new DataFrame
    filtered_df = pd.DataFrame(normalized_data.T, columns=data.columns)                         # Create a new DataFrame with the same column names
    normalized_df = pd.concat([coords,filtered_df], axis=1)
    return normalized_df

#%% combine predictions with actual data
for var in ['P','NDVI','LST','SM']:                                                             # loop over variables
    predictions = np.array(pd.read_csv(os.path.join(pred_dir, f'{var}_combined.csv'))['yhat'])  # read df_combined
    frame = []                                                                                  # create a list to store results
    for p in predictions:                                                                       # loop over predictions
        values = p.strip('[]').split()                                                          # parse strings
        numeric_array = np.array([float(value) for value in values])                            # convert strings to numerics
        frame.append(numeric_array)                                                             # append results to frame
    df_pred = pd.DataFrame(frame)                                                               # create a df
    df_pred = df_pred.clip(lower=0)                                                             # Clip the DataFrame to replace negative values with zero
    # df_pred.to_csv(os.path.join(pred_dir, f'{var}_predictions.csv'))                            # save as csv
    
    for folder in ['1_normalized', '1_combined']:
        target_dir = os.path.join(data_dir, f'{var}//{folder}')
        if os.path.exists(target_dir) == False:                                                 # check if output_dir exists; if not, create one.
            os.mkdir(target_dir)
    df_actu = pd.read_csv(os.path.join(data_dir, f'{var}//1_resampled//1.csv'))                 # read actual data
    df_smoothed = remove_outliers(df_actu, z_threshold=5)                                       # smooth data
    df_normalized = normalize(df_smoothed)                                                      # normalize data
    df_normalized.to_csv(os.path.join(data_dir, f'{var}//1_normalized//1.csv'), index=None)     # save normalized data as csv
    df_out = df_normalized
    df_out.iloc[:,-12:] = df_pred                                                               # replace last 12 months with predicted data
    df_out.to_csv(os.path.join(data_dir, f'{var}//1_combined//1.csv'), index=None)              # save combined data as csv

#%% calculate SPI or Z-Score and store in 2_zscore
for var in ['LST','NDVI','SM','P']:                                                                 # calculate zscore for LST, NDVI, and SM
    input_dir = os.path.join(data_dir, var, '1_combined')
    output_dir = os.path.join(data_dir, var, '2_zscore')
    zscore_batch(input_dir, output_dir)

# input_dir = os.path.join(data_dir, 'P', '1_combined')
# output_dir = os.path.join(data_dir, 'P', '2_zscore')
# spi_batch(input_dir, output_dir, thresh=3)                                                      # calculate SPI for P

#%% stack input files into 3D numpy arrays
for var in ['P','LST','NDVI','SM']:
    input_dir = os.path.join(data_dir, var, '2_zscore')
    coords_dir = os.path.join(data_dir, var, '3_coords')
    output_dir = os.path.join(data_dir, var, '4_stacked')
    stack_monthly_batch(input_dir, output_dir, coords_dir, thresh=3)

#%% calculate weights of each input in each month using PCA
files_list = [os.path.join(data_dir, 'P', '1_combined'),
              os.path.join(data_dir, 'LST', '1_combined'),
              os.path.join(data_dir, 'NDVI', '1_combined'),
              os.path.join(data_dir, 'SM', '1_combined')]

output_dir = os.path.join(cdi_dir, 'weights')                                                   # output_dir
get_weights(files_list, output_dir)

#%% calculate CDI values
input_dir = data_dir
output_dir = os.path.join(cdi_dir, 'output')
weights_dir = os.path.join(cdi_dir, 'weights')
cdi_batch(input_dir, output_dir, weights_dir, w='pca', thresh=3)
cdi_batch(input_dir, output_dir, weights_dir, w='constant', thresh=3)

#%% create tiff images from CDI values
for w in ['pca', 'constant']:
    input_dir = os.path.join(cdi_dir, f'output\\{w}')
    coords_dir = os.path.join(wd, 'data\\Texas\\LST\\3_coords')
    maps_dir = os.path.join(cdi_dir, 'maps')
    output_dir = os.path.join(maps_dir, f'{w}')
    save_tiff_batch(input_dir, coords_dir, maps_dir, output_dir,
                    start_date='2001-03-01', end_date='2022-12-01')
    
#%% create tiff images from SPI values
input_dir = os.path.join(data_dir, 'P', '4_stacked')
coords_dir = os.path.join(wd, 'data\\Texas\\LST\\3_coords')
maps_dir = os.path.join(data_dir, 'P', 'maps')
output_dir = os.path.join(maps_dir, 'pca')
save_tiff_batch(input_dir, coords_dir, maps_dir, output_dir, 
                start_date='2001-03-01', end_date='2022-12-01')



