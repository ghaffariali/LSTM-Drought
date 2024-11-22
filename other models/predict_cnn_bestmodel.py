# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:18:31 2023

@author: alg721

this scripts does the following:
    1- download data for an area based on given bounding box
    2- regrid all data to remove any mismatch between grids
    
"""
# import libraries and packages
wd = r'C:\Projects\Drought\Code'
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import random
import tensorflow as tf
from sklearn.metrics import r2_score

sys.path.append(wd)                                                                         # append working directory to path
from functions import download_data, regrid
from fn_prediction import remove_outliers, normalize, roll_raw, getXy_raw, \
    predict_cnn, get_nse, save_tiff

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)                                                                 # Set random seed for NumPy
tf.random.set_seed(random_seed)                                                             # Set random seed for TensorFlow
           
#%% 1- download monthly data at scale=27830m and store in 0_original
data_file = r'C:\Projects\Drought\Code\data\RS_data.csv'                                    # data_file contains details for downloading data
data_dir = r'C:\Projects\Drought\Code\data\Texas'                                           # downloaded data will be saved to data_dir
bounding_box = [-109.0, 32.0, -103.0, 37.0]                                                 # bounding box for the region of interest
var_ids = [0,1,2,3,4]                                                                       # variables to download - refer to RS_data.csv for more info
download_data(data_file, data_dir, bounding_box, var_ids)                                   # download data

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

#%% 0- predict future values using an overlap between train and prediction data
p_data = pd.read_csv(output_files[0])
t_data = pd.read_csv(output_files[1])
n_data = pd.read_csv(output_files[2])
s_data = pd.read_csv(output_files[3])
e_data = pd.read_csv(output_files[4])

n_test=1; lag=12; complexity='low'; n_window=24; cut=0; var='P'; loc='NM'
"""
these parameters control the settings of the model:
    1- n_window: number of months in each segment of data. larger n_window allows
    the model to capture longer term trends but could also lead to generalization
    and make the model go for trend rather than accurate predictions.
    
    2- n_test: number of samples used for testing the CNN model. this is crutial
    for making sure the model does not overfit but since we are short on data,
    I only use 12 samples for test.
    
    3- lag: number of months that overlap between X and y data. CNN gets lost
    when there is no overlap between X and y data and goes for predicting the
    train rather than actual predictions. So, a starting overlap can help the
    CNN make more accurate predictions. lag here coincides with the concept of 
    prediction horizon. larg↕er lag values indicate longer prediction horizons
    and therefore, comes with higher uncertainty and less accuracy.
    
    4- complexity: has two options: low and high, which relate to the complexity
    of the CNN model. a high complexity model can get to more accurate predictions
    but at the cost of overfitting and also higher computational expense. if
    there is a clear pattern in the training data, a high complexity model is
    suggested. if not, use low complexity.
"""
### cut data to select a certain period for prediction
if cut!=0:
    p_cut = p_data.iloc[:,:-cut]
    t_cut = t_data.iloc[:,:-cut]
    n_cut = n_data.iloc[:,:-cut]
    s_cut = s_data.iloc[:,:-cut]
    e_cut = e_data.iloc[:,:-cut]
else:
    p_cut = p_data
    t_cut = t_data
    n_cut = n_data
    s_cut = s_data
    e_cut = e_data

### smooth data to remove outliers
p_smoothed = remove_outliers(p_cut, z_threshold=5)
t_smoothed = remove_outliers(t_cut, z_threshold=5)
ndvi_smoothed = remove_outliers(n_cut, z_threshold=5)
sm_smoothed = remove_outliers(s_cut, z_threshold=5)
et_smoothed = remove_outliers(e_cut, z_threshold=5)

### normalize data using MinMaxScaler
p_normalized = normalize(p_smoothed)
t_normalized = normalize(t_smoothed)
ndvi_normalized = normalize(ndvi_smoothed)
sm_normalized = normalize(sm_smoothed)
et_normalized = normalize(et_smoothed)

### create rolling windows of data
p_roll = roll_raw(p_normalized, n_window)
t_roll = roll_raw(t_normalized, n_window)
ndvi_roll = roll_raw(ndvi_normalized, n_window)
sm_roll = roll_raw(sm_normalized, n_window)
et_roll = roll_raw(et_normalized, n_window)

frame_out = []                                                                                  # create a list to store results for each cell
for cell in p_normalized.index:
    print(f'Started predicting future {var} values for cell {cell}:')
    ### predict values and evaluate results using NSE
    X_p, y_p = getXy_raw(p_roll, cell, lag)
    X_t, y_t = getXy_raw(t_roll, cell, lag)
    X_n, y_n = getXy_raw(ndvi_roll, cell, lag)
    X_s, y_s = getXy_raw(sm_roll, cell, lag)
    X_e, y_e = getXy_raw(et_roll, cell, lag)
    X = np.concatenate((X_p, X_t, X_n, X_s, X_e), axis=-1)
    pred_train, pred_test, y_train, y_test, X_train, X_test = \
        predict_cnn(X, y_p, n_test, complexity=complexity, loss='mae')                          # predict future values
    train_nse = get_nse(y_train, pred_train, dtype='train')                                     # calculate NSE for train data
    test_nse = get_nse(y_test, pred_test, dtype='test')                                         # calculate NSE for test data
    nse = r2_score(p_normalized.iloc[cell,-lag:], pred_test[-1,-lag:])
    print(f'NSE for prediction = {np.round(nse,2)}\n')
    frame_out.append([cell, train_nse[1], test_nse[1], nse, pred_test[-1,-lag:]])

df_out = pd.DataFrame(frame_out, columns=['cell','train_nse','test_nse','pred_nse','prediction'])
df_out.to_csv(os.path.join(wd,f'{loc}_pred_{lag}_{var}.csv'))
save_tiff(df_out['pred_nse'], s_data, os.path.join(wd, f'{loc}_nse_{lag}_{var}.tiff'))

#%% plot results for one cell
cell=0
fig, ax = plt.subplots()
ax.plot(range(n_window), p_normalized.iloc[cell,-n_window:], label='actual')
ax.plot(range(n_window), pred_test[-1,:], label='predicted')
ax.axvline(x=n_window-lag, color='r', linestyle='--')
nse = r2_score(p_normalized.iloc[cell,-lag:], pred_test[-1,-lag:])
fig.suptitle(f'Cell {cell}; lag={lag}; complexity={complexity}; NSE={np.round(nse,2)}')
print(f'NSE for prediction = {np.round(nse,2)}\n')
fig.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.07))
ax.set_ylabel('Precipitation (normalized)')
ax.set_xlabel('month')
plt.show()
# fig.savefig(os.path.join(output_dir, f'prediction_{cell}.png'), dpi=300, bbox_inches='tight')