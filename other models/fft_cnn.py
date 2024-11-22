# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:57:13 2023

@author: alg721

this script contains codes for predicting FFT using CNN. the predicted FFT can
be converted to the original data.
"""
# import libraries
wd = r'C:\Projects\Drought\Code'
import os
import sys
import numpy as np
import pandas as pd
import rasterio
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
sys.path.append(wd)                                                                             # append working directory to path
from functions import resample_batch, get_boxes

p_file = r'C:\Projects\Drought\Code\data\monthly_27830_r2\P\0_original\1.csv'
t_file = r'C:\Projects\Drought\Code\data\monthly_27830_r2\LST\0_original\1.csv'
ndvi_file = r'C:\Projects\Drought\Code\data\monthly_27830_r2\NDVI\0_original\1.csv'
sm_file = r'C:\Projects\Drought\Code\data\monthly_27830_r2\SM\0_original\1.csv'
et_file = r'C:\Projects\Drought\Code\data\monthly_deep\ET\0_original\1.csv'

output_dir = r'C:\Projects\Drought\Code'
df1 = pd.read_csv(p_file)
df2 = pd.read_csv(t_file)
df3 = pd.read_csv(ndvi_file)
df4 = pd.read_csv(sm_file)
df5 = pd.read_csv(et_file)

# filter df1 and df2 and select rows that have the same lat and lon in both
p_data = df1[df1[['latitude', 'longitude']].apply(tuple, axis=1).isin(df4[['latitude', 'longitude']].apply(tuple, axis=1))].reset_index(drop=True)
t_data = df2[df2[['latitude', 'longitude']].apply(tuple, axis=1).isin(df4[['latitude', 'longitude']].apply(tuple, axis=1))].reset_index(drop=True)
ndvi_data = df3[df3[['latitude', 'longitude']].apply(tuple, axis=1).isin(df4[['latitude', 'longitude']].apply(tuple, axis=1))].reset_index(drop=True)
sm_data = df4[df4[['latitude', 'longitude']].apply(tuple, axis=1).isin(df4[['latitude', 'longitude']].apply(tuple, axis=1))].reset_index(drop=True)
et_data = df5[df5[['latitude', 'longitude']].apply(tuple, axis=1).isin(df4[['latitude', 'longitude']].apply(tuple, axis=1))].reset_index(drop=True)

# interpolate each df to replace nan values
p_data = p_data.interpolate(method='nearest', axis=0)
t_data = t_data.interpolate(method='nearest', axis=0)
ndvi_data = ndvi_data.interpolate(method='nearest', axis=0)
sm_data = sm_data.interpolate(method='nearest', axis=0)
et_data = et_data.interpolate(method='nearest', axis=0)
    
# =============================================================================
# remove outliers and replace with interpolation
# =============================================================================
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

# =============================================================================
# normalize data using minmax method
# =============================================================================
from sklearn.preprocessing import MinMaxScaler
def normalize(df):
    data = df.iloc[:,2:]                                                                        # get only precipitation values
    coords = df.iloc[:,0:2]                                                                     # get coordination
    normalized_data = MinMaxScaler().fit_transform(data.T)                                      # Normalize each row separately and store in a new DataFrame
    filtered_df = pd.DataFrame(normalized_data.T, columns=data.columns)                         # Create a new DataFrame with the same column names
    normalized_df = pd.concat([coords,filtered_df], axis=1)
    return normalized_df

# =============================================================================
# calculate FFT values using a sliding window
# =============================================================================
def roll_fft(df, n_window, dt=1):
    data = df.iloc[:,2:]                                                                    # separate variable data from coordinates
    n_cells = len(data)                                                                     # number of cells (location tags)
    Fs = 1/dt
    fft_rol, f_rol, data_rol = [], [], []
    for n in range(np.shape(data)[1]-n_window*12+1):
        data_win = data.iloc[:,n:n+n_window*12].values
        data_length = np.shape(data_win)[1]
        fft_win = np.zeros((n_cells,data_length), dtype=np.complex128)
        f_win = np.zeros((n_cells,data_length), dtype=np.complex128)
        for i in range(n_cells):
            Y = np.fft.fft(data_win[i, :])
            fft_win[i,:] = Y
            f_win[i,:] = Fs * np.arange(0, data_length) / data_length
        f_rol.append(f_win)    
        fft_rol.append(fft_win)
        data_rol.append(data_win)
    return data_rol, fft_rol, f_rol

# =============================================================================
# roll data in moving windows of size n_window
# =============================================================================
def roll_raw(df, n_window):
    data = df.iloc[:,2:]
    data_rol = []
    n_cells = len(data)
    for n in range(np.shape(data)[1]-n_window+1):
        data_win = data.iloc[:,n:n+n_window].values
        data_rol.append(data_win)
    return data_rol

# =============================================================================
# get X and y arrays
# =============================================================================
def getXy_fft(fft, cell, lag):
    fft_cell = []
    for arr in fft:
        magnitude = np.abs(arr[cell,:])
        phase = np.angle(arr[cell,:])
        fft_cell.append([magnitude, phase])
    fft_stacked = np.stack(fft_cell)
    fft_stacked = np.transpose(fft_stacked, (0,2,1))                                            # reshape input data into (n_samples, n_steps, n_features)
    X = fft_stacked[:-lag,:,:]
    y = fft_stacked[lag:,:,:]
    return X, y

# =============================================================================
# get X and y arrays from raw data
# =============================================================================
def getXy_raw(data_rol, cell, lag):
    data_cell = []
    for arr in data_rol:
        data_cell.append(arr[cell,:])
    data_stacked = np.stack(data_cell)
    X = data_stacked[:-lag,:]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = data_stacked[lag:,:]
    return X, y

# =============================================================================
# train model and predict future FFT values
# =============================================================================
import tensorflow as tf
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# R2 score from sklearn.metrics is in fact the NSE metric ranging from -inf to 1.
from sklearn.metrics import r2_score
import keras.layers as layers

def split_data(X, y, n_test):
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]
    return X_train, X_test, y_train, y_test

def predict(X, y, n_test, complexity='high', loss='mse', verbose=0):
    # build model architecture
    n_steps, n_features = X.shape[1], X.shape[2]                                                # number of steps and number of features
    model = Sequential()
    model.add(layers.Input(shape=(n_steps, n_features)))
    
    ### more complex architecture
    if complexity=='high':
        # Convolutional layers
        model.add(layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'))
        model.add(layers.MaxPooling1D(pool_size=2))
        
        # Flatten the output of the convolutional layers
        model.add(layers.Flatten())
        
        # Add dense (fully connected) layers
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        
        # Output layer
        model.add(layers.Dense(128, activation=None))
        model.add(layers.Dense(np.shape(y)[1], activation='linear'))
    
    ### less complex architecture
    if complexity=='low':
        # Convolutional layers
        model.add(layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'))
        model.add(layers.MaxPooling1D(pool_size=2))
        
        # Flatten layer
        model.add(layers.Flatten())
        
        # Dense layers
        model.add(layers.Dense(128, activation='relu'))
        
        # Output layer for a sequence of 37 values
        model.add(layers.Dense(np.shape(y)[1], activation='linear'))
    
    # Compile the model
    model.compile(optimizer='adam', loss=loss)
    # model.summary()
    
    X_train, X_test, y_train, y_test = split_data(X, y, n_test)                                 # divide into train and test sets
    
    # define callbacks
    model_dir = r'C:\Projects\Drought\Code\model'
    save_filename = os.path.join(model_dir, 'best_model.h5')                                    # define a name for output model
    model_checkpoint = ModelCheckpoint(filepath=save_filename, monitor='val_loss', 
                                       save_best_only=True)                                     # save the best model if val_loss improves
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=verbose, 
                               restore_best_weights=True)                                       # stop training to avoid overfitting
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, 
                                  verbose=verbose, mode='auto', min_lr=0.0001)                  # reduce learning_rate if val_loss doesn't improve
    
    # run with cross validations
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.1)
    batch_size = 3
    epochs = 100
    history = model.fit(X_t, y_t, batch_size, epochs, verbose=verbose, 
                        validation_data=(X_v, y_v),
                        callbacks=[model_checkpoint, early_stop, reduce_lr])
    
    # load and evaluate best model
    best_model = load_model(save_filename)                                                      # load best model (lowest validation error)
    pred_test = best_model.predict(X_test, verbose=verbose)                                     # make predictions for X_test
    pred_train = best_model.predict(X_train, verbose=verbose)                                   # make predictions for X_train
    
    return pred_train, pred_test, y_train, y_test, X_train, X_test
    
# =============================================================================
# calculate NSE
# =============================================================================
def get_nse(y_pred, y_test, dtype):
    nse = []
    for c in range(np.shape(y_test)[0]):
        nse.append(r2_score(y_pred[c,:], y_test[c,:]))
    mean_nse = np.mean(nse)
    print(f'NSE_{dtype} = {np.round(mean_nse, 3)}')
    return nse, mean_nse

# =============================================================================
# calculate R2
# =============================================================================
from scipy.stats import linregress
def cal_r2(x,y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)                              # Calculate the linear regression    
    r_squared = r_value ** 2                                                                    # Calculate R-squared
    return r_squared

# =============================================================================
# optimize lag values for a set of n_pred values
# =============================================================================
def optimize_lag(f, pred_horiz, cell_data, magnitude, cell_frame):
    pred_p, pred_f = [], []                                                                     # create lists to store predictions for each segment
    best_lags = []
    n_pred = f                                                                                  # prediction segments in months

    ### create an array of nth segment of actual data for each FFT
    # e.g., 2nd segments (months 7-12) of actual data corresponding to each FFT
    n_range = np.arange(0,int(pred_horiz/n_pred))                                               # number of segments in prediction horizon
    for n in n_range:
        actual_data = []
        for i in range(len(cell_data)):
            segment = cell_data[i+n*n_pred:i+n*n_pred+n_pred]
            if len(segment)==n_pred:
                actual_data.append(segment)
            else:
                break
            
        ### pass each segment over FFT to find the best lag values for that segment
        frame = []
        for i in range(np.shape(magnitude)[0]):
            fft_month = i%12 + 1                                                                # calculate corresponding month
            mag = magnitude[i,:]                                                                # predicted FFT magnitude
            pha = phase[i,:]                                                                    # phase data from previous FFT
            complex_data = mag * np.exp(1j * pha)                                               # create FFT from magntiude and phase
            recon_data = np.abs(np.fft.ifft(complex_data))                                      # reconstruct original data from FFT
            # recon_data = np.abs(np.fft.ifft(mag))                                             # reconstruct original data using only magnitude data
            org_data = actual_data[i]                                                           # actual data for future n_pred months
            l_range = len(recon_data) - len(org_data) + 1                                       # range of lag values
            # l_range = 12
            org_month = pd.to_datetime(org_data.index[0]).month                                 # starting month of segment
            for l in range(l_range):
                lag_month = l%12 + org_month                                                    # calculate corresponding month
                try:
                    r2 = cal_r2(org_data, recon_data[l:l+len(org_data)])                        # compare original data with a chunk of reconstructed data
                except ValueError:
                    r2 = np.nan
                frame.append([i, fft_month, l, lag_month, lag_month-fft_month, r2])             # append r2 and corresponding indices to frame
        lag_df = pd.DataFrame(frame, columns=['fft_id', 'fft_month', 'lag_id', 
                                              'lag_month', 'lag_value', 'r2'])                  # create a df of lag values
        lags_ranked = lag_df.groupby(by='lag_id').mean()                                        # rank lag values based on lag_id
        best_lag = lags_ranked['r2'].idxmax()                                                   # get best lag value with max R2
        pred_p.append(recon_p[best_lag:best_lag+len(segment)+1])                                # append corresponding chunk of recon_f to prediction
        pred_f.append(recon_f[best_lag:best_lag+len(segment)+1])                                # append corresponding chunk of recon_f to prediction
        best_lags.append([n, best_lag])                                                         # save best_lag value for segment n
    
    ### evaluate model for train data
    pred_p_arr = np.concatenate(pred_p)
    p_data = cell_data[-2*pred_horiz:-pred_horiz]
    r2_pp = cal_r2(pred_p_arr, p_data)
    
    ### evaluate model for test data
    pred_f_arr = np.concatenate(pred_f)
    f_data = cell_data[-pred_horiz:]
    r2_ff = cal_r2(pred_f_arr, f_data)
    
    return {'n_pred':f, 'best_lags':best_lags, 'r2_pp':r2_pp, 'pred_p':pred_p,
            'r2_ff':r2_ff, 'pred_f':pred_f}

# =============================================================================
# smooth array
# =============================================================================
def smooth_row(row, thresh=6, window_size=3, sigma=1.0):
    row = pd.Series(row)
    z_scores = (row - row.mean()) / row.std()                                                   # calculate z-score values
    outliers = np.where(np.abs(z_scores) > thresh)[0]                                           # Identify outliers using z-score values
    row_without_outliers = row.copy()
    row_without_outliers[outliers] = np.nan                                                     # Remove outliers from the row
    interpolated_row = row_without_outliers.interpolate()                                       # Interpolate missing values in the row
    smoothed_row = gaussian_filter1d(interpolated_row, sigma)                                   # Smooth the row using Gaussian smoothing
    return smoothed_row

# =============================================================================
# save an array to a tiff file
# =============================================================================
from scipy.interpolate import griddata
def save_tiff(data, coords, output_file, arc_flip=False):    
    x = pd.unique(coords['longitude'])                                                          # get longitudes
    y = pd.unique(coords['latitude'])                                                           # get latitudes
    xx, yy = np.meshgrid(x, y)                                                                  # create a meshgrid of lat and lon
    grid = griddata((coords['longitude'], coords['latitude']), data, (xx, yy), 
                    method='nearest')                                                           # reshaoe array into a grid
    num_rows, num_cols = grid.shape                                                             # get number of rows and columns
    left, bottom = np.min(x), np.min(y)                                                         # get coordinates for left bottom
    pixel_width, pixel_height = x[1] - x[0], y[0] - y[1]
    transform = rasterio.transform.from_origin(left, bottom, pixel_width, pixel_height)
    if arc_flip:
        grid = np.flipud(grid)
    
    # write the grid to the output file as a GeoTIFF
    with rasterio.open(
            output_file,
            mode='w',
            driver='GTiff',
            height=num_rows,
            width=num_cols,
            count=1,
            dtype=grid.dtype,
            crs = 'EPSG:4326',
            transform=transform,
    ) as dst:
        dst.write(grid, 1)
    
    # add coordinate arrays as metadata
    with rasterio.open(output_file, mode='r+') as dst:
        dst.update_tags(1, x=x.tolist(), y=y.tolist())

#%% 0- predict future values using an overlap between train and prediction data
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)                                                             # Set random seed for NumPy
tf.random.set_seed(random_seed)                                                         # Set random seed for TensorFlow

n_test=1; lag=12; complexity='high'; n_window=24; cut=0; var='P'
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
    ndvi_cut = ndvi_data.iloc[:,:-cut]
    sm_cut = sm_data.iloc[:,:-cut]
    et_cut = et_data.iloc[:,:-cut]
else:
    p_cut = p_data
    t_cut = t_data
    ndvi_cut = ndvi_data
    sm_cut = sm_data
    et_cut = et_data
### smooth data to remove outliers
p_smoothed = remove_outliers(p_cut, z_threshold=5)
t_smoothed = remove_outliers(t_cut, z_threshold=5)
ndvi_smoothed = remove_outliers(ndvi_cut, z_threshold=5)
sm_smoothed = remove_outliers(sm_cut, z_threshold=5)
et_smoothed = remove_outliers(et_cut, z_threshold=5)
### normalize data using MinMaxScaler
p_normalized = normalize(p_smoothed)
t_normalized = normalize(t_smoothed)
ndvi_normalized = normalize(ndvi_smoothed)
sm_normalized = normalize(sm_smoothed)
et_normalized = normalize(et_smoothed)
# p_normalized = normalize(p_data)
# t_normalized = normalize(t_data)
### create rolling windows of data
p_roll = roll_raw(p_normalized, n_window)
t_roll = roll_raw(t_normalized, n_window)
ndvi_roll = roll_raw(ndvi_normalized, n_window)
sm_roll = roll_raw(sm_normalized, n_window)
et_roll = roll_raw(et_normalized, n_window)
# p_roll = roll_raw(p_smoothed, n_window)
# t_roll = roll_raw(t_smoothed, n_window)
frame_out = []                                                                                  # create a list to store results for each cell

for cell in p_normalized.index:
# cell = 1579
    print(f'Started predicting future {var} values for cell {cell}:')
    ### predict values and evaluate results using NSE
    X_p, y_p = getXy_raw(p_roll, cell, lag)
    X_t, y_t = getXy_raw(t_roll, cell, lag)
    X_n, y_n = getXy_raw(ndvi_roll, cell, lag)
    X_s, y_s = getXy_raw(sm_roll, cell, lag)
    X_e, y_e = getXy_raw(et_roll, cell, lag)
    X = np.concatenate((X_p, X_t, X_n, X_s, X_e), axis=-1)
    pred_train, pred_test, y_train, y_test, X_train, X_test = \
        predict(X, y_p, n_test, complexity=complexity, loss='mae')                              # predict future values
    train_nse = get_nse(y_train, pred_train, dtype='train')                                     # calculate NSE for train data
    test_nse = get_nse(y_test, pred_test, dtype='test')                                         # calculate NSE for test data
    nse = r2_score(p_normalized.iloc[cell,-lag:], pred_test[-1,-lag:])
    print(f'NSE for prediction = {np.round(nse,2)}\n')
    frame_out.append([cell, train_nse[1], test_nse[1], nse, pred_test[-1,-lag:]])

df_out = pd.DataFrame(frame_out, columns=['cell','train_nse','test_nse','pred_nse','prediction'])
df_out.to_csv(os.path.join(wd,f'pred_{var}_{lag}.csv'))
save_tiff(df_out['pred_nse'], sm_data, os.path.join(wd, f'nse_{lag}_{var}.tiff'))

#%%
# plot results for one cell
fig, ax = plt.subplots()
p = n_window
ax.plot(range(p), sm_normalized.iloc[cell,-p:], label='actual')
ax.plot(range(p), pred_test[-1,:], label='predicted')
ax.axvline(x=p-lag, color='r', linestyle='--')
nse = r2_score(sm_normalized.iloc[cell,-lag:], pred_test[-1,-lag:])
fig.suptitle(f'Cell {cell}; lag={lag}; complexity={complexity}; NSE={np.round(nse,2)}')
print(f'NSE for prediction = {np.round(nse,2)}\n')

# ax.plot(range(12), prediction, label='prediction')
fig.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.07))
ax.set_ylabel('Precipitation (normalized)')
ax.set_xlabel('month')
plt.show()
# fig.savefig(os.path.join(output_dir, f'prediction_{cell}.png'), dpi=300, bbox_inches='tight')

#%% predict future values without any overlap
def getXy_nolag(df, cell, x_l, y_l):
    cell_data = df.iloc[cell,2:].values                                                            # extract data for cell
    frame_X, frame_y = [], []
    for i in range(len(cell_data)-x_l-y_l+1):
        frame_X.append(cell_data[i:i+x_l])
        frame_y.append(cell_data[i+x_l:i+x_l+y_l])
    
    X = np.stack(frame_X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = np.stack(frame_y)
    return X, y

n_test=1; complexity='low'; x_l=12; y_l=12; cell=1200

frame_out = []
# for cell in p_normalized.index[-4:]:
print(f'Started predicting future FFT values for cell {cell}:')
### predict values and evaluate results using NSE
X_p, y_p = getXy_nolag(p_normalized, cell, x_l, y_l)
X_t, y_t = getXy_nolag(t_normalized, cell, x_l, y_l)
X = np.concatenate((X_p, X_t), axis=-1)
pred_train, pred_test, y_train, y_test, X_train, X_test = \
    predict(X, y_p, n_test, complexity=complexity, loss='mae')                              # predict future values
# train_nse = get_nse(y_train, pred_train, dtype='train')                                     # calculate NSE for train data
# test_nse = get_nse(y_test, pred_test, dtype='test')                                         # calculate NSE for test data
nse = r2_score(pred_test, y_test)

# frame_out.append([cell, train_nse[1], test_nse[1], pred_test])

fig, ax = plt.subplots()
p = n_window
ax.plot(y_test[-1,:], label='actual')
ax.plot(pred_test[-1,:], label='predicted')
# ax.plot(range(12), prediction, label='prediction')
fig.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.07))
ax.set_ylabel('Precipitation (normalized)')
ax.set_xlabel('month')
plt.show()
fig.savefig(os.path.join(output_dir, f'prediction_{cell}.png'), dpi=300, bbox_inches='tight')


#%% predict future values using whole year as X and y data (limited training data)
def getXy_limited(df, cell, x_l, y_l):
    cell_data = df.iloc[cell,2:].values                                                            # extract data for cell
    frame_X, frame_y = [], []
    for i in range(int(len(cell_data)/12)):
        frame_X.append(cell_data[i*12:i*12+x_l])
        frame_y.append(cell_data[i*12+x_l:i*12+x_l+y_l])
    
    X = np.stack(frame_X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = np.stack(frame_y)
    return X, y

n_test=1; complexity='low'; x_l=6; y_l=6; cell=1200

frame_out = []
# for cell in p_normalized.index[-4:]:
print(f'Started predicting future FFT values for cell {cell}:')
### predict values and evaluate results using NSE
X_p, y_p = getXy_limited(p_normalized, cell, x_l, y_l)
X_t, y_t = getXy_limited(t_normalized, cell, x_l, y_l)
X = np.concatenate((X_p, X_t), axis=-1)
pred_train, pred_test, y_train, y_test, X_train, X_test = \
    predict(X_p, y_p, n_test, complexity=complexity, loss='mae')                              # predict future values
# train_nse = get_nse(y_train, pred_train, dtype='train')                                     # calculate NSE for train data
# test_nse = get_nse(y_test, pred_test, dtype='test')                                         # calculate NSE for test data
nse = r2_score(pred_test, y_test)

# frame_out.append([cell, train_nse[1], test_nse[1], pred_test])

fig, ax = plt.subplots()
p = n_window
ax.plot(y_test[-1,:], label='actual')
ax.plot(pred_test[-1,:], label='predicted')
# ax.plot(range(12), prediction, label='prediction')
fig.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.07))
ax.set_ylabel('Precipitation (normalized)')
ax.set_xlabel('month')
plt.show()
fig.savefig(os.path.join(output_dir, f'prediction_{cell}.png'), dpi=300, bbox_inches='tight')












#%% predict future data given best lag values
smoothed_df = remove_outliers(data, z_threshold=5)                                              # smooth data
normalized_df = normalize(smoothed_df)                                                          # normalize data using MinMaxScaler
fft_rol, f_rol = roll_fft(smoothed_df, n_window=4)                                              # calculate FFT values using
frame_out = []
parallel = True

n_test=24; lag=48; pred_horiz=48
# for cell in smoothed_df.index:
for cell in [0,500,1000,1500,2000]:
    ### predict FFT values and evaluate results using NSE
    print(f'\nStarted predicting future FFT values for cell {cell}:')
    X, y = getXy_fft(fft_rol, cell=cell, lag=lag)                                               # get X and y arrays
    X_t = X[:,:,0].reshape(X.shape[0], X.shape[1], 1)                                           # X_t for model training
    y_t = y[:,:,0]                                                                              # y_t for model training
    pred_train, pred_test, y_train, y_test, X_train, X_test = pred_fft(X_t, y_t, n_test)        # predict future FFT values
    train_nse = get_nse(y_train, pred_train, dtype='train')                                     # calculate NSE for train data
    test_nse = get_nse(y_test, pred_test, dtype='test')                                         # calculate NSE for test data
    
    ### reconstruct original data and find best lag values
    print(f'Calculating best lag values for cell {cell}:')
    ### reconstruct data for last train data
    mag_p = pred_train[-1,:]                                                                        # magnitude
    pha_p = X[len(pred_train)-1,:,1]                                                                # phase
    complex_p = mag_p * np.exp(1j * pha_p)                                                          # create FFT from magntiude and phase
    recon_p = np.abs(np.fft.ifft(complex_p))                                                        # reconstruct original data from FFT
    
    ### reconstruct data for last test data
    mag_f = pred_test[-1,:]                                                                         # magnitude
    pha_f = X[-1,:,1]                                                                               # phase
    complex_f = mag_f * np.exp(1j * pha_f)                                                          # create FFT from magntiude and phase
    recon_f = np.abs(np.fft.ifft(complex_f))                                                        # reconstruct original data from FFT
    
    magnitude = pred_train                                                                          # historical magnitudes
    # magnitude = X_train[:,:,0]
    phase = X[:,:,1]                                                                                # historical phases
    cell_data = smoothed_df.iloc[cell, 2+lag:]                                                      # cell data
    cell_frame = []
    
    if parallel:
        f_frame = Parallel(n_jobs=-1)\
            (delayed(optimize_lag)\
             (f, pred_horiz, cell_data, magnitude, cell_frame)
             for f in tqdm([3,4,6,8,12,16,24,48]))
        frame_out.append(f_frame)
    else:
        for f in tqdm([3,4,6,8,12,16,24,48]):
            f_frame = optimize_lag(f, pred_horiz, cell_data, magnitude)
            cell_frame.append(f_frame)
        frame_out.append(cell_frame)

#%% predict data based on r2 values for n_preds
cells = [0,500,1000,1500,2000]
frame_desc = []
for i in range(len(frame_out)):
    cell = frame_out[i]
    for f in cell:
        frame_desc.append([cells[i], f['n_pred'], f['best_lags'], f['r2_pp'], 
                          f['pred_p'], f['r2_ff'], f['pred_f']])

df_desc = pd.DataFrame(frame_desc, columns=['cell', 'n_pred', 'best_lags', 
                                            'r2_pp', 'pred_p', 'r2_ff', 'pred_f'])
df_desc.to_csv('C:\Projects\Drought\Code\df_desc.csv', index=None)

frame_pred = []
for cell in cells:
    df_cell = df_desc[df_desc['cell']==cell]
    if df_cell['r2_pp'].max() >= 0.6:
        prediction = np.concatenate(df_cell['pred_f'][df_cell['r2_pp'].idxmax()])
    else:
        prediction = np.concatenate(df_cell['pred_f'].iloc[-1])
    frame_pred.append([cell, prediction])   

#%%
var = 'SM'
for i in range(5):
    cell = frame_pred[i][0]
    fig, ax = plt.subplots(3,1, constrained_layout=True)
    fig.suptitle(f'Prediction results for cell {cell} for 2019-01 to 2022-12')
    r2 = np.round(cal_r2(smoothed_df.iloc[cell,-48:], frame_pred[i][1]),2)
    nse = np.round(r2_score(smoothed_df.iloc[cell,-48:], frame_pred[i][1]),2)
    ax[0].set_title(f'Soil moisture; R2={r2}; NSE={nse}')
    ax[0].plot(np.arange(1,49), smoothed_df.iloc[cell,-48:], label='Actual')
    ax[0].plot(np.arange(1,49), frame_pred[i][1], label='Prediction')
    # ax[0].set_xlabel('month')
    ax[0].set_ylabel('SM')
    
    r2 = np.round(cal_r2(smooth_row(smoothed_df.iloc[cell,-48:]), smooth_row(frame_pred[i][1])),2)
    nse = np.round(r2_score(smooth_row(smoothed_df.iloc[cell,-48:]), smooth_row(frame_pred[i][1])),2)
    ax[1].set_title(f'Soil moisture (Smoothed); R2={r2}; NSE={nse}')
    ax[1].plot(np.arange(1,49), smooth_row(smoothed_df.iloc[cell,-48:]))
    ax[1].plot(np.arange(1,49), smooth_row(frame_pred[i][1]))
    # ax[1].set_xlabel('month')
    ax[1].set_ylabel('SM')
    
    ax[2].set_title(f'Soil moisture time series for cell {cell} from 2001 to 2022')
    ax[2].plot(np.arange(1,265), smoothed_df.iloc[cell, 2:])
    ax[2].set_xlabel('month')
    ax[2].set_ylabel('SM')
    ax[2].axvline(x = 216, color = 'red')
    fig.legend(bbox_to_anchor=(0.75,0), ncol=2)

    plt.show()
    plt.subplots_adjust(hspace=0.4)  # Adjust hspace as needed

    fig.savefig(os.path.join(output_dir, f'prediction_{cell}_{var}.png'), dpi=300, bbox_inches='tight')

