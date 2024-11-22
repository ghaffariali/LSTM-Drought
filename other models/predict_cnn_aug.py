# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:46:01 2023

@author: ghaff

this script contains codes for predicting future values of time series using
LSTM method. The code uses four features and loops over different values of 
batchsize to get the best accuracy.
"""
# import libraries and packages
import os
import sys
import warnings
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from keras.models import Sequential
from keras.layers import LSTM, Dense
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
import random
from tqdm import tqdm

# sys.path.append(r'C:\Projects\Drought\code')
# from fn_prediction import remove_outliers, normalize

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


def get_Xy(sequence, n_steps, n_pred):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix+n_pred > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+n_pred]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))                                          # reshape from [samples, timesteps] into [samples, timesteps, features]
    y = np.stack(y)
    return X, y

def split_data(X, y, n_pred=12):
    X_train = X[:-n_pred,:,:]; y_train = y[:-n_pred,:]
    X_test = X[-n_pred:,:,:]; y_test = y[-n_pred:,:]
    return X_train, y_train, X_test, y_test

def get_traintest(p_normalized, t_normalized, n_normalized, e_normalized, s_normalized, 
                  cell, n_steps, n_pred):
    ### get X and y for each variable
    X_p, y_p = get_Xy(p_normalized.iloc[cell, 2:], n_steps, n_pred)
    X_t, y_t = get_Xy(t_normalized.iloc[cell, 2:], n_steps, n_pred)
    X_n, y_n = get_Xy(n_normalized.iloc[cell, 2:], n_steps, n_pred)
    X_e, y_e = get_Xy(e_normalized.iloc[cell, 2:], n_steps, n_pred)
    X_s, y_s = get_Xy(s_normalized.iloc[cell, 2:], n_steps, n_pred)

    X = np.concatenate((X_p, X_t, X_n, X_e, X_s), axis=2)                                    # concat data
    X_train = X[:-n_pred,:,:]; y_train = y_p[:-n_pred,:]
    X_test = X[-n_pred:,:,:]; y_test = y_p[-n_pred:,:]
    return X_train, y_train, X_test, y_test

import keras.layers as layers
def predict_cell(cell, p_normalized, t_normalized, n_normalized, e_normalized, 
                 frame_cell):
    X_train, y_train, X_test, y_test = get_traintest(
        p_normalized, t_normalized, n_normalized, e_normalized, cell)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame_b = []
        for b in tqdm(np.arange(1,9)):
            random.seed(42)
            np.random.seed(42)
            tf.random.set_seed(42)
            
            # define model with low complexity
            model = Sequential()
            model.add(layers.Input(shape=(n_steps, n_features)))
            model.add(layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'))
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'))
            model.add(layers.MaxPooling1D(pool_size=2))            
            model.add(layers.Flatten())                                                     # Flatten layer            
            model.add(layers.Dense(128, activation='relu'))                                 # Dense layers
            model.add(layers.Dense(n_pred, activation='linear'))                            # output layer
                    
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=100, verbose=0, batch_size=b)                # fit model
            yhat = model.predict(X_test, verbose=0)[-1,:]                                   # demonstrate prediction
            mse = mean_squared_error(y_test[-1,:], yhat)
            nse = r2_score(y_test[-1,:], yhat)
            frame_b.append([cell, b, mse, nse])
            # print(f'batchsize={b}', f'MSE={np.round(mse,4)}', f'NSE={np.round(nse,4)}')
            
            # Reset the model for the next iteration
            del model
            tf.keras.backend.clear_session()
        frame_cell.append(frame_b)
    
def predict_batchsize(frame_b, b):
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # define model with low complexity
    model = Sequential()
    model.add(layers.Input(shape=(n_steps, n_features)))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))            
    model.add(layers.Flatten())                                                             # Flatten layer            
    model.add(layers.Dense(128, activation='relu'))                                         # Dense layers
    model.add(layers.Dense(n_pred, activation='linear'))                                    # output layer
    
    # define model with high complexity
    # model = Sequential()
    # model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    # model.add(LSTM(units=50, activation='relu', return_sequences=True))
    # model.add(LSTM(units=50, activation='relu'))
    # model.add(Dense(n_pred))
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, verbose=0, batch_size=b)                # fit model
    yhat = model.predict(X_test, verbose=0)[-1,:]                                   # demonstrate prediction
    mse = mean_squared_error(y_test[-1,:], yhat)
    nse = r2_score(y_test[-1,:], yhat)
    # frame_b.append([cell, b, mse, nse])
    # print(f'batchsize={b}', f'MSE={np.round(mse,4)}', f'NSE={np.round(nse,4)}')
    
    # Reset the model for the next iteration
    del model
    tf.keras.backend.clear_session()
    return cell, b, mse, nse, y_test[-1,:], yhat

def augment_data(df_original, p_rate):
    df = df_original.copy()
    random.seed(42)
    np.random.seed(42)
    coords = df.iloc[:,:2]                                                          # coordinates
    data = df.iloc[:,2:-12]                                                         # original data except the last year
    last_yr = df.iloc[:,-12:]                                                       # last year
    aug_data = pd.DataFrame(data.values*np.random.normal(loc=1, scale=p_rate, 
                                                         size=np.shape(data.values)),
                            columns=data.columns)
    df_out = pd.concat([coords, aug_data, data, last_yr], axis=1)
    return df_out

#%% parallel version
start_time = time.time()
n_steps=24; n_pred=12; n_features=5; var='P'                                            # data preparation parameters
epochs=100; verbose=0; complexity='low'                                                 # LSTM parameters

### read, smooth, and normalize data
data_dir = r'C:\Projects\Drought\Code\data\Texas'
p_data = pd.read_csv(os.path.join(data_dir,'P//1_resampled//1.csv'))
t_data = pd.read_csv(os.path.join(data_dir,'LST//1_resampled//1.csv'))
n_data = pd.read_csv(os.path.join(data_dir,'NDVI//1_resampled//1.csv'))
e_data = pd.read_csv(os.path.join(data_dir,'ET//1_resampled//1.csv'))
s_data = pd.read_csv(os.path.join(data_dir,'SM//1_resampled//1.csv'))

### remove Elnio years before preprocessing
# el_yrs = [2003, 2005, 2007, 2010, 2015, 2016, 2019]
# p_data = sep_elnino(p_data, el_yrs)
# t_data = sep_elnino(t_data, el_yrs)
# n_data = sep_elnino(n_data, el_yrs)
# e_data = sep_elnino(e_data, el_yrs)

p_smoothed = remove_outliers(p_data, z_threshold=5)
t_smoothed = remove_outliers(t_data, z_threshold=5)
n_smoothed = remove_outliers(n_data, z_threshold=5)
e_smoothed = remove_outliers(e_data, z_threshold=5)
s_smoothed = remove_outliers(s_data, z_threshold=5)

p_normalized = normalize(p_smoothed)
t_normalized = normalize(t_smoothed)
n_normalized = normalize(n_smoothed)
e_normalized = normalize(e_smoothed)
s_normalized = normalize(s_smoothed)

### data augmentation
p_rate = 0.05
p_augmented = augment_data(p_normalized, p_rate)
t_augmented = augment_data(t_normalized, p_rate)
n_augmented = augment_data(n_normalized, p_rate)
e_augmented = augment_data(e_normalized, p_rate)
s_augmented = augment_data(s_normalized, p_rate)

frame_cell = []
m, n = 0, 5
for cell in p_normalized.index[m:n]:
    print(f'\nStarted predicting for cell {cell}:')
    X_train, y_train, X_test, y_test = get_traintest(
        p_augmented, t_augmented, n_augmented, e_augmented, s_augmented, cell, n_steps, n_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame_b = []
        results = Parallel(n_jobs=-1)\
            (delayed(predict_batchsize)(frame_b, b) for b in tqdm(np.arange(1,17)))
    frame_cell.append(results)

### separate metrics from predictions
metrics, predictions = [], []
for cell in frame_cell:
    frame_m , frame_p = [], []
    for batch in cell:
        frame_m.append(batch[0:4])
        frame_p.append(batch[4:])
    metrics.append(frame_m)
    predictions.append(frame_p)

### select best batchsize for all cells
frame_best = []
for cell in frame_cell:
    df_cell = pd.DataFrame(cell, columns=['cell','batchsize','mse','nse', 'y_test', 'yhat'])
    best_b = df_cell['nse'].idxmax()
    frame_best.append(df_cell.iloc[best_b,:].values)

### save results
df_p = pd.DataFrame(frame_best, columns=['cell','batchsize','mse','nse', 'y_test', 'yhat'])
df_p.to_csv(f'C:\Projects\Drought\Code\output\cnn\{var}\lstm_{m}_{n}.csv', index=None)   
np.save(f'C:\Projects\Drought\Code\output\cnn\{var}\metrics_{m}_{n}.npy', metrics)
np.save(f'C:\Projects\Drought\Code\output\cnn\{var}\predictions_{m}_{n}.npy', predictions)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'\nElapsed time: {np.round(elapsed_time/60,2)} minutes')

#%% non-parallel version
start_time = time.time()
n_steps=24; n_pred=12; n_features=5; var='P'                                            # data preparation parameters
epochs=100; verbose=0; complexity='low'                                                 # LSTM parameters

### read, smooth, and normalize data
data_dir = r'C:\Projects\Drought\Texas'
p_data = pd.read_csv(os.path.join(data_dir,'P//1_resampled//1.csv'))
t_data = pd.read_csv(os.path.join(data_dir,'LST//1_resampled//1.csv'))
n_data = pd.read_csv(os.path.join(data_dir,'NDVI//1_resampled//1.csv'))
e_data = pd.read_csv(os.path.join(data_dir,'ET//1_resampled//1.csv'))
s_data = pd.read_csv(os.path.join(data_dir,'SM//1_resampled//1.csv'))

### remove Elnio years before preprocessing
# el_yrs = [2003, 2005, 2007, 2010, 2015, 2016, 2019]
# p_data = sep_elnino(p_data, el_yrs)
# t_data = sep_elnino(t_data, el_yrs)
# n_data = sep_elnino(n_data, el_yrs)
# e_data = sep_elnino(e_data, el_yrs)

p_smoothed = remove_outliers(p_data, z_threshold=5)
t_smoothed = remove_outliers(t_data, z_threshold=5)
n_smoothed = remove_outliers(n_data, z_threshold=5)
e_smoothed = remove_outliers(e_data, z_threshold=5)
s_smoothed = remove_outliers(s_data, z_threshold=5)

p_normalized = normalize(p_smoothed)
t_normalized = normalize(t_smoothed)
n_normalized = normalize(n_smoothed)
e_normalized = normalize(e_smoothed)
s_normalized = normalize(s_smoothed)

m, n = 0, 500
frame_cell = []
for cell in p_normalized.index[m:n]:
    print(f'\nStarted predicting for cell {cell}:')
    X_train, y_train, X_test, y_test = get_traintest(
        p_normalized, t_normalized, n_normalized, e_normalized, s_normalized, cell, n_steps, n_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame_b = []
        for b in tqdm(np.arange(1,17)):
            random.seed(42)
            np.random.seed(42)
            tf.random.set_seed(42)
            
            # define model with low complexity
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
            model.add(Dense(n_pred))
            
            # define model with high complexity
            # model = Sequential()
            # model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
            # model.add(LSTM(units=50, activation='relu', return_sequences=True))
            # model.add(LSTM(units=50, activation='relu'))
            # model.add(Dense(n_pred))
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=100, verbose=0, batch_size=b)                # fit model
            yhat = model.predict(X_test, verbose=0)[-1,:]                                   # demonstrate prediction
            mse = mean_squared_error(y_test[-1,:], yhat)
            nse = r2_score(y_test[-1,:], yhat)
            frame_b.append([cell, b, mse, nse, y_test[-1,:], yhat])
            # print(f'batchsize={b}', f'MSE={np.round(mse,4)}', f'NSE={np.round(nse,4)}')
            
            # Reset the model for the next iteration
            del model
            tf.keras.backend.clear_session()
        frame_cell.append(frame_b)

### separate metrics from predictions
metrics, predictions = [], []
for cell in frame_cell:
    frame_m , frame_p = [], []
    for batch in cell:
        frame_m.append(batch[0:4])
        frame_p.append(batch[4:])
    metrics.append(frame_m)
    predictions.append(frame_p)

### select best batchsize for all cells
frame_best = []
for cell in frame_cell:
    df_cell = pd.DataFrame(cell, columns=['cell','batchsize','mse','nse', 'y_test', 'yhat'])
    best_b = df_cell['nse'].idxmax()
    frame_best.append(df_cell.iloc[best_b,:].values)

### save results
df_p = pd.DataFrame(frame_best, columns=['cell','batchsize','mse','nse', 'y_test', 'yhat'])
df_p.to_csv(f'C:\Projects\Drought\Code\output\cnn\{var}\lstm_{m}_{n}.csv', index=None)   
np.save(f'C:\Projects\Drought\Code\output\cnn\{var}\metrics_{m}_{n}.npy', metrics)
np.save(f'C:\Projects\Drought\Code\output\cnn\{var}\predictions_{m}_{n}.npy', predictions)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'\nElapsed time: {np.round(elapsed_time/60,2)} minutes')

