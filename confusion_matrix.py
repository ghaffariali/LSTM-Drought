# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:08:02 2024

@author: ghaff
"""

import os
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import rasterio
wd = r'C:\Projects\Drought\Code'

cdi_path = r'C:\Projects\Drought\Code\output\cdi_predicted\output\pca\1.npy'
usdm_path = r'C:\Projects\Drought\Code\data\Texas\P\5_usdm\1.npy'
spi_path = r'C:\Projects\Drought\Code\data\Texas\P\4_stacked\1.npy'

cdi_arr = np.load(cdi_path)
usdm_arr = np.load(usdm_path)
spi_arr = np.load(spi_path)



def discretize_array(array, ranges):
    ranges = [(-np.inf, ranges[0]), (ranges[0], ranges[1]), (ranges[1], ranges[2]), 
              (ranges[2], ranges[3]), (ranges[3], ranges[4]), (ranges[4], np.inf)]              # define range

    labels = [4, 3, 2, 1, 0, -9999]                                                             # define labels
    discretized_array = np.empty(array.shape, dtype=array.dtype)                                # initialize an array to store discretized values
    for r, l in zip(ranges, labels):                                                            # Iterate over each range and label
        mask = (r[0] <= array) & (array < r[1])                                                 # Create a boolean mask for values within the current range
        discretized_array[mask] = l                                                             # Assign the corresponding label to values within the range
    return discretized_array

from sklearn.metrics import confusion_matrix

def compare_arrays(array1, array2):
    labels = [4, 3, 2, 1, 0, -9999]                                                             # define labels
    confusion_mat = confusion_matrix(array1.flatten(), array2.flatten(), labels=labels)         # Create confusion matrix
    return confusion_mat

def predict_metrics(confusion_matrix, drop_nan=True):
    warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
    # confusion_matrix = confusion_matrix[:-1,:-1]                                                # drop last row and column from confusion matrix
    num_classes = confusion_matrix.shape[0]
    accuracy = np.diag(confusion_matrix).sum() / confusion_matrix.sum()

    precision = []
    recall = []
    f1 = []
    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP
        TN = confusion_matrix.sum() - TP - FP - FN
        
        # Calculate precision, recall, and F1 for each class
        precision.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
        recall.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0)
    
    # Average precision, recall, and F1 across all classes
    precision_avg = sum(precision) / num_classes
    recall_avg = sum(recall) / num_classes
    f1_avg = sum(f1) / num_classes
    
    return [accuracy, precision_avg, recall_avg, f1_avg]

def generate_ranges(t, n, mean, std, upper_limit):
    np.random.seed(42)
    rand_numbers = np.zeros((t,n))
    for i in range(t):
        numbers = np.random.normal(mean, std, n)  # Generate n random numbers from a normal distribution
        numbers = np.maximum(numbers, 0)  # Ensure all numbers are positive
        while sum(numbers) >= upper_limit:  # Check if the sum exceeds the upper limit
            numbers = np.random.normal(mean, std, n)  # Regenerate the numbers if the condition is not met
            numbers = np.maximum(numbers, 0)  # Ensure all numbers are positive
        rand_numbers[i,:] =numbers
    ranges = -2 + np.cumsum(rand_numbers, axis=1)
    return ranges

def calibrate_index(t, ranges, cdi_arr, usdm_disc):
    cdi_disc = discretize_array(cdi_arr[:-12,:,:], ranges[t,:])                                      # discritize CDI
    frame_j = []
    for j in range(np.shape(usdm_disc)[0]):
        conf_mat = compare_arrays(usdm_disc[j,:,:], cdi_disc[j,:,:])                                # calculate confusion matrix
        metrics = predict_metrics(conf_mat)                                                         # calculate metrics
        frame_j.append(metrics)
    df_metrics = pd.DataFrame(frame_j, columns=['accuracy', 'precision_avg', 'recall_avg', 'f1_avg'])
    mean_metrics = np.insert(df_metrics.mean().values, 0, t)
    return mean_metrics

def mask_roi():
    boundary_img = os.path.join(wd, 'data\\Texas_State_Boundary\\State.tif')                    # boundary tiff file
    with rasterio.open(boundary_img) as src:                                                    # Open the TIFF file using rasterio
        bnd_array = src.read(1)                                                                 # Read the image as a NumPy array
        # bnd_array[0,:,:] = np.flipud(bnd_array[0,:,:])                                          # flip array upside down for plotting
        # Get image metadata for setting axis labels
        # transform = src.transform
        # bounds = src.bounds
        # width, height = src.width, src.height
        # extent_bnd = [bounds.left, bounds.right, bounds.bottom, bounds.top]                     # Define the extent of the image in terms of lon and lat
    
    ### fill 0 values in the US with 1
    last_ones_indices = np.argmax(bnd_array[::-1] == 1, axis=0)                                 # Find the indices of the last occurrence of value 1 in each column
    mask = np.zeros_like(bnd_array, dtype=bool)                                                 # create a boolean mask to replace 0s with 1s
    for col, last_idx in enumerate(last_ones_indices):
        mask[-last_idx-1:, col] = True
    mask_array = np.where(mask, bnd_array, 1)                                                   # Replace the elements in the original array with 1 where the mask is True
    return mask_array

#%% calibrate CDI model
usdm_disc = usdm_arr[:-12,:,:]
ranges = generate_ranges(1000, 5, 0.5, 0.1, 2.5)
frame_t = []
parallel = True
if parallel:
    results = Parallel(n_jobs=-1)\
        (delayed(calibrate_index)(t, ranges, cdi_arr, usdm_disc)
         for t in tqdm(range(len(ranges))))
else:
    results = []
    for t in tqdm(range(len(ranges))):
        metrics = calibrate_index(t, ranges, cdi_arr, usdm_disc)
        results.append(metrics)

#%% select ranges from calibration
df_results = pd.DataFrame(results, columns=['t', 'accuracy', 'precision_avg', 'recall_avg', 'f1_avg'])
df_results.to_csv(r'C:\Projects\Drought\Code\output\results_cdi.csv', index=None)
t = int(df_results['t'][df_results['accuracy'].idxmax()])
cdi_disc = discretize_array(cdi_arr[-12:,:,:], ranges[t,:])                                      # discritize CDI
usdm_disc = usdm_arr[-12:,:,:]
frame_j = []
for j in range(np.shape(usdm_disc)[0]):
    conf_mat = compare_arrays(usdm_disc[j,:,:], cdi_disc[j,:,:])                                # calculate confusion matrix
    metrics = predict_metrics(conf_mat)                                                         # calculate metrics
    frame_j.append(metrics)
df_metrics = pd.DataFrame(frame_j, columns=['accuracy', 'precision_avg', 'recall_avg', 'f1_avg'])
mean_metrics = np.insert(df_metrics.mean().values, 0, t)

#%% select subjective ranges
mask_array = mask_roi()  

# ranges_cdi = [-1.5, -1, -0.5, 0, 0.5]
ranges_cdi = [-1.55, -1.06, -0.65, -0.2, 0.2]
ranges_spi = [-2, -1.6, -1.3, -0.8, -0.5]
cdi_disc = discretize_array(cdi_arr[-12:,:,:], ranges_cdi)                                      # discritize CDI
spi_disc = discretize_array(spi_arr[-12:,:,:], ranges_spi)                                      # discritize SPI

cdi_filtered = np.where(mask_array==0, -9999, cdi_disc)                                 # mask CDI to remove Mexico and Gulf of Mexico from the map
spi_filtered = np.where(mask_array==0, -9999, spi_disc)                                 # mask SPI to remove Mexico and Gulf of Mexico from the map

usdm_disc = usdm_arr[-12:,:,:]
frame_cdi, frame_spi = [], []
for j in range(np.shape(usdm_disc)[0]):
    # conf_mat_cdi = compare_arrays(usdm_disc[j,:,:], cdi_disc[j,:,:])                                # calculate confusion matrix
    # conf_mat_spi = compare_arrays(usdm_disc[j,:,:], spi_disc[j,:,:])                                # calculate confusion matrix
    conf_mat_cdi = compare_arrays(usdm_disc[j,:,:], cdi_filtered[j,:,:])                                # calculate confusion matrix
    conf_mat_spi = compare_arrays(usdm_disc[j,:,:], spi_filtered[j,:,:])                                # calculate confusion matrix
    
    metrics_cdi = predict_metrics(conf_mat_cdi)                                                         # calculate metrics
    metrics_spi = predict_metrics(conf_mat_spi)                                                         # calculate metrics

    frame_cdi.append(metrics_cdi)
    frame_spi.append(metrics_spi)
df_metrics_cdi = pd.DataFrame(frame_cdi, columns=['Accuracy', 'Precision', 'Recall', 'F1'])
df_metrics_spi = pd.DataFrame(frame_spi, columns=['Accuracy', 'Precision', 'Recall', 'F1'])
print('CDI results', df_metrics_cdi.mean())
print('SPI results', df_metrics_spi.mean())

#%%
import matplotlib.pyplot as plt
output_dir = os.path.join(wd, 'output//plots')
# Sample data
for metric in df_metrics_cdi.columns:
    array1 = df_metrics_cdi[metric]
    array2 = df_metrics_spi[metric]
    labels = [f'2022-{m}' for m in np.arange(1,13)]
    
    # Plotting
    # Generate positions for the bars
    x = np.arange(len(labels))
    bar_width = 0.35
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(x - bar_width/2, array1, color='blue', width=bar_width, label='ECDI vs USDM')
    plt.bar(x + bar_width/2, array2, color='red', width=bar_width, label='SPI vs USDM')
    plt.xlabel('month', fontsize=16)
    plt.ylabel(f'{metric}', fontsize=16)
    # plt.title('Comparison of Values between Array 1 and Array 2')
    plt.xticks(x, labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig(os.path.join(output_dir, f'es_{metric}.png'), bbox_inches='tight', dpi=300)
    plt.show()











