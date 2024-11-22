# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:43:44 2023

@author: alg721

this script does the following:
    1- plot input data for one cell from one variable
    2- seasonal trend decomposition for one cell and plot the result
"""
# import libraries and packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
wd = r'C:\Projects\Drought\Code'
output_dir = os.path.join(wd, 'output\\plots')
data_dir = os.path.join(wd, 'data\\Texas')

# =============================================================================
# plot data for one cell: original, outliers removed, smoothed
# =============================================================================
from scipy.ndimage import gaussian_filter1d
from matplotlib.dates import DateFormatter
def plot_data(cell, data, thresh, sigma=1.0, show_plot=False):
    row = np.array(data.iloc[cell,2:])                                                          # select data for one cell
    z_scores = (row - row.mean()) / row.std()                                                   # calculate z-score values
    outliers = np.where(np.abs(z_scores) > thresh)[0]                                           # Identify outliers using z-score values
    row_without_outliers = row.copy()
    row_without_outliers[outliers] = np.nan                                                     # Remove outliers from the row
    interpolated_row = pd.Series(row_without_outliers).interpolate()                            # Interpolate missing values in the row
    smoothed_row = gaussian_filter1d(interpolated_row, sigma)                                   # Smooth the row using Gaussian smoothing
    df_monthly = pd.DataFrame([row, interpolated_row, smoothed_row],
                              index=['Original','Outliers removed', 'Smoothed'],
                              columns=pd.to_datetime(data.columns[2:])).T                       # store values as a df
    
    if show_plot:                                                                               # Plot the original, outlier-removed, and smoothed data
        fig, ax = plt.subplots()
        ax.plot(df_monthly['Original'], label='Original', color='b')
        ax.plot(df_monthly['Outliers removed'], label='Outliers removed', color='y')
        ax.plot(df_monthly['Smoothed'], label='Smoothed', color='r')
        date_format = DateFormatter('%Y')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_title(f'Monthly averaged values for cell {cell}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Precipitation (mm/hr)')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.85,0.1), ncol=3, 
                   borderaxespad=3, fontsize=7.5)
        fig.tight_layout()
        plt.show()
    
# =============================================================================
# seasonal trend decomposition
# =============================================================================
from statsmodels.tsa.seasonal import STL
def extract_seasonality(cell, data, plot_results=False):
    cell_data = data.iloc[cell, 2:]                                                             # select data for cell
    result = STL(endog=cell_data, period=12, seasonal=13, robust=True).fit()                    # fit data for seasonal analysis
    seasonal, trend, resid = result.seasonal, result.trend, result.resid

    if plot_results:
        fig, axs = plt.subplots(nrows=4, sharex=True, figsize=(15,10))
        axs[0].plot(cell_data)
        axs[0].set_title('Original')
        axs[1].plot(seasonal)
        axs[1].set_title('Seasonal')
        axs[2].plot(trend)
        axs[2].set_title('Trend')
        axs[3].plot(resid)
        axs[3].set_title('Residual')
    
        # correct x-ticks and x-labels
        x_values = cell_data.index
        xtick_positions = x_values[::12]                                                        # Select every 12th x value
        first_yr, last_yr = pd.to_datetime(x_values[0]).year, pd.to_datetime(x_values[-1]).year # get first and last year
        xtick_labels = [str(year) for year in range(first_yr, last_yr+1, 1)]                    # Labels for years
        axs[-1].set_xticks(xtick_positions)                                                     # Apply x-tick positions and labels to the last subplot
        axs[-1].set_xticklabels(xtick_labels, rotation=45, ha='right')                          # Rotate for readability
        plt.show()
    return seasonal, trend, resid

#%% plot the process of training LSTM
p_data = pd.read_csv(os.path.join(data_dir, 'P\\1_resampled\\1.csv'))
cell=1200; m=24
fig, ax = plt.subplots(4,1)
for a in ax:
    a.set_ylim(-0.1, 0.4)  # Adjust the limits according to your data
    a.set_yticks([])
    a.set_yticklabels([])
    
ax[0].plot(range(m), p_data.iloc[cell, 2:2+m], color='b')
ax[0].plot(range(m-1,m+12-1), p_data.iloc[cell, 2+m-1:2+m-1+12], color='r')
ax[0].axvline(x=23, color='black', linestyle='--', linewidth=3)
ax[0].set_xticks([0, 23, 35])                                                                   # Specify the indices for ticks
ax[0].set_xticklabels([X[0:7] for X in p_data.columns[[2, 25, 37]]])                            # Specify the labels for ticks
ax[0].tick_params(axis='both', which='major', labelsize=14)


ax[1].plot(range(m), p_data.iloc[cell, 3:3+m], color='b')
ax[1].plot(range(m-1,m+12-1), p_data.iloc[cell, 3+m-1:3+m-1+12], color='r')
ax[1].axvline(x=23, color='black', linestyle='--', linewidth=3)
ax[1].set_xticks([0, 23, 35])                                                                   # Specify the indices for ticks
ax[1].set_xticklabels([X[0:7] for X in p_data.columns[[3, 26, 38]]])                            # Specify the labels for ticks
ax[1].tick_params(axis='both', which='major', labelsize=14)


ax[2].plot(range(m), p_data.iloc[cell, -24-m:-24], color='b')
ax[2].plot(range(m-1,m+12-1), p_data.iloc[cell, -24:-12], color='r')
ax[2].axvline(x=23, color='black', linestyle='--', linewidth=3)
ax[2].set_xticks([0, 23, 35])                                                                   # Specify the indices for ticks
ax[2].set_xticklabels([X[0:7] for X in p_data.columns[[-48, -25, -13]]])                            # Specify the labels for ticks
ax[2].tick_params(axis='both', which='major', labelsize=14)

ax[3].plot(range(m), p_data.iloc[cell, -12-m:-12], color='b')
ax[3].plot(range(m-1,m+12-1), p_data.iloc[cell, -12:], color='r')
ax[3].axvline(x=23, color='black', linestyle='--', linewidth=3)
ax[3].set_xticks([0, 23, 35])                                                                   # Specify the indices for ticks
ax[3].set_xticklabels([X[0:7] for X in p_data.columns[[-36, -13, -1]]])                            # Specify the labels for ticks
ax[3].tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'lstm_process.png'), bbox_inches='tight', dpi=300)
plt.show()

#%% plot time series of all four variables for two cells
p_data = pd.read_csv(os.path.join(data_dir,'P//1_resampled//1.csv'))
t_data = pd.read_csv(os.path.join(data_dir,'LST//1_resampled//1.csv'))
t_data.iloc[:,2:] = t_data.iloc[:,2:]*0.02-273.15
n_data = pd.read_csv(os.path.join(data_dir,'NDVI//1_resampled//1.csv'))
e_data = pd.read_csv(os.path.join(data_dir,'ET//1_resampled//1.csv'))
s_data = pd.read_csv(os.path.join(data_dir,'SM//1_resampled//1.csv'))

cells = [1773, 2027]
fig, ax = plt.subplots(4,2, sharex=True)
fig.suptitle('Time series of climatic variables for two cells')
ax[0,0].set_title(f'Cell {cells[0]}')
ax[0,0].plot(p_data.columns[2:].values, p_data.iloc[cells[0],2:].values, color='b', label='P')
ax[0,0].set_ylim(0,0.5)
ax[0,0].text(-0.2, 0.5, 'P', rotation='vertical', va='center', ha='center', transform=ax[0,0].transAxes)
# ax[0,0].set_ylabel('Prec.')
ax[1,0].plot(t_data.columns[2:].values, t_data.iloc[cells[0],2:].values, color='r', label='T')
ax[1,0].set_ylim(15,50)
ax[1,0].text(-0.2, 0.5, 'T', rotation='vertical', va='center', ha='center', transform=ax[1,0].transAxes)
# ax[1,0].set_ylabel('Temp.')
ax[2,0].plot(n_data.columns[2:].values, n_data.iloc[cells[0],2:].values*0.01, color='g', label='NDVI')
ax[2,0].set_ylim(10,100)
ax[2,0].text(-0.2, 0.5, 'NDVI', rotation='vertical', va='center', ha='center', transform=ax[2,0].transAxes)
# ax[2,0].set_ylabel('NDVI')
ax[3,0].plot(s_data.columns[2:].values, s_data.iloc[cells[0],2:].values, color='brown', label='SM')
ax[3,0].set_ylim(50,200)
ax[3,0].text(-0.2, 0.5, 'SM', rotation='vertical', va='center', ha='center', transform=ax[3,0].transAxes)
# ax[3,0].set_ylabel('Soil moisture')
ax[3,0].set_xlabel('Year')
ax[3,0].set_xticklabels(p_data.columns[::24])

ax[0,1].set_title(f'Cell {cells[1]}')
ax[0,1].plot(p_data.columns[2:].values, p_data.iloc[cells[1],2:].values, color='b', label='P')
ax[0,1].set_ylim(0,0.5)
# ax[0,1].set_ylabel('Precipitation')
ax[1,1].plot(t_data.columns[2:].values, t_data.iloc[cells[1],2:].values, color='r', label='T')
ax[1,1].set_ylim(15,50)
# ax[0,1].set_ylabel('Precipitation')
ax[2,1].plot(n_data.columns[2:].values, n_data.iloc[cells[1],2:].values*0.01, color='g', label='NDVI')
ax[2,1].set_ylim(10,100)
# ax[0,1].set_ylabel('Precipitation')
ax[3,1].plot(s_data.columns[2:].values, s_data.iloc[cells[1],2:].values, color='brown', label='SM')
# ax[0,1].set_ylabel('Precipitation')
ax[3,1].set_ylim(50,200)
ax[3,1].set_xlabel('Year')

for i in range(4):
    for j in range(2):
        ax[i,j].set_xticks([])
        ax[i,j].set_xticklabels([])
        ax[i,1].set_yticklabels([])

for j in range(2):
    ax[3,j].set_xticks(np.arange(0, len(p_data.columns[2:]), 48))
    ax[3,j].set_xticklabels([X[:4] for X in p_data.columns[2:][::48]], rotation=90)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, f'variables{cells[0]}.png'), bbox_inches='tight', dpi=300)
plt.show()


