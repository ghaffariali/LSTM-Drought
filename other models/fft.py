# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:04:30 2023

@author: alg721

this script contains codes for calculating FFT of the dataset.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_file = r'C:\Projects\Drought\Code\data\Texas\P\0_original\1.csv'
output_dir = r'C:\Projects\Drought\Code\output\plots'
data = pd.read_csv(data_file)
coords = data.iloc[:,0:2]                                                               # get coordinations
prcp = data.iloc[:,2:]                                                                  # get precipitation data

# calculate Fourier transform using a cascading window
def casc_fft(df):
    data = df.iloc[:,2:]                                                                    # separate variable data from coordinates
    n_yrs = int(np.shape(data)[1]/12)                                                       # number of years
    n_cells = len(data)                                                                     # number of cells (location tags)
    dt = 1                                                                                  # 1 month sampling rate
    Fs = 1/dt
    fft_cas, f_cas = [], []
    for n in range(n_yrs):
        data_win = data.iloc[:,0:12*(n+1)].values                                           # select data for n_yr
        data_length = np.shape(data_win)[1]
        fft_win = np.zeros((n_cells,data_length//2+1))
        f_win = np.zeros((n_cells,data_length//2+1))
        for i in range(n_cells):
            Y = np.fft.fft(data_win[i, :])
            P2 = np.abs(Y) / data_length
            P1 = P2[:data_length//2 + 1]
            P1[1:-1] = 2 * P1[1:-1]
            # fft.append(P1)
            fft_win[i, :] = P1
            f_win[i, :] = Fs * np.arange(0, (data_length//2) + 1) / data_length
        f_cas.append(f_win)
        fft_cas.append(fft_win)
    return fft_cas, f_cas

fft_cas, f_cas = casc_fft(data)

# calculate Fourier tra'nsform using a rolling window
def roll_fft_yearly(df, n_window):
    data = df.iloc[:,2:]                                                                    # separate variable data from coordinates
    n_yrs = int(np.shape(data)[1]/12)                                                       # number of years
    n_cells = len(data)                                                                     # number of cells (location tags)
    dt = 1                                                                                  # 1 month sampling rate
    Fs = 1/dt
    fft_rol, f_rol = [], []
    for n in range(n_yrs-n_window+1):
        data_win = data.iloc[:,n*12:(n+n_window)*12].values
        data_length = np.shape(data_win)[1]
        fft_win = np.zeros((n_cells,data_length//2+1))
        f_win = np.zeros((n_cells,data_length//2+1))
        for i in range(n_cells):
            Y = np.fft.fft(data_win[i, :])
            P2 = np.abs(Y) / data_length
            P1 = P2[:data_length//2 + 1]
            P1[1:-1] = 2 * P1[1:-1]
            fft_win[i, :] = P1
            f_win[i, :] = Fs * np.arange(0, (data_length//2) + 1) / data_length
        f_rol.append(f_win)
        fft_rol.append(fft_win)
    return fft_rol, f_rol

def roll_fft_monthly(df, n_window):
    data = df.iloc[:,2:]                                                                    # separate variable data from coordinates
    n_cells = len(data)                                                                     # number of cells (location tags)
    dt = 1                                                                                  # 1 month sampling rate
    Fs = 1/dt
    fft_rol, f_rol = [], []
    for n in range(np.shape(data)[1]-n_window*12+1):
        data_win = data.iloc[:,n:n+n_window*12].values
        data_length = np.shape(data_win)[1]
        fft_win = np.zeros((n_cells,data_length//2+1))
        f_win = np.zeros((n_cells,data_length//2+1))
        for i in range(n_cells):
            Y = np.fft.fft(data_win[i, :])
            P2 = np.abs(Y) / data_length
            P1 = P2[:data_length//2 + 1]
            P1[1:-1] = 2 * P1[1:-1]
            fft_win[i, :] = P1
            f_win[i, :] = Fs * np.arange(0, (data_length//2) + 1) / data_length
        f_rol.append(f_win)
        fft_rol.append(fft_win)
    return fft_rol, f_rol

fft_rol_6, f_rol_6 = roll_fft_yearly(data, n_window=6)
fft_rol_1, f_rol_1 = roll_fft_yearly(data, n_window=1)

fft_rol1_mo, f_rol_mo = roll_fft_monthly(data, n_window=6)

#%% plot
def plot_multiple_fft(f, fft, output_dir, n_window, var, idx, n_p):
    fig, ax = plt.subplots(n_p, 1, sharex=True)
    date_range = np.arange(2001,2023)
    for c in range(n_p):
        freq, loc = np.meshgrid(f[c][idx, :], np.arange(1, np.shape(fft[0])[0]+1))
        ax[c].contour(freq, loc, fft[c])
        ax[c].set_xlabel('Frequency (1/month)')
        ax[c].set_ylabel(f'{date_range[c]}-{date_range[c]+n_window-1}',fontsize=9)
        ax[c].yaxis.label.set(rotation='horizontal', ha='right');
        ax[c].plot(freq, fft[c], 'r', linewidth=2)
        ax[c].tick_params(axis='y', labelsize=6)
        fig.legend(['Frequency Contour', f'Line for loc = {idx}'], bbox_to_anchor=[0.675,0.01])
        fig.suptitle(f'n_window={n_window}, variable={var}')
    fig.text(-0.11, 0.5, 'Location tag', va='center', rotation='vertical')
    fig.savefig(os.path.join(output_dir, f'{var}_fft_{n_window}.png'), dpi=300, bbox_inches='tight')
    plt.show()

plot_multiple_fft(f_rol_6, fft_rol_6, output_dir, n_window=6, var='P', idx=500, n_p=17)

def plot_fft_casc(f, fft, var, idx, output_dir):
    fig, ax = plt.subplots(int(len(fft)/2),2,sharex=True)
    for c in range(len(fft)):
        freq, loc = np.meshgrid(f[c][idx,:], np.arange(1, np.shape(fft[c][0]+1)))
        ax[c].contour(freq, loc, fft[c])
        ax[c].set_xlabel('Frequency (1/month)')
        ax[c].set_ylabel()

#%%
var = 'P'
f = f_cas
fft = fft_cas
idx = 500
fig, ax = plt.subplots(int(len(fft)/2),2,sharex=True)
for c in range(0,11):
    freq, loc = np.meshgrid(f[c][idx,:], np.arange(1, np.shape(fft[c])[0]+1))
    ax[c,0].contour(freq, loc, fft[c])
    ax[c,0].set_xlabel('Frequency (1/month)')
    ax[c,0].set_ylabel(f'{int((np.shape(fft[c])[1]-1)*2/12)}')
    ax[c,0].tick_params(axis='y', labelsize=7)
    
for c in range(0,11):
    freq, loc = np.meshgrid(f[c+11][idx,:], np.arange(1, np.shape(fft[c+11])[0]+1))
    ax[c,1].contour(freq, loc, fft[c+11])
    ax[c,1].set_xlabel('Frequency (1/month)')
    ax[c,1].yaxis.set_label_position("right")
    ax[c,1].yaxis.tick_right()
    ax[c,1].set_ylabel(f'{int((np.shape(fft[c+11])[1]-1)*2/12)}')
    ax[c,1].tick_params(axis='y', labelsize=7)

fig.savefig(os.path.join(output_dir, f'{var}_fft_cascading.png'), dpi=300, bbox_inches='tight')

#%%


n_window = 6
### plot two fft values for previous years and the next year
idx = 200
yr1 = 0
yr2 = yr1 + n_window
arr1, arr2 = f_rol_1, f_rol_6
arr3, arr4 = fft_rol_1, fft_rol_6
fig, ax = plt.subplots(2,1,sharex=True)
fig.suptitle(f'FFT plots for cell {idx}')
ax[0].plot(arr1[yr2][idx,:], arr3[yr2][idx,:], color='r', label='Present year')
ax[1].plot(arr2[yr1][idx,:], arr4[yr1][idx,:], color='b', label='Previous years')
ax[1].set_xlabel('Frequency (1/month)')
fig.text(0.01, 0.5, 'FFT', va='center', rotation='vertical')
fig.legend()
plt.show()

#%% plot
yr = 20
var = 'P'
idx = 1000

plt.figure(figsize=(10, 15))
plot_dic = {'P':'Precipitation', 'LST':'Temperature', 'NDVI':'NDVI', 'SM': 'Soil Moisture'}

T = np.arange(1,np.shape(prcp)[1]+1)
plt.subplot(6, 2, 1)
plt.plot(T, prcp.iloc[idx, :])
plt.legend([f'Time domain {plot_dic[var]} of cell={idx}'])
plt.xlabel('Time (month)')
plt.ylabel('Precipitation')

plt.subplot(6, 2, 3)
plt.plot(f_cas[yr][idx,:], fft_cas[yr][idx,:])
plt.legend([f'Frequency domain {plot_dic[var]} of cell={idx}'])
plt.xlabel('Frequency (1/month)')
plt.ylabel('|P1(f)|')

plt.subplot(6, 2, 5)
freq, loc = np.meshgrid(f_cas[yr][idx,:], np.arange(1, np.shape(data)[0]+1))
f_inv = 1.0 / freq
f_inv[:,0] = f_inv[:,1]                                                                 # inf values are replaced with next column
contour_plot = plt.contour(f_inv, loc, fft_cas[yr])
plt.xlabel('Period (month)')
plt.ylabel('Cell')
# loc_idx = fft[yr][idx,:]
# plt.plot(1.0 / freq, fft_cas[yr], 'r', linewidth=2)
# plt.legend(['Period Contour', f'Line for cell={idx}'], bbox_to_anchor=[0.54,0.3])
plt.legend('Period Contour', bbox_to_anchor=[0.54,0.3])

plt.subplot(6, 2, 7)
freq, loc = np.meshgrid(f_cas[yr][idx, :], np.arange(1, np.shape(data)[0]+1))
contour_plot = plt.contour(freq, loc, fft_cas[yr])
plt.xlabel('Frequency (1/month)')
plt.ylabel('Cell')
# loc_idx = fft_results[idx, :]
# plt.plot(freq, fft_cas[yr], 'r', linewidth=2)
# plt.legend(['Frequency Contour', f'Line for cell={idx}'], bbox_to_anchor=[0.54,0.1])
plt.legend('Frequency Contour', bbox_to_anchor=[0.54,0.1])


plt.tight_layout()
plt.savefig(f'C:\Projects\Drought\Code\{var}_fft.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
yr = 20
var = 'P'
idx = 1000
# fig, ax = plt.subplots(figsize=(5, 4))                                                     # Create a figure with one subplot
plt.subplots()
# plt.subplot(figsize=(5,12))
freq, loc = np.meshgrid(f_cas[yr][idx,:], np.arange(1, np.shape(data)[0]+1))
f_inv = 1.0 / freq
f_inv[:,0] = f_inv[:,1]                                                                 # inf values are replaced with next column
contour_plot = plt.contour(f_inv, loc, fft_cas[yr])
plt.xlabel('Recurring interval (month)')
plt.ylabel('Cell')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{var}_fft'), dpi=300, bbox_inches='tight')
plt.show()