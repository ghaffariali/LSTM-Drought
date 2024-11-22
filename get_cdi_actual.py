# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:29:57 2023

@author: ghaff
"""
wd = r'C:\Projects\Drought\Code'
import os
import sys
import numpy as np
sys.path.append(wd)
from functions import zscore_batch, spi_batch, stack_monthly_batch, get_weights, \
    cdi_batch, save_tiff_batch

data_dir = os.path.join(wd, 'data\\Texas')
cdi_dir = os.path.join(wd, 'output\\cdi_actual')
if os.path.exists(cdi_dir):
    print('Directory for actual CDI values exists.')
else:
    os.mkdir(cdi_dir)

#%% calculate SPI or Z-Score and store in 2_zscore
for var in ['LST','NDVI','SM']:                                                                 # calculate zscore for LST, NDVI, and SM
    input_dir = os.path.join(data_dir, var, '1_resampled')
    output_dir = os.path.join(data_dir, var, '2_zscore')
    zscore_batch(input_dir, output_dir)

input_dir = os.path.join(data_dir, 'P', '1_resampled')
output_dir = os.path.join(data_dir, 'P', '2_zscore')
spi_batch(input_dir, output_dir, thresh=3)                                                      # calculate SPI for P

#%% stack input files into 3D numpy arrays
for var in ['P','LST','NDVI','SM']:
    input_dir = os.path.join(data_dir, var, '2_zscore')
    coords_dir = os.path.join(data_dir, var, '3_coords')
    output_dir = os.path.join(data_dir, var, '4_stacked')
    stack_monthly_batch(input_dir, output_dir, coords_dir, thresh=3)

#%% calculate weights of each input in each month using PCA
files_list = [os.path.join(data_dir, 'P', '1_resampled'),
              os.path.join(data_dir, 'LST', '1_resampled'),
              os.path.join(data_dir, 'NDVI', '1_resampled'),
              os.path.join(data_dir, 'SM', '1_resampled')]

output_dir = os.path.join(cdi_dir, 'weights')                                                   # weights dir
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
maps_dir = os.path.join(data_dir, 'P', 'spi')
output_dir = os.path.join(maps_dir, 'pca')
save_tiff_batch(input_dir, coords_dir, maps_dir, output_dir, 
                start_date='2001-03-01', end_date='2022-12-01')

#%% create tiff images from monthly USDM values
usdm = np.load(os.path.join(wd, 'usdm\\usdm.npy'))
input_dir = os.path.join(data_dir, 'P\\5_usdm')
if os.path.exists(input_dir)==False:
    os.mkdir(input_dir)
np.save(os.path.join(input_dir, '1.npy'), usdm[2:264,:,:])                                       # save USDM data from 2001 to 2022
coords_dir = os.path.join(wd, 'data\\Texas\\LST\\3_coords')
maps_dir = os.path.join(data_dir, 'P', 'usdm')
output_dir = os.path.join(maps_dir, 'pca')
save_tiff_batch(input_dir, coords_dir, maps_dir, output_dir, 
                start_date='2001-03-01', end_date='2022-12-01')
