# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:47:17 2024

@author: download input data for the region of interest.
"""
wd = r'C:\Projects\Drought\Code'
import os
import sys
import numpy as np
from functions import download_data, regrid
import random

sys.path.append(wd)                                                                         # append working directory to path
random.seed(42)
np.random.seed(42)

#%% 1- download monthly data at scale=27830m and store in 0_original
data_file = os.path.join(wd, 'RS_data.csv')                                                 # data_file contains details for downloading data
data_dir = os.path.join(wd, 'data\Texas')                                                   # downloaded data will be saved to data_dir
bounding_box = [-106.6468, 25.8371, -93.5083, 36.5007]                                      # bounding box for the region of interest
if not os.path.exists(data_dir):
    print('Data folder does not exist')
else:
    print('Data directory exists. Downloading data ...')
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