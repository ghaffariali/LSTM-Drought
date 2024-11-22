# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:55:17 2023

@author: alg721

THERE IS NO NEED TO RUN THIS CODE IF YOU HAVE THE FILE usdm.npy

this script contains the codes for preparing USDM values for comparison with
CDI. USDM values are converted to tif from shapefiles and then averaged. Final
values are stored in one array named usdm.npy.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import gdal, ogr, osr
import rasterio
from tqdm import tqdm
import sys
wd = r'C:\Projects\RS-Drought\3_Code'
sys.path.append(wd)                                                                             # append working directory to path
from functions import spi_batch, zscore_batch, regrid, crop_batch, get_weights, \
    stack_monthly_batch, cdi_batch

#%%
"""this section converts shapefiles to tif images. in this project, USDM files
are as shapefiles and need to be converted to tif. I wrote this code based on 
this tutorial:
    https://www.youtube.com/watch?v=EHRnB-9fHlg"""
    
def shp2tif_batch(input_dir, output_dir, coord_file, attr, pixel_size):
    print(f'Started converting shapefiles to tiff images for {os.path.basename(input_dir)[:4]}:')
    def shp2tif(input_files, output_dir, bounding_box, n, attr, pixel_size, plot_data=False):
        output_tiff = os.path.join(output_dir, os.path.basename(input_files[n])[:-3] + 'tif')       # output filename
        
        # plot shapefile
        if plot_data:
            shapefile = gpd.read_file(input_files[n])                                               # read input shapefile
            fig, ax = plt.subplots(1, figsize=(8,12))
            shapefile.plot(ax=ax, column='DM', cmap='jet', legend=True)
            plt.xlabel('lon')
            plt.ylabel('lat')
            plt.show()
        
        input_shp = ogr.Open(input_files[n])                                                        # open input shapefile
        source_layer = input_shp.GetLayer()                                                         # get layers
        defn = source_layer.GetLayerDefn()                                                          # get layer definitions
        # get column names
        column_names = []
        for n in range(defn.GetFieldCount()):
            fdefn = defn.GetFieldDefn(n)
            column_names.append(fdefn.name)
        
        # xmin, xmax, ymin, ymax = source_layer.GetExtent()                                           # get shapefile extent;  use this line for the complete maps
        xmin, xmax, ymin, ymax = bounding_box                                                       # bounding box for Texas
        x_res = int((xmax-xmin)/pixel_size)                                                         # get image width
        y_res = int((ymax-ymin)/pixel_size)                                                         # get image height
        target_ds = gdal.GetDriverByName('GTiff').Create(output_tiff, x_res, y_res, 1, 
                                                         gdal.GDT_Float32,['COMPRESS=LZW'])         # create target tiff image
        target_ds.SetGeoTransform((xmin,pixel_size,0.0,ymax,0.0,-pixel_size))                       # set transformation
        srse = osr.SpatialReference()                                                               # get source reference
        proj = 'EPSG:4326'                                                                          # projection: WGS84
        srse.SetWellKnownGeogCS(proj)
        target_ds.SetProjection(srse.ExportToWkt())
        band = target_ds.GetRasterBand(1)
        target_ds.GetRasterBand(1).SetNoDataValue(-9999)                                            # set nan values
        band.Fill(-9999)
        gdal.RasterizeLayer(target_ds, [1], source_layer, None, None, [1], 
                            options=['ALL_TOUCHED=TRUE', f'ATTRIBUTE={attr}'])
        target_ds = None
        
        if plot_data:
            # Open the TIFF file using rasterio
            with rasterio.open(output_tiff) as src:
                tiff_data = src.read()                                                              # Read the image data as a NumPy array
                transform = src.transform                                                           # Get the spatial transformation information
                crs = src.crs                                                                       # Get the coordinate reference system (CRS)        
                metadata = src.meta                                                                 # Get metadata
            
            fig, ax = plt.subplots(1, figsize=(8,12))
            plt.imshow(tiff_data[0,:,:], cmap = 'jet')
            plt.xlabel('lon')
            plt.ylabel('lat')
            plt.show()

    if os.path.exists(output_dir) == False:                                                         # check if output_dir exists; if not, create one.
        os.mkdir(output_dir)
    
    coords = pd.read_csv(coord_file)                                                                # read coordinates as *.csv
    lon = pd.unique(coords['longitude'])                                                            # get longitudes
    lat = pd.unique(coords['latitude'])                                                             # get latitudes
    bounding_box = [lon[0]-pixel_size/2, lon[-1]+pixel_size/2, 
                    lat[0]-pixel_size/2, lat[-1]+pixel_size/2]                                      # get bounding box for the region of interest
    input_files = [f.path for f in os.scandir(input_dir) if f.path.endswith('.shp')]                # list all *.shp files in output_dir
    for n in tqdm(range(len(input_files)), miniters=1):
        shp2tif(input_files, output_dir, bounding_box, n, attr, pixel_size)
    print('Converting shapefiles into tiff images completed.')

shp_dir = r'C:\Projects\RS-Drought\usdm\shp'
tif_dir = r'C:\Projects\RS-Drought\usdm\tif'
coord_file = r'C:\Projects\RS-Drought\3_Code\data\SM\0_original\1.csv'
years = [f.path for f in os.scandir(shp_dir) if f.is_dir()]
for year in years:
    input_dir = year
    output_dir = os.path.join(tif_dir, os.path.basename(year))
    shp2tif_batch(input_dir, output_dir, coord_file, attr='DM', pixel_size=0.25)

#%% resample USDM data into monthly tif images to compare with CDI
def tif_avg(input_dir, output_file):
    tif_dirs = [f.path for f in os.scandir(input_dir) if f.is_dir()]                                # get tif directories
    frame1 = []                                                                                     # create a list to store monthly data
    for tif_dir in tif_dirs:                                                                        # loop over tif_dirs
        print(f'Calculating average for year {os.path.basename(tif_dir)[:4]}...')
        input_files = [f.path for f in os.scandir(tif_dir) if f.path.endswith('.tif')]              # list all *.tif files in output_dir
        months = [int(X[-8:-6]) for X in input_files]                                               # get month index
        df_files = pd.DataFrame([input_files, months], index=['file','month']).T                    # create a df of files and corresponding months
        for m in np.arange(1, 13):                                                                  # loop over months
            month_files = df_files['file'][df_files['month']==m]                                    # select files for month m
            frame2 = []                                                                             # create a list to store weekly data
            for file in month_files:                                                                # loop over month_files
                with rasterio.open(file) as src:                                                
                    tiff_data = src.read()                                                          # Read the image data for one weeek as a NumPy array
                frame2.append(tiff_data[0,:,:])                                                     # append data to frame2
            try:
                stacked_w = np.stack(frame2)                                                        # stack weekly data into one array
                month_arr = np.round(np.mean(stacked_w, axis=0))                                    # calculate monthly average data
                month_arr = np.where(month_arr<0, -9999, month_arr)                                 # set negative values to missing data
                frame1.append(month_arr)                                                            # append monthly data to frame1
            except ValueError:
                print(f'Incomplete data for year {os.path.basename(tif_dir)[:4]}, month {m}.')
    stacked_m = np.stack(frame1)                                                                    # stack monthly data to one array
    np.save(output_file, stacked_m)                                                                 # save array as *.npy
    
    return stacked_m

input_dir = r'C:\Projects\RS-Drought\usdm\tif'
output_file = r'C:\Projects\RS-Drought\usdm\usdm.npy'
usdm = tif_avg(input_dir, output_file)
usdm = usdm[2:-8]                                                                                   # exclude first 2 months due to SPI and incomplete year of 2023

#%% plot the shapefile and the tiff file for one image
# Load shapefile and TIFF file paths
shp_file = r'C:\Projects\RS-Drought\usdm\shp\2001_USDM_M\USDM_20010102.shp'
tif_file = r'C:\Projects\RS-Drought\usdm\tif\2001_USDM_M\USDM_20010102.tif'

# Load shapefile data
shapefile = gpd.read_file(shp_file)
coords = pd.read_csv(coord_file)

# Get unique longitude and latitude values
lon = pd.unique(coords['longitude'])
lat = pd.unique(coords['latitude'])

# Set the plot limits based on the bounding box
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot the shapefile within the bounding box
ax[0].set_xlim(np.min(lon), np.max(lon))
ax[0].set_ylim(np.min(lat), np.max(lat))
shapefile.plot(ax=ax[0], column='DM', cmap='jet', legend=True)
ax[0].set_xlabel('Longitude')
ax[0].set_ylabel('Latitude')
ax[0].set_title('Shapefile Plot')

# Load TIFF data
with rasterio.open(tif_file) as src:
    tiff_data = src.read(1)

# Define a custom colormap for the TIFF data
cmap = plt.get_cmap('jet', 6)
bounds = [-1, 0, 1, 2, 3, 4, 5]
norm = plt.Normalize(bounds[0], bounds[-1])
# cmap = ListedColormap(cmap(norm(i)) for i in bounds)

# Create a plot for the TIFF data with the custom colormap
im = ax[1].imshow(tiff_data, cmap=cmap, interpolation='none', vmin=-1, vmax=4)
cbar = fig.colorbar(im, ax=ax[1], ticks=[-1, 0, 1, 2, 3, 4], orientation='vertical')
cbar.ax.set_yticklabels(['NaN', '0', '1', '2', '3', '4'])
ax[1].set_xlabel('X Coordinate')
ax[1].set_ylabel('Y Coordinate')
ax[1].set_title('TIFF Data Plot')

# Show the combined figure with two subplots
plt.tight_layout()
plt.show()
fig.savefig(r'C:\Projects\RS-Drought\usdm\map.png', dpi=300, bbox_inches='tight')

#%% save USDM to tif files
def save_tiff(grid, coords, output_file, arc_flip=False):
    x = pd.unique(coords['longitude'])                                                          # get longitudes
    y = pd.unique(coords['latitude'])                                                           # get latitudes
    num_rows, num_cols = grid.shape                                                             # get number of rows and columns
    left, bottom = x[0], y[0]
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

input_file = r'C:\Projects\RS-Drought\usdm\usdm.npy'
output_dir = r'C:\Projects\RS-Drought\usdm\monthly'
coords = pd.read_csv(r'C:\Projects\Drought\Code\data\monthly_27830_r1\P\5_coords\1_comb.csv')
start_date, end_date = ['2001-03-01','2022-12-01']
date_range = pd.date_range(start_date, end_date, freq='MS')
usdm = np.load(input_file)[2:-8]
for m in range(len(usdm)):
    output_file = os.path.join(output_dir, date_range[m].strftime('%Y-%m-%d')+'.tif')
    grid = usdm[m,:,:]
    save_tiff(grid, coords, output_file)

#%% calculate CDI for historic values from 2001 to 2020


### regrid input data to remove any mismatch in coordinates
input_files = [r'C:\Projects\RS-Drought\3_Code\data\P\0_original\1.csv',
               r'C:\Projects\RS-Drought\3_Code\data\LST\0_original\1.csv',
               r'C:\Projects\RS-Drought\3_Code\data\NDVI\0_original\1.csv',
               r'C:\Projects\RS-Drought\3_Code\data\SM\0_original\1.csv']
output_files = [r'C:\Projects\RS-Drought\3_Code\data\P\1_resampled\1.csv',
               r'C:\Projects\RS-Drought\3_Code\data\LST\1_resampled\1.csv',
               r'C:\Projects\RS-Drought\3_Code\data\NDVI\1_resampled\1.csv',
               r'C:\Projects\RS-Drought\3_Code\data\SM\1_resampled\1.csv']

regrid(input_files, output_files)

#%% calculate Zscore and SPI
spi_batch(r'C:\Projects\RS-Drought\3_Code\data\P\1_resampled', 
          r'C:\Projects\RS-Drought\3_Code\data\P\2_zscore', thresh=3)
zscore_batch(r'C:\Projects\RS-Drought\3_Code\data\LST\1_resampled', 
             r'C:\Projects\RS-Drought\3_Code\data\LST\2_zscore')
zscore_batch(r'C:\Projects\RS-Drought\3_Code\data\NDVI\1_resampled', 
             r'C:\Projects\RS-Drought\3_Code\data\NDVI\2_zscore')
zscore_batch(r'C:\Projects\RS-Drought\3_Code\data\SM\1_resampled', 
             r'C:\Projects\RS-Drought\3_Code\data\SM\2_zscore')

#%% crop to a desired period
period = ['2001-03-01', '2022-12-01']
crop_batch(r'C:\Projects\RS-Drought\3_Code\data\P\2_zscore', 
             r'C:\Projects\RS-Drought\3_Code\data\P\3_cropped', period)
crop_batch(r'C:\Projects\RS-Drought\3_Code\data\LST\2_zscore', 
             r'C:\Projects\RS-Drought\3_Code\data\LST\3_cropped', period)
crop_batch(r'C:\Projects\RS-Drought\3_Code\data\NDVI\2_zscore', 
             r'C:\Projects\RS-Drought\3_Code\data\NDVI\3_cropped', period)
crop_batch(r'C:\Projects\RS-Drought\3_Code\data\SM\2_zscore', 
             r'C:\Projects\RS-Drought\3_Code\data\SM\3_cropped', period)

#%% get weights using PCA
files_list = [r'C:\Projects\RS-Drought\3_Code\data\P\3_cropped',
              r'C:\Projects\RS-Drought\3_Code\data\LST\3_cropped',
              r'C:\Projects\RS-Drought\3_Code\data\NDVI\3_cropped',
              r'C:\Projects\RS-Drought\3_Code\data\SM\3_cropped']

output_dir = r'C:\Projects\RS-Drought\3_Code\cdi\weights'                                             # output_dir
get_weights(files_list, output_dir)

#%% stack data into 3d arrays
stack_monthly_batch(r'C:\Projects\RS-Drought\3_Code\data\P\3_cropped', 
                    r'C:\Projects\RS-Drought\3_Code\data\P\4_stacked', 
                    r'C:\Projects\RS-Drought\3_Code\data\P\5_coords', thresh=1)
stack_monthly_batch(r'C:\Projects\RS-Drought\3_Code\data\LST\3_cropped', 
                    r'C:\Projects\RS-Drought\3_Code\data\LST\4_stacked', 
                    r'C:\Projects\RS-Drought\3_Code\data\LST\5_coords', thresh=1)
stack_monthly_batch(r'C:\Projects\RS-Drought\3_Code\data\NDVI\3_cropped', 
                    r'C:\Projects\RS-Drought\3_Code\data\NDVI\4_stacked', 
                    r'C:\Projects\RS-Drought\3_Code\data\NDVI\5_coords', thresh=1)
stack_monthly_batch(r'C:\Projects\RS-Drought\3_Code\data\SM\3_cropped', 
                    r'C:\Projects\RS-Drought\3_Code\data\SM\4_stacked', 
                    r'C:\Projects\RS-Drought\3_Code\data\SM\5_coords', thresh=1)

#%% calculate CDI values
input_dir = r'C:\Projects\RS-Drought\3_Code\data'
output_dir = r'C:\Projects\RS-Drought\3_Code\cdi\output'
weights_dir = r'C:\Projects\RS-Drought\3_Code\cdi\weights'

cdi_batch(input_dir, output_dir, weights_dir, thresh=3, w='pca')
cdi_batch(input_dir, output_dir, weights_dir, thresh=3, w='constant')

#%% create a mask for extracting only US
mask_file = r'C:\Projects\RS-Drought\usdm\Texas_State_Boundary\State.tif'
with rasterio.open(mask_file) as src:
    us_mask = src.read(1)


usdm = np.load(r'C:\Projects\RS-Drought\usdm\usdm.npy')[2:-8]
# cdi = np.load(r'C:\Projects\RS-Drought\3_Code\cdi\output\pca\1.npy')
cdi = np.load(r'C:\Projects\RS-Drought\3_Code\cdi\output\constant\1.npy')

for i in range(len(usdm)):
    usdm[i,:,:][us_mask==0] = 99
    cdi[i,:,:][us_mask==0] = 99

# Sample data
discrete_array = usdm
continuous_array = cdi

# Define a mapping from ranges to discrete values
range_to_value = {
    (-1000, -2.0): 4,
    (-2.0, -1.5): 3,
    (-1.5, -1.0): 2,
    (-1.0, -0.5): 1,
    (-0.5, 0.0): 0,
    (0.0, 50): -9999
}

# Create an array to store the discrete values based on ranges
discrete_from_ranges = np.full(continuous_array.shape, 99, dtype=int)

# Map the continuous values to discrete values based on ranges
for (lower, upper), value in range_to_value.items():
    mask = (continuous_array >= lower) & (continuous_array < upper)
    discrete_from_ranges[mask] = value

#%%
import numpy as np
import matplotlib.pyplot as plt
usdm[usdm==99] = -2
usdm[usdm==-9999] = -1
discrete_from_ranges[discrete_from_ranges==99] = -2
discrete_from_ranges[discrete_from_ranges==-9999] = -1

months = pd.date_range('2001-03-01', '2022-12-01', freq='MS')
m = 130
# Sample data (replace with your actual arrays)
array1 = usdm[m,:,:]
array2 = discrete_from_ranges[m,:,:]

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot the first array in the left subplot
cmap1 = plt.get_cmap('jet', 7)  # Choose your colormap
img1 = axes[0].imshow(array1, cmap=cmap1, vmin=-2, vmax=4)
axes[0].set_title(f'USDM {months[m].strftime("%Y-%m")}')  # Set subplot title

# Plot the second array in the right subplot
cmap2 = plt.get_cmap('jet', 7)  # Adjust the colormap as needed
img2 = axes[1].imshow(array2, cmap=cmap2, vmin=-2, vmax=4)  # Set min and max values
axes[1].set_title(f'ECDI {months[m].strftime("%Y-%m")}')  # Set subplot title

# Create colorbars for both subplots
cbar1 = plt.colorbar(img1, ax=axes[0], ticks=[-2, -1, 0, 1, 2, 3, 4])
cbar1.ax.set_yticklabels(['Outside Texas', 'None', 'D0', 'D1', 'D2', 'D3', 'D4'])  # Set colorbar labels for array 1

cbar2 = plt.colorbar(img2, ax=axes[1], ticks=[-2, -1, 0, 1, 2, 3, 4])
cbar2.ax.set_yticklabels(['Outside Texas', '> 0.0', '[-0.5, 0.0]', '[-1.0,-0.5]', '[-1.5,-1.0]', '[-2.0,-1.5]', '<-2.0'])  # Set colorbar labels for array 2

plt.show()

fig.savefig(r'C:\Projects\RS-Drought\usdm\drought_map.png', dpi=300, bbox_inches='tight')

