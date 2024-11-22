# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:07:34 2023

@author: aligh

this code does the following:
    1- save tif image from a column in a df
    2- plot the saved tif image
"""
# import libraries and define directories
import os
import sys
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from rasterio.plot import show
from geo_northarrow import add_north_arrow
from shapely.geometry.point import Point
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FixedLocator
wd = r'C:\Projects\Drought\Code'
output_dir = r'C:\Projects\Drought\Code\output\plots'
coords = pd.read_csv(os.path.join(wd, 'data\\Texas\\P\\1_resampled\\1.csv'))
sys.path.append(wd)

# =============================================================================
# save data as tif image
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
            # dtype=np.int32,
            crs = 'EPSG:4326',
            transform=transform,
    ) as dst:
        dst.write(grid, 1)
    
    # add coordinate arrays as metadata
    with rasterio.open(output_file, mode='r+') as dst:
        dst.update_tags(1, x=x.tolist(), y=y.tolist())

# =============================================================================
# draw boundary shapefile on plot axs
# =============================================================================
import geopandas as gpd
def plot_boundary(axs):
    # plot boundary shapefile
    shapefile_path = os.path.join(wd, 'data\\Texas_State_Boundary\\State.shp')                  # Path to your shapefile
    gdf = gpd.read_file(shapefile_path)                                                         # Create a GeoDataFrame from the shapefile
    gdf_extent = gdf.total_bounds                                                               # Get the actual data extent from the GeoDataFrame
    try:                                                                                        # more than one axis
        for j in range(len(axs)):
            gdf.plot(ax=axs[j], color='none', edgecolor='black', linewidth=4)                   # Plot the shapefile on the same axis
            # Set the extent of the plot to match the actual data extent
            axs[j].set_xlim(gdf_extent[0], gdf_extent[2])
            axs[j].set_ylim(gdf_extent[1], gdf_extent[3])
    except TypeError:                                                                           # just one axis
        gdf.plot(ax=axs, color='none', edgecolor='black', linewidth=4)                          # Plot the shapefile on the same axis
        axs.set_xlim(gdf_extent[0], gdf_extent[2])
        axs.set_ylim(gdf_extent[1], gdf_extent[3])

def plot_scalebar(axs, size="large"):
    """tutorial:
        https://geopandas.org/en/stable/gallery/matplotlib_scalebar.html"""
    points = gpd.GeoSeries([Point(-106, 26), Point(-105, 26)], crs=4326)                        # Geographic WGS 84 - degrees
    points = points.to_crs(32619)                                                               # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])                                             # calculate distance in m
    try:
        for j in range(len(axs)):
            axs[j].add_artist(ScaleBar(distance_meters, location='lower right', 
                                       font_properties={"size": size,}, 
                                       scale_formatter=lambda value, unit: f"> {value} {unit} <",))
    except TypeError:
        axs.add_artist(ScaleBar(distance_meters, location='lower right', 
                                   font_properties={"size": size,}, 
                                   scale_formatter=lambda value, unit: f"> {value} {unit} <",))
        
def plot_northarrow(axs, scale=0.75, ylim_pos=0.14):
    try:
        for j in range(len(axs)):
            add_north_arrow(axs[j], scale=.75, xlim_pos=.92, ylim_pos=ylim_pos, color='#000', text_scaler=3, text_yT=-1.25)
    except TypeError:
        add_north_arrow(axs, scale=scale, xlim_pos=.92, ylim_pos=ylim_pos, color='#000', text_scaler=3, text_yT=-1.25)

#%% plot land use map for 2022
df_map = pd.read_csv(os.path.join(wd, 'data\\Texas\\LULC\\0_original\\1.csv'))                  # read landuse data
output_file = os.path.join(output_dir, 'landuse.tif')                                           # create output name
save_tiff(df_map['2022-01-01'], df_map, output_file, arc_flip=True)                             # save as tif

# Define land use labels for values 1 to 17
landuse_labels = {1: 'Evergreen Needleleaf Forests', 
                  2: 'Evergreen Broadleaf Forests', 
                  3: 'Deciduous Needleleaf Forests', 
                  4: 'Deciduous Broadleaf Forests', 
                  5: 'Mixed Forests', 
                  6: 'Closed Shrublands', 
                  7: 'Open Shrublands', 
                  8: 'Woody Savannas', 
                  9: 'Savannas', 
                  10: 'Grasslands', 
                  11: 'Permanent Wetlands', 
                  12: 'Croplands', 
                  13: 'Urban and Built-up Lands', 
                  14: 'Cropland/Natural Vegetation Mosaics', 
                  15: 'Permanent Snow and Ice', 
                  16: 'Barren', 
                  17: 'Water Bodies'}
landuse_cmap = ListedColormap(['#05450a', '#086a10', '#54a708', '#78d203', '#009900', '#c6b044', '#dcd159',
                               '#dade48', '#fbff13', '#b6ff05', '#27ff87', '#c24f44', '#a5a5a5', '#ff6d4c', 
                               '#69fff8', '#f9ffa4', '#1c0dff'])

with rasterio.open(output_file) as src:                                                         # Open the TIFF file using rasterio
    img_array = src.read(1)                                                                     # Read the image as a NumPy array
    # img_array[0,:,:] = np.flipud(landuse_array[0,:,:])                                          # flip array upside down for plotting
    transform = src.transform
    bounds = src.bounds
    width, height = src.width, src.height
    extent = [bounds.left, bounds.right, bounds.top, bounds.bottom]                             # Define the extent of the image in terms of lon and lat

fig, ax = plt.subplots(figsize=(8, 8))                                                          # Create a figure with one subplot
im = ax.imshow(img_array, cmap=landuse_cmap, extent=extent, aspect='auto', vmin=1, vmax=17)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=-0.3)
cbar = fig.colorbar(im, cax=cax, ticks=range(1, 18))                                            # Add a colorbar to the figure
cbar.ax.tick_params(labelsize=16)                                                               # Set font size for colorbar tick labels
cbar.set_ticklabels([landuse_labels[i] for i in range(1, 18)])                                  # Set custom tick labels for the colorbar
ax.set_xlabel('Longitude', fontsize=16)
ax.set_ylabel('Latitude', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
plot_boundary(ax)                                                                               # plot boundary shapefile
add_north_arrow(ax, scale=.75, xlim_pos=.92, ylim_pos=.1, color='#000', text_scaler=3, text_yT=-1.25)
plot_scalebar(ax)
fig.savefig(os.path.join(output_dir, 'landuse_2022.png'), bbox_inches='tight', dpi=300)
plt.show()

#%% plot mean precipitation map for 2022
df_map = pd.read_csv(os.path.join(wd, 'data\\Texas\\P\\0_original\\1.csv'))                     # read precipitation data
df_map['mean'] = df_map.iloc[:,-12:].mean(axis=1)                                               # calculate average of last 12 months
output_file = os.path.join(output_dir, 'mean_prec.tif')                                         # create output name
save_tiff(df_map['mean'], df_map, output_file, arc_flip=True)                                   # save as tif

with rasterio.open(output_file) as src:                                                         # Open the TIFF file using rasterio
    img_array = src.read(1)                                                                     # Read the image as a NumPy array
    # img_array[0,:,:] = np.flipud(img_array[0,:,:])                                              # flip array upside down for plotting
    transform = src.transform
    bounds = src.bounds
    width, height = src.width, src.height
    extent = [bounds.left, bounds.right, bounds.top, bounds.bottom]                             # Define the extent of the image in terms of lon and lat
    projection = src.crs

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(img_array, cmap='Blues', extent=extent, aspect='auto', vmin=0, vmax=0.3, rasterized=True)
ax.set_xlabel('Longitude', fontsize=16)
ax.set_ylabel('Latitude', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=-0.3)
cbar = fig.colorbar(im, cax=cax)                                                                # Add a colorbar to the figure
cbar.ax.tick_params(labelsize=16)                                                               # Set font size for colorbar tick labels
cbar.set_label('Precipitation', fontsize=16)
plot_northarrow(ax)
plot_scalebar(ax)
plot_boundary(ax)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'mean_P_2022.png'), bbox_inches='tight', dpi=300, pad_inches=0.1)
plt.show()

#%%
df_map = pd.read_csv(os.path.join(wd, 'output\\sarima\\p_sarima.csv'))                          # read SARIMA output for prediction of precipitation
output_file = os.path.join(output_dir, 'nse_sarima.tif')                                        # create output name
save_tiff(df_map['nse'], coords, output_file, arc_flip=True)                                    # save as tif

#%% plot CDI/SPI for pairs of two months
index = 'cdi'
cdi_dir = os.path.join(wd, 'output\\cdi_predicted\maps\\pca\\1')                          # predicted CDI values using SPI for P
# cdi_dir = os.path.join(wd, 'output\\cdi_predicted_allzscore\\maps\\pca\\1                       # predicted CDI values using Z-Score for P
spi_dir = os.path.join(wd, 'data\\Texas\\P\\spi\\pca\\1')                                       # actual SPI values based on actual P

if index=='spi':
    index_dir = spi_dir
else:
    index_dir = cdi_dir
months = ['07','10']
file1 = f'2022-{months[0]}-01.tif'
file2 = f'2022-{months[1]}-01.tif'

cdi_img = os.path.join(index_dir, file1)
spi_img = os.path.join(index_dir, file2)

with rasterio.open(cdi_img) as src:                                                         # Open the TIFF file using rasterio
    cdi_array = src.read()                                                                  # Read the image as a NumPy array
    cdi_array[0,:,:] = np.flipud(cdi_array[0,:,:])                                          # flip array upside down for plotting
    # Get image metadata for setting axis labels
    transform = src.transform
    bounds = src.bounds
    width, height = src.width, src.height
    extent_cdi = [bounds.left, bounds.right, bounds.bottom, bounds.top]                     # Define the extent of the image in terms of lon and lat

with rasterio.open(spi_img) as src:                                                         # Open the TIFF file using rasterio
    spi_array = src.read()                                                                  # Read the image as a NumPy array
    spi_array[0,:,:] = np.flipud(spi_array[0,:,:])                                          # flip array upside down for plotting
    # Get image metadata for setting axis labels
    transform = src.transform
    bounds = src.bounds
    width, height = src.width, src.height
    extent_spi = [bounds.left, bounds.right, bounds.bottom, bounds.top]                     # Define the extent of the image in terms of lon and lat

# Create a figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.075]})
# fig.suptitle(f'type: {v}; date: {file[:7]}', fontsize=18)
im_left = axs[0].imshow(cdi_array[0, :, :], cmap='RdBu', extent=extent_cdi,             
                        aspect='auto', vmin=-2, vmax=2)                                     # Plot the first image on the left subplot
axs[0].set_xlabel('Longitude', fontsize=18) 
axs[0].set_ylabel('Latitude', fontsize=18)
axs[0].set_title(f'{file1[:7]}', fontsize=16)
axs[0].tick_params(axis='both', which='major', labelsize=14)

im_right = axs[1].imshow(spi_array[0, :, :], cmap='RdBu', extent=extent_spi, 
                         aspect='auto', vmin=-2, vmax=2)                                    # Plot the second image on the right subplot
axs[1].set_xlabel('Longitude', fontsize=18)
# axs[1].set_ylabel('Latitude', fontsize=18)
axs[1].set_title(f'{file2[:7]}', fontsize=16)
axs[1].tick_params(axis='x', which='major', labelsize=14)
axs[1].set_yticklabels([])

# Create a common colorbar for both subplots
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(im_right, cax=cax)
# cbar.ax.set_ylabel('Values', fontsize=14)
cbar.ax.tick_params(labelsize=14)  # Set font size for colorbar tick labels
ticks = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
cbar.set_ticks(ticks)
cbar.set_ticklabels(['≤ -2.0', '-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5', '≥ 2.0'])
cbar.locator = FixedLocator(ticks)
plot_boundary(axs)
plot_northarrow(axs)
plot_scalebar(axs)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, f'{index}_{file1[:-4]}_{file2[:-4]}.png'), 
            bbox_inches='tight', dpi=300)
plt.show()


#%% plot CDI and SPI
cdi_dir = os.path.join(wd, 'output\\cdi_predicted\\maps\\pca\\1')                          # predicted CDI values using SPI for P
# cdi_dir = os.path.join(wd, 'output\\cdi_predicted_allzscore\\maps\\pca\\1                       # predicted CDI values using Z-Score for P
spi_dir = os.path.join(wd, 'data\\Texas\\P\\spi\\pca\\1')                                       # actual SPI values based on actual P

months = [str(i).zfill(2) for i in range(1, 13)]
for m in months:
    file = f'2022-{m}-01.tif'; v='predicted'
    cdi_img = os.path.join(cdi_dir, file)
    spi_img = os.path.join(spi_dir, file)
    
    with rasterio.open(cdi_img) as src:                                                         # Open the TIFF file using rasterio
        cdi_array = src.read()                                                                  # Read the image as a NumPy array
        cdi_array[0,:,:] = np.flipud(cdi_array[0,:,:])                                          # flip array upside down for plotting
        # Get image metadata for setting axis labels
        transform = src.transform
        bounds = src.bounds
        width, height = src.width, src.height
        extent_cdi = [bounds.left, bounds.right, bounds.bottom, bounds.top]                     # Define the extent of the image in terms of lon and lat
    
    with rasterio.open(spi_img) as src:                                                         # Open the TIFF file using rasterio
        spi_array = src.read()                                                                  # Read the image as a NumPy array
        spi_array[0,:,:] = np.flipud(spi_array[0,:,:])                                          # flip array upside down for plotting
        # Get image metadata for setting axis labels
        transform = src.transform
        bounds = src.bounds
        width, height = src.width, src.height
        extent_spi = [bounds.left, bounds.right, bounds.bottom, bounds.top]                     # Define the extent of the image in terms of lon and lat
    
    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # fig.suptitle(f'type: {v}; date: {file[:7]}', fontsize=18)
    im_left = axs[0].imshow(cdi_array[0, :, :], cmap='RdBu', extent=extent_cdi,             
                            aspect='auto', vmin=-2, vmax=2)                                     # Plot the first image on the left subplot
    axs[0].set_xlabel('Longitude', fontsize=18) 
    axs[0].set_ylabel('Latitude', fontsize=18)
    axs[0].set_title(f'ECDI for {file[:7]}', fontsize=16)
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    
    im_right = axs[1].imshow(spi_array[0, :, :], cmap='RdBu', extent=extent_spi, 
                             aspect='auto', vmin=-2, vmax=2)                                    # Plot the second image on the right subplot
    axs[1].set_xlabel('Longitude', fontsize=18)
    # axs[1].set_ylabel('Latitude', fontsize=18)
    axs[1].set_title(f'SPI-3 for {file[:7]}', fontsize=16)
    axs[1].tick_params(axis='x', which='major', labelsize=14)
    axs[1].set_yticklabels([])
    
    # Create a common colorbar for both subplots
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im_right, cax=cax)
    # cbar.ax.set_ylabel('Values', fontsize=14)
    cbar.ax.tick_params(labelsize=14)  # Set font size for colorbar tick labels
    ticks = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(['≤ -2.0', '-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5', '≥ 2.0'])
    cbar.locator = FixedLocator(ticks)
    plot_boundary(axs)
    plot_northarrow(axs)
    plot_scalebar(axs)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'cdi_spi_{file[:-4]}_{v}.png'), 
                bbox_inches='tight', dpi=300)
    plt.show()

#%% plot USDM
usdm_dir = os.path.join(wd, 'data\\Texas\\P\\usdm\\pca\\1')
months = [str(i).zfill(2) for i in range(1, 13)]
for m in months:
    file = f'2022-{m}-01.tif'
    usdm_img = os.path.join(usdm_dir, file)
    with rasterio.open(usdm_img) as src:                                                        # Open the TIFF file using rasterio
        usdm_arr = src.read()                                                                   # Read the image as a NumPy array
        usdm_arr = np.where(usdm_arr == -9999, -1, usdm_arr)                                    # replace -9999 with -1 for plotting
        transform = src.transform
        bounds = src.bounds
        width, height = src.width, src.height
        extent = [bounds.left, bounds.right, bounds.top, bounds.bottom]                         # Define the extent of the image in terms of longitudes and latitudes
    
    # Define custom colormap values and labels
    categories = [-1, 0, 1, 2, 3, 4]
    cmap_values = [0, 0.2, 0.4, 0.6, 0.8, 1]
    cmap_labels = ['Normal or Wet conditions', 'Abnormally dry', 'Moderate Drought',
                   'Severe Drought', 'Extreme drought', 'Exceptional drought']
    cmap = ListedColormap(['grey', 'yellow', '#d6b575', 'orange', 'red', '#400000'])            # Create a custom colormap with specified color values
    
    fig, axs = plt.subplots(figsize=(10, 6))
    im = axs.imshow(usdm_arr[0, :, :], cmap=cmap, extent=extent, aspect='auto', vmin=min(categories), vmax=max(categories))
    axs.set_xlabel('Longitude', fontsize=18)
    axs.set_ylabel('Latitude', fontsize=18)
    axs.set_title(f'USDM for {file[:7]}', fontsize=18)
    axs.tick_params(axis='both', which='major', labelsize=16)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.1)                                      # Create a colorbar for the custom colormap
    cbar = fig.colorbar(im, cax=cax, ticks=categories)
    cbar.set_ticklabels(cmap_labels)                                                            # Display labels for each category on the colorbar
    cbar.ax.tick_params(labelsize=16)                                                           # Set font size for colorbar tick labels
    plot_boundary(axs)
    plot_northarrow(axs)
    plot_scalebar(axs)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'usdm_{file[:-4]}.png'), bbox_inches='tight', dpi=300)
    plt.show()

#%% discretize CDI and SPI - then, plot CDI, SPI, and USDM
def discretize_array(array, ranges):
    ranges = [(-np.inf, ranges[0]), (ranges[0], ranges[1]), (ranges[1], ranges[2]), 
              (ranges[2], ranges[3]), (ranges[3], ranges[4]), (ranges[4], np.inf)]              # define range

    labels = [4, 3, 2, 1, 0, -9999]                                                             # define labels
    discretized_array = np.empty(array.shape, dtype=array.dtype)                                # initialize an array to store discretized values
    for r, l in zip(ranges, labels):                                                            # Iterate over each range and label
        mask = (r[0] <= array) & (array < r[1])                                                 # Create a boolean mask for values within the current range
        discretized_array[mask] = l                                                             # Assign the corresponding label to values within the range
    return discretized_array

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

mask_array = mask_roi()                                                                         # get mask after removing Mexico and Gulf of Mexico

ranges_cdi = [-1.5, -1, -0.5, 0, 0.5]                                                           # ranges for CDI
ranges_spi = [-2, -1.6, -1.3, -0.8, -0.5]                                                       # ranges for SPI

cdi_dir = os.path.join(wd, 'output\\cdi_predicted\\maps\\pca\\1')                          # predicted CDI values using SPI for P
usdm_dir = os.path.join(wd, 'data\\Texas\\P\\usdm\\pca\\1')                                     # monthly averaged USDM
spi_dir = os.path.join(wd, 'data\\Texas\\P\\spi\\pca\\1')                                       # actual SPI values based on actual P

categories = [-1, 0, 1, 2, 3, 4]
cmap_values = [0, 0.2, 0.4, 0.6, 0.8, 1]
cmap_labels = ['Normal or Wet conditions', 'Abnormally dry', 'Moderate Drought',
               'Severe Drought', 'Extreme drought', 'Exceptional drought']
cmap = ListedColormap(['grey', 'yellow', '#d6b575', 'orange', 'red', '#400000'])                # Create a custom colormap with specified color values

months = [str(i).zfill(2) for i in range(1, 13)]
for m in months:
    file = f'2022-{m}-01.tif'; v='predicted'
    cdi_img = os.path.join(cdi_dir, file)
    spi_img = os.path.join(spi_dir, file)
    usdm_img = os.path.join(usdm_dir, file)
    
    with rasterio.open(cdi_img) as src:                                                         # Open the TIFF file using rasterio
        cdi_array = src.read()                                                                  # Read the image as a NumPy array
        # cdi_array[0,:,:] = np.flipud(cdi_array[0,:,:])                                          # flip array upside down for plotting
        # Get image metadata for setting axis labels
        transform = src.transform
        bounds = src.bounds
        width, height = src.width, src.height
        extent_cdi = [bounds.left, bounds.right, bounds.top, bounds.bottom]                     # Define the extent of the image in terms of lon and lat
    
    with rasterio.open(spi_img) as src:                                                         # Open the TIFF file using rasterio
        spi_array = src.read()                                                                  # Read the image as a NumPy array
        # spi_array[0,:,:] = np.flipud(spi_array[0,:,:])                                          # flip array upside down for plotting
        # Get image metadata for setting axis labels
        transform = src.transform
        bounds = src.bounds
        width, height = src.width, src.height
        extent_spi = [bounds.left, bounds.right, bounds.top, bounds.bottom]                     # Define the extent of the image in terms of lon and lat
    
    with rasterio.open(usdm_img) as src:                                                        # Open the TIFF file using rasterio
        usdm_arr = src.read()                                                                   # Read the image as a NumPy array
        usdm_arr = np.where(usdm_arr == -9999, -1, usdm_arr)                                    # replace -9999 with -1 for plotting
        transform = src.transform
        bounds = src.bounds
        width, height = src.width, src.height
        extent_usdm = [bounds.left, bounds.right, bounds.top, bounds.bottom]                    # Define the extent of the image in terms of longitudes and latitudes
    
    cdi_disc = discretize_array(cdi_array, ranges_cdi)                                          # discritize CDI
    spi_disc = discretize_array(spi_array, ranges_spi)                                          # discritize SPI
    cdi_filtered = np.where(mask_array==0, -1, cdi_disc[0,:,:])                                 # mask CDI to remove Mexico and Gulf of Mexico from the map
    spi_filtered = np.where(mask_array==0, -1, spi_disc[0,:,:])                                 # mask SPI to remove Mexico and Gulf of Mexico from the map

    fig, axs = plt.subplots(1, 3, figsize=(14,5), gridspec_kw={'width_ratios': [1, 1, 1.08]})  # Adjust the width ratio of the third subplot
    im_left = axs[0].imshow(cdi_filtered, cmap=cmap, extent=extent_cdi, aspect='auto', vmin=min(categories), vmax=max(categories))                                     # Plot the first image on the left subplot
    axs[0].set_xlabel('Longitude', fontsize=18) 
    axs[0].set_ylabel('Latitude', fontsize=18)
    axs[0].set_title(f'ECDI for {file[:7]}', fontsize=16)
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    
    im_mid = axs[1].imshow(spi_filtered, cmap=cmap, extent=extent_spi, aspect='auto', vmin=min(categories), vmax=max(categories))                                    # Plot the second image on the right subplot
    axs[1].set_xlabel('Longitude', fontsize=18)
    # axs[1].set_ylabel('Latitude', fontsize=18)
    axs[1].set_title(f'SPI-3 for {file[:7]}', fontsize=16)
    axs[1].tick_params(axis='x', which='major', labelsize=14)
    axs[1].set_yticklabels([])
    
    im_right = axs[2].imshow(usdm_arr[0,:,:], cmap=cmap, extent=extent_usdm, aspect='auto', vmin=min(categories), vmax=max(categories))
    axs[2].set_xlabel('Longitude', fontsize=18)
    # axs[2].set_ylabel('Latitude', fontsize=18)
    axs[2].set_title(f'USDM for {file[:7]}', fontsize=16)
    axs[2].tick_params(axis='x', which='major', labelsize=14)
    axs[2].set_yticklabels([])
    
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)                                      # Create a colorbar for the custom colormap
    cbar = fig.colorbar(im_right, cax=cax, ticks=categories)
    cbar.set_ticklabels(cmap_labels)                                                            # Display labels for each category on the colorbar
    cbar.ax.tick_params(labelsize=14)                                                           # Set font size for colorbar tick labels
    plot_boundary(axs)
    plot_northarrow(axs, scale=0.4, ylim_pos=0.17)
    plot_scalebar(axs, size="small")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'cdi_spi_usdm_{file[:-4]}.png'), bbox_inches='tight', dpi=300)
    plt.show()

#%% plot time series and compare actual vs predicted for two cells
var = 'P'; v='P'
data_dir = os.path.join(wd, 'data\\Texas_pred')
df_pred = pd.read_csv(os.path.join(data_dir, f'{var}//1_combined//1.csv'))
df_actu = pd.read_csv(os.path.join(data_dir, f'{var}//1_normalized//1.csv'))
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
cell0 = 5
labels = [X[:7] for X in df_pred.columns[-96:]]

# Plot for Cell 0
ax[0].plot(labels, df_pred.iloc[cell0, -96:], color='r', label='Prediction')
ax[0].plot(labels, df_actu.iloc[cell0, -96:], color='b', label='Actual')
ax[0].set_title(f'Cell {cell0}', fontsize=16)
ax[0].axvline(x=83, color='black', linestyle='--', linewidth=3)

# Plot for Cell 1
cell1 = 950
ax[1].plot(labels, df_pred.iloc[cell1, -96:], color='r', label='Prediction')
ax[1].plot(labels, df_actu.iloc[cell1, -96:], color='b', label='Actual')
ax[1].set_title(f'Cell {cell1}', fontsize=16)
ax[1].axvline(x=83, color='black', linestyle='--', linewidth=3)

# Common settings for both plots
for i in range(2):
    xticks_interval = 36
    ax[i].set_ylim(0, 1)
    ax[i].set_xticks(ax[i].get_xticks()[::xticks_interval])
    ax[i].set_xticklabels(labels[::xticks_interval], rotation=45, ha='center', fontsize=14)
    # ax[i].set_xlabel('Date', fontsize=f)
    ax[0].set_ylabel(f'{v} (normalized)', fontsize=16)
    ax[i].tick_params(axis='y', labelsize=14)

ax[1].legend(fontsize=14)
plt.subplots_adjust(wspace=0.08)                                                                # You can adjust the value of wspace as needed
fig.savefig(os.path.join(output_dir, f'timeseries_{var}.png'), bbox_inches='tight', dpi=300)
plt.show()

#%% plot time series and compare actual vs predicted for two groups of cells
var = 'SM'; v='SM'
data_dir = os.path.join(wd, 'data\\Texas_pred')
df_pred = pd.read_csv(os.path.join(data_dir, f'{var}//1_combined//1.csv'))
df_actu = pd.read_csv(os.path.join(data_dir, f'{var}//1_normalized//1.csv'))
fig, ax = plt.subplots(9, 2, sharey=True)
cells_west = [5,6,7,5+43,6+43,7+43,5+2*43,6+2*43,7+2*43]                                        # a 3*3 window of cells in West Texas
cells_east = [2027-2*43,2028-2*43,2029-2*43,2027-43,2028-43,2029-43,2027,2028,2029]             # a 3*3 window of cells in West Texas
labels = [X[:7] for X in df_pred.columns[-96:]]
def plot_timeseries(ax, i ,j, df_pred, df_actu, cell):
    ax[i,j].plot(labels, df_pred.iloc[cell, -96:], color='r', label='Prediction')
    ax[i,j].plot(labels, df_actu.iloc[cell, -96:], color='b', label='Actual')
    ax[i,j].set_title(f'Cell {cell}', fontsize=8, y=0.7)
    ax[i,j].axvline(x=83, color='black', linestyle='--', linewidth=2)
    if j==0:
        # ax[i,j].text(-10, 0.5, f'{cell}', va='center', ha='center', rotation='vertical', fontsize=8)
        ax[i,j].tick_params(axis='y', labelsize=7, labelright=False, labelleft=True)
    else:
        # ax[i,j].text(110, 0.5, f'{cell}', va='center', ha='center', rotation='vertical', fontsize=8)
        ax[i,j].tick_params(axis='y', labelsize=7, labelright=False, labelleft=False)
    if i!=8:
        ax[i,j].set_xticks([])
    ax[i,j].set_xticklabels([])
    
for i in range(9):
    plot_timeseries(ax, i, 0, df_pred, df_actu, cells_west[i])
    plot_timeseries(ax, i, 1, df_pred, df_actu, cells_east[i])
    
plt.subplots_adjust(hspace=0.8)  # Adjust the vertical spacing between subplots
xticks_interval = 36
for j in range(2):
    ax[8,j].set_ylim(0, 1)
    ax[8,j].set_xticks(ax[8,j].get_xticks()[::xticks_interval])
    ax[8,j].set_xticklabels(labels[::xticks_interval], rotation=45, ha='center', fontsize=10)
    
# ax[4,0].text(-20, 0.5, 'Cell number', va='center', ha='center', rotation='vertical', fontsize=14)   # text as "Cell number"
# ax[4,1].text(120, 0.5, 'Cell number', va='center', ha='center', rotation='vertical', fontsize=14)   # text as "Cell number"

# ax[1].legend(fontsize=14)
# Get handles and labels from the last subplot to create a single legend
handles, labels = ax[0,0].get_legend_handles_labels()

# Create a single legend for all subplots at the top of the figure
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
plt.subplots_adjust(wspace=0.08)                                                                # You can adjust the value of wspace as needed
fig.savefig(os.path.join(output_dir, f'timeseries9v9_{var}.png'), bbox_inches='tight', dpi=300)
plt.show()

#%% plot time series and compare actual vs predicted for two groups of cells for comparison between LSTM and SARIMA
var = 'P'; v='P'
data_dir = os.path.join(wd, 'data\\Texas_pred')
df_lstm = pd.read_csv(os.path.join(data_dir, f'{var}//1_combined//1.csv'))
df_sarima = pd.read_csv(f'C:\\Projects\\Drought\\Code\\output\\sarima\\{var}\\1_combined.csv')
df_actu = pd.read_csv(os.path.join(data_dir, f'{var}//1_normalized//1.csv'))
fig, ax = plt.subplots(9, 2, sharey=True)
cells_west = [5,6,7,5+43,6+43,7+43,5+2*43,6+2*43,7+2*43]                                        # a 3*3 window of cells in West Texas
cells_east = [2027-2*43,2028-2*43,2029-2*43,2027-43,2028-43,2029-43,2027,2028,2029]             # a 3*3 window of cells in West Texas
labels = [X[:7] for X in df_actu.columns[-96:]]
def plot_timeseries(ax, i ,j, df_pred, df_actu, cell):
    ax[i,j].plot(labels, df_pred.iloc[cell, -96:], color='r', label='Prediction')
    ax[i,j].plot(labels, df_actu.iloc[cell, -96:], color='b', label='Actual')
    ax[i,j].set_title(f'Cell {cell}', fontsize=8, y=0.7)
    ax[i,j].axvline(x=83, color='black', linestyle='--', linewidth=2)
    if j==0:
        # ax[i,j].text(-10, 0.5, f'{cell}', va='center', ha='center', rotation='vertical', fontsize=8)
        ax[i,j].tick_params(axis='y', labelsize=7, labelright=False, labelleft=True)
    else:
        # ax[i,j].text(110, 0.5, f'{cell}', va='center', ha='center', rotation='vertical', fontsize=8)
        ax[i,j].tick_params(axis='y', labelsize=7, labelright=False, labelleft=False)
    if i!=8:
        ax[i,j].set_xticks([])
    ax[i,j].set_xticklabels([])
    
for i in range(9):
    plot_timeseries(ax, i, 0, df_lstm, df_actu, cells_east[i])
    plot_timeseries(ax, i, 1, df_sarima, df_actu, cells_east[i])
    
plt.subplots_adjust(hspace=0.8)  # Adjust the vertical spacing between subplots
xticks_interval = 36
for j in range(2):
    ax[8,j].set_ylim(0, 1)
    ax[8,j].set_xticks(ax[8,j].get_xticks()[::xticks_interval])
    ax[8,j].set_xticklabels(labels[::xticks_interval], rotation=45, ha='center', fontsize=10)
    
# ax[4,0].text(-20, 0.5, 'Cell number', va='center', ha='center', rotation='vertical', fontsize=14)   # text as "Cell number"
# ax[4,1].text(120, 0.5, 'Cell number', va='center', ha='center', rotation='vertical', fontsize=14)   # text as "Cell number"

# ax[1].legend(fontsize=14)
# Get handles and labels from the last subplot to create a single legend
handles, labels = ax[0,0].get_legend_handles_labels()

# Create a single legend for all subplots at the top of the figure
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
plt.subplots_adjust(wspace=0.08)                                                                # You can adjust the value of wspace as needed
fig.savefig(os.path.join(output_dir, f'lstm_vs_sarima_{var}.png'), bbox_inches='tight', dpi=300)
plt.show()

#%% plot time series and compare actual vs predicted for two groups of cells for comparison between LSTM and SARIMA
df_lstm = pd.read_csv(os.path.join(wd, 'data//Texas_pred//P//1_combined//1.csv'))               # predictions from LSTM
df_sarima = pd.read_csv(os.path.join(wd, 'output//sarima//P//1_combined.csv'))                     # predictions from SARIMA
df_actu = pd.read_csv(os.path.join(wd, 'data//Texas_pred//P//1_normalized//1.csv'))                    # actual data
fig, ax = plt.subplots(9, 2, sharey=True)
cells_west = [5,6,7,5+43,6+43,7+43,5+2*43,6+2*43,7+2*43]                                        # a 3*3 window of cells in West Texas
cells_east = [2027-2*43,2028-2*43,2029-2*43,2027-43,2028-43,2029-43,2027,2028,2029]             # a 3*3 window of cells in West Texas
labels = [X[:7] for X in df_actu.columns[-96:]]                                                 # X-tick labels

def plot_timeseries(ax, i ,j, df_actu, df_lstm, df_sarima, cells):
    cell = cells[i]
    ax[i,j].plot(labels, df_actu.iloc[cell, -96:], color='b', label='Actual')
    ax[i,j].plot(labels[-12:], df_lstm.iloc[cell, -12:], color='r', label='LSTM')
    ax[i,j].plot(labels[-12:], df_sarima.iloc[cell, -12:], color='g', label='SARIMA')
    ax[i,j].axvline(x=83, color='black', linestyle='--', linewidth=2)
    ax[i,j].set_title(f'Cell {cell}', fontsize=8, y=0.7)

    if j==0:
        # ax[i,j].text(-10, 0.5, f'{cell}', va='center', ha='center', rotation='vertical', fontsize=8)
        ax[i,j].tick_params(axis='y', labelsize=7, labelright=False, labelleft=True)
    else:
        # ax[i,j].text(110, 0.5, f'{cell}', va='center', ha='center', rotation='vertical', fontsize=8)
        ax[i,j].tick_params(axis='y', labelsize=7, labelright=False, labelleft=False)
    if i!=8:
        ax[i,j].set_xticks([])
    ax[i,j].set_xticklabels([])
    
for i in range(9):
    plot_timeseries(ax, i, 0, df_actu, df_lstm, df_sarima, cells_west)
    plot_timeseries(ax, i, 1, df_actu, df_lstm, df_sarima, cells_east)

plt.subplots_adjust(hspace=0.8)  # Adjust the vertical spacing between subplots
xticks_interval = 36
for j in range(2):
    ax[8,j].set_ylim(0, 1)
    ax[8,j].set_xticks(ax[8,j].get_xticks()[::xticks_interval])
    ax[8,j].set_xticklabels(labels[::xticks_interval], rotation=45, ha='center', fontsize=10)
    
# ax[4,0].text(-20, 0.5, 'Cell number', va='center', ha='center', rotation='vertical', fontsize=14)   # text as "Cell number"
# ax[4,1].text(120, 0.5, 'Cell number', va='center', ha='center', rotation='vertical', fontsize=14)   # text as "Cell number"

# Get handles and labels from the last subplot to create a single legend
handles, labels = ax[0,0].get_legend_handles_labels()

# Create a single legend for all subplots at the top of the figure
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)
plt.subplots_adjust(wspace=0.08)                                                                # You can adjust the value of wspace as needed
fig.savefig(os.path.join(output_dir, 'lstm_vs_sarima_ql.png'), bbox_inches='tight', dpi=300)
plt.show()

#%% Quantile loss for comparing LSTM to SARIMA
# Calculate quantile loss (e.g., for 90th percentile)
def quantile_loss(y_true, y_pred, q=0.95):
    errors = y_true - y_pred
    loss = np.maximum(q * errors, (q - 1) * errors)
    return np.mean(loss)
cells_west = [5,6,7,5+43,6+43,7+43,5+2*43,6+2*43,7+2*43]                                        # a 3*3 window of cells in West Texas
cells_east = [2027,2028,2029,2027-43,2028-43,2029-43,2027-2*43,2028-2*43,2029-2*43]             # a 3*3 window of cells in East Texas
ql = []
for cell in cells_west:
    ql_model1 = quantile_loss(df_actu.iloc[cell, -12:], df_lstm.iloc[cell, -12:])
    ql_model2 = quantile_loss(df_actu.iloc[cell, -12:], df_sarima.iloc[cell, -12:])
    ql.append([cell, ql_model1, ql_model2])

p_lstm = pd.read_csv(os.path.join(wd,'output\\lstm\\predictions\\P_combined.csv'))
df_ql = pd.DataFrame(ql, columns=['cell','ql_lstm', 'ql_sarima'])
print('ql_lstm=', df_ql['ql_lstm'].mean())
print('ql_sarima=', df_ql['ql_sarima'].mean())

df_ql_pos = df_ql[p_lstm['nse']>=0]
print('ql_lstm=', df_ql_pos['ql_lstm'].mean())
print('ql_sarima=', df_ql_pos['ql_sarima'].mean())

#%% plot SARIMA vs LSTM
import os
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

cdi_img = r'C:\Projects\RS-Drought\output\tiff\poster\nse_sarima.tif'
spi_img = r'C:\Projects\RS-Drought\output\tiff\poster\nse_P.tif'

# Open the TIFF file using rasterio
with rasterio.open(cdi_img) as src:
    # Read the image as a NumPy array
    cdi_array = src.read()

    # Get image metadata for setting axis labels
    transform = src.transform
    bounds = src.bounds
    width, height = src.width, src.height

    # Define the extent of the image in terms of longitudes and latitudes
    extent_cdi = [bounds.left, bounds.right, bounds.top, bounds.bottom]

# Open the TIFF file using rasterio
with rasterio.open(spi_img) as src:
    # Read the image as a NumPy array
    spi_array = src.read()

    # Get image metadata for setting axis labels
    transform = src.transform
    bounds = src.bounds
    width, height = src.width, src.height

    # Define the extent of the image in terms of longitudes and latitudes
    extent_spi = [bounds.left, bounds.right, bounds.top, bounds.bottom]

# Create a figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot the first image on the left subplot
im_left = axs[0].imshow(cdi_array[0, :, :], cmap='RdYlGn', extent=extent_cdi, aspect='auto', vmin=-1, vmax=1)
axs[0].set_xlabel('Longitude', fontsize=18)
axs[0].set_ylabel('Latitude', fontsize=18)
axs[0].set_title('Prediction accuracy from SARIMA', fontsize=18)
axs[0].tick_params(axis='both', which='major', labelsize=16)

# Plot the second image on the right subplot
im_right = axs[1].imshow(spi_array[0, :, :], cmap='RdYlGn', extent=extent_spi, aspect='auto', vmin=-1, vmax=1)
axs[1].set_xlabel('Longitude', fontsize=18)
# axs[1].set_ylabel('Latitude', fontsize=18)
axs[1].set_title('Prediction accuracy from LSTM', fontsize=18)
axs[1].tick_params(axis='both', which='major', labelsize=16)

# Create a common colorbar for both subplots
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = fig.colorbar(im_right, cax=cax, label='Values')
cbar.ax.tick_params(labelsize=16)  # Set font size for colorbar tick labels

# Adjust layout for better spacing
plt.tight_layout()
fig.savefig(os.path.join(r'C:\Projects\RS-Drought\output\png', f'cdi_spi_{file[:-4]}_{v}.png'), 
            bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

#%% plot SARIMA vs LSTM vs CNN
var = 'P'
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
    mask_array[:,:34] = 1
    return mask_array

mask_array = mask_roi()                                                                         # get mask after removing Mexico and Gulf of Mexico

data_sarima = pd.read_csv(os.path.join(wd, f'output\\sarima\\{var}\\{var}_sarima.csv'))
sarima_img = os.path.join(os.path.join(wd, f'output\\sarima\\{var}\\{var}_nse.tif'))
save_tiff(data_sarima['nse'], coords, sarima_img)                                               # save SARIMA as tiff
lstm_img = os.path.join(wd, 'output\\lstm\\accuracy_maps\\tiff\\{var}_nse.tif')
cnn_img = os.path.join(wd, 'output\\cnn\\accuracy_maps\\tiff\\{var}_nse.tif')

with rasterio.open(lstm_img) as src:                                                             # Open the TIFF file using rasterio
    lstm_arr = src.read()                                                                      # Read the image as a NumPy array
    lstm_arr[0,:,:] = np.flipud(lstm_arr[0,:,:])                                          # flip image for visualization
    transform = src.transform
    bounds = src.bounds
    width, height = src.width, src.height
    extent_lstm = [bounds.left, bounds.right, bounds.top, bounds.bottom]                         # Define the extent of the image in terms of longitudes and latitudes

with rasterio.open(cnn_img) as src:                                                             # Open the TIFF file using rasterio
    cnn_arr = src.read()                                                                      # Read the image as a NumPy array
    cnn_arr[0,:,:] = np.flipud(cnn_arr[0,:,:])                                          # flip image for visualization
    transform = src.transform
    bounds = src.bounds
    width, height = src.width, src.height
    extent_cnn = [bounds.left, bounds.right, bounds.top, bounds.bottom]                         # Define the extent of the image in terms of longitudes and latitudes

with rasterio.open(sarima_img) as src:                                                             # Open the TIFF file using rasterio
    sarima_arr = src.read()                                                                        # Read the image as a NumPy array
    sarima_arr[0,:,:] = np.flipud(sarima_arr[0,:,:])

### set water as nan values in the map
lstm_filtered = np.where(mask_array==0, np.nan, lstm_arr[0,:,:])                                 # mask CDI to remove Mexico and Gulf of Mexico from the map
cnn_filtered = np.where(mask_array==0, np.nan, cnn_arr[0,:,:])                                 # mask CDI to remove Mexico and Gulf of Mexico from the map
sarima_filtered = np.where(mask_array==0, np.nan, sarima_arr[0,:,:])                                 # mask CDI to remove Mexico and Gulf of Mexico from the map

### set negative values to 0
lstm_filtered = np.where(lstm_arr[0,:,:]<0, 0, lstm_arr[0,:,:])                                 # mask CDI to remove Mexico and Gulf of Mexico from the map
cnn_filtered = np.where(cnn_arr[0,:,:]<0, 0, cnn_arr[0,:,:])                                 # mask CDI to remove Mexico and Gulf of Mexico from the map
sarima_filtered = np.where(sarima_arr[0,:,:]<0, 0, sarima_arr[0,:,:])                                 # mask CDI to remove Mexico and Gulf of Mexico from the map

fig, axs = plt.subplots(1, 3, figsize=(14,5), sharey=True, gridspec_kw={'width_ratios': [1, 1, 1.08]})  # Adjust the width ratio of the third subplot
im_left = axs[0].imshow(lstm_filtered, cmap='RdYlGn', extent=extent_lstm, 
                        aspect='auto', vmin=-1, vmax=1)                                         # Plot the first image on the left subplot
axs[0].set_xlabel('Longitude', fontsize=18)
axs[0].set_ylabel('Latitude', fontsize=18)
axs[0].set_title('LSTM', fontsize=18)
axs[0].tick_params(axis='both', which='major', labelsize=16)

img_mid = axs[1].imshow(cnn_filtered, cmap='RdYlGn', extent=extent_cnn, 
                         aspect='auto', vmin=-1, vmax=1)                                        # Plot the second image on the right subplot
axs[1].set_xlabel('Longitude', fontsize=18)
# axs[1].set_ylabel('Latitude', fontsize=18)
axs[1].set_title('1D-CNN', fontsize=18)
axs[1].tick_params(axis='both', which='major', labelsize=16)

# Plot the second image on the right subplot
im_right = axs[2].imshow(sarima_filtered, cmap='RdYlGn', extent=extent_cnn, aspect='auto', vmin=-1, vmax=1)
axs[2].set_xlabel('Longitude', fontsize=18)
# axs[1].set_ylabel('Latitude', fontsize=18)
axs[2].set_title('SARIMA', fontsize=18)
axs[2].tick_params(axis='both', which='major', labelsize=16)

# Create a common colorbar for both subplots
divider = make_axes_locatable(axs[2])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(im_right, cax=cax, label='NSE')
cbar.ax.tick_params(labelsize=14)  # Set font size for colorbar tick labels
ticks = [-1, -0.5, 0.0, 0.5, 1]
cbar.set_ticks(ticks)
cbar.set_ticklabels(['≤-1.0', '-0.5', '0.0', '0.5', '1.0'])
plot_boundary(axs)
plot_northarrow(axs)
plot_scalebar(axs, size="small")
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'sarima_cnn_lstm.png'), bbox_inches='tight', dpi=300)
plt.show()

#%% box plot for LSTM vs CNN vs SARIMA
df_lstm = pd.read_csv(os.path.join(wd, 'output//lstm//predictions//P_combined.csv'))
df_cnn = pd.read_csv(os.path.join(wd, 'output//cnn//predictions//P_combined.csv'))
df_sarima = pd.read_csv(os.path.join(wd, 'output//sarima//P//p_sarima.csv'))

# Assuming you have three DataFrames df1, df2, and df3, each containing the arrays you want to plot
data1 = df_lstm['nse'].values
data2 = df_cnn['nse'].values
data3 = df_sarima['nse'].values

# Combine data into a list
data = [data1, data2, data3]

# Plotting box plots
plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=['LSTM', '1D-CNN', 'SARIMA'])
# plt.title('Box plots for three arrays from three DataFrames')
plt.xlabel('DataFrame')
plt.ylabel('Array Values')
plt.grid(True)
plt.show()

#%% get the dimension of image for one of the Tables in the manuscript
import ee
from datetime import datetime

# Initialize Earth Engine
ee.Initialize()

# Define the Image Collection
# image_collection = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
image_collection = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")


# Define the time range
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 1, 2)

# Filter the Image Collection by date
filtered_collection = image_collection.filterDate(start_date, end_date)

# Get the list of images in the collection
image_list = filtered_collection.toList(filtered_collection.size())

# Iterate over each image and print its dimensions
for i in range(image_list.size().getInfo()):
    image = ee.Image(image_list.get(i))
    info = image.getInfo()
    width = info['bands'][0]['dimensions'][0]
    height = info['bands'][0]['dimensions'][1]
    print(f"Image {i+1}: Width = {width}, Height = {height}")
