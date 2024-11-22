# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:53:43 2024

@author: ghaff

this script does the following:
    1- merge predictions for all cells into one csv file
    2- combine predictions for all n_steps to get the best result
    3- plot accuracy maps
    
Run this code with spatial environment.
"""
wd = r'C:\Projects\Drought\Code'
output_dir = r'C:\Projects\Drought\Code\output\lstm\accuracy_maps'
import os
import sys
sys.path.append(wd)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geo_northarrow import add_north_arrow
from shapely.geometry.point import Point
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FixedLocator
# =============================================================================
# merge predictions from different cells
# =============================================================================
def merge_preds(pred_dir, var, cap_zero=True):
    csv_files = [f.path for f in os.scandir(os.path.join(pred_dir,var)) 
                 if f.path.endswith('.csv')]                                                # list all csv files in the folder
    frame = []                                                                              # create a list to store results
    for csv_file in csv_files:
        df_result = pd.read_csv(csv_file)
        frame.append(df_result)
    df_merged = pd.concat(frame).sort_values(by='cell')                                     # merge all results
    df_merged.to_csv(os.path.join(pred_dir, f'{var}_merged.csv'), index=None)               # save df_merged as csv
    df_merged = pd.read_csv(os.path.join(pred_dir, f'{var}_merged.csv'))                    # read df_merged (this line is to avoid errors in Spyder)
    return df_merged

# =============================================================================
# combine predictions from different n_steps
# =============================================================================
def combine_preds(pred_dir, var):
    csv_files = [f.path for f in os.scandir(os.path.join(pred_dir, var)) 
                 if f.path.endswith('.csv')]
    # read all csv files in one framme
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        n_steps = int(os.path.basename(csv_file)[0:2])
        df['n_steps'] = n_steps
        dfs.append(df)
    max_nse_rows = []
    for row_index in range(len(df)):                                                        # Iterate through rows
        max_nse_value = float('-inf')                                                       # Initialize with negative infinity
        max_nse_row = None
        for df in dfs:                                                                      # Compare 'nse' values for each DataFrame
            current_nse_value = df.loc[row_index, 'nse']
            if current_nse_value > max_nse_value:
                max_nse_value = current_nse_value
                max_nse_row = df.loc[row_index]
        max_nse_rows.append(max_nse_row)

    df_combined = pd.DataFrame(max_nse_rows)                                                # Create a DataFrame from the selected rows
    df_combined.to_csv(os.path.join(pred_dir, f'{var}_combined.csv'), index=None)           # save as csv
    return df_combined

# =============================================================================
# extract metrics, predictions and actual data from LSTM and CNN results
# =============================================================================
"""note that predictions and actual arrays are normalized."""
def extract_metrics(df, cap_zero=True):
    metrics = df[['cell','batchsize','mse','nse']]                                          # extract metrics
    df_pred = df[['y_test','yhat']]                                                         # extract predictions and actual data
    # Function to convert array strings to NumPy arrays
    def convert_array_string_to_array(array_string):
        array_values_str = array_string[1:-1].split()
        return np.array([float(value) for value in array_values_str])
    
    # Apply the conversion function and create a new column
    df_pred['prediction'] = df_pred['yhat'].apply(convert_array_string_to_array)
    df_pred['actual'] = df_pred['y_test'].apply(convert_array_string_to_array)
    prediction = np.vstack(df_pred['prediction'].to_numpy())
    if cap_zero:
        prediction = np.clip(prediction, a_min=0.0, a_max=None)
    actual = np.vstack(df_pred['actual'].to_numpy())
    return metrics, prediction, actual

from sklearn.preprocessing import MinMaxScaler
def normalize(df):
    scaler = MinMaxScaler()
    data = df.iloc[:,2:]                                                                        # get only precipitation values
    coords = df.iloc[:,0:2]                                                                     # get coordination
    normalized_data = scaler.fit_transform(data.T)                                              # Normalize each row separately and store in a new DataFrame
    filtered_df = pd.DataFrame(normalized_data.T, columns=data.columns)                         # Create a new DataFrame with the same column names
    normalized_df = pd.concat([coords,filtered_df], axis=1)
    return normalized_df

# =============================================================================
# extract metrics, predictions and actual data from SARIMA results
# =============================================================================
"""note that predictions and actual arrays are normalized."""
from io import StringIO
def extract_metrics_sarima(df, cap_zero=True):
    metrics = df[['cell','config','mse','nse']]                                          # extract metrics
    df_pred = df[['predictions']]                                                        # extract predictions
    # Function to convert array strings to NumPy arrays
    def convert_array_string_to_array(data_str):
        lines = data_str.strip().split('\n')                                            # Split the string into lines
        return np.array([float(line.split()[1]) for line in lines[:-1]])                # Parse each line to extract the value part
    
    # Apply the conversion function and create a new column
    df_pred['prediction'] = df_pred['predictions'].apply(convert_array_string_to_array)
    prediction = np.vstack(df_pred['prediction'].to_numpy())
    return metrics, prediction

# df = pd.read_csv(r'C:\Projects\Drought\Code\output\sarima\p_sarima.csv')
# metrics, prediction = extract_metrics_sarima(df)
df_out = pd.read_csv(r'C:\Projects\Drought\Code\data\Texas\P\1_resampled\1.csv')
# df_out.iloc[:,-12:] = prediction                                                               # replace last 12 months with predicted data
# df_normalized = normalize(df_out)
# df_normalized.to_csv(r'C:\Projects\Drought\Code\output\sarima\1_combined.csv', index=None)              # save combined data as csv

#%%
# =============================================================================
# replace LSTM predictions with monthly averages for cells with negative NSE
# =============================================================================
from sklearn.metrics import r2_score
def replace_lstm(df_raw, df_lstm):
    df_hist = df_raw.iloc[:,2:-12]
    index = pd.date_range(start='2001-01-01', periods=df_hist.shape[1], freq='MS')
    
    # Iterate through rows and calculate monthly averages
    monthly_averages_list = []
    for _, row in df_hist.iterrows():
        time_series = pd.Series(row.values, index=index)
        monthly_averages = time_series.groupby(time_series.index.month).mean()
        monthly_averages_list.append(monthly_averages)
    df_monthly_average = pd.DataFrame(monthly_averages_list)
    
    r2_frame = []
    for i in range(len(df_hist)):
        r2 = r2_score(df_raw.iloc[i,-12:], df_monthly_average.iloc[i,:])
        r2_frame.append(r2)
    
    negative_nse_indices = df_lstm.index[df_lstm['nse'] < 0]                            # get the cells with negative NSE vales
    df_out = pd.DataFrame(columns=['lstm','average','lstm+average'])
    df_out['lstm'] = df_lstm['nse']
    df_out['average'] = r2_frame
    df_out['lstm+average'] = df_lstm['nse']
    df_out['lstm+average'][negative_nse_indices] = np.array(r2_frame)[negative_nse_indices]

# =============================================================================
# save data as tiff using coordinations from resampled data
# =============================================================================
import rasterio
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

def plot_scalebar(axs):
    """tutorial:
        https://geopandas.org/en/stable/gallery/matplotlib_scalebar.html"""
    points = gpd.GeoSeries([Point(-106, 26), Point(-105, 26)], crs=4326)                        # Geographic WGS 84 - degrees
    points = points.to_crs(32619)                                                               # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])                                             # calculate distance in m
    try:
        for j in range(len(axs)):
            axs[j].add_artist(ScaleBar(distance_meters, location='lower right', 
                                       font_properties={"size": "large",}, 
                                       scale_formatter=lambda value, unit: f"> {value} {unit} <",))
    except TypeError:
        axs.add_artist(ScaleBar(distance_meters, location='lower right', 
                                   font_properties={"size": "large",}, 
                                   scale_formatter=lambda value, unit: f"> {value} {unit} <",))
        
def plot_northarrow(axs):
    try:
        for j in range(len(axs)):
            add_north_arrow(axs[j], scale=.75, xlim_pos=.92, ylim_pos=.14, color='#000', text_scaler=3, text_yT=-1.25)
    except TypeError:
        add_north_arrow(axs, scale=.75, xlim_pos=.92, ylim_pos=.14, color='#000', text_scaler=3, text_yT=-1.25)

#%% mainscript
pred_dir = os.path.join(wd, 'output\\lstm\\predictions')
coords = pd.read_csv(os.path.join(wd, 'data\\Texas\\P\\1_resampled\\1.csv'))
for var in ['P','NDVI','LST','SM']:
    ### combine results for different n_steps
    df_combined = combine_preds(pred_dir, var)

    ### export tiff images of prediction accuracy
    for metric in ['mse', 'nse']:
        df_acc = pd.read_csv(os.path.join(pred_dir, f'{var}_combined.csv'))
        output_tiff = os.path.join(output_dir,'tiff', f'{var}_{metric}.tif')
        save_tiff(df_acc[metric], coords, output_tiff)

# metrics, prediction, actual = extract_metrics(df_combined)
# data_12 = pd.read_csv(os.path.join(pred_dir, 'P\\12_P_0_2067.csv'))
# output_file = os.path.join(output_dir,'tiff', f'{var}_{metric}_12.tif')
# save_tiff(data_12[metric], coords, output_file)

#%% plot prediction accuracy by NSE
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

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import matplotlib.patches as patches

f = 18                                                                                          # set value for fontsize
metric='nse'                                                                                    # select accuracy metric
labels = ['P', 'NDVI', 'T', 'SM']
variables = ['P','NDVI','LST','SM']
# Create a Rectangle patch

for v in range(len(variables)):
    var = variables[v]
    image = os.path.join(output_dir, 'tiff', f'{var}_{metric}.tif')
    
    ### plot accuracy map from tiff file
    with rasterio.open(image) as src:                                                           # Open the TIFF file using rasterio
        img_array = src.read()                                                                  # Read the image as a NumPy array
        img_array[0,:,:] = np.flipud(img_array[0,:,:])                                          # flip image for visualization
        transform = src.transform                                                               # Get image metadata for setting axis labels
        bounds = src.bounds
        width, height = src.width, src.height
        extent_cdi = [bounds.left, bounds.right, bounds.top, bounds.bottom]                     # Define the extent of the image in terms of longitudes and latitudes
    img_filtered = np.where(mask_array==0, np.nan, img_array[0,:,:])                                 # mask CDI to remove Mexico and Gulf of Mexico from the map
    fig, ax = plt.subplots(figsize=(10, 8))                                                     # Create a figure with one subplot
    im = ax.imshow(img_filtered, cmap='RdYlGn', extent=extent_cdi, aspect='auto', vmin=-1, vmax=1)
    ax.set_xlabel('Longitude', fontsize=f)
    ax.set_ylabel('Latitude', fontsize=f)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(labels[v], fontsize=25)


    # Add a colorbar to the figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, label='NSE')
    cbar.ax.tick_params(labelsize=16)  # Set font size for colorbar tick labels
    ticks = [-1, -0.5, 0.0, 0.5, 1]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(['≤-1.0', '-0.5', '0.0', '0.5', '1.0'])
    
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # cbar = fig.colorbar(im, cax=cax, label='NSE')
    # cbar.ax.tick_params(labelsize=14)  # Set font size for colorbar tick labels
    # ticks = [-1, -0.5, 0.0, 0.5, 1]
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(['≤-1.0', '-0.5', '0.0', '0.5', '1.0'])
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.ax.tick_params(labelsize=16)                                                           # Set font size for colorbar tick labels
    cbar.set_label('NSE', fontsize=16)
    
    ### plot Texas boundary as a shapefile
    shapefile_path = r'C:\Projects\Drought\Code\data\Texas_State_Boundary\State.shp'            # Path to boundary shapefile
    gdf = gpd.read_file(shapefile_path)                                                         # Create a GeoDataFrame from the shapefile
    gdf_extent = gdf.total_bounds                                                               # Get the actual data extent from the GeoDataFrame
    gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=4)                               # Plot the shapefile on the same axis
    ax.set_xlim(gdf_extent[0], gdf_extent[2])                                                   # Set the extent of the plot to match the actual data extent
    ax.set_ylim(gdf_extent[1], gdf_extent[3])
    plot_northarrow(ax)
    plot_scalebar(ax)
    dist = df_out['latitude'][8] - df_out['latitude'][5]
    west_rec = plt.Rectangle((df_out['longitude'][5], df_out['latitude'][5]), dist, dist, linewidth=5, edgecolor='r', facecolor='none')
    east_rec = plt.Rectangle((df_out['longitude'][1941], df_out['latitude'][1941]), dist, dist, linewidth=5, edgecolor='r', facecolor='none')
    ax.add_patch(west_rec)
    ax.add_patch(east_rec)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'png\\{var}_{metric}.png'), bbox_inches='tight', dpi=300)
    plt.show()

