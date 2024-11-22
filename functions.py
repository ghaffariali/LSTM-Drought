# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:01:58 2023

@author: Ali Ghaffari           alg721@lehigh.edu

this script contains codes for the remote sensing project.
"""
# =============================================================================
# import libraries and packages
# =============================================================================
# import os
import ee
import os
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import zscore
from scipy import stats as st
from scipy.interpolate import griddata
from scipy import interpolate
from tqdm import tqdm
from functools import reduce
from scipy.signal import convolve2d
from sklearn.decomposition import PCA
import rasterio
import matplotlib.pyplot as plt

# =============================================================================
# workspace initialization
# =============================================================================
# ee.Authenticate()                                                                       # Trigger the authentication flow. This is only required once.
ee.Initialize()                                                                         # Initialize the library

# =============================================================================
# get_regions
# =============================================================================
def get_geometries(regions_link, boundary_link, geometries_dir, dataset_name, 
                   state_code=-99, save_to_pkl=True):
    """
    this function gets geometries from regions_link dataset and filters it by
    geometry from boundary_link. Then, saves a *.pkl file with two lists as 
    region_names and region_geometries.

    Parameters
    ----------
    regions_link : string
        regions dataset link.
    boundary_link : string
        boundary geometry link.
    geometries_dir : string
        directory to save geometries as *.pkl.
    dataset_name : string
        name of dataset (e.g. counties or huc).
    state_code: int, optional
        if not None, then retreives counties from a state based on state_code.
    save_to_pkl : Boolean, optional
        Save output to *.pkl. The default is True.

    Returns
    -------
    None.

    """
    
    print('Started getting geometries for {}:'.format(dataset_name))
    if state_code >= 0:
        regions = ee.FeatureCollection(regions_link)\
            .filter(ee.Filter.eq('STATEFP',str(state_code)))                                # filter regions by state code
    else:
        boundary = ee.FeatureCollection(boundary_link).geometry()                           # get geometry for boundary region
        regions = ee.FeatureCollection(regions_link)\
            .filter(ee.Filter.geometry(boundary))                                           # get geometries for the regions
    
    # Get a list of features in the filtered FeatureCollection
    features = regions.toList(regions.size())

    # Loop through the features and get the name and geometry of each one
    geometries = []                                                                         # create a list to store geometries
    names = []                                                                              # create a list to store names
    n_features = regions.size().getInfo()                                                   # get number of features
    for i in tqdm(range(n_features),miniters=1):                                            # loop over features
        print(i)
        feature = ee.Feature(features.get(i))
        name = feature.getInfo()['properties']['name']
        geometry = feature.geometry()
        if geometry is not None:
            names.append(name)
            geometries.append(geometry)
    
    if save_to_pkl:                                                                         # save names and geometries to *.pkl
        with open(os.path.join(geometries_dir, '{}.pkl'.format(dataset_name)), 
                  'wb') as f:
            pickle.dump([names, geometries], f)
    print('Retrival of geometries completed for {}.\n'.format(dataset_name))

# =============================================================================
# get_boxes
# =============================================================================
def get_boxes(bounding_box, n_rows, n_cols):
    """
    this function divides a large bounding box for a region into smaller 
    bounding boxes to override memory limit on GEE.

    Parameters
    ----------
    bounding_box : list
        coordinates of the original bounding box.
    n_rows : integer
        number of rows.
    n_cols : integer
        number of columns.

    Returns
    -------
    list
        list of smaller bounding boxes.

    """
    r_vals = np.linspace(bounding_box[0], bounding_box[2], n_cols)                          # get the list of longitudes
    c_vals = np.linspace(bounding_box[1], bounding_box[3], n_rows)                          # get the list of latitudes
    geometries = []                                                                         # create a list to store bounding box coordinates
    for i in range(len(r_vals)-1):
        for j in range(len(c_vals)-1):
            geo_box = ee.Geometry.Rectangle([r_vals[i], c_vals[j], 
                                             r_vals[i+1], c_vals[j+1]])                     # convert bounding box to Rectangle geometry
            geometries.append(geo_box)                                                      # append coordinates of each box to frame
    names = [str(i+1) for i in range(len(geometries))]
    
    return names, geometries

# =============================================================================
# get_data_monthly_batch
# =============================================================================
def get_data_monthly_batch(period, image, band, region_names, region_geometries, 
                   scale, output_dir, scale_a, scale_b, parallel=False):
    """
    this function retrieves data from GEE and resamples the resulting images 
    monthly and stores the result for each region_geometry as a .csv file.

    Parameters
    ----------
    period : list
        period of the desired data: [first_year, last_year]
    image : string
        image snippet from GEE.
    band : string
        desired band.
    region_names : list
        list of names region geometries.
    region_geometries : list
        list of region geometries.
    scale : float
        spatial resolution from GEE.
    output_dir : string
        output directory.
    scale_a: float, optional
        a in y=ax+b for scale correction. The default is 1.
    scale_b: float, optional
        b in y=ax+b for scale correction. The default is 0.
    parallel : boolean, optional
        parallel run. The default is False.

    Returns
    -------
    None.

    """
    
    print('Started retrieving {} data for scale={} using parallelization:'
          .format(os.path.basename(os.path.dirname(output_dir)), scale))
    def correct_scale(df_K, scale_a, scale_b):
        """
        this function corrects scale using y = scale_a*x + scale_b.
        For example, for K to C: y = 0.02x - 273.15

        Parameters
        ----------
        df_K : dataframe
            dataframe with incorrect scale.
        scale_a: float
            a in y=ax+b for scale correction.
        scale_b: float
        b in y=ax+b for scale correction.

        Returns
        -------
        df_C : dataframe
            dataframe with corrected scale.

        """
        # df_C = df_K.multiply(scale_a).add(scale_b).round(4)                                 # correct scale
        df_C = df_K.multiply(scale_a).add(scale_b)                                          # correct scale
        return df_C
    
    def calculate_monthly_mean(image_collection, year):
        """
        this function resamples images in an image collection monthly.

        Parameters
        ----------
        image_collection : imagecollection
            image collection.
        year : integer
            year of data.

        Returns
        -------
        monthly_images : image collection
            resampled image collection.

        """
        months = ee.List.sequence(1, 12)                                                    # Define a sequence of months
        monthly_images = ee.ImageCollection.fromImages(
            months.map(lambda month:
                image_collection.filter(ee.Filter.calendarRange(month, month, 'month'))
                    .mean()
                    .set('system:time_start', ee.Date.fromYMD(year, month, 1).format('YYYY-MM-dd'))
            ).flatten())

        return monthly_images
    
    def get_data_yearly(period, image, band, region_names, region_geometries,
                        scale, output_dir, n, scale_a, scale_b):
        """
        this function retrives data for each year. Due to memory limitation, 
        data retrieval process is done annually.

        Parameters
        ----------
        period : list
            period of the desired data: [first_year, last_year]
        image : string
            image snippet from GEE.
        band : string
            desired band.
        region_names : list
            list of names region geometries.
        region_geometries : list
            list of region geometries.
        scale : float
            spatial resolution from GEE.
        output_dir : string
            output directory.
        n : int
            region index.

        Returns
        -------
        monthly_images : dataframe
            image data for 12 months including lon and lat of points.

        """
        
        ee.Initialize()
        years = range(period[0], period[1]+1)                                                   # create a list of years
        frame1 = []                                                                             # create a list to store resampled data in df_year
        for year in years:                                                                      # loop over years
            collection = ee.ImageCollection(image).select(band)\
                .filterDate('{}-01-01'.format(year), '{}-01-01'.format(year+1))\
                    .filterBounds(region_geometries[n])                                         # import image collection from GEE for one year and clip to geometry
                    
            monthly_images = calculate_monthly_mean(collection, year)                           # Calculate the monthly mean of the images in the collection
            data_list = monthly_images.getRegion(region_geometries[n], scale).getInfo()         # get image data as a list
            data_df = pd.DataFrame(data_list[1:], columns=data_list[0]).round(5)                # get image data as a df
            frame1.append(data_df)                                                              # append annual data to frame1
        
        df_total = pd.concat(frame1)                                                            # concat annual dfs to get df_total
        df_pivot = df_total.pivot_table(values=band, index=['longitude', 'latitude'], 
                                        columns='time', aggfunc='first')                        # reshape df_total based on monthly values
        df_out = correct_scale(df_pivot, scale_a, scale_b)                                      # correct scale
        df_out.sort_values(['longitude','latitude'], ascending=[True,True], 
                           inplace=True, ignore_index=False)                                     # sort by longitude and latitude
        df_out.to_csv(os.path.join(output_dir,'{}.csv'.format(region_names[n])), 
                      index=True)                                                               # save file as *.csv
    
    if os.path.exists(output_dir) == False:                                                     # check if output_dir exists; if not, create one.
        os.mkdir(output_dir)
    
    if parallel:
        Parallel(n_jobs=-1)\
            (delayed(get_data_yearly)\
             (period, image, band, region_names, region_geometries, scale, 
              output_dir, n, scale_a, scale_b)
             for n in tqdm(range(len(region_names)),miniters=1))
    else:
        for n in tqdm(range(len(region_names)),miniters=1):
            get_data_yearly(period, image, band, region_names, region_geometries, 
                            scale, output_dir, n, scale_a, scale_b)
    print('Retrieval of {} data completed for scale={} using parallelization.\n'
          .format(os.path.basename(os.path.dirname(output_dir)), scale))

# =============================================================================
# get_data_daily_batch
# =============================================================================
def get_data_daily_batch(period, image, band, region_names, region_geometries, 
                   scale, output_dir, scale_a=1, scale_b=0, parallel=False):
    """
    this function retrieves data from GEE and resamples the resulting images 
    monthly and stores the result for each region_geometry as a .csv file.

    Parameters
    ----------
    period : list
        period of the desired data: [first_year, last_year]
    image : string
        image snippet from GEE.
    band : string
        desired band.
    region_names : list
        list of names region geometries.
    region_geometries : list
        list of region geometries.
    scale : float
        spatial resolution from GEE.
    output_dir : string
        output directory.
    scale_a: float, optional
        a in y=ax+b for scale correction. The default is 1.
    scale_b: float, optional
        b in y=ax+b for scale correction. The default is 0.
    parallel : boolean, optional
        parallel run. The default is False.

    Returns
    -------
    None.

    """
    
    print('Started retrieving {} data for scale={} using parallelization:'
          .format(os.path.basename(os.path.dirname(output_dir)), scale))
    def correct_scale(df_K, scale_a, scale_b):
        """
        this function corrects scale using y = scale_a*x + scale_b.
        For example, for K to C: y = 0.02x - 273.15

        Parameters
        ----------
        df_K : dataframe
            dataframe with incorrect scale.
        scale_a: float
            a in y=ax+b for scale correction.
        scale_b: float
        b in y=ax+b for scale correction.

        Returns
        -------
        df_C : dataframe
            dataframe with corrected scale.

        """
        df_C = df_K.multiply(scale_a).add(scale_b).round(5)                                 # correct scale
        return df_C
    
    def calculate_daily_mean(image_collection, year):
        """
        This function resamples images in an image collection daily.
    
        Parameters
        ----------
        image_collection : imagecollection
            Image collection.
        year : integer
            Year of data.
    
        Returns
        -------
        daily_images : image collection
            Resampled image collection.
    
        """
        days = ee.List.sequence(1, 365)                                                     # Define a sequence of days
        daily_images = ee.ImageCollection.fromImages(                                       # Map over the months to calculate the monthly mean
            days.map(lambda day:
                image_collection.filter(ee.Filter.calendarRange(day, day, 'day_of_year'))
                .mean()
                .set('system:time_start', ee.Date.fromYMD(year, 1, 1).advance(ee.Number(day).subtract(1), 'day').format('YYYY-MM-dd'))
            ).flatten())
    
        return daily_images

    
    def get_data_yearly(period, image, band, region_names, region_geometries,
                           scale, output_dir, n, scale_a=1, scale_b=0):
        """
        this function retrives data for each year. Due to memory limitation, 
        data retrieval process is done annually.

        Parameters
        ----------
        period : list
            period of the desired data: [first_year, last_year]
        image : string
            image snippet from GEE.
        band : string
            desired band.
        region_names : list
            list of names region geometries.
        region_geometries : list
            list of region geometries.
        scale : float
            spatial resolution from GEE.
        output_dir : string
            output directory.
        n : int
            region index.

        Returns
        -------
        monthly_images : dataframe
            image data for 12 months including lon and lat of points.

        """
        
        ee.Initialize()
        years = range(period[0], period[1]+1)                                                   # create a list of years
        for year in years:                                                                      # loop over years
            collection = ee.ImageCollection(image).select(band)\
                .filterDate('{}-01-01'.format(year), '{}-01-01'.format(year+1))\
                    .filterBounds(region_geometries[n])                                         # import image collection from GEE for one year and clip to geometry
                    
            daily_images = calculate_daily_mean(collection, year)                               # Calculate the monthly mean of the images in the collection
            data_list = daily_images.getRegion(region_geometries[n], scale).getInfo()           # get image data as a list
            data_df = pd.DataFrame(data_list[1:], columns=data_list[0]).round(4)                # get image data as a df
            df_pivot = data_df.pivot_table(values=band, index=['longitude', 'latitude'], 
                                            columns='time', aggfunc='first')                    # reshape df_total based on monthly values
            df_out = correct_scale(df_pivot, scale_a, scale_b)                                  # correct scale
            df_out.sort_values(['longitude','latitude'], ascending=[True,True], 
                               inplace=True, ignore_index=False)                                # sort by longitude and latitude
            df_out.to_csv(os.path.join(output_dir,'{}_{}.csv'
                                       .format(region_names[n], year)), 
                          index=True)
                
    if os.path.exists(output_dir) == False:                                                     # check if output_dir exists; if not, create one.
        os.mkdir(output_dir)
    
    if parallel:
        Parallel(n_jobs=-1)\
            (delayed(get_data_yearly)\
             (period, image, band, region_names, region_geometries, scale, 
              output_dir, n, scale_a, scale_b)
             for n in tqdm(range(len(region_names)),miniters=1))
    else:
        for n in tqdm(range(len(region_names)),miniters=1):
            get_data_yearly(period, image, band, region_names, region_geometries, 
                            scale, output_dir, n)
    print('Retrieval of {} data completed for scale={} using parallelization.\n'
          .format(os.path.basename(os.path.dirname(output_dir)), scale))

# =============================================================================
# download data
# =============================================================================
def download_data(data_file, data_dir, bounding_box, var_ids):
    """
    downloads data based on RS_data.csv and a given bounding box.

    Parameters
    ----------
    data_file : string
        path to RS_data.csv
    data_dir : string
        directory to save downloaded data.
    bounding_box : array
        coordinates of the original bounding box..
    var_ids : array
        ids of variables to download based on RS_data.csv.

    Returns
    -------
    None.

    """
    if os.path.exists(data_dir)==True:
        print(f'Data already exists for {os.path.basename(data_dir)}.')
    else:
        os.mkdir(data_dir)                                                                      # create data_dir
        rs_data = pd.read_csv(data_file)                                                        # read file containing detail for RS data
        n_rows, n_cols = 2,2                                                                    # number of lat and lon values for rows and columns
        region_names, region_geometries = get_boxes(bounding_box, n_rows, n_cols)               # get region names and geometries
        for var_id in var_ids:                                                                  # loop over variables ids
            var = rs_data.loc[var_id]['variable']
            var_dir = os.path.join(data_dir, var)
            if os.path.exists(var_dir) == False:                                                # check if var_dir exists; if not, create one
                print(f'Created variable directory for {var}.')
                os.mkdir(var_dir)
            output_dir = os.path.join(var_dir, '0_original')                                    # create 0_original to store downloaded data
            period = [rs_data.loc[var_id]['period_start'], rs_data.loc[var_id]['period_end']]   # period
            image = rs_data.loc[var_id]['image']                                                # image dataset
            band = rs_data.loc[var_id]['band']                                                  # variable name in dataset
            scale = int(rs_data.loc[var_id]['scale'])                                           # spatial scale
            scale_a = int(rs_data.loc[var_id]['scale_a'])
            scale_b = int(rs_data.loc[var_id]['scale_b'])
            get_data_monthly_batch(period, image, band, region_names, region_geometries, 
                                   scale, output_dir, scale_a, scale_b)                         # download data

# =============================================================================
# regrid 
# =============================================================================
def regrid(input_files, output_files):
    """
    regrids data based on the input with lowest number of points. output files
    have the same number of cells and coordinates.

    Parameters
    ----------
    input_files : list
        list of input csv files.
    output_files : list
        list pf output csv files.

    Returns
    -------
    None.

    """
    df1 = pd.read_csv(input_files[0])
    df2 = pd.read_csv(input_files[1])
    df3 = pd.read_csv(input_files[2])
    df4 = pd.read_csv(input_files[3])
    df5 = pd.read_csv(input_files[4])

    ### identify df with lowest number of points
    df_lens = [len(X) for X in [df1,df2,df3,df4,df5]]
    df_base = pd.read_csv(input_files[df_lens.index(min(df_lens))])

    # filter df1 and df2 and select rows that have the same lat and lon in both
    df1_r = df1[df1[['latitude', 'longitude']].apply(tuple, axis=1).isin(df_base[['latitude', 'longitude']].apply(tuple, axis=1))].reset_index(drop=True)
    df2_r = df2[df2[['latitude', 'longitude']].apply(tuple, axis=1).isin(df_base[['latitude', 'longitude']].apply(tuple, axis=1))].reset_index(drop=True)
    df3_r = df3[df3[['latitude', 'longitude']].apply(tuple, axis=1).isin(df_base[['latitude', 'longitude']].apply(tuple, axis=1))].reset_index(drop=True)
    df4_r = df4[df4[['latitude', 'longitude']].apply(tuple, axis=1).isin(df_base[['latitude', 'longitude']].apply(tuple, axis=1))].reset_index(drop=True)
    df5_r = df5[df5[['latitude', 'longitude']].apply(tuple, axis=1).isin(df_base[['latitude', 'longitude']].apply(tuple, axis=1))].reset_index(drop=True)

    # interpolate each df to replace nan values
    df1_r = df1_r.interpolate(method='nearest', axis=0)
    df2_r = df2_r.interpolate(method='nearest', axis=0)
    df3_r = df3_r.interpolate(method='nearest', axis=0)
    df4_r = df4_r.interpolate(method='nearest', axis=0)
    df5_r = df5_r.interpolate(method='nearest', axis=0)

    for X in output_files:
        if os.path.exists(os.path.dirname(X))==False:
            os.mkdir(os.path.dirname(X))
    # save each file to 1_resampled
    df1_r.to_csv(output_files[0],index=None)
    df2_r.to_csv(output_files[1],index=None)
    df3_r.to_csv(output_files[2],index=None)
    df4_r.to_csv(output_files[3],index=None)
    df5_r.to_csv(output_files[4],index=None)


# =============================================================================
# zscore_batch
# =============================================================================
def zscore_batch(input_dir, output_dir, parallel=False):
    """
    this function reads files from input_dir, computes z-scores for each point 
    and saves the output in output_dir with the same name as input file.

    Parameters
    ----------
    input_dir : string
        input files directory.
    output_dir : string
        output files directory.
    parallel : boolean, optional
        parallel run. The default is False.

    Returns
    -------
    None.

    """
    print('Started calculating Z-score values for {}:'
          .format(os.path.basename(os.path.dirname(output_dir))))
    def zscore_(file, output_dir):
        df_raw = pd.read_csv(file)                                                      # read input file
        unique_points = df_raw[['longitude','latitude']]                                # extract lon and lat
        df_values = df_raw.drop(['longitude','latitude'], axis=1)                       # extract data values
        df_zscore = df_values.apply(zscore, axis=1, result_type='broadcast').round(4)   # calculate z-scores along each row and round by 4 decimals
        df_out = pd.concat([unique_points, df_zscore], axis=1)                          # create df_out from unique_points and z-scores
        df_out.sort_values(['longitude','latitude'], ascending=[True,True], 
                           inplace=True, ignore_index=False)                            # sort by longitude and latitude
        df_out.to_csv(os.path.join(output_dir,os.path.basename(file)), 
                      index=None)                                                       # save df_out to *.csv
        
    if os.path.exists(output_dir) == False:                                             # check if output_dir exists; if not, create one.
        os.mkdir(output_dir)
    
    files = [f.path for f in os.scandir(input_dir) if f.path.endswith('.csv')]          # list all *.csv files in output_dir
    
    if parallel:
        Parallel(n_jobs=-1)\
            (delayed(zscore_)\
                (file, output_dir) for file in tqdm(files, miniters=1))
    else:
        for file in tqdm(files, miniters=1):
            zscore_(file, output_dir)
    print('Calculation of Z-score values completed for {}.\n'
          .format(os.path.basename(os.path.dirname(output_dir))))

# =============================================================================
# spi_batch
# =============================================================================
def spi_batch(input_dir, output_dir, thresh):
    """
    this function calculates SPI values for a dataframe which consists of 
    precipitation values for multiple points.

    Parameters
    ----------
    input_dir : string
        directory for precipitation values.
    output_dir : string
        directory to save SPI values.
    thresh : integer
        SPI interval.

    Returns
    -------
    None.

    """
    print('Started calculating SPI values:')
    def spi(ds, thresh):
        """
        this function computes SPI values from precipitation values as a Series. 
        This code is based on YT tutorial: 
            https://www.youtube.com/watch?v=ruD73APLNAc

        Parameters
        ----------
        ds : Series
            precipitation values.
        thresh : integer
            time interval.

        Returns
        -------
        norm_spi : array
            SPI values.

        """
        ds_ma = ds.rolling(thresh, center=False).mean()                                 # Rolling Mean / Moving Averages
        ds_In = np.log(ds_ma)                                                           # Natural log of moving averages
        ds_In[ np.isinf(ds_In) == True] = np.nan                                        # Change infinity to NaN
        ds_mu = np.nanmean(ds_ma)                                                       # Overall Mean of Moving Averages
        ds_sum = np.nansum(ds_In)                                                       # Summation of Natural log of moving averages
            
        # Computing essentials for gamma distribution
        n = len(ds_In[thresh-1:])                                                       # size of data
        A = np.log(ds_mu) - (ds_sum/n)                                                  # Compute A
        alpha = (1/(4*A))*(1+(1+((4*A)/3))**0.5)                                        # Compute alpha  (a)
        beta = ds_mu/alpha                                                              # Compute beta (scale)
        
        gamma = st.gamma.cdf(ds_ma, a=alpha, scale=beta)                                # Gamma Distribution (CDF)
        
        # Standardized Precipitation Index   (Inverse of CDF)
        norm_spi = st.norm.ppf(gamma, loc=0, scale=1)                                   # loc is mean and scale is standard dev.
        
        return norm_spi
    
    if os.path.exists(output_dir) == False:                                             # check if output_dir exists; if not, create one.
        os.mkdir(output_dir)
    
    files = [f.path for f in os.scandir(input_dir) if f.path.endswith('.csv')]          # list all *.csv files in output_dir
    print('Calculating SPI-{} values:'.format(thresh))
    for file in tqdm(files, miniters=1):                                                # loop over input files
        df_raw = pd.read_csv(file)                                                      # read input file
        unique_points = df_raw[['longitude','latitude']]                                # extract lon and lat
        df_values = df_raw.drop(['longitude','latitude'], axis=1).T                     # extract data values
    
        ##### calculate SPI values
        frame = []                                                                      # create a list to store norm_spi values for each point
        for col in df_values.columns:                                                   # loop over columns (points)
            norm_spi = spi(df_values[col], thresh)                                      # calculate SPI for each point
            frame.append(norm_spi)                                                      # append norm_spi to frame
        
        df_spi = pd.DataFrame(frame).round(4)                                           # create df_spi from monthly SPI values
        df_out = pd.concat([unique_points, df_spi], axis=1)                             # add unique_points and create df_out
        df_out.columns = df_raw.columns
        df_out.sort_values(['longitude','latitude'], ascending=[True,True], 
                           inplace=True, ignore_index=False)                            # sort by longitude and latitude
        df_out.to_csv(os.path.join(output_dir,os.path.basename(file)), 
                      index=None)                                                       # save df_out as *.csv
    print('Calculation of SPI-{} values completed.\n'.format(thresh))

# =============================================================================
# resample_batch
# =============================================================================
def resample_batch(input_dir, output_dir, coords_dir, region_geometries, 
                   method='nearest', new_res=1000, parallel=False):
    """
    this function resamples data in input_dir based on region_geometry and saves 
    the results in output_dir.

    Parameters
    ----------
    input_dir : string
        input files directory.
    output_dir : string
        output directory.
    region_geometries : list
        list of region geometries
    method : string. The default is 'nearest'.
        resampling method.
    new_res : float, optional
        new resolution in meters. The default is 1000m.
    parallel : boolean, optional
        parallel run. The default is False.

    Returns
    -------
    None.

    """
    print('Started resampling of {} data:'
          .format(os.path.basename(os.path.dirname(output_dir))))
    def resample(input_files, output_dir, coords_dir, region_geometries, n, method):
        ee.Initialize()
        ##### get points from input_file
        input_df = pd.read_csv(input_files[n])                                              # read input_file
        x = input_df['longitude']                                                           # get x values from input_file
        y = input_df['latitude']                                                            # get y values from input_file
        
        ##### create a new grid based on bounding box and new resolution
        try:                                                                                # get the region from input_file name
            r = int(os.path.basename(input_files[n]).split('_')[0])                         # daily
        except ValueError:
            r = int(os.path.basename(input_files[n])[:-4])                                  # monthly
        coords = region_geometries[r-1].getInfo()['coordinates'][0]                         # get the coordinates of four corners of the bounding box
        x_min, y_min = coords[0]                                                            # get x and y for bottom left corner
        x_max, y_max = coords[2]                                                            # get x and y for top right corner
        d = 0.1*new_res/11132                                                               # distance interval in m: 0.1degrees = 11132m
        x_new = np.arange(x_min, x_max, d)
        y_new = np.arange(y_min, y_max, d)
        xx, yy = np.meshgrid(x_new, y_new)
        
        frame = []                                                                          # create a list to store resampled data for each interval
        for c in input_df.columns[2:]:                                                      # loop over data in different times
            z_new = griddata((x, y), input_df[c], (xx, yy), method=method)                  # interpolate using IDW method
            frame.append(z_new.ravel())                                                     # append z_new to frame
        
        frame.insert(0,xx.ravel())                                                          # add new x values
        frame.insert(1,yy.ravel())                                                          # add new y values
        
        df_out = pd.DataFrame(frame).T                                                      # create a df from frame
        df_out.columns = input_df.columns                                                   # rename columns to original names
        df_out.sort_values(['longitude','latitude'], ascending=[True,True], 
                           inplace=True, ignore_index=False)                                # sort by longitude and latitude
        df_out.to_csv(os.path.join(output_dir,os.path.basename(input_files[n])), 
                          index=None)                                                       # save df_out as *.csv
        
        ### extract unique lon and lat for future referencing
        uni_lon = pd.unique(df_out['longitude'])                                            # get unique lon
        uni_lat = pd.unique(df_out['latitude'])                                             # get unique lat
        uni_df = pd.DataFrame([uni_lon, uni_lat], index=['longitude','latitude']).T         # create a df of unique lon and lat
        uni_df.to_csv(os.path.join(coords_dir, os.path.basename(input_files[n])),
                      index=None)                                                           # save as *.csv
        
    for d in [output_dir, coords_dir]:
        if os.path.exists(d) == False:                                                      # check if directories exist; if not, create them
            os.mkdir(d)
    
    input_files = [f.path for f in os.scandir(input_dir) if f.path.endswith('.csv')]        # list all *.csv files in input_dir
    
    if parallel:
        Parallel(n_jobs=-1)\
            (delayed(resample)(input_files, output_dir, coords_dir, region_geometries, n, method)
             for n in tqdm(range(len(input_files)),miniters=1))
    else:
        for n in tqdm(range(len(input_files)), miniters=1):
            resample(input_files, output_dir, coords_dir, region_geometries, n, method)
    print('Resampling of {} data into {} resolution completed.\n'
          .format(os.path.basename(os.path.dirname(output_dir)), new_res))

# =============================================================================
# mask_batch 
# =============================================================================
def mask_batch(input_dir, output_dir, mask_dir, filter_values, parallel=False):
    print('Started masking of data based on land cover:')
    def mask_lc(input_files, mask_files, output_dir, filter_values, n):
        df_mask = pd.read_csv(mask_files[n])                                                # read mask_file as *.csv
        df_input = pd.read_csv(input_files[n])                                              # read input_file as *.csv
        df_out = df_input.copy()                                                            # create df_out to save results
        for c in df_input.columns[2:]:                                                      # loop over df_input months
            mask_col = [f for f in df_mask.columns if f.startswith(c[:4])][0]               # get the corresponding column in df_mask
            mask = df_mask[mask_col].isin(filter_values)                                    # find undesired values
            df_out[c][mask] = np.nan                                                        # set undesired values to NaN
        df_out.sort_values(['longitude','latitude'], ascending=[True,True], 
                           inplace=True, ignore_index=False)                                # sort by longitude and latitude
        df_out.to_csv(os.path.join(output_dir,os.path.basename(input_files[n])), 
                      index=None)                                                           # save df_out as *.csv
    
    if os.path.exists(output_dir) == False:                                                 # check if output_dir exists; if not, create one.
        os.mkdir(output_dir)

    input_files = [f.path for f in os.scandir(input_dir) if f.path.endswith('.csv')]        # list all *.csv files in output_dir    
    mask_files = [f.path for f in os.scandir(mask_dir) if f.path.endswith('.csv')]          # list all *.csv files in mask_dir
    
    if parallel:
        Parallel(n_jobs=-1)\
            (delayed(mask_lc)(input_files, mask_files, output_dir, filter_values, n)
             for n in tqdm(range(len(input_files)),miniters=1))
    else:
        for n in tqdm(range(len(input_files)), miniters=1):
            mask_lc(input_files, mask_files, output_dir, filter_values, n)
    print('Masking of data based on land cover completed.\n')

# =============================================================================
# crop_batch
# =============================================================================
def crop_batch(input_dir, output_dir, period, parallel=False):
    """
    this function crops time series based on period. This is done to unify the 
    time range of all inputs to be used in CNN.

    Parameters
    ----------
    input_dir : string
        input directory.
    output_dir : string
        output directory.
    period : list
        desired time range (e.g., ['2003-01-01', '2014-12-01'])
    parallel : boolean, optional
        parallel run. The default is False.

    Returns
    -------
    None.

    """
    print('Started cropping {} time series from {} to {}:'
          .format(os.path.basename(os.path.dirname(output_dir)), period[0], period[1]))
    def crop_df(input_files, output_dir, period, n):
        df_input = pd.read_csv(input_files[n])                                              # read input file
        i1 = df_input.columns.get_loc(period[0])                                            # get index of period start
        i2 = df_input.columns.get_loc(period[1])                                            # get index of period end
        df_out = df_input.iloc[:, 0:2].join(df_input.iloc[:, i1:i2+1], how='outer')         # crop data
        df_out.sort_values(['longitude','latitude'], ascending=[True,True], 
                           inplace=True, ignore_index=False)                                # sort by longitude and latitude
        df_out.to_csv(os.path.join(output_dir,os.path.basename(input_files[n])), 
                      index=None)                                                           # save df_out as *.csv
        
    if os.path.exists(output_dir) == False:                                                 # check if output_dir exists; if not, create one.
        os.mkdir(output_dir)
        
    input_files = [f.path for f in os.scandir(input_dir) if f.path.endswith('.csv')]        # list all *.csv files in input_dir
    
    if parallel:
        Parallel(n_jobs=-1)\
            (delayed(crop_df)(input_files, output_dir, period, n)
             for n in tqdm(range(len(input_files)),miniters=1))
    else:
        for n in tqdm(range(len(input_files)), miniters=1):
            crop_df(input_files, output_dir, period, n)
    print('Cropping {} time series from {} to {} completed.\n'
          .format(os.path.basename(os.path.dirname(output_dir)), period[0], period[1]))
    
# =============================================================================
# stack_batch
# =============================================================================
def stack_monthly_batch(input_dir, output_dir, coords_dir, thresh, parallel=False):
    """
    this function stacks gridded monthly data into 3D numpy arrays with 
    size=(m,h,w) where m=number of months, h=image height, w=image width.

    Parameters
    ----------
    input_dir : string
        input directory.
    output_dir : string
        output directory.
    coords_dir: string
        coordinations directory
    thresh: int
        threshold value in SPI.
    parallel : boolean, optional
        parallel run. The default is False.

    Returns
    -------
    None.

    """
    print('Started stacking monthly {} data into 3D numpy arrays:'
          .format(os.path.basename(os.path.dirname(output_dir))))

    def interpolate_nans(arr):
        kernel = np.ones((3, 3))                                                                # create a 3*3 kernel
        kernel[1, 1] = 0                                                                        # set the center of the kernel to 0
        mask = np.logical_or(np.isnan(arr), np.isinf(arr))                                      # get a mask for Nan, inf, and -inf values
        arr = np.where(mask, np.nan, arr)                                                       # replace all invalid values with NaN
        counts = convolve2d(~mask, kernel, mode='same', boundary='fill', fillvalue=0)
        counts[counts == 0] = 1                                                                 # to avoid division by zero
        filled = convolve2d(np.nan_to_num(arr), kernel, mode='same', 
                            boundary='fill', fillvalue=0) / counts                              # fill NaN values
        filled[np.logical_not(mask)] = arr[np.logical_not(mask)]                                # replace the original nonNaN values
        return np.round(filled,4)
    
    def stack_df(input_files, output_dir, coords_dir, n, thresh=thresh):
        df_input = pd.read_csv(input_files[n])                                                  # read input file
        
        ### extract unique lon and lat for future referencing
        uni_lon = pd.unique(df_input['longitude'])                                              # get unique lon
        uni_lat = pd.unique(df_input['latitude'])                                               # get unique lat
        uni_df = pd.DataFrame([uni_lon, uni_lat], index=['longitude','latitude']).T             # create a df of unique lon and lat
        uni_df.to_csv(os.path.join(coords_dir, os.path.basename(input_files[n])),
                      index=None)                                                               # save as *.csv
        
        frame = []                                                                              # create a list to store data for each month
        for c in df_input.columns[2+thresh-1:]:                                                 # loop over columns
            df_month = df_input[['longitude', 'latitude', c]]                                   # get data for one month along with lon and lat
            map_array = df_month.pivot(index='latitude', columns='longitude', 
                                       values=c).sort_index(ascending=False).values
            map_array_filled = interpolate_nans(map_array)
            frame.append(map_array_filled)
        np.save(os.path.join(output_dir, 
                             os.path.basename(input_files[n])[:-3]+'npy'), frame)               # save array as *.npy

    for t_dir in [output_dir, coords_dir]:                                                      # ckeck for both output_dir and coords_dir
        if os.path.exists(t_dir) == False:                                                      # if t_dir doesn't exit create one
            os.mkdir(t_dir)

    input_files = [f.path for f in os.scandir(input_dir) if f.path.endswith('.csv')]            # list all *.csv files in input_dir
    
    if parallel:
        Parallel(n_jobs=-1)\
            (delayed(stack_df)(input_files, output_dir, coords_dir, n)
             for n in tqdm(range(len(input_files)),miniters=1))
    else:
        for n in tqdm(range(len(input_files)), miniters=1):
            stack_df(input_files, output_dir, coords_dir, n)
    print('Stacking monthly {} data into 3D numpy arrays completed.\n'
          .format(os.path.basename(os.path.dirname(output_dir))))

# =============================================================================
# stack_daily_batch
# =============================================================================
def stack_daily_batch(input_dir, region_names, output_dir, parallel=False):
    """
    this function stacks gridded monthly data into 3D numpy arrays with 
    size=(m,h,w) where m=number of months, h=image height, w=image width.

    Parameters
    ----------
    input_dir : string
        input directory.
    filter_dir_list : list
        list of directories for files to be used as filters.
    output_dir : string
        output directory.
    n_cores : integer, optional
        number of cores. The default is -1.

    Returns
    -------
    None.

    """
    print('Started stacking {} data into 3D numpy arrays:'
          .format(os.path.basename(os.path.dirname(output_dir))))

    def interpolate_nans(arr):
        kernel = np.ones((3, 3))                                                                # create a 3*3 kernel
        kernel[1, 1] = 0                                                                        # set the center of the kernel to 0
        mask = np.logical_or(np.isnan(arr), np.isinf(arr))                                      # get a mask for Nan, inf, and -inf values
        arr = np.where(mask, np.nan, arr)                                                       # replace all invalid values with NaN
        counts = convolve2d(~mask, kernel, mode='same', boundary='fill', fillvalue=0)
        counts[counts == 0] = 1                                                                 # to avoid division by zero
        filled = convolve2d(np.nan_to_num(arr), kernel, mode='same', 
                            boundary='fill', fillvalue=0) / counts                              # fill NaN values
        filled[np.logical_not(mask)] = arr[np.logical_not(mask)]                                # replace the original nonNaN values
        return np.round(filled,4)
    
    def stack_df(input_files, region_names, output_dir, n):
        region_files = [f for f in input_files if int(os.path.basename(f).split('_')[0])==n+1]  # filter files for region n
        
        frame = []                                                                              # create list to store results
        for r in region_files:                                                                  # loop over region_files
            df = pd.read_csv(r)                                                                 # read file r
            for c in df.columns[2:]:                                                            # loop over columns
                df_month = df[['longitude', 'latitude', c]]                                     # get data for one month along with lon and lat
                arr_pivot = df_month.pivot_table(index='longitude', columns='latitude', 
                                                values=c).values                                # get values in a matrix
                arr_filled = interpolate_nans(arr_pivot)                                        # interpolate invalid values
                frame.append(arr_filled)                                                        # append arr_filled to frame
        np.save(os.path.join(output_dir, region_names[n]+'.npy'), frame)                        # save array as *.npy

    if os.path.exists(output_dir) == False:                                                     # if t_dir doesn't exit create one
        os.mkdir(output_dir)

    input_files = [f.path for f in os.scandir(input_dir) if f.path.endswith('.csv')]            # list all *.csv files in input_dir
    
    if parallel:
        Parallel(n_jobs=-1)\
            (delayed(stack_df)(input_files, region_names, output_dir, n)
             for n in tqdm(range(len(region_names)),miniters=1))
    else:
        for n in tqdm(range(len(region_names)), miniters=1):
            stack_df(input_files, region_names, output_dir, n)
    print('Stacking {} data into 3D numpy arrays completed.\n'
          .format(os.path.basename(os.path.dirname(output_dir))))

# =============================================================================
# get_weights
# =============================================================================
def get_weights(files_list, output_dir, parallel=False):
    print('Started calculating weights using PCA:')
    """
    this function calculates weights of each input using PCA.

    Parameters
    ----------
    files_list : list
        list of input directories.
    output_dir : string
        output dir.
    n_cores : int, optional
        number of cores. The default is -1.

    Returns
    -------
    None.

    """
    def w_pca(df):
        """
        this function calculates weights of each column in a df.

        Parameters
        ----------
        df : dataframe
            input df.

        Returns
        -------
        weights : list
            list of weights for each column.

        """
        df_norm = (df - df.mean()) / df.std()                                                   # normalize the data
        n_comp = len(df_norm.columns)                                                           # get number of components
        pca = PCA(n_components=n_comp)                                                          # fit PCA model
        pca.fit(df_norm)
        
        # get the weights of each feature
        weights = pca.explained_variance_ratio_[:n_comp] / sum(pca.explained_variance_ratio_[:n_comp])

        return weights

    def get_weights_monthly(p_files, lst_files, ndvi_files, sm_files, n, output_dir):
        ### read input files
        p = pd.read_csv(p_files[n])                                                             # read P
        lst = pd.read_csv(lst_files[n])                                                         # read LST
        ndvi = pd.read_csv(ndvi_files[n])                                                       # read NDVI
        sm = pd.read_csv(sm_files[n])                                                           # read SM
        
        if not len(p)==len(lst) or len(p)==len(ndvi) or len(p)==len(sm):                        # check length of data points
            min_len = np.min([len(p),len(lst),len(ndvi),len(sm)])
            p = p.iloc[:min_len,:]
            lst = lst.iloc[:min_len,:]
            ndvi = ndvi.iloc[:min_len,:]
            sm = sm.iloc[:min_len,:]

        n_years = int((len(p.columns) - 2)/12)                                                  # get number of years
        frame = []                                                                              # create a list to store weights for each month
        for m in range(12):                                                                     # loop over months
            frame_m = []                                                                        # create a list to store data for each month
            for y in range(n_years):                                                            # loop over years
                frame_m.append([p.iloc[:,y*12+m+2], lst.iloc[:,y*12+m+2], 
                                ndvi.iloc[:,y*12+m+2], sm.iloc[:,y*12+m+2]])                    # get monthly data for each variable
            ### stack and remove NaN values
            df_month = pd.DataFrame(np.hstack(frame_m)).T                                       # stack monthly data to get four columns of data for each input
            df_month = df_month.dropna()                                                        # drop NaN values
            df = df_month.copy()                                                                # make a copy of df_month to remove -inf values
            
            try:
                ### remove -inf values
                neginf_values = np.isneginf(df).sum()                                               # get -inf values in each column
                if neginf_values.sum()>0:                                                           # if there are any -inf values
                    for c in df.columns:                                                            # loop over columns
                        if neginf_values[c]>0:                                                      # find the column with -inf values
                            mask = np.isneginf(df)                                                  # Create a mask of the -inf values
                            df[mask] = np.nan                                                       # Replace -inf values with NaN
                            mean = np.nanmean(df[c])                                                # Calculate the mean of column c ignoring NaN values
                            df = pd.DataFrame(np.nan_to_num(df, nan=mean), 
                                              columns=['P','LST','NDVI','SM'])                      # replace NaN values with column mean
            except:
                print(os.path.basename(p_files[n]))
                            
            weights = w_pca(df)                                                                 # get weights by PCA
            frame.append(weights)                                                               # append weights to frame
        df_out = pd.DataFrame(frame, columns=['P','LST','NDVI','SM'])                           # create a df from frame
        df_out.insert(0, 'Month', np.arange(1,13))                                              # add one column as month
        df_out.to_csv(os.path.join(output_dir, os.path.basename(p_files[n])), index=None)       # save file as *.csv

    if os.path.exists(output_dir) == False:                                                     # check if output_dir exists; if not, create one.
        os.mkdir(output_dir)
        
    ### get list of files
    p_files = [f.path for f in os.scandir(files_list[0]) if f.path.endswith('.csv')]            # list all P files
    lst_files = [f.path for f in os.scandir(files_list[1]) if f.path.endswith('.csv')]          # list all P files
    ndvi_files = [f.path for f in os.scandir(files_list[2]) if f.path.endswith('.csv')]         # list all P files
    sm_files = [f.path for f in os.scandir(files_list[3]) if f.path.endswith('.csv')]           # list all P files

    if parallel:    
        Parallel(n_jobs=-1)\
            (delayed(get_weights_monthly)(p_files, lst_files, ndvi_files, sm_files, 
                                          n, output_dir)
              for n in tqdm(range(len(p_files)),miniters=1))   
    else:
        for n in tqdm(range(len(p_files)), miniters=1):
            get_weights_monthly(p_files, lst_files, ndvi_files, sm_files, n, output_dir)
    print('Calculation of weights for input variables based on PCA completed.\n')

# =============================================================================
# cdi_batch
# =============================================================================
def cdi_batch(input_dir, output_dir, weights_dir, thresh, w='pca', 
              w_values=[0.4,0.2,0.2,0.2], parallel=False):
    print(f'Calculating CDI based on {w} weights:')
    
    def initialize_dir(output_dir):                                                         # create required folders to output results
        for d in ['pca','constant']:
            if os.path.exists(os.path.join(output_dir, d)) == False:                        # check if output_dir exists; if not, create one.
                os.mkdir(os.path.join(output_dir, d))

    def cdi(input_dir, output_dir, files, weights_dir, n):
        ### read files
        t_file = os.path.basename(files[n])
        p = np.load(os.path.join(input_dir, 'P', '4_stacked', t_file))                      # input file for Precipitation
        lst = np.load(os.path.join(input_dir, 'LST', '4_stacked', t_file))                  # input file for Land Surface Temperature
        ndvi = np.load(os.path.join(input_dir, 'NDVI', '4_stacked', t_file))                # input file for NDVI
        sm = np.load(os.path.join(input_dir, 'SM', '4_stacked', t_file))                    # input file for SM from GLDAS
        df_w = pd.read_csv(os.path.join(weights_dir, f'{t_file[:-4]}.csv'))                 # read weight values for region n
        
        frame = []                                                                          # create a list to store monthly CDI values
        n_months = np.size(p, 0)                                                            # get number of months

        if w=='pca':                                                                        # if PCA weights are used
            for m in range(n_months):                                                       # loop over months
                weights = df_w.iloc[(m+thresh-1) % 12, :]                                   # get the weights for the corresponding month
                cdi = p[m,:,:]*weights['P'] + lst[m,:,:]*weights['LST'] + \
                    ndvi[m,:,:]*weights['NDVI'] + sm[m,:,:]*weights['SM']                   # calculate CDI
                frame.append(cdi)                                                           # append CDI for month m to frame

        elif w=='constant':                                                                 # if constant weights are used
            for m in range(n_months):                                                       # loop over months
                weights = df_w.iloc[(m+thresh-1) % 12, :]                                   # get the weights for the corresponding month
                cdi = p[m,:,:]*w_values[0] + lst[m,:,:]*w_values[1] + \
                    ndvi[m,:,:]*w_values[2] + sm[m,:,:]*w_values[3]                         # calculate CDI-SSM
                frame.append(cdi)                                                           # append CDI for month m to frame
        
        cdi_arr = np.stack(frame)                                                           # stack CDI-SSM values to form a 3D array
        
        np.save(os.path.join(output_dir, w, t_file), cdi_arr)                               # save CDI array as *.npy
    
    if os.path.exists(output_dir) == False:                                                 # check if output_dir exists; if not, create one.
        os.mkdir(output_dir)
    
    initialize_dir(output_dir)
    files = [f.path for f in os.scandir(os.path.join(input_dir, 'P', '4_stacked')) 
                if f.path.endswith('.npy')]                                                 # list all predicted SM files
    
    if parallel:
        Parallel(n_jobs=-1)\
            (delayed(cdi)(input_dir, output_dir, files, weights_dir, n)
             for n in tqdm(range(len(files))))
    else:
        for n in tqdm(range(len(files))):
            cdi(input_dir, output_dir, files, weights_dir, n)
    print('Completed calculating CDI.\n')

# =============================================================================
# save_tiff
# =============================================================================
def save_tiff_batch(input_dir, coords_dir, maps_dir, output_dir, start_date, 
                    end_date, arc_flip=False, parallel=False):
    print('Started creating maps as tiff files from CDI:')
    
    def initialize_dir(maps_dir):                                                           # create required folders to output results
        if os.path.exists(maps_dir) == False:                                                     # check if output_dir exists; if not, create one.
            os.mkdir(maps_dir)
        for d in ['pca','constant']:
            if os.path.exists(os.path.join(maps_dir, d)) == False:                          # check if output_dir exists; if not, create one.
                os.mkdir(os.path.join(maps_dir, d))            
    
    def save_tiff(input_files, coords_dir, output_dir, labels, n, arc_flip):
        output_dir = os.path.join(output_dir, f'{os.path.basename(input_files[n])[:-4]}')
        if os.path.exists(output_dir) == False:                                             # check if output_dir exists; if not, create one.
            os.mkdir(output_dir)

        cdi = np.load(os.path.join(input_files[n]))                                         # read CDI 3D array
        coords = pd.read_csv(os.path.join(
            coords_dir, f'{os.path.basename(input_files[n])[:-4]}.csv'))                    # read coordinates
        x = coords['longitude'].values                                                      # get longitudes
        y = coords['latitude'].values                                                       # get latitudes
        for m in range(np.size(cdi,0)):                                                     # loop over months
            grid = cdi[m,:,:]                                                               # read data for month m as a grid
            if arc_flip:                                                                    # if intended for demonstration in ArcGIS
                grid = np.flipud(grid)                                                      # flip grid upside down
            num_rows, num_cols = grid.shape
            left, bottom = x[0], y[0]
            pixel_width, pixel_height = x[1] - x[0], y[0] - y[1]
            transform = rasterio.transform.from_origin(left, bottom, pixel_width, pixel_height)
            output_file = os.path.join(output_dir, labels[m].strftime('%Y-%m-%d')+'.tif')   # create output name for tiff image

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
    
    initialize_dir(maps_dir)
         
    labels = pd.date_range(start=start_date, end=end_date, freq='MS')
    input_files = [f.path for f in os.scandir(input_dir) if f.path.endswith('.npy')]                 # list all input files
    
    if parallel:
        Parallel(n_jobs=-1)\
            (delayed(save_tiff)(input_files, coords_dir, output_dir, labels, n, arc_flip)
             for n in tqdm(range(len(input_files))))
    else:
        for n in tqdm(range(len(input_files)), miniters=1):
            save_tiff(input_files, coords_dir, output_dir, labels, n, arc_flip)
    print('Completed creating tiff files.\n')

# =============================================================================
# save_tiff_range
# =============================================================================
def save_tiff_range(input_file, coords_file, output_file):
    cdi = np.load(input_file)                                                           # read CDI 3D array
    coords = pd.read_csv(coords_file)                                                   # read coords
    x = coords['longitude'].values                                                      # get longitudes
    y = coords['latitude'].values                                                       # get latitudes
    m = 0        
    grid = cdi[m,:,:]                                                                   # read data for month m as a grid
    # change min and max for visualization in ArcGIS Pro
    grid[0] = -5
    grid[-1] = 5
    
    num_rows, num_cols = grid.shape
    left, bottom = x[0], y[0]
    pixel_width, pixel_height = x[1] - x[0], y[1] - y[0]
    transform = rasterio.transform.from_origin(left, bottom, pixel_width, pixel_height)

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



