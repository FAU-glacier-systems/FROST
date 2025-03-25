#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann, Johannes J. Fuerst
# Published under the GNU GPL (Version 3), check the LICENSE file

# import required packages
import argparse
import json
import subprocess
import os
import utm
import math
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import zoom
import scipy.interpolate
import rasterio
from rasterio.windows import from_bounds
from rasterio.merge import merge

"""
TODOs:
- make netCDF file names variables (input_saved.nc, observations.nc)
- check if scale_raster produces equidistant grids???
- constant timeline for obs.: make flexible and link to timeline in observation.nc
"""


def main(rgi_id,
         target_resolution,
         download_oggm_shop_flag,
         download_hugonnet_flag,
         hugonnet_directory,
         year_interval):  # Parse command-line arguments

    # Define the path using os.path.join
    rgi_id_dir = os.path.join('..', '..', 'Data', 'Glaciers', rgi_id)

    # Call functions based on flags
    if download_oggm_shop_flag:
        print(f"Downloading OGGM shop data for RGI ID: {rgi_id}...")
        download_OGGM_shop(rgi_id_dir, rgi_id)
        print("OGGM shop data download completed.")

    if download_hugonnet_flag:
        print(f"Downloading Hugonnet data with the following parameters:")

        print(f"  RGI directory: {rgi_id_dir}")
        print(f"  Year interval: {year_interval}")
        download_hugonnet(rgi_id_dir, year_interval, hugonnet_directory)
        print("Hugonnet data download completed.")

    # Rescale all output netCDF to a given target resolution
    if args.target_resolution:
        print(
            f"  Scale output netCDF to target resolution: {target_resolution}")
        target_resolution = float(target_resolution)
        with Dataset(os.path.join(rgi_id_dir, 'OGGM_shop', 'input_saved.nc'),
                     'r') as scaled_dataset:
            x = scaled_dataset.variables['x'][:]
            resolution = abs(x[1] - x[2])
        if resolution != target_resolution:
            scale_factor = resolution / target_resolution
            print('  Scale factor : ', scale_factor)
            scale_raster(
                os.path.join(rgi_id_dir, 'OGGM_shop', 'input_saved.nc'),
                os.path.join(rgi_id_dir, 'OGGM_shop', 'input_scaled.nc'),
                scale_factor)
            os.rename(os.path.join(rgi_id_dir, 'OGGM_shop', 'input_saved.nc'),
                      os.path.join(rgi_id_dir, 'OGGM_shop', 'input_saved_OGGM.nc'))
            os.rename(os.path.join(rgi_id_dir, 'OGGM_shop', 'input_scaled.nc'),
                      os.path.join(rgi_id_dir, 'OGGM_shop', 'input_saved.nc'))

            scale_raster(
                os.path.join(rgi_id_dir, 'observations.nc'),
                os.path.join(rgi_id_dir, 'observations_scaled.nc'),
                scale_factor)
            os.rename(os.path.join(rgi_id_dir, 'observations.nc'),
                      os.path.join(rgi_id_dir, 'observations_OGGM.nc'))
            os.rename(os.path.join(rgi_id_dir, 'observations_scaled.nc'),
                      os.path.join(rgi_id_dir, 'observations.nc'))
            # scale_raster('input_saved.nc', 'input_scaled.nc', scale_factor)
        else:
            # Make back-up file for original OGGM netCDF
            # Source and destination
            src = os.path.join(rgi_id_dir, 'observations.nc')
            dst = os.path.join(rgi_id_dir, 'observations_OGGM.nc')

            # Check the operating system and use the respective command
            if os.name == 'nt':  # Windows
                cmd = f'copy "{src}" "{dst}"'
            else:  # Unix/Linux
                cmd = f'cp "{src}" "{dst}"'

            # Copy File
            os.system(cmd)

            # Source and destination
            src = os.path.join(rgi_id_dir, 'OGGM_shop', 'input_saved.nc')
            dst = os.path.join(rgi_id_dir, 'OGGM_shop', 'input_saved_OGGM.nc')

            # Check the operating system and use the respective command
            if os.name == 'nt':  # Windows
                cmd = f'copy "{src}" "{dst}"'
            else:  # Unix/Linux
                cmd = f'cp "{src}" "{dst}"'

            # Copy File
            os.system(cmd)


def scale_raster(input_file, output_file, scale_factor):
    """
    script to spatially up- or downsample the OGGMshop standard netCDF file

    Authors: Oskar Herrmann, Johannes J. Fuerst

    Args:
           input_file(str)     - refers to directory & filename of standard OGGMshop netCDF
           scale_factor(float) - values smaller than 1 result in dx_out > dx_in
                                 and vice versa
           output_file(str)    - output is written in same directory as input;
                                 also in netCDF format
                                 file is already within this routine

    Returns:
           none
    """

    # Load the NetCDF file
    with Dataset(input_file, 'r') as input_dataset:
        # Downscale coordinates
        new_x = zoom(input_dataset.variables['x'][:], scale_factor, order=1)
        new_y = zoom(input_dataset.variables['y'][:], scale_factor, order=1)

        # Create output NetCDF file
        with Dataset(output_file, 'w') as scaled_dataset:
            # Create dimensions
            scaled_dataset.createDimension('x', len(new_x))
            scaled_dataset.createDimension('y', len(new_y))

            # Check if 'time' dimension exists
            if 'time' in input_dataset.dimensions:
                scaled_dataset.createDimension('time',
                                               len(input_dataset.dimensions['time']))
                time_var = scaled_dataset.createVariable('time', 'f4', ('time',))
                time_var[:] = input_dataset.variables['time'][:]
                time_var.setncatts(
                    {attr: input_dataset.variables['time'].getncattr(attr)
                     for attr in
                     input_dataset.variables['time'].ncattrs()})

            # Create coordinate variables
            x_var = scaled_dataset.createVariable('x', 'f4', ('x',))
            y_var = scaled_dataset.createVariable('y', 'f4', ('y',))
            x_var[:] = new_x
            y_var[:] = new_y

            # Copy attributes for x and y
            x_var.setncatts(
                {attr: input_dataset.variables['x'].getncattr(attr) for attr in
                 input_dataset.variables['x'].ncattrs()})
            y_var.setncatts(
                {attr: input_dataset.variables['y'].getncattr(attr) for attr in
                 input_dataset.variables['y'].ncattrs()})

            # Copy other variables and downscale
            for var_name in input_dataset.variables:
                if var_name in ['x', 'y', 'time']:
                    continue

                var = input_dataset.variables[var_name]
                dims = var.dimensions

                # Create a new variable in the output dataset
                scaled_var = scaled_dataset.createVariable(var_name, var.datatype,
                                                           dims)

                # Downscale data if it has 'x' and 'y' dimensions
                if 'x' in dims and 'y' in dims:
                    scale_factors = [
                        scale_factor if dim in ['y', 'x'] else 1
                        for dim in dims
                    ]
                    scaled_data = zoom(var[:], scale_factors, order=0)
                    scaled_var[:] = scaled_data
                else:
                    scaled_var[:] = var[:]

                # Copy variable attributes
                scaled_var.setncatts(
                    {attr: var.getncattr(attr) for attr in var.ncattrs()})

            # Copy global attributes (e.g., CRS, title, etc.)
            scaled_dataset.setncatts(
                {attr: input_dataset.getncattr(attr) for attr in
                 input_dataset.ncattrs()})

            # Get global pyproj attribute
            dst_proj = scaled_dataset.pyproj_crs
            # adjust dxdx values§
            dst_proj2 = str(
                scaled_dataset.pyproj_crs.split('dxdy')[0]) + "dxdy\': [" + str(
                1.0 / scale_factor * (input_dataset.variables['x'][1] -
                                      input_dataset.variables['x'][
                                          0])) + ', -' + str(1.0 / scale_factor * (
                    input_dataset.variables['y'][1] -
                    input_dataset.variables['y'][0])) + '],' + "\'pixel " + str(
                scaled_dataset.pyproj_crs.split('pixel')[1])
            # adjust nxny values
            dst_proj3 = str(dst_proj2.split('nxny')[0]) + "nxny\': [" + str(
                len(scaled_dataset.variables['x'][:])) + ', ' + str(
                len(scaled_dataset.variables['y'][:])) + ']' + str(
                "\'dxdy" + str(dst_proj2.split('dxdy')[1]))
            scaled_dataset.setncattr('pyproj_crs', str(dst_proj3))

            # Handle CRS explicitly, if available
            if 'crs' in input_dataset.variables:
                crs_var = input_dataset.variables['crs']
                scaled_crs = scaled_dataset.createVariable('crs', crs_var.datatype)
                scaled_crs.setncatts(
                    {attr: crs_var.getncattr(attr) for attr in crs_var.ncattrs()})
                scaled_dataset.variables['crs'] = crs_var[:]

    print(f"Scaled raster saved to {output_file} with metadata.")


# Function to handle the main logic
def download_OGGM_shop(rgi_id_dir, rgi_id):
    """
    wrapper to call 'igm_run'
    - JSON file needs to specify the oggm_shop routine of IGM
    - function generates generic OGGMshop netcdf

    Authors: Oskar Herrmann

    Args:
           rgi_id(str)    - specify glacier ID

    Returns:
           none
    """

    # Define the params to be saved in params.json
    json_file_path = os.path.join('..', '..', 'Experiments', rgi_id,
                                  'params_download.json')

    with open(json_file_path, 'r') as file:
        params = json.load(file)

    params["oggm_RGI_ID"] = rgi_id
    # Check if the directory exists, and create it if not
    oggm_shop_dir = os.path.join(rgi_id_dir, 'OGGM_shop')

    if not os.path.exists(oggm_shop_dir):
        os.makedirs(oggm_shop_dir)

    # Change directory to the correct location
    original_dir = os.getcwd()
    os.chdir(oggm_shop_dir)

    # Write the params dictionary to the params.json file
    with open('params.json', 'w') as json_file:
        json.dump(params, json_file, indent=4)

    # Run the igm_run command
    subprocess.run(['igm_run', '--param_file', 'params.json'])

    os.chdir(original_dir)


def crop_hugonnet_to_glacier(rgi_region, date_range, hugonnet_dir, oggm_shop_dir,
                             oggm_shop_dataset):
    """
    Fuse multiple dh/dt tiles and crop to a specified OGGM dataset area.

    Authors: Oskar Herrmann, Johannes J. Fuerst

    Args:
        date_range (str): The date range for the dh/dt dataset.
        oggm_shop_dir (str): relative directory of OGGM dataset
        oggm_shop_dataset (xarray.Dataset): OGGM dataset with spatial coordinates.

    Returns:
        np.ndarray: Cropped and filtered dh/dt map.
    """

    # Define the folder containing dh/dt files
    dhdt_folder = os.path.join(hugonnet_dir, 'dhdt')
    dhdt_err_folder = os.path.join(hugonnet_dir, 'dhdt_err')
    print('... retrieving Hugonnet data from: ', hugonnet_dir)

    # Extract UTM coordinates from the NetCDF file (adjust according to your dataset)
    x_coords = oggm_shop_dataset['x'][:]
    y_coords = oggm_shop_dataset['y'][:]
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()

    # Use netCDF file from OGGMshop and extract projection details
    # (no idea what happens if another DEM source is taken - instead of SRTM)
    zone_number = int(oggm_shop_dataset.epsg.split(':')[1][3:5])

    # Determine if glacier is in northern or southern hemisphere
    # for that use y-corner coordinate from OGGM projection details
    if float(oggm_shop_dataset.pyproj_crs.split(':')[2].split(']')[0].split(',')[
                 1]) > 0:
        zone_letter = "N"
    else:
        zone_letter = "S"

    # Determine if glacier is in western or eastern longitude range
    if zone_number <= 30:
        east_west = "W"
    else:
        east_west = "E"

    # Determine maximum and minimum values for longitude and latitude
    x_range = np.array([min_x, min_x, max_x, max_x])
    # ATTENTION: Hugonnet uses 'S' labels for UTM so all y-values are positive for Huggonet
    if zone_letter == "S":
        # ATTENTION: Hugonnet uses 'S' label for UTM zone (EPSG:327??) in southern hemisphere
        # so all y-values are positive
        # (I do not understand why values have to be put negative and no subtraction from 1.0e7 as defined for UTM-South)
        y_range = np.array([-1.0 * max_y, -1.0 * max_y, -1.0 * min_y, -1.0 * min_y])
    else:
        y_range = np.array([min_y, min_y, max_y, max_y])

    # OGGM file format uses exclusive northern hemisphere UTM (EPSG:326??)
    # --> y-coordinate is negative for southern hemisphere
    lat_lon_corner = utm.to_latlon(x_range, y_range, zone_number, "N")
    lat_lon_corner = np.abs(lat_lon_corner)
    min_lat, max_lat = min(lat_lon_corner[0]), max(lat_lon_corner[0])
    min_lon, max_lon = min(lat_lon_corner[1]), max(lat_lon_corner[1])

    # If glacier is in the southern hemispher or in western territory
    # a correction of the tile number by one is necessary
    if zone_letter == "S":
        min_lat += 1
        max_lat += 1

    if east_west == "W":
        min_lon += 1
        max_lon += 1

    # Create a list to store overlapping tile names
    tile_names = []

    # Iterate over possible tiles
    for lat in range(int(min_lat), int(max_lat) + 1):
        for lon in range(int(min_lon), int(max_lon) + 1):
            # Construct the tile name
            tile_name = f'{zone_letter}{lat:02d}{east_west}{lon:03d}'
            tile_names.append(tile_name)

    # Collect all dh/dt files for the specified tiles
    dhdt_files = [os.path.join(dhdt_folder, f'{tile}_{date_range}_dhdt.tif') for tile
                  in tile_names]
    dhdt_err_files = [os.path.join(dhdt_err_folder,
                                   f'{tile}_{date_range}_dhdt_err.tif')
                      for tile in tile_names]

    # Merge dh/dt tiles across UTMzone and crop accordingly for output
    dst_array = tile_merge_reproject(dhdt_files, oggm_shop_dataset)
    dst_err_array = tile_merge_reproject(dhdt_err_files, oggm_shop_dataset)

    # Re-assign
    cropped_map = np.squeeze(dst_array)
    cropped_err_map = np.squeeze(dst_err_array)

    # Replace invalid values (-9999) with NaN
    filtered_map = np.where(cropped_map == -9999, np.nan, cropped_map)
    filtered_err_map = np.where(cropped_err_map == -9999, np.nan, cropped_err_map)

    return filtered_map, filtered_err_map


def tile_merge_reproject(flist, oggm_shop_dataset):
    """
    Script to collect all geoTIF tiles for a certain glacier
    - check all coordinate reference systems (CRS)
    - reproject all tiles into same CRS
    - merge all reprojected tiles
    - crop to target region (as defined by OGGMshop standard netCDF)
    (
     large portion of the routines is taken from OGGM routine 'hugonnet_maps.py',
     particularly from function 'hugonnet_to_gdir'
    )

    Authors: Oskar Herrmann, Johannes J. Fuerst

    Args:
           flist(str)         - file list of all relevant tiles for a specific glacier
           oggm_shop_dataset  - OGGM dataset loaded from a netCDF

    Returns:
           none
    """

    from packaging.version import Version
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio import MemoryFile
    try:
        # rasterio V > 1.0
        from rasterio.merge import merge as merge_tool
    except ImportError:
        from rasterio.tools.merge import merge as merge_tool

    # A glacier area can cover more than one tile:
    if len(flist) == 1:
        dem_dss = [rasterio.open(flist[0])]  # if one tile, just open it
        file_crs = dem_dss[0].crs
        dhdt_data = rasterio.band(dem_dss[0], 1)
        if Version(rasterio.__version__) >= Version('1.0'):
            src_transform = dem_dss[0].transform
        else:
            src_transform = dem_dss[0].affine
        nodata = dem_dss[0].meta.get('nodata', None)
    else:
        dem_dss = [rasterio.open(s) for s in flist]  # list of rasters

        # make sure all files have the same crs and reproject if needed;
        # defining the target crs to the one most commonly used, minimizing
        # the number of files for reprojection
        crs_list = np.array([dem_ds.crs.to_string() for dem_ds in dem_dss])
        unique_crs, crs_counts = np.unique(crs_list, return_counts=True)
        file_crs = rasterio.crs.CRS.from_string(
            unique_crs[np.argmax(crs_counts)])

        if len(unique_crs) != 1:
            # more than one crs, we need to do reprojection
            memory_files = []
            for i, src in enumerate(dem_dss):
                if file_crs != src.crs:
                    transform, width, height = calculate_default_transform(
                        src.crs, file_crs, src.width, src.height, *src.bounds)
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'crs': file_crs,
                        'transform': transform,
                        'width': width,
                        'height': height
                    })

                    reprojected_array = np.empty(shape=(src.count, height, width),
                                                 dtype=src.dtypes[0])
                    # just for completeness; even the data only has one band
                    for band in range(1, src.count + 1):
                        reproject(source=rasterio.band(src, band),
                                  destination=reprojected_array[band - 1],
                                  src_transform=src.transform,
                                  src_crs=src.crs,
                                  dst_transform=transform,
                                  dst_crs=file_crs,
                                  resampling=Resampling.nearest)

                    memfile = MemoryFile()
                    with memfile.open(**kwargs) as mem_dst:
                        mem_dst.write(reprojected_array)
                    memory_files.append(memfile)
                else:
                    memfile = MemoryFile()
                    with memfile.open(**src.meta) as mem_src:
                        mem_src.write(src.read())
                    memory_files.append(memfile)

            with rasterio.Env():
                datasets_to_merge = [memfile.open() for memfile in memory_files]
                nodata = datasets_to_merge[0].meta.get('nodata', None)
                dhdt_data, src_transform = merge_tool(datasets_to_merge,
                                                      nodata=nodata)
        else:
            # only one single crs occurring, no reprojection needed
            nodata = dem_dss[0].meta.get('nodata', None)
            dhdt_data, src_transform = merge_tool(dem_dss, nodata=nodata)

    # Read global attributes from OGGMshop netcdf (as written by IGM oggm_shop.py)
    dst_array = np.zeros(np.shape(oggm_shop_dataset['usurf'][:]))
    dst_crs = oggm_shop_dataset.epsg
    dst_transform = rasterio.transform.from_origin(float(
        oggm_shop_dataset.pyproj_crs.split(':')[2].split('[')[1].split(',')[0]),
        float(
            oggm_shop_dataset.pyproj_crs.split(
                ':')[2].split(']')[
                0].split(',')[1]),
        abs(float(
            oggm_shop_dataset.pyproj_crs.split(
                ':')[4].split('[')[
                1].split(',')[0])),
        abs(float(
            oggm_shop_dataset.pyproj_crs.split(
                ':')[4].split(']')[
                0].split(',')[1])))

    resampling = Resampling.bilinear

    # crop an reproject to target netCDF
    with MemoryFile() as dest:
        reproject(
            # Source parameters
            source=dhdt_data,
            src_crs=file_crs,
            src_transform=src_transform,
            src_nodata=nodata,
            # Destination parameters
            destination=dst_array,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            # Configuration
            resampling=resampling)
        dest.write(dst_array)

    for dem_ds in dem_dss:
        dem_ds.close()

    return dst_array


def interpolate_nans(grid):
    """
    Script to bilinearly interpolate NaNs in input field

    Authors: Oskar Herrmann

    Args:
           grid(numpy array)              - file list of all relevant tiles for a specific glacier

    Returns:
           grid_interpolated(numpy array) - same array as input (created by scipy)
    """
    # Get x, y coordinates of valid values
    x, y = np.indices(grid.shape)
    valid_mask = ~np.isnan(grid)  # Mask of non-NaN values

    # Interpolate NaN values using 'linear' method
    grid_interpolated = scipy.interpolate.griddata(
        (x[valid_mask], y[valid_mask]),  # Points with valid values
        grid[valid_mask],  # Known values
        (x, y),  # Grid of all points
        method='linear'  # Linear interpolation
    )

    return grid_interpolated


def download_hugonnet(rgi_id_dir, year_interval, hugonnet_directory):
    """
    Script to bilinearly interpolate NaNs in input field
    - it creates a netCDF file (observations.nc)

    Authors: Oskar Herrmann

    Args:
           rgi_id_dir(str)    - relative directory of IGM input folder
                                (expectation is filename 'input_saved.nc'; TO DO)
           year_interval(int) - time interval of Hugonnet input files (5 or 20 years)
           hugonnet_firectory (str) - absolute or relative path to Hugonnet dat on local drive

    Returns:
           none
    """

    # Retrieve relative directory path for OGGMshop
    oggm_shop_dir = os.path.join(rgi_id_dir, 'OGGM_shop')

    # Join directory and filename
    oggm_shop_file = os.path.join(oggm_shop_dir, 'input_saved.nc')

    # Load file from oggm_shop and retrieve relevant variables
    oggm_shop_dataset = Dataset(oggm_shop_file, 'r')
    icemask_2000 = oggm_shop_dataset['icemask'][:]
    usurf_2000 = oggm_shop_dataset['usurf'][:]
    thk_2000 = oggm_shop_dataset['thkinit'][:]

    # List folder names depending on time period
    rgi_region = rgi_id_dir.split('/')[-1].split("-")[3]

    # Define time difference between obserations
    # (still constant) TODO: make flexible and link to timeline in observation.nc
    data_interval = 20
    if data_interval == 20:
        folder_names = [rgi_region + '_rgi60_2000-01-01_2020-01-01']

    elif data_interval == 5:
        folder_names = [rgi_region + '_rgi60_2000-01-01_2005-01-01',
                        rgi_region + '_rgi60_2005-01-01_2010-01-01',
                        rgi_region + '_rgi60_2010-01-01_2015-01-01',
                        rgi_region + '_rgi60_2015-01-01_2020-01-01']

    else:
        raise ValueError(
            'Invalid time period: {}. Please choose either 5 or 20.'.format(
                year_interval))

    # Load dhdts data sets
    dhdts = []
    dhdts_err = []
    for folder_name in folder_names:
        # Load dhdt
        date_range = folder_name.split('_', 2)[-1]

        # Set local file path to Hugonnet data, if default is set
        if hugonnet_directory == '../../Data/Hugonnet/':
            rgi_region = rgi_id_dir.split('/')[-1].split("-")[3]
            fname = f'{rgi_region}_rgi60_{date_range}'
            hugonnet_dir = os.path.join('..', '..', 'Data', 'Hugonnet', fname)
        else:
            hugonnet_dir = str(hugonnet_directory)

        ### MERGE TILES AND CROP to oggmshop area ###
        print('Hugonnet dh/dt filename : ', folder_name)
        cropped_dhdt, cropped_dhdt_err = crop_hugonnet_to_glacier(rgi_region,
                                                                  date_range,
                                                                  hugonnet_dir,
                                                                  oggm_shop_dir,
                                                                  oggm_shop_dataset)

        dhdt_masked = cropped_dhdt[::-1] * icemask_2000
        dhdts.append(dhdt_masked)

        dhdt_err_masked = cropped_dhdt_err[::-1] * icemask_2000
        dhdts_err.append(dhdt_err_masked)

    usurf_change = [usurf_2000]  # initialise with 2000 state #TODO ASTER ?
    dhdt_change = [np.zeros_like(usurf_2000)]
    dhdt_err_change = [np.zeros_like(usurf_2000)]
    usurf_err_change = []  # TODO

    bedrock = usurf_2000 - thk_2000
    year_range = np.arange(2000, 2021, data_interval)

    for i, year in enumerate(year_range[1:]):
        # compute surface change based on dhdt and provide uncertainties
        # change the dhdt field every year_interval

        dhdt_index = math.floor((year - 2001) / data_interval)
        dhdt = dhdts[dhdt_index]

        # check if an interpolation is required (only if both NaNs and actually valid values exist)
        if np.nansum(~np.isnan(dhdt)) > 0 and np.nansum(np.isnan(dhdt)) > 0:
            dhdt = interpolate_nans(dhdt)
        dhdt = np.where(icemask_2000 == 1, dhdt, 0)
        dhdt_change.append(dhdt)

        # either bedrock or last usurf + current dhdt
        usurf = np.maximum(bedrock, usurf_change[-1] + dhdt * data_interval)
        usurf_change.append(usurf)

        # compute uncertainty overtime
        dhdt_err = dhdts_err[dhdt_index]

        # check if an interpolation is required (only if both NaNs and actually valid values exist)
        if np.nansum(~np.isnan(dhdt_err)) > 0 and np.nansum(np.isnan(dhdt_err)) > 0:
            dhdt_err = interpolate_nans(dhdt_err)

        dhdt_err = np.where(icemask_2000 == 1, dhdt_err, 0)
        dhdt_err_change.append(dhdt_err)

        # assuming the error is termporal independet
        # the square root of the sum of variance should be the right err for the
        # surface
        # usurf_err_new = dhdt_err * year_interval / 2

        if not usurf_err_change:
            usurf_err = dhdt_err * data_interval / 2
        else:
            usurf_err = ((dhdt_err_change[-2] * data_interval / 2
                          + dhdt_err_change[-1] * data_interval / 2)) / 2

        usurf_err_change.append(usurf_err)

    # usurf error of final year
    usurf_err = dhdt_err_change[-1] * data_interval / 2
    usurf_err_change.append(usurf_err)

    # transform to numpy array
    usurf_change = np.array(usurf_change)
    usurf_err_change = np.array(usurf_err_change)
    dhdt_change = np.array(dhdt_change)
    dhdt_err_change = np.array(dhdt_err_change)

    # compute velocity magnitude
    uvelo = oggm_shop_dataset.variables['uvelsurfobs'][:]
    vvelo = oggm_shop_dataset.variables['vvelsurfobs'][:]
    velo = np.sqrt(uvelo ** 2 + vvelo ** 2)

    # Create a new netCDF file
    observation_file = os.path.join(rgi_id_dir, 'observations.nc')
    with Dataset(observation_file, 'w') as merged_dataset:
        # Create dimensions
        merged_dataset.createDimension('time', len(year_range))
        merged_dataset.createDimension('x', oggm_shop_dataset.dimensions['x'].size)
        merged_dataset.createDimension('y', oggm_shop_dataset.dimensions['y'].size)

        # Create variables
        time_var = merged_dataset.createVariable('time', 'f4', ('time',))
        x_var = merged_dataset.createVariable('x', 'f4', ('x',))
        y_var = merged_dataset.createVariable('y', 'f4', ('y',))
        usurf_var = merged_dataset.createVariable('usurf', 'f4', ('time', 'y', 'x'))
        usurf_err_var = merged_dataset.createVariable('usurf_err', 'f4',
                                                      ('time', 'y',
                                                       'x'))
        icemask_var = merged_dataset.createVariable('icemask', 'f4',
                                                    ('time', 'y', 'x'))
        dhdt_var = merged_dataset.createVariable('dhdt', 'f4', ('time', 'y', 'x'))
        dhdt_err_var = merged_dataset.createVariable('dhdt_err', 'f4',
                                                     ('time', 'y', 'x'))
        velsurf_mag_var = merged_dataset.createVariable('velsurf_mag', 'f4',
                                                        ('time', 'y', 'x'))

        # Assign data to variables
        time_var[:] = year_range
        x_var[:] = oggm_shop_dataset.variables['x'][:]
        y_var[:] = oggm_shop_dataset.variables['y'][:]
        usurf_var[:] = usurf_change
        usurf_err_var[:] = usurf_err_change
        icemask_var[:] = icemask_2000
        dhdt_var[:] = dhdt_change
        dhdt_err_var[:] = dhdt_err_change
        velsurf_mag_var[:] = velo

        # Write globale attribute in OGGMshop netCDF file
        dst_crs = oggm_shop_dataset.epsg
        dst_proj = oggm_shop_dataset.pyproj_crs
        merged_dataset.setncattr('pyproj_crs', str(dst_proj))
        merged_dataset.setncattr('epsg', str(dst_crs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script generates params.json for downloading data with '
                    'oggm shop as the igm module and runs igm_run.'
    )

    # Add argument for RGI ID
    parser.add_argument('--rgi_id', type=str,
                        default='RGI2000-v7.0-G-11-01706',
                        help='The RGI ID of the glacier to be calibrated '
                             '(default: RGI2000-v7.0-G-11-01706).')

    # Add argument for specific target resolution
    parser.add_argument('--target_resolution', type=float,
                        help='user-specific resolution for IGM run [meters] '
                             '(defaul: 100m)')

    # Add flags to control function execution
    parser.add_argument('--download_oggm_shop', action='store_true',
                        help='Flag to control execution of download_OGGM_shop.')

    parser.add_argument('--download_hugonnet', action='store_true',
                        help='Flag to control execution of download_Hugonnet.')

    # Add argument to specify user directory for Hugonnet elevation changes
    parser.add_argument('--hugonnet_directory', type=str,
                        default='../../Data/Hugonnet/',
                        help='User-specific directory on your local file system '
                             '(default: ../../Data/Hugonnet/).')

    # select between 5-year or 20-year dhdt
    parser.add_argument('--year_interval', type=int, default=20,
                        help='Select between 5-year or 20-year dhdt (5, 20)')

    # Parse arguments
    args = parser.parse_args()

    main(rgi_id=args.rgi_id,
         target_resolution=args.target_resolution,
         download_oggm_shop_flag=args.download_oggm_shop,
         download_hugonnet_flag=args.download_hugonnet,
         hugonnet_directory=args.hugonnet_directory,
         year_interval=args.year_interval
         )
