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
import shutil
import yaml
import xarray as xr
import rioxarray

"""
TODOs:
- make netCDF file names variables (input_saved.nc, observations.nc)
- check if scale_raster produces equidistant grids???
- constant timeline for obs.: make flexible and link to timeline in observation.nc
"""


def main(rgi_id,
         rgi_id_dir,
         smb_model,
         target_resolution,
         oggm_shop,
         hugonnet,
         hugonnet_directory,
         year_interval,
         zone_letter):  # Parse command-line arguments

    # Create output folder
    os.makedirs(rgi_id_dir, exist_ok=True)

    # SMB model
    if str(smb_model) == "ELA":
        flag_OGGM_climate=False
    elif str(smb_model) == "TI":
        flag_OGGM_climate = True
    else:
        flag_OGGM_climate = False

    # Call functions based on flags
    if oggm_shop:
        print(f"Downloading OGGM shop data for RGI ID: {rgi_id}...")
        download_OGGM_shop(rgi_id, rgi_id_dir,flag_OGGM_climate)
        print("OGGM shop data download completed.")

    if hugonnet:
        # check if OGGM original input file exists already
            # Define input and output file names
        input_orig_file = os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input_OGGM_orig.nc')
        input_file = os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input.nc')
        if os.path.exists(input_orig_file):
            print('OGGM original file exists.')
            os.remove(input_file)
            os.rename(input_orig_file, input_file)
        else:
            print('OGGM original file does not exist')

        print(f"Downloading Hugonnet data with the following parameters:")
        print(f"  RGI directory: {rgi_id_dir}")
        print(f"  Year interval: {year_interval}")
        download_hugonnet(rgi_id_dir, year_interval, hugonnet_directory, zone_letter)
        print("Hugonnet data download completed.")

    # TODO
    # Check input file from download for consistency
    # - no 'NaN' in the thickness fields (thkinit, thk)
    # - 90% percentile of observed velocities must be above 10m/yr (mangitude of both x- and y-components)

    # Define input and output file names
    input_orig_file = os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input_OGGM_orig.nc')
    input_file = os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input.nc')
    if os.path.exists(input_orig_file):
        print('OGGM original file exists.')
    else:
        print('OGGM original file does not exist')
        print(input_file)
        os.rename(input_file, input_orig_file)
   
    # Open the input netCDF file in read mode
    with Dataset(input_orig_file, 'r') as src:
        # Create a new netCDF file in write mode
        with Dataset(input_file, 'w') as dst:
   
            # Copy all dimensions from the source file to the destination file
            for name, dimension in src.dimensions.items():
                dst.createDimension(name,
                                    len(dimension) if not dimension.isunlimited() else None)
   
            # Copy all variables from the source file to the destination file
            for name, variable in src.variables.items():
                # dst_var = dst.createVariable(name, variable.datatype, variable.dimensions)
                # dst_var[:] = variable[:]  # Copy variable data
   
                # Check if variable is 'thi' and its original values are all NaN
                if name == 'thk':
                    # Set values to zero where they are NaN
                    original_data = variable[:]
                    # Create a mask for NaN values and set them to zero
                    dst_var = dst.createVariable(name, variable.datatype, variable.dimensions)
                    dst_var[:] = np.where(np.abs(original_data) > 10000, 0, original_data)
                    dst_var[:] = np.where(np.abs(original_data) < 0, 0, dst_var)
                elif name == 'thkinit':
                    # Set values to zero where they are NaN
                    original_data = variable[:]
                    # Create a mask for NaN values and set them to zero
                    dst_var = dst.createVariable(name, variable.datatype, variable.dimensions)
                    dst_var[:] = np.where(np.abs(original_data) > 10000, 0, original_data)
                    dst_var[:] = np.where(np.abs(original_data) < 0, 0, dst_var)
                elif name == 'uvelsurfobs':
                    # Set values to zero where they are NaN
                    original_data = np.array(variable[:])
                    modified_data = np.where(np.abs(original_data) == 0, np.nan, original_data)
                    if np.nansum(np.nansum(modified_data)) != 0 and not(np.isnan(np.nansum(np.nansum(modified_data)))) :
                        if np.nanpercentile(np.abs(original_data.flatten()), 90) > 10.0:
                            # Create a mask for NaN values and set them to zero
                            dst_var = dst.createVariable(name, variable.datatype, variable.dimensions)
                            dst_var[:] = np.where(np.abs(original_data) == 0, np.nan, original_data)
                elif name == 'vvelsurfobs':
                    # Set values to zero where they are NaN
                    original_data = np.array(variable[:])
                    modified_data = np.where(np.abs(original_data) == 0, np.nan, original_data)
                    if np.nansum(np.nansum(modified_data)) != 0 and not(np.isnan(np.nansum(np.nansum(modified_data)))) :
                        #print('PERCENTILE PERCENTILE 90 : ',np.nanpercentile(np.abs(modified_data.flatten()), 90))
                        if np.nanpercentile(np.abs(modified_data.flatten()), 90) > 10.0:
                            # Create a mask for NaN values and set them to zero
                            dst_var = dst.createVariable(name, variable.datatype, variable.dimensions)
                            dst_var[:] = np.where(np.abs(original_data) == 0, np.nan,original_data)
                else:
                    dst_var = dst.createVariable(name, variable.datatype,variable.dimensions)
                    dst_var[:] = variable[:]  # Copy variable data
   
            # Copy global attributes from the source file to the destination file
            dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

    # Rescale all output netCDF to a given target resolution
    # check if target resolution is defined as a float
    try:
        float(target_resolution)
        if np.isnan(float(target_resolution)) or np.isinf(float(target_resolution)):
            target_resolution_float = False
        else:
            target_resolution_float = True
    except ValueError:
        print(
            "--target resolution is not a float. Standard OGGM resolution is taken.")
        target_resolution_float = False

    if target_resolution_float:

        input_nc = os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input.nc')
        obs_nc = os.path.join(rgi_id_dir, 'observations.nc')

        with Dataset(input_nc, 'r') as ds:
            x = ds.variables['x'][:]
            resolution = abs(x[1] - x[2])

        if resolution != target_resolution:
            print(f"  Scale to target resolution: {target_resolution}")
            scale_factor = resolution / target_resolution
            print(f"  Scale factor: {scale_factor:.3f}")

            # scale input.nc
            scale_raster(input_nc, input_nc.replace('.nc', '_scaled.nc'),
                         scale_factor)
            shutil.move(input_nc, input_nc.replace('.nc', '_OGGM.nc'))
            shutil.move(input_nc.replace('.nc', '_scaled.nc'), input_nc)

            # scale observations.nc
            scale_raster(obs_nc, obs_nc.replace('.nc', '_scaled.nc'), scale_factor)
            shutil.move(obs_nc, obs_nc.replace('.nc', '_OGGM.nc'))
            shutil.move(obs_nc.replace('.nc', '_scaled.nc'), obs_nc)


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
            x_attrs = {
                attr: input_dataset.variables['x'].getncattr(attr)
                for attr in input_dataset.variables['x'].ncattrs()
                if attr != '_FillValue'
            }
            x_var.setncatts(x_attrs)

            y_attrs = {
                attr: input_dataset.variables['y'].getncattr(attr)
                for attr in input_dataset.variables['y'].ncattrs()
                if attr != '_FillValue'
            }
            y_var.setncatts(y_attrs)

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
                attrs_to_copy = {
                    attr: var.getncattr(attr)
                    for attr in var.ncattrs()
                    if attr != '_FillValue'
                }
                scaled_var.setncatts(attrs_to_copy)

            # Copy global attributes (e.g., CRS, title, etc.)
            scaled_dataset.setncatts(
                {attr: input_dataset.getncattr(attr) for attr in
                 input_dataset.ncattrs()})

            # TODO
            ## Get global pyproj attribute
            #dst_proj = scaled_dataset.pyproj_crs
            ## adjust dxdx values§
            #dst_proj2 = str(
            #    scaled_dataset.pyproj_crs.split('dxdy')[0]) + "dxdy\': [" + str(
            #    1.0 / scale_factor * (input_dataset.variables['x'][1] -
            #                          input_dataset.variables['x'][
            #                              0])) + ', -' + str(1.0 / scale_factor * (
            #        input_dataset.variables['y'][1] -
            #        input_dataset.variables['y'][0])) + '],' + "\'pixel " + str#(
            #    scaled_dataset.pyproj_crs.split('pixel')[1])
            ## adjust nxny values
            #dst_proj3 = str(dst_proj2.split('nxny')[0]) + "nxny\': [" + str(
            #    len(scaled_dataset.variables['x'][:])) + ', ' + str(
            #    len(scaled_dataset.variables['y'][:])) + ']' + str(
            #    "\'dxdy" + str(dst_proj2.split('dxdy')[1]))
            #scaled_dataset.setncattr('pyproj_crs', str(dst_proj3))
            #
            ## Handle CRS explicitly, if available
            #if 'crs' in input_dataset.variables:
            #    crs_var = input_dataset.variables['crs']
            #    scaled_crs = scaled_dataset.createVariable('crs', crs_var.datatype)
            #    scaled_crs.setncatts(
            #        {attr: crs_var.getncattr(attr) for attr in crs_var.ncattrs()})
            #    scaled_dataset.variables['crs'] = crs_var[:]

    print(f"Scaled raster saved to {output_file} with metadata.")


# Function to handle the main logic
def download_OGGM_shop(rgi_id, rgi_id_dir, flag_OGGM_climate):
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



    # Check if the directory exists, and create it if not
    preprocess_dir = os.path.join(rgi_id_dir, 'Preprocess')
    exp_dir = os.path.join(preprocess_dir, 'experiment')
    os.makedirs(exp_dir, exist_ok=True)

    # Change directory to the correct location
    original_dir = os.getcwd()
    os.chdir(preprocess_dir)

    # create params.yaml
    params = {
        "core": {
            "url_data": ""
        },
        "defaults": [
            {"override /inputs": ["oggm_shop", "local"]},
            {"override /processes": []},
            {"override /outputs": []}
        ],
        "inputs": {
            "oggm_shop": {
                "RGI_ID": rgi_id,
                "thk_source": "millan_ice_thickness",
                "incl_glathida": True,
            }
        },
        "outputs": {},
        "processes": {}
    }

    # Write YAML
    with open(os.path.join('experiment', 'params_oggm_shop.yaml'), 'w') as f:
        f.write("# @package _global_\n")
        yaml.dump(params, f, sort_keys=False)

    # Run the igm_run command
    import sys
    from igm.igm_run import main as igm_main
    sys.argv = ["igm_run", "+experiment=params_oggm_shop"]
    igm_main()
    # subprocess.run(["igm_run", "+experiment=params"], check=True)
    # TODO remove unnecessary files

    if flag_OGGM_climate:
        # Rescue 'climate_historical.nc' created by oggm_shop.py
        # this option requires that "oggm_remove_RGI_folder": false
        #print('flag_OGGM_climate', flag_OGGM_climate, type(flag_OGGM_climate))
        if flag_OGGM_climate:
            print('OGGM climate data is retained for SMB model')
            os.system('pwd')
            os.listdir('./')
            src = os.path.join('.', 'data', rgi_id,'climate_historical.nc')
            dst = os.path.join('..', 'climate_historical.nc')

            # Check the operating system and use the respective command
            if os.name == 'nt':  # Windows
                cmd = f'copy "{src}" "{dst}"'
            else:  # Unix/Linux
                cmd = f'cp "{src}" "{dst}"'

            # Copy File
            os.system(cmd)

    # Replace the usurf with NASADEM  data
    dem_path = f"data/{rgi_id}/NASADEM/dem.tif"
    input_nc = 'data/input.nc'

    # Load and update usurf in input.nc after IGM creates it
    if os.path.exists(dem_path) and os.path.exists(input_nc):
        # Read DEM
        dem = rioxarray.open_rasterio(dem_path)
        dem = dem.squeeze()

        # Read input.nc
        with Dataset(input_nc, 'r+') as input_dataset:
            # Update usurf with DEM data - flip y axis
            input_dataset['usurf'][:] = dem.data[::-1]


    os.chdir(original_dir)


def crop_hugonnet_to_glacier(date_range, hugonnet_dir, oggm_shop_dataset,
                             zone_letter):
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

    # TODO
    # Use netCDF file from OGGMshop and extract projection details
    # (no idea what happens if another DEM source is taken - instead of SRTM)
    zone_number = int(oggm_shop_dataset.pyproj_srs.split('=')[2][0:2])
    # Convert to CRS object
    from pyproj import CRS
    crs = CRS.from_proj4(oggm_shop_dataset.pyproj_srs)

    # Get the EPSG code
    ## use EPSG number from proj_srs
    ## (for some glaciers proj_srs is corrupted)
    #epsg_code = crs.to_epsg()
    # use EPSG number from reference DEM
    # (might be more robust if on southern hemisphere)
    epsg_code = oggm_shop_dataset.epsg.split(':')[1]
    hemisphere_code = epsg_code[2]
    oggm_shop_dataset.epsg = f"EPSG:{epsg_code}"

    if int(hemisphere_code) == 6:
        print('UTM hemisphere code is 6 (northern hemisphere).')
    elif int(hemisphere_code) == 7:
        print('UTM hemisphere code is 7 (southern hemisphere).')
    else:
        print('UTM hemisphere code has no expected value (6 or 7) but is : ', hemisphere_code)
        print('EPSG code : ', epsg_code)

    # set zone letter
    if min_y > 0 and int(hemisphere_code) == 6:
        zone_letter = "N"
    elif max_y < 0 and int(hemisphere_code) == 7:
        zone_letter = "N"
    else:
        zone_letter = "S"

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
    lat_lon_corner = utm.to_latlon(x_range, y_range, zone_number, zone_letter)
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
    x = oggm_shop_dataset['x'][:]
    y = oggm_shop_dataset['y'][:]
    dst_transform = rasterio.transform.from_origin(x[0], y[-1], x[1] - x[0],
                                                   y[1] - y[0])

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


def download_hugonnet(rgi_id_dir, year_interval, hugonnet_directory, zone_letter):
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
    preprocess_dir = os.path.join(rgi_id_dir, 'Preprocess')

    # Join directory and filename
    oggm_shop_file = os.path.join(preprocess_dir, 'data', 'input.nc')

    # Load file from oggm_shop and retrieve relevant variables
    oggm_shop_dataset = Dataset(oggm_shop_file, 'a')
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
        dhdt, dhdt_err = crop_hugonnet_to_glacier(date_range=date_range,
                                                  hugonnet_dir=hugonnet_dir,
                                                  oggm_shop_dataset=oggm_shop_dataset,
                                                  zone_letter=zone_letter)

        dhdt_masked = dhdt[::-1] * icemask_2000
        dhdts.append(dhdt_masked)

        dhdt_err_masked = dhdt_err[::-1] * icemask_2000
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
            usurf_err = dhdt_err * np.sqrt(year_interval)
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
        uvelsurfobs_var = merged_dataset.createVariable('uvelsurfobs', 'f4',
                                                        ('time', 'y', 'x'))
        vvelsurfobs_var = merged_dataset.createVariable('vvelsurfobs', 'f4',
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
        uvelsurfobs_var[:] = uvelo
        vvelsurfobs_var[:] = vvelo

        # Write globale attribute in OGGMshop netCDF file
        dst_crs = oggm_shop_dataset.epsg
        dst_proj = oggm_shop_dataset.pyproj_srs
        merged_dataset.setncattr('pyproj_srs', str(dst_proj))
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

    parser.add_argument('--rgi_id_dir', type=str,
                        default="../../Data/Results/Test_default/Glaciers/RGI2000"
                                "-v7.0-G-11-01706/")

    # Add argument for SMB model decision
    parser.add_argument('--smb_model', type=str,
                        default="ELA",
                        help='Flag to decide for SMB model (ELA, TI, ...).')

    # Add argument for specific target resolution
    parser.add_argument('--target_resolution', type=str,
                        help='user-specific resolution for IGM run [meters] '
                             '(default: 100m)')

    # Add flags to control function execution
    parser.add_argument('--download_oggm_shop', action='store_true',
                        help='Flag to control execution of download_OGGM_shop.')

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
         rgi_id_dir=args.rgi_id_dir,
         smb_model=args.smb_model,
         target_resolution=args.target_resolution,
         oggm_shop=args.download_oggm_shop,
         hugonnet=args.download_hugonnet,
         hugonnet_directory=args.hugonnet_directory,
         year_interval=args.year_interval,
         )
