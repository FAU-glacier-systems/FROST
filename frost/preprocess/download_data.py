#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann, Johannes J. Fuerst
# Published under the GNU GPL (Version 3), check the LICENSE file

# import required packages
import argparse
import os
import numpy as np
import netCDF4
from netCDF4 import Dataset
from scipy.ndimage import zoom
import shutil
import yaml
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
         oggm_shop):  # Parse command-line arguments

    # Create output folder
    os.makedirs(rgi_id_dir, exist_ok=True)

    # SMB model
    if str(smb_model) == "ELA":
        flag_OGGM_climate = False
    elif str(smb_model) == "TI":
        flag_OGGM_climate = True
    else:
        flag_OGGM_climate = False

    # Call functions based on flags
    if oggm_shop:
        print(f"Downloading OGGM shop data for RGI ID: {rgi_id}...")
        download_OGGM_shop(rgi_id, rgi_id_dir, flag_OGGM_climate)
        print("OGGM shop data download completed.")

    # TODO
    # Check input file from download for consistency
    # - no 'NaN' in the thickness fields (thkinit, thk)
    # - 90% percentile of observed velocities must be above 10m/yr (mangitude of both x- and y-components)

    # Define input and output file names
    input_orig_file = os.path.join(rgi_id_dir, 'Preprocess', 'data',
                                   'input_OGGM_orig.nc')
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
                    dst_var = dst.createVariable(name, variable.datatype,
                                                 variable.dimensions)
                    dst_var[:] = np.where(np.abs(original_data) > 10000, 0,
                                          original_data)
                    dst_var[:] = np.where(np.abs(original_data) < 0, 0, dst_var)

                    try:
                        fillvalue = variable.get_fill_value()
                    except:
                        fillvalue = netCDF4.default_fillvals['f8']

                    mask = src.variables["icemask"][:]

                    if np.nansum(np.nansum(original_data * mask)) == 0 :
                        dst_var[:] = np.where(np.abs(original_data * mask) == fillvalue, 10, dst_var)
                        dst_var[:] = np.where(np.abs(original_data+mask) == 1, 10, dst_var)
                    if np.isinf(np.nansum(np.nansum(original_data * mask))) :
                        dst_var[:] = mask*10

                elif name == 'thkinit':
                    # Set values to zero where they are NaN
                    original_data = variable[:]
                    # Create a mask for NaN values and set them to zero
                    dst_var = dst.createVariable(name, variable.datatype,
                                                 variable.dimensions)
                    dst_var[:] = np.where(np.abs(original_data) > 10000, 0,
                                          original_data)
                    dst_var[:] = np.where(np.abs(original_data) < 0, 0, dst_var)
                    dst_var[:] = np.where(np.isinf(original_data), np.nan, dst_var)

                    try:
                        fillvalue = variable.get_fill_value()
                    except:
                        fillvalue = netCDF4.default_fillvals['f8']

                    mask = src.variables["icemask"][:]
                    if np.nansum(np.nansum(original_data * mask)) == 0 :
                        dst_var[:] = np.where(np.abs(original_data * mask) == fillvalue, 10, dst_var)
                        dst_var[:] = np.where(np.abs(original_data+mask) == 1, 10, dst_var)
                    if np.isinf(np.nansum(np.nansum(original_data * mask))) :
                        dst_var[:] = mask*10
                        #print('JOHANNES JOHANNES JOHANNES JOHANNES : corrected thickness init')

                else:
                    dst_var = dst.createVariable(name, variable.datatype,
                                                 variable.dimensions)
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
            # dst_proj = scaled_dataset.pyproj_crs
            ## adjust dxdx values§
            # dst_proj2 = str(
            #    scaled_dataset.pyproj_crs.split('dxdy')[0]) + "dxdy\': [" + str(
            #    1.0 / scale_factor * (input_dataset.variables['x'][1] -
            #                          input_dataset.variables['x'][
            #                              0])) + ', -' + str(1.0 / scale_factor * (
            #        input_dataset.variables['y'][1] -
            #        input_dataset.variables['y'][0])) + '],' + "\'pixel " + str#(
            #    scaled_dataset.pyproj_crs.split('pixel')[1])
            ## adjust nxny values
            # dst_proj3 = str(dst_proj2.split('nxny')[0]) + "nxny\': [" + str(
            #    len(scaled_dataset.variables['x'][:])) + ', ' + str(
            #    len(scaled_dataset.variables['y'][:])) + ']' + str(
            #    "\'dxdy" + str(dst_proj2.split('dxdy')[1]))
            # scaled_dataset.setncattr('pyproj_crs', str(dst_proj3))
            #
            ## Handle CRS explicitly, if available
            # if 'crs' in input_dataset.variables:
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
        # print('flag_OGGM_climate', flag_OGGM_climate, type(flag_OGGM_climate))
        if flag_OGGM_climate:
            print('OGGM climate data is retained for SMB model')
            os.system('pwd')
            os.listdir('./')
            src = os.path.join('.', 'data', rgi_id, 'climate_historical.nc')
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
