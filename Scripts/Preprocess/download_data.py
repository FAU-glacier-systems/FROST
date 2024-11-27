import argparse
import json
import subprocess
import os
import rasterio
from rasterio.windows import from_bounds
import math
from netCDF4 import Dataset
import numpy as np
from scipy.ndimage import zoom
from rasterio.merge import merge
import matplotlib.pyplot as plt


def scale_raster(input_file, output_file, scale_factor):
    # Load the NetCDF file
    with Dataset(input_file, 'r') as input_ds:
        # Downscale coordinates
        new_x = input_ds.variables['x'][::int(1 / scale_factor)]
        new_y = input_ds.variables['y'][::int(1 / scale_factor)]

        # Create output NetCDF file
        with Dataset(output_file, 'w') as scaled_ds:
            # Create dimensions
            scaled_ds.createDimension('x', len(new_x))
            scaled_ds.createDimension('y', len(new_y))

            # Check if 'time' dimension exists
            if 'time' in input_ds.dimensions:
                scaled_ds.createDimension('time', len(input_ds.dimensions['time']))
                time_var = scaled_ds.createVariable('time', 'f4', ('time',))
                time_var[:] = input_ds.variables['time'][:]
                time_var.setncatts({attr: input_ds.variables['time'].getncattr(attr)
                                    for attr in input_ds.variables['time'].ncattrs()})

            # Create coordinate variables
            x_var = scaled_ds.createVariable('x', 'f4', ('x',))
            y_var = scaled_ds.createVariable('y', 'f4', ('y',))
            x_var[:] = new_x
            y_var[:] = new_y

            # Copy attributes for x and y
            x_var.setncatts(
                {attr: input_ds.variables['x'].getncattr(attr) for attr in
                 input_ds.variables['x'].ncattrs()})
            y_var.setncatts(
                {attr: input_ds.variables['y'].getncattr(attr) for attr in
                 input_ds.variables['y'].ncattrs()})

            # Copy other variables and downscale
            for var_name in input_ds.variables:
                if var_name in ['x', 'y', 'time']:
                    continue

                var = input_ds.variables[var_name]
                dims = var.dimensions

                # Create a new variable in the output dataset
                scaled_var = scaled_ds.createVariable(var_name, var.datatype, dims)

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
            scaled_ds.setncatts(
                {attr: input_ds.getncattr(attr) for attr in input_ds.ncattrs()})

            # Handle CRS explicitly, if available
            if 'crs' in input_ds.variables:
                crs_var = input_ds.variables['crs']
                scaled_crs = scaled_ds.createVariable('crs', crs_var.datatype)
                scaled_crs.setncatts(
                    {attr: crs_var.getncattr(attr) for attr in crs_var.ncattrs()})
                scaled_ds.variables['crs'] = crs_var[:]

    print(f"Scaled raster saved to {output_file} with metadata.")


# Function to handle the main logic
def download_OGGM_shop(rgi_id, scale_factor, rgi_id_directory):
    # Define the params to be saved in params.json
    json_file_path = os.path.join('..', '..', 'Experiments', rgi_id,
                                  'params_download.json')
    with open(json_file_path, 'r') as file:
        params = json.load(file)

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


def crop_hugonnet_to_glacier(date_range, tile_names, oggm_shop_ds):
    """
    Fuse multiple dh/dt tiles and crop to a specified OGGM dataset area.

    Args:
        date_range (str): The date range for the dh/dt dataset.
        tile_names (list): List of tile names to process.
        oggm_shop_ds (xarray.Dataset): OGGM dataset with spatial coordinates.

    Returns:
        np.ndarray: Cropped and filtered dh/dt map.
    """
    # Define the folder containing dh/dt files
    folder_name = f'11_rgi60_{date_range}'
    dhdt_folder = os.path.join('..', '..', 'Data', 'Hugonnet', folder_name, 'dhdt')
    dhdt_err_folder = os.path.join('..', '..', 'Data', 'Hugonnet', folder_name, 'dhdt_err')

    # Collect all dh/dt files for the specified tiles
    dhdt_files = [os.path.join(dhdt_folder, f'{tile}_{date_range}_dhdt.tif') for tile
                  in tile_names]
    dhdt_err_files = [os.path.join(dhdt_err_folder,
                                   f'{tile}_{date_range}_dhdt_err.tif')
                      for tile in tile_names]

    # Open all the dh/dt tiles and merge them
    datasets = [rasterio.open(file) for file in dhdt_files]
    datasets_err = [rasterio.open(file) for file in dhdt_err_files]

    merged_map, merged_transform = merge(datasets)
    merged_err_map, merged_err_transform = merge(datasets_err)

    # Get bounds of the OGGM shop dataset area
    area_x = oggm_shop_ds['x'][:]
    area_y = oggm_shop_ds['y'][:]
    min_x, max_x = area_x.min(), area_x.max()
    min_y, max_y = area_y.min(), area_y.max()

    # Define the window to crop the merged dataset using these bounds
    window = from_bounds(min_x, min_y, max_x, max_y, merged_transform)

    # Ensure window indices are integers, and handle off-by-one errors
    row_off = int(window.row_off)  # Ensure the row offset is an integer
    col_off = int(window.col_off)  # Ensure the column offset is an integer
    height = int(window.height)  # Ensure height is integer and within bounds
    width = int(window.width)

    # Crop the merged map using the calculated window
    cropped_map = merged_map[0,  # Band 1
                            row_off:row_off + height+1,
                             col_off:col_off + width+1]

    cropped_err_map = merged_err_map[0,
                                     row_off:row_off + height+1,
                                     col_off:col_off + width+1]

    # Replace invalid values (-9999) with NaN
    filtered_map = np.where(cropped_map == -9999, np.nan, cropped_map)
    filtered_err_map = np.where(cropped_err_map == -9999, np.nan, cropped_err_map)

    # Close all open datasets
    for dataset in datasets:
        dataset.close()

    return filtered_map, filtered_err_map

def download_hugonnet(scale_factor, rgi_id_dir, year_interval,
                      tile_names):
    oggm_shop_dir = os.path.join(rgi_id_dir, 'OGGM_shop')

    oggm_shop_file = os.path.join(oggm_shop_dir, 'input_saved.nc')

    # load file form oggm_shop
    oggm_shop_ds = Dataset(oggm_shop_file, 'r')
    icemask_2000 = oggm_shop_ds['icemask'][:]
    usurf_2000 = oggm_shop_ds['usurf'][:]
    thk_2000 = oggm_shop_ds['thkinit'][:]

    # list folder names depending on time period
    if year_interval == 20:
        folder_names = ['11_rgi60_2000-01-01_2020-01-01']

    elif year_interval == 5:
        folder_names = ['11_rgi60_2000-01-01_2005-01-01',
                        '11_rgi60_2005-01-01_2010-01-01',
                        '11_rgi60_2010-01-01_2015-01-01',
                        '11_rgi60_2015-01-01_2020-01-01']

    else:
        raise ValueError(
            'Invalid time period: {}. Please choose either 5 or 20.'.format(
                year_interval))

    # load dhdts data sets
    dhdts = []
    dhdts_err = []
    for folder_name in folder_names:
        # load dhdt
        date_range = folder_name.split('_', 2)[-1]
        cropped_dhdt, cropped_dhdt_err = crop_hugonnet_to_glacier(date_range,
                                                                  tile_names,
                                                                  oggm_shop_ds)
        dhdt_masked = cropped_dhdt[::-1] * icemask_2000
        dhdts.append(dhdt_masked)

        dhdt_err_masked = cropped_dhdt_err[::-1] * icemask_2000
        dhdts_err.append(dhdt_err_masked)

    usurf_change = [usurf_2000]  # initialise with 2000 state
    thk_change = [thk_2000]
    dhdt_change = [np.zeros_like(usurf_2000)]
    dhdt_err_change = [np.zeros_like(usurf_2000)]
    usurf_err_change = [np.zeros_like(usurf_2000)]  # TODO

    bedrock = usurf_2000 - thk_2000

    year_range = np.arange(2000, 2021)

    for i, year in enumerate(year_range[1:]):
        # compute surface change based on dhdt and provide uncertainties
        # change the dhdt field every year_interval
        dhdt_index = math.floor(i / year_interval)
        dhdt = dhdts[dhdt_index]
        dhdt = np.where(icemask_2000 == 1, dhdt, 0)
        dhdt_change.append(dhdt)


        # either bedrock or last usurf + current dhdt
        usurf = np.maximum(bedrock, usurf_change[-1] + dhdt)
        usurf_change.append(usurf)
        thk = usurf - bedrock
        thk_change.append(thk)


        # compute uncertainty overtime
        dhdt_err = dhdts_err[dhdt_index]
        dhdt_err= np.where(icemask_2000 == 1, dhdt_err, 0)
        dhdt_err_change.append(dhdt_err)

        # assuming the error is termporal independet
        # the square root of the sum of variance should be the right err for the
        # surface
        usurf_err = np.sqrt(sum([dhdt_err_i ** 2 for dhdt_err_i in dhdt_err_change]))
        usurf_err_change.append(usurf_err)

    # transform to numpy array
    usurf_change = np.array(usurf_change)
    usurf_err_change = np.array(usurf_err_change)
    thk_change = np.array(thk_change)
    dhdt_change = np.array(dhdt_change)
    dhdt_err_change = np.array(dhdt_err_change)

    # compute velocity magnitude
    uvelo = oggm_shop_ds.variables['uvelsurfobs'][:]
    vvelo = oggm_shop_ds.variables['vvelsurfobs'][:]
    velo = np.sqrt(uvelo ** 2 + vvelo ** 2)
    # create placeholder smb
    smb = np.zeros_like(dhdt)

    # Create a new netCDF file
    observation_file = os.path.join(rgi_id_dir, 'observations.nc')
    with Dataset(observation_file, 'w') as merged_ds:
        # Create dimensions
        merged_ds.createDimension('time', len(year_range))
        merged_ds.createDimension('x', oggm_shop_ds.dimensions['x'].size)
        merged_ds.createDimension('y', oggm_shop_ds.dimensions['y'].size)

        # Create variables
        time_var = merged_ds.createVariable('time', 'f4', ('time',))
        x_var = merged_ds.createVariable('x', 'f4', ('x',))
        y_var = merged_ds.createVariable('y', 'f4', ('y',))
        thk_var = merged_ds.createVariable('thk', 'f4', ('time', 'y', 'x'))
        usurf_var = merged_ds.createVariable('usurf', 'f4', ('time', 'y', 'x'))
        usurf_err_var = merged_ds.createVariable('usurf_err', 'f4', ('time', 'y',
                                                                     'x'))
        topg_var = merged_ds.createVariable('topg', 'f4', ('time', 'y', 'x'))
        icemask_var = merged_ds.createVariable('icemask', 'f4', ('time', 'y', 'x'))
        dhdt_var = merged_ds.createVariable('dhdt', 'f4', ('time', 'y', 'x'))
        dhdt_err_var = merged_ds.createVariable('dhdt_err', 'f4', ('time', 'y', 'x'))
        smb_var = merged_ds.createVariable('smb', 'f4', ('time', 'y', 'x'))
        velsurf_mag_var = merged_ds.createVariable('velsurf_mag', 'f4',
                                                   ('time', 'y', 'x'))

        # Assign data to variables
        time_var[:] = year_range
        x_var[:] = oggm_shop_ds.variables['x'][:]
        y_var[:] = oggm_shop_ds.variables['y'][:]
        thk_var[:] = thk_change
        usurf_var[:] = usurf_change
        usurf_err_var[:] = usurf_err_change
        topg_var[:] = bedrock
        icemask_var[:] = icemask_2000
        dhdt_var[:] = dhdt_change
        dhdt_err_var[:] = dhdt_err_change
        smb_var[:] = smb
        velsurf_mag_var[:] = velo


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='This script generates params.json for downloading data with '
                    'oggm shop as the igm module and runs igm_run.'
    )

    # Add argument for RGI ID
    parser.add_argument('--rgi_id', type=str,
                        default='RGI2000-v7.0-G-11-01706',
                        help='The RGI ID of the glacier to be calibrated '
                             '(default: RGI2000-v7.0-G-11-01706).')

    # Add argument for scale factor
    parser.add_argument('--scale_factor', type=float,
                        default=1.0,
                        help='Factor to scale the resolution of the glacier. '
                             'OGGM scales the resolution according to the glacier '
                             'size.')

    # Add flags to control function execution
    parser.add_argument('--download_oggm_shop', action='store_true',
                        help='Flag to control execution of download_OGGM_shop.')
    parser.add_argument('--download_hugonnet', action='store_true',
                        help='Flag to control execution of download_Hugonnet.')

    # select between 5-year or 20-year dhdt
    parser.add_argument('--year_interval', type=int, default=5,
                        help='Select between 5-year or 20-year dhdt (5, 20)')
    parser.add_argument('--tile_names', type=str, default='N46E008',
                        nargs='+',
                        help='Specify one or more tile names (e.g. N46E008 N46E007)')

    # Parse arguments
    args = parser.parse_args()

    # Define the path using os.path.join
    rgi_id_dir = os.path.join('..', '..', 'Data', 'Glaciers', args.rgi_id)

    # Call functions based on flags
    if args.download_oggm_shop:
        print(f"Downloading OGGM shop data for RGI ID: {args.rgi_id}...")
        download_OGGM_shop(args.rgi_id, args.scale_factor, rgi_id_dir)
        print("OGGM shop data download completed.")

    if args.download_hugonnet:
        print(f"Downloading Hugonnet data with the following parameters:")
        print(f"  Scale factor: {args.scale_factor}")
        print(f"  RGI directory: {rgi_id_dir}")
        print(f"  Year interval: {args.year_interval}")
        print(f"  Tile name: {args.tile_names}")
        download_hugonnet(args.scale_factor, rgi_id_dir, args.year_interval,
                          args.tile_names)
        print("Hugonnet data download completed.")

    if args.scale_factor != 1.0:
        scale_raster(
            os.path.join(rgi_id_dir, 'OGGM_shop', 'input_saved.nc'),
            os.path.join(rgi_id_dir, 'OGGM_shop', 'input_scaled.nc'),
            args.scale_factor)
        scale_raster(os.path.join(rgi_id_dir, 'observations.nc'),
                     os.path.join(rgi_id_dir, 'observations_scaled.nc'),
                     args.scale_factor)
