import argparse
import json
import subprocess
import os
import numpy as np
import xarray as xr
from scipy.ndimage import zoom
from netCDF4 import Dataset
import rasterio
from rasterio.windows import from_bounds
import math


def scale_raster(input_file, scale_factor, output_file):
    # Load the NetCDF file
    ds = xr.open_dataset(input_file)

    # Downscale each data variable (assuming 2D or 3D arrays)
    downscaled_data_vars = {}
    for var in ds.data_vars:
        data = ds[var].values
        if data.ndim == 2:  # For 2D data
            downscaled_data = zoom(data, zoom=scale_factor, order=0)
        elif data.ndim == 3:  # For 3D data
            downscaled_data = np.array(
                [zoom(layer, zoom=scale_factor, order=0) for layer in data])
        else:
            raise ValueError(f'Unsupported data dimensions: {data.ndim}')
        if var == 'icemask':
            downscaled_data[downscaled_data >= 0.5] = 1
            downscaled_data[downscaled_data < 0.5] = 0
        downscaled_data_vars[var] = (ds[var].dims, downscaled_data)

    # Create a new xarray dataset with the downscaled data
    downscaled_ds = xr.Dataset(
        downscaled_data_vars,
        coords={
            'y': np.linspace(ds.y.min(), ds.y.max(), downscaled_data.shape[-2]),
            'x': np.linspace(ds.x.min(), ds.x.max(), downscaled_data.shape[-1]),
            # Add other coordinates if needed
        }
    )

    # Save the downscaled dataset to a new NetCDF file
    downscaled_ds.to_netcdf(output_file)
    print(f'Downscaled data saved to {output_file}')


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

    # scale the downloaded file
    if scale_factor != 1:
        scale_raster(input_file='input_saved.nc', scale_factor=scale_factor,
                     output_file='input_scaled.nc')
    os.chdir(original_dir)


def crop_hugonnet_to_glacier(hugonnet_dataset, oggm_shop_ds):
    # Get bounds of the OGGM shop dataset area
    area_x = oggm_shop_ds['x'][:]
    area_y = oggm_shop_ds['y'][:]

    # Calculate the bounds from the min and max coordinates of oggm_shop_ds
    min_x, max_x = area_x.min(), area_x.max()
    min_y, max_y = area_y.min(), area_y.max()

    # Define the window to crop the hugonnet dataset using these bounds
    window = from_bounds(min_x, min_y, max_x, max_y, hugonnet_dataset.transform)

    # Read the data from the specified window (cropped area)
    cropped_map = hugonnet_dataset.read(1, window=window)
    filtered_map = np.where(cropped_map == -9999, np.nan, cropped_map)

    return filtered_map


def download_hugonnet(scale_factor, rgi_id_dir, year_interval,
                      tile_name):
    oggm_shop_dir = os.path.join(rgi_id_dir, 'OGGM_shop')

    if scale_factor != 1:
        oggm_shop_file = os.path.join(oggm_shop_dir, 'input_scaled.nc')
    else:
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
        dhdt_file = f'{tile_name}_{date_range}_dhdt.tif'
        dhdt_path = os.path.join('..', '..', 'Data', 'Hugonnet', folder_name, 'dhdt',
                                 dhdt_file)
        with rasterio.open(dhdt_path) as dhdt_dataset:
            cropped_dhdt = crop_hugonnet_to_glacier(dhdt_dataset, oggm_shop_ds)
            dhdt_masked = cropped_dhdt[::-1] * icemask_2000
            dhdts.append(dhdt_masked)

        # load dhdt error
        dhdt_err_file = dhdt_file.replace('.tif', '_err.tif')
        dhdt_err_path = os.path.join('..', '..', 'Data', 'Hugonnet', folder_name,
                                     'dhdt_err', dhdt_err_file)
        with rasterio.open(dhdt_err_path) as dhdt_err_dataset:
            cropped_dhdt_err = crop_hugonnet_to_glacier(dhdt_err_dataset,
                                                        oggm_shop_ds)
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
        dhdt_change.append(dhdt)

        # either bedrock or last usurf + current dhdt
        usurf = np.maximum(bedrock, usurf_change[-1] + dhdt)
        usurf_change.append(usurf)
        thk_change.append(usurf - bedrock)

        # compute uncertainty overtime
        dhdt_err = dhdts_err[dhdt_index]
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
    parser.add_argument('--tile_name', type=str, default='N46E008')

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
        print(f"  Tile name: {args.tile_name}")
        download_hugonnet(args.scale_factor, rgi_id_dir, args.year_interval,
                          args.tile_name)
        print("Hugonnet data download completed.")
