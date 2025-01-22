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
import utm


def scale_raster(input_file, output_file, scale_factor):
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

            # Handle CRS explicitly, if available
            if 'crs' in input_dataset.variables:
                crs_var = input_dataset.variables['crs']
                scaled_crs = scaled_dataset.createVariable('crs', crs_var.datatype)
                scaled_crs.setncatts(
                    {attr: crs_var.getncattr(attr) for attr in crs_var.ncattrs()})
                scaled_dataset.variables['crs'] = crs_var[:]

    print(f"Scaled raster saved to {output_file} with metadata.")


# Function to handle the main logic
def download_OGGM_shop(rgi_id):
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

    with Dataset('input_saved.nc', 'r') as scaled_dataset:
        x = scaled_dataset.variables['x'][:]
        resolution = abs(x[1] - x[2])
        if resolution != 100:
            scale_factor = resolution / 100
            scale_raster('input_saved.nc', 'input_scaled.nc', scale_factor)
            os.remove('input_saved.nc')
            os.rename('input_scaled.nc', 'input_saved.nc')

    os.chdir(original_dir)


def crop_hugonnet_to_glacier(rgi_region, date_range, oggm_shop_dataset):
    """
    Fuse multiple dh/dt tiles and crop to a specified OGGM dataset area.

    Args:
        date_range (str): The date range for the dh/dt dataset.
        oggm_shop_dataset (xarray.Dataset): OGGM dataset with spatial coordinates.

    Returns:
        np.ndarray: Cropped and filtered dh/dt map.
    """
    # Define the folder containing dh/dt files
    folder_name = f'{rgi_region}_rgi60_{date_range}'
    dhdt_folder = os.path.join('..', '..', 'Data', 'Hugonnet', folder_name, 'dhdt')
    dhdt_err_folder = os.path.join('..', '..', 'Data', 'Hugonnet', folder_name,
                                   'dhdt_err')

    # Extract UTM coordinates from the NetCDF file (adjust according to your dataset)
    x_coords = oggm_shop_dataset['x'][:]
    y_coords = oggm_shop_dataset['y'][:]
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()

    zone_number = int(oggm_shop_dataset.pyproj_srs.split('+')[2].split("=")[1])
    zone_letter = "N"  # TODO for south regions
    x_range = np.array([min_x, min_x, max_x, max_x])
    y_range = np.array([min_y, min_y, max_y, max_y])

    lat_lon_corner = utm.to_latlon(x_range, y_range, zone_number, zone_letter)
    lat_lon_corner = np.abs(lat_lon_corner)
    min_lat, max_lat = min(lat_lon_corner[0]), max(lat_lon_corner[0])
    min_lon, max_lon = min(lat_lon_corner[1]), max(lat_lon_corner[1])

    # Create a list to store overlapping tile names
    tile_names = []

    # Iterate over possible tiles

    for lat in range(int(min_lat), int(max_lat) + 1):
        for lon in range(int(min_lon), int(max_lon) + 1):
            # Construct the tile name
            tile_name = f'N{lat:02d}E{lon:03d}'
            tile_names.append(tile_name)
    print(tile_names)
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

    # Define the window to crop the merged dataset using these bounds
    window = from_bounds(min_x, min_y, max_x, max_y, merged_transform)

    # Ensure window indices are integers, and handle off-by-one errors
    row_off = round(window.row_off)  # Ensure the row offset is an integer
    col_off = round(window.col_off)  # Ensure the column offset is an integer
    height = len(y_coords)  # Ensure height is integer and within bounds
    width = len(x_coords)

    # Crop the merged map using the calculated window
    cropped_map = merged_map[0,
                  row_off:row_off + height,
                  col_off:col_off + width]

    cropped_err_map = merged_err_map[0,
                      row_off:row_off + height,
                      col_off:col_off + width]

    # Replace invalid values (-9999) with NaN
    filtered_map = np.where(cropped_map == -9999, np.nan, cropped_map)
    filtered_err_map = np.where(cropped_err_map == -9999, np.nan, cropped_err_map)

    # Close all open datasets
    for dataset in datasets:
        dataset.close()

    return filtered_map, filtered_err_map


def download_hugonnet(rgi_id_dir, year_interval):
    oggm_shop_dir = os.path.join(rgi_id_dir, 'OGGM_shop')

    oggm_shop_file = os.path.join(oggm_shop_dir, 'input_saved.nc')

    # load file form oggm_shop
    oggm_shop_dataset = Dataset(oggm_shop_file, 'r')
    icemask_2000 = oggm_shop_dataset['icemask'][:]
    usurf_2000 = oggm_shop_dataset['usurf'][:]
    thk_2000 = oggm_shop_dataset['thkinit'][:]

    # list folder names depending on time period
    rgi_region = rgi_id_dir.split('/')[-1].split("-")[3]

    data_interval = 5
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

    # load dhdts data sets
    dhdts = []
    dhdts_err = []
    for folder_name in folder_names:
        # load dhdt
        date_range = folder_name.split('_', 2)[-1]

        ### MERGE TILES AND CROP to oggmshop area ###
        cropped_dhdt, cropped_dhdt_err = crop_hugonnet_to_glacier(rgi_region,
                                                                  date_range,
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
        dhdt = np.where(icemask_2000 == 1, dhdt, 0)
        dhdt_change.append(dhdt)

        # either bedrock or last usurf + current dhdt
        usurf = np.maximum(bedrock, usurf_change[-1] + dhdt * data_interval)
        usurf_change.append(usurf)

        # compute uncertainty overtime
        dhdt_err = dhdts_err[dhdt_index]
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
                          + dhdt_err_change[-1] *data_interval / 2)) / 2


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
                        help='1.0 mean 100m resolution, 0.5 mean 200m resolution, ')

    # Add flags to control function execution
    parser.add_argument('--download_oggm_shop', action='store_true',
                        help='Flag to control execution of download_OGGM_shop.')
    parser.add_argument('--download_hugonnet', action='store_true',
                        help='Flag to control execution of download_Hugonnet.')

    # select between 5-year or 20-year dhdt
    parser.add_argument('--year_interval', type=int, default=20,
                        help='Select between 5-year or 20-year dhdt (5, 20)')

    # Parse arguments
    args = parser.parse_args()

    # Define the path using os.path.join
    rgi_id_dir = os.path.join('..', '..', 'Data', 'Glaciers', args.rgi_id)

    # Call functions based on flags
    if args.download_oggm_shop:
        print(f"Downloading OGGM shop data for RGI ID: {args.rgi_id}...")
        download_OGGM_shop(args.rgi_id)
        print("OGGM shop data download completed.")

    if args.download_hugonnet:
        print(f"Downloading Hugonnet data with the following parameters:")

        print(f"  RGI directory: {rgi_id_dir}")
        print(f"  Year interval: {args.year_interval}")
        download_hugonnet(rgi_id_dir, args.year_interval)
        print("Hugonnet data download completed.")

    print(f"  Scale factor: {args.scale_factor}")

    if args.scale_factor != 1.0:
        scale_raster(
            os.path.join(rgi_id_dir, 'OGGM_shop', 'input_saved.nc'),
            os.path.join(rgi_id_dir, 'OGGM_shop', 'input_scaled.nc'),
            args.scale_factor)

        scale_raster(os.path.join(rgi_id_dir, 'observations.nc'),
                     os.path.join(rgi_id_dir, 'observations_scaled.nc'),
                     args.scale_factor)
