import os
from netCDF4 import Dataset
import numpy as np
import rasterio
import utm
from frost.preprocess.download_data import scale_raster
import shutil


def main(rgi_id_dir, year_interval, hugonnet_directory, target_resolution):
    """
    Write observations.nc: the Hugonnet et al. (2021) elevation change rate
    dhdt and its 1-sigma error dhdt_err, reprojected onto the grid of the
    inversion result, as they are (NaN where there is no data; no gap
    filling). The ObservationProvider aggregates them to elevation bands.

    Authors: Oskar Herrmann

    Args:
           rgi_id_dir(str)          - glacier directory
           year_interval(int)       - unused; the period comes from the
                                      Hugonnet folder name
           hugonnet_directory (str) - Hugonnet folder of one period, e.g.
                                      .../11_rgi60_2000-01-01_2020-01-01
    """
    inversion_output = os.path.join(rgi_id_dir, 'Preprocess', 'outputs',
                                    'output.nc')
    # 'a': crop_hugonnet_to_glacier corrects the epsg attribute
    with Dataset(inversion_output, 'a') as inversion_dataset:
        icemask = np.array(inversion_dataset['icemask'][:]) > 0.5
        usurf = np.array(inversion_dataset['usurf'][:])
        topg = usurf - np.array(inversion_dataset['thk'][:])
        velsurf_mag = np.array(inversion_dataset['velsurfobs_mag'][:])

        # period from the folder name, e.g. 11_rgi60_2000-01-01_2020-01-01
        date_range = os.path.basename(
            os.path.normpath(str(hugonnet_directory))).split('_', 2)[-1]
        years = [int(date[:4]) for date in date_range.split('_')]
        print('Hugonnet dh/dt', date_range, 'from', hugonnet_directory)

        dhdt, dhdt_err = crop_hugonnet_to_glacier(
            date_range=date_range, hugonnet_dir=str(hugonnet_directory),
            inversion_dataset=inversion_dataset)
        # rasterio rows run north to south, the model grid south to north
        dhdt, dhdt_err = dhdt[::-1], dhdt_err[::-1]

        write_observation_file(
            os.path.join(rgi_id_dir, 'observations.nc'),
            x=inversion_dataset['x'][:], y=inversion_dataset['y'][:],
            years=years, usurf=usurf, topg=topg, icemask=icemask,
            dhdt=np.where(icemask, dhdt, np.nan),
            dhdt_err=np.where(icemask, dhdt_err, np.nan),
            velsurf_mag=velsurf_mag, epsg=inversion_dataset.epsg,
            pyproj_srs=inversion_dataset.pyproj_srs)

    rescale_observations(rgi_id_dir, target_resolution)


def write_observation_file(path, x, y, years, usurf, topg, icemask, dhdt,
                           dhdt_err, velsurf_mag, epsg, pyproj_srs):
    """
    observations.nc as read by the ObservationProvider.

    Args:
        years          - start and end of the observation period
        usurf          - surface at the start of the period (m)
        dhdt, dhdt_err - elevation change rate over the period and its
                         1-sigma error (m/yr), NaN where not observed
    """
    with Dataset(path, 'w') as nc:
        nc.createDimension('time', 2)
        nc.createDimension('x', len(x))
        nc.createDimension('y', len(y))
        variables = {
            'time': (('time',), years, 'year', 'start and end of the period'),
            'x': (('x',), x, 'm', ''),
            'y': (('y',), y, 'm', ''),
            'usurf': (('y', 'x'), usurf, 'm',
                      'surface elevation at the start of the period'),
            'topg': (('y', 'x'), topg, 'm', 'bed elevation'),
            'icemask': (('y', 'x'), icemask, '', 'ice at the start'),
            'dhdt': (('y', 'x'), dhdt, 'm/yr',
                     'elevation change rate over the period'),
            'dhdt_err': (('y', 'x'), dhdt_err, 'm/yr', '1-sigma error of dhdt'),
            'velsurf_mag': (('y', 'x'), velsurf_mag, 'm/yr',
                            'observed surface speed'),
        }
        for name, (dims, values, units, long_name) in variables.items():
            dtype = 'f8' if name in ('x', 'y') else 'f4'
            var = nc.createVariable(name, dtype, dims, fill_value=(
                np.float32(np.nan) if dims == ('y', 'x') else None))
            var[:] = np.asarray(values, dtype=np.float64)
            var.units = units
            if long_name:
                var.long_name = long_name
        nc.setncattr('pyproj_srs', str(pyproj_srs))
        nc.setncattr('epsg', str(epsg))



def rescale_observations(rgi_id_dir, target_resolution):
    """Resample observations.nc to target_resolution, if it is a number."""
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
    # scale observations.nc
    if target_resolution_float:
        obs_nc = os.path.join(rgi_id_dir, 'observations.nc')

        with Dataset(obs_nc, 'r') as ds:
            x = ds.variables['x'][:]
            resolution = abs(x[1] - x[2])
            scale_factor = resolution / target_resolution

        scale_raster(obs_nc, obs_nc.replace('.nc', '_scaled.nc'), scale_factor)
        shutil.move(obs_nc, obs_nc.replace('.nc', '_OGGM.nc'))
        shutil.move(obs_nc.replace('.nc', '_scaled.nc'), obs_nc)


def crop_hugonnet_to_glacier(date_range, hugonnet_dir, inversion_dataset):
    """
    Fuse multiple dh/dt tiles and crop to a specified OGGM dataset area.

    Authors: Oskar Herrmann, Johannes J. Fuerst

    Args:
        date_range (str): The date range for the dh/dt dataset.
        inversion_dataset (str): relative directory of OGGM dataset
        inversion_dataset (xarray.Dataset): OGGM dataset with spatial coordinates.

    Returns:
        np.ndarray: Cropped and filtered dh/dt map.
    """

    # Define the folder containing dh/dt files
    dhdt_folder = os.path.join(hugonnet_dir, 'dhdt')
    dhdt_err_folder = os.path.join(hugonnet_dir, 'dhdt_err')
    print('... retrieving Hugonnet data from: ', hugonnet_dir)

    # Extract UTM coordinates from the NetCDF file (adjust according to your dataset)
    x_coords = inversion_dataset['x'][:]
    y_coords = inversion_dataset['y'][:]
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()

    # TODO
    # Use netCDF file from OGGMshop and extract projection details
    # (no idea what happens if another DEM source is taken - instead of SRTM)
    zone_number = int(inversion_dataset.pyproj_srs.split('=')[2][0:2])
    # Convert to CRS object
    from pyproj import CRS
    crs = CRS.from_proj4(inversion_dataset.pyproj_srs)

    # Get the EPSG code
    ## use EPSG number from proj_srs
    ## (for some glaciers proj_srs is corrupted)
    # epsg_code = crs.to_epsg()
    # use EPSG number from reference DEM
    # (might be more robust if on southern hemisphere)
    epsg_code = inversion_dataset.epsg.split(':')[1]
    hemisphere_code = epsg_code[2]
    inversion_dataset.epsg = f"EPSG:{epsg_code}"

    if int(hemisphere_code) == 6:
        print('UTM hemisphere code is 6 (northern hemisphere).')
    elif int(hemisphere_code) == 7:
        print('UTM hemisphere code is 7 (southern hemisphere).')
    else:
        print('UTM hemisphere code has no expected value (6 or 7) but is : ',
              hemisphere_code)
        print('EPSG code : ', epsg_code)

    # set zone letter
    if min_y > 0 and int(hemisphere_code) == 6:
        zone_letter = "N"
    elif max_y < 0 and int(hemisphere_code) == 7:
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
    dst_array = tile_merge_reproject(dhdt_files, inversion_dataset)
    dst_err_array = tile_merge_reproject(dhdt_err_files, inversion_dataset)

    # Re-assign
    cropped_map = np.squeeze(dst_array)
    cropped_err_map = np.squeeze(dst_err_array)

    # Replace invalid values (-9999) with NaN
    filtered_map = np.where(cropped_map == -9999, np.nan, cropped_map)
    filtered_err_map = np.where(cropped_err_map == -9999, np.nan, cropped_err_map)

    return filtered_map, filtered_err_map


def tile_merge_reproject(flist, inversion_dataset):
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
           inversion_dataset  - OGGM dataset loaded from a netCDF

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
    dst_array = np.zeros(np.shape(inversion_dataset['usurf'][:]))
    dst_crs = inversion_dataset.epsg
    x = inversion_dataset['x'][:]
    y = inversion_dataset['y'][:]
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
