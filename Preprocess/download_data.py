import argparse
import json
import subprocess
import os
import numpy as np
import xarray as xr
from scipy.ndimage import zoom

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
            raise ValueError(f"Unsupported data dimensions: {data.ndim}")
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
    print(f"Downscaled data saved to {output_file}")


# Function to handle the main logic
def main(rgi_id, scale_factor):
    # Define the params to be saved in params.json
    params = {
        "modules_preproc": ["oggm_shop"],
        "modules_process": [],
        "modules_postproc": [],
        "oggm_RGI_version": 7,
        "oggm_RGI_product": "G",
        "oggm_thk_source": "millan_ice_thickness",
        "oggm_incl_glathida": True,
        "oggm_RGI_ID": rgi_id
    }

    # Define the path using os.path.join
    rgi_id_directory = os.path.join("../Glaciers", rgi_id, "OGGM_shop")

    # Check if the directory exists, and create it if not
    if not os.path.exists(rgi_id_directory):
        os.makedirs(rgi_id_directory)

        # Change directory to the correct location
    os.chdir(rgi_id_directory)

    # Write the params dictionary to the params.json file
    with open("params.json", 'w') as json_file:
        json.dump(params, json_file, indent=4)

    # Run the igm_run command
    subprocess.run(["igm_run", "--param_file", "params.json"])

    # scale the downloaded file
    if scale_factor != 1:
        scale_raster(input_file="input_saved.nc", scale_factor=scale_factor,
                     output_file="input_scaled.nc")

if __name__ == "__main__":
    # Parse command-line argumentsds
    parser = argparse.ArgumentParser(
        description="This script generates params.json for downloading data with "
                    "oggm shop as the igm module and runs igm_run.")

    # Add argument for RGI ID
    parser.add_argument('--rgi_id', type=str,
                        default="RGI2000-v7.0-G-11-01706",
                        help="The RGI ID of the glacier to be calibrated "
                             "(default: RGI2000-v7.0-G-11-01706).")

    # Add argument for RGI ID
    parser.add_argument('--scale_factor', type=float,
                        default="1",
                        help="Factor to scale the resolution of the glacier"
                             "OGGM scales the resolution according to the glacier "
                             "size")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.rgi_id, args.scale_factor)
