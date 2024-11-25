import argparse
import json
import subprocess
import os


# Function to handle the main logic
def main(rgi_id):
    # Define the params to be saved in params.json
    params = {
        "modules_preproc": ["load_ncdf"],
        "modules_process": ["iceflow"],
        "modules_postproc": [],
        "iflo_run_data_assimilation": True,
        "lncd_input_file": "../OGGM_shop/input_scaled.nc",
        "opti_control": ["thk", "usurf","slidingco"],
        "opti_cost": ["velsurf","icemask", "usurf", "thk", "divfluxfcz"],
        "opti_usurfobs_std"			: 0.3,
        "opti_velsurfobs_std" 	: 0.25,
        "opti_thkobs_std"			  : 1,
        "opti_divfluxobs_std"   : 0.1,
        "opti_regu_param_thk"			: 1,
        "opti_regu_param_slidingco"     : 1.0e6,
        "opti_smooth_anisotropy_factor"	: 0.2,
        "opti_convexity_weight"		: 500,
        "opti_nbitmax"			: 500,
        "iflo_init_slidingco"      		: 0.045,
        "iflo_save_model": True,
        "opti_vars_to_save": ["usurf", "thk", "slidingco", "velsurf_mag", "velsurfobs_mag",
                            "divflux", "icemask", "arrhenius", "thkobs", "dhdt", "topg"]
    }

    # Define the path using os.path.join
    rgi_id_directory = os.path.join("../Glaciers", rgi_id, "Inversion")

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


if __name__ == "__main__":
    # Parse command-line argumentsds
    parser = argparse.ArgumentParser(
        description="This script generates params.json for igm inversion and "
                    "runs igm_run.")

    # Add argument for RGI ID
    parser.add_argument('--rgi_id', type=str,
                        default="RGI2000-v7.0-G-11-01706",
                        help="The RGI ID of the glacier to be calibrated "
                             "(default: RGI2000-v7.0-G-11-01706).")


    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.rgi_id)
