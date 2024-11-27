import argparse
import os
import json

# Function to handle the main logic
def main(rgi_id, year_interval, seed):
    print(f"Running calibration for glacier: {rgi_id}")

    json_file_path = os.path.join("..", "Experiments", rgi_id,
                                  "params_calibration.json")
    with open(json_file_path, 'r') as file:
        params = json.load(file)

    # overwrite default
    params['year_interval'] = year_interval
    params['seed'] = seed


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run glacier calibration experiments.")

    # Add arguments for parameters
    parser.add_argument('--rgi_id', type=str,
                        default="RGI2000-v7.0-G-11-01706",
                        help="RGI ID of the glacier for the model.")

    parser.add_argument('--year_interval', type=int, default=5,
                        help="Select between 5-year or 20-year dhdt (5, 20)")

    parser.add_argument("-i", "--inflation", type=float,
                        default='1', help="Inflation rate for the model." )

    parser.add_argument('--seed', type=int, default=1,
                        help="Random seed for the model.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.rgi_id, args.year_interval, args.seed)
