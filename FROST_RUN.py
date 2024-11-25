import argparse


# Function to handle the main logic
def main(rgi_id: str):
    print(f"Running calibration for glacier: {rgi_id}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run glacier calibration experiments.")

    # Add arguments for parameters
    parser.add_argument('--rgi_id', type=str,
                        default="RGI2000-v7.0-G-11-01706",
                        help="Name of the glacier for the model.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.rgi_id)
