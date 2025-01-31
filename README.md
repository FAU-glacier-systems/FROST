# My Python Project

A short description of what your project does.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/FAU-glacier-systems/FROST.git
   cd FROST

2. Create a virtual environment with conda
   ```bash
   conda env create -f environment.yml
   conda activate frost_env
      ```

## FROST Pipeline

1. Download data e.g. Rhone Glacier (RGI2000-v7.0-G-11-1706 )
   ```bash
   cd Scripts/Preprocess
   python download_data.py --rgi_id RGI2000-v7.0-G-11-1706 
   --download_oggm --download_hugonnet
   cd ../..
      ```

2. IGM inversion for thickness and sliding
   ```bash
   cd Scripts/Preprocess
   python igm_inversion.py --rgi_id RGI2000-v7.0-G-11-1706 
   cd ../..
   ```

3. Run calibration
   ```bash
   python run_calibration.py --rgi_id RGI2000-v7.0-G-11-1706  --ensemble_size 3 
   --iterations 5
   ```
   Results can be seen in Experitment/RGI2000-v7.0-G-11-1706/..
![Alt text](Plots/status_5_2020.png)
