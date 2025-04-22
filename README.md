# FROST 

Framework for assimilating Remote-sensing Observations for Surface Mass Balance Tuning


## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:FAU-glacier-systems/FROST.git
   cd FROST

2. Create a virtual environment with conda
   ```bash
   conda env create -f environment.yml
   conda activate frost_env
      ```

## FROST Pipeline

1. Download data e.g. Rhone Glacier (RGI2000-v7.0-G-11-01706 )
   ```bash
   cd Scripts/Preprocess
   python download_data.py --rgi_id RGI2000-v7.0-G-11-01706 --download_oggm --download_hugonnet
   cd ../..
      ```
   If you want to calibrate other glaciers you have to provide all dhdt tiles from Hugonnet: https://www.sedoo.fr/theia-publication-products/?uuid=c428c5b9-df8f-4f86-9b75-e04c778e29b9

2. IGM inversion for thickness and sliding
   ```bash
   cd Scripts/Preprocess
   python igm_inversion.py --rgi_id RGI2000-v7.0-G-11-01706 
   cd ../..
   ```

3. Run calibration
   ```bash
   python run_calibration.py --rgi_id RGI2000-v7.0-G-11-01706  --ensemble_size 3 --iterations 5
   ```
   Results can be seen in Experitment/RGI2000-v7.0-G-11-1706/..
![Alt text](Plots/status_6_2020_real.pdf)
