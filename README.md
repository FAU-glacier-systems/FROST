<h1>
  <img src="assets/frost_log.png" alt="FROST logo" width="100" align="left">
  <b>F</b>ramework for assimilating <b>R</b>emote-sensing <b>O</b>bservations for <b>S</b>urface
  mass balance <b>T</b>uning<br>
</h1>


FROST is a data assimilation framework tailored for glacier modeling.
It couples the 3D glacier model [IGM](https://github.com/instructed-glacier-model/igm)
with an Ensemble Kalman Filter to calibrate glacier-specific surface mass
balance parameters using remote sensing observations. The method is
derivative-free, and scalable. It also provides uncertainty estimates alongside 
calibrated results.

---

## 🏗️ Installation

1. Clone the repository

   ```bash
   git clone git@github.com:FAU-glacier-systems/FROST.git
   cd FROST
   ```

2. Create a virtual environment with conda

   ```bash
   conda env create -f environment.yml
   conda activate frost_env
   ```

3. Install the IGM model next to FROST (Hydra-compatible)

   ```bash
   cd ..
   git clone https://github.com/jouvetg/igm 
   pip install -e igm/
   ```

---

## 🚀 Pipeline for Calibration 

1. Duplicate the `experiments/test_default` folder and rename it to your custom
   experiment name, e.g., `experiments/my_run`.
   Adapt the `config.yml` to your target glacier and desired setup e.g rgi_id
2. Download elevation change product and adapt the path in `config.yml`:
    https://www.sedoo.fr/theia-publication-products/?uuid=c428c5b9-df8f-4f86-9b75-e04c778e29b9
3. Run the pipeline

   ```bash
   python frost_pipeline.py --config experiments/<experiment-name>/config.yml 
   ```

4. View the results:

* **Calibration Results**
  `data/results/<experiment-name>/glaciers/<rgi-id>/calibration_results.json`

* **Monitoring Images**
  `monitor/status.png`

* **Example**
  ![Status Example](assets/status_006_2020.png)

---

## 🏛️ Architecture

A schematic overview of the FROST calibration workflow:
![FROST Architecture](assets/FROST_architecture.svg)

---

## 📎 Reference

If you use FROST, please cite:

> Herrmann et al. (2025). A Kalman Filter-based Framework for Assimilating Remote
> Sensing Observations into a Surface Mass Balance Model. *Annals of Glaciology*, 66(
> 94).
> \[DOI & bibtex placeholder]

