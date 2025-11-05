import os
import xarray as xr
import csv
import numpy as np

# Input directory with glacier subfolders
data_dir = "../../data/results/central_europe_submit/glaciers"

# Single combined CSV
out_csv = "../central_europe_submit/tables/inversion_results.csv"

# Header (velocity + thickness-at-observations)
HEADER = [
    "rgi_id",
    # Velocity (masked by icemask)
    "Mean_velsurf_mag", "Std_velsurf_mag", "Min_velsurf_mag",
    "Q1_velsurf_mag", "Median_velsurf_mag", "Q3_velsurf_mag", "Max_velsurf_mag",
    "Mean_velsurfobs_mag", "Std_velsurfobs_mag", "Min_velsurfobs_mag",
    "Q1_velsurfobs_mag", "Median_velsurfobs_mag", "Q3_velsurfobs_mag",
    "Max_velsurfobs_mag", "MAE_velsurf_mag",
    # Thickness (only at observed points)
    "n_obs_thk", "coverage_pct",
    "Mean_thk_model_at_obs", "Std_thk_model_at_obs",
    "Mean_thk_obs", "Std_thk_obs",
    "MAE_thk", "Bias_thk", "RMSE_thk",
    "MedianAE_thk", "Q95AE_thk",
    "Min_thk_obs", "Q1_thk_obs", "Median_thk_obs", "Q3_thk_obs", "Max_thk_obs",
]

# Create/overwrite CSV with header
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)

def to_scalar(x):
    """Return Python float or np.nan from xarray reduction."""
    try:
        return x.item()
    except Exception:
        return float(x) if np.isscalar(x) else np.nan

for rgi_file in os.listdir(data_dir):
    rgi_path = os.path.join(data_dir, rgi_file)
    if not os.path.isdir(rgi_path):
        continue

    print(f"Processing RGI: {rgi_file}")
    inversion_path = os.path.join(rgi_path, "Preprocess", "outputs")
    nc_path = os.path.join(inversion_path, "output.nc")
    if not os.path.isfile(nc_path):
        print(f"  output.nc not found")
        continue

    row = {k: np.nan for k in HEADER}
    row["rgi_id"] = rgi_file

    try:
        ds = xr.open_dataset(nc_path)

        # -------- Velocity block (masked by icemask) --------
        if all(v in ds.variables for v in ["velsurf_mag", "velsurfobs_mag", "icemask"]):
            mask_ice = (ds["icemask"] == 1)
            vmod = ds["velsurf_mag"].where(mask_ice)
            vobs = ds["velsurfobs_mag"].where(mask_ice)

            row["Mean_velsurf_mag"]    = to_scalar(vmod.mean(skipna=True))
            row["Std_velsurf_mag"]     = to_scalar(vmod.std(skipna=True))
            row["Min_velsurf_mag"]     = to_scalar(vmod.min(skipna=True))
            row["Max_velsurf_mag"]     = to_scalar(vmod.max(skipna=True))
            row["Q1_velsurf_mag"]      = to_scalar(vmod.quantile(0.25, skipna=True))
            row["Median_velsurf_mag"]  = to_scalar(vmod.quantile(0.50, skipna=True))
            row["Q3_velsurf_mag"]      = to_scalar(vmod.quantile(0.75, skipna=True))

            row["Mean_velsurfobs_mag"]   = to_scalar(vobs.mean(skipna=True))
            row["Std_velsurfobs_mag"]    = to_scalar(vobs.std(skipna=True))
            row["Min_velsurfobs_mag"]    = to_scalar(vobs.min(skipna=True))
            row["Max_velsurfobs_mag"]    = to_scalar(vobs.max(skipna=True))
            row["Q1_velsurfobs_mag"]     = to_scalar(vobs.quantile(0.25, skipna=True))
            row["Median_velsurfobs_mag"] = to_scalar(vobs.quantile(0.50, skipna=True))
            row["Q3_velsurfobs_mag"]     = to_scalar(vobs.quantile(0.75, skipna=True))

            row["MAE_velsurf_mag"] = to_scalar(np.abs(vmod - vobs).mean(skipna=True))
        else:
            print("  Missing velocity variables")

        # -------- Thickness block (only where thkobs is present) --------
        if all(v in ds.variables for v in ["thk", "thkobs", "icemask"]):
            ice_mask = (ds["icemask"] == 1)
            obs_mask = ds["thkobs"].notnull()
            valid = ice_mask & obs_mask

            thk_mod = ds["thk"].where(valid)
            thk_obs = ds["thkobs"].where(valid)

            n_obs = int(valid.sum().item())
            total_ice = int(ice_mask.sum().item()) if "icemask" in ds.variables else 0
            coverage_pct = (100.0 * n_obs / total_ice) if total_ice > 0 else np.nan

            diff = thk_mod - thk_obs
            absdiff = np.abs(diff)

            row["n_obs_thk"]               = n_obs
            row["coverage_pct"]            = coverage_pct
            row["Mean_thk_model_at_obs"]   = to_scalar(thk_mod.mean(skipna=True))
            row["Std_thk_model_at_obs"]    = to_scalar(thk_mod.std(skipna=True))
            row["Mean_thk_obs"]            = to_scalar(thk_obs.mean(skipna=True))
            row["Std_thk_obs"]             = to_scalar(thk_obs.std(skipna=True))
            row["MAE_thk"]                 = to_scalar(absdiff.mean(skipna=True))
            row["Bias_thk"]                = to_scalar(diff.mean(skipna=True))
            row["RMSE_thk"]                = to_scalar(np.sqrt((diff**2).mean(skipna=True)))
            row["MedianAE_thk"]            = to_scalar(absdiff.quantile(0.50, skipna=True))
            row["Q95AE_thk"]               = to_scalar(absdiff.quantile(0.95, skipna=True))

            row["Min_thk_obs"]             = to_scalar(thk_obs.min(skipna=True))
            row["Q1_thk_obs"]              = to_scalar(thk_obs.quantile(0.25, skipna=True))
            row["Median_thk_obs"]          = to_scalar(thk_obs.quantile(0.50, skipna=True))
            row["Q3_thk_obs"]              = to_scalar(thk_obs.quantile(0.75, skipna=True))
            row["Max_thk_obs"]             = to_scalar(thk_obs.max(skipna=True))
        else:
            print("  Missing thickness variables")

        ds.close()

    except Exception as e:
        print(f"  Error processing {nc_path}: {e}")

    # Append row
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row[k] for k in HEADER])

print(f"Done. Wrote {out_csv}")
