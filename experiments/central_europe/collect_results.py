import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# Paths
# ============================================================
# Ensemble and inversion statistics are slow to gather (one NetCDF per glacier),
# so they are cached; set these to True to rebuild the caches.
RECOMPUTE_ENSEMBLE = False
RECOMPUTE_INVERSION = False
ENSEMBLE_CACHE_CSV = Path("tables/ensemble_stats.csv")
INVERSION_CACHE_CSV = Path("tables/inversion_results.csv")
RGI_FILES_PATH = Path("../../data/raw/central_europe/Split_Files")
SLA_PATH = Path("../../data/raw/central_europe/Alps_Glacier_EoS_SLA_2000-2019_stats_v2.csv")
WGMS_GLAMOS_PATH = Path("validation/tables/combined_ela_gradients.csv")

EXPERIMENTS_PATH = Path("../../data/results/central_europe_submit/glaciers")
OUTPUT_CSV = Path("tables/aggregated_results.csv")


# ============================================================
# Small utilities
# ============================================================

def safe_read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        print(f"Warning: file not found: {path}")
        return pd.DataFrame(columns=columns or [])

    df = pd.read_csv(path)
    if columns is None:
        return df

    keep = [c for c in columns if c in df.columns]
    return df.loc[:, keep].copy() if keep else pd.DataFrame(columns=columns)


def merge_drop_duplicates(left: pd.DataFrame, right: pd.DataFrame, on: str, suffix: str) -> pd.DataFrame:
    merged = pd.merge(left, right, on=on, how="left", suffixes=("", suffix))

    duplicate_cols = []
    for col in merged.columns:
        if col.endswith(suffix):
            base = col[:-len(suffix)]
            if base in merged.columns:
                duplicate_cols.append(col)

    if duplicate_cols:
        merged = merged.drop(columns=duplicate_cols)

    return merged


def aspect_to_components(aspect_deg: float) -> tuple[float, float]:
    aspect_rad = np.radians(aspect_deg)
    return np.sin(aspect_rad), np.cos(aspect_rad)


def empty_time_series_array(n_times: int = 3) -> np.ndarray:
    return np.full(n_times, np.nan)


# ============================================================
# Reading single-glacier outputs
# ============================================================

def read_calibration_json(result_path: Path) -> tuple[list, list] | None:
    if not result_path.exists():
        return None

    try:
        with open(result_path, "r") as file:
            data = json.load(file)
        final_mean = data.get("final_mean", [])
        final_std = data.get("final_std", [])
        return final_mean, final_std

    except json.JSONDecodeError:
        print(f"Warning: could not parse JSON: {result_path}")
        return None


def collect_ensemble_stats(glacier_dir: Path) -> dict:
    ensemble_path = glacier_dir / "Ensemble"

    vel_ensemble = []
    smb_ensemble = []
    dhdt_ensemble = []

    if not ensemble_path.exists():
        return {
            "vel_mean": empty_time_series_array(),
            "smb_mean": empty_time_series_array(),
            "smb_std": empty_time_series_array(),
            "dhdt_mean": np.nan,
            "dhdt_std": np.nan,
        }

    for member in sorted(ensemble_path.iterdir())[::2]:
        output_path = member / "outputs" / "output.nc"
        if not output_path.exists():
            continue

        try:
            with xr.open_dataset(output_path) as ds:
                if "velsurf_mag" not in ds:
                    continue

                vel = ds["velsurf_mag"]
                smb = ds["smb"]
                usurf = ds["usurf"]
                thk = ds["thk"]

                vel_member = []
                smb_member = []

                for idx in [0, 1, 2]:
                    if idx < len(vel):
                        mask = thk[idx].values > 1
                        vel_member.append(np.nanmean(vel[idx].values[mask]))
                        smb_member.append(np.nanmean(smb[idx].values[mask]))

                if len(usurf) > 2:
                    dhdt = (usurf[2].values - usurf[0].values) / 20
                    dhdt_ensemble.append(np.nanmean(dhdt[thk[0].values > 1]))

                vel_ensemble.append(vel_member)
                smb_ensemble.append(smb_member)

        except Exception as exc:
            print(f"Warning: failed reading {output_path}: {exc}")

    vel_ensemble = np.array(vel_ensemble) if vel_ensemble else np.full((1, 3), np.nan)
    smb_ensemble = np.array(smb_ensemble) if smb_ensemble else np.full((1, 3), np.nan)
    dhdt_ensemble = np.array(dhdt_ensemble) if dhdt_ensemble else np.array([np.nan])

    return {
        "vel_mean": np.nanmean(vel_ensemble, axis=0),
        "smb_mean": np.nanmean(smb_ensemble, axis=0),
        "smb_std": np.nanstd(smb_ensemble, axis=0),
        "dhdt_mean": np.nanmean(dhdt_ensemble),
        "dhdt_std": np.nanstd(dhdt_ensemble),
    }


def collect_observation_stats(glacier_dir: Path) -> dict:
    observations_path = glacier_dir / "observations.nc"

    if not observations_path.exists():
        return {
            "dhdt_obs_mean": np.nan,
            "dhdt_obs_std": np.nan,
        }

    try:
        with xr.open_dataset(observations_path) as ds:
            usurf_obs = ds["usurf"]
            thk0 = ds["thk"][0].values
            mask = thk0 > 1

            dhdt = (usurf_obs[1].values - usurf_obs[0].values) / 20
            dhdt_mean = np.nanmean(dhdt[mask])

            dhdt_err = ds["dhdt_err"][1].to_numpy()[mask]
            dhdt_std = np.nanmean(dhdt_err)

        return {
            "dhdt_obs_mean": dhdt_mean,
            "dhdt_obs_std": dhdt_std,
        }

    except Exception as exc:
        print(f"Warning: failed reading {observations_path}: {exc}")
        return {
            "dhdt_obs_mean": np.nan,
            "dhdt_obs_std": np.nan,
        }


# ============================================================
# Glacier row builder
# ============================================================

def build_glacier_result_row(rgi_row: pd.Series) -> dict | None:
    rgi_id = rgi_row["rgi_id"]
    glacier_dir = EXPERIMENTS_PATH / rgi_id
    result_path = glacier_dir / "calibration_results.json"

    calibration = read_calibration_json(result_path)
    if calibration is None:
        print(f"Warning: missing or invalid calibration for {rgi_id}")
        return None

    final_mean, final_std = calibration
    obs_stats = collect_observation_stats(glacier_dir)

    row_data = rgi_row.to_dict()
    aspect_deg = row_data.pop("aspect_deg")
    east_west, south_north = aspect_to_components(aspect_deg)

    row_data.update(
        {
            "EastWest": east_west,
            "SouthNorth": south_north,
            "ela": final_mean[0] if len(final_mean) > 0 else np.nan,
            "gradabl": final_mean[1] if len(final_mean) > 1 else np.nan,
            "gradacc": final_mean[2] if len(final_mean) > 2 else np.nan,
            "ela_std": final_std[0] if len(final_std) > 0 else np.nan,
            "gradabl_std": final_std[1] if len(final_std) > 1 else np.nan,
            "gradacc_std": final_std[2] if len(final_std) > 2 else np.nan,
            "dhdt_mean": obs_stats["dhdt_obs_mean"],
            "dhdt_std": obs_stats["dhdt_obs_std"],
        }
    )

    return row_data


def build_ensemble_row(rgi_id: str) -> dict:
    glacier_dir = EXPERIMENTS_PATH / rgi_id
    ensemble_stats = collect_ensemble_stats(glacier_dir)

    return {
        "rgi_id": rgi_id,
        "vel_ensemble_year0": ensemble_stats["vel_mean"][0],
        "vel_ensemble_year10": ensemble_stats["vel_mean"][1],
        "vel_ensemble_year20": ensemble_stats["vel_mean"][2],
        "smb_ensemble_year0": ensemble_stats["smb_mean"][0],
        "smb_ensemble_year10": ensemble_stats["smb_mean"][1],
        "smb_ensemble_year20": ensemble_stats["smb_mean"][2],
        "smb_ensemble_mean": np.nanmean(ensemble_stats["smb_mean"]),
        "smb_ensemble_std": np.nanmean(ensemble_stats["smb_std"]),
        "dhdt_ensemble_mean": ensemble_stats["dhdt_mean"],
        "dhdt_ensemble_std": ensemble_stats["dhdt_std"],
    }


def collect_or_load_ensemble_stats(rgi_ids, recompute: bool = False) -> pd.DataFrame:
    if ENSEMBLE_CACHE_CSV.exists() and not recompute:
        print(f"Loading ensemble cache from {ENSEMBLE_CACHE_CSV}")
        return pd.read_csv(ENSEMBLE_CACHE_CSV)

    print("Recomputing ensemble statistics")
    rows = []
    for rgi_id in rgi_ids:
        print(f"Ensemble: {rgi_id}")
        rows.append(build_ensemble_row(rgi_id))

    df = pd.DataFrame(rows)
    ENSEMBLE_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ENSEMBLE_CACHE_CSV, index=False, float_format="%.4f")
    print(f"Saved ensemble cache to {ENSEMBLE_CACHE_CSV}")
    return df


# ============================================================
# Inversion statistics (velocity and thickness vs observations)
# ============================================================
def to_scalar(x) -> float:
    """Return Python float or np.nan from xarray reduction."""
    try:
        return x.item()
    except Exception:
        return float(x) if np.isscalar(x) else np.nan


def build_inversion_row(rgi_id: str) -> dict:
    row = {"rgi_id": rgi_id}
    nc_path = EXPERIMENTS_PATH / rgi_id / "Preprocess" / "outputs" / "output.nc"
    if not nc_path.exists():
        print(f"Warning: inversion output not found: {nc_path}")
        return row

    try:
        with xr.open_dataset(nc_path) as ds:
            # Velocity (masked by icemask)
            if all(v in ds.variables for v in ["velsurf_mag", "velsurfobs_mag", "icemask"]):
                mask_ice = ds["icemask"] == 1
                for name, var in [("velsurf_mag", ds["velsurf_mag"]),
                                  ("velsurfobs_mag", ds["velsurfobs_mag"])]:
                    v = var.where(mask_ice)
                    row[f"Mean_{name}"] = to_scalar(v.mean(skipna=True))
                    row[f"Std_{name}"] = to_scalar(v.std(skipna=True))
                    row[f"Min_{name}"] = to_scalar(v.min(skipna=True))
                    row[f"Q1_{name}"] = to_scalar(v.quantile(0.25, skipna=True))
                    row[f"Median_{name}"] = to_scalar(v.quantile(0.50, skipna=True))
                    row[f"Q3_{name}"] = to_scalar(v.quantile(0.75, skipna=True))
                    row[f"Max_{name}"] = to_scalar(v.max(skipna=True))
                vmod = ds["velsurf_mag"].where(mask_ice)
                vobs = ds["velsurfobs_mag"].where(mask_ice)
                row["MAE_velsurf_mag"] = to_scalar(np.abs(vmod - vobs).mean(skipna=True))
            else:
                print(f"Warning: missing velocity variables for {rgi_id}")

            # Thickness (only where thkobs is present)
            if all(v in ds.variables for v in ["thk", "thkobs", "icemask"]):
                ice_mask = ds["icemask"] == 1
                valid = ice_mask & ds["thkobs"].notnull()
                thk_mod = ds["thk"].where(valid)
                thk_obs = ds["thkobs"].where(valid)

                n_obs = int(valid.sum().item())
                total_ice = int(ice_mask.sum().item())
                diff = thk_mod - thk_obs
                absdiff = np.abs(diff)

                row["n_obs_thk"] = n_obs
                row["coverage_pct"] = 100.0 * n_obs / total_ice if total_ice > 0 else np.nan
                row["Mean_thk_model_at_obs"] = to_scalar(thk_mod.mean(skipna=True))
                row["Std_thk_model_at_obs"] = to_scalar(thk_mod.std(skipna=True))
                row["Mean_thk_obs"] = to_scalar(thk_obs.mean(skipna=True))
                row["Std_thk_obs"] = to_scalar(thk_obs.std(skipna=True))
                row["MAE_thk"] = to_scalar(absdiff.mean(skipna=True))
                row["Bias_thk"] = to_scalar(diff.mean(skipna=True))
                row["RMSE_thk"] = to_scalar(np.sqrt((diff ** 2).mean(skipna=True)))
                row["MedianAE_thk"] = to_scalar(absdiff.quantile(0.50, skipna=True))
                row["Q95AE_thk"] = to_scalar(absdiff.quantile(0.95, skipna=True))
                row["Min_thk_obs"] = to_scalar(thk_obs.min(skipna=True))
                row["Q1_thk_obs"] = to_scalar(thk_obs.quantile(0.25, skipna=True))
                row["Median_thk_obs"] = to_scalar(thk_obs.quantile(0.50, skipna=True))
                row["Q3_thk_obs"] = to_scalar(thk_obs.quantile(0.75, skipna=True))
                row["Max_thk_obs"] = to_scalar(thk_obs.max(skipna=True))
            else:
                print(f"Warning: missing thickness variables for {rgi_id}")
    except Exception as exc:
        print(f"Warning: failed reading {nc_path}: {exc}")

    return row


def collect_or_load_inversion_stats(rgi_ids, recompute: bool = False) -> pd.DataFrame:
    if INVERSION_CACHE_CSV.exists() and not recompute:
        print(f"Loading inversion cache from {INVERSION_CACHE_CSV}")
        return pd.read_csv(INVERSION_CACHE_CSV)

    print("Recomputing inversion statistics")
    rows = []
    for rgi_id in rgi_ids:
        print(f"Inversion: {rgi_id}")
        rows.append(build_inversion_row(rgi_id))

    df = pd.DataFrame(rows)
    INVERSION_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INVERSION_CACHE_CSV, index=False)
    print(f"Saved inversion cache to {INVERSION_CACHE_CSV}")
    return df
# ============================================================
# Collect model results
# ============================================================

def load_rgi_part(part: int) -> pd.DataFrame:
    file_path = RGI_FILES_PATH / f"RGI_SELECT_PART_{part}.csv"
    if not file_path.exists():
        print(f"Warning: missing RGI split file: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    if df.empty:
        print(f"Warning: empty RGI split file: {file_path}")
    return df


def collect_model_results(parts) -> pd.DataFrame:
    all_results = []

    for part in parts:
        print(f"Processing part {part}")
        rgi_data = load_rgi_part(part)
        if rgi_data.empty:
            continue

        for _, row in rgi_data.iterrows():
            print(row['glac_name'])
            glacier_result = build_glacier_result_row(row)
            if glacier_result is not None:
                all_results.append(glacier_result)

    return pd.DataFrame(all_results)


# ============================================================
# Load external tables
# ============================================================

def load_external_tables() -> tuple[pd.DataFrame, pd.DataFrame]:

    sla_df = safe_read_csv(
        SLA_PATH,
        columns=["rgi_id", "eos_sla_mean", "eos_sla_n"],
    )

    wgms_glamos_df = safe_read_csv(
        WGMS_GLAMOS_PATH,
        columns=[
            "rgi_id",
            "glacier_name",
            "country",
            "ela",
            "ela_source",
            "gradabl",
            "gradabl_source",
            "gradacc",
            "gradacc_source",
            "annual_mass_balance",
            "annual_mass_balance_std",
            "annual_mass_balance_source",
            "n_years",
        ],
    )

    return sla_df, wgms_glamos_df


# ============================================================
# Formatting
# ============================================================

def format_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "glacier_name_x" in out.columns or "glacier_name_y" in out.columns:
        out["glacier_name"] = out.get("glacier_name_x").combine_first(out.get("glacier_name_y"))
        out = out.drop(columns=[c for c in ["glacier_name_x", "glacier_name_y"] if c in out.columns])

    # merge glacier_name + glac_name
    if "glacier_name" in out.columns and "glac_name" in out.columns:
        out["glacier_name"] = out["glacier_name"].combine_first(out["glac_name"])
        out = out.drop(columns=["glac_name"])

    if "glacier_name" in out.columns:
        cols = out.columns.tolist()
        cols.insert(1, cols.pop(cols.index("glacier_name")))
        out = out[cols]

    if "ela" in out.columns:
        out["ela"] = out["ela"].round().astype("Int64")

    return out


# ============================================================
# Main
# ============================================================

def main():
    parts = np.arange(1, 42)

    model_results = collect_model_results(parts)
    if model_results.empty:
        print("No results to save. Exiting.")
        return

    ensemble_df = collect_or_load_ensemble_stats(
        model_results["rgi_id"].unique(),
        recompute=RECOMPUTE_ENSEMBLE,
    )

    if not ensemble_df.empty:
        model_results = pd.merge(model_results, ensemble_df, on="rgi_id", how="left")

    inversion_df = collect_or_load_inversion_stats(
        model_results["rgi_id"].unique(),
        recompute=RECOMPUTE_INVERSION,
    )
    sla_df, wgms_glamos_df = load_external_tables()

    merged = model_results.copy()

    if not inversion_df.empty:
        merged = merge_drop_duplicates(merged, inversion_df, on="rgi_id", suffix="_inv")

    if not sla_df.empty:
        merged = merge_drop_duplicates(merged, sla_df, on="rgi_id", suffix="_sla")

    if not wgms_glamos_df.empty:
        merged = pd.merge(merged, wgms_glamos_df, on="rgi_id", how="left", suffixes=("", "_wgmsglamos"))

    merged = format_output(merged)

    merged.to_csv(OUTPUT_CSV, index=False, float_format="%.4f")
    print(f"Results successfully saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()