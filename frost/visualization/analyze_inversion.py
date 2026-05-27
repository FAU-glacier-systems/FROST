
import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

# Load CSV
glacier_list = "../../data/raw/glamos/GLAMOS_RGI.csv"
df = pd.read_csv(glacier_list)

outdir = "Plots"
os.makedirs(outdir, exist_ok=True)

vars_to_plot = {
    "velsurfobs_mag": {
            "title": "Surface Velocity\n (Observed)",
            "cmap": "magma",
            "cbar_label": "Velocity (m yr$^{-1}$)",
            "vmin": 0,
            "vmax": None,
            },
    "velsurf_mag": {
        "title": "Surface Velocity\n(Model)",
        "cmap": "magma",
        "cbar_label": "Velocity (m yr$^{-1}$)",
        "vmin": 0,
        "vmax": None,
    },
    "thk": {
        "title": "Ice Thickness",
        "cmap": "Blues",
        "cbar_label": "Thickness (m)",
        "vmin": 0,
        "vmax": None,
    },


    "smb": {
        "title": "Surface Mass Balance",
        "cmap": "RdBu",
        "cbar_label": "SMB (m yr$^{-1}$)",
        "vmin": -7.5,
        "vmax": 7.5,
    },
}

for row in df.itertuples(index=False):
    rgi_id = str(row.rgi_id)
    glamos_name = row.glamos_name

    print(rgi_id, glamos_name)
    rgi_dir = f"../../data/results/central_europe_submit/glaciers/{rgi_id}"
    nc_path = rgi_dir+"/Preprocess/outputs/output.nc"
    smb_results = rgi_dir+"/calibration_results.json"




    try:
        ds = xr.open_dataset(nc_path)
    except Exception:
        print(f"{rgi_id} does not exist or cannot be opened")
        continue

    with open(smb_results, "r") as f:
        data = json.load(f)
        ela = data["final_mean"][0]
        grad_abl = data["final_mean"][1]
        grad_acc = data["final_mean"][2]

    smb = np.array(ds["usurf"] - ela)
    smb[smb<0] *= grad_abl/1000
    smb[smb>0] *= grad_acc/1000

    # Pick a 2D ice mask (handle possible time dimension)
    icem = ds["icemask"]
    if "time" in icem.dims:
        icem = icem.isel(time=0)
    icem_np = np.array(icem)

    # ensure same extend of colorbar
    max_vel = max(np.max(ds["velsurfobs_mag"]).data,np.max(ds["velsurf_mag"]).data)
    vars_to_plot["velsurf_mag"]["vmax"] = max_vel
    vars_to_plot["velsurfobs_mag"]["vmax"] = max_vel

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes = axes.ravel()

    for ax, (name, meta) in zip(axes, vars_to_plot.items()):

        if name == "smb":
            da = smb


        else:
            da = ds[name]
            if "time" in da.dims:
                da = da.isel(time=0)

        arr = np.array(da, dtype=float)
        arr[icem_np == 0] = np.nan

        im = ax.imshow(
            arr, zorder=3,
            origin="lower",
            cmap=meta["cmap"],
            vmin=meta["vmin"],
            vmax=meta["vmax"],
        )

        ax.set_title(meta["title"])

        from matplotlib.ticker import FuncFormatter

        resolution = float(ds["x"][1] - ds["x"][0])  # meters per pixel
        crop_padding = 25

        ny, nx = da.shape[-2], da.shape[-1]
        x0, x1 = crop_padding, nx - 1 - crop_padding
        y0, y1 = crop_padding, ny - 1 - crop_padding


        def ticks_start1km(p0, p1, res_m_per_pix, max_ticks=4, min_km=1.0):
            pmin, pmax = (p0, p1) if p0 <= p1 else (p1, p0)

            span_km = (pmax - pmin) * res_m_per_pix / 1000.0
            if span_km <= min_km:
                return np.array([])

            # choose step so we never exceed max_ticks
            step_km = max(min_km, span_km / max_ticks)
            step_km = np.ceil(step_km)  # round to full km

            # ticks in km starting at 1 km
            ticks_km = np.arange(1, span_km + 1e-9, step_km)
            ticks_km = ticks_km[:max_ticks]

            # convert back to pixels
            ticks_pix = pmin + (ticks_km * 1000.0 / res_m_per_pix)

            return ticks_pix


        x_ticks = ticks_start1km(x0, x1, resolution)
        y_ticks = ticks_start1km(y0, y1, resolution)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # labels in km relative to cropped origin
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda pix, _: f"{(pix - x0) * resolution / 1000:.0f}"
        ))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda pix, _: f"{(pix - y0) * resolution / 1000:.0f}"
        ))

        ax.set_xlabel("Distance (km)")
        ax.grid(axis="y", color="black", linestyle="--", zorder=-1, alpha=.2)
        ax.grid(axis="x", color="black", linestyle="--", zorder=-1, alpha=.2)
        ax.xaxis.set_tick_params(bottom=False)
        ax.yaxis.set_tick_params(left=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(meta["cbar_label"])

    import string
    axes_with_label = [axes[0], axes[1], axes[2],  axes[3]]
    labels_subplot = [f"{letter})" for letter in string.ascii_lowercase[:len(axes_with_label)]]
    for ax, label in zip(axes_with_label, labels_subplot):
        ax.text(-.1, 1.02, label, transform=ax.transAxes,
                fontsize=12, va='bottom', ha='left', fontweight='bold')

    fig.suptitle(glamos_name)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{rgi_id}_4panel.pdf"))
    plt.close(fig)
