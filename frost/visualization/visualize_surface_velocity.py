import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import TwoSlopeNorm

plt.rcParams["font.family"] = "monospace"
# --------------------------------------------------
# paths
# --------------------------------------------------
path_to_netcdf = "../../data/results/central_europe_submit/glaciers/RGI2000-v7.0-G-11-01706/Preprocess/data/input.nc"
output_pdf = "Plots/glacier_velocity_dhdt_map.pdf"

# --------------------------------------------------
# open dataset
# --------------------------------------------------
ds = xr.open_dataset(path_to_netcdf)

# --------------------------------------------------
# helper
# --------------------------------------------------
def find_first_existing(dataset, candidates):
    for name in candidates:
        if name in dataset.variables:
            return name
    return None

u_name = find_first_existing(ds, ["uvelsurfobs", "uvelsurfaobs"])
v_name = find_first_existing(ds, ["vvelsurfobs", "vvelsurfaobs"])
dhdt_name = find_first_existing(ds, ["dhdt", "dhdtobs", "dhdt", "dhdt_observation"])

if u_name is None or v_name is None:
    raise ValueError(f"Could not find velocity fields. Available variables: {list(ds.variables)}")

if dhdt_name is None:
    raise ValueError(
        f"Could not find a dhdt field. Available variables: {list(ds.variables)}"
    )

for required in ["usurf", "icemask"]:
    if required not in ds.variables:
        raise ValueError(f"Could not find '{required}' in dataset.")

if "x" not in ds.coords or "y" not in ds.coords:
    raise ValueError(f"Could not find x/y coordinates. Available coords: {list(ds.coords)}")

# --------------------------------------------------
# variables
# --------------------------------------------------
u = ds[u_name].copy()
v = ds[v_name].copy()
dhdt = ds[dhdt_name].copy()
usurf = ds["usurf"]
icemask = ds["icemask"]

x = ds["x"].values
y = ds["y"].values

crop = 15  # pixels to remove on each side

# crop coordinates
x = x[crop:-crop]
y = y[crop:-crop]

# crop fields
usurf = usurf[crop:-crop, crop:-crop]
u = u[crop:-crop, crop:-crop]
v = v[crop:-crop, crop:-crop]
dhdt = dhdt[crop:-crop, crop:-crop]
icemask = icemask[crop:-crop, crop:-crop]

# --------------------------------------------------
# clean invalid values
# --------------------------------------------------
bad_threshold = 1e6

u = u.where(np.abs(u) < bad_threshold)
v = v.where(np.abs(v) < bad_threshold)
dhdt = dhdt.where(np.abs(dhdt) < bad_threshold)

if "_FillValue" in u.attrs:
    u = u.where(u != u.attrs["_FillValue"])
if "_FillValue" in v.attrs:
    v = v.where(v != v.attrs["_FillValue"])
if "_FillValue" in dhdt.attrs:
    dhdt = dhdt.where(dhdt != dhdt.attrs["_FillValue"])

# --------------------------------------------------
# mask to glacier only
# --------------------------------------------------
ice = icemask > 0
u = u.where(ice)
v = v.where(ice)
dhdt = dhdt.where(ice)

vel_mag = np.sqrt(u**2 + v**2).where(ice)

# --------------------------------------------------
# cell center coordinates for quiver

# --------------------------------------------------
dx = np.median(np.diff(x)) if len(x) > 1 else 0
dy = np.median(np.diff(y)) if len(y) > 1 else 0

Xc, Yc = np.meshgrid(x + dx / 2, y + dy / 2)

step = 10

Xq = Xc[::step, ::step]
Yq = Yc[::step, ::step]
Uq = u.values[::step, ::step]
Vq = v.values[::step, ::step]
Mq = ice.values[::step, ::step]

valid = np.isfinite(Uq) & np.isfinite(Vq) & (Mq > 0)

Xq = Xq[valid]
Yq = Yq[valid]
Uq = Uq[valid]
Vq = Vq[valid]

# --------------------------------------------------
# coordinate transformer
# EPSG:32632 -> EPSG:4326
# --------------------------------------------------
transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)

def format_lon(x_val, pos):
    lon, lat = transformer.transform(x_val, np.mean(y))
    return f"{lon:.2f}°E"

def format_lat(y_val, pos):
    lon, lat = transformer.transform(np.mean(x), y_val)
    return f"{lat:.2f}°N"

# --------------------------------------------------
# dhdt normalization centered at 0
# --------------------------------------------------
dhdt_vals = dhdt.values[np.isfinite(dhdt.values)]
dhdt_abs = np.nanpercentile(np.abs(dhdt_vals), 98) if dhdt_vals.size > 0 else 1.0
dhdt_norm = TwoSlopeNorm(vmin=-dhdt_abs, vcenter=0.0, vmax=dhdt_abs)

# --------------------------------------------------
# figure
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharex=True, sharey=True)

# ==========================
# panel 1: velocity
# ==========================
ax = axes[0]

im0 = ax.pcolormesh(
    x, y, usurf.values,
    cmap="gray",
    shading="auto",
    rasterized=True
)

im1 = ax.pcolormesh(
    x, y, vel_mag.values,
    cmap="magma",
    shading="auto",
    zorder=2,
    rasterized=True
)

q = ax.quiver(
    Xq, Yq, Uq, Vq,
    color="white",
    scale=800,
    width=0.007,
    headwidth=3.0,
    headlength=4.0,
    headaxislength=3.5,
    pivot="middle",
    zorder=3
)

cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
cbar1.set_label(r"Surface velocity magnitude (m yr$^{-1}$)", labelpad=10)

ax.set_title("Observed surface velocity\n2017-2018")

# ==========================
# panel 2: dhdt
# ==========================
ax2 = axes[1]

im0b = ax2.pcolormesh(
    x, y, usurf.values,
    cmap="gray",
    shading="auto",
    rasterized=True
)

im2 = ax2.pcolormesh(
    x, y, dhdt.values,
    cmap="RdBu",
    norm=dhdt_norm,
    shading="auto",
    zorder=2,
    rasterized=True
)

cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label(r"$dh/dt$ (m yr$^{-1}$)", labelpad=10)
# ==========================
# shared usurf colorbar
# ==========================

ax2.set_title("Observed elevation change rate\n2000-2019")

# --------------------------------------------------
# shared styling
# --------------------------------------------------
for ax in axes:
    ax.tick_params(axis='both', which='both', length=0)
    # ax.tick_params(axis='y', labelrotation=90)
    # for label in ax.get_yticklabels():
    #     label.set_verticalalignment('center')
        #label.set_horizontalalignment('center')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.grid(axis="y", color="lightgray", linestyle="--", zorder=-10)
    ax.grid(axis="x", color="lightgray", linestyle="--", zorder=-10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")

axes[0].set_ylabel("Latitude")

plt.tight_layout()
plt.savefig(output_pdf, format="pdf", bbox_inches="tight")

print(f"Saved figure to: {output_pdf}")