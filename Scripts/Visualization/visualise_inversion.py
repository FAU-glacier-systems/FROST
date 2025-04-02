import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# output_file = '../ReferenceSimulation/output.nc'
optimized_file = (
    '../../Data/Glaciers/RGI2000-v7.0-G-11-01706/Inversion/geology-optimized.nc')
figure_path = "../../Plots/inversion_result"

# ds = xr.open_dataset(output_file)
ds_optimized = xr.open_dataset(optimized_file)

icemask = np.array(ds_optimized["icemask"])
surface = np.array(ds_optimized["usurf"])

arrhenius = np.array(ds_optimized["arrhenius"])
slidingco = np.array(ds_optimized["slidingco"])
velocity = np.array(ds_optimized["velsurf_mag"])
thickness = np.array(ds_optimized["thk"])

velocity_obs = np.array(ds_optimized["velsurfobs_mag"])

fig, ax = plt.subplots(2, 3, figsize=(11, 8))
plt.subplots_adjust(left=0, bottom=0.05, right=0.95, top=0.95, wspace=0.1,
                    hspace=0.1)

for i in range(6):
    surface_im = ax[int(i/3), int(i%3)].imshow(surface, cmap='gray', vmin=1450,
                                          vmax=3600,
                                   origin='lower')

cbar = fig.colorbar(surface_im)
cbar.ax.set_ylabel('Surface Elevation (m)', rotation=90)
ax[1,2].set_title("Surface Elevation [NASADEM]")


velocity_obs[icemask < 0.01] = None
vel_img = ax[0,0].imshow(velocity_obs, vmin=0, vmax=np.nanmax(velocity_obs),
                       cmap="magma", zorder=2, origin="lower")
cbar = fig.colorbar(vel_img)
cbar.ax.set_ylabel('Surface Velocity (m a$^{-1}$)', rotation=90)
ax[0,0].set_title("Observed Velocity [Millan22]")


velocity_iter = velocity
velocity_obs[icemask < 0.01] = None
vel_img = ax[0,1].imshow(velocity_iter-velocity_obs, vmin=-10, vmax=10,
                       cmap="RdBu", zorder=2, origin='lower')
cbar = fig.colorbar(vel_img)
cbar.ax.set_ylabel('Surface Velocity Difference (m a$^{-1}$)', rotation=90)
ax[0,1].set_title("Velocity Difference\nmodelled - observed")

velocity_iter[icemask < 0.01] = None
vel_img = ax[0,2].imshow(velocity_iter, vmin=0, vmax=np.nanmax(velocity_obs),
                       cmap="magma", zorder=2, origin='lower')
cbar = fig.colorbar(vel_img)
cbar.ax.set_ylabel('Surface Velocity (m a$^{-1}$)', rotation=90)
ax[0,2].set_title("Modelled Velocity")

thickness[icemask < 0.01] = None
img = ax[1,0].imshow(thickness, cmap='Blues', zorder=2, origin='lower')
cbar = fig.colorbar(img)
cbar.ax.invert_yaxis()
cbar.ax.set_ylabel('Thickness (m)', rotation=90)

ax[1,0].set_title("Optimised Thickness ")

slidingco_iter = slidingco
slidingco_iter[icemask < 0.01] = None
img = ax[1,1].imshow(slidingco_iter, zorder=2, origin='lower')
cbar = fig.colorbar(img)
cbar.ax.set_ylabel('Sliding Co. (MPa a$^{3}$ m$^{-3}$)', rotation=90)

ax[1,1].set_title("Optimised Sliding Co.")
resolution = ds_optimized.x[1].data - ds_optimized.x[0].data




def formatter(x, pos):
    del pos
    return str(int(x * resolution / 1000))


for i in range(6):
    #ax[int(i/3), int(i%3)].invert_yaxis()
    # ax[i].set_xlim(15, 65)
    # ax[i].set_ylim(15, 105)
    # ax[i].yaxis.set_ticks([20, 40, 60, 80, 100])
    # ax[i].xaxis.set_ticks([ 20, 40, 60])
    ax[int(i/3), int(i%3)].xaxis.set_major_formatter(formatter)
    ax[int(i/3), int(i%3)].yaxis.set_major_formatter(formatter)
    ax[int(i/3), int(i%3)].grid(axis="y", color="black", linestyle="--", zorder=0, alpha=.2)
    ax[int(i/3), int(i%3)].grid(axis="x", color="black", linestyle="--", zorder=0, alpha=.2)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[int(i/3), int(i%3)].spines[axis].set_linewidth(0)
    ax[int(i/3), int(i%3)].set_xlabel('km', color='black')
    ax[int(i/3), int(i%3)].tick_params(axis='x', colors='black')
    ax[int(i/3), int(i%3)].tick_params(axis='y', colors='black')

# fig.suptitle("inversion", fontsize=32)
plt.tight_layout()
plt.savefig(figure_path + '.pdf', format='pdf')
plt.savefig(figure_path + '.png', format='png')
