import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

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
divflux = np.array(ds_optimized["divflux"])
thkobs= np.array(ds_optimized["thkobs"])


velocity_obs = np.array(ds_optimized["velsurfobs_mag"])

fig, ax = plt.subplots(2, 4, figsize=(10, 6))



p = 25
for i,axi in enumerate(ax.flatten()):
    surface_im = axi.imshow(surface[p:-p,p:-p], cmap='gray',
                                               vmin=1450,
                                          vmax=3600,
                                   origin='lower')
    if i==3:

        cbar = fig.colorbar(surface_im)
        cbar.ax.set_ylabel('Surface Elevation (m)', rotation=90)
        ax[0,3].set_title("Surface Elevation\n[NASADEM]")


velocity_obs[icemask < 0.01] = None
vel_img = ax[0,0].imshow(velocity_obs[p:-p,p:-p], vmin=0, vmax=np.nanmax(velocity_obs),
                       cmap="magma", zorder=2, origin="lower")
cbar = fig.colorbar(vel_img)
cbar.ax.set_ylabel('Surface Velocity (m a$^{-1}$)', rotation=90)
ax[0,0].set_title("Observed Velocity\n[Millan22]")



velocity_iter = velocity
velocity_obs[icemask < 0.01] = None
dif = velocity_iter[p:-p,p:-p]-velocity_obs[p:-p,p:-p]
vel_img = ax[0,1].imshow(dif, vmin=-50, vmax=50,
                       cmap="RdBu_r", zorder=2, origin='lower')
cbar = fig.colorbar(vel_img)
cbar.ax.set_ylabel('Velocity Difference (m a$^{-1}$)', rotation=90)
ax[0,1].set_title("Velocity Difference\nModel - Observed")

velocity_iter[icemask < 0.01] = None
vel_img = ax[1,0].imshow(velocity_iter[p:-p,p:-p], vmin=0, vmax=np.nanmax(
    velocity_obs),
                       cmap="magma", zorder=2, origin='lower')
cbar = fig.colorbar(vel_img)
cbar.ax.set_ylabel('Surface Velocity (m a$^{-1}$)', rotation=90)
ax[1,0].set_title("Model Velocity")

slidingco_iter = slidingco
slidingco_iter[icemask < 0.01] = None
img = ax[1,1].imshow(slidingco_iter[p:-p,p:-p], zorder=2, origin='lower')
cbar = fig.colorbar(img)
cbar.ax.set_ylabel('Sliding Co. (MPa a$^{3}$ m$^{-3}$)', rotation=90)
cbar.formatter = mticker.FuncFormatter(lambda x, _: f"{x:.3f}")
cbar.update_ticks()
ax[1,1].set_title("Model Sliding Co.")
resolution = ds_optimized.x[1].data - ds_optimized.x[0].data


thickness[icemask < 0.01] = None
img = ax[1,2].imshow(thickness[p:-p,p:-p], cmap='Blues', zorder=2, origin='lower')
cbar = fig.colorbar(img)
cbar.ax.invert_yaxis()
cbar.ax.set_ylabel('Thickness (m)', rotation=90)
ax[1,2].set_title("Model Thickness ")

thkobs
img = ax[0,2].imshow(thkobs[p:-p,p:-p], cmap='Blues', zorder=2, origin='lower',
                     interpolation='nearest')
cbar = fig.colorbar(img)
cbar.ax.invert_yaxis()
cbar.ax.set_ylabel('Thickness (m)', rotation=90)
ax[0,2].set_title("Observed Thickness\n[GlaThiDa]")



divflux[icemask < 0.01] = None
img = ax[1,3].imshow(divflux[p:-p,p:-p], cmap='RdBu_r', zorder=2, origin='lower')
cbar = fig.colorbar(img)
cbar.ax.set_ylabel('Flux Divergence (m a$^{-1}$)', rotation=90)


ax[1,3].set_title("Flux Divergence")




def formatter(x, pos):
    del pos
    return str(int(x * resolution / 1000))


for axi in ax.flatten():
    #ax[int(i/3), int(i%3)].invert_yaxis()
    axi.set_xticks(np.arange(25, dif.shape[1] , step=resolution))  # From 25 to
    # 130, step 20                             step=self.resolution)
    axi.set_yticks(np.arange(25, dif.shape[0] , step=resolution)  )
    # ax[i].yaxis.set_ticks([20, 40, 60, 80, 100])
    # ax[i].xaxis.set_ticks([ 20, 40, 60])
    axi.xaxis.set_major_formatter(formatter)
    axi.yaxis.set_major_formatter(formatter)
    axi.grid(axis="y", color="black", linestyle="--", zorder=0, alpha=.2)
    axi.grid(axis="x", color="black", linestyle="--", zorder=0, alpha=.2)

    for axis in ['top', 'bottom', 'left', 'right']:
        axi.spines[axis].set_linewidth(0)
    axi.set_xlabel('km', color='black')
    axi.tick_params(axis='x', colors='black')
    axi.tick_params(axis='y', colors='black')

for i in range(4):
    ax[0,i].set_xlabel('')

ax[0,0].set_ylabel('km')
ax[1,0].set_ylabel('km')

# fig.suptitle("inversion", fontsize=32)
import string
axes = ax.flatten()  # Flatten for easy iteration

labels_subplot = [f"{letter})" for letter in
                  string.ascii_lowercase[:len(axes)]]

for ax, label in zip(axes, labels_subplot):
    # Add label to lower-left corner (relative coordinates)
    ax.text(-0.35, 1.02, label, transform=ax.transAxes,
            fontsize=12, va='bottom', ha='left', fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(wspace=0.5, left=0.05, )

plt.savefig(figure_path + '.pdf', format='pdf')
plt.savefig(figure_path + '.png', format='png')
