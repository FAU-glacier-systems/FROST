import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import rasterio
import cartopy.crs as ccrs
from rasterio.plot import show
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from adjustText import adjust_text


# -----------------------------------------------------
# 1. Load and prepare the DEM (GeoTIFF)
# -----------------------------------------------------
def load_dem(dem_path):
    """
    Load DEM (Digital Elevation Model) data from a GeoTIFF file.
    """
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        transform = src.transform
        crs = src.crs
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom,
                  src.bounds.top]

        # Mask invalid data
        dem_data[dem_data > 30000] = 0

    return dem_data, extent


# -----------------------------------------------------
# 2. Load and prepare glacier data from CSV
# -----------------------------------------------------
def load_glacier_data(csv_path):
    """
    Load glacier data from a CSV file and prepare it as a GeoDataFrame.
    """
    df = pd.read_csv(csv_path)

    # Filter and sort by area

    df = df[pd.to_numeric(df["area_km2"], errors="coerce").notna()]
    df = df.sort_values(by="area_km2", ascending=False).reset_index(drop=True)

    # Create GeoDataFrame and reproject to UTM (EPSG:32632)
    geometry = [Point(xy) for xy in zip(df['cenlon'], df['cenlat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=32632)

    return gdf


# -----------------------------------------------------
# 3. Load and plot country borders
# -----------------------------------------------------
def plot_country_borders(ax, country_paths):
    """
    Plot country borders (e.g., Switzerland, Italy) on the axes.
    """
    for path in country_paths:
        country_gdf = gpd.read_file(path).to_crs(epsg=32632)
        country_gdf.boundary.plot(ax=ax, color="white", linewidth=1, zorder=3,
                                  alpha=0.5,)


# -----------------------------------------------------
# 4. Generalized Plotting Function with Annotations
# -----------------------------------------------------
def plot_map_with_annotations(
    dem_data, extent, gdf, 
    save_path, country_paths, 
    value_column, color_map, colorbar_label, num_dec=1
):
    """
    Generalized function for plotting a map with glacier data and text annotations.
    Arguments:
    - value_column: the column name in gdf to be used for color coding
    - color_map: the colormap to use for visualizing `value_column`
    - colorbar_label: label for the colorbar
    """
    # Set up UTM projection for Cartopy
    utm_crs = ccrs.UTM(zone=32)
    fig, ax = plt.subplots(figsize=(10, 5.5), subplot_kw={'projection': utm_crs})

    # Plot DEM

    img = ax.imshow(
        dem_data,
        origin='upper',
        extent=extent,
        cmap='gray',
        transform=utm_crs,
        vmin=-100,
        vmax=3100
    )

    # Plot country borders
    #plot_country_borders(ax, country_paths)

    # Plot glacier points (scaled by area, colored by the selected value_column)
    sizes = gdf["area_km2"] * 20
    if value_column=='ela_sla':
        colors = gdf['ela']  # Dynamic value to colormap
    else:
        colors = gdf[value_column]  # Dynamic value to colormap

    from matplotlib.markers import MarkerStyle
    from matplotlib.patches import Wedge
    # Function to create a half-circle marker path
    from matplotlib.path import Path

    def half_circle_marker(theta1=0, theta2=np.pi, n=32):
        angles = np.linspace(theta1, theta2, n)
        verts = np.column_stack([np.cos(angles), np.sin(angles)])
        verts = np.vstack([[0, 0], verts, [0, 0]])
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
        return Path(verts, codes)

    upper_half_path = half_circle_marker(0, np.pi)  # flat edge at bottom, arc on top
    lower_half_path = half_circle_marker(np.pi, 2 * np.pi)  # flat edge at top, arc on bottom

    if value_column == "ela" or value_column == "sla":
        scatter = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=colors, s=sizes, vmin=2850, vmax=3550,
            transform=utm_crs, cmap=color_map,
            zorder=10, alpha=1, edgecolor='k', linewidth=0.8,
        )


    elif value_column == "ela_sla":
        # both in the same plot: ELA on upper half, SLA on lower half

        # same vmin/vmax, cmap, paths defined earlier
        vmin, vmax = 2700, 3700

        x = gdf.geometry.x.to_numpy()
        y = gdf.geometry.y.to_numpy()
        ela = gdf["ela"].to_numpy()
        sla = gdf["sla"].to_numpy()
        s = sizes.to_numpy() if hasattr(sizes, "to_numpy") else np.asarray(sizes)

        # plot smaller first, larger last
        #order = np.argsort(s)

        scatter_ela_last = None
        scatter_sla_last = None
        for i in np.arange(len(x)):
            # lower half first (SLA)
            scatter_sla_last = ax.scatter(
                [x[i]], [y[i]],
                c=[sla[i]], s=[s[i]],
                vmin=vmin, vmax=vmax, cmap=color_map, transform=utm_crs,
                marker=lower_half_path, edgecolor='k', linewidth=0.2,
                alpha=1,
            )
            # upper half on top (ELA)
            scatter_ela_last = ax.scatter(
                [x[i]], [y[i]],
                c=[ela[i]], s=[s[i]],
                vmin=vmin, vmax=vmax, cmap=color_map, transform=utm_crs,
                marker=upper_half_path, edgecolor='k', linewidth=0.2,
                alpha=1,
            )

        scatter = scatter_ela_last

    elif value_column == "sla_ela_diff":
        scatter = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=colors, s=sizes, vmin=-300, vmax=300,
            transform=utm_crs, cmap=color_map,
            zorder=10, alpha=1, edgecolor='k', linewidth=0.8,
        )
    elif value_column == "gradabl":
        scatter = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=colors, s=sizes, vmin=0, vmax=16,
            transform=utm_crs, cmap=color_map,
            zorder=10, alpha=1, edgecolor='k', linewidth=0.8,
        )
    elif value_column == "gradacc":
        scatter = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=colors, s=sizes, vmin=0, vmax=6,
            transform=utm_crs, cmap=color_map,
            zorder=10, alpha=1, edgecolor='k', linewidth=0.8,
        )
    else:

        scatter = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=colors, s=sizes,
            transform=utm_crs, cmap=color_map,
            zorder=10, alpha=1, edgecolor='k', linewidth=0.8,
        )
    if value_column == 'ela_sla':
        value_column = 'ela'
    # Text Annotations (e.g., glacier names or IDs)
    texts = []
    for _, row in gdf.head(5).iterrows():
        # Add text annotations slightly offset from points
        text = ax.text(
            row.geometry.x, row.geometry.y,
            f"{row['glac_name']} ({row[value_column]:.{num_dec}f})",
            ha='center', va='center',
            color='white',
            fontsize=8,
            zorder=12,
            bbox=dict(
                facecolor='black',  # Background color
                edgecolor='none',  # No border
                boxstyle='round,pad=0.2',  # Rounded corners, padded
                alpha=0.6  # Optional transparency
            )
        )

        texts.append(text)

    # Set dynamic extent based on glacier points
    buffer = 0  # in meters
    min_x, min_y, max_x, max_y = (
        gdf.geometry.x.min() - buffer, gdf.geometry.y.min() - buffer,
        gdf.geometry.x.max() + buffer, gdf.geometry.y.max() + buffer
    )
    ax.set_extent([min_x, max_x, min_y, max_y], crs=utm_crs)

    # Adjust text annotations to avoid overlapping
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='->', color='white', lw=0.5, zorder=11),
        expand=(3, 3), avoid_self=True, only_move={"text":"y"},
    )
    # Set dynamic extent based on glacier points
    buffer = 20000  # in meters
    min_x, min_y, max_x, max_y = (
        gdf.geometry.x.min() - buffer, gdf.geometry.y.min() - buffer,
        gdf.geometry.x.max() + buffer, gdf.geometry.y.max() + buffer
    )
    ax.set_extent([min_x, max_x, min_y, max_y], crs=utm_crs)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Add colorbar centered at bottom
    cax = inset_axes(
        ax,
        width="50%", height="5%",
        loc='lower center',
        bbox_to_anchor=(0, 0.1, 1, 1),  # centered under x-axis
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    cbar = plt.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_label_position('top')  # label on top
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.set_label(colorbar_label, labelpad=10, loc='center', color='white')

    # Make tick labels white
    cbar.ax.tick_params(colors='white')

    import matplotlib.ticker as mticker

    # Add gridlines and labels
    gl = ax.gridlines(
        draw_labels=True,
        color='gray',
        alpha=0.7,
        linestyle='--',
        zorder=1
    )

    # Hide top and right labels
    gl.top_labels = gl.right_labels = False

    # Set ticks only at full degrees
    gl.xlocator = mticker.MultipleLocator(1)  # Every 1° in longitude
    gl.ylocator = mticker.MultipleLocator(1)  # Every 1° in latitude

    # Add colorbar for elevation
    cax2 = inset_axes(
        ax,
        width="2%", height="40%",  # adjust size
        loc='lower right',
        bbox_to_anchor=(-0.06, 0.01, 1, 1),  # relative to axes
        bbox_transform=ax.transAxes,
        borderpad=1
    )
    cbar2 = plt.colorbar(img, cax=cax2)  # vertical by default

    # keep tick labels
    cbar2.ax.tick_params(colors="white", labelsize=8)

    # instead of set_label, use set_title for horizontal text on top
    cbar2.ax.set_title("Elevation (m)",
                       color="white",
                       fontsize=10,
                       pad=10)  # move it up a bit

    sizes = [40, 200, 400]  # marker sizes
    labels = [f"{int(s / 20)}" for s in sizes]

    handles = [plt.scatter([], [], s=s, color="none", alpha=1, edgecolors='white') for s in sizes]

    leg = ax.legend(handles, labels,
                    title="Area (km$^2$)",
                    loc="lower right",
                    bbox_to_anchor=(0.89, 0.05),  # move slightly left
                    frameon=False)

    # make text white
    plt.setp(leg.get_title(), color="white")  # title
    plt.setp(leg.get_texts(), color="white")  # labels

    # --- Small Europe overview inset with main-extent box ---
    import cartopy.feature as cfeature
    from cartopy.mpl.geoaxes import GeoAxes
    from shapely.geometry import box

    crs = ccrs.PlateCarree(central_longitude=10)

    # Create a GeoAxes inset in the upper-left
    ov = inset_axes(
        ax, width="15%", height="25%", loc='upper left',
        axes_class=GeoAxes,
        axes_kwargs=dict(map_projection=ccrs.epsg(3035)),
        borderpad=0.6
    )

    # Show Europe (tweak to taste)
    ov.set_extent([-25, 25, 32, 69], crs=crs)
    ov.add_feature(cfeature.OCEAN, facecolor='1', zorder=0)
    ov.add_feature(cfeature.LAND, facecolor='0.5', zorder=1)
    #ov.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='white', zorder=2)
    #ov.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='white', zorder=2)

    # Remove ticks/grid on the inset
    ov.set_xticks([]);
    ov.set_yticks([])
    #ov.outline_patch.set_edgecolor('white')
    #ov.outline_patch.set_linewidth(1)

    # Get the main axes extent in PlateCarree and draw it on the inset
    xmin, xmax, ymin, ymax = ax.get_extent(crs=crs)
    main_box = box(xmin, ymin, xmax, ymax)
    ov.add_geometries(
        [main_box], crs=crs,
        facecolor='none', edgecolor='yellow', linewidth=1.5, zorder=3
    )

    # Save the figure
    #plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Map saved to {save_path}")
    plt.close()


# -----------------------------------------------------
# 5. Main Function
# -----------------------------------------------------
def main():
    # File paths
    dem_path = "../../data/raw/visualization_context/alpsDEM.tif"
    csv_path = "../central_europe_submit/tables/aggregated_results.csv"
    sla_path = "../../data/raw/central_europe/Alps_EOS_SLA_2000-2019_mean.csv"
    country_paths = [
        "../../data/raw/visualization_context/gadm41_CHE_shp/gadm41_CHE_0.shp",
        "../../data/raw/visualization_context/gadm41_ITA_shp/gadm41_ITA_0.shp"
    ]

    # Load data
    print("Loading DEM...")
    dem_data, extent = load_dem(dem_path)

    print("Loading glacier data...")
    gdf = load_glacier_data(csv_path)
    sla_df = load_glacier_data(sla_path)
    # Filter rows where 'sla' > 4000
    #sla_df = sla_df[sla_df['sla'] < 4000]

    merged_df = pd.merge(gdf, sla_df, on="rgi_id", how="left",
                         suffixes=('', '_drop'))
    
    
    # Calculate bias correction as mean difference
    # bias_correction = np.mean(merged_df['ela'] - merged_df['sla'])
    # # Apply bias correction to SLA values
    # merged_df['sla_corrected'] = merged_df['sla'] + bias_correction
    # # Calculate corrected difference
    merged_df['sla_ela_diff'] = merged_df['ela']-merged_df['sla']#-
    # merged_df[
    # 'sla_corrected']

    gdf = merged_df.loc[:, ~merged_df.columns.str.endswith('_drop')]
    print(
        f"Mean absolute difference after bias correction: {np.mean(abs(gdf['sla_ela_diff'])):.2f}")

    # Map for "ela"
    print("Plotting map for 'ela'...")
    plot_map_with_annotations(
        dem_data, extent, gdf,
        save_path="../central_europe_submit/plots/ALPS_ela_sla_scatter.pdf",
        country_paths=country_paths,
        value_column="ela_sla",  # Column for ELA
        color_map="viridis_r",  # Colormap
        colorbar_label="Equilibrium Line Altitude (m)",
        num_dec=0,
    )

    plot_map_with_annotations(
        dem_data, extent, gdf,
        save_path="../central_europe_submit/plots/ALPS_ela_Scatter.pdf",
        country_paths=country_paths,
        value_column="ela",  # Column for ELA
        color_map="viridis_r",  # Colormap
        colorbar_label="Equilibrium Line Altitude (m)",
        num_dec=0,
    )

    # Map for "gradabl" (reds)
    print("Plotting map for 'gradabl'...")
    plot_map_with_annotations(
        dem_data, extent, gdf,
        save_path="../central_europe_submit/plots/ALPS_gradabl_Scatter.pdf",
        country_paths=country_paths,
        value_column="gradabl",  # Column for Gradient Ablation
        color_map="Reds",       # Colormap
        colorbar_label="Gradient Ablation (m$\,$yr$^{-1}\,$km$^{-1}$)"
    )

    # Map for "gradacc" (blues)
    print("Plotting map for 'gradacc'...")
    plot_map_with_annotations(
        dem_data, extent, gdf,
        save_path="../central_europe_submit/plots/ALPS_gradacc_Scatter.pdf",
        country_paths=country_paths,
        value_column="gradacc",  # Column for Gradient Accumulation
        color_map="Blues",       # Colormap
        colorbar_label="Gradient Accumulation (m$\,$yr$^{-1}\,$km$^{-1}$)"
    )

    plot_map_with_annotations(
        dem_data, extent, gdf,
        save_path="../central_europe_submit/plots/ALPS_sla_Scatter.pdf",
        country_paths=country_paths,
        value_column="sla",  # Column for ELA
        color_map="viridis_r",  # Colormap
        colorbar_label="End of summer snowline altitude (m)",
        num_dec=0,
    )

    plot_map_with_annotations(
        dem_data, extent, gdf,
        save_path="../central_europe_submit/plots/ALPS_difslaela_Scatter.pdf",
        country_paths=country_paths,
        value_column="sla_ela_diff",  # Column for ELA
        color_map="RdBu",  # Colormap
        colorbar_label="Difference between SLA and ELA (m)",
        num_dec=0,
    )


if __name__ == "__main__":
    main()