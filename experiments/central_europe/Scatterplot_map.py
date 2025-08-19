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
        vmax=4000
    )

    # Plot country borders
    plot_country_borders(ax, country_paths)

    # Plot glacier points (scaled by area, colored by the selected value_column)
    sizes = gdf["area_km2"] * 10  # Scale size by area
    colors = gdf[value_column]  # Dynamic value to colormap
    if value_column == "ela" or value_column == "sla":
        scatter = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=colors, s=sizes, vmin=2600, vmax=3700,
            transform=utm_crs, cmap=color_map,
            zorder=10, alpha=1, edgecolor='k', linewidth=0.8,
        )
    elif value_column == "sla_ela_diff":
        scatter = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=colors, s=sizes, vmin=-300, vmax=300,
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


    # Add colorbar
    cax = inset_axes(
        ax,
        width="100%", height="50%",
        loc='upper center',
        bbox_to_anchor=(0, 1.05, 1, 0.1),
        bbox_transform=ax.transAxes, borderpad=0
    )
    cbar = plt.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_label_position('top')  # Move label above the colorbar
    cbar.ax.xaxis.set_ticks_position('bottom')  # Keep ticks below
    cbar.set_label(colorbar_label, labelpad=10, loc='center')  #

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
    cbar2 = plt.colorbar(img, ax=ax, shrink=0.7)
    cbar2.set_label("Elevation (m)")

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Map saved to {save_path}")
    plt.close()


# -----------------------------------------------------
# 5. Main Function
# -----------------------------------------------------
def main():
    # File paths
    dem_path = "../../data/raw/visualization_context/alpsDEM.tif"
    csv_path = "../glamos_run/aggregated_results.csv"
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
    sla_df = sla_df[sla_df['sla'] < 4000]

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
        save_path="../glamos_run/plots/ALPS_ela_Scatter.pdf",
        country_paths=country_paths,
        value_column="ela",  # Column for ELA
        color_map="viridis",  # Colormap
        colorbar_label="Equilibrium Line Altitude (m)",
        num_dec=0,
    )

    # Map for "gradabl" (reds)
    print("Plotting map for 'gradabl'...")
    plot_map_with_annotations(
        dem_data, extent, gdf,
        save_path="../glamos_run/plots/ALPS_gradabl_Scatter.pdf",
        country_paths=country_paths,
        value_column="gradabl",  # Column for Gradient Ablation
        color_map="Reds",       # Colormap
        colorbar_label="Gradient Ablation (m/a/km)"
    )

    # Map for "gradacc" (blues)
    print("Plotting map for 'gradacc'...")
    plot_map_with_annotations(
        dem_data, extent, gdf,
        save_path="../glamos_run/plots/ALPS_gradacc_Scatter.pdf",
        country_paths=country_paths,
        value_column="gradacc",  # Column for Gradient Accumulation
        color_map="Blues",       # Colormap
        colorbar_label="Gradient Accumulation (m/a/km)"
    )

    plot_map_with_annotations(
        dem_data, extent, gdf,
        save_path="../glamos_run/plots/ALPS_sla_Scatter.pdf",
        country_paths=country_paths,
        value_column="sla",  # Column for ELA
        color_map="viridis",  # Colormap
        colorbar_label="End of summer snowline altitude (m)",
        num_dec=0,
    )

    plot_map_with_annotations(
        dem_data, extent, gdf,
        save_path="../glamos_run/plots/ALPS_difslaela_Scatter.pdf",
        country_paths=country_paths,
        value_column="sla_ela_diff",  # Column for ELA
        color_map="RdBu",  # Colormap
        colorbar_label="Difference between SLA and ELA (m)",
        num_dec=0,
    )


if __name__ == "__main__":
    main()