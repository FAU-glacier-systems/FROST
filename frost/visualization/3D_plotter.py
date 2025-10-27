import numpy as np
import copy
import plotly.graph_objects as go
import pyproj
from netCDF4 import Dataset
import argparse

property = 'thk'
color_scale = 'Blues'
min_property = 0
max_property = 600



def main(file_path):
    with Dataset(file_path, 'r') as ds:
        glacier_surface = ds.variables['usurf'][:]
        # dhdt = (ds.variables['usurf'][-1] - glacier_surface)/20
        #dhdt = ds.variables['dhdt'][:]
        dhdt = np.zeros_like(glacier_surface)[0]
        bedrock = ds.variables['topg'][:][0]
        #color_map = ds.variables[property][:]
        #color_map = (glacier_surface - 3100)*0.01


        x = ds.variables['x'][:]
        y = ds.variables['y'][:]

    visualise_3d(dhdt, glacier_surface, bedrock, x, y)
    return


def visualise_3d(dhdt, glacier_surface, bedrock, x, y):
    # choose property that is displayed on the glacier surface

    thickness = glacier_surface - bedrock
    lat_range = x
    lon_range = y
    # dhdt[thickness < 0.001] = None

    max_property_map = np.nanmax(dhdt)
    min_property_map = np.nanmin(dhdt)

    # make edges equal so that it looks like a volume
    max_bedrock = np.max(bedrock)
    min_bedrock = np.min(bedrock)
    bedrock_border = copy.copy(bedrock)
    bedrock_border[0, :] = min_bedrock
    bedrock_border[-1, :] = min_bedrock
    bedrock_border[:, 0] = min_bedrock
    bedrock_border[:, -1] = min_bedrock




    for i, year in enumerate(range(2000, 2101, 1)):
        # create 3D surface plots with property as surface color

        # glacier_surface += dhdt * 2
        # thickness = glacier_surface - bedrock

        glacier_surface_year = glacier_surface[i]
        thickness_year = thickness[i]

        # create time frames for slider
        glacier_surface_year[thickness_year < 1] = None
        surface_fig = go.Surface(
            z=glacier_surface_year,
            x=lat_range,
            y=lon_range,
            colorscale=color_scale,
            cmin=min_property,
            cmax=max_property,
            surfacecolor=thickness_year,
            showlegend=False,
            name="glacier surface",
            colorbar=dict(title= f"{year} Ice Thickness (m)",
                          titleside="top", thickness=25, orientation="h", y=0.75,
                          len=0.75,
                          titlefont=dict(size=40), tickfont=dict(size=40),
                          tickvals=[min_property, max_property],
                          # tickformat=".0f"

                          # This limits decimal places to 3
                          ),
            showscale=True,
        )

        # create 3D bedrock plots
        bedrock_fig = go.Surface(
            z=bedrock_border,
            x=lat_range,
            y=lon_range,
            colorscale='gray',
            opacity=1,
            showlegend=False,
            name="bedrock",
            cmax=max_bedrock,
            cmin=1000,
            colorbar=dict(title="Sliding Coefficient",
                          titleside="top", thickness=25, orientation="h", y=0.75,
                          len=0.75,
                          titlefont=dict(size=40), tickfont=dict(size=40),
                          tickvals=[min_property, max_property],
                          tickformat=".0f"),

            showscale=False,
        )

        # compute aspect ratio of the base
        resolution = int(lat_range[1] - lat_range[0])
        ratio_y = bedrock.shape[0] / bedrock.shape[1]
        ratio_z = (max_bedrock - min_bedrock) / (bedrock.shape[0] * resolution)
        ratio_z *= 2  # emphasize z-axis to make mountians look twice as steep

        # # transform angle[0-180] into values between [0, 1] for camera postion
        # radians = math.radians(camera_angle - 180)
        # camera_x = math.sin(-radians) - 1
        # camera_y = math.cos(-radians) - 1

        # transform angle[0-180] into values between [0, 1] for camera postion
        # theta = 2 * math.pi * camera_angle / 100

        # Define the UTM projection (UTM zone 32N)
        utm_proj = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')

        # Define the WGS84 projection
        wgs84_proj = pyproj.Proj(proj='latlong', datum='WGS84')

        # Example coordinate in UTM zone 32N (replace these values with your coordinates)
        utm_easting = lat_range  # example easting value
        utm_northing = lon_range  # example northing value

        # Reproject the coordinate
        lon_x, lat_x = pyproj.transform(utm_proj, wgs84_proj, utm_easting,
                                        np.ones_like(utm_easting) * utm_northing[0])
        lon_y, lat_y = pyproj.transform(utm_proj, wgs84_proj,
                                        np.ones_like(utm_northing) * utm_easting[0],
                                        utm_northing)

        # Output the WGS84 coordinate

        fig_dict = dict(
            data=[surface_fig, bedrock_fig],
            layout=dict(
                autosize=True,
                title={  # 'text': 'Kanderfirn                            '
                    #        '' + str(year),
                    'font': {'size': 50, 'color': 'white'},
                    'x': 0.1,
                    'y': 0.1},
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',  # Make outer background transparent
                plot_bgcolor='rgba(0,0,0,0)',
                width=1080,
                height=1080,
                font=dict(family="monospace", size=10),
                legend={"orientation": "h", "yanchor": "bottom", "xanchor": "left"},
                scene=dict(
                    zaxis=dict(showbackground=False, showticklabels=False, title="",
                               showgrid=False,  # Remove grid lines
                               zeroline=False,  # Remove axis zero line
                               showline=False,  # Remove axis line
                               ),
                    xaxis=dict(
                        showbackground=False,
                        showticklabels=True,
                        showgrid=False,  # Remove grid lines
                        zeroline=False,  # Remove axis zero line
                        showline=False,  # Remove axis line
                        visible=False,
                        range=[lat_range[0], lat_range[-1]],
                        tickvals=[ticks for ticks in lat_range[::42]],
                        ticktext=["%.2fE" % ticks for ticks in lon_x[::42]],

                        title="Longitude",

                    ),
                    yaxis=dict(
                        showbackground=False,
                        showticklabels=True,
                        showgrid=False,  # Remove grid lines
                        zeroline=False,  # Remove axis zero line
                        showline=False,  # Remove axis line
                        visible=False,
                        range=[lon_range[0], lon_range[-1]],
                        title="Latitude",
                        tickvals=[ticks for ticks in lon_range[::42]],
                        ticktext=["%.2fN" % ticks for ticks in lat_y[::42]],

                    ),
                ),
                scene_aspectratio=dict(x=1, y=ratio_y, z=ratio_z),
                scene_camera_eye=dict(x=0, y=-1.5, z=1.5),
                scene_camera_center=dict(x=0, y=0, z=0),

            ),
        )
        # create figure
        fig = go.Figure(fig_dict)
        # return
        # import dash
        # from dash import dcc, html
        # # Dash app layout
        # app = dash.Dash(__name__)
        # app.layout = html.Div([
        #     html.H1("3D Glacier Surface"),
        #     dcc.Graph(figure=fig)
        # ])
        #
        # app.run_server(debug=True)
        fig.write_image(f"Plots/glacier_surface{year}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run glacier calibration experiments.')

    # Add arguments for parameters
    parser.add_argument('--file_path', type=str,
                        help='Path to netcdf file with glacier properties')

    args = parser.parse_args()
    file_path = args.file_path
    main(file_path)
