from setuptools import setup, find_packages

setup(
    name="frost",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
    # Pinned to the versions running in frost_env, the environment actually
    # used for calibration experiments - keep these in sync with each other.
    install_requires=[
        "numpy==1.26.4",
        "scipy==1.15.3",
        "xarray==2025.4.0",
        "matplotlib==3.10.7",
        "plotly==6.3.0",
        "dash==2.14.2",
        "rioxarray==0.18.1",
        "rasterio==1.4.3",
        "pyproj==3.6.1",
        "netCDF4==1.6.0",
        "PyYAML==6.0.3",
        "utm==0.8.1",
        "gstools==1.7.0",
        "pyvista==0.46.4",
    ],
    extras_require={
        "dev": ["pytest==9.1.1"],
    },
)
