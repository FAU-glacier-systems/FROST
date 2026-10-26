#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
import os
import shutil
import yaml
import numpy as np
from netCDF4 import Dataset
import netCDF4

from frost.preprocess.thickness_validation import validate_thickness


def emulator_path(rgi_id_dir):
    """Iceflow network saved by the inversion, to be used by forward runs."""
    return os.path.join(rgi_id_dir, 'Preprocess', 'outputs', 'emulator.keras')


def thickness_at_2000(input_file, product_year=None):
    """Shift the thickness product of input.nc (thk of the download, see
    thickness_products.py) from its reference year back to 2000 with the
    observed dh/dt: thk_2000 = thk - dhdt * (product_year - 2000). The year
    is the thk_source_year of input.nc; product_year is needed only for
    downloads without it (Millan, before 2026-10-22) and must match it. Gaps of
    the product inside the ice mask take the nearest value. Writes the result
    to thk (start of the inversion) and thkprior (reference for a thickness
    misfit); the unshifted product is kept as thk_product."""
    from scipy.ndimage import distance_transform_edt

    with Dataset(input_file, 'r+') as ds:
        source_year = ds.getncattr('thk_source_year') \
            if 'thk_source_year' in ds.ncattrs() else None
        if source_year is None and product_year is None:
            raise ValueError(f'{input_file} has no thk_source_year; set '
                             'frost: thk_product_year in params_inversion.yaml')
        if source_year is not None and product_year is not None \
                and int(source_year) != int(product_year):
            raise ValueError(
                f'thk_product_year {product_year} of params_inversion.yaml, but '
                f'the thickness of {input_file} is {ds.getncattr("thk_source")} '
                f'of {source_year}')
        product_year = int(source_year if source_year is not None else product_year)
        if 'thk_product' not in ds.variables:
            product = ds.createVariable('thk_product', 'f4', ('y', 'x'))
            product[:] = ds['thk'][:]
        thk = np.array(ds['thk_product'][:].filled(np.nan)
                       if np.ma.isMaskedArray(ds['thk_product'][:])
                       else ds['thk_product'][:], dtype=float)
        icemask = np.array(ds['icemask'][:]) > 0.5
        dhdt = np.array(ds['dhdt'][:], dtype=float) if 'dhdt' in ds.variables \
            else np.zeros_like(thk)

        valid = np.isfinite(thk) & (thk > 0)
        _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
        thk = np.where(icemask & ~valid, thk[iy, ix], thk)
        dhdt = np.where(np.isfinite(dhdt), dhdt, 0.0)
        thk_2000 = np.where(icemask,
                            np.maximum(thk - dhdt * (product_year - 2000), 0.0),
                            0.0)

        ds['thk'][:] = thk_2000
        if 'thkprior' not in ds.variables:
            ds.createVariable('thkprior', 'f4', ('y', 'x'))
        ds['thkprior'][:] = np.where(icemask, thk_2000, np.nan)
        ds['thkprior'].setncattr(
            'description', f'thickness product of {product_year} shifted to '
                           '2000 with the observed dh/dt')


def velocity_year(input_file):
    """Year of the surface velocities of input.nc (velocity_source_year): an
    annual mosaic's year, the middle of a composite or period
    ('2014-2022 composite' -> 2018, Millan '2017-2018' -> 2017.5)."""
    import re
    with Dataset(input_file, 'r') as ds:
        label = str(ds.getncattr('velocity_source_year'))
    years = [int(year) for year in re.findall(r'\d{4}', label)]
    if not years:
        raise ValueError(f'No year in velocity_source_year "{label}" of '
                         f'{input_file}')
    return (min(years) + max(years)) / 2


def geometry_at(input_file, year):
    """Move the start geometry of input.nc (surface of 2000, thickness of
    2000 from thickness_at_2000) to 'year' with the observed dh/dt, bed
    unchanged: thk = max(thk_2000 + dhdt * (year - 2000), 0) on the ice,
    usurf = bed + thk. The inversion then fits tau_ref to velocities on the
    geometry of their own epoch instead of the thicker 2000 geometry (on
    Aletsch the 2000 geometry gave a tau_ref too high and a starving tongue).
    Keeps the 2000 fields as usurf_2000 and thk_2000 for restore_2000."""
    with Dataset(input_file, 'r+') as ds:
        # usurf is overwritten below, so keep the 2000 surface of the first
        # call; thk was just reset to 2000 by thickness_at_2000
        if 'usurf_2000' not in ds.variables:
            ds.createVariable('usurf_2000', 'f4', ('y', 'x'))
            ds['usurf_2000'][:] = ds['usurf'][:]
        if 'thk_2000' not in ds.variables:
            ds.createVariable('thk_2000', 'f4', ('y', 'x'))
        ds['thk_2000'][:] = ds['thk'][:]
        usurf_2000 = np.array(ds['usurf_2000'][:], dtype=float)
        thk_2000 = np.array(ds['thk_2000'][:], dtype=float)
        icemask = np.array(ds['icemask'][:]) > 0.5
        dhdt = np.array(ds['dhdt'][:], dtype=float)
        dhdt = np.where(np.isfinite(dhdt), dhdt, 0.0)
        cell_area = abs(float(ds['x'][1] - ds['x'][0])) ** 2

        bed = usurf_2000 - thk_2000
        thk = np.where(icemask,
                       np.maximum(thk_2000 + dhdt * (year - 2000), 0.0), 0.0)
        ds['thk'][:] = thk
        ds['usurf'][:] = bed + thk
        ds['thkprior'][:] = np.where(icemask, thk, np.nan)
        ds['thkprior'].setncattr(
            'description', f'thickness of 2000 moved to {year:g} with the '
                           'observed dh/dt (inversion geometry)')
    print(f'Inversion geometry of {year:g}: ice volume '
          f'{np.sum(thk_2000) * cell_area * 1e-9:.3f} -> '
          f'{np.sum(thk) * cell_area * 1e-9:.3f} km3, mean thickness change '
          f'{np.mean((thk - thk_2000)[icemask]):.1f} m')


def restore_2000(output_file, input_file, year):
    """Put the 2000 surface and thickness back into the inversion result,
    which the forward runs start from (bed and tau_ref unchanged); the
    inversion geometry is kept as usurf_inversion and thk_inversion."""
    with Dataset(input_file, 'r') as src, Dataset(output_file, 'r+') as out:
        for name in ('usurf', 'thk'):
            kept = out.createVariable(f'{name}_inversion', 'f4', ('y', 'x'))
            kept[:] = out[name][:]
            kept.setncattr('description', f'{name} of the inversion ({year:g})')
            out[name][:] = src[f'{name}_2000'][:]
        out['topg'][:] = out['usurf'][:] - out['thk'][:]
        out.setncattr('inversion_year', float(year))


def main(rgi_id_dir, params_inversion_path, min_velocity_p99=0.0):
    """
    Generates params.json for IGM inversion and runs igm_run.

    Args:
        rgi_id (str) - Glacier RGI ID
        min_velocity_p99 (float) - velocity observations are used only if
                                   their 99th percentile on the ice exceeds
                                   this (m/yr); 0 (default since 2026-10-22):
                                   whenever there are any. 10 dropped the slow
                                   glaciers with ITS_LIVE (p99 1-8 m/yr).
                                   params_inversion.yaml can set it as
                                   frost: min_velocity_p99. Without usable
                                   velocities and without another misfit
                                   term, tau_ref keeps its default (an
                                   inversion of 0 iterations): ITS_LIVE on
                                   Jamtal (p99 1.8 m/yr, noise) made the ice
                                   immobile and the forward runs unstable.
    """

    # Check if velocity observations are available

    # Define input and output file names
    input_file = os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input.nc')

    with open(params_inversion_path, 'r') as file:
        min_velocity_p99 = float((yaml.safe_load(file).get('frost') or {})
                                 .get('min_velocity_p99', min_velocity_p99))

    flag_velsurfobs = False
    velocity_p99 = np.nan
    with Dataset(input_file, 'r') as input_dataset:
        icemask = np.array(input_dataset['icemask'][:]) > 0.5

        #flag_velsurfobs = "uvelsurfobs" in input_dataset.variables and "vvelsurfobs" in input_dataset.variables
        epsg = input_dataset.getncattr('epsg')
        pyproj_srs = input_dataset.getncattr('pyproj_srs')

        for name in ['uvelsurfobs']:
            # Read variable
            variable = input_dataset.variables[name]
            try:
                fillvalue = variable.get_fill_value()
            except:
                fillvalue = netCDF4.default_fillvals['f8']

            # Set values to zero where they are NaN
            original_data_v1 = np.array(variable[:])
            original_data_v2 = np.array(input_dataset.variables['vvelsurfobs'][:])

            # Remove zero values
            modified_data_v1 = np.where(original_data_v1 == 0,
                                        np.nan, original_data_v1)
            modified_data_v2 = np.where(original_data_v2 == 0,
                                        np.nan, original_data_v2)

            # Remove fillvalues
            modified_data_v1 = np.where(np.abs(modified_data_v1) == fillvalue,
                                        np.nan, modified_data_v1)
            modified_data_v2 = np.where(np.abs(modified_data_v2) == fillvalue,
                                        np.nan, modified_data_v2)

            # Compute magnitude of vecotr field
            modified_data = np.sqrt(modified_data_v1 ** 2 + modified_data_v2 ** 2)

            # Check if velocity input is meaningful (nonzero, not NaN, 99th
            # percentile on the ice above min_velocity_p99)
            on_ice = modified_data[icemask]
            if np.isfinite(on_ice).any() and np.nansum(on_ice) != 0:
                velocity_p99 = float(np.nanpercentile(np.abs(on_ice), 99))
                if velocity_p99 > min_velocity_p99:
                    flag_velsurfobs = True

    # Load base parameters from params_inversion.yaml
    with open(params_inversion_path, 'r') as file:
        inv_params = yaml.safe_load(file)

    # FROST options, not passed to IGM
    frost_options = inv_params.pop('frost', None) or {}
    if frost_options.get('thk_start') == 'product_2000':
        thickness_at_2000(input_file, frost_options.get('thk_product_year'))

    # Inversion on the geometry of another year (inversion_year: 'velocity'
    # for the year of the velocity product, or a year); the forward runs
    # still start from 2000 (restore_2000 below)
    inversion_year = frost_options.get('inversion_year')
    if inversion_year is not None:
        if frost_options.get('thk_start') != 'product_2000':
            raise ValueError('inversion_year needs thk_start: product_2000 '
                             '(fixed thickness of 2000)')
        inversion_year = velocity_year(input_file) \
            if inversion_year == 'velocity' else float(inversion_year)
        geometry_at(input_file, inversion_year)
    else:
        # a previous run with inversion_year moved usurf; take 2000 again
        with Dataset(input_file, 'r+') as ds:
            if 'usurf_2000' in ds.variables:
                ds['usurf'][:] = ds['usurf_2000'][:]

    # frost: invert: false skips the inversion: 0 iterations, tau_ref keeps
    # its start value everywhere (params_inversion.yaml: iceflow sliding
    # tau_ref); the forward runs start from the 2000 geometry as usual
    velocity_fit = 'yes'
    if not frost_options.get('invert', True):
        inv_params['assimilations']['field_inversion'].setdefault(
            'optimization', {})['nbitmax'] = 0
        velocity_fit = 'skipped'
        print(f'Inversion skipped for {rgi_id_dir} (frost: invert: false): '
              'uniform tau_ref')
    # Drop the velsurf misfit term if no usable velocity observations are
    # available; if it is the only term, invert nothing (tau_ref and the
    # other variables keep their start values)
    elif not flag_velsurfobs:
        field_inversion = inv_params['assimilations']['field_inversion']
        misfit = [term for term in field_inversion['objective']['misfit']
                  if term['name'] != 'velsurf']
        if misfit:
            field_inversion['objective']['misfit'] = misfit
            velocity_fit = 'dropped'
        else:
            field_inversion.setdefault('optimization', {})['nbitmax'] = 0
            velocity_fit = 'skipped'
        print(f'No usable surface velocities for {rgi_id_dir} (99th '
              f'percentile on the ice {velocity_p99:.1f} m/yr, needs > '
              f'{min_velocity_p99:g}): velocity misfit {velocity_fit}')

    # Fixed hydra run dir so reruns overwrite instead of creating
    # outputs/<date>/<time>; params_inversion.yaml may set its own
    run_dir = inv_params.setdefault('hydra', {}).setdefault('run', {}) \
        .setdefault('dir', os.path.join('outputs', 'inversion'))

    # Prepare inversion directory
    preprocess_dir = os.path.join(rgi_id_dir, 'Preprocess')
    # shutil.rmtree(inversion_dir, ignore_errors=True)
    exp_dir = os.path.join(preprocess_dir, 'experiment')
    os.makedirs(exp_dir, exist_ok=True)

    # Save the iceflow network the inversion used (the pretrained emulator,
    # fine-tuned on this glacier at initialisation), so the forward runs
    # continue with the same ice flow instead of the raw pretrained one.
    # IGM loads the save_emulator user module from Preprocess/user; it has
    # to be both a processes key and in the defaults list (module order).
    frost_user_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'igm_user')
    shutil.copytree(
        os.path.join(frost_user_dir, 'code', 'processes', 'save_emulator'),
        os.path.join(preprocess_dir, 'user', 'code', 'processes', 'save_emulator'),
        dirs_exist_ok=True)
    os.makedirs(os.path.join(preprocess_dir, 'user', 'conf', 'processes'),
                exist_ok=True)
    shutil.copy(
        os.path.join(frost_user_dir, 'conf', 'processes', 'save_emulator.yaml'),
        os.path.join(preprocess_dir, 'user', 'conf', 'processes'))
    inv_params['processes']['save_emulator'] = {
        'path': os.path.abspath(emulator_path(rgi_id_dir))}
    for default in inv_params['defaults']:
        if 'override /processes' in default \
                and 'save_emulator' not in default['override /processes']:
            default['override /processes'].append('save_emulator')
    # Change to inversion directory and save params
    original_dir = os.getcwd()
    os.chdir(preprocess_dir)
    print(preprocess_dir)
    # Write to YAML file
    with open(os.path.join('experiment', 'params_inversion.yaml'), 'w') as file:
        file.write("# @package _global_\n")
        yaml.dump(inv_params, file, sort_keys=False)

    # Run IGM inversion
    # Run the igm_run command
    import sys
    from igm.igm_run import main as igm_main
    sys.argv = ["igm_run", "+experiment=params_inversion"]
    igm_main()
    # subprocess.run(["igm_run", "+experiment=params"], check=True)

    # TODO remove unnecessary files
    latest = run_dir

    # field_inversion (IGM >= 3.2) writes its result to optimize.nc inside the
    # hydra run dir, not to a configurable save_result_in_ncdf path like the
    # old data_assimilation strategy did. Copy it to outputs/output.nc so the
    # rest of FROST (create_observation.py, analyze_inversion.py,
    # ensemble_kalman_filter.py, ...) can keep expecting that fixed path.
    # optimize.nc stores snapshots along an 'iterations' dimension; keep only
    # the last one so output.nc holds plain 2D (y, x) fields.
    os.makedirs('outputs', exist_ok=True)
    with Dataset(os.path.join(latest, 'optimize.nc'), 'r') as src, \
            Dataset(os.path.join('outputs', 'output.nc'), 'w') as output:
        output.setncatts(src.__dict__)
        for name, dim in src.dimensions.items():
            if name != 'iterations':
                output.createDimension(name, None if dim.isunlimited() else len(dim))
        for name, var in src.variables.items():
            if name == 'iterations':
                continue
            dims = tuple(d for d in var.dimensions if d != 'iterations')
            fill = var.getncattr('_FillValue') if '_FillValue' in var.ncattrs() else None
            out_var = output.createVariable(name, var.dtype, dims, fill_value=fill)
            out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()
                               if k != '_FillValue'})
            out_var[:] = var[-1] if 'iterations' in var.dimensions else var[:]
        # field_inversion changes thk under a fixed usurf but saves the
        # original topg. The forward runs rebuild thk as usurf - topg, so
        # store the bed that belongs to the inverted thickness.
        output['topg'][:] = output['usurf'][:] - output['thk'][:]
        output.setncattr('epsg', epsg)
        output.setncattr('pyproj_srs', pyproj_srs)
        output.setncattr('velocity_fit', velocity_fit)
        output.setncattr('velocity_p99_ice', velocity_p99)

    os.chdir(original_dir)

    if inversion_year is not None:
        restore_2000(os.path.join(preprocess_dir, 'outputs', 'output.nc'),
                     input_file, inversion_year)

    # Independent check against thickness observations, where there are any
    validate_thickness(rgi_id_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates params.json for IGM inversion and runs igm_run.')
    parser.add_argument('--rgi_id_dir', type=str,
                        default="../../results/test_default/RGI2000"
                                "-v7.0-G-11-01706/",
                        help='Path to Glacier dir with OGGM_shop output')
    parser.add_argument('--params_inversion_path', type=str,
                        default='../../experiments/test_default/params_inversion.yaml')
    args = parser.parse_args()
    main(args.rgi_id_dir, args.params_inversion_path)
