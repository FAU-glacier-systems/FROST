#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Monthly W5E5 climate of the calibration (climate_historical.nc of the OGGM
shop: one 0.5 degree cell, temperature at ref_hgt) against nearby weather
stations, 2000-2019.

Stations (monthly means / sums, open data): MeteoSwiss SwissMetNet
(data.geo.admin.ch, ch.meteoschweiz.ogd-smn: tre200m0, rre150m0) and
GeoSphere Austria (klima-v2-1m: tl_mittel, rr). Per glacier the closest high
station with temperature, and the closest with precipitation (summit
stations such as Jungfraujoch or Gornergrat have no rain gauge). The two
Italian glaciers (Careser, Vedretta Lunga) only have Sta. Maria (Val
Muestair, 21-27 km, other valley).

W5E5 temperature is moved to the station height with the -6.5 K/km of the TI
model (no_flow.TEMP_GRADIENT). Scores: temperature bias station - W5E5 (the
temp_bias a station would suggest), per season; r of the monthly anomalies
(seasonal cycle removed); precipitation ratio station / W5E5 (compare with
prcp_fac; gauges catch too little snow and stand below the glacier).

Run from the repository root (downloads into data/raw/stations once):
    python experiments/central_europe_TI/compare_stations.py [--dark]
Writes tables/stations.tsv, plots/stations.{pdf,png} and per glacier the annual
and seasonal series: plots/stations_{temperature,precipitation}_{annual,seasonal}.
"""

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
from frost.glacier_model import no_flow  # noqa: E402
from frost.visualization.plot_style import COLORS as PALETTE, DARK_THEME  # noqa: E402

RAW = os.path.join('data', 'raw', 'stations')
METEOSWISS = 'https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/{s}/ogd-smn_{s}_m.csv'
GEOSPHERE = ('https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-1m?parameters='
             'tl_mittel,rr&station_ids={s}&start=1999-01-01T00:00&end=2020-01-01T00:00&output_format=csv')
YEARS = ('2000-01-01', '2019-12-01')
# glacier: (temperature station, precipitation station); MeteoSwiss
# abbreviations, GeoSphere numbers
STATIONS = {
    'Grosser_Aletschgletscher': ('JUN', 'BLA'), 'Pasterze': ('82', '82'),
    'Glacier_de_Corbassière': ('GSB', 'GSB'), 'Rhonegletscher': ('GRH', 'GRH'),
    'Vadret_Pers': ('BEH', 'BEH'), 'Findelgletscher': ('GOR', 'ZER'),
    'Allalingletscher': ('GOR', 'ZER'), 'Kesselwandferner': ('184', '184'),
    'Glacier_du_Giétro': ('ATT', 'EVO'), 'Vernagtferner': ('184', '184'),
    'Hintereisferner': ('184', '184'), 'Glacier_de_la_Plaine_Morte': ('MVE', 'MVE'),
    'Ghiacciaio_di_Malevalle': ('66', '66'), 'Griesgletscher': ('GRH', 'ULR'),
    'Schwarzberggletscher': ('GOR', 'ZER'), 'Jamtalferner': ('28', '28'),
    'Zettalunitzkees': ('82', '15210'), 'Hallstätter_Gletscher': ('179', '179'),
    'Glacier_de_Tsanfleuron': ('DIA', 'CDM'), 'Silvrettagletscher': ('28', '28'),
    'Careser': ('SMM', 'SMM'), 'Claridenfirn': ('GUE', 'GUE'),
    'Vedretta_Lunga': ('SMM', 'SMM'), 'Adlergletscher': ('GOR', 'ZER'),
    'Ghiacciaio_del_Basòdino': ('ROE', 'ROE')}
SEASONS = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}


# ------------------------------------------------------------ stations

def _fetch(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(['curl', '-sf', '-m', '120', '-o', path, url], check=True)


def station_meta():
    """Name, height (m) and coordinates of all stations of both networks."""
    swiss = os.path.join(RAW, 'meteoswiss', 'ogd-smn_meta_stations.csv')
    austria = os.path.join(RAW, 'geosphere', 'klima-v2-1m_metadata.json')
    _fetch('https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_stations.csv', swiss)
    _fetch('https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-1m/metadata', austria)
    m = pd.read_csv(swiss, sep=';', encoding='latin1')
    meta = {r.station_abbr: (r.station_name, r.station_height_masl, r.station_coordinates_wgs84_lon,
                             r.station_coordinates_wgs84_lat) for r in m.itertuples()}
    with open(austria) as f:
        for s in json.load(f)['stations']:
            meta[str(s['id'])] = (s['name'], s['altitude'], s['lon'], s['lat'])
    return meta


def station_series(station):
    """Monthly temperature (degC) and precipitation (mm) 2000-2019."""
    if station.isdigit():
        path = os.path.join(RAW, 'geosphere', f'klima-v2-1m_{station}.csv')
        _fetch(GEOSPHERE.format(s=station), path)
        d = pd.read_csv(path)
        time = pd.to_datetime(d.time.str[:10])
        d = pd.DataFrame({'temp': d.tl_mittel.values, 'prcp': d.rr.values}, index=time)
    else:
        path = os.path.join(RAW, 'meteoswiss', f'ogd-smn_{station.lower()}_m.csv')
        _fetch(METEOSWISS.format(s=station.lower()), path)
        d = pd.read_csv(path, sep=';', encoding='latin1')
        time = pd.to_datetime(d.reference_timestamp, format='%d.%m.%Y %H:%M')
        d = pd.DataFrame({'temp': d.tre200m0.values, 'prcp': d.rre150m0.values}, index=time)
    return d.loc[YEARS[0]:YEARS[1]]


# ------------------------------------------------------------ comparison

def anomaly(series):
    return series - series.groupby(series.index.month).transform('mean')


def glacier_row(results_dir, rgi_id, name, meta):
    folder = os.path.join(results_dir, rgi_id)
    climate = xr.open_dataset(os.path.join(folder, 'climate_historical.nc'))
    w5e5 = climate[['temp', 'prcp']].to_dataframe().loc[YEARS[0]:YEARS[1]]
    t_station, p_station = STATIONS[name]
    t_name, t_z, t_lon, t_lat = meta[t_station]
    p_name, p_z = meta[p_station][:2]
    row = {'rgi_id': rgi_id, 'glacier': name.replace('_', ' '), 't_station': t_name,
           't_station_z': t_z, 'p_station': p_name, 'p_station_z': p_z,
           'ref_hgt': climate.attrs['ref_hgt']}
    obs = station_series(t_station)
    model = w5e5.temp + no_flow.TEMP_GRADIENT * (t_z - climate.attrs['ref_hgt'])
    both = pd.concat([obs.temp, model], axis=1, keys=['obs', 'model'], sort=True).dropna()
    diff = both.obs - both.model
    row['t_months'] = len(both)
    row['t_bias'] = diff.mean()
    for season, months in SEASONS.items():
        row[f't_bias_{season}'] = diff[diff.index.month.isin(months)].mean()
    row['t_r_anom'] = anomaly(both.obs).corr(anomaly(both.model))
    temp = both
    obs = station_series(p_station)
    both = pd.concat([obs.prcp, w5e5.prcp], axis=1, keys=['obs', 'model'], sort=True).dropna()
    row['p_months'] = len(both)
    row['p_ratio'] = both.obs.sum() / both.model.sum()
    winter = both.index.month.isin([10, 11, 12, 1, 2, 3, 4])
    row['p_ratio_winter'] = both.obs[winter].sum() / both.model[winter].sum()
    row['p_r_anom'] = anomaly(both.obs).corr(anomaly(both.model))
    prcp = both
    with open(os.path.join(folder, 'Preprocess', 'data', rgi_id, 'mb_calib.json')) as f:
        oggm = json.load(f)
    row['oggm_temp_bias'], row['oggm_prcp_fac'] = oggm['temp_bias'], oggm['prcp_fac']
    results = os.path.join(folder, 'calibration_results.json')
    row['frost_temp_bias'] = row['frost_prcp_fac'] = np.nan
    if os.path.exists(results):
        with open(results) as f:
            mean = json.load(f)['final_mean']
        row['frost_prcp_fac'], row['frost_temp_bias'] = mean[1], mean[2]
    return row, temp, prcp


def plot_series(table, series, variable, path):
    """One panel per glacier: annual values 2000-2019 (left figure) and the
    mean seasonal cycle (right figure) of the station and of W5E5 (temperature
    at the station height)."""
    temperature = variable == 'temp'
    unit = '°C' if temperature else 'mm'
    for kind in ['annual', 'seasonal']:
        n = len(table)
        cols = 5
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.5 * rows), squeeze=False)
        for ax, (_, r), both in zip(axes.flat, table.iterrows(), series):
            if kind == 'annual':
                full = both.groupby(both.index.year).count().obs == 12
                values = both.groupby(both.index.year).mean() if temperature else \
                    both.groupby(both.index.year).sum()
                values = values[full]
            else:
                values = both.groupby(both.index.month).mean()
            ax.plot(values.index, values.obs, '-o', color=plt.rcParams['text.color'], ms=2.5, lw=1, label='station')
            ax.plot(values.index, values.model, '-o', color=PALETTE['red'], ms=2.5, lw=1,
                    label='W5E5 (calibration)')
            station = r.t_station if temperature else r.p_station
            z = r.t_station_z if temperature else r.p_station_z
            score = f'bias {r.t_bias:+.1f} °C, r {r.t_r_anom:.2f}' if temperature else \
                f'ratio {r.p_ratio:.2f}, r {r.p_r_anom:.2f}'
            ax.set_title(f'{r.glacier}\n{station} ({z:.0f} m): {score}', fontsize=7.5)
            ax.tick_params(labelsize=7)
            if kind == 'seasonal':
                ax.set_xticks(range(1, 13), list('JFMAMJJASOND'))
            else:
                ax.set_xticks([2000, 2005, 2010, 2015, 2019])
                ax.set_xlim(1999.5, 2019.5)
        for ax in axes.flat[n:]:
            ax.axis('off')
        label = {('temp', 'annual'): 'annual mean temperature (°C)',
                 ('temp', 'seasonal'): 'monthly mean temperature 2000-2019 (°C)',
                 ('prcp', 'annual'): 'annual precipitation (mm)',
                 ('prcp', 'seasonal'): 'monthly precipitation 2000-2019 (mm)'}[variable, kind]
        fig.supylabel(label)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        for ext in ['pdf', 'png']:
            fig.savefig(f'{path}_{kind}.{ext}', dpi=130)
        plt.close(fig)


def plot(table, path):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    ax = axes[0]
    for col, marker, label in [('oggm_temp_bias', 'o', 'OGGM'), ('frost_temp_bias', 's', 'FROST posterior')]:
        ax.scatter(table.t_bias, table[col], marker=marker, s=24, label=label)
    lim = [min(table.t_bias.min(), table.oggm_temp_bias.min()) - 0.5,
           max(table.t_bias.max(), table.frost_temp_bias.max()) + 0.5]
    ax.plot(lim, lim, '--', color=plt.rcParams['text.color'], lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('station - W5E5 temperature (°C)')
    ax.set_ylabel('calibrated temp_bias (°C)')
    ax.legend(fontsize=8)
    ax.set_title('temperature offset: stations vs calibration', fontsize=10)

    ax = axes[1]
    months = list(SEASONS)
    for _, r in table.iterrows():
        ax.plot(months, [r[f't_bias_{s}'] for s in months], color='grey', lw=0.7, alpha=0.6)
    ax.plot(months, [table[f't_bias_{s}'].median() for s in months], '-o', color=plt.rcParams['text.color'], lw=2, label='median')
    ax.axhline(0, color=plt.rcParams['text.color'], lw=0.8)
    ax.set_ylabel('station - W5E5 temperature (°C)')
    ax.legend(fontsize=8)
    ax.set_title('temperature offset per season', fontsize=10)

    ax = axes[2]
    for col, marker, label in [('oggm_prcp_fac', 'o', 'OGGM'), ('frost_prcp_fac', 's', 'FROST posterior')]:
        ax.scatter(table.p_ratio, table[col], marker=marker, s=24, label=label)
    lim = [0, max(table.p_ratio.max(), table.oggm_prcp_fac.max()) + 0.3]
    ax.plot(lim, lim, '--', color=plt.rcParams['text.color'], lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('station / W5E5 precipitation (-)')
    ax.set_ylabel('calibrated prcp_fac (-)')
    ax.legend(fontsize=8)
    ax.set_title('precipitation: stations vs calibration', fontsize=10)

    ax = axes[3]
    for i, (col, label) in enumerate([('t_r_anom', 'temperature'), ('p_r_anom', 'precipitation')]):
        ax.scatter(np.full(len(table), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(table)),
                   table[col], s=18)
    ax.set_xticks([0, 1], ['temperature', 'precipitation'])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('r of the monthly anomalies')
    ax.set_title('year-to-year agreement', fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(f'{path}.{ext}', dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--results', default=os.path.join('data', 'results', 'central_europe_TI'))
    parser.add_argument('--out', default=HERE)
    parser.add_argument('--dark', action='store_true', help='black plots (*_dark)')
    args = parser.parse_args()
    suffix = ''
    if args.dark:
        plt.rcParams.update(DARK_THEME)
        suffix = '_dark'

    meta = station_meta()
    with open(os.path.join(HERE, 'glaciers.txt')) as f:
        glaciers = [tuple(line.split()[:2]) for line in f if line.strip()]
    rows, temps, prcps = zip(*[glacier_row(args.results, rgi_id, name, meta) for rgi_id, name in glaciers])
    table = pd.DataFrame(rows)
    os.makedirs(os.path.join(args.out, 'tables'), exist_ok=True)
    table.to_csv(os.path.join(args.out, 'tables', 'stations.tsv'), sep='\t', index=False,
                 float_format='%.3g')
    plot(table, os.path.join(args.out, 'plots', f'stations{suffix}'))
    plot_series(table, temps, 'temp', os.path.join(args.out, 'plots', f'stations_temperature{suffix}'))
    plot_series(table, prcps, 'prcp', os.path.join(args.out, 'plots', f'stations_precipitation{suffix}'))
    columns = ['glacier', 't_station', 't_bias', 't_bias_JJA', 'oggm_temp_bias', 'frost_temp_bias',
               't_r_anom', 'p_station', 'p_ratio', 'oggm_prcp_fac', 'frost_prcp_fac', 'p_r_anom']
    print(table[columns].to_string(index=False, float_format='%.2f'))
    print(table.drop(columns=['rgi_id', 'glacier', 't_station', 'p_station']).median().to_string(
        float_format='%.2f'))


if __name__ == '__main__':
    main()
