#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Evaluation of the TI calibration against the GLAMOS / WGMS mass balance per
elevation band and year, over the periods the measurements actually cover.

Every reference record is one band (elevation bin) in one balance year with
its dates: GLAMOS massbalance_observation_elevationbins.csv (floating dates
of the field surveys: start, end of winter, end of year; e.g. Rhone
2018-09-12 / 2019-04-18 / 2019-08-12), WGMS mass_balance_band.csv with the
dates of mass_balance.csv (mostly the fixed hydrological year 1 Oct - 30 Apr
- 30 Sep; a missing midseason date leaves the band without winter and
summer balance). The TI model of every posterior member (OGGM monthly TI as
in IGM's clim_1D_3D + smb_1D_3D, frost/glacier_model/no_flow.smb_ti) runs at
the band's mid elevation on the monthly climate of the glacier
(climate_historical.nc), each month weighted by its days inside the period.
So the winter and summer balance are compared too, not only the annual one.

Three model variants of the annual balance:
- TI: the measured period (as above)
- TI calendar year: January-December of the balance year, the year the
  calibration and the IGM runs use; the difference to TI is the timing
  error of comparing calendar-year model output with the measurements
- ELA: the constant SMB of the ELA calibration (central_europe_combined_retrain),
  scaled to the length of the period; no year-to-year change

Scores per glacier and variable (band-year records): bias, RMSE, r, share
of records inside +-2 std of the members; glacier-wide annual balance per
year (area-weighted over the measured bands): r between the years and RMSE;
mean profile (2000-2019 mean per band, bands with >= 10 years): RMSE, as
experiments/central_europe_esmda/evaluate.py.

Run from the repository root:
    python experiments/central_europe_TI/evaluate_bands.py [--dark]
Writes tables/band_records.tsv, tables/band_scores.tsv, tables/band_summary.tsv
and plots/band_*.{pdf,png} into experiments/central_europe_TI.
"""

import argparse
import calendar
import json
import os
import sys
from datetime import date, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'validation'))
import reference_profiles as rp  # noqa: E402
from frost.glacier_model import no_flow  # noqa: E402
from frost.visualization.plot_style import DARK_THEME  # noqa: E402

RUNS = {'TI': os.path.join('data', 'results', 'central_europe_TI'),
        'ELA': os.path.join('data', 'results', 'central_europe_combined_retrain')}
YEARS = (2000, 2019)  # balance years (end of the period); W5E5 ends in December 2019
MIN_YEARS = rp.MIN_YEARS
ICE_TO_WE = 0.91
VARIANTS = ['TI', 'TI calendar year', 'ELA']


# ------------------------------------------------------------ references

def _date(value):
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    return None if pd.isna(stamp) else stamp.date()


def glamos_records(name):
    df = pd.read_csv(rp.GLAMOS_BINS, delimiter=';', skiprows=6)[2:].copy()
    df.columns = [c.strip() for c in df.columns]
    df = df[df['glacier name'] == name]
    rows = []
    for _, r in df.iterrows():
        end = _date(r['end date of observation'])
        if end is None or not YEARS[0] <= end.year <= YEARS[1]:
            continue
        ba = pd.to_numeric(r['annual mass balance'], errors='coerce')
        if not ba > -20000:
            continue
        rows.append(dict(
            year=end.year, z_min=float(r['lower elevation of bin']),
            z_max=float(r['upper elevation of bin']),
            area=pd.to_numeric(r['area of elevation bin'], errors='coerce'),
            start=_date(r['start date of observation']),
            winter_end=_date(r['end date of winter observation']), end=end,
            ba=ba / 1000, bw=pd.to_numeric(r['winter mass balance'], errors='coerce') / 1000,
            bs=pd.to_numeric(r['summer mass balance'], errors='coerce') / 1000))
    return pd.DataFrame(rows)


def wgms_records(name):
    bands = pd.read_csv(rp.WGMS_BANDS, low_memory=False)
    bands = bands[(bands.glacier_name == name) & bands.year.between(*YEARS)]
    dates = pd.read_csv(rp.WGMS, low_memory=False)
    dates = dates[dates.glacier_name == name].set_index('year')
    rows = []
    for _, r in bands.iterrows():
        ba = pd.to_numeric(r.annual_balance, errors='coerce')
        if not np.isfinite(ba):
            continue
        d = dates.loc[r.year] if r.year in dates.index else None
        if isinstance(d, pd.DataFrame):
            d = d.iloc[0]
        # missing dates: the fixed hydrological year (WGMS time_system 'fixed')
        start = _date(d.begin_date) if d is not None else None
        end = _date(d.end_date) if d is not None else None
        winter_end = _date(d.midseason_date) if d is not None else None
        start = start or date(r.year - 1, 10, 1)
        end = end or date(r.year, 9, 30)
        rows.append(dict(
            year=int(r.year), z_min=float(r.lower_elevation), z_max=float(r.upper_elevation),
            area=pd.to_numeric(r.area, errors='coerce'), start=start, winter_end=winter_end,
            end=end, ba=ba, bw=pd.to_numeric(r.winter_balance, errors='coerce'),
            bs=pd.to_numeric(r.summer_balance, errors='coerce')))
    return pd.DataFrame(rows)


def reference_records(rgi_id, names):
    """Band-year records of the glacier, GLAMOS where it has the glacier,
    else WGMS (as reference_profiles.mass_balance_profiles)."""
    if rgi_id not in names.index:
        return None, None
    row = names.loc[rgi_id]
    records = glamos_records(row.glamos_name) if isinstance(row.glamos_name, str) else pd.DataFrame()
    if len(records):
        return records, 'GLAMOS'
    records = wgms_records(row.wgms_name) if isinstance(row.wgms_name, str) else pd.DataFrame()
    return (records, 'WGMS') if len(records) else (None, None)


# ----------------------------------------------------------------- model

def month_weights(start, end, yr_0, n_months):
    """Weight of every climate month (index from January yr_0) inside
    [start, end): days inside / days of the month; None if the period
    leaves the climate record."""
    weights = {}
    day = start
    while day < end:
        last = calendar.monthrange(day.year, day.month)[1]
        month_end = date(day.year, day.month, last) + timedelta(days=1)
        stop = min(month_end, end)
        index = (day.year - yr_0) * 12 + day.month - 1
        if index < 0 or index >= n_months:
            return None
        weights[index] = weights.get(index, 0) + (stop - day).days / last
        day = stop
    return weights


def ti_period(climate, z, members, start, end):
    """Balance (m w.e.) of [start, end) at the elevations z for every member
    (rows: melt_f, prcp_fac, temp_bias): the monthly terms of no_flow.smb_ti,
    weighted by month_weights."""
    temp = climate['temp'].ravel()
    prcp = climate['prcp'].ravel()
    weights = month_weights(start, end, climate['yr_0'], temp.size)
    if weights is None:
        return None
    melt_f, prcp_fac, temp_bias = (members[:, k, None] for k in range(3))
    lapse = no_flow.TEMP_GRADIENT * (z[None] - climate['ref_hgt'])
    balance = np.zeros((len(members), len(z)))
    for index, w in weights.items():
        t = temp[index] + temp_bias + lapse
        solid = np.clip((no_flow.TEMP_ALL_LIQUID - t)
                        / (no_flow.TEMP_ALL_LIQUID - no_flow.TEMP_ALL_SOLID), 0, 1)
        melt = melt_f * np.maximum(t - no_flow.TEMP_MELT, 0) * no_flow.DAYS_PER_YEAR / 12 / 1000
        balance += w * (prcp[index] * prcp_fac * solid / 1000 - melt)
    return balance


def ela_annual(members, z):
    """Annual SMB (m w.e./yr) of the ELA members (ela, abl_grad, acc_grad;
    gradients m ice per km), IGM's smb simple."""
    ela, abl, acc = (members[:, k, None] for k in range(3))
    smb = z[None] - ela
    smb = smb * np.where(smb < 0, abl, acc) / 1000
    return ICE_TO_WE * np.clip(smb, -100, 100)


def final_ensemble(results, rgi_id):
    path = os.path.join(results, rgi_id, 'calibration_results.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return np.array(json.load(f)['final_ensemble'], dtype=float)


def model_records(records, ti, ela, climate):
    """Model mean and std of the members for every record and variant."""
    out = records.copy()
    z = ((records.z_min + records.z_max) / 2).to_numpy()
    for column in ['TI_ba', 'TI_bw', 'TI_bs', 'TI calendar year_ba', 'ELA_ba']:
        out[column] = np.nan
        out[column + '_std'] = np.nan
    for i, r in enumerate(records.itertuples()):
        zi = z[i:i + 1]
        values = {}
        if ti is not None:
            values['TI_ba'] = ti_period(climate, zi, ti, r.start, r.end) if r.start else None
            if r.start and r.winter_end and r.start < r.winter_end < r.end:
                values['TI_bw'] = ti_period(climate, zi, ti, r.start, r.winter_end)
                values['TI_bs'] = ti_period(climate, zi, ti, r.winter_end, r.end)
            values['TI calendar year_ba'] = ti_period(climate, zi, ti, date(r.year, 1, 1),
                                                      date(r.year + 1, 1, 1))
        if ela is not None:
            days = (r.end - r.start).days if r.start else 365.25
            values['ELA_ba'] = ela_annual(ela, zi) * days / 365.25
        for column, v in values.items():
            if v is not None:
                out.loc[out.index[i], column] = v[:, 0].mean()
                out.loc[out.index[i], column + '_std'] = v[:, 0].std()
    return out


# ---------------------------------------------------------------- scores

def stats(obs, mod, std=None):
    ok = np.isfinite(obs) & np.isfinite(mod)
    if ok.sum() < 2:
        return dict(n=int(ok.sum()), bias=np.nan, rmse=np.nan, r=np.nan, inside_2std_pct=np.nan)
    diff = mod[ok] - obs[ok]
    inside = (100 * np.mean(np.abs(diff) <= 2 * std[ok]) if std is not None else np.nan)
    return dict(n=int(ok.sum()), bias=diff.mean(), rmse=np.sqrt(np.mean(diff ** 2)),
                r=np.corrcoef(obs[ok], mod[ok])[0, 1], inside_2std_pct=inside)


def glacier_wide(records, column):
    """Area-weighted balance per year over the bands with both values."""
    rows = []
    for year, group in records.groupby('year'):
        ok = np.isfinite(group[column]) & np.isfinite(group.ba)
        if not ok.any():
            continue
        g = group[ok]
        w = g.area.to_numpy(dtype=float)
        w = w if np.all(np.isfinite(w)) and w.sum() > 0 else np.ones(len(g))
        rows.append(dict(year=year, obs=np.average(g.ba, weights=w),
                         mod=np.average(g[column], weights=w)))
    return pd.DataFrame(rows)


def glacier_scores(records, rgi_id, name, source):
    rows = []
    for variant in VARIANTS:
        column = f'{variant}_ba'
        if records[column].notna().sum() == 0:
            continue
        base = dict(rgi_id=rgi_id, glacier=name, source=source, variant=variant)
        rows.append({**base, 'quantity': 'band annual',
                     **stats(records.ba.to_numpy(), records[column].to_numpy(),
                             records[column + '_std'].to_numpy())})
        wide = glacier_wide(records, column)
        rows.append({**base, 'quantity': 'glacier-wide annual',
                     **stats(wide.obs.to_numpy(), wide['mod'].to_numpy())})
        z = (records.z_min + records.z_max) / 2
        profile = records.assign(z=z).groupby('z').agg(
            obs=('ba', 'mean'), mod=(column, 'mean'), years=('year', 'nunique'))
        profile = profile[profile.years >= MIN_YEARS]
        rows.append({**base, 'quantity': 'mean profile',
                     **stats(profile.obs.to_numpy(), profile['mod'].to_numpy())})
    for season, obs in [('winter', 'bw'), ('summer', 'bs')]:
        column = f'TI_{obs}'
        if records[column].notna().sum() and records[obs].notna().sum():
            rows.append(dict(rgi_id=rgi_id, glacier=name, source=source, variant='TI',
                             quantity=f'band {season}',
                             **stats(records[obs].to_numpy(), records[column].to_numpy(),
                                     records[column + '_std'].to_numpy())))
    return rows


def summary_table(scores):
    """Per variant and quantity: mean over the glaciers of bias and RMSE,
    median r, mean share inside +-2 std."""
    return scores.groupby(['quantity', 'variant']).agg(
        glaciers=('glacier', 'nunique'), records=('n', 'sum'), bias=('bias', 'mean'),
        rmse=('rmse', 'mean'), r_median=('r', 'median'),
        inside_2std_pct=('inside_2std_pct', 'mean')).reset_index()


# ----------------------------------------------------------------- plots

COLORS = {'obs': None, 'TI': '#ff6b6b', 'TI calendar year': '#f4a261', 'ELA': '0.6'}


def plot_timeseries(records_by_glacier, path):
    """Glacier-wide annual balance per year: measured, TI (measured periods),
    TI calendar year, ELA."""
    names = list(records_by_glacier)
    cols = 5
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 2.6 * rows), squeeze=False)
    fg = plt.rcParams['text.color']
    for ax, name in zip(axes.flat, names):
        records = records_by_glacier[name]
        obs = glacier_wide(records, 'TI_ba')
        ax.plot(obs.year, obs.obs, 'o-', color=fg, ms=2.5, lw=0.8, label='measured')
        for variant in VARIANTS:
            wide = glacier_wide(records, f'{variant}_ba')
            if len(wide):
                ax.plot(wide.year, wide['mod'], color=COLORS[variant], lw=1,
                        ls='--' if variant == 'TI calendar year' else '-', label=variant)
        ax.axhline(0, color=fg, lw=0.4)
        ax.set_title(name.replace('_', ' ')[:28], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for ax in axes.flat[len(names):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=6, loc='lower left')
    fig.supylabel('Glacier-wide annual balance (m w.e.)')
    fig.supxlabel('Balance year (end of the measured period)')
    fig.tight_layout()
    save(fig, path)


def plot_scatter(records, path):
    """Band-year records: model against measured for the annual (TI, TI
    calendar year, ELA), winter and summer balance (TI)."""
    panels = [('ba', 'TI_ba', 'Annual, TI (measured periods)'),
              ('ba', 'TI calendar year_ba', 'Annual, TI calendar year'),
              ('ba', 'ELA_ba', 'Annual, ELA'),
              ('bw', 'TI_bw', 'Winter, TI'), ('bs', 'TI_bs', 'Summer, TI')]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4))
    fg = plt.rcParams['text.color']
    for ax, (obs, mod, title) in zip(axes, panels):
        ok = records[obs].notna() & records[mod].notna()
        x, y = records.loc[ok, obs], records.loc[ok, mod]
        ax.plot(x, y, '.', ms=1.5, alpha=0.4, color=COLORS['TI'] if 'TI' in mod else '0.6',
                rasterized=False)
        # a few bins far below the rest (e.g. -10 to -20 m w.e. at retreating
        # tongues) would squeeze the cloud: limits from the 0.5-99.5 %
        values = np.concatenate([x, y])
        lim = [np.percentile(values, 0.5) - 0.3, np.percentile(values, 99.5) + 0.3]
        outside = int(np.sum((x < lim[0]) | (x > lim[1]) | (y < lim[0]) | (y > lim[1])))
        ax.plot(lim, lim, '--', color=fg, lw=0.6)
        s = stats(x.to_numpy(), y.to_numpy())
        ax.set_title(f'{title}\nbias {s["bias"]:+.2f}, RMSE {s["rmse"]:.2f}, r {s["r"]:.2f}, '
                     f'n {s["n"]}' + (f' ({outside} off the axes)' if outside else ''), fontsize=9)
        ax.set_xlabel('Measured (m w.e.)', fontsize=8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('Model (m w.e.)', fontsize=8)
    fig.tight_layout()
    save(fig, path)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(f'{path}.{ext}', dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--ti', default=RUNS['TI'], help='results of the TI calibration')
    parser.add_argument('--ela', default=RUNS['ELA'], help='results of the ELA calibration')
    parser.add_argument('--glaciers', default=os.path.join(HERE, 'glaciers.txt'))
    parser.add_argument('--out', default=HERE)
    parser.add_argument('--dark', action='store_true', help='black plots (*_dark)')
    args = parser.parse_args()
    suffix = ''
    if args.dark:
        plt.rcParams.update(DARK_THEME)
        suffix = '_dark'

    names = rp._names()
    with open(args.glaciers) as f:
        glaciers = [tuple(line.split()[:2]) for line in f if line.strip()]
    all_records, scores, by_glacier = [], [], {}
    for rgi_id, name in glaciers:
        records, source = reference_records(rgi_id, names)
        ti = final_ensemble(args.ti, rgi_id)
        if records is None or ti is None:
            print(f'{name}: {"no reference" if records is None else "no TI result"}, skipped')
            continue
        climate = no_flow.load_climate(os.path.join(args.ti, rgi_id, 'climate_historical.nc'))
        records = model_records(records, ti, final_ensemble(args.ela, rgi_id), climate)
        records.insert(0, 'glacier', name)
        records.insert(0, 'rgi_id', rgi_id)
        records['source'] = source
        all_records.append(records)
        by_glacier[name] = records
        scores += glacier_scores(records, rgi_id, name, source)
        print(f'{name}: {source}, {len(records)} band-years, {records.year.nunique()} years')

    records = pd.concat(all_records, ignore_index=True)
    scores = pd.DataFrame(scores)
    summary = summary_table(scores)
    os.makedirs(os.path.join(args.out, 'tables'), exist_ok=True)
    records.to_csv(os.path.join(args.out, 'tables', 'band_records.tsv'), sep='\t', index=False,
                   float_format='%.4g')
    scores.to_csv(os.path.join(args.out, 'tables', 'band_scores.tsv'), sep='\t', index=False,
                  float_format='%.3g')
    summary.to_csv(os.path.join(args.out, 'tables', 'band_summary.tsv'), sep='\t', index=False,
                   float_format='%.3g')
    plot_timeseries(by_glacier, os.path.join(args.out, 'plots', f'band_timeseries{suffix}'))
    plot_scatter(records, os.path.join(args.out, 'plots', f'band_scatter{suffix}'))
    print(summary.to_string(index=False, float_format='%.2f'))


if __name__ == '__main__':
    main()
