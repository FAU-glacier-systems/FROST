#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Two things of the GLAMOS / WGMS references that the ELA and gradients of
tables/combined_ela_gradients.csv hide:

1. Censored ELA: in years without accumulation area GLAMOS reports the ELA
   at the glacier top (AAR 0-2 %, e.g. Plaine Morte 2915-2945 m with its top
   at 2952 m) and WGMS flags it with ela_position 'above'. The 2000-2019
   mean of such a glacier is a lower bound of its ELA, and the GLAMOS
   gradients are fitted through the capped ELAs. ela_censored_share() gives
   the share of those years per glacier (Plaine Morte 0.8, Careser 0.75,
   Tsanfleuron 0.7, ...), from the source of the reference ELA (GLAMOS
   before WGMS, as the combined table).
2. Mass-balance profile: the mean annual mass balance per elevation bin
   2000-2019 (GLAMOS massbalance_observation_elevationbins.csv, WGMS
   mass_balance_band.csv), m w.e./yr, to compare modelled SMB profiles
   directly instead of ELA and gradients. Bins observed in fewer than
   MIN_YEARS years are dropped (bins at the retreating tongue or the
   lowering top). Both are themselves stake measurements extrapolated by a
   model.

    from reference_profiles import ela_censored_share, mass_balance_profiles
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from frost.paths import DATA_RAW

HERE = os.path.dirname(os.path.abspath(__file__))
LOOKUP = os.path.join(HERE, 'tables', 'RGI6-7.csv')
COMBINED = os.path.join(HERE, 'tables', 'combined_ela_gradients.csv')
GLAMOS = DATA_RAW / 'glamos' / 'massbalance_observation.csv'
GLAMOS_BINS = DATA_RAW / 'glamos' / 'massbalance_observation_elevationbins.csv'
GLAMOS_RGI = DATA_RAW / 'glamos' / 'GLAMOS_RGI.csv'
WGMS = DATA_RAW / 'DOI-WGMS-FoG-2025-02b' / 'data' / 'mass_balance.csv'
WGMS_BANDS = DATA_RAW / 'DOI-WGMS-FoG-2025-02b' / 'data' / 'mass_balance_band.csv'
YEARS = (2000, 2019)
MIN_YEARS = 10
AAR_NO_ACCUMULATION = 2  # % (GLAMOS)


def _glamos(path):
    df = pd.read_csv(path, delimiter=';', skiprows=6)[2:].copy()
    df.columns = [c.strip() for c in df.columns]
    df['year'] = pd.to_datetime(df['end date of observation'], errors='coerce').dt.year
    for column in df.columns[5:]:
        if column != 'observer':
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df[(df.year >= YEARS[0]) & (df.year <= YEARS[1])]


def _wgms(path):
    df = pd.read_csv(path, low_memory=False)
    return df[(df.year >= YEARS[0]) & (df.year <= YEARS[1])]


def _names():
    """rgi_id -> (glamos_name, wgms_name, ela_source)."""
    lookup = pd.read_csv(LOOKUP).set_index('rgi_id')
    # GLAMOS names as extract_glamos_ela_gradients.py (RGI6-7.csv has
    # e.g. 'Plaine Morte' for 'Glacier de la Plaine Morte')
    glamos = pd.read_csv(GLAMOS_RGI).set_index('rgi_id')['glamos_name']
    lookup['glamos_name'] = glamos.reindex(lookup.index).combine_first(lookup.glamos_name)
    source = pd.read_csv(COMBINED).set_index('rgi_id')['ela_source']
    return lookup.join(source, how='left')


def ela_censored_share(rgi_ids):
    """Share of the reference years (with an ELA) whose ELA is at or above
    the glacier top, per rgi_id (NaN without a reference ELA)."""
    names = _names()
    glamos = _glamos(GLAMOS).dropna(subset=['equilibrium line altitude'])
    wgms = _wgms(WGMS)
    wgms = wgms[pd.to_numeric(wgms.ela, errors='coerce').notna()]
    share = {}
    for rgi_id in rgi_ids:
        share[rgi_id] = np.nan
        if rgi_id not in names.index:
            continue
        row = names.loc[rgi_id]
        if row.ela_source == 'GLAMOS':
            years = glamos[glamos['glacier name'] == row.glamos_name]
            if len(years):
                share[rgi_id] = float((years['accumulation area ratio']
                                       <= AAR_NO_ACCUMULATION).mean())
        elif row.ela_source == 'WGMS':
            years = wgms[wgms.glacier_name == row.wgms_name]
            if len(years):
                share[rgi_id] = float((years.ela_position == 'above').mean())
    return share


def mass_balance_profiles(rgi_ids):
    """Mean annual mass balance per elevation bin 2000-2019 (m w.e./yr),
    per rgi_id: DataFrame with elevation (bin centre), mb, years, source;
    GLAMOS where it has the glacier, else WGMS; missing glaciers left out."""
    names = _names()
    glamos = _glamos(GLAMOS_BINS)
    glamos = glamos[glamos['annual mass balance'] > -20000]
    wgms = _wgms(WGMS_BANDS)
    wgms = wgms[pd.to_numeric(wgms.annual_balance, errors='coerce').notna()]
    profiles = {}
    for rgi_id in rgi_ids:
        if rgi_id not in names.index:
            continue
        row = names.loc[rgi_id]
        bins = glamos[glamos['glacier name'] == row.glamos_name]
        if len(bins):
            table = pd.DataFrame({
                'elevation': (bins['lower elevation of bin'] + bins['upper elevation of bin']) / 2,
                'mb': bins['annual mass balance'] / 1000, 'year': bins.year})
            source = 'GLAMOS'
        else:
            bins = wgms[wgms.glacier_name == row.wgms_name]
            if not len(bins):
                continue
            table = pd.DataFrame({
                'elevation': (bins.lower_elevation + bins.upper_elevation) / 2,
                'mb': pd.to_numeric(bins.annual_balance), 'year': bins.year})
            source = 'WGMS'
        profile = table.groupby('elevation').agg(mb=('mb', 'mean'), years=('year', 'nunique'))
        profile = profile[profile.years >= MIN_YEARS].reset_index()
        if len(profile):
            profile['source'] = source
            profiles[rgi_id] = profile
    return profiles


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('glaciers', help='glaciers.txt (RGI id and name per line)')
    args = parser.parse_args()
    with open(args.glaciers) as f:
        glaciers = [tuple(line.split()[:2]) for line in f if line.strip()]
    ids = [g[0] for g in glaciers]
    share, profiles = ela_censored_share(ids), mass_balance_profiles(ids)
    for rgi_id, name in glaciers:
        p = profiles.get(rgi_id)
        print(f'{name[:26]:26s} censored {share[rgi_id]:5.2f}  profile '
              + (f'{p.source.iloc[0]} {len(p)} bins {p.elevation.min():.0f}-{p.elevation.max():.0f} m'
                 if p is not None else '-'))
