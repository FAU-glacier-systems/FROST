#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
README figure: SMB profiles of FROST v2 (central_europe_combined_retrain)
and FROST v1 (the submitted Brief Communication run, central_europe_submit)
against the GLAMOS / WGMS elevation-band mass balance 2000-2019, one panel
per glacier. Same numbers as plots/profiles.png of evaluate.py.

Reads only calibration_results.json and the reference profiles (no
ensembles, fine on the login node). Run from the repository root:
    python experiments/central_europe_combined_retrain/readme_profiles.py
Writes assets/smb_profiles.png (light) and assets/smb_profiles_dark.png.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'experiments', 'central_europe_esmda')]
import evaluate  # noqa: E402
from frost.visualization import plot_style  # noqa: E402

RUN_V2 = os.path.join('data', 'results', 'central_europe_combined_retrain')
LABELS = {'new': 'FROST v2', 'submitted': 'FROST v1'}
LIGHT = dict(bg='#FFFFFF', fg='#111111', muted='#555555', grid='#D0D0D0',
             new='#1F6FB2', submitted='#8A8A8A')
DARK = dict(bg='#0A0A0A', fg='#EDEDED', muted='#C8C8C8', grid='#4A4A4A',
            new=plot_style.COLORS['blue'], submitted='#9A9A9A')



def short_name(name):
    """Glacier name without the generic prefix (Glacier de la Plaine Morte ->
    Plaine Morte), as in most glacier figures."""
    name = name.replace('_', ' ')
    for prefix in ('Glacier de la ', 'Glacier de ', 'Glacier du ', 'Ghiacciaio di ',
                   'Ghiacciaio del '):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def draw(profiles, curves, c, path):
    plt.rcdefaults()
    plt.rcParams.update({k: v for k, v in plot_style.DARK_THEME.items()
                         if k.startswith(('axes.spines', 'xtick.', 'ytick.major',
                                          'ytick.minor', 'axes.grid', 'axes.axisbelow',
                                          'grid.linestyle', 'grid.linewidth',
                                          'legend.frameon'))})
    plot_style.use_font()
    plt.rcParams.update({'figure.facecolor': c['bg'], 'axes.facecolor': c['bg'],
                         'savefig.facecolor': c['bg'], 'text.color': c['fg'],
                         'axes.labelcolor': c['fg'], 'xtick.color': c['muted'],
                         'ytick.color': c['muted'], 'grid.color': c['grid'],
                         'xtick.major.size': 0, 'xtick.minor.size': 0})
    glaciers = list(dict.fromkeys(profiles.glacier))
    cols = 5
    rows = int(np.ceil(len(glaciers) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.1 * rows + 1.4), squeeze=False)
    for ax, name in zip(axes.flat, glaciers):
        group = profiles[profiles.glacier == name]
        rgi_id = group.rgi_id.iloc[0]
        for run in ('submitted', 'new'):
            if (rgi_id, run) not in curves:
                continue
            z, mean, std, reference = curves[(rgi_id, run)]
            ax.fill_betweenx(z, mean - 2 * std, mean + 2 * std, color=c[run],
                             alpha=0.22, lw=0)
            ax.plot(mean, z, color=c[run], lw=2.6)
            rmse = group[group.run == run].rmse.iloc[0]
            ax.text(0.03, 0.97 - 0.13 * (run == 'new'), f'{LABELS[run]}  {rmse:.2f}',
                    transform=ax.transAxes, fontsize=12, va='top', color=c[run],
                    weight='bold' if run == 'new' else 'normal', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc=c['bg'], ec='none', alpha=0.8))
        ax.plot(reference, z, 'o', ms=5, color=c['fg'], zorder=4)
        ax.axvline(0, color=c['muted'], lw=0.7)
        ax.set_title(short_name(name), fontsize=14, color=c['fg'], loc='left')
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(labelsize=12)
    for ax in axes.flat[len(glaciers):]:
        ax.axis('off')
    handles = [plt.Line2D([], [], color=c["new"], lw=3,
                          label='FROST v2 (IGM 3.2, ES-MDA), posterior mean ± 2 std'),
               plt.Line2D([], [], color=c['submitted'], lw=2,
                          label='FROST v1 (Brief Communication)'),
               plt.Line2D([], [], color=c['fg'], marker='o', ls='', ms=4,
                          label='GLAMOS / WGMS elevation bands'),
               plt.Line2D([], [], color='none', label='number: RMSE (m w.e./yr)')]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=15,
               bbox_to_anchor=(0.5, 0.995), labelcolor=c['fg'])
    mean_rmse = profiles.groupby('run').rmse.mean()
    fig.supxlabel('Annual surface mass balance 2000-2019 (m w.e./yr). Mean RMSE: '
                  f'v2 {mean_rmse["new"]:.2f}, v1 {mean_rmse["submitted"]:.2f}',
                  fontsize=16, color=c["fg"])
    fig.supylabel('Elevation (m)', fontsize=16, color=c['fg'])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f'Wrote {path}')


def main():
    evaluate.RUNS['new'] = RUN_V2
    with open(os.path.join(evaluate.HERE, 'glaciers.txt')) as f:
        glaciers = [tuple(line.split()[:2]) for line in f if line.strip()]
    profiles, curves = evaluate.profile_rows(glaciers)
    both = profiles.groupby('rgi_id').run.nunique() == 2
    profiles = profiles[profiles.rgi_id.isin(both[both].index)]
    print(profiles.groupby('run')[['rmse', 'inside_2std_pct']].mean())
    draw(profiles, curves, LIGHT, os.path.join(ROOT, 'assets', 'smb_profiles.png'))
    draw(profiles, curves, DARK, os.path.join(ROOT, 'assets', 'smb_profiles_dark.png'))


if __name__ == '__main__':
    main()
