#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""Two-page PDF summary of the FROST ContinuIX submission.

Glacier means of every submitted file come from the check report
(check_GROUP_FAU2.txt, written by package_and_check.sh on a compute node);
the EXP01 model dh/dt and the G03 profile from summary_data.json.
Writes ContinuIX_FROST_summary.pdf and page1/2.png to
data/results/continuix/summary/. Light, runs on the login node.

Run from the repository root:
    python experiments/continuix/make_summary.py
"""
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

RESULTS = os.path.join('data', 'results', 'continuix')
HERE = os.path.join(RESULTS, 'summary')
STATUS_DATE = '8 October 2026'


def load():
    """Glacier means of every file from the check report, the EXP01 model
    dh/dt and the G03 profile from summary_data.json, GPU hours from the log."""
    global D, M, F, GPU_HOURS
    D = json.load(open(os.path.join(HERE, 'summary_data.json')))
    M = D['model']
    F = {}
    for line in open(os.path.join(RESULTS, 'check_GROUP_FAU2.txt')):
        m = re.match(r'(EXP\d\d)\s+([GS]\d\d)\s+\d+' + r'\s+(\S+)' * 5, line)
        if m:
            e, g, smb, unct, fdiv, obs, res = m.groups()
            F[f'{e}_{g}'] = dict(smb=float(smb), unct=float(unct), fdiv=float(fdiv),
                                 obs=float(obs))
    log = open(os.path.join(RESULTS, 'GROUP_FAU2', 'log_GROUP_FAU2.txt')).read()
    GPU_HOURS = re.search(r'([\d.]+) GPU hours in total', log).group(1)


INK, INK2, MUTED, GRID = '#1f1f1f', '#5f5e5a', '#8a8984', '#e3e2dc'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'text.color': INK,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'axes.linewidth': 0.6,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False,
    'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.titlecolor': INK})
A4 = (8.27, 11.69)


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def text_block(fig, x, y, title, lines, width=0.86, size=7.6):
    fig.text(x, y, title, fontsize=9.5, weight='bold')
    fig.text(x, y - 0.012, '\n'.join(lines), fontsize=size, va='top',
             color=INK, linespacing=1.45, wrap=True)


def header(fig, subtitle):
    fig.text(0.07, 0.955, 'FROST in ContinuIX: submission summary',
             fontsize=16, weight='bold')
    fig.text(0.07, 0.935, subtitle, fontsize=8.5, color=INK2)
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.925, 0.925], color=GRID, lw=1))


def footer(fig, page, total=2):
    fig.text(0.07, 0.025, f'GROUP_FAU2 · method01 · status {STATUS_DATE} · '
             'details: experiments/continuix/STATUS.md and README_GROUP_FAU2_method01.txt',
             fontsize=6.5, color=MUTED)
    fig.text(0.93, 0.025, f'{page}/{total}', fontsize=6.5, color=MUTED, ha='right')


def page1(total=2):
    fig = plt.figure(figsize=A4)
    header(fig, 'Oskar Herrmann (ORCID 0000-0002-0319-9065) · JRG Glacier Systems & '
                'Natural Hazards, Institute of Geography, FAU')

    kpis = [(str(len(F)), 'result files'), ('EXP01–20', 'all experiments'),
            ('8', 'glaciers (6 real, 2 synth.)'), (GPU_HOURS, 'GPU hours (A40)'),
            ('end Oct', 'deadline 2026')]
    for i, (v, l) in enumerate(kpis):
        x = 0.07 + i * 0.178
        fig.text(x, 0.885, v, fontsize=17, weight='bold')
        fig.text(x, 0.868, l, fontsize=7.2, color=INK2)

    text_block(fig, 0.07, 0.835, 'Method', [
        '• IGM 3.2 ice flow (CNN emulator) on a 2.5D grid: 50 m (G01 100 m; S02 at its 100 m data); provided thickness kept fixed,',
        '   basal sliding (tau_ref) inverted from the surface velocities (Laplacian regularisation, λ = 1e11 for all glaciers).',
        '• Three-parameter SMB (ELA, ablation and accumulation gradient) calibrated with ES-MDA: 36 members, 6 iterations,',
        '   observed dh/dt averaged in 50 m elevation bands; correlated dh/dt error + model error 0.5 m/yr per band (G03 1.3, G05 1.5).',
        '• SMB = calibrated SMB model on the mid-period surface; FDIV = time-mean flux divergence of the calibrated ensemble;',
        '   UNCT_SMB / UNCT_FDIV = ensemble standard deviation; ice equivalent, 910 kg m⁻³. Citation: Herrmann et al. (2025), AoG.'])

    # EXP01 chart
    ax = fig.add_axes([0.10, 0.47, 0.83, 0.235])
    gl = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'S01', 'S02']
    x = np.arange(len(gl))
    obs = [F[f'EXP01_{g}']['obs'] for g in gl]
    mod = [M[f'EXP01_{g}']['dh_mod'] for g in gl]
    smb = [F[f'EXP01_{g}']['smb'] for g in gl]
    unc = [F[f'EXP01_{g}']['unct'] for g in gl]
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.scatter(x - 0.18, obs, s=46, facecolor='white', edgecolor=INK, lw=1.2,
               zorder=3, label='observed dh/dt')
    ax.scatter(x, mod, s=46, color=BLUE, edgecolor='white', lw=1.2, zorder=3,
               label='modelled dh/dt (calibrated ensemble)')
    ax.errorbar(x + 0.18, smb, yerr=unc, fmt='D', ms=5.5, color=ORANGE,
                mec='white', mew=1, ecolor=ORANGE, elinewidth=1.4, capsize=0,
                zorder=3, label='submitted SMB ± UNCT_SMB')
    ax.set_xticks(x, gl)
    ax.set_ylabel('glacier mean (m i.e. yr⁻¹)')
    ax.set_title('EXP01 reference: the calibration matches the observed thinning except on G03')
    grid(ax)
    ax.legend(loc='lower left', ncol=1, fontsize=7)
    ax.annotate('G03: upper half thins\n~1.2 m/yr too fast', (2, mod[2]),
                xytext=(2.45, -3.1), fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
    ax.set_ylim(-4.6, 2.4)

    # EXP03-15 sensitivity: glaciers x experiments, SMB minus EXP01
    from matplotlib.colors import LinearSegmentedColormap
    ax = fig.add_axes([0.10, 0.125, 0.80, 0.255])
    exps, rows, delta = sensitivity()
    labels = ['THK\n±30 %', 'THK\n±30 %\n2× corr', 'THK\n±16 %\nmean', 'THK\n×1.3',
              'THK\n×0.7', 'VEL\n10 m/a\n100 m', 'VEL\n10 m/a\n1 km', 'VEL\n10 %\n100 m',
              'VEL\n10 % xy\n100 m', 'VEL\n10 % xy\n1 km', 'VEL\nbias\nramp',
              'RES\n3×', 'RES\n100 m']
    cmap = LinearSegmentedColormap.from_list('div', [ORANGE, '#f1f0ea', BLUE])
    ax.imshow(delta, cmap=cmap, vmin=-0.5, vmax=0.5, aspect='auto')
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            v = delta[i, j]
            ax.text(j, i, f'{v:+.2f}'.replace('+0.00', '0.00').replace('-0.00', '0.00'),
                    ha='center', va='center', fontsize=6.4,
                    color='white' if abs(v) > 0.3 else INK)
    ax.set_xticks(np.arange(len(exps)), [f'{e[3:]}\n{l}' for e, l in zip(exps, labels)],
                  fontsize=6.3)
    ax.set_yticks(np.arange(len(rows)), [g + (' *' if g in ('G01', 'G05', 'S01', 'S02') else '')
                                         for g in rows])
    ax.set_xticks(np.arange(-0.5, len(exps)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which='minor', color='white', lw=2)
    ax.tick_params(which='both', length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.axhline(3.5, color=INK2, lw=0.8)
    ax.set_title(f'EXP03–15: glacier-mean SMB minus EXP01 (m i.e. yr⁻¹); '
                 f'{np.mean(np.abs(delta) <= 0.07):.0%} of the {delta.size} runs within ±0.07')
    fig.text(0.10, 0.048, '* mandatory glaciers (above the line); G02, G03, G04, G06 optional.\n'
             'Largest shifts: local thickness noise (EXP03/04) and 100 m resolution on the small glaciers.',
             fontsize=6.6, color=INK2, linespacing=1.4)
    footer(fig, 1, total)
    return fig


def sensitivity():
    """EXP03-15 glacier-mean SMB minus EXP01 per glacier and experiment."""
    exps = [f'EXP{i:02d}' for i in range(3, 16)]
    rows = ['G01', 'G05', 'S01', 'S02', 'G02', 'G03', 'G04', 'G06']
    return exps, rows, np.array([[F[f'{e}_{g}']['smb'] - F[f'EXP01_{g}']['smb']
                                   for e in exps] for g in rows])


def page2(total=2):
    exps, rows, delta = sensitivity()
    i_max = np.unravel_index(np.argmax(delta), delta.shape)
    res = delta[:, -2:]
    i_res = np.unravel_index(np.argmin(res), res.shape)
    fig = plt.figure(figsize=A4)
    header(fig, 'Raw data (EXP02), global data (EXP16–20), the G03 limitation and what is left')

    ax = fig.add_axes([0.10, 0.60, 0.83, 0.28])
    real = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06']
    x = np.arange(len(real))
    ax.axhline(0, color=MUTED, lw=0.6)
    for off, exp, c, lab in [(-0.22, 'EXP01', BLUE, 'EXP01 (site data)'),
                             (-0.08, 'EXP02', ORANGE, 'EXP02 (raw data)')]:
        ax.errorbar(x + off, [F[f'{exp}_{g}']['smb'] for g in real],
                    yerr=[F[f'{exp}_{g}']['unct'] for g in real], fmt='o', ms=5.5,
                    color=c, mec='white', mew=0.8, elinewidth=1.4, label=lab, zorder=3)
    for j, e in enumerate(['EXP16', 'EXP17', 'EXP18', 'EXP19', 'EXP20']):
        ax.errorbar(x + 0.06 + j * 0.05, [F[f'{e}_{g}']['smb'] for g in real],
                    yerr=[F[f'{e}_{g}']['unct'] for g in real], fmt='s', ms=3.8,
                    color=AQUA, mec='white', mew=0.6, elinewidth=0.9, alpha=0.9,
                    label='EXP16–20 (global data)' if j == 0 else None, zorder=3)
    ax.scatter(x - 0.22, [F[f'EXP01_{g}']['obs'] for g in real], marker='_', s=160,
               color=INK, lw=1.4, zorder=4, label='observed dh/dt (site)')
    ax.scatter(x + 0.16, [F[f'EXP16_{g}']['obs'] for g in real], marker='_', s=260,
               color=INK2, lw=1.4, zorder=4, label='observed dh/dt (Hugonnet 2000–20)')
    ax.set_xticks(x, real)
    ax.set_ylabel('glacier-mean SMB ± UNCT_SMB (m i.e. yr⁻¹)')
    ax.set_title('SMB per glacier: raw data close to EXP01; global data constrain the SMB only weakly')
    grid(ax)
    ax.legend(loc='upper right', ncol=2, fontsize=6.8)

    # G03 profile
    ax = fig.add_axes([0.10, 0.30, 0.36, 0.215])
    p = D['g03']
    z = [b['z'] for b in p]
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.plot([b['need'] for b in p], z, '-o', color=ORANGE, ms=3.5, lw=1.6,
            label='required SMB (obs dh/dt + FDIV)')
    ax.plot([b['smb'] for b in p], z, '-o', color=BLUE, ms=3.5, lw=1.6,
            label='calibrated SMB')
    ax.set_xlabel('m i.e. yr⁻¹')
    ax.set_ylabel('elevation (m)')
    ax.set_title('G03 (EXP01): a two-gradient profile\ncannot follow the required SMB', fontsize=8.5)
    grid(ax, 'both')
    ax.legend(loc='lower right', fontsize=6.5)

    text_block(fig, 0.53, 0.53, 'Key findings', [
        '• EXP01: dh/dt matched within ±0.25 m/yr on 7 of 8',
        '   glaciers; posterior spread ~10–50 % of the prior.',
        f'• EXP03–15 (all 8 glaciers): {np.mean(np.abs(delta) <= 0.07):.0%} of the runs within',
        f'   ±0.07 m/yr of EXP01; up to {delta[i_max]:+.2f} ({rows[i_max[0]]} {exps[i_max[1]]}),',
        f'   {res[i_res]:+.2f} in RES ({rows[i_res[0]]} {exps[-2:][i_res[1]]}).',
        '• EXP02: GPR-only glaciers (G02, G03, G05, S02) get',
        '   a thickness shaped like IGM\'s start thickness,',
        '   scaled to the profiles (r = 0.69–0.94 vs EXP01).',
        '• GLOB: Hugonnet errors 1.2–4.3 m/yr per pixel;',
        '   posterior 65–108 % of the prior, UNCT_SMB',
        '   0.4–1.3 m/yr (G02 1.4–3.6); 12 of 30 runs miss',
        '   the glacier-mean dh/dt by > 0.5 m/yr.'], size=7.3)

    text_block(fig, 0.07, 0.235, 'Known limitations (all stated in the README)', [
        '• G03: required SMB flat (~−4 m/yr) at 2300–2600 m, then rising ~11 m/yr per km to 3100 m; a weaker sliding regularisation fixed',
        '   the upper-basin velocities (1.33 → 1.09 × obs) but not the dh/dt fit, so it was reverted. Basins at one elevation behave alike.',
        '• G01: one ELA for an ice cap with 11 outlet basins. S01: velocities ~28 % too low after the inversion.',
        '• FDIV has local extremes in thick ice (< 0.1 % of cells, e.g. EXP04 G01); kept, not clipped.',
        '• GLOB: with large, spatially correlated dh/dt errors a glacier-wide offset is cheap, so the SMB follows the thickness product',
        '   (G01 EXP19 Maffezzoli: −2.9 vs −0.9 m/yr; G02 ELA near the summit, some members lose their ice within 20 years).'])

    text_block(fig, 0.07, 0.11, 'Open', [
        f'Upload data/results/continuix/GROUP_FAU2/ ({len(F)} files, README, log, checklist) to the ContinuIX SharePoint.',
        'GLOB accepted as documented; model grid chosen after the tests of 8 October (see the resolution section).'])
    footer(fig, 2, total)
    return fig


def main():
    load()
    with PdfPages(os.path.join(HERE, 'ContinuIX_FROST_summary.pdf')) as pdf:
        for n, fig in enumerate([page1(), page2()], start=1):
            pdf.savefig(fig)
            fig.savefig(os.path.join(HERE, f'page{n}.png'), dpi=110)
            plt.close(fig)
    print('ok')


if __name__ == '__main__':
    main()
