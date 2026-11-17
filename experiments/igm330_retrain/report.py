#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
PDF report: emulator retraining in the forward runs on IGM 3.3.0 (IGM default
rule and the new adaptive 'credit' rule) against the v3-dev rule of IGM 3.2.1
+ fork (20 iterations every year of model time, central_europe_retrain_t1):
speed against the solver (forward_test.py), calibration wall time, posterior
SMB parameters, misfit and GLAMOS/WGMS evaluation (evaluate_sbatch.sh); and
the final setup, IGM default rule with the live IGM model (igm330_live,
live_model_test.py, worker_sweep.py).

Reads saved results and logs only (fine on a login node). From the repo root:
    python experiments/igm330_retrain/report.py
Writes experiments/igm330_retrain/igm330_retrain_report.pdf and
tables/{forward,calibration}.tsv.
"""

import argparse
import json
import os
import re
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from netCDF4 import Dataset  # noqa: E402

sys.path.insert(0, os.getcwd())
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from frost.visualization import plot_style  # noqa: E402
from forward_test import SOURCE, TARGET, OLD_TARGET  # noqa: E402

TITLE = 'FROST on IGM 3.3.0: emulator retraining and live model'
PAGE = (8.27, 11.69)
L, R = 0.09, 0.91
BODY = 9.5
T1_JOB = 4988518  # central_europe_retrain_t1
JOBS = {'igm_default': 5014441, 'credit': 5014442, 'live': 5022120}  # calibration arrays
LIVE_TEST_JOBS = {'fresh': 5021885, 'live': 5022119}  # live_model_test.py (Rhone)

# rules: (label, colour); 't1' = IGM 3.2.1 + fork, the reference
RULES = {
    'solver': ('Solver (reference, no emulator)', '#111111'),
    't1': ('20 it. every model year (IGM 3.2.1 + fork, v3-dev until now)', '#1f6fb2'),
    'igm_default': ('IGM default: 5 it. every 5 steps', '#e08a00'),
    'credit': ('Credit (IGM 3.3.0): 5 it. when inputs changed by kappa 1e-3', '#2a9a63'),
    'credit_x20': ('Credit, 20 it. per retraining', '#7b4fa0'),
    'credit_k1e-2': ('Credit, kappa 1e-2', '#c62f35'),
    'live': ('IGM default + live IGM model (final setup)', '#8c564b'),
}
# forward runs: rule -> (results root, variant folder)
FORWARD = {'solver': (TARGET, 'solver'), 't1': (OLD_TARGET, 'retrain_t1x20'),
           'igm_default': (TARGET, 'igm_default'), 'credit': (TARGET, 'credit'),
           'credit_x20': (TARGET, 'credit_x20'), 'credit_k1e-2': (TARGET, 'credit_k1e-2')}
SOLVER_OLD = (OLD_TARGET, 'solver')  # IGM 3.2.1 solver: reference of t1
# calibrations: rule -> (results, log prefix, evaluation folder)
CALIB = {'t1': ('data/results/central_europe_retrain_t1', 'ce_t1', 'eval_central_europe_retrain_t1'),
         'igm_default': ('data/results/igm330_default', 'igm330', 'eval_igm330_default'),
         'credit': ('data/results/igm330_credit', 'igm330', 'eval_igm330_credit'),
         'live': ('data/results/igm330_live', 'igm330', 'eval_igm330_live')}

# Written after the results (2026-11-16, jobs 5014381, 5014441, 5014442)
CONCLUSIONS = [
    ('b', 'IGM 3.3.0 runs FROST with one line changed in IGM (igm_run read an ice-flow '
          'option also in runs without ice flow, so the OGGM-shop download failed). A second '
          '3.3.0 bug (a second IGM run in one process fails on a stale jit tf.function) is '
          'worked around in FROST: igm_wrapper.reset_igm_tf_functions.'),
    ('b', 'Speed: both 3.3.0 rules calibrate faster than the fork rule: 254 (IGM default) '
          'and 265 min (credit) against 307 min summed over the 17 glaciers (-17 % / -14 %); '
          'longest task 22 / 23 min against 28 (Aletsch 21 min against 28).'),
    ('b', 'Calibration: the same answer. Posterior ELA within 0.5 posterior std of the fork '
          'rule on every glacier (mean |dELA| 18 / 17 m; largest Silvretta -63 m, Pasterze '
          '+35 m), gradients within 0.9 std; mean misfit 0.42 / 0.42 against 0.45.'),
    ('b', 'Forward runs against the solver: all rules within 4-6 % (Rhone, Aletsch) and '
          '5-9 % (Corbassiere). IGM 3.3.0 removed the jumps of "5 every 5 steps" on slow '
          'glaciers: Corbassiere 6 % on 3.3.0 against 27 % on 3.2.1 (speed 2001: 12 against '
          '3.9 m/yr), the reason the fork rule was introduced.'),
    ('b', 'Credit retrains every ~3 steps (kappa 1e-3) and buys nothing over the default '
          'here; 20 iterations per retraining is the most accurate (3.6-4.7 %) for +5-10 % '
          'forward time; kappa 1e-2 is too coarse on Rhone (9 %).'),
    ('b', 'GLAMOS/WGMS SMB profiles (17 glaciers): RMSE 0.99 / 0.99 against 0.98 m w.e./yr, '
          'bias +0.33 against +0.32, bins within 2 std 79 % against 83 %. Slightly worse with '
          'both 3.3.0 rules alike (so from IGM 3.3.0 or run-to-run noise, not the rule): '
          'Kesselwandferner +0.17, Pasterze +0.12, Gietro +0.10; better Tsanfleuron -0.10, '
          'Claridenfirn -0.07. For scale, the fork rule against v2 moved the RMSE by 0.004.'),
    ('b', 'Choice: IGM default (5 every 5 steps) for the forward runs: plain IGM settings, '
          'fastest, same calibration (igm_wrapper.DEFAULT_RETRAIN_RULE). The small GLAMOS '
          'loss is common to both 3.3.0 rules. Download + inversion on 3.3.0: end-to-end test.'),
    ('b', 'Live IGM model (2026-11-17, last page): x2 faster calibration (127 against 254 min), '
          'same posterior and GLAMOS profile RMSE (0.992 / 0.992); 8 workers saturate the GPU.'),
]

C = dict(fg='#111111', muted='#555555', grid='#c8c8c8')


# ---------------------------------------------------------------- layout

def theme():
    plt.rcdefaults()
    plt.rcParams.update({k: v for k, v in plot_style.DARK_THEME.items()
                         if k.startswith(('axes.spines', 'xtick.major', 'ytick.major',
                                          'axes.grid', 'axes.axisbelow', 'grid.linestyle',
                                          'grid.linewidth', 'legend.frameon'))})
    plt.rcParams['grid.color'] = C['grid']
    plot_style.use_font()
    plt.rcParams['font.monospace'] = ['JetBrains Mono', 'DejaVu Sans Mono']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 9


def page(pdf, n, total, topic, draw):
    fig = plt.figure(figsize=PAGE)
    fig.text(L, 0.975, TITLE, fontsize=7.5, color=C['muted'], va='top')
    fig.text(R, 0.975, topic, fontsize=7.5, color=C['muted'], va='top', ha='right')
    fig.text(R, 0.02, f'{n} / {total}', fontsize=7.5, color=C['muted'], ha='right')
    fig.text(L, 0.02, 'experiments/igm330_retrain/report.py', fontsize=6, color=C['muted'])
    draw(fig)
    pdf.savefig(fig)
    if os.environ.get('PREVIEW_DIR'):
        fig.savefig(os.path.join(os.environ['PREVIEW_DIR'], f'igm330_page{n}.png'), dpi=80)
    plt.close(fig)


def text_block(fig, y, blocks, width=98):
    line = 0.0158
    for kind, text in blocks:
        if kind == 'h':
            y -= 0.008
            fig.text(L, y, text, fontsize=13, weight='bold', color=C['fg'], va='top')
            y -= 0.025
            continue
        indent = L + 0.022 if kind == 'b' else L
        lines = textwrap.wrap(text, width - (3 if kind == 'b' else 0))
        if kind == 'b':
            fig.text(L + 0.006, y, '-', fontsize=BODY, color=C['muted'], va='top')
        for i, ln in enumerate(lines):
            fig.text(indent, y - i * line, ln, fontsize=BODY, color=C['fg'], va='top')
        y -= len(lines) * line + 0.006
    return y


def heading(fig, text, sub=None, y=0.945):
    fig.text(L, y, text, fontsize=14, weight='bold', color=C['fg'], va='top')
    if sub:
        for i, ln in enumerate(textwrap.wrap(sub, 105)):
            fig.text(L, y - 0.025 - 0.014 * i, ln, fontsize=8.5, color=C['muted'], va='top')


def short(name):
    return (name.replace('Glacier_de_la_', '').replace('Glacier_de_', '')
            .replace('Glacier_du_', '').replace('Grosser_', '').replace('gletscher', '').replace('_', ' '))


# ---------------------------------------------------------------- data

def glaciers(path):
    with open(path) as f:
        return [line.split()[:2] for line in f if line.strip()]


def speed_mask(rgi_id):
    """Ice with observed speeds on the inversion geometry (as emulator_test)."""
    with Dataset(os.path.join(SOURCE, rgi_id, 'Preprocess', 'outputs', 'output.nc')) as ds:
        obs = np.array(ds['velsurfobs_mag'][:], dtype=float)
        thk = np.array(ds['thk_inversion' if 'thk_inversion' in ds.variables else 'thk'][:],
                       dtype=float)
    return (thk > 1) & np.isfinite(obs) & (obs > 0)


def forward_run(root, variant, rgi_id):
    path = os.path.join(root, rgi_id, variant, 'outputs', 'output.nc')
    if not os.path.exists(path):
        return None
    with Dataset(path) as ds:
        t = np.array(ds['time'][:], dtype=float)
        v = np.array(ds['velsurf_mag'][:], dtype=float)
        thk = np.array(ds['thk'][:], dtype=float)
    mask = speed_mask(rgi_id)
    speed = np.array([np.nanmean(vi[mask & (h > 1)]) for vi, h in zip(v, thk)])
    out = dict(t=t, speed=speed, runtime=np.nan, retrains=np.nan, steps=np.nan)
    rt = os.path.join(root, rgi_id, variant, 'runtime.txt')
    if os.path.exists(rt):
        out['runtime'] = float(open(rt).read().split()[0])
    log = os.path.join(root, rgi_id, variant, 'credit_log.jsonl')
    if os.path.exists(log):
        recs = [json.loads(line) for line in open(log) if line.strip()]
        out['steps'] = len(recs)
        # after the start-up training (step 0)
        out['retrains'] = sum(r['is_retrain'] for r in recs if r['step'] > 0)
    return out


def speed_error(run, ref):
    """Mean |run - ref| / ref of the glacier-mean speed, 2001-2020 (%)."""
    n = min(len(run['speed']), len(ref['speed']))
    sel = slice(1, n)
    return 100 * np.nanmean(np.abs(run['speed'][sel] - ref['speed'][sel]) / ref['speed'][sel])


def forward_table():
    rows, series = [], {}
    for rgi_id, name in glaciers(os.path.join(HERE, 'glaciers_forward.txt')):
        runs = {rule: forward_run(root, var, rgi_id) for rule, (root, var) in FORWARD.items()}
        old_solver = forward_run(*SOLVER_OLD, rgi_id)
        series[name] = runs
        for rule, run in runs.items():
            if run is None or rule == 'solver':
                continue
            ref = old_solver if rule == 't1' else runs['solver']
            rows.append(dict(glacier=name, rule=rule, runtime_s=run['runtime'],
                             error_pct=speed_error(run, ref) if ref is not None else np.nan,
                             steps=run['steps'], retrains=run['retrains'],
                             speed_2020=run['speed'][-1]))
        for label, run in [('solver', runs['solver']), ('solver_321', old_solver)]:
            if run is not None:
                rows.append(dict(glacier=name, rule=label, runtime_s=run['runtime'],
                                 error_pct=0.0, speed_2020=run['speed'][-1]))
    return pd.DataFrame(rows), series


def elapsed_minutes(log):
    try:
        with open(log) as f:
            m = re.search(r'Elapsed runtime\s*:\s*(\d+):(\d+):(\d+)', f.read())
    except OSError:
        return np.nan
    return int(m[1]) * 60 + int(m[2]) + int(m[3]) / 60 if m else np.nan


def calibration(rgi_dir):
    path = os.path.join(rgi_dir, 'calibration_results.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    post = [r for r in d['diagnostics'] if r['posterior']]
    return dict(mean=np.array(d['final_mean']), std=np.array(d['final_std']),
                misfit=post[-1]['misfit'] if post else np.nan)


def profiles(folder):
    path = os.path.join(HERE, folder, 'tables', 'profiles.tsv')
    if not os.path.exists(path):
        return {}
    p = pd.read_csv(path, sep='\t')
    p = p[p['run'] == 'new']
    return dict(zip(p['rgi_id'], p['rmse']))


def calibration_table(jobs):
    rows = []
    rmse = {rule: profiles(ev) for rule, (_, _, ev) in CALIB.items()}
    gl = glaciers(os.path.join(os.path.dirname(HERE), 'igm330_default', 'glaciers.txt'))
    for task, (rgi_id, name) in enumerate(gl, start=1):
        for rule, (root, prefix, _) in CALIB.items():
            res = calibration(os.path.join(root, rgi_id))
            job = jobs.get(rule)
            log = os.path.join('logs', f'{prefix}_{job}_{task}.out') if job else ''
            row = dict(glacier=name, rgi_id=rgi_id, rule=rule, minutes=elapsed_minutes(log),
                       glamos_rmse=rmse[rule].get(rgi_id, np.nan))
            if res is not None:
                row.update(ela=res['mean'][0], abl=res['mean'][1], acc=res['mean'][2],
                           ela_std=res['std'][0], abl_std=res['std'][1], acc_std=res['std'][2],
                           misfit=res['misfit'])
            rows.append(row)
    table = pd.DataFrame(rows)
    # differences to t1 in units of t1's posterior spread
    ref = table[table.rule == 't1'].set_index('glacier')
    for p in ['ela', 'abl', 'acc']:
        if p in table:
            table[f'd_{p}'] = table[p] - table.glacier.map(ref[p])
            table[f'd_{p}_sd'] = table[f'd_{p}'] / table.glacier.map(ref[f'{p}_std'])
    return table


# ---------------------------------------------------------------- pages

def summary_rows(fwd, cal):
    out = []
    for rule in ['t1', 'igm_default', 'credit', 'credit_x20', 'credit_k1e-2', 'live']:
        f = fwd[fwd.rule == rule]
        c = cal[cal.rule == rule] if rule in CALIB else cal.iloc[:0]
        done = c.dropna(subset=['misfit']) if 'misfit' in c else c.iloc[:0]
        out.append([RULES[rule][0].split(' (')[0] if rule != 't1' else '20 it./yr (3.2.1 fork)',
                    f'{f.error_pct.mean():.1f}' if len(f) else '-',
                    f'{f.runtime_s.mean():.0f}' if len(f) else '-',
                    f'{c.minutes.sum():.0f}' if c.minutes.notna().any() else '-',
                    f'{c.minutes.max():.0f}' if c.minutes.notna().any() else '-',
                    f'{done.misfit.mean():.2f}' if len(done) else '-',
                    f'{c.glamos_rmse.mean():.3f}' if c.glamos_rmse.notna().any() else '-',
                    f'{done.d_ela_sd.abs().max():.2f}' if rule != 't1' and len(done) else '-'])
    return out


def draw_summary(fig, fwd, cal, jobs):
    heading(fig, 'Emulator retraining on IGM 3.3.0',
            'Update from IGM 3.2.1 + fork (retrain_interval) to plain IGM 3.3.0: which '
            'retraining rule for the forward runs of the calibration?')
    y = text_block(fig, 0.89, [
        ('p', 'FROST v3-dev retrained the ice-flow emulator in the forward runs 20 iterations '
              'every year of model time, an option of our IGM fork. IGM 3.3.0 brings its own '
              'adaptive rule ("credit": retrain once the normalised change of the inputs thk, '
              'usurf, ... accumulated since the last retraining reaches kappa). The fork is '
              'dropped; FROST now chooses between IGM options only (igm_wrapper.RETRAIN_RULES).'),
        ('b', 'Forward test: one posterior member of Aletsch, Rhone and Corbassiere, '
              '2000-2020, each rule against the solver of the same IGM version. Error = '
              'mean |emulator - solver| / solver of the glacier-mean speed 2001-2020.'),
        ('b', 'Calibration: the 17 glaciers with ice flow, inputs, inversion and emulator '
              'copied from central_europe_retrain (as t1), so only IGM and the rule differ. '
              'Wall time of the Slurm task (1 A40), ES-MDA posterior, misfit of the posterior '
              'run, GLAMOS/WGMS SMB profile RMSE (evaluate.py).'),
    ])
    cols = ['Rule', 'Fwd\nerror %', 'Fwd\ns', 'Calib\nmin', 'Longest\nmin', 'Misfit',
            'GLAMOS\nRMSE', 'max\n|dELA|/sd']
    ax = fig.add_axes([L, y - 0.15, R - L, 0.14])
    ax.axis('off')
    tab = ax.table(cellText=summary_rows(fwd, cal), colLabels=cols, loc='upper left',
                   cellLoc='right', colLoc='right',
                   colWidths=[0.30] + [0.10] * 7)
    for (r, c), cell in tab.get_celld().items():
        cell.set_height(0.16 if r == 0 else 0.11)
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('white' if r else C['grid'])
        cell.set_linewidth(0.6)
        if c == 0:
            cell.set_text_props(ha='left')
            cell.PAD = 0.02
        if r == 0:
            cell.set_text_props(weight='bold')
    y -= 0.135
    note = ('Fwd: mean of the 3 glaciers (t1 against the IGM 3.2.1 solver). Calib: sum and '
            'longest of the 17 tasks; misfit and RMSE: means; dELA: posterior mean minus t1 in '
            'units of t1\'s posterior std.')
    for i, ln in enumerate(textwrap.wrap(note, 120)):
        fig.text(L, y - 0.012 * i, ln, fontsize=7.5, color=C['muted'], va='top')
    y -= 0.022
    concl = CONCLUSIONS or [('p', 'Conclusions pending: written once the forward test and both '
                                  'calibrations have run.')]
    y = text_block(fig, y, [('h', 'Conclusions')] + concl)
    missing = [r for r in ['igm_default', 'credit', 'live'] if not jobs.get(r)]
    if missing:
        text_block(fig, y - 0.01, [('p', f'No Slurm job id given for {", ".join(missing)}: '
                                         'wall times missing.')])


def draw_forward_series(fig, series):
    heading(fig, 'Forward test: glacier-mean speed',
            'One posterior member, 2000-2020. Solver = IGM 3.3.0 solver; the IGM 3.2.1 solver '
            '(reference of the fork rule) as a thin dashed line.')
    names = list(series)
    h = 0.76 / max(len(names), 1)
    handles = {}
    for i, name in enumerate(names):
        ax = fig.add_axes([L + 0.04, 0.82 - (i + 1) * h + 0.035, R - L - 0.04, h - 0.06])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        runs = series[name]
        rgi_id = [g for g, n in glaciers(os.path.join(HERE, 'glaciers_forward.txt')) if n == name][0]
        old = forward_run(*SOLVER_OLD, rgi_id)
        if old is not None:
            handles['solver_321'], = ax.plot(old['t'], old['speed'], color='#111111', lw=0.8,
                                             ls='--', label='Solver IGM 3.2.1')
        for rule, run in runs.items():
            if run is None:
                continue
            label, colour = RULES[rule]
            handles[rule], = ax.plot(run['t'], run['speed'], color=colour, label=label,
                                     lw=2.2 if rule == 'solver' else 1.3)
        ax.set_title(short(name), fontsize=10, loc='left')
        ax.set_ylabel('Mean speed (m/yr)')
        if i == len(names) - 1:
            ax.set_xlabel('Year')
    order = [k for k in ['solver', 'solver_321', 't1', 'igm_default', 'credit', 'credit_x20',
                         'credit_k1e-2'] if k in handles]
    fig.legend([handles[k] for k in order], [handles[k].get_label() for k in order],
               loc='upper left', bbox_to_anchor=(L, 0.885), ncol=2, fontsize=7.5)


def draw_forward_cost(fig, fwd):
    heading(fig, 'Forward test: accuracy against cost',
            'Each point one rule on one glacier. Lower left is better. Credit rules: number of '
            'retrainings after the start-up training over the time steps of the run.')
    ax = fig.add_axes([L + 0.04, 0.56, R - L - 0.04, 0.30])
    markers = {g: m for g, m in zip(fwd.glacier.unique(), ['o', 's', '^', 'D', 'v'])}
    for _, r in fwd[~fwd.rule.isin(['solver', 'solver_321'])].iterrows():
        ax.scatter(r.runtime_s, r.error_pct, color=RULES[r.rule][1], marker=markers[r.glacier],
                   s=45, zorder=3)
    for g, m in markers.items():
        ax.scatter([], [], color='#888888', marker=m, label=short(g))
    for rule in ['t1', 'igm_default', 'credit', 'credit_x20', 'credit_k1e-2']:
        if (fwd.rule == rule).any():
            ax.scatter([], [], color=RULES[rule][1], marker='o', label=RULES[rule][0])
    ax.set_xlabel('Wall time of the 20-year run alone on one A40 (s)')
    ax.set_ylabel('Speed error vs solver (%)')
    ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    sol = fwd[fwd.rule.isin(['solver', 'solver_321'])]
    lines = [f'{short(r.glacier)} {"IGM 3.3.0" if r.rule == "solver" else "IGM 3.2.1"} solver: '
             f'{r.runtime_s:.0f} s' for _, r in sol.iterrows()]
    cr = fwd[fwd.rule.str.startswith('credit')].dropna(subset=['steps'])
    lines += [f'{short(r.glacier)} {r.rule}: {r.retrains:.0f} retrainings in {r.steps:.0f} steps'
              for _, r in cr.iterrows()]
    text_block(fig, 0.36, [('p', '; '.join(lines) + '.')] if lines else [])


def draw_calibration(fig, cal):
    heading(fig, 'Calibration: wall time and posterior',
            '17 glaciers with ice flow, one Slurm task each (1 A40, 8 workers, 32 members, '
            '4 ES-MDA iterations + posterior run). Bottom: posterior mean ELA minus t1, in units '
            'of t1\'s posterior std (|x| < 1: inside the spread).')
    rules = [r for r in CALIB if (cal.rule == r).any()]
    names = list(dict.fromkeys(cal.glacier))
    x = np.arange(len(names))
    w = 0.8 / len(rules)
    ax = fig.add_axes([L + 0.04, 0.52, R - L - 0.04, 0.33])
    for k, rule in enumerate(rules):
        c = cal[cal.rule == rule].set_index('glacier').reindex(names)
        ax.bar(x + (k - (len(rules) - 1) / 2) * w, c.minutes, w, color=RULES[rule][1],
               label=RULES[rule][0])
    ax.axhline(20, color='#c62f35', ls='--', lw=0.8, label='Target 20 min per glacier')
    ax.set_xticks(x, [short(n) for n in names], rotation=60, ha='right', fontsize=7.5)
    ax.set_ylabel('Wall time (min)')
    ax.legend(fontsize=7, loc='upper right')
    ax = fig.add_axes([L + 0.04, 0.12, R - L - 0.04, 0.27])
    others = [r for r in rules if r != 't1']
    w2 = 0.8 / max(len(others), 1)
    for k, rule in enumerate(others):
        c = cal[cal.rule == rule].set_index('glacier').reindex(names)
        if 'd_ela_sd' in c:
            ax.bar(x + (k - (len(others) - 1) / 2) * w2, c.d_ela_sd, w2, color=RULES[rule][1],
                   label=RULES[rule][0])
    ax.axhspan(-1, 1, color='#888888', alpha=0.12, lw=0, label='Inside t1 posterior +-1 std')
    ax.axhline(0, color='#111111', lw=0.6)
    ax.set_xticks(x, [short(n) for n in names], rotation=60, ha='right', fontsize=7.5)
    ax.set_ylabel('ELA - ELA(t1) / std(t1)')
    ax.legend(fontsize=7, loc='upper right')


def draw_fit(fig, cal):
    heading(fig, 'Calibration: fit',
            'Top: misfit of the posterior run (dh/dt bands, lower is better). Bottom: RMSE of '
            'the posterior SMB against the GLAMOS/WGMS profile 2000-2019 (m w.e./yr; glaciers '
            'with a reference only).')
    rules = [r for r in CALIB if (cal.rule == r).any()]
    names = list(dict.fromkeys(cal.glacier))
    x = np.arange(len(names))
    w = 0.8 / len(rules)
    for top, col, label in [(0.52, 'misfit', 'Posterior misfit'),
                            (0.12, 'glamos_rmse', 'GLAMOS profile RMSE')]:
        ax = fig.add_axes([L + 0.04, top, R - L - 0.04, 0.30])
        for k, rule in enumerate(rules):
            c = cal[cal.rule == rule].set_index('glacier').reindex(names)
            if col in c:
                ax.bar(x + (k - (len(rules) - 1) / 2) * w, c[col], w, color=RULES[rule][1],
                       label=RULES[rule][0])
        ax.set_xticks(x, [short(n) for n in names], rotation=60, ha='right', fontsize=7.5)
        ax.set_ylabel(label)
        ax.legend(fontsize=7, loc='upper right')


def evaluation(folder):
    """evaluate.py summary (tables/evaluation.tsv) or None if it has not run."""
    path = os.path.join(HERE, folder, 'tables', 'evaluation.tsv')
    return pd.read_csv(path, sep='\t') if os.path.exists(path) else None


def glamos_rows():
    """One row per calibration (+ the ContinuIX submission as a reference):
    posterior against GLAMOS/WGMS (evaluate.py, 'new' run of each folder)."""
    def metrics(ev, run):
        e = ev[ev.run == run].set_index(['parameter', 'unit'])
        g = lambda p, u, c: e.loc[(p, u), c] if (p, u) in e.index else np.nan  # noqa: E731
        return [g('ela', 'ice', 'bias'), g('ela', 'ice', 'mae'), g('ela', 'ice', 'inside_2sigma_pct'),
                g('abl', 'w.e.', 'bias'), g('acc', 'w.e.', 'bias'),
                g('profile', 'w.e.', 'mae'), g('profile', 'w.e.', 'bias'),
                g('profile', 'w.e.', 'inside_2sigma_pct')]
    rows, labels = [], []
    ref = evaluation(CALIB['t1'][2])
    if ref is not None and (ref.run == 'submitted').any():
        rows.append(metrics(ref, 'submitted'))
        labels.append(('ContinuIX submission (v1)', '#888888'))
    for rule, (_, _, folder) in CALIB.items():
        ev = evaluation(folder)
        label = RULES[rule][0].split(' (')[0] if rule != 't1' else '20 it./yr (3.2.1 fork)'
        rows.append(metrics(ev, 'new') if ev is not None else [np.nan] * 8)
        labels.append((label.replace(' (final setup)', '') + ('' if ev is not None else ' (pending)'), RULES[rule][1]))
    return labels, rows


def draw_glamos(fig):
    heading(fig, 'GLAMOS/WGMS evaluation',
            'Posterior against the in-situ references (evaluate.py; GLAMOS for the Swiss '
            'glaciers, WGMS for the others): ELA, ablation and accumulation gradients and the '
            'SMB profile 2000-2019 per 50 m band. Not used in the calibration.')
    labels, rows = glamos_rows()
    cols = ['Run', 'ELA bias\nm', 'ELA MAE\nm', 'ELA in\n2 sd %', 'abl. grad\nbias', 'acc. grad\nbias',
            'Profile\nRMSE', 'Profile\nbias', 'Bins in\n2 sd %']
    fmt = ['{:.0f}', '{:.0f}', '{:.0f}', '{:+.2f}', '{:+.2f}', '{:.3f}', '{:+.3f}', '{:.0f}']
    cells = [[lab] + ['-' if not np.isfinite(v) else f.format(v) for v, f in zip(r, fmt)]
             for (lab, _), r in zip(labels, rows)]
    ax = fig.add_axes([L, 0.70, R - L, 0.17])
    ax.axis('off')
    tab = ax.table(cellText=cells, colLabels=cols, loc='upper left', cellLoc='right',
                   colLoc='right', colWidths=[0.32] + [0.085] * 8)
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.5)
    for (r, c), cell in tab.get_celld().items():
        cell.set_height(0.20 if r == 0 else 0.12)
        cell.set_edgecolor('white' if r else C['grid'])
        if c == 0:
            cell.set_text_props(ha='left', color=labels[r - 1][1] if r else C['fg'])
            cell.PAD = 0.02
        if r == 0:
            cell.set_text_props(weight='bold')
    note = ('Gradients in m w.e./yr/100 m, profile in m w.e./yr. ELA: 13 glaciers with a '
            'reference ELA inside the glacier (4 censored). Bins in 2 sd: share of profile bands '
            'where the reference lies inside 2 posterior std.')
    for i, ln in enumerate(textwrap.wrap(note, 125)):
        fig.text(L, 0.685 - 0.012 * i, ln, fontsize=7, color=C['muted'], va='top')
    # profile RMSE per glacier, difference to t1
    rmse = {rule: profiles(ev) for rule, (_, _, ev) in CALIB.items()}
    gl = glaciers(os.path.join(os.path.dirname(HERE), 'igm330_default', 'glaciers.txt'))
    names = [n for _, n in gl]
    x = np.arange(len(names))
    others = [r for r in CALIB if r != 't1' and rmse[r]]
    ax = fig.add_axes([L + 0.04, 0.17, R - L - 0.04, 0.42])
    w = 0.8 / max(len(others), 1)
    for k, rule in enumerate(others):
        d = [rmse[rule].get(g, np.nan) - rmse['t1'].get(g, np.nan) for g, _ in gl]
        ax.bar(x + (k - (len(others) - 1) / 2) * w, d, w, color=RULES[rule][1],
               label=RULES[rule][0])
    ax.axhline(0, color='#111111', lw=0.6)
    ax.set_xticks(x, [short(n) for n in names], rotation=60, ha='right', fontsize=7.5)
    ax.set_ylabel('Profile RMSE minus t1 (m w.e./yr)')
    ax.set_title('Per glacier: SMB profile RMSE against the fork rule (t1); below 0 = better',
                 fontsize=9, loc='left')
    ax.legend(fontsize=7, loc='upper right')


def live_test_timing():
    """Seconds per member of live_model_test.py: fresh (warm) and live, Rhone."""
    out = {}
    for mode, job in LIVE_TEST_JOBS.items():
        try:
            with open(os.path.join('logs', f'igm330_live_{job}.out'), errors='ignore') as f:
                t = [float(m[1]) for m in re.finditer(rf'^TIMING {mode} \S+ ela \S+ ([\d.]+) s',
                                                       f.read(), re.M)]
        except OSError:
            t = []
        out[mode] = t
    return out


def draw_live(fig, cal):
    heading(fig, 'Final setup: live IGM model',
            'Each calibration worker keeps its compiled IGM model for all members of a glacier '
            '(igm_wrapper.LiveIGM): weights reset, optimizer zeroed, new geometry and SMB, run. '
            'No IGM source change.')
    t = live_test_timing()
    fresh = t['fresh'][1:8] if len(t['fresh']) > 1 else []  # warm runs
    live = t['live'][1:] if len(t['live']) > 1 else []
    c = cal.groupby('rule').minutes.sum()
    lines = []
    if fresh and live:
        lines.append(('b', f'One Rhone member alone on an A40 (2000-2020, 300 start iterations): '
                           f'{np.mean(fresh):.1f} s as a fresh in-process run, {np.mean(live):.1f} s '
                           f'with the live model (x{np.mean(fresh) / np.mean(live):.1f}); '
                           f'results bit-identical for all 8 test members (thk, speed, SMB).'))
    if 'live' in c and 'igm_default' in c:
        d = cal[cal.rule.isin(['igm_default', 'live'])].pivot(index='glacier', columns='rule',
                                                               values='ela')
        dsd = cal[cal.rule == 'igm_default'].set_index('glacier').ela_std
        rel = ((d.live - d.igm_default).abs() / dsd).dropna()
        lines.append(('b', f'Calibration of the 17 flow glaciers: {c["live"]:.0f} min against '
                           f'{c["igm_default"]:.0f} min (IGM default without the live model, '
                           f'x{c["igm_default"] / c["live"]:.1f}); small glaciers x3, large x1.4 '
                           f'(there the compute dominates). Posterior identical on '
                           f'{(rel == 0).sum()} glaciers, max difference {rel.max():.2f} posterior '
                           f'std elsewhere (probably GPU nondeterminism with 8 workers on one GPU).'))
    lines.append(('b', 'Safety: after every IGM update rerun live_model_test_sbatch.sh (must stay '
                       'bit-identical); FROST_LIVE_IGM=0 falls back to fresh runs.'))
    y = text_block(fig, 0.88, lines)
    path = os.path.join(HERE, 'tables', 'worker_sweep.tsv')
    if not os.path.exists(path):
        return
    s = pd.read_csv(path, sep='\t')
    ax = fig.add_axes([L + 0.04, 0.40, (R - L) / 2 - 0.07, y - 0.47])
    ax2 = fig.add_axes([L + (R - L) / 2 + 0.05, 0.40, (R - L) / 2 - 0.05, y - 0.47])
    colours = {'Silvretta': '#2a9a63', 'Rhone': '#1f6fb2', 'Aletsch': '#e08a00'}
    for g, grp in s.groupby('glacier'):
        ok = grp[grp.ok.astype(str) == 'True']
        ax.plot(ok.workers, ok.warm_pass_s, 'o-', color=colours.get(g), label=g)
        ax2.plot(ok.workers, ok.gpu_mem_peak_gib, 'o-', color=colours.get(g), label=g)
        bad = grp[grp.ok.astype(str) != 'True']
        ax2.scatter(bad.workers, [48] * len(bad), marker='x', color=colours.get(g), s=50)
    ax.set_xlabel('Forward workers')
    ax.set_ylabel('Pass of 32 members, warm (s)')
    ax.set_title('Worker sweep: time per pass', fontsize=9, loc='left')
    ax.legend(fontsize=7)
    ax2.axhline(48, color='#c62f35', ls='--', lw=0.8)
    ax2.set_xlabel('Forward workers')
    ax2.set_ylabel('Peak GPU memory (GiB)')
    ax2.set_title('GPU memory (x: out of memory)', fontsize=9, loc='left')
    text_block(fig, 0.33, [
        ('b', 'Worker sweep (worker_sweep.py, one A40 + 16 cores): the GPU is busy 94-99 % from '
              '4-8 workers on, so more workers do not shorten a pass (+-5 % from 8 to 32). Each '
              'worker holds 2.3-2.8 GiB of GPU memory; 24 and 32 workers ran out of the '
              '48 GB on Aletsch (Rhone at the limit). Host memory 11 GiB at 8 workers (limit 54). -> 8 workers stay.'),
        ('b', 'What is left is GPU compute per member (Rhone ~3.7 s in parallel); further speed '
              'would need less retraining or several members per GPU call. Not pursued.'),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--default_job', type=int, default=JOBS['igm_default'],
                        help='Slurm array job of igm330_default')
    parser.add_argument('--credit_job', type=int, default=JOBS['credit'],
                        help='Slurm array job of igm330_credit')
    parser.add_argument('--live_job', type=int, default=JOBS['live'],
                        help='Slurm array job of igm330_live')
    args = parser.parse_args()
    jobs = {'t1': T1_JOB, 'igm_default': args.default_job, 'credit': args.credit_job,
            'live': args.live_job}

    fwd, series = forward_table()
    cal = calibration_table(jobs)
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    fwd.to_csv(os.path.join(HERE, 'tables', 'forward.tsv'), sep='\t', index=False,
               float_format='%.4g')
    cal.to_csv(os.path.join(HERE, 'tables', 'calibration.tsv'), sep='\t', index=False,
               float_format='%.4g')
    print(fwd.to_string(index=False, float_format='%.3g'))
    print(cal.groupby('rule')[['minutes', 'misfit', 'glamos_rmse']].agg(['sum', 'mean', 'max'])
          .to_string(float_format='%.3g'))

    theme()
    pages = [('Summary', lambda f: draw_summary(f, fwd, cal, jobs)),
             ('Forward test', lambda f: draw_forward_series(f, series)),
             ('Forward test', lambda f: draw_forward_cost(f, fwd)),
             ('Calibration', lambda f: draw_calibration(f, cal)),
             ('Calibration', lambda f: draw_fit(f, cal)),
             ('GLAMOS/WGMS', lambda f: draw_glamos(f)),
             ('Live model', lambda f: draw_live(f, cal))]
    path = os.path.join(HERE, 'igm330_retrain_report.pdf')
    with PdfPages(path) as pdf:
        for n, (topic, draw) in enumerate(pages, start=1):
            page(pdf, n, len(pages), topic, draw)
    print(f'Report: {path}')


if __name__ == '__main__':
    main()
