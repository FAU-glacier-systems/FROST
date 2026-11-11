#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
README diagrams of FROST v2: the calibration workflow (assets/pipeline*.png)
and the code and file layout (assets/architecture*.png), each in a light and
a dark version.

Run from the repository root:
    python tools/readme_diagrams.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from frost.visualization import plot_style  # noqa: E402

THEMES = {
    '': dict(bg='#FFFFFF', fg='#111111', muted='#555555', line='#777777',
             blue='#1F6FB2', red='#C62F35', yellow='#B58900', green='#2A8A5A',
             grey='#888888', fill=0.08),
    '_dark': dict(bg='#0A0A0A', fg='#EDEDED', muted='#BBBBBB', line='#9A9A9A',
                  blue=plot_style.COLORS['blue'], red=plot_style.COLORS['red'],
                  yellow=plot_style.COLORS['yellow'], green=plot_style.COLORS['green'],
                  grey='#9A9A9A', fill=0.14),
}


class Canvas:
    """Axes in box units (x 0..W, y 0..H) with boxes, text and arrows."""

    def __init__(self, c, width, height, scale=0.1):
        self.c = c
        self.fig = plt.figure(figsize=(width * scale, height * scale), facecolor=c['bg'])
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis('off')

    @staticmethod
    def height(title, lines, h=None):
        """Box height for a title and lines (or h if given)."""
        if h is not None:
            return h
        return 1.5 + (2.9 if title else 0) + 2.45 * len(lines) + (0.5 if lines else 0.3)

    def box(self, x, y, w, color, title=None, lines=(), h=None, dashed=False, size=9.5,
            mono_title=False, fill=True):
        """Rounded box with its top-left corner at (x, y); title in bold, then
        the lines (lines in backticks in monospace). Returns the bottom y."""
        c = self.c
        h = self.height(title, lines, h)
        patch = FancyBboxPatch((x, y - h), w, h, boxstyle='round,pad=0,rounding_size=1.2',
                               linewidth=1.6, edgecolor=c[color],
                               facecolor=matplotlib.colors.to_rgba(c[color], c['fill'] if fill else 0.0),
                               linestyle='--' if dashed else '-', zorder=2)
        self.ax.add_patch(patch)
        ty = y - 1.5
        if title:
            self.ax.text(x + 1.4, ty, title, fontsize=size + 0.5, weight='bold', va='top',
                         color=c['fg'], family='monospace' if mono_title else None, zorder=3)
            ty -= 2.9
        for line in lines:
            mono = line.startswith('`')
            self.ax.text(x + 1.4, ty, line.strip('`'), fontsize=size - (0.7 if mono else 0),
                         va='top', color=c['fg'], family='monospace' if mono else None,
                         zorder=3)
            ty -= 2.45
        return y - h

    def text(self, x, y, s, size=9, color='muted', **kw):
        self.ax.text(x, y, s, fontsize=size, color=self.c[color], zorder=3, **kw)

    def arrow(self, start, end, color='line', dashed=False, rad=0.0):
        self.ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle='-|>', mutation_scale=12, lw=1.4, color=self.c[color],
            linestyle='--' if dashed else '-', connectionstyle=f'arc3,rad={rad}', zorder=4,
            shrinkA=0, shrinkB=0))

    def save(self, path, bottom=0.0):
        # crop the empty part below `bottom`
        width, height = self.ax.get_xlim()[1], self.ax.get_ylim()[1]
        self.ax.set_ylim(bottom, height)
        self.fig.set_size_inches(width * 0.1, (height - bottom) * 0.1)
        self.fig.savefig(path, dpi=150, facecolor=self.c['bg'])
        plt.close(self.fig)
        print(f'Wrote {path}')


def pipeline(c, path):
    d = Canvas(c, 118, 72)
    d.text(2, 70, 'FROST v2: calibration of one glacier', size=12.5, color='fg',
           weight='bold', va='top')
    d.text(2, 66.4, 'python frost_pipeline.py --config experiments/<experiment>/pipeline_config.yml',
           size=8.5, family='monospace', va='top')

    # left column: preparation
    lx, lw = 2, 40
    b1 = d.box(lx, 61, lw, 'blue', '1  Download', [
        'RGI 7.0 outline, NASADEM, W5E5 climate (OGGM)',
        'Ice thickness: IceBoost v2.0',
        'Surface velocity: ITS_LIVE 2014-2022',
        'Elevation change: Hugonnet et al., 2000-2020'])
    t2 = b1 - 3.5
    b2 = d.box(lx, t2, lw, 'blue', '2  Inversion (IGM)', [
        'Thickness fixed: IceBoost moved to 2000',
        'Sliding (tau_ref) fitted to the velocity',
        'Glaciers below 5 m/yr: no ice flow'])
    t3 = b2 - 3.5
    b3 = d.box(lx, t3, lw, 'red', '3  Observations', [
        'dh/dt in 50 m elevation bands',
        'Error covariance from the spatial',
        'error correlation, plus a model error'])
    d.arrow((lx + lw / 2, b1), (lx + lw / 2, t2))
    d.arrow((lx + lw / 2, b2), (lx + lw / 2, t3))

    # right: ES-MDA loop (forward runs top left, parameters top right)
    cx, cw, ct = 50, 66, 61
    ix1, ix2, iw = cx + 3, cx + 35, 28
    loop_top = ct - 6.5
    fb = d.box(ix1, loop_top, iw, 'blue', 'Forward runs 2000-2020', [
        'IGM 3D ice flow, emulator', 'retrained on the glacier', 'or no flow: dh/dt = SMB'],
        size=8.8)
    d.box(ix2, loop_top, iw, 'yellow', 'SMB parameters', [
        'ELA model: ELA, 2 gradients', 'or temperature index: melt,', 'precipitation, temp. bias'],
        size=8.8)
    low_top = fb - 5
    mb = d.box(ix1, low_top, iw, 'red', 'Modelled dh/dt', [
        'band means of every member'], size=8.8)
    d.box(ix2, low_top, iw, 'yellow', 'Kalman update', [
        'observation error x iterations'], size=8.8)
    cb = mb - 7
    d.box(cx, ct, cw, 'yellow', '4  ES-MDA: 32 members, 4 iterations, then the posterior run',
          h=ct - cb, fill=True)
    ymid_top = loop_top - 7
    ymid_low = low_top - 4
    d.arrow((ix2, ymid_top), (ix1 + iw, ymid_top))                 # parameters -> runs
    d.arrow((ix1 + iw / 2, fb), (ix1 + iw / 2, low_top))           # runs -> dh/dt
    d.arrow((ix1 + iw, ymid_low), (ix2, ymid_low))                 # dh/dt -> update
    d.arrow((ix2 + iw / 2, low_top), (ix2 + iw / 2, fb))            # update -> parameters
    # inputs: geometry and sliding into the runs; observations along a line
    # below the loop and up into the update
    d.arrow((lx + lw, (t2 + b2) / 2), (ix1, ymid_top - 2))
    y_obs = mb - 4.2
    ox = ix2 + 6
    d.ax.plot([lx + lw, ox], [y_obs, y_obs], color=c['line'], lw=1.4, zorder=4)
    d.arrow((ox, y_obs), (ox, mb))
    d.text(ix1, y_obs + 0.5, 'observations (3)', size=8.3, va='bottom')
    d.text(lx + lw + 1.0, (t2 + b2) / 2 - 1.2, 'geometry,', size=8.3, va='top')
    d.text(lx + lw + 1.0, (t2 + b2) / 2 - 3.5, 'sliding (2)', size=8.3, va='top')

    # outputs
    rt = cb - 3.5
    rb = d.box(cx, rt, 33, 'green', 'Results', [
        '`calibration_results.json', '`Monitor/status.png, maps_prognostic.png'], size=8.8)
    d.box(cx + 36, rt, 30, 'grey', 'Evaluation only', [
        'GLAMOS / WGMS mass balance', 'GlaThiDa ice thickness'], size=8.8, dashed=True,
          fill=False)
    d.arrow((cx + 16.5, cb), (cx + 16.5, rt))
    d.arrow((cx + 33, (rt + rb) / 2), (cx + 36, (rt + rb) / 2), dashed=True)
    d.save(path, bottom=min(rb, b3) - 2)


def architecture(c, path):
    d = Canvas(c, 124, 96)
    d.text(2, 94, 'FROST v2: code and files', size=12.5, color='fg', weight='bold', va='top')
    d.text(2, 90.4, 'Files of every glacier in data/results/<experiment>/<rgi_id>/',
           size=8.5, family='monospace', va='top')

    x, w = 42, 42          # code
    fx, fw = 92, 30        # files
    gap = 3
    rows = []              # (top, bottom, files, color)

    def module(top, color, title, lines, files, xx=x, ww=w, size=8.6):
        bottom = d.box(xx, top, ww, color, title, lines, size=size, mono_title=True)
        if files:
            fbottom = d.box(fx, top, fw, color, None, files, size=8.6, fill=False)
            ymid = top - min(top - bottom, top - fbottom) / 2
            d.arrow((xx + ww, ymid), (fx, ymid))
        return bottom

    t = 84
    b = module(t, 'blue', 'preprocess/download_data.py', ['OGGM shop and data products'],
               ['`Preprocess/data/input.nc', '`climate_historical.nc'])
    rows.append((t, b))
    t = b - gap
    b = module(t, 'blue', 'preprocess/igm_inversion.py',
               ['sliding inversion with IGM', '(no_flow_setup.py without ice flow)'],
               ['`Preprocess/outputs/output.nc', '`Preprocess/outputs/emulator.keras'])
    rows.append((t, b))
    t = b - gap
    b = module(t, 'red', 'preprocess/create_observation.py', ['dh/dt on the model grid'],
               ['`observations.nc'])
    rows.append((t, b))
    for (t0, b0), (t1, _) in zip(rows[:-1], rows[1:]):
        d.arrow((x + w / 2, b0), (x + w / 2, t1))

    # calibration container
    ct = b - gap
    it = ct - 7
    ix, iw = x + 2, w - 4
    ib = module(it, 'red', 'calibration/observation_provider.py',
                ['band means and their covariance'], None, ix, iw, 8.2)
    it2 = ib - 1.8
    ib = module(it2, 'yellow', 'calibration/ensemble_kalman_filter.py',
                ['prior, ES-MDA update, forward runs'], ['`calibration_results.json'],
                ix, iw, 8.2)
    it3 = ib - 1.8
    ib = module(it3, 'blue', 'glacier_model/igm_wrapper.py',
                ['IGM run of one member (no_flow.py', 'without ice flow)'],
                ['`Ensemble/Member_<i>/'], ix, iw, 8.2)
    it4 = ib - 1.8
    ib = module(it4, 'green', 'visualization/monitor.py', ['plots of every iteration'],
                ['`Monitor/status.png', '`Monitor/maps_prognostic.png'], ix, iw, 8.2)
    cb = ib - 2
    d.box(x, ct, w, 'yellow', 'frost_calibration.py', ['ES-MDA loop'], h=ct - cb,
          size=8.6, mono_title=True)
    d.arrow((x + w / 2, b), (x + w / 2, ct))

    # configuration and driver, left
    cfg_b = d.box(2, 84, 32, 'grey', 'experiments/<experiment>/', [
        '`pipeline_config.yml', '`params_inversion.yaml'], size=8.6, mono_title=True,
        fill=False)
    pt = cfg_b - 6
    pb = d.box(2, pt, 32, 'grey', 'frost_pipeline.py', [
        'download, inversion, calibrate', 'Slurm: hpc_sbatch.sh'], size=8.6, mono_title=True)
    d.arrow((18, cfg_b), (18, pt))
    ymid = (pt + pb) / 2
    for (t0, b0) in rows:
        d.arrow((34, ymid), (x, (t0 + b0) / 2), rad=0.0)
    d.arrow((34, ymid), (x, ct - 3.5))
    d.save(path, bottom=cb - 2)


def main():
    plot_style.use_font()
    plt.rcParams['font.monospace'] = ['JetBrains Mono', 'DejaVu Sans Mono']
    for suffix, c in THEMES.items():
        pipeline(c, os.path.join(ROOT, 'assets', f'pipeline{suffix}.png'))
        architecture(c, os.path.join(ROOT, 'assets', f'architecture{suffix}.png'))


if __name__ == '__main__':
    main()
