import glob, json, os, warnings
import numpy as np
from netCDF4 import Dataset
warnings.filterwarnings('ignore')

R = 'data/results/continuix/EXP01'
rows = []
for g in ['G02', 'G03', 'G04', 'G05', 'G06', 'S01', 'S02']:
    d = os.path.join(R, g)
    meta = json.load(open(os.path.join(d, 'continuix.json')))
    cal = json.load(open(os.path.join(d, 'calibration_results.json')))
    # --- inversion
    with Dataset(os.path.join(d, 'Preprocess/outputs/output.nc')) as nc:
        m = np.array(nc['icemask'][:]) > 0.5
        vm = np.array(nc['velsurf_mag'][:]); vo = np.array(nc['velsurfobs_mag'][:])
        tau = np.array(nc['tau_ref'][:]); thk = np.array(nc['thk'][:])
        vb = np.array(nc['velbase_mag'][:])
        cd = np.array(nc['da_cost_data_hist'][:]); cr = np.array(nc['da_cost_reg_hist'][:])
    ok = m & np.isfinite(vo) & (vo > 0)
    e = vm[ok] - vo[ok]
    inv = dict(v_obs=np.mean(vo[ok]), v_mod=np.mean(vm[ok]), v_rms=np.sqrt(np.mean(e**2)),
               v_bias=np.mean(e), v_r=np.corrcoef(vm[ok], vo[ok])[0, 1],
               v_rel_rms=np.sqrt(np.mean(e**2)) / np.mean(vo[ok]),
               slide_frac=np.mean(vb[m]) / max(np.mean(vm[m]), 1e-9),
               tau_p5=np.percentile(tau[m], 5), tau_p95=np.percentile(tau[m], 95),
               tau_at_bound=np.mean((tau[m] <= 0.0011) | (tau[m] >= 9.9)),
               thk_mean=np.mean(thk[m]), cost_drop=cd[-1] / cd[0] if cd[0] else np.nan)
    # --- calibration (posterior forward runs vs observed dh/dt)
    with Dataset(os.path.join(d, 'observations.nc')) as nc:
        dh = np.array(nc['dhdt'][:]).squeeze(); err = np.array(nc['dhdt_err'][:]).squeeze()
        us0 = np.array(nc['usurf'][:]); us0 = us0[0] if us0.ndim == 3 else us0
    if dh.ndim == 3: dh = dh[-1]
    if err.ndim == 3: err = err[-1]
    obs = np.isfinite(dh) & m
    bins = np.floor(us0 / 50) * 50
    ub = np.unique(bins[obs])
    mod_mean, mod_band = [], []
    for out in sorted(glob.glob(os.path.join(d, 'Ensemble/Member_*/outputs/output.nc'))):
        with Dataset(out) as nc:
            t = np.array(nc['time'][:]); us = np.array(nc['usurf'][:])
        r = (us[-1] - us[0]) / (t[-1] - t[0])
        mod_mean.append(np.mean(r[obs]))
        mod_band.append([np.mean(r[obs & (bins == b)]) for b in ub])
    mod_band = np.array(mod_band); obs_band = np.array([np.mean(dh[obs & (bins == b)]) for b in ub])
    area = np.array([np.sum(obs & (bins == b)) for b in ub]); w = area / area.sum()
    mb = mod_band.mean(0)
    band_rms = np.sqrt(np.sum(w * (mb - obs_band)**2))
    # sign of misfit vs elevation: low half vs high half (area-weighted)
    mid = np.searchsorted(np.cumsum(w), 0.5)
    lo = np.sum((w * (mb - obs_band))[:mid + 1]) / w[:mid + 1].sum()
    hi = np.sum((w * (mb - obs_band))[mid + 1:]) / max(w[mid + 1:].sum(), 1e-9)
    fe, fs = np.array(cal['final_mean']), np.array(cal['final_std'])
    ps = cal['initial_spread']
    cal_d = dict(period=f"{meta['year_start']}-{meta['year_end']}", n_ens=len(mod_mean),
                 dh_obs=np.mean(dh[obs]), dh_mod=np.mean(mod_mean), dh_spread=np.std(mod_mean),
                 dh_err=np.nanmean(err[obs]), band_rms=band_rms, band_obs_std=np.sqrt(np.sum(w*(obs_band-np.sum(w*obs_band))**2)),
                 misfit_low=lo, misfit_high=hi, cover=obs.sum() / m.sum(),
                 ela=fe[0], ela_sd=fs[0], abl=fe[1], abl_sd=fs[1], acc=fe[2], acc_sd=fs[2],
                 ela_prior=cal['initial_smb']['ela'], usurf_range=f"{meta['usurf_min']:.0f}-{meta['usurf_max']:.0f}",
                 sd_ratio=[fs[0] / ps['ela'], fs[1] / ps['abl_grad'], fs[2] / ps['acc_grad']])
    rows.append((g, inv, cal_d))
    print(g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in {**inv, **cal_d}.items()})
    print('   bands', [f'{b:.0f}:{o:+.2f}/{mm:+.2f}' for b, o, mm in zip(ub, obs_band, mb)][::max(1, len(ub)//12)])
