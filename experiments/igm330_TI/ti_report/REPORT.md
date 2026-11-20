# TI calibration report: data/results/igm330_TI

The full report is ti_report.pdf (A4); numbers in ti_summary.tsv. Flags:

- **prior_shift**: a posterior mean more than 2 prior std from OGGM's calibration
- **spread_ratio_min**: posterior spread of a parameter below 15 % of the prior (possible ensemble collapse)
- **last_change**: a mean still moved by > 0.5 posterior std in the last update
- **misfit_posterior**: band dh/dt of the posterior outside the data and model error (misfit chi2 / n > 1)
- **unphysical**: members with a melt factor or precipitation factor <= 0
- **gw_r**: year-to-year glacier-wide balance not reproduced (r < 0.5)
- **gw_bias**: glacier-wide annual balance off by > 0.3 m w.e./yr
- **winter_bias**: winter balance (bands) off by > 0.5 m w.e.
- **summer_bias**: summer balance (bands) off by > 0.5 m w.e.
- **dhdt_chi2_data**: modelled band dh/dt outside the data error (chi2/n > 4)

| Glacier | Flow | melt_f | prcp_fac | temp_bias | gw r | gw RMSE | winter / summer bias | profile RMSE | dh/dt chi2 | Flags |
|---|---|---|---|---|---|---|---|---|---|---|
| Grosser_Aletschgletscher | igm | 3.11 | 2.20 | +2.40 | 0.69 | 0.42 | -0.50 / +1.72 | 2.73 | 6.6 | spread_ratio_min, misfit_posterior, summer_bias, dhdt_chi2_data |
| Pasterze | igm | 3.39 | 1.54 | +2.06 | 0.67 | 0.29 | +nan / +nan | 0.84 | 0.7 | - |
| Glacier_de_Corbassière | igm | 3.37 | 1.53 | -0.29 | 0.76 | 0.30 | -0.15 / +0.49 | 0.60 | 1.0 | - |
| Rhonegletscher | igm | 4.38 | 1.07 | +0.91 | 0.67 | 0.45 | -0.23 / +0.26 | 0.70 | 1.7 | - |
| Vadret_Pers | igm | 3.06 | 1.48 | +2.62 | 0.67 | 0.53 | -0.21 / +0.93 | 1.31 | 0.7 | gw_bias, summer_bias |
| Findelgletscher | igm | 4.98 | 2.66 | -0.56 | 0.58 | 0.40 | +0.21 / +0.32 | 0.94 | 4.0 | misfit_posterior |
| Allalingletscher | igm | 3.64 | 1.39 | +1.40 | 0.75 | 0.45 | -0.18 / +0.41 | 0.37 | 0.8 | - |
| Hintereisferner | igm | 3.43 | 3.57 | +2.70 | 0.85 | 0.46 | +0.03 / +0.81 | 1.27 | 2.1 | last_change, misfit_posterior, gw_bias, summer_bias |
| Glacier_de_la_Plaine_Morte | igm | 3.55 | 3.18 | +1.16 | 0.82 | 0.41 | -0.01 / -0.44 | 0.69 | 1.2 | - |
| Griesgletscher | igm | 3.66 | 1.68 | +1.49 | 0.84 | 0.41 | -0.18 / +0.55 | 0.61 | 0.7 | summer_bias |
| Glacier_du_Giétro | igm | 4.30 | 1.92 | -0.21 | 0.70 | 0.45 | -0.07 / +0.62 | 0.88 | 0.5 | summer_bias |
| Schwarzberggletscher | igm | 4.19 | 2.93 | +1.49 | 0.84 | 0.34 | +0.29 / -0.08 | 0.50 | 0.7 | - |
| Kesselwandferner | igm | 3.92 | 2.48 | +1.93 | 0.88 | 0.32 | +nan / +nan | 1.25 | 1.8 | - |
| Hallstätter_Gletscher | igm | 3.87 | 3.13 | -0.26 | 0.83 | 0.40 | -0.05 / +0.63 | 0.83 | 0.1 | summer_bias |
| Glacier_de_Tsanfleuron | igm | 4.45 | 3.06 | +2.68 | 0.95 | 0.36 | -0.06 / +0.30 | 0.29 | 1.3 | - |
| Silvrettagletscher | igm | 4.28 | 3.56 | +1.74 | 0.75 | 0.44 | -0.42 / +0.53 | 0.35 | 0.5 | summer_bias |
| Claridenfirn | igm | 5.37 | 1.57 | +0.49 | 0.65 | 0.57 | +0.03 / +0.57 | 2.35 | 0.3 | gw_bias, summer_bias |
| Vernagtferner | none | 3.53 | 2.19 | +2.58 | 0.81 | 0.29 | -0.17 / +0.47 | 0.59 | 1.7 | - |
| Ghiacciaio_di_Malevalle | none | 3.39 | 2.04 | +1.88 | 0.91 | 0.29 | -0.37 / +0.23 | 0.27 | 1.0 | - |
| Jamtalferner | none | 3.86 | 3.40 | +2.45 | 0.75 | 0.45 | -0.04 / +0.17 | 0.42 | 1.2 | - |
| Zettalunitzkees | none | 5.96 | 1.89 | -2.24 | 0.75 | 0.57 | -0.12 / -0.30 | 0.60 | 1.3 | gw_bias |
| Careser | none | 5.84 | 2.74 | +3.55 | 0.81 | 0.44 | +0.43 / -0.11 | 0.35 | 1.2 | - |
| Vedretta_Lunga | none | 5.76 | 2.39 | +3.48 | 0.86 | 0.52 | +0.13 / -0.48 | 0.53 | 0.7 | gw_bias |
| Adlergletscher | none | 4.78 | 1.44 | +1.29 | 0.67 | 0.38 | +0.16 / +0.27 | 0.62 | 0.7 | last_change |
| Ghiacciaio_del_Basòdino | none | 6.16 | 2.36 | +1.11 | 0.83 | 1.20 | +0.38 / -1.61 | 1.71 | 0.1 | gw_bias, summer_bias |
