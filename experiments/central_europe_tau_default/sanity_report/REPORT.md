# Sanity report: data/results/central_europe_tau_default

The full report is report.pdf (A4). Flags:

- **thk_bias_glathida_m**: thickness off GlaThiDa by > 30 m on average
- **speed_ratio_obs**: forward speed in the observation year off the observed by > 33 %
- **speed_start_ratio**: forward speed of 2000 > 2x that of the observation year (start-up shock)
- **dhdt_chi2_data**: modelled band dh/dt outside the data error (chi2/n > 4)
- **dead_bands**: bands that melt out in every member
- **ela_ref_capped**: reference ELA at the glacier top in > 25 % of the years (a lower bound, left out of the ELA scores)

| Glacier | Thk bias | Speed obs / fwd 2000 / fwd obs. year / fwd 2020 | dh/dt obs / model | chi2 data | Melted bands | Flags | Figures |
|---|---|---|---|---|---|---|---|
| Grosser_Aletschgletscher | -10 | 33.5 / 106.1 / 27.6 / 26.5 | -1.30 / -2.07 | 30.0 | 0 | speed_start_ratio, dhdt_chi2_data | [glacier](Grosser_Aletschgletscher/glacier.png) |
| Pasterze | +50 | 4.6 / 28.0 / 13.3 / 12.5 | -1.28 / -1.57 | 1.7 | 1 | thk_bias_glathida_m, speed_ratio_obs, speed_start_ratio, dead_bands | [glacier](Pasterze/glacier.png) |
| Glacier_de_Corbassière | -2 | 7.0 / 48.4 / 19.2 / 18.7 | -0.88 / -1.04 | 1.1 | 1 | speed_ratio_obs, speed_start_ratio, dead_bands | [glacier](Glacier_de_Corbassière/glacier.png) |
| Rhonegletscher | +2 | 10.9 / 44.7 / 24.6 / 23.9 | -1.21 / -1.33 | 2.0 | 0 | speed_ratio_obs | [glacier](Rhonegletscher/glacier.png) |
| Vadret_Pers | +6 | 10.8 / 67.7 / 29.6 / 28.9 | -1.09 / -1.24 | 0.7 | 0 | speed_ratio_obs, speed_start_ratio | [glacier](Vadret_Pers/glacier.png) |
| Findelgletscher | +3 | 9.9 / 21.7 / 14.6 / 14.2 | -0.70 / -0.95 | 2.7 | 1 | speed_ratio_obs, dead_bands | [glacier](Findelgletscher/glacier.png) |
| Allalingletscher | -6 | 7.1 / 30.4 / 16.4 / 16.2 | -0.61 / -0.69 | 1.7 | 0 | speed_ratio_obs | [glacier](Allalingletscher/glacier.png) |
| Vernagtferner | +37 | 0.7 / 17.5 / 7.1 / 6.7 | -1.05 / -1.07 | 4.3 | 0 | thk_bias_glathida_m, speed_ratio_obs, speed_start_ratio, dhdt_chi2_data | [glacier](Vernagtferner/glacier.png) |
| Hintereisferner | +23 | 1.0 / 2.6 / 1.9 / 1.9 | -1.21 / -1.10 | 1.7 | 0 | speed_ratio_obs | [glacier](Hintereisferner/glacier.png) |
| Glacier_de_la_Plaine_Morte | +51 | 1.8 / 6.2 / 2.4 / 2.1 | -1.49 / -2.03 | 4.0 | 0 | thk_bias_glathida_m, speed_ratio_obs, speed_start_ratio, dhdt_chi2_data, ela_ref_capped | [glacier](Glacier_de_la_Plaine_Morte/glacier.png) |
| Ghiacciaio_di_Malevalle | -18 | 0.5 / 6.5 / 2.0 / 1.8 | -1.24 / -1.36 | 0.7 | 0 | speed_ratio_obs, speed_start_ratio | [glacier](Ghiacciaio_di_Malevalle/glacier.png) |
| Griesgletscher | +4 | 1.7 / 14.3 / 5.6 / 5.4 | -1.38 / -1.31 | 0.2 | 0 | speed_ratio_obs, speed_start_ratio, ela_ref_capped | [glacier](Griesgletscher/glacier.png) |
| Glacier_du_Giétro | -3 | 2.4 / 13.9 / 7.3 / 7.3 | -0.85 / -0.77 | 0.5 | 0 | speed_ratio_obs | [glacier](Glacier_du_Giétro/glacier.png) |
| Schwarzberggletscher | -2 | 2.3 / 9.3 / 5.7 / 5.5 | -0.97 / -0.92 | 0.3 | 0 | speed_ratio_obs | [glacier](Schwarzberggletscher/glacier.png) |
| Kesselwandferner | +9 | 2.8 / 16.7 / 6.5 / 6.2 | -0.59 / -0.91 | 5.8 | 0 | speed_ratio_obs, speed_start_ratio, dhdt_chi2_data | [glacier](Kesselwandferner/glacier.png) |
| Jamtalferner | -3 | 0.4 / 18.1 / 4.2 / 4.4 | -1.18 / -1.44 | 0.4 | 0 | speed_ratio_obs, speed_start_ratio, ela_ref_capped | [glacier](Jamtalferner/glacier.png) |
| Zettalunitzkees | -1 | 0.8 / 8.0 / 4.0 / 4.0 | -0.75 / -0.79 | 0.4 | 0 | speed_ratio_obs | [glacier](Zettalunitzkees/glacier.png) |
| Hallstätter_Gletscher | -25 | 1.4 / 4.9 / 1.5 / 1.3 | -0.50 / -0.81 | 0.1 | 0 | speed_start_ratio | [glacier](Hallstätter_Gletscher/glacier.png) |
| Glacier_de_Tsanfleuron | -8 | 1.9 / 10.6 / 2.8 / 2.7 | -1.46 / -1.67 | 1.3 | 0 | speed_ratio_obs, speed_start_ratio, ela_ref_capped | [glacier](Glacier_de_Tsanfleuron/glacier.png) |
| Silvrettagletscher | +7 | 1.1 / 14.9 / 7.6 / 7.2 | -1.06 / -1.09 | 0.3 | 0 | speed_ratio_obs, ela_ref_capped | [glacier](Silvrettagletscher/glacier.png) |
| Careser | +nan | 0.4 / 0.6 / 0.4 / 0.4 | -1.36 / -1.74 | 1.0 | 0 | ela_ref_capped | [glacier](Careser/glacier.png) |
| Claridenfirn | -42 | 1.4 / 9.8 / 3.1 / 2.8 | -0.90 / -0.99 | 0.1 | 0 | thk_bias_glathida_m, speed_ratio_obs, speed_start_ratio | [glacier](Claridenfirn/glacier.png) |
| Vedretta_Lunga | -19 | 0.5 / 2.1 / 1.6 / 1.6 | -1.06 / -1.14 | 0.3 | 0 | speed_ratio_obs, ela_ref_capped | [glacier](Vedretta_Lunga/glacier.png) |
| Adlergletscher | -25 | 0.7 / 12.7 / 5.8 / 5.7 | -0.45 / -0.61 | 0.9 | 0 | speed_ratio_obs, speed_start_ratio | [glacier](Adlergletscher/glacier.png) |
| Ghiacciaio_del_Basòdino | +6 | 0.8 / 6.7 / 1.7 / 1.5 | -0.95 / -0.99 | 0.0 | 0 | speed_ratio_obs, speed_start_ratio, ela_ref_capped | [glacier](Ghiacciaio_del_Basòdino/glacier.png) |
