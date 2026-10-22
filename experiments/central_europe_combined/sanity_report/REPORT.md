# Sanity report: data/results/central_europe_combined

The full report is report.pdf (A4). Flags:

- **thk_bias_glathida_m**: thickness off GlaThiDa by > 30 m on average
- **speed_ratio_obs**: forward speed in the observation year off the observed by > 33 %
- **speed_start_ratio**: forward speed of 2000 > 2x that of the observation year (start-up shock)
- **dhdt_chi2_data**: modelled band dh/dt outside the data error (chi2/n > 4)
- **dead_bands**: bands that melt out in every member
- **ela_ref_capped**: reference ELA at the glacier top in > 25 % of the years (a lower bound, left out of the ELA scores)

| Glacier | Thk bias | Speed obs / fwd 2000 / fwd obs. year / fwd 2020 | dh/dt obs / model | chi2 data | Melted bands | Flags | Figures |
|---|---|---|---|---|---|---|---|
| Grosser_Aletschgletscher | -10 | 33.5 / 78.4 / 37.3 / 37.2 | -1.30 / -1.38 | 3.7 | 3 | speed_start_ratio, dead_bands | [glacier](Grosser_Aletschgletscher/glacier.png) |
| Pasterze | +50 | 4.7 / 42.3 / 6.5 / 6.4 | -1.28 / -1.25 | 1.8 | 1 | thk_bias_glathida_m, speed_ratio_obs, speed_start_ratio, dead_bands | [glacier](Pasterze/glacier.png) |
| Glacier_de_Corbassière | -2 | 7.0 / 11.2 / 7.9 / 7.8 | -0.88 / -0.74 | 1.1 | 2 | dead_bands | [glacier](Glacier_de_Corbassière/glacier.png) |
| Rhonegletscher | +2 | 10.9 / 16.1 / 9.5 / 9.1 | -1.21 / -1.14 | 3.8 | 0 | - | [glacier](Rhonegletscher/glacier.png) |
| Vadret_Pers | +6 | 10.8 / 29.9 / 12.5 / 12.0 | -1.09 / -0.85 | 2.6 | 0 | speed_start_ratio | [glacier](Vadret_Pers/glacier.png) |
| Findelgletscher | +3 | 9.9 / 12.9 / 7.3 / 7.1 | -0.70 / -0.81 | 4.1 | 0 | speed_ratio_obs, dhdt_chi2_data | [glacier](Findelgletscher/glacier.png) |
| Allalingletscher | -6 | 7.1 / 10.3 / 6.0 / 6.0 | -0.61 / -0.52 | 0.4 | 0 | - | [glacier](Allalingletscher/glacier.png) |
| Vernagtferner | +37 | 0.7 / nan / nan / nan | -1.05 / -1.09 | 1.1 | 0 | thk_bias_glathida_m | [glacier](Vernagtferner/glacier.png) |
| Hintereisferner | +23 | 1.0 / nan / nan / nan | -1.21 / -1.11 | 2.0 | 0 | - | [glacier](Hintereisferner/glacier.png) |
| Glacier_de_la_Plaine_Morte | +51 | 1.8 / nan / nan / nan | -1.49 / -1.47 | 0.4 | 0 | thk_bias_glathida_m, ela_ref_capped | [glacier](Glacier_de_la_Plaine_Morte/glacier.png) |
| Ghiacciaio_di_Malevalle | -18 | 0.5 / nan / nan / nan | -1.24 / -1.37 | 0.6 | 0 | - | [glacier](Ghiacciaio_di_Malevalle/glacier.png) |
| Griesgletscher | +4 | 1.7 / nan / nan / nan | -1.38 / -1.55 | 0.5 | 0 | ela_ref_capped | [glacier](Griesgletscher/glacier.png) |
| Schwarzberggletscher | -2 | 2.3 / nan / nan / nan | -0.97 / -1.01 | 0.5 | 0 | - | [glacier](Schwarzberggletscher/glacier.png) |
| Kesselwandferner | +9 | 2.8 / 4.2 / 3.1 / 2.9 | -0.59 / -0.63 | 3.1 | 0 | - | [glacier](Kesselwandferner/glacier.png) |
| Jamtalferner | -3 | 0.4 / nan / nan / nan | -1.18 / -1.50 | 0.7 | 0 | ela_ref_capped | [glacier](Jamtalferner/glacier.png) |
| Zettalunitzkees | -1 | 0.8 / nan / nan / nan | -0.75 / -0.76 | 0.3 | 0 | - | [glacier](Zettalunitzkees/glacier.png) |
| Hallstätter_Gletscher | -25 | 1.4 / nan / nan / nan | -0.50 / -0.71 | 0.1 | 0 | - | [glacier](Hallstätter_Gletscher/glacier.png) |
| Glacier_de_Tsanfleuron | -8 | 1.9 / nan / nan / nan | -1.46 / -1.70 | 0.6 | 0 | ela_ref_capped | [glacier](Glacier_de_Tsanfleuron/glacier.png) |
| Silvrettagletscher | +7 | 1.1 / nan / nan / nan | -1.06 / -1.26 | 0.4 | 0 | ela_ref_capped | [glacier](Silvrettagletscher/glacier.png) |
| Careser | +nan | 0.4 / nan / nan / nan | -1.36 / -1.46 | 0.6 | 0 | ela_ref_capped | [glacier](Careser/glacier.png) |
| Claridenfirn | -42 | 1.4 / nan / nan / nan | -0.90 / -0.93 | 0.2 | 0 | thk_bias_glathida_m | [glacier](Claridenfirn/glacier.png) |
| Vedretta_Lunga | -19 | 0.5 / nan / nan / nan | -1.06 / -1.12 | 0.4 | 0 | ela_ref_capped | [glacier](Vedretta_Lunga/glacier.png) |
| Adlergletscher | -25 | 0.7 / nan / nan / nan | -0.45 / -0.17 | 0.2 | 0 | - | [glacier](Adlergletscher/glacier.png) |
| Ghiacciaio_del_Basòdino | +6 | 0.8 / nan / nan / nan | -0.95 / -0.91 | 0.0 | 0 | ela_ref_capped | [glacier](Ghiacciaio_del_Basòdino/glacier.png) |
