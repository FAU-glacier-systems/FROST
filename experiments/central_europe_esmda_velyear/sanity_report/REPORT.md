# Sanity report: data/results/central_europe_esmda_velyear

The full report is report.pdf (A4). Flags:

- **thk_bias_glathida_m**: thickness off GlaThiDa by > 30 m on average
- **speed_ratio_obs**: forward speed in the observation year off the observed by > 33 %
- **speed_start_ratio**: forward speed of 2000 > 2x that of the observation year (start-up shock)
- **dhdt_chi2_data**: modelled band dh/dt outside the data error (chi2/n > 4)
- **dead_bands**: bands that melt out in every member

| Glacier | Thk bias | Speed obs / fwd 2000 / fwd obs. year / fwd 2020 | dh/dt obs / model | chi2 data | Melted bands | Flags | Figures |
|---|---|---|---|---|---|---|---|
| Grosser_Aletschgletscher | -10 | 33.5 / 78.4 / 37.3 / 37.2 | -1.30 / -1.38 | 3.7 | 3 | speed_start_ratio, dead_bands | [glacier](Grosser_Aletschgletscher/glacier.png) |
| Pasterze | +50 | 4.7 / 42.3 / 6.5 / 6.4 | -1.28 / -1.25 | 1.8 | 1 | thk_bias_glathida_m, speed_ratio_obs, speed_start_ratio, dead_bands | [glacier](Pasterze/glacier.png) |
| Glacier_de_Corbassière | -2 | 7.0 / 11.2 / 7.9 / 7.8 | -0.88 / -0.74 | 1.1 | 2 | dead_bands | [glacier](Glacier_de_Corbassière/glacier.png) |
| Rhonegletscher | +2 | 10.9 / 16.1 / 9.5 / 9.1 | -1.21 / -1.14 | 3.8 | 0 | - | [glacier](Rhonegletscher/glacier.png) |
| Vadret_Pers | +6 | 10.8 / 29.9 / 12.5 / 12.0 | -1.09 / -0.85 | 2.6 | 0 | speed_start_ratio | [glacier](Vadret_Pers/glacier.png) |
| Findelgletscher | +3 | 9.9 / 12.9 / 7.3 / 7.1 | -0.70 / -0.81 | 4.1 | 0 | speed_ratio_obs, dhdt_chi2_data | [glacier](Findelgletscher/glacier.png) |
| Allalingletscher | -6 | 7.1 / 10.3 / 6.0 / 6.0 | -0.61 / -0.52 | 0.4 | 0 | - | [glacier](Allalingletscher/glacier.png) |
| Vernagtferner | +37 | 0.7 / 3.3 / 1.2 / 1.1 | -1.05 / -1.16 | 1.0 | 0 | thk_bias_glathida_m, speed_ratio_obs, speed_start_ratio | [glacier](Vernagtferner/glacier.png) |
| Hintereisferner | +23 | 1.0 / 3.1 / 1.4 / 1.3 | -1.21 / -1.14 | 1.8 | 0 | speed_ratio_obs, speed_start_ratio | [glacier](Hintereisferner/glacier.png) |
| Glacier_de_la_Plaine_Morte | +51 | 1.8 / 3.0 / 1.9 / 1.9 | -1.49 / -1.49 | 1.4 | 0 | thk_bias_glathida_m | [glacier](Glacier_de_la_Plaine_Morte/glacier.png) |
| Ghiacciaio_di_Malevalle | -18 | 0.5 / 3.0 / 0.9 / 0.9 | -1.24 / -1.31 | 0.3 | 0 | speed_ratio_obs, speed_start_ratio | [glacier](Ghiacciaio_di_Malevalle/glacier.png) |
| Griesgletscher | +4 | 1.7 / 4.5 / 2.3 / 2.2 | -1.38 / -1.35 | 0.1 | 0 | speed_ratio_obs | [glacier](Griesgletscher/glacier.png) |
| Schwarzberggletscher | -2 | 2.4 / 5.2 / 3.1 / 3.0 | -0.97 / -0.93 | 0.5 | 0 | - | [glacier](Schwarzberggletscher/glacier.png) |
| Kesselwandferner | +9 | 2.8 / 4.2 / 3.1 / 2.9 | -0.59 / -0.63 | 3.1 | 0 | - | [glacier](Kesselwandferner/glacier.png) |
| Jamtalferner | -3 | 0.4 / 6.3 / 22.9 / 28.0 | -1.18 / -1.42 | 0.9 | 0 | speed_ratio_obs | [glacier](Jamtalferner/glacier.png) |
| Zettalunitzkees | -1 | 0.8 / 2.0 / 1.0 / 1.0 | -0.75 / -0.77 | 0.5 | 0 | - | [glacier](Zettalunitzkees/glacier.png) |
| Hallstätter_Gletscher | -25 | 1.4 / 2.6 / 0.8 / 0.9 | -0.50 / -0.69 | 0.1 | 0 | speed_ratio_obs, speed_start_ratio | [glacier](Hallstätter_Gletscher/glacier.png) |
| Glacier_de_Tsanfleuron | -8 | 1.9 / 4.0 / 1.4 / 1.4 | -1.46 / -1.56 | 1.0 | 0 | speed_ratio_obs, speed_start_ratio | [glacier](Glacier_de_Tsanfleuron/glacier.png) |
| Silvrettagletscher | +7 | 1.1 / 2.8 / 1.2 / 1.2 | -1.06 / -1.09 | 0.3 | 0 | speed_start_ratio | [glacier](Silvrettagletscher/glacier.png) |
| Careser | +nan | 0.4 / 1.5 / 0.5 / 0.5 | -1.36 / -1.60 | 0.9 | 0 | speed_ratio_obs, speed_start_ratio | [glacier](Careser/glacier.png) |
| Claridenfirn | -42 | 1.4 / 4.0 / 1.9 / 1.8 | -0.90 / -0.99 | 0.1 | 0 | thk_bias_glathida_m, speed_start_ratio | [glacier](Claridenfirn/glacier.png) |
| Vedretta_Lunga | -19 | 0.5 / 2.1 / 0.8 / 0.7 | -1.06 / -1.17 | 0.3 | 0 | speed_ratio_obs, speed_start_ratio | [glacier](Vedretta_Lunga/glacier.png) |
| Adlergletscher | -25 | 0.7 / 2.5 / 1.6 / 1.6 | -0.45 / -0.33 | 0.4 | 0 | speed_ratio_obs | [glacier](Adlergletscher/glacier.png) |
| Ghiacciaio_del_Basòdino | +6 | 0.8 / 2.8 / 1.1 / 1.1 | -0.95 / -0.84 | 0.0 | 0 | speed_ratio_obs, speed_start_ratio | [glacier](Ghiacciaio_del_Basòdino/glacier.png) |
