# Sanity report: data/results/igm330_credit

The full report is report.pdf (A4). Flags:

- **speed_ratio_obs**: forward speed in the observation year off the observed by > 33 %
- **speed_start_ratio**: forward speed of 2000 > 2x that of the observation year (start-up shock)
- **dhdt_chi2_data**: modelled band dh/dt outside the data error (chi2/n > 4)
- **dead_bands**: bands that melt out in every member
- **ela_ref_capped**: reference ELA at the glacier top in > 25 % of the years (a lower bound, left out of the ELA scores)

| Glacier | Thk r | Speed obs / fwd 2000 / fwd obs. year / fwd 2020 | dh/dt obs / model | chi2 data | Melted bands | Flags | Figures |
|---|---|---|---|---|---|---|---|
| Grosser_Aletschgletscher | 0.88 | 33.5 / 44.7 / 27.5 / 26.7 | -1.30 / -1.49 | 4.9 | 1 | dhdt_chi2_data, dead_bands | [glacier](Grosser_Aletschgletscher/glacier.png) |
| Pasterze | 0.67 | 4.7 / 4.4 / 3.0 / 2.9 | -1.28 / -1.24 | 0.6 | 0 | speed_ratio_obs | [glacier](Pasterze/glacier.png) |
| Glacier_de_Corbassière | 0.87 | 7.0 / 12.2 / 8.6 / 8.4 | -0.88 / -0.71 | 0.9 | 2 | dead_bands | [glacier](Glacier_de_Corbassière/glacier.png) |
| Rhonegletscher | 0.89 | 10.9 / 20.8 / 11.8 / 11.4 | -1.21 / -1.17 | 1.6 | 0 | - | [glacier](Rhonegletscher/glacier.png) |
| Vadret_Pers | 0.89 | 10.8 / 19.5 / 12.1 / 11.8 | -1.09 / -0.97 | 0.8 | 0 | - | [glacier](Vadret_Pers/glacier.png) |
| Findelgletscher | 0.91 | 9.9 / 12.3 / 8.8 / 8.5 | -0.70 / -0.82 | 3.3 | 0 | - | [glacier](Findelgletscher/glacier.png) |
| Allalingletscher | 0.87 | 7.1 / 10.2 / 7.2 / 7.2 | -0.61 / -0.54 | 0.6 | 0 | - | [glacier](Allalingletscher/glacier.png) |
| Kesselwandferner | 0.59 | 2.8 / 3.6 / 2.6 / 2.5 | -0.59 / -0.58 | 1.4 | 0 | - | [glacier](Kesselwandferner/glacier.png) |
| Glacier_du_Giétro | 0.88 | 2.4 / 3.6 / 2.5 / 2.4 | -0.85 / -0.67 | 0.3 | 0 | - | [glacier](Glacier_du_Giétro/glacier.png) |
| Schwarzberggletscher | 0.90 | 2.4 / 4.0 / 2.8 / 2.7 | -0.97 / -0.90 | 0.5 | 0 | - | [glacier](Schwarzberggletscher/glacier.png) |
| Griesgletscher | 0.57 | 1.7 / 4.1 / 1.9 / 1.7 | -1.38 / -1.31 | 0.2 | 0 | speed_start_ratio, ela_ref_capped | [glacier](Griesgletscher/glacier.png) |
| Hallstätter_Gletscher | 0.45 | 1.4 / 1.9 / 0.6 / 0.5 | -0.50 / -0.69 | 0.1 | 0 | speed_ratio_obs, speed_start_ratio | [glacier](Hallstätter_Gletscher/glacier.png) |
| Claridenfirn | 0.62 | 1.4 / 3.8 / 1.5 / 1.5 | -0.90 / -0.98 | 0.1 | 0 | speed_start_ratio | [glacier](Claridenfirn/glacier.png) |
| Hintereisferner | nan | 1.0 / 3.1 / 1.6 / 1.5 | -1.21 / -1.17 | 1.6 | 0 | speed_ratio_obs | [glacier](Hintereisferner/glacier.png) |
| Glacier_de_la_Plaine_Morte | 0.72 | 1.8 / 3.5 / 1.4 / 1.3 | -1.49 / -1.52 | 0.9 | 0 | speed_start_ratio, ela_ref_capped | [glacier](Glacier_de_la_Plaine_Morte/glacier.png) |
| Glacier_de_Tsanfleuron | 0.86 | 1.9 / 4.5 / 1.7 / 1.5 | -1.46 / -1.59 | 0.7 | 0 | speed_start_ratio, ela_ref_capped | [glacier](Glacier_de_Tsanfleuron/glacier.png) |
| Silvrettagletscher | 0.68 | 1.1 / 3.9 / 2.0 / 1.8 | -1.06 / -1.09 | 0.3 | 0 | speed_ratio_obs, speed_start_ratio, ela_ref_capped | [glacier](Silvrettagletscher/glacier.png) |
