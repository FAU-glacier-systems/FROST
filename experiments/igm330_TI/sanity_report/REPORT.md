# Sanity report: data/results/igm330_TI

The full report is report.pdf (A4). Flags:

- **speed_ratio_obs**: forward speed in the observation year off the observed by > 33 %
- **speed_start_ratio**: forward speed of 2000 > 2x that of the observation year (start-up shock)
- **dhdt_chi2_data**: modelled band dh/dt outside the data error (chi2/n > 4)
- **dead_bands**: bands that melt out in every member
- **ela_ref_capped**: reference ELA at the glacier top in > 25 % of the years (a lower bound, left out of the ELA scores)

| Glacier | Thk r | Speed obs / fwd 2000 / fwd obs. year / fwd 2020 | dh/dt obs / model | chi2 data | Melted bands | Flags | Figures |
|---|---|---|---|---|---|---|---|
| Grosser_Aletschgletscher | 0.88 | 33.5 / 44.7 / 27.4 / 26.2 | -1.30 / -1.51 | 6.6 | 2 | dhdt_chi2_data, dead_bands | [glacier](Grosser_Aletschgletscher/glacier.png) |
| Pasterze | 0.67 | 4.7 / 4.4 / 3.3 / 3.2 | -1.28 / -1.12 | 0.7 | 0 | speed_ratio_obs | [glacier](Pasterze/glacier.png) |
| Glacier_de_Corbassière | 0.87 | 7.0 / 12.2 / 9.6 / 9.3 | -0.88 / -0.71 | 1.0 | 3 | speed_ratio_obs, dead_bands | [glacier](Glacier_de_Corbassière/glacier.png) |
| Rhonegletscher | 0.89 | 10.9 / 20.8 / 12.8 / 12.0 | -1.21 / -1.18 | 1.7 | 0 | - | [glacier](Rhonegletscher/glacier.png) |
| Vadret_Pers | 0.89 | 10.8 / 19.5 / 16.4 / 15.7 | -1.09 / -0.74 | 0.7 | 0 | speed_ratio_obs | [glacier](Vadret_Pers/glacier.png) |
| Findelgletscher | 0.91 | 9.9 / 12.3 / 8.6 / 8.1 | -0.70 / -0.87 | 4.0 | 1 | dead_bands | [glacier](Findelgletscher/glacier.png) |
| Allalingletscher | 0.87 | 7.1 / 10.2 / 9.0 / 8.8 | -0.61 / -0.38 | 0.8 | 0 | - | [glacier](Allalingletscher/glacier.png) |
| Hintereisferner | nan | 1.0 / 3.1 / 2.4 / 2.3 | -1.21 / -0.95 | 2.1 | 0 | speed_ratio_obs | [glacier](Hintereisferner/glacier.png) |
| Glacier_de_la_Plaine_Morte | 0.72 | 1.8 / 3.5 / 1.4 / 1.3 | -1.49 / -1.60 | 1.2 | 0 | speed_start_ratio, ela_ref_capped | [glacier](Glacier_de_la_Plaine_Morte/glacier.png) |
| Griesgletscher | 0.57 | 1.7 / 4.1 / 2.0 / 1.8 | -1.38 / -1.23 | 0.7 | 0 | speed_start_ratio, ela_ref_capped | [glacier](Griesgletscher/glacier.png) |
| Glacier_du_Giétro | 0.88 | 2.4 / 3.6 / 2.7 / 2.6 | -0.85 / -0.78 | 0.5 | 1 | dead_bands | [glacier](Glacier_du_Giétro/glacier.png) |
| Schwarzberggletscher | 0.90 | 2.4 / 4.0 / 4.0 / 3.9 | -0.97 / -0.79 | 0.7 | 0 | speed_ratio_obs | [glacier](Schwarzberggletscher/glacier.png) |
| Kesselwandferner | 0.59 | 2.8 / 3.6 / 2.7 / 2.5 | -0.59 / -0.45 | 1.8 | 0 | - | [glacier](Kesselwandferner/glacier.png) |
| Hallstätter_Gletscher | 0.45 | 1.4 / 1.9 / 0.8 / 0.7 | -0.50 / -0.60 | 0.1 | 0 | speed_ratio_obs, speed_start_ratio | [glacier](Hallstätter_Gletscher/glacier.png) |
| Glacier_de_Tsanfleuron | 0.86 | 1.9 / 4.5 / 1.8 / 1.6 | -1.46 / -1.48 | 1.3 | 0 | speed_start_ratio, ela_ref_capped | [glacier](Glacier_de_Tsanfleuron/glacier.png) |
| Silvrettagletscher | 0.68 | 1.1 / 3.9 / 2.5 / 2.3 | -1.06 / -0.90 | 0.5 | 0 | speed_ratio_obs, ela_ref_capped | [glacier](Silvrettagletscher/glacier.png) |
| Claridenfirn | 0.62 | 1.4 / 3.8 / 1.9 / 1.8 | -0.90 / -0.94 | 0.3 | 0 | speed_ratio_obs | [glacier](Claridenfirn/glacier.png) |
| Vernagtferner | -0.26 | 0.7 / nan / nan / nan | -1.05 / -0.95 | 1.7 | 0 | - | [glacier](Vernagtferner/glacier.png) |
| Ghiacciaio_di_Malevalle | 0.75 | 0.5 / nan / nan / nan | -1.24 / -1.16 | 1.0 | 1 | dead_bands | [glacier](Ghiacciaio_di_Malevalle/glacier.png) |
| Jamtalferner | 0.40 | 0.4 / nan / nan / nan | -1.18 / -1.23 | 1.2 | 0 | ela_ref_capped | [glacier](Jamtalferner/glacier.png) |
| Zettalunitzkees | 0.09 | 0.8 / nan / nan / nan | -0.75 / -1.06 | 1.3 | 0 | - | [glacier](Zettalunitzkees/glacier.png) |
| Careser | nan | 0.4 / nan / nan / nan | -1.36 / -1.63 | 1.2 | 0 | ela_ref_capped | [glacier](Careser/glacier.png) |
| Vedretta_Lunga | 0.63 | 0.5 / nan / nan / nan | -1.06 / -1.19 | 0.7 | 1 | dead_bands, ela_ref_capped | [glacier](Vedretta_Lunga/glacier.png) |
| Adlergletscher | 0.50 | 0.7 / nan / nan / nan | -0.45 / -0.22 | 0.7 | 0 | - | [glacier](Adlergletscher/glacier.png) |
| Ghiacciaio_del_Basòdino | 0.59 | 0.8 / nan / nan / nan | -0.95 / -1.12 | 0.1 | 0 | ela_ref_capped | [glacier](Ghiacciaio_del_Basòdino/glacier.png) |
