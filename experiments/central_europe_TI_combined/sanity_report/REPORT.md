# Sanity report: data/results/central_europe_TI_combined

The full report is report.pdf (A4). Flags:

- **speed_ratio_obs**: forward speed in the observation year off the observed by > 33 %
- **speed_start_ratio**: forward speed of 2000 > 2x that of the observation year (start-up shock)
- **dhdt_chi2_data**: modelled band dh/dt outside the data error (chi2/n > 4)
- **dead_bands**: bands that melt out in every member
- **ela_ref_capped**: reference ELA at the glacier top in > 25 % of the years (a lower bound, left out of the ELA scores)

| Glacier | Thk r | Speed obs / fwd 2000 / fwd obs. year / fwd 2020 | dh/dt obs / model | chi2 data | Melted bands | Flags | Figures |
|---|---|---|---|---|---|---|---|
| Grosser_Aletschgletscher | 0.88 | 33.5 / 45.6 / 27.4 / 26.2 | -1.30 / -1.50 | 6.7 | 2 | dhdt_chi2_data, dead_bands | [glacier](Grosser_Aletschgletscher/glacier.png) |
| Pasterze | 0.67 | 4.7 / 8.9 / 6.1 / 5.8 | -1.28 / -1.18 | 1.1 | 1 | dead_bands | [glacier](Pasterze/glacier.png) |
| Glacier_de_Corbassière | 0.87 | 7.0 / 8.9 / 9.7 / 9.3 | -0.88 / -0.70 | 1.0 | 2 | speed_ratio_obs, dead_bands | [glacier](Glacier_de_Corbassière/glacier.png) |
| Rhonegletscher | 0.89 | 10.9 / 19.5 / 12.6 / 12.1 | -1.21 / -1.19 | 1.6 | 0 | - | [glacier](Rhonegletscher/glacier.png) |
| Vadret_Pers | 0.89 | 10.8 / 19.9 / 16.8 / 16.0 | -1.09 / -0.74 | 0.7 | 0 | speed_ratio_obs | [glacier](Vadret_Pers/glacier.png) |
| Findelgletscher | 0.91 | 9.9 / 12.4 / 8.7 / 8.2 | -0.70 / -0.87 | 3.9 | 1 | dead_bands | [glacier](Findelgletscher/glacier.png) |
| Allalingletscher | 0.87 | 7.1 / 9.9 / 9.0 / 9.0 | -0.61 / -0.37 | 0.7 | 0 | - | [glacier](Allalingletscher/glacier.png) |
| Vernagtferner | -0.26 | 0.7 / nan / nan / nan | -1.05 / -0.95 | 1.7 | 0 | - | [glacier](Vernagtferner/glacier.png) |
| Hintereisferner | nan | 1.0 / 2.1 / 1.9 / 1.8 | -1.21 / -0.98 | 2.3 | 0 | speed_ratio_obs | [glacier](Hintereisferner/glacier.png) |
| Glacier_de_la_Plaine_Morte | 0.72 | 1.8 / 2.5 / 1.4 / 1.3 | -1.49 / -1.58 | 1.3 | 0 | ela_ref_capped | [glacier](Glacier_de_la_Plaine_Morte/glacier.png) |
| Ghiacciaio_di_Malevalle | 0.75 | 0.5 / nan / nan / nan | -1.24 / -1.16 | 1.0 | 1 | dead_bands | [glacier](Ghiacciaio_di_Malevalle/glacier.png) |
| Griesgletscher | 0.57 | 1.7 / 2.3 / 1.5 / 1.4 | -1.38 / -1.25 | 0.7 | 0 | ela_ref_capped | [glacier](Griesgletscher/glacier.png) |
| Glacier_du_Giétro | 0.88 | 2.4 / 4.4 / 3.4 / 3.2 | -0.85 / -0.81 | 0.5 | 1 | speed_ratio_obs, dead_bands | [glacier](Glacier_du_Giétro/glacier.png) |
| Schwarzberggletscher | 0.90 | 2.4 / 4.7 / 4.2 / 4.3 | -0.97 / -0.79 | 0.6 | 0 | speed_ratio_obs | [glacier](Schwarzberggletscher/glacier.png) |
| Kesselwandferner | 0.59 | 2.8 / 5.7 / 3.2 / 2.9 | -0.59 / -0.51 | 2.5 | 0 | - | [glacier](Kesselwandferner/glacier.png) |
| Jamtalferner | 0.40 | 0.4 / nan / nan / nan | -1.18 / -1.23 | 1.2 | 0 | ela_ref_capped | [glacier](Jamtalferner/glacier.png) |
| Zettalunitzkees | 0.09 | 0.8 / nan / nan / nan | -0.75 / -1.06 | 1.3 | 0 | - | [glacier](Zettalunitzkees/glacier.png) |
| Hallstätter_Gletscher | 0.45 | 1.4 / 0.9 / 0.6 / 0.5 | -0.50 / -0.66 | 0.2 | 0 | speed_ratio_obs | [glacier](Hallstätter_Gletscher/glacier.png) |
| Glacier_de_Tsanfleuron | 0.86 | 1.9 / 1.8 / 1.4 / 1.3 | -1.46 / -1.49 | 1.4 | 0 | speed_ratio_obs, ela_ref_capped | [glacier](Glacier_de_Tsanfleuron/glacier.png) |
| Silvrettagletscher | 0.68 | 1.1 / 2.1 / 1.7 / 1.5 | -1.06 / -0.90 | 0.9 | 0 | speed_ratio_obs, ela_ref_capped | [glacier](Silvrettagletscher/glacier.png) |
| Careser | nan | 0.4 / nan / nan / nan | -1.36 / -1.63 | 1.2 | 0 | ela_ref_capped | [glacier](Careser/glacier.png) |
| Claridenfirn | 0.62 | 1.4 / 1.1 / 1.0 / 0.9 | -0.90 / -0.95 | 0.5 | 0 | speed_ratio_obs | [glacier](Claridenfirn/glacier.png) |
| Vedretta_Lunga | 0.63 | 0.5 / nan / nan / nan | -1.06 / -1.19 | 0.7 | 1 | dead_bands, ela_ref_capped | [glacier](Vedretta_Lunga/glacier.png) |
| Adlergletscher | 0.50 | 0.7 / nan / nan / nan | -0.45 / -0.22 | 0.7 | 0 | - | [glacier](Adlergletscher/glacier.png) |
| Ghiacciaio_del_Basòdino | 0.59 | 0.8 / nan / nan / nan | -0.95 / -1.12 | 0.1 | 0 | ela_ref_capped | [glacier](Ghiacciaio_del_Basòdino/glacier.png) |
