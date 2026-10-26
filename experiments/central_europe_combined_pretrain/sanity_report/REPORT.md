# Sanity report: data/results/central_europe_combined_pretrain

The full report is report.pdf (A4). Flags:

- **speed_ratio_obs**: forward speed in the observation year off the observed by > 33 %
- **speed_start_ratio**: forward speed of 2000 > 2x that of the observation year (start-up shock)
- **dhdt_chi2_data**: modelled band dh/dt outside the data error (chi2/n > 4)
- **dead_bands**: bands that melt out in every member
- **ela_ref_capped**: reference ELA at the glacier top in > 25 % of the years (a lower bound, left out of the ELA scores)

| Glacier | Thk r | Speed obs / fwd 2000 / fwd obs. year / fwd 2020 | dh/dt obs / model | chi2 data | Melted bands | Flags | Figures |
|---|---|---|---|---|---|---|---|
| Grosser_Aletschgletscher | 0.88 | 33.5 / 51.8 / 32.3 / 31.4 | -1.30 / -1.42 | 4.6 | 1 | dhdt_chi2_data, dead_bands | [glacier](Grosser_Aletschgletscher/glacier.png) |
| Pasterze | 0.67 | 4.7 / 10.4 / 6.9 / 6.7 | -1.28 / -1.32 | 1.2 | 1 | speed_ratio_obs, dead_bands | [glacier](Pasterze/glacier.png) |
| Glacier_de_Corbassière | 0.87 | 7.0 / 11.7 / 8.2 / 8.1 | -0.88 / -0.70 | 1.1 | 2 | dead_bands | [glacier](Glacier_de_Corbassière/glacier.png) |
| Rhonegletscher | 0.89 | 10.9 / 28.2 / 16.4 / 15.9 | -1.21 / -1.25 | 1.4 | 0 | speed_ratio_obs | [glacier](Rhonegletscher/glacier.png) |
| Vadret_Pers | 0.89 | 10.8 / 26.5 / 15.1 / 14.8 | -1.09 / -1.03 | 0.6 | 0 | speed_ratio_obs | [glacier](Vadret_Pers/glacier.png) |
| Findelgletscher | 0.91 | 9.9 / 12.3 / 8.8 / 8.6 | -0.70 / -0.80 | 2.9 | 0 | - | [glacier](Findelgletscher/glacier.png) |
| Allalingletscher | 0.87 | 7.1 / 5.0 / 4.4 / 4.3 | -0.61 / -0.45 | 0.7 | 0 | speed_ratio_obs | [glacier](Allalingletscher/glacier.png) |
| Vernagtferner | -0.26 | 0.7 / nan / nan / nan | -1.05 / -1.09 | 1.1 | 0 | - | [glacier](Vernagtferner/glacier.png) |
| Hintereisferner | nan | 1.0 / nan / nan / nan | -1.21 / -1.11 | 2.0 | 0 | - | [glacier](Hintereisferner/glacier.png) |
| Glacier_de_la_Plaine_Morte | 0.72 | 1.8 / nan / nan / nan | -1.49 / -1.47 | 0.4 | 0 | ela_ref_capped | [glacier](Glacier_de_la_Plaine_Morte/glacier.png) |
| Ghiacciaio_di_Malevalle | 0.75 | 0.5 / nan / nan / nan | -1.24 / -1.37 | 0.6 | 0 | - | [glacier](Ghiacciaio_di_Malevalle/glacier.png) |
| Griesgletscher | 0.57 | 1.7 / nan / nan / nan | -1.38 / -1.55 | 0.5 | 0 | ela_ref_capped | [glacier](Griesgletscher/glacier.png) |
| Glacier_du_Giétro | 0.88 | 2.4 / 10.6 / 5.7 / 5.6 | -0.85 / -0.70 | 0.4 | 0 | speed_ratio_obs | [glacier](Glacier_du_Giétro/glacier.png) |
| Schwarzberggletscher | 0.90 | 2.3 / nan / nan / nan | -0.97 / -1.01 | 0.5 | 0 | - | [glacier](Schwarzberggletscher/glacier.png) |
| Kesselwandferner | 0.59 | 2.8 / 11.4 / 4.4 / 4.3 | -0.59 / -0.89 | 4.7 | 0 | speed_ratio_obs, speed_start_ratio, dhdt_chi2_data | [glacier](Kesselwandferner/glacier.png) |
| Jamtalferner | 0.40 | 0.4 / nan / nan / nan | -1.18 / -1.50 | 0.7 | 0 | ela_ref_capped | [glacier](Jamtalferner/glacier.png) |
| Zettalunitzkees | 0.09 | 0.8 / nan / nan / nan | -0.75 / -0.76 | 0.3 | 0 | - | [glacier](Zettalunitzkees/glacier.png) |
| Hallstätter_Gletscher | 0.45 | 1.4 / nan / nan / nan | -0.50 / -0.71 | 0.1 | 0 | - | [glacier](Hallstätter_Gletscher/glacier.png) |
| Glacier_de_Tsanfleuron | 0.86 | 1.9 / nan / nan / nan | -1.46 / -1.70 | 0.6 | 0 | ela_ref_capped | [glacier](Glacier_de_Tsanfleuron/glacier.png) |
| Silvrettagletscher | 0.68 | 1.1 / nan / nan / nan | -1.06 / -1.26 | 0.4 | 0 | ela_ref_capped | [glacier](Silvrettagletscher/glacier.png) |
| Careser | nan | 0.4 / nan / nan / nan | -1.36 / -1.46 | 0.6 | 0 | ela_ref_capped | [glacier](Careser/glacier.png) |
| Claridenfirn | 0.62 | 1.4 / nan / nan / nan | -0.90 / -0.93 | 0.2 | 0 | - | [glacier](Claridenfirn/glacier.png) |
| Vedretta_Lunga | 0.63 | 0.5 / nan / nan / nan | -1.06 / -1.12 | 0.4 | 0 | ela_ref_capped | [glacier](Vedretta_Lunga/glacier.png) |
| Adlergletscher | 0.50 | 0.7 / nan / nan / nan | -0.45 / -0.17 | 0.2 | 0 | - | [glacier](Adlergletscher/glacier.png) |
| Ghiacciaio_del_Basòdino | 0.59 | 0.8 / nan / nan / nan | -0.95 / -0.91 | 0.0 | 0 | ela_ref_capped | [glacier](Ghiacciaio_del_Basòdino/glacier.png) |
