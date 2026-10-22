# Sanity report: data/results/central_europe_esmda_velyear

Band height 50 m, model error 0.5 m/yr, tau_ref bounds (0.05, 2.0). Glaciers sorted as in glaciers.txt. Flags:

- **bed_roughness_ratio**: bed rougher than the ice-free terrain
- **bed_pits**: single-cell pits in the bed
- **tau_at_bounds_pct**: tau_ref at its bounds on > 10 % of the ice
- **tau_noise_corr**: tau_ref copies the small-scale pattern of the observed speed
- **speed_ratio_inv_year**: forward speed in the inversion year off the inverted speed by > 25 %
- **dhdt_chi2_data**: modelled band dh/dt outside the data error (chi2/n > 4)
- **dead_bands**: bands that melt out in every member

![overview](overview.png)

| Glacier | Bed rough. ratio | Pits | tau at bounds % | tau noise corr | Speed obs / inv / fwd(inv yr) / fwd 2020 | dh/dt obs / model | chi2 data | Dead bands | Flags | Figures |
|---|---|---|---|---|---|---|---|---|---|---|
| Grosser_Aletschgletscher | 0.65 | 7 | 1 | -0.09 | 33.5 / 34.6 / 37.3 / 37.2 | -1.30 / -1.38 | 3.7 | 3 | bed_pits, dead_bands | [bed](Grosser_Aletschgletscher/bed.png) · [velocity](Grosser_Aletschgletscher/velocity.png) · [tau_ref](Grosser_Aletschgletscher/tau_ref.png) · [dhdt](Grosser_Aletschgletscher/dhdt.png) |
| Pasterze | 0.81 | 6 | 0 | -0.17 | 4.7 / 5.2 / 6.5 / 6.4 | -1.28 / -1.25 | 1.8 | 1 | bed_pits, speed_ratio_inv_year, dead_bands | [bed](Pasterze/bed.png) · [velocity](Pasterze/velocity.png) · [tau_ref](Pasterze/tau_ref.png) · [dhdt](Pasterze/dhdt.png) |
| Glacier_de_Corbassière | 0.78 | 2 | 1 | -0.11 | 7.0 / 8.0 / 7.9 / 7.8 | -0.88 / -0.74 | 1.1 | 2 | bed_pits, dead_bands | [bed](Glacier_de_Corbassière/bed.png) · [velocity](Glacier_de_Corbassière/velocity.png) · [tau_ref](Glacier_de_Corbassière/tau_ref.png) · [dhdt](Glacier_de_Corbassière/dhdt.png) |
| Rhonegletscher | 0.86 | 6 | 0 | -0.02 | 10.9 / 11.4 / 9.5 / 9.1 | -1.21 / -1.14 | 3.8 | 0 | bed_pits | [bed](Rhonegletscher/bed.png) · [velocity](Rhonegletscher/velocity.png) · [tau_ref](Rhonegletscher/tau_ref.png) · [dhdt](Rhonegletscher/dhdt.png) |
| Vadret_Pers | 0.96 | 1 | 1 | -0.18 | 10.8 / 12.2 / 12.5 / 12.0 | -1.09 / -0.85 | 2.6 | 0 | bed_pits | [bed](Vadret_Pers/bed.png) · [velocity](Vadret_Pers/velocity.png) · [tau_ref](Vadret_Pers/tau_ref.png) · [dhdt](Vadret_Pers/dhdt.png) |
| Findelgletscher | 0.62 | 2 | 1 | -0.27 | 9.9 / 9.9 / 7.3 / 7.1 | -0.70 / -0.81 | 4.1 | 0 | bed_pits, speed_ratio_inv_year, dhdt_chi2_data | [bed](Findelgletscher/bed.png) · [velocity](Findelgletscher/velocity.png) · [tau_ref](Findelgletscher/tau_ref.png) · [dhdt](Findelgletscher/dhdt.png) |
| Allalingletscher | 0.78 | 1 | 1 | -0.30 | 7.1 / 7.5 / 6.0 / 6.0 | -0.61 / -0.52 | 0.4 | 0 | bed_pits | [bed](Allalingletscher/bed.png) · [velocity](Allalingletscher/velocity.png) · [tau_ref](Allalingletscher/tau_ref.png) · [dhdt](Allalingletscher/dhdt.png) |
| Vernagtferner | 0.69 | 0 | 0 | 0.14 | 0.7 / 1.0 / 1.2 / 1.1 | -1.05 / -1.16 | 1.0 | 0 | speed_ratio_inv_year | [bed](Vernagtferner/bed.png) · [velocity](Vernagtferner/velocity.png) · [tau_ref](Vernagtferner/tau_ref.png) · [dhdt](Vernagtferner/dhdt.png) |
| Hintereisferner | 1.06 | 0 | 0 | -0.17 | 1.0 / 1.4 / 1.4 / 1.3 | -1.21 / -1.14 | 1.8 | 0 | - | [bed](Hintereisferner/bed.png) · [velocity](Hintereisferner/velocity.png) · [tau_ref](Hintereisferner/tau_ref.png) · [dhdt](Hintereisferner/dhdt.png) |
| Glacier_de_la_Plaine_Morte | 0.65 | 1 | 0 | -0.03 | 1.8 / 1.9 / 1.9 / 1.9 | -1.49 / -1.49 | 1.4 | 0 | bed_pits | [bed](Glacier_de_la_Plaine_Morte/bed.png) · [velocity](Glacier_de_la_Plaine_Morte/velocity.png) · [tau_ref](Glacier_de_la_Plaine_Morte/tau_ref.png) · [dhdt](Glacier_de_la_Plaine_Morte/dhdt.png) |
| Ghiacciaio_di_Malevalle | 0.74 | 0 | 0 | -0.14 | 0.5 / 1.0 / 0.9 / 0.9 | -1.24 / -1.31 | 0.3 | 0 | - | [bed](Ghiacciaio_di_Malevalle/bed.png) · [velocity](Ghiacciaio_di_Malevalle/velocity.png) · [tau_ref](Ghiacciaio_di_Malevalle/tau_ref.png) · [dhdt](Ghiacciaio_di_Malevalle/dhdt.png) |
| Griesgletscher | 0.78 | 2 | 0 | -0.37 | 1.7 / 2.0 / 2.3 / 2.2 | -1.38 / -1.35 | 0.1 | 0 | bed_pits | [bed](Griesgletscher/bed.png) · [velocity](Griesgletscher/velocity.png) · [tau_ref](Griesgletscher/tau_ref.png) · [dhdt](Griesgletscher/dhdt.png) |
| Schwarzberggletscher | 0.59 | 1 | 0 | -0.27 | 2.4 / 2.2 / 3.1 / 3.0 | -0.97 / -0.93 | 0.5 | 0 | bed_pits, speed_ratio_inv_year | [bed](Schwarzberggletscher/bed.png) · [velocity](Schwarzberggletscher/velocity.png) · [tau_ref](Schwarzberggletscher/tau_ref.png) · [dhdt](Schwarzberggletscher/dhdt.png) |
| Kesselwandferner | 0.89 | 0 | 0 | -0.01 | 2.8 / 3.3 / 3.1 / 2.9 | -0.59 / -0.63 | 3.1 | 0 | - | [bed](Kesselwandferner/bed.png) · [velocity](Kesselwandferner/velocity.png) · [tau_ref](Kesselwandferner/tau_ref.png) · [dhdt](Kesselwandferner/dhdt.png) |
| Jamtalferner | 0.80 | 0 | 0 | 0.25 | 0.4 / 0.8 / 22.9 / 28.0 | -1.18 / -1.42 | 0.9 | 0 | speed_ratio_inv_year | [bed](Jamtalferner/bed.png) · [velocity](Jamtalferner/velocity.png) · [tau_ref](Jamtalferner/tau_ref.png) · [dhdt](Jamtalferner/dhdt.png) |
| Zettalunitzkees | 0.54 | 0 | 0 | -0.37 | 0.8 / 0.9 / 1.0 / 1.0 | -0.75 / -0.77 | 0.5 | 0 | - | [bed](Zettalunitzkees/bed.png) · [velocity](Zettalunitzkees/velocity.png) · [tau_ref](Zettalunitzkees/tau_ref.png) · [dhdt](Zettalunitzkees/dhdt.png) |
| Hallstätter_Gletscher | 0.47 | 0 | 0 | 0.14 | 1.4 / 0.8 / 0.8 / 0.9 | -0.50 / -0.69 | 0.1 | 0 | - | [bed](Hallstätter_Gletscher/bed.png) · [velocity](Hallstätter_Gletscher/velocity.png) · [tau_ref](Hallstätter_Gletscher/tau_ref.png) · [dhdt](Hallstätter_Gletscher/dhdt.png) |
| Glacier_de_Tsanfleuron | 0.41 | 0 | 0 | 0.10 | 1.9 / 1.6 / 1.4 / 1.4 | -1.46 / -1.56 | 1.0 | 0 | - | [bed](Glacier_de_Tsanfleuron/bed.png) · [velocity](Glacier_de_Tsanfleuron/velocity.png) · [tau_ref](Glacier_de_Tsanfleuron/tau_ref.png) · [dhdt](Glacier_de_Tsanfleuron/dhdt.png) |
| Silvrettagletscher | 0.66 | 0 | 0 | -0.14 | 1.1 / 1.1 / 1.2 / 1.2 | -1.06 / -1.09 | 0.3 | 0 | - | [bed](Silvrettagletscher/bed.png) · [velocity](Silvrettagletscher/velocity.png) · [tau_ref](Silvrettagletscher/tau_ref.png) · [dhdt](Silvrettagletscher/dhdt.png) |
| Careser | 0.74 | 1 | 0 | 0.17 | 0.4 / 0.4 / 0.5 / 0.5 | -1.36 / -1.60 | 0.9 | 0 | bed_pits | [bed](Careser/bed.png) · [velocity](Careser/velocity.png) · [tau_ref](Careser/tau_ref.png) · [dhdt](Careser/dhdt.png) |
| Claridenfirn | 0.84 | 0 | 0 | 0.00 | 1.4 / 1.8 / 1.9 / 1.8 | -0.90 / -0.99 | 0.1 | 0 | - | [bed](Claridenfirn/bed.png) · [velocity](Claridenfirn/velocity.png) · [tau_ref](Claridenfirn/tau_ref.png) · [dhdt](Claridenfirn/dhdt.png) |
| Vedretta_Lunga | 0.79 | 0 | 0 | -0.12 | 0.5 / 0.4 / 0.8 / 0.7 | -1.06 / -1.17 | 0.3 | 0 | speed_ratio_inv_year | [bed](Vedretta_Lunga/bed.png) · [velocity](Vedretta_Lunga/velocity.png) · [tau_ref](Vedretta_Lunga/tau_ref.png) · [dhdt](Vedretta_Lunga/dhdt.png) |
| Adlergletscher | 0.94 | 0 | 0 | 0.15 | 0.7 / 1.7 / 1.6 / 1.6 | -0.45 / -0.33 | 0.4 | 0 | - | [bed](Adlergletscher/bed.png) · [velocity](Adlergletscher/velocity.png) · [tau_ref](Adlergletscher/tau_ref.png) · [dhdt](Adlergletscher/dhdt.png) |
| Ghiacciaio_del_Basòdino | 0.23 | 0 | 0 | -0.08 | 0.8 / 0.4 / 1.1 / 1.1 | -0.95 / -0.84 | 0.0 | 0 | speed_ratio_inv_year | [bed](Ghiacciaio_del_Basòdino/bed.png) · [velocity](Ghiacciaio_del_Basòdino/velocity.png) · [tau_ref](Ghiacciaio_del_Basòdino/tau_ref.png) · [dhdt](Ghiacciaio_del_Basòdino/dhdt.png) |
