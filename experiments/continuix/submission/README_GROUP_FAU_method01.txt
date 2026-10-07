=============================================================================================================================
METHOD DESCRIPTION:
FROST (Framework for assimilating Remote-sensing Observations for Surface mass balance Tuning) couples the Instructed Glacier Model (IGM 3.2.0) with an ensemble smoother with multiple data assimilation (ES-MDA) and calibrates a three-parameter elevation-dependent SMB model (equilibrium line altitude, ablation gradient, accumulation gradient) against the observed dh/dt. The spatial discretization is a regular 2.5D grid (50 m; 100 m for G01, 25 m for S01; coarser input grids are kept), on which the ice flow is computed by IGM's physics-informed CNN emulator of a higher-order Stokes approximation (MOLHO vertical basis). The provided ice thickness is kept fixed and the basal sliding parameter (tau_ref) is inverted from the observed surface velocities with a Laplacian regularisation. 36 ensemble members, each with different SMB parameters, are run forward through the dh/dt period with IGM's mass conservation, and the modelled dh/dt is compared with the observed dh/dt averaged in 50 m elevation bands; the parameters are updated in 6 ES-MDA iterations. The band errors combine the provided dh/dt error, spatially correlated as in Hugonnet et al. (2021), with a model error of 0.5 m/yr per band (1.3 m/yr for G03 and 1.5 m/yr for G05, scaled so that chi2/n is about 1). FDIV is the time-mean flux divergence of the forward runs of the calibrated ensemble, and SMB is the calibrated SMB model evaluated on the surface at the middle of the dh/dt period. SMB is therefore a smooth function of elevation, not the pixel-wise residual DHDT + FDIV, and the two agree only in the elevation-band mean. All fields are in m ice equivalent per year with an ice density of 910 kg m-3; no firn or snow density is applied. UNCT_SMB and UNCT_FDIV are the standard deviations over the calibrated ensemble, i.e. the parameter uncertainty constrained by the dh/dt and its errors; they do not include errors of the provided thickness or the structural error of the SMB model.

=============================================================================================================================
DATA PRE-PROCESSING:
The same steps were applied to all experiments (EXP01, EXP03-EXP15); the perturbed fields were used as provided.

1. Ice mask: ICEMASK cells that have THK, velocity (VX or VY) or DHDT data. ContinuIX fills missing values with 0, so zero values inside the ICEMASK are treated as gaps. This removes about a third of the S01 domain (ice-free beyond the terminus) and at most 0.3 % of the cells of the other glaciers. Experiments on the EXP01 grid (EXP03-EXP13) use the EXP01 ice mask, so that the perturbations change the data but not the glacier extent; EXP14 and EXP15 use their own mask.
2. Grid: no reprojection (the provided CRS is kept). The fields are resampled to the model grid with rasterio, using block averaging (Resampling.average) where the model grid is coarser than the data and bilinear interpolation otherwise. Model grid spacing: 50 m, except G01 100 m and S01 25 m; grids coarser than this are kept (S02 100 m, EXP14 S02 300 m, EXP15 100 m). Five ice-free cells are added around the domain. A model cell is ice if at least half of it is ice in the data.
3. Thickness (THK): gaps inside the ice mask are filled by linear interpolation (scipy griddata), with the nearest value outside the convex hull of the data. THK is not smoothed and is held fixed in the inversion.
4. Surface (DEM): gaps are filled with the nearest valid value. For the forward runs, the DEM is shifted with dh/dt from its acquisition date to the start of the dh/dt period.
5. Velocity (VX, VY): cells with zero in both components are treated as missing and left out of the misfit; no gap filling or smoothing. A uniform misfit std of 1 m/yr is used instead of the provided velocity uncertainties, because the latter are standard deviations over the period, not errors of the mean field.
6. Inversion of the sliding: tau_ref (log10 transform, bounds 0.001-10 MPa) minimises a Huber misfit to the surface velocities plus a squared-Laplacian regularisation (lambda = 1e11, from an L-curve on G03 and G05), the same for all glaciers and experiments.
7. dh/dt (DHDT): not gap-filled for the calibration; band means use only the pixels with data. Gaps are filled (linear interpolation) only for the DEM shift in step 4 and for the evaluation surface of the SMB. Period from the DHDT timestamp attribute; 2000-2010 for the synthetic glaciers S01 and S02. The dh/dt error is UNCT_DHDT where provided (S01, S02), otherwise the 'uncertainty' attribute if given as a rate (G03: 0.07 m/yr), otherwise 0.2 m/yr.
8. SMB prior: ELA at the median ice elevation with a standard deviation of a third of the elevation range; ablation gradient 7.6 +- 9.0 and accumulation gradient 3.1 +- 3.4 m/yr per km.
9. Output: FDIV and UNCT_FDIV are interpolated bilinearly from the model grid to the provided grid; SMB and UNCT_SMB are evaluated directly on the provided grid. All fields are set to NaN outside the ice mask of step 1. THK, VX, VY, DHDT, DEM and BED were not modified and are not included.

=============================================================================================================================
DOI/CITATION:
Herrmann, O., et al. (2025): A Kalman filter-based framework for assimilating remote sensing observations into a surface mass balance model. Annals of Glaciology, 66, e23. doi:10.1017/aog.2025.10020
Code: https://github.com/FAU-glacier-systems/FROST
IGM: https://github.com/instructed-glacier-model/igm

=============================================================================================================================
CONTRIBUTORS:
Oskar Herrmann, ORCID 0000-0002-0319-9065 (https://orcid.org/0000-0002-0319-9065)

=============================================================================================================================
INSTITUTION:
Junior Research Group Glacier Systems & Natural Hazards, Institute of Geography, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Erlangen, Germany

=============================================================================================================================
ACKNOWLEDGEMENTS:
FROST was developed by Oskar Herrmann and Johannes J. Fürst (FAU, johannes.fuerst@fau.de). We thank the developers of IGM (Guillaume Jouvet and co-workers). Computations were carried out on the Alex cluster of the Erlangen National High Performance Computing Center (NHR@FAU).
Corresponding author: Oskar Herrmann, oskar.herrmann@fau.de

=============================================================================================================================
PERMISSIONS:
confirm

=============================================================================================================================
ADDITIONAL INFORMATION:
- Submitted experiments: EXP01 for all eight glaciers; EXP03-EXP15 for the mandatory glaciers G01, G05, S01 and S02. EXP02 and EXP16-EXP20 were not run.
- FROST is not a pixel-wise continuity inversion. The SMB is a calibrated function of elevation (one ELA and two gradients per glacier), so it shows no spatial pattern at a given elevation. On G01 (Langjökull) it cannot represent differences between the outlet basins.
- The glacier-mean SMB is pinned by the observed dh/dt (the glacier-mean FDIV is close to zero). The thickness perturbations (EXP03-EXP07) therefore mainly shift the calibrated SMB parameters (e.g. the G05 ELA by up to 190 m, the S02 ELA by 250 m) and change the glacier-mean SMB by at most 0.07 m/yr, except S01 EXP03/EXP04 (+0.28/+0.22 m/yr) and G05 EXP06 (-0.19 m/yr). The velocity perturbations (EXP08-EXP13) change it by at most 0.04 m/yr, as the regularisation of tau_ref smooths out the velocity noise. S01 at 100 m (EXP15) has only 219 ice cells and its narrow margins are poorly resolved (-0.35 m/yr).
- Known limitations: on G03 the upper half of the glacier thins about 1.2 m/yr faster in the model than observed, which the three SMB parameters cannot correct. On S01 the modelled surface velocities are about 29 % lower than the observed ones after the inversion.
- The global attributes of each file give the model resolution, the dh/dt period and the ensemble mean and standard deviation of the calibrated SMB parameters (ela in m a.s.l., abl_grad and acc_grad in m ice eq./yr per km).
