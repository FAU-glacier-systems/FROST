# Findings (2026-10-22)

Ice flow only where the glacier moves: 8 glaciers (ITS_LIVE p99 on the ice >= 10 m/yr) with IGM (tau_ref inverted on the 2018 geometry, forward runs from 2000, 32 members), 16 without ice flow (dh/dt = SMB with melt-out, 128 members, marked *). ES-MDA, 4 iterations, absolute-sill dh/dt covariance, model error 0.5 m/yr. 24 of 25 glaciers (Gietro missing).

## Thickness

- Thickness of 2000 (IceBoost moved back with dh/dt) against GlaThiDa moved to 2000: r 0.83, bias +5 m, RMSE 45 m on 7152 cells; median glacier bias 8 m. Off by > 30 m: Pasterze +50, Plaine Morte +51, Vernagt +37 (too thick), Claridenfirn -42 (too thin).
- Bed without artefacts; NASADEM and COPDEM align, the shift to 2000 is consistent with the DEM difference (+5-8 m radar penetration). But 19-77 m of ice in the outline cells: cliffs at the margins.

## Velocity (8 glaciers with ice flow)

- Forward speed in 2018 within 33 % of the observed on 6 of 8 (0.85-1.16; Pasterze 1.40 too fast, Findel 0.73 too slow).
- Start-up shock: the speeds of 2000 are 1.3-6.5x those of 2018 (Pasterze 6.5, Pers 2.4, Aletsch 2.1), the slowdown 2000-2020 is 31-85 %. Most likely the frozen ice-flow emulator extrapolating on the thicker 2000 geometry, not the data (shallow-ice estimate 1.2-1.7x).

## Sliding (tau_ref)

- No sign of overfitting: <= 1.5 % of the ice at the bounds, small-scale roughness 0.10-0.16 decades, small-scale correlation with the observed speed -0.30 to -0.01. Sliding share 0.4-0.6. Fitted through the emulator, so tau_ref belongs to that network.

## Elevation change

- Glacier means: r 0.93, MAE 0.12 m/yr (largest Jamtal 0.32). Band chi2/n against the data error alone 1-4 with ice flow (Aletsch 3.7, Rhone 3.8, Findel 4.1), 0-2 without; 6 bands melt out in every member (Aletsch, Pasterze, Corbassiere tongues).

## Agreement with GLAMOS / WGMS

- The reference ELA is capped at the glacier top in years without accumulation area; on 8 glaciers in > 25 % of the years (a lower bound, open markers, not scored). ELA on the other 15: r 0.67, bias +43 m, MAE 125 m. Ablation gradient MAE 2.4 (r -0.21), accumulation gradient MAE 0.85 per km: no skill in the gradients.
- SMB profiles against the mean mass balance per elevation bin: RMSE 0.83 m w.e./yr, 89 % of the bins inside +-2 std; worst Claridenfirn 2.8, Aletsch 2.1, Basodino 2.0.

## Limitations and open topics

- Ice-flow emulator frozen (never retrained, no physics solve): start-up shock, mass jumps at steep margins; Jamtal unstable with ice flow (now without).
- Thickness at the margins (cliffs) and the 2000 start geometry at the tongues.
- Without ice flow the accumulation area keeps its ice (no export): upper SMB and accumulation gradient biased low on the slow glaciers.
- References: capped ELAs, gradients fitted through them, mass balances from stakes extrapolated by a model; GLAMOS and WGMS differ in method.
- Model error 0.5 m/yr set by hand; the posteriors of the two setups differ in width (128 vs 32 members).
- Gietro to rerun; density (ice vs firn) not yet treated.

## Next steps

1. Emulator test (frozen / retrained / solver; experiments/emulator_test) and, if retraining removes the shock, rerun the 8 glaciers with ice flow.
2. Velocity threshold for the inversion (min_velocity_p99 3 m/yr) in that rerun.
3. Choose the selection rule for ice flow (p99 10 m/yr, or flux divergence against the band error) and apply it to all 409 glaciers.
