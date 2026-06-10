# Experiment: Fig-4C/D source-sink campaign (Monodomain V5.4)

Run scripts for the four-condition Fig-4C/D 2-D reproduction campaign.

**Backlinks**
- Research question: [source_sink_mismatch_investigation](../../../../Research/Active/source_sink_mismatch_investigation/README.md)
- Plan: [PLAN.md](../../../../Research/Active/source_sink_mismatch_investigation/PLAN.md) · [test plan](../../../../Research/Active/source_sink_mismatch_investigation/FIG4C_BLOCK_TEST_PLAN.md)
- Project dashboard: [MASTER.md](../../../../MASTER.md)

**Premise (corrected 2026-06-08):** the Ciaccio source-sink effect is 2-D in-plane
cross-section curvature (Fig 4), NOT thickness (Fig 5 = IBZ measurable proxy). No
thickness anywhere in these runs. Analysis via `cardiac_core.analysis`
(`activation_time_interp`, `front_metrics`, `fit_eikonal`).

| Script | Stage | Status |
|---|---|---|
| `run_s0_eikonal.py` | S0 — measure CV0, D_eik, r* on an expanding circle | **DONE** — CV0=62.4cm/s, D_eik=0.00084, r*≈134µm, R²=0.99997 |
| `render_s0_video.py` | render S0 expanding wave to MP4 (from cache) | DONE |
| `run_s0b_param_demo.py` | S0b — same circle at hourglass params: proves resolution+stencil caused the failure | **DONE** — 50µm/iso D_eik/D=+0.83; 250µm/cardinal4 (hourglass) D_eik/D=−0.93 (inverted) |
| `run_s0c_obstacle_tuning.py` | S0c — obstacle leading/trailing crescent vs r*/dx: diagonal conn. NOT sufficient; D/dx tuning is the other part | **DONE** — lead −110→−163µs as r*/dx 0.8→3.2; cardinal4≈moore8_iso here (axial wave) |
| `run_s0d_hourglass_confirm.py` | S0d — re-run actual hourglass, orig vs fixed | DONE — centerline dilation CV dip present even at 250µm |
| `run_s0d2/3/4_*.py` + `render_s0d_matched_video.py` | S0d isochrones + matched-dx control | **DONE — inverse crescent is RESOLUTION-dependent (needs dx~50µm), NOT stencil; both cardinal4 & moore8_iso show it at 50µm, neither at 250µm (PI visually confirmed). LAT-derived crescent metrics FAILED; visual front = ground truth.** |
| `run_s0e_dx_sweep.py` + `render_s0e_video.py` | S0e — dx sweep isochrones/video (full hourglass) | DONE — diverging fan trivial; converging is the real test |
| `run_s0f_converging.py` + `run_s0f_metric.py` + `render_s0f_video.py` | S0f — CONVERGING-half crescent vs dx (clean metric) | **DONE — −115→−236µs as dx 250→25µm; CONVERGES not dies (physical)** |
| `run_s0g_wavelength.py` | S0g (Step 2) — λ via APD at fixed dx | **DONE — crescent EXACTLY −175µs across 3.9× λ; λ INERT (absent from r*=D/CV)** |
| `run_s0h_scale_discriminator.py` | S0h — dx/constriction vs dx/r* | **DONE — fixed dx/constriction, crescent still moves −117→−180µs → NOT dx/constriction; it's dx/r*** |
| `run_s0i_cv_sweep.py` | S0i — CV channel via GNa at fixed dx,D | **DONE — crescent −338→−90µs as CV 49→81; CV operative (enters r*=D/CV). CONTROL PARAM = dx/r* = dx·CV/D** |
| `run_s1_nucleus.py` | S1 — critical-nucleus dx sweep -> dx_resolved | todo |
| `run_s2_regime.py` | S2 — hourglass vs strand->bulk | todo |
| `run_s3_excitability.py` | S3 — excitability sweep to block | todo |
| `run_s4_figure4.py` | S4 — Fig-4 A-D + eikonal/cross-section law | todo |

Run: `/opt/miniforge3/bin/conda run -n heart-conduction python <script>`
(conda is not on the non-interactive PATH).
