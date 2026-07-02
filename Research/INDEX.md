# Research Index

Master index for the research knowledge base. Start here when looking up theory, debugging simulation issues, or adding new papers.

## Question Map

| Folder | Question | Quick Answer |
|--------|----------|--------------|
| [Q1](Q1_spatial_discretization/) | How do I discretize the cardiac PDEs? | FDM (structured, fast), FEM (unstructured, flexible), FVM (conservative). Face-based BCs. |
| [Q2](Q2_linear_solvers/) | How do I solve the linear systems? | 3 tiers: Spectral direct → PCG+spectral → PCG+GMG/AMG. Elliptic solve is the bottleneck. |
| [Q3](Q3_time_integration/) | What time stepper should I use? | Operator splitting (Strang/Godunov) + Rush-Larsen for ionic + CN/BDF2 for diffusion. |
| [Q4](Q4_lbm_cardiac/) | Can LBM solve cardiac EP? | Yes for monodomain (D2Q5/D2Q9). Bidomain feasible but no production implementation yet. |
| [Q5](Q5_boundary_conduction_speedup/) | Does CV increase at tissue boundaries? | Yes — reduced electrotonic load → higher safety factor → ~7-13% faster CV. |
| [Q6](Q6_scar_bc_validity/) | Are Dirichlet BCs valid at scar? | No — scar is inert, use Neumann (no-flux). Dirichlet creates unphysical artifacts. |
| [Q7](Q7_fetal_heart_development/) | How does the fetal conduction system develop? | Sequential morphogenesis, gap junction transitions (Cx40/43/45), slow→fast conduction. |
| [Q8](Q8_ionic_model_optimization/) | How do I tune TTP06/ORd for target CV and APD? | Multi-objective (NSGA-II) or Bayesian (HMC/GP emulator). Never fit to single AP alone — use multi-rate pacing + tissue CV. |

## Debugging Quick-Reference

| Symptom | Likely Cause | Go To |
|---------|-------------|-------|
| PCG diverges on elliptic solve | Wrong preconditioner or non-SPD matrix | Q2/BIDOMAIN_LINEAR_SOLVERS.md |
| Too many PCG iterations (>50) | Missing spectral preconditioner | Q2/QUICK_START.txt |
| CV ~35% too high in LBM vs FDM | Numerical dispersion (expected, not a bug) | Q4/04_LBM_EP_Implementation.md |
| CV changes near tissue edge | Electrotonic load effect (Kleber) | Q5/README.md |
| Explicit time stepper blows up | CFL violation or stiff source term | Q3/BIDOMAIN_EXPLICIT_METHODS.md |
| 9-pt stencil wrong at boundary | Face-based BC not applied correctly | Q1/01_FDM_Stencils_and_Implementation.md |
| Kleber ratio doesn't converge to 1.131 | dx too coarse (need dx < 0.01) | Q5/CARDIAC_BOUNDARY_CONDUCTION_BIBLIOGRAPHY.md |
| Bidomain phi_e solve has null space | All-Neumann BCs → pin one node or remove null space | Q2/BIDOMAIN_LINEAR_SOLVERS.md |
| Splitting order matters for accuracy | Godunov is O(dt), Strang is O(dt^2) | Q3/BIDOMAIN_SOLVER_METHODS.md |
| LBM tau calibration off | Check sigma_to_D() and tau_from_D() conversion | Q4/04_LBM_EP_Implementation.md |
| CV/APD don't match target after tuning | Single-AP fitting is non-unique; need multi-rate or tissue data | Q8/README.md |
| Multiple param sets give same AP but different CV | IKr/IKs compensation; add Rm or CV as objective | Q8/pouranbarani_2019_multiobjective_rm.md |
| Parameter uncertainty too large | Use Bayesian methods (HMC) with multi-CL data | Q8/nietoramos_2023_bayesian_hmc.md |

## Citation Registry

| Key | PDF | Summary Location | Topics |
|-----|-----|-------------------|--------|
| rapaka_2012_lbm_ep | [PDF](papers/rapaka_2012_lbm_ep.pdf) | Q4/04_LBM_EP_Implementation.md | LBM, GPU, monodomain |
| belmiloudi_2019_coupled_lbm_fv | [PDF](papers/belmiloudi_2019_coupled_lbm_fv.pdf) | Q4/LBM_BIDOMAIN.md | LBM, bidomain, coupling |
| campos_2016_lbm_gpu | [PDF](papers/campos_2016_lbm_gpu.pdf) | Q4/overview.md | LBM, GPU, parallel |
| lbm_review_macro_flows | [PDF](papers/lbm_review_macro_flows.pdf) | Q4/overview.md | LBM, theory |
| 12859_2023_article_5513 | [PDF](papers/12859_2023_article_5513.pdf) | — | (to be summarized) |
| pouranbarani_2019_multiobjective_rm | [PDF](papers/multiobjective_optimization_membrane_resistance_2019_pouranbarani.pdf) | Q8/pouranbarani_2019_multiobjective_rm.md | NSGA-II, multi-objective, Rm, CV |
| coveney_2021_bayesian_restitution | [PDF](papers/bayesian_calibration_restitution_emulators_2021_coveney.pdf) | Q8/coveney_2021_bayesian_restitution.md | Bayesian, GP emulator, PCA, restitution |
| nietoramos_2023_bayesian_hmc | [PDF](papers/bayesian_hmc_abc_cardiac_fitting_2023_nietoramos.pdf) | Q8/nietoramos_2023_bayesian_hmc.md | HMC, ABC-SMC, Bayesian, posterior |
| groenendaal_2015_cell_specific | [PDF](papers/cell_specific_electrophysiology_models_2015_groenendaal.pdf) | Q8/groenendaal_2015_cell_specific.md | GA, cell-specific, non-uniqueness |
| zhang_2024_gradient_two_waveform | [PDF](papers/gradient_based_hipsc_two_waveform_fitting_2024_zhang.pdf) | Q8/zhang_2024_gradient_two_waveform.md | gradient PO, two-waveform, hiPSC-CM |
| chang_2017_uq_cipa | [PDF](papers/uncertainty_quantification_proarrhythmia_cipa_2017_chang.pdf) | Q8/chang_2017_uq_cipa.md | UQ, ORd, CiPA, MCMC |
| nietoramos_2022_hmc_cinc | [PDF](papers/hamiltonian_monte_carlo_ap_parameters_2022_nietoramos.pdf) | Q8/nietoramos_2022_hmc_cinc.md | HMC, NUTS, proof-of-concept |
| cairns_2017_ga_parameterization | [PDF](papers/genetic_algorithm_ap_parameterization_2017_cairns.pdf) | Q8/cairns_2017_ga_parameterization.md | GA, parameterization |
| bishop_2011_augmented_monodomain | [PDF](papers/augmented_monodomain_bath_loading_2011_bishop.pdf) | Q5/bishop_2011_augmented_monodomain.md | bath-loading, augmented monodomain, CV speedup |
| bishop_2011_bath_loading_arrhythmias | [PDF](papers/bath_loading_arrhythmia_wavefront_curvature_2011_bishop.pdf) | Q5/bishop_2011_bath_loading_arrhythmias.md | bath-loading, arrhythmias, fiber rotation |
| rossi_2018_thickness_curvature | [PDF](papers/muscle_thickness_curvature_conduction_velocity_2018_rossi.pdf) | Q5/rossi_2018_thickness_curvature.md | wall thickness, curvature, atrial CV |
| patel_2005_bidomain_boundary | [PDF](papers/overdetermined_boundary_condition_bidomain_2005_patel.pdf) | Q5/patel_2005_bidomain_boundary.md | overdetermined BC, tissue-bath interface, analytical |
| roberts_2008_discrete_multidomain | [PDF](papers/discrete_multidomain_interstitial_space_2008_roberts.pdf) | Q5/roberts_2008_discrete_multidomain.md | discrete multidomain, extracellular cleft, microscale |
| tranquillo_2005_analytical_bath | [PDF](papers/analytical_extracellular_potential_finite_bath_2005_tranquillo.pdf) | Q5/tranquillo_2005_analytical_bath.md | analytical phi_e, finite bath, Fourier series |
| johnston_2008_approximate_bidomain | [PDF](papers/approximate_bidomain_boundary_solutions_2008_johnston.pdf) | Q5/johnston_2008_approximate_bidomain.md | approximate bidomain, boundary layer, Patel-Roth extension |
| sambelashvili_2004_scroll_wave | [PDF](papers/scroll_wave_reentry_boundary_dynamics_2004_sambelashvili.pdf) | Q5/sambelashvili_2004_scroll_wave.md | scroll wave, filament, boundary type, reentry |

| hipsc_cm_maturation_models | (multi-paper survey) | Q8/hipsc_cm_maturation_models.md | hiPSC-CM, maturation, IK1, If, automaticity, quiescence |

*Add new papers using `/summarize-paper` or `/research`. Each entry maps citation key → PDF → summary file → topic tags.*

## Other Resources

| Resource | Path | Contents |
|----------|------|----------|
| Reference implementations | [code_examples/](code_examples/) | MonoAlg3D (C/CUDA), torchcor (PyTorch), lettuce (LBM), pyamg, amgcl, etc. |
| Textbook | [Active/textbook/](Active/textbook/) | Cardiac Computational Modeling textbook HTML/PDF, style guide, audits (now a tracked research question; migrated 2026-07-02). Use `/textbook-edit` and `/textbook-compile`. |
| Simulation figures | [figures/](figures/) | Legacy screenshots and wave propagation visualizations |
