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
| pouranbarani_2019_multiobjective_rm | DOI: 10.1371/journal.pone.0225245 | Q8/pouranbarani_2019_multiobjective_rm.md | NSGA-II, multi-objective, Rm, CV |
| coveney_2021_bayesian_restitution | DOI: 10.3389/fphys.2021.693015 | Q8/coveney_2021_bayesian_restitution.md | Bayesian, GP emulator, PCA, restitution |
| nietoramos_2023_bayesian_hmc | DOI: 10.1007/s11517-022-02685-y | Q8/nietoramos_2023_bayesian_hmc.md | HMC, ABC-SMC, Bayesian, posterior |
| groenendaal_2015_cell_specific | DOI: 10.1371/journal.pcbi.1004242 | Q8/groenendaal_2015_cell_specific.md | GA, cell-specific, non-uniqueness |
| zhang_2024_gradient_two_waveform | DOI: 10.1038/s41598-024-63413-0 | Q8/zhang_2024_gradient_two_waveform.md | gradient PO, two-waveform, hiPSC-CM |
| chang_2017_uq_cipa | DOI: 10.3389/fphys.2017.00917 | Q8/chang_2017_uq_cipa.md | UQ, ORd, CiPA, MCMC |
| nietoramos_2022_hmc_cinc | DOI: 10.23919/cinc53138.2021.9662836 | Q8/nietoramos_2022_hmc_cinc.md | HMC, NUTS, proof-of-concept |
| cairns_2017_ga_parameterization | DOI: 10.1063/1.5000354 | Q8/cairns_2017_ga_parameterization.md | GA, parameterization |
| bishop_2011_augmented_monodomain | DOI: 10.1109/TBME.2010.2096425 | Q5/bishop_2011_augmented_monodomain.md | bath-loading, augmented monodomain, CV speedup |
| bishop_2011_bath_loading_arrhythmias | DOI: 10.1016/j.bpj.2011.10.052 | Q5/bishop_2011_bath_loading_arrhythmias.md | bath-loading, arrhythmias, fiber rotation |
| rossi_2018_thickness_curvature | DOI: 10.3389/fphys.2018.01344 | Q5/rossi_2018_thickness_curvature.md | wall thickness, curvature, atrial CV |
| patel_2005_bidomain_boundary | DOI: 10.1103/PhysRevE.72.051931 | Q5/patel_2005_bidomain_boundary.md | overdetermined BC, tissue-bath interface, analytical |
| roberts_2008_discrete_multidomain | DOI: 10.1529/biophysj.108.137349 | Q5/roberts_2008_discrete_multidomain.md | discrete multidomain, extracellular cleft, microscale |
| tranquillo_2005_analytical_bath | DOI: 10.1109/TBME.2004.840467 | Q5/tranquillo_2005_analytical_bath.md | analytical phi_e, finite bath, Fourier series |
| johnston_2008_approximate_bidomain | DOI: 10.1103/PhysRevE.78.041904 | Q5/johnston_2008_approximate_bidomain.md | approximate bidomain, boundary layer, Patel-Roth extension |
| sambelashvili_2004_scroll_wave | DOI: 10.1152/ajpheart.01108.2003 | Q5/sambelashvili_2004_scroll_wave.md | scroll wave, filament, boundary type, reentry |

*Add new papers using `/summarize-paper` or `/research`. Each entry maps citation key → PDF → summary file → topic tags.*

## Other Resources

| Resource | Path | Contents |
|----------|------|----------|
| Reference implementations | [code_examples/](code_examples/) | MonoAlg3D (C/CUDA), torchcor (PyTorch), lettuce (LBM), pyamg, amgcl, etc. |
| Textbook | [textbook/](textbook/) | Bidomain textbook HTML/PDF, style guide, audits. Use `/textbook-edit` and `/textbook-compile`. |
| Simulation figures | [figures/](figures/) | Legacy screenshots and wave propagation visualizations |
