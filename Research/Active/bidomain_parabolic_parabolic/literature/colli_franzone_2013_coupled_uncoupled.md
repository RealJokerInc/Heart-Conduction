---
paper: colli_franzone_2013_coupled_uncoupled
title: "A Comparison of Coupled and Uncoupled Solvers for the Cardiac Bidomain Model"
authors: "P. Colli Franzone, L. F. Pavarino, S. Scacchi"
year: 2013
journal: "ESAIM: Mathematical Modelling and Numerical Analysis (M2AN)"
doi: "10.1051/m2an/2012055"
pdf: ../papers/esaim_m2an_2013_coupled_uncoupled_bidomain.pdf
questions: [bidomain_parabolic_parabolic, engine_consolidation]
---

## Key Findings

- **Uncoupled (Gauss-Seidel) PE bidomain is 2.5–3× faster than the fully coupled formulation** at equivalent accuracy on 3D structured and unstructured meshes. The uncoupled solver solves the parabolic PDE twice and the elliptic PDE once per timestep.
- **Uncoupled is as scalable as coupled.** Parallel benchmarks on Linux clusters show equivalent strong/weak scaling for both strategies. Decoupling does not cost scalability.
- **Faster PCG convergence for the decoupled linear systems.** The elliptic system from the decoupled formulation is better-conditioned for Multilevel Hybrid Schwarz preconditioning than the coupled 2×2 block system.

## Method

- **Formulation**: Standard **parabolic-elliptic** (PE) bidomain — the paper is NOT about parabolic-parabolic formulations. It addresses how to *solve* the PE system efficiently.
- **Spatial discretization**: P1 (linear) isoparametric trilinear finite elements on hex meshes, linear FE on tet meshes. Mass-lumping used (identity mass = trapezoidal quadrature).
- **Time discretization**: IMEX (implicit-explicit), decoupling ODEs (gating/concentrations) from PDEs (Vm, φ_e). Diffusion implicit, reaction explicit. Luo-Rudy I ionic model.
- **Coupled method**: Solve one SPD linear system with unknowns (Vm^{n+1}, φ_e^{n+1}) at each timestep.
- **Uncoupled method**: Solve parabolic PDE for Vm (with lagged φ_e), then elliptic PDE for φ_e using the new Vm, then a second parabolic update. This is the standard Gauss-Seidel decoupling.
- **Linear solver**: PCG with Multilevel Hybrid Schwarz preconditioning.

## Key Equations / Results

The PE bidomain as stated in the paper (eq 1):
```
C_m ∂v/∂t - ∇·(D_i∇v) - ∇·(D_i∇u_e) + I_ion(v, w, c) = 0
∇·(D_i∇v) + ∇·((D_i+D_e)∇u_e) = -I_app^e
∂w/∂t - R(v, w) = 0
∂c/∂t - S(v, w, c) = 0
```
with Neumann BC `n·D_i∇(v+u_e) = 0`, `n·(D_i+D_e)∇u_e + n·D_i∇v = 0` and zero-mean compatibility on I_app and u_e.

Discrete block system (eq 4):
```
C_m M [ v'; u_e' ]  +  A [ v; u_e ]  +  M·[ I_ion; 0 ]  =  [ 0; M·I_app^e ]

where  A = [ A_i   A_i     ]
           [ A_i   A_i+A_e ]
```

## Connections to Our Models

### Relevant Engine Components

- **Bidomain V1** (`Bidomain/Engine_V1/cardiac_sim/simulation/classical/solver/diffusion_stepping/decoupled_gs.py`): implements exactly the decoupled strategy described in this paper (Gauss-Seidel split — parabolic solve for Vm, then elliptic solve for φ_e). Their coupled vs uncoupled comparison is the reference for justifying this design choice.

### Agreements

- Our Bidomain V1 uses the decoupled GS approach and observes the same pattern: spectral/PCG solves on the elliptic subproblem converge cleanly; the parabolic is solved via DCT/DST/FFT in the isotropic uniform-grid regime (faster than preconditioned CG).
- The mass-lumping assumption (identity mass matrix) aligns with our FDM convention — we skip mass matrices entirely since FDM already uses pointwise evaluation.

### Disagreements or Gaps

- They do not address the parabolic-parabolic vs parabolic-elliptic distinction — this paper is purely about solver strategy within the PE formulation.
- No hyperbolic formulation, no tissue-bath boundary artifact discussion, no LBM.

### Actionable Insights

- **Low priority: Justifies our Gauss-Seidel decoupling architecturally.** Cite this paper as the primary reference for why Bidomain V1 does PE decoupled GS instead of the coupled 2×2 block.
- **Low priority: For the hyperbolic bidomain solver, preserve the decoupled architecture.** The 2×2 block structure in the hyperbolic case has the same "upper-left = parabolic-like, lower-right = elliptic-like, off-diagonals couple them" shape that this paper established works well for GS decoupling. See `HYPERBOLIC_BIDOMAIN_MAPPING.md` §3.7 — GS decoupling still applies to the hyperbolic system.

## Limitations / Caveats

- **Not directly about our question.** The paper addresses solver efficiency *within* the PE formulation, not the physical validity of PE vs PP vs hyperbolic.
- **LR1 ionic model only** — no TTP06 or similar modern models. Solver behavior on modern ionic models may differ.
- **No tissue-bath or boundary artifact discussion.**
