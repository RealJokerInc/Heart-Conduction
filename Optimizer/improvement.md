# Engine Tuner — Improvement Roadmap

Design decisions for multi-engine integration (Bidomain V1, Monodomain V5.4, LBM V1). These are validated designs from the initial architecture phase, deferred from V1 to keep the first pass simple. Implement in V2+ after V1 proves the optimization strategy works.

---

## 1. Engine Adapter Abstraction

V1 calls Monodomain V5.4 directly. V2 introduces an `EngineAdapter` ABC so the optimizer is engine-agnostic.

```python
class EngineAdapter(ABC):
    """Interface between the optimizer and any simulation engine."""

    @abstractmethod
    def run_single_cell(
        self,
        theta_ionic: dict[str, float],
        cls: list[float],
        n_beats: int = 20,
    ) -> dict:
        """Run single-cell pacing. Return {apd_90, dvdt_max, V_trace, ...}"""

    @abstractmethod
    def run_1d_cable(
        self,
        theta_ionic: dict[str, float],
        theta_tissue: dict[str, float],
        cable_length: float = 2.0,
        dx: float = 0.02,
    ) -> float:
        """Run 1D cable, return CV (cm/s)."""

    @abstractmethod
    def run_2d_tissue(
        self,
        theta_ionic: dict[str, float],
        theta_tissue: dict[str, float],
        domain_size: tuple[float, float] = (1.0, 1.0),
        dx: float = 0.02,
    ) -> dict:
        """Run 2D tissue sim, return {cv_long, cv_trans, tissue_apd, ...}"""

    @abstractmethod
    def run_s1s2(
        self,
        theta_ionic: dict[str, float],
        theta_tissue: dict[str, float],
        s1_cl: float,
        di_values: list[float],
    ) -> list[tuple[float, float]]:
        """Run S1S2 restitution protocol in tissue. Return [(DI, APD), ...]"""

    @property
    @abstractmethod
    def tissue_param_names(self) -> list[str]:
        """Names of tissue parameters this engine exposes."""

    @property
    @abstractmethod
    def tissue_param_bounds(self) -> dict[str, tuple[float, float]]:
        """Default bounds for tissue parameters."""
```

---

## 2. Monodomain V5.4 Adapter

```python
class MonodomainAdapter(EngineAdapter):
    """Wraps Monodomain Engine V5.4 (FDM, Godunov, Rush-Larsen)."""

    tissue_param_names = ['D_long', 'D_trans']
    tissue_param_bounds = {
        'D_long':  (0.0005, 0.005),   # cm²/ms
        'D_trans': (0.0001, 0.002),
    }

    def run_1d_cable(self, theta_ionic, theta_tissue, ...):
        grid = StructuredGrid.create_rectangle(
            Lx=cable_length, Ly=dx, Nx=int(cable_length/dx), Ny=1,
            device=self.device
        )
        sim = MonodomainSimulation(
            mesh=grid,
            ionic_model=self.ionic_model,
            cell_type=self.cell_type,
            D=theta_tissue['D_long'],  # or D_trans for transverse cable
            splitting='godunov',
            ionic_solver='rush_larsen',
            diffusion_solver='forward_euler',
            dt=self.dt,
            device=self.device,
        )
        # Apply conductance scaling to ionic model
        self._apply_theta_ionic(sim.ionic, theta_ionic)
        # Run and measure activation times
        ...
```

---

## 3. Bidomain V1 Adapter

The bidomain engine has 4 tissue parameters instead of 2: intracellular and extracellular diffusion per axis.

```python
class BidomainAdapter(EngineAdapter):
    """Wraps Bidomain Engine V1 (decoupled GS splitting, spectral/PCG solver)."""

    tissue_param_names = ['D_i_long', 'D_i_trans', 'D_e_long', 'D_e_trans']
    tissue_param_bounds = {
        'D_i_long':  (0.0003, 0.003),
        'D_i_trans': (0.00005, 0.001),
        'D_e_long':  (0.001, 0.01),
        'D_e_trans': (0.0003, 0.005),
    }

    def run_1d_cable(self, theta_ionic, theta_tissue, ...):
        grid = StructuredGrid.create_rectangle(
            Lx=cable_length, Ly=dx, Nx=int(cable_length/dx), Ny=1,
            device=self.device
        )
        sim = BidomainSimulation(
            grid=grid,
            ionic_model=self.ionic_model,
            D_i=theta_tissue['D_i_long'],  # per-axis
            D_e=theta_tissue['D_e_long'],
            splitting='godunov',
            ionic_solver='rush_larsen',
            elliptic_solver='auto',
            dt=self.dt,
            device=self.device,
        )
        self._apply_theta_ionic(sim.ionic, theta_ionic)
        ...
```

### Bidomain-specific considerations

- **chi=1.0, Cm=1.0 in operators**: D_i, D_e are pre-scaled by chi*Cm (convention from Bidomain V1)
- **Effective diffusion**: For monodomain comparison, D_eff = D_i * D_e / (D_i + D_e)
- **Elliptic solver**: `auto` selects spectral for isotropic uniform grids. For anisotropic, falls back to PCG+spectral preconditioner.
- **Parabolic coupling**: RHS uses `L_i * phi_e` (full coupling, NOT `theta * L_i * phi_e`)
- **4 params vs 2**: The optimizer has more freedom but also more degeneracy. Constraint: D_i/D_e ratio typically 0.25-0.5 (can be used as a prior or hard bound).

### Bidomain tissue parameter decomposition

```
    D_i_long  ─┬─── CV_longitudinal (with D_e_long)
    D_e_long  ─┘

    D_i_trans ─┬─── CV_transverse (with D_e_trans)
    D_e_trans ─┘

    Anisotropy ratio = CV_long / CV_trans
                     = sqrt((D_i_L * D_e_L / (D_i_L + D_e_L)) /
                            (D_i_T * D_e_T / (D_i_T + D_e_T)))

    Constraint: D_i/D_e ratio is physiologically bounded (~0.25-0.5)
    This reduces 4 free params to effectively ~2 + prior on ratio.
```

---

## 4. LBM V1 Adapter

LBM uses relaxation times (tau) instead of diffusion coefficients. The conversion is handled by existing utility functions.

```python
class LBMAdapter(EngineAdapter):
    """Wraps LBM Engine V1 (D2Q5/D2Q9, BGK/MRT)."""

    tissue_param_names = ['tau_long', 'tau_trans']
    tissue_param_bounds = {
        'tau_long':  (0.501, 2.0),    # tau > 0.5 required for stability
        'tau_trans': (0.501, 2.0),
    }

    def run_1d_cable(self, theta_ionic, theta_tissue, ...):
        from src.simulation import LBMSimulation
        from src.diffusion import tau_from_D, sigma_to_D

        sim = LBMSimulation(
            domain_size=(int(cable_length/dx), 1),
            lattice=D2Q5,
            ionic_model=self.ionic_model,
            collision='bgk',
            tau=theta_tissue['tau_long'],
            dx=dx, dt=self.dt,
            device=self.device,
        )
        self._apply_theta_ionic(sim.ionic, theta_ionic)
        ...
```

### LBM-specific considerations

- **Known CV offset**: LBM produces ~35% higher CV than FDM at the same resolution due to numerical dispersion. The optimizer must account for this — either:
  - (a) Accept LBM CV as "LBM-correct" and tune to LBM-adjusted targets
  - (b) Apply a correction factor derived from pure-diffusion benchmarks
- **D2Q5 vs D2Q9**: D2Q5 is isotropic only (no D_xy cross-term). D2Q9 needed for full anisotropic tensor. For per-axis anisotropy (D_L ≠ D_T, D_xy = 0), D2Q5 with MRT suffices.
- **tau_from_D()**: `tau = 0.5 + D / (cs² * dt)` where cs² = 1/3 for D2Q5. This conversion is in `Monodomain/LBM_V1/src/diffusion.py`.
- **Stability**: tau must be > 0.5. Very large tau (>2) causes numerical diffusion. Optimal range: 0.55-1.5.

---

## 5. Cross-Engine Validation

After tuning on one engine, validate that the tuned parameters produce consistent results across all engines.

```python
class CrossEngineValidator:
    """Run the same θ* on all engines and compare."""

    def validate(
        self,
        theta_ionic: dict[str, float],
        theta_tissue_mono: dict[str, float],
        engines: list[str] = ['monodomain', 'bidomain', 'lbm'],
    ) -> dict:
        results = {}
        for engine in engines:
            adapter = self._get_adapter(engine)
            theta_tissue = self._convert_tissue_params(theta_tissue_mono, engine)
            cv = adapter.run_1d_cable(theta_ionic, theta_tissue)
            apd = adapter.run_2d_tissue(theta_ionic, theta_tissue)
            results[engine] = {'cv': cv, 'tissue_apd': apd}

        return {
            'mono_vs_bidomain_cv': abs(results['monodomain']['cv'] -
                                       results['bidomain']['cv']) /
                                       results['monodomain']['cv'] * 100,
            'mono_vs_lbm_cv': abs(results['monodomain']['cv'] -
                                  results['lbm']['cv']) /
                                  results['monodomain']['cv'] * 100,
            # Expected: mono ↔ bidomain < 1%, mono ↔ LBM ~35%
        }
```

### Tissue parameter conversion between engines

```
    Monodomain → Bidomain:
        D_mono = D_i * D_e / (D_i + D_e)
        With ratio r = D_i/D_e ≈ 0.3 (typical):
            D_i = D_mono * (1 + r) / r
            D_e = D_mono * (1 + r)

    Monodomain → LBM:
        tau = tau_from_D(D_mono, dx, dt)
        = 0.5 + D_mono / (cs² * dt)
```

---

## 6. Anisotropic Diffusion Tensor

V1 uses per-axis isotropic diffusion (D_L, D_T with D_xy = 0). V2 extends to full tensor for fiber-oriented anisotropy.

```python
# V1: diagonal tensor
D = torch.tensor([[D_long, 0],
                   [0, D_trans]])

# V2: full tensor with fiber orientation
def build_diffusion_tensor(D_long, D_trans, fiber_angle):
    """Rotate diffusion tensor to fiber direction."""
    c, s = torch.cos(fiber_angle), torch.sin(fiber_angle)
    R = torch.tensor([[c, -s], [s, c]])
    D_fiber = torch.tensor([[D_long, 0], [0, D_trans]])
    return R @ D_fiber @ R.T

# V2 tissue params: D_long, D_trans, fiber_angle (3 params)
# Or: per-node fiber field from Builder mesh (no extra params, fixed geometry)
```

### Engine support for anisotropic tensor

| Engine | Diagonal (V1) | Full tensor (V2) |
|--------|--------------|-----------------|
| Monodomain V5.4 FDM | 9-pt stencil with harmonic averaging | Same stencil, cross-derivative terms |
| Monodomain V5.4 FEM | Mass + stiffness assembly | Natural support via element-level integration |
| Monodomain V5.4 FVM | TPFA (two-point flux approx.) | Needs MPFA for non-K-orthogonal meshes |
| Bidomain V1 | Separate D_i, D_e tensors | Same framework, more params |
| LBM V1 D2Q5 | Per-axis tau only | Cannot do full tensor (no D_xy) |
| LBM V1 D2Q9 | Per-axis tau | Full tensor via MRT relaxation |

---

## 7. Pipeline Architecture (Multi-Engine)

```
                          ┌──────────────────┐
                          │   USER INPUTS    │
                          │  targets + engine │
                          └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │  Engine Adapter  │
                          │  Factory         │
                          └────────┬─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
            ┌──────────┐  ┌──────────┐  ┌──────────┐
            │ Monodomain│  │ Bidomain │  │   LBM    │
            │ Adapter   │  │ Adapter  │  │ Adapter  │
            └─────┬────┘  └─────┬────┘  └─────┬────┘
                  │              │              │
                  └──────────────┼──────────────┘
                                 │
                                 ▼
    ╔════════════════════════════════════════════════════╗
    ║  PHASE 1: Cell Fit (engine-independent)            ║
    ║  BayesOpt qNEHVI over 8 ionic conductances         ║
    ╚════════════════════╤═══════════════════════════════╝
                         │
                         ▼
    ╔════════════════════════════════════════════════════╗
    ║  PHASE 2: Tissue Fit (engine-specific params)      ║
    ║  Mono: D_L, D_T (2 params)                         ║
    ║  Bidomain: D_i_L, D_i_T, D_e_L, D_e_T (4 params) ║
    ║  LBM: tau_L, tau_T (2 params)                      ║
    ╚════════════════════╤═══════════════════════════════╝
                         │
                         ▼
    ╔════════════════════════════════════════════════════╗
    ║  PHASE 3: Joint Refinement (10-14 params)          ║
    ║  GP emulator + NSGA-II on surrogate                ║
    ╚════════════════════╤═══════════════════════════════╝
                         │
                         ▼
    ╔════════════════════════════════════════════════════╗
    ║  PHASE 4: Validate                                 ║
    ║  + PHASE 5 (V2): Cross-Engine Validation           ║
    ║    Run same θ* on all engines, compare CV/APD      ║
    ╚════════════════════════════════════════════════════╝
```

---

## 8. HMC on GP Surrogate (V2+)

After the GP emulator is built in Phase 3, use HMC/NUTS to sample the full posterior distribution over parameters.

```python
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

def emulator_model(emulator, targets):
    """Pyro model for HMC sampling on the GP emulator."""
    theta = pyro.sample('theta', dist.Uniform(bounds_low, bounds_high))

    # Emulator prediction (fast)
    pred = emulator.predict(theta)

    # Likelihood
    pyro.sample('cv_L', dist.Normal(pred['cv_L'], 1.0), obs=targets.cv_longitudinal)
    pyro.sample('cv_T', dist.Normal(pred['cv_T'], 1.0), obs=targets.cv_transverse)
    pyro.sample('apd', dist.Normal(pred['tissue_apd'], 3.0), obs=targets.tissue_apd_90)

nuts = NUTS(lambda: emulator_model(emulator, targets))
mcmc = MCMC(nuts, num_samples=2000, warmup_steps=1000)
mcmc.run()

# Posterior analysis
samples = mcmc.get_samples()
# → parameter distributions, correlations, identifiability
```

This gives full uncertainty quantification without running HMC through the stiff ODE solver directly.

---

## 9. Sensitivity Pre-Screen (V2+)

Before optimization, run Sobol sensitivity analysis to identify which of the 17+ TTP06 conductances actually matter for CV and APD.

```python
from SALib.sample import saltelli
from SALib.analyze import sobol

problem = {
    'num_vars': 17,
    'names': ['GNa', 'GNaL', 'GCaL', ...],
    'bounds': [[0.5, 2.0], [0.3, 3.0], ...],
}

# Generate samples and evaluate
X = saltelli.sample(problem, 256)  # 256 × (2×17+2) = 9216 evaluations
Y_apd = np.array([run_single_cell(x)['apd_90'] for x in X])
Y_cv = np.array([run_tissue_cv(x) for x in X])

# Analyze
Si_apd = sobol.analyze(problem, Y_apd)
Si_cv = sobol.analyze(problem, Y_cv)

# Result: rank parameters by total-order Sobol index
# Keep only those with S_T > 0.01 for optimization
# Typically reduces 17 → 6-8 parameters
```

This justifies the 8-parameter subset used in V1 and may reveal that fewer parameters suffice.
