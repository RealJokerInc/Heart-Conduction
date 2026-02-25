# Engine V5.1: PyTorch GPU Implementation

## Overview

Engine V5.1 is a complete rewrite of the cardiac electrophysiology simulation engine using PyTorch for GPU acceleration. This document provides detailed technical specifications for the implementation.

### Design Principles

1. **GPU-First**: All computations run on CUDA GPU (no CPU fallback)
2. **Float64 Precision**: Double precision for numerical stability in stiff ODEs
3. **Backward Compatible**: Can load/compare states from V5 for validation
4. **Batch Operations**: Vectorized operations over entire tissue grid
5. **Modular Architecture**: Separable ionic, diffusion, and simulation components

### Target Performance

| Metric | V5 (CPU) | V5.1 (GPU) Target |
|--------|----------|-------------------|
| Cells/second | 8.4M | 50-100M |
| 500x500 step time | 30 ms | 3-5 ms |
| Memory | 90 MB (CPU) | 500 MB (GPU) |

---

## Mathematical Foundation

### Monodomain Equation

The monodomain model describes electrical propagation in cardiac tissue:

```
χ · Cm · ∂V/∂t = -χ · Iion(V, u) + ∇·(D·∇V) + Istim
```

Where:
- `V`: Transmembrane potential (mV)
- `u`: Vector of 40 state variables (gating, concentrations)
- `χ = 1400 cm⁻¹`: Surface-to-volume ratio
- `Cm = 1.0 µF/cm²`: Membrane capacitance
- `Iion`: Total ionic current density (µA/µF)
- `D`: Diffusion tensor (cm²/ms)
- `Istim`: Stimulus current (µA/µF)

### Operator Splitting

We use Godunov (first-order) splitting:

```
V^(n+1) = D(dt) · I(dt) · V^n
```

Where:
- `I(dt)`: Ionic operator (ODE integration)
- `D(dt)`: Diffusion operator (PDE discretization)

For each timestep:
1. **Ionic Step**: Integrate `dV/dt = -Iion/Cm` and update gating/concentrations
2. **Diffusion Step**: Apply `dV/dt = ∇·(D·∇V) / (χ·Cm)`

---

## Stage 1: Ionic Model (ORd 2011)

### Overview

The O'Hara-Rudy dynamic (ORd) model contains:
- 41 state variables (1 voltage + 40 internal states)
- 15 ionic currents
- CaMKII signaling cascade
- Detailed calcium handling (SR, cytosol, subspace)

### Stage 1.1: Tensor Infrastructure

**Objective**: Establish PyTorch tensor conventions and device management.

#### State Tensor Layout

```python
# Shape: (ny, nx, 41) for tissue, (41,) for single cell
# Stored as contiguous float64 tensor on GPU

class StateIndex:
    """Indices into the state tensor (last dimension)."""
    V = 0       # Membrane potential (mV)
    nai = 1     # Intracellular Na+ (mM)
    nass = 2    # Subspace Na+ (mM)
    ki = 3      # Intracellular K+ (mM)
    kss = 4     # Subspace K+ (mM)
    cai = 5     # Intracellular Ca2+ (mM)
    cass = 6    # Subspace Ca2+ (mM)
    cansr = 7   # NSR Ca2+ (mM)
    cajsr = 8   # JSR Ca2+ (mM)
    m = 9       # INa activation
    hf = 10     # INa fast inactivation
    hs = 11     # INa slow inactivation
    j = 12      # INa recovery
    hsp = 13    # INa phosphorylated fast inact
    jp = 14     # INa phosphorylated recovery
    mL = 15     # INaL activation
    hL = 16     # INaL inactivation
    hLp = 17    # INaL phosphorylated inact
    a = 18      # Ito activation
    iF = 19     # Ito fast inactivation
    iS = 20     # Ito slow inactivation
    ap = 21     # Ito phosphorylated activation
    iFp = 22    # Ito phosphorylated fast inact
    iSp = 23    # Ito phosphorylated slow inact
    d = 24      # ICaL activation
    ff = 25     # ICaL fast inactivation
    fs = 26     # ICaL slow inactivation
    fcaf = 27   # ICaL Ca-dependent fast inact
    fcas = 28   # ICaL Ca-dependent slow inact
    jca = 29    # ICaL Ca-dependent recovery
    nca = 30    # ICaL Ca-dependent factor
    ffp = 31    # ICaL phosphorylated fast inact
    fcafp = 32  # ICaL phosphorylated Ca-dep inact
    xrf = 33    # IKr fast activation
    xrs = 34    # IKr slow activation
    xs1 = 35    # IKs activation 1
    xs2 = 36    # IKs activation 2
    xk1 = 37    # IK1 activation
    Jrelnp = 38 # SR release (non-phosphorylated)
    Jrelp = 39  # SR release (phosphorylated)
    CaMKt = 40  # CaMKII trapped fraction
```

#### Device Management

```python
class DeviceManager:
    """Centralized GPU device management."""

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for V5.1")

        self.device = torch.device('cuda')
        self.dtype = torch.float64

        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    def tensor(self, data):
        """Create tensor on GPU with float64."""
        return torch.tensor(data, dtype=self.dtype, device=self.device)

    def zeros(self, *shape):
        """Create zero tensor on GPU."""
        return torch.zeros(*shape, dtype=self.dtype, device=self.device)
```

#### Validation 1.1

- [ ] Verify CUDA availability check works
- [ ] Confirm tensors created on correct device
- [ ] Verify float64 precision maintained
- [ ] Test tensor indexing matches V5 StateIndex

---

### Stage 1.2: Gating Kinetics

**Objective**: Implement steady-state and time constant functions for all gates.

#### Mathematical Form

Most gates follow Hodgkin-Huxley kinetics:

```
dx/dt = (x_inf - x) / tau_x
```

Where:
- `x_inf(V)`: Voltage-dependent steady-state
- `tau_x(V)`: Voltage-dependent time constant

#### Steady-State Functions

Typically sigmoidal:
```
x_inf = 1 / (1 + exp(-(V - V_half) / k))
```

#### Implementation Pattern

```python
# All functions operate on tensors (batched over tissue)

def INa_m_inf(V: torch.Tensor) -> torch.Tensor:
    """INa activation steady-state."""
    return 1.0 / (1.0 + torch.exp(-(V + 39.57) / 9.871))

def INa_m_tau(V: torch.Tensor) -> torch.Tensor:
    """INa activation time constant (ms)."""
    return 1.0 / (6.765 * torch.exp((V + 11.64) / 34.77) +
                  8.552 * torch.exp(-(V + 77.42) / 5.955))
```

#### Safe Exponential

Prevent overflow in exponential calculations:

```python
def safe_exp(x: torch.Tensor, limit: float = 80.0) -> torch.Tensor:
    """Clamped exponential to prevent overflow."""
    return torch.exp(torch.clamp(x, -limit, limit))
```

#### Gate List (15 gating variables)

| Gate | Current | Type | Has Phosphorylated Form |
|------|---------|------|------------------------|
| m | INa | activation | No |
| hf, hs | INa | inactivation | Yes (hsp) |
| j | INa | recovery | Yes (jp) |
| mL | INaL | activation | No |
| hL | INaL | inactivation | Yes (hLp) |
| a | Ito | activation | Yes (ap) |
| iF, iS | Ito | inactivation | Yes (iFp, iSp) |
| d | ICaL | activation | No |
| ff, fs | ICaL | inactivation | Yes (ffp) |
| fcaf, fcas | ICaL | Ca-inactivation | Yes (fcafp) |
| jca | ICaL | recovery | No |
| xrf, xrs | IKr | activation | No |
| xs1, xs2 | IKs | activation | No |
| xk1 | IK1 | activation | No |

#### Validation 1.2

For each gating function, compare against V5 at voltage range [-100, +60] mV:

```python
def validate_gating(v5_func, v51_func, name):
    V_test = torch.linspace(-100, 60, 161, dtype=torch.float64, device='cuda')
    V_numpy = V_test.cpu().numpy()

    v5_result = np.array([v5_func(v) for v in V_numpy])
    v51_result = v51_func(V_test).cpu().numpy()

    max_error = np.max(np.abs(v5_result - v51_result))
    assert max_error < 1e-10, f"{name}: max error {max_error}"
```

- [ ] All 15 steady-state functions match V5 (< 1e-10 error)
- [ ] All 15 time constant functions match V5 (< 1e-10 error)
- [ ] Phosphorylated variants match V5

---

### Stage 1.3: Ion Currents

**Objective**: Implement all 15 ionic currents.

#### Reversal Potentials

Nernst equation:
```
E_X = (R·T/F) · ln([X]_o / [X]_i)
```

At 37°C: `R·T/F = 8.314 · 310.15 / 96485 = 26.71 mV`

```python
RTF = 26.71  # mV (R*T/F at 37°C)

def E_Na(nai: torch.Tensor, nao: float = 140.0) -> torch.Tensor:
    """Sodium reversal potential."""
    return RTF * torch.log(nao / nai)

def E_K(ki: torch.Tensor, ko: float = 5.4) -> torch.Tensor:
    """Potassium reversal potential."""
    return RTF * torch.log(ko / ki)

def E_Ca(cai: torch.Tensor, cao: float = 1.8) -> torch.Tensor:
    """Calcium reversal potential (2+ valence)."""
    return 0.5 * RTF * torch.log(cao / cai)
```

#### Current Implementations

**INa (Fast Sodium)**
```python
def I_Na(V, m, hf, hs, j, hsp, jp, nai, nao, fCaMKp, GNa):
    """Fast sodium current."""
    ENa = RTF * torch.log(nao / nai)

    # Non-phosphorylated
    h = 0.99 * hf + 0.01 * hs
    INa_np = GNa * m**3 * h * j * (V - ENa)

    # Phosphorylated
    hp = 0.99 * hsp + 0.01 * hs
    INa_p = GNa * m**3 * hp * jp * (V - ENa)

    # Weighted by CaMKII
    return (1.0 - fCaMKp) * INa_np + fCaMKp * INa_p
```

**ICaL (L-type Calcium)**

Most complex current with:
- Voltage-dependent gating (d, ff, fs)
- Calcium-dependent inactivation (fcaf, fcas, jca, nca)
- Phosphorylation effects
- Permeation of Ca²⁺, Na⁺, K⁺

```python
def I_CaL(V, d, ff, fs, fcaf, fcas, jca, nca, ffp, fcafp,
          cass, nass, kss, cao, nao, ko, fCaMKp, params):
    """L-type calcium current (with Na and K components)."""

    # Voltage-dependent driving force (GHK)
    vfrt = V / RTF
    vffrt = V * F / (R * T)  # For GHK

    # ... (detailed GHK permeation equations)

    # Gating
    f = 0.4 * ff + 0.6 * fs
    fca = 0.3 * fcaf + 0.7 * fcas

    # ... (phosphorylation weighting)

    return ICaL, ICaNa, ICaK
```

#### Current List

| Current | Description | Key Dependencies |
|---------|-------------|------------------|
| INa | Fast sodium | m, hf, hs, j, nai |
| INaL | Late sodium | mL, hL, nai |
| Ito | Transient outward K+ | a, iF, iS, ki |
| ICaL | L-type Ca2+ | d, ff, fs, fcaf, cass |
| ICaNa | ICaL Na+ component | (same as ICaL) |
| ICaK | ICaL K+ component | (same as ICaL) |
| IKr | Rapid delayed rectifier | xrf, xrs, ki |
| IKs | Slow delayed rectifier | xs1, xs2, ki, nai |
| IK1 | Inward rectifier K+ | xk1, ki |
| INaCa | Na+/Ca2+ exchanger | nai, cai, cass |
| INaK | Na+/K+ pump | nai, ki |
| INab | Background Na+ | nai |
| ICab | Background Ca2+ | cai |
| IpCa | Sarcolemmal Ca2+ pump | cai |
| IKb | Background K+ | ki |

#### Validation 1.3

For each current, test at standard state conditions:

```python
def validate_current(v5_model, v51_func, current_name):
    # Test at multiple voltages with standard concentrations
    V_test = torch.linspace(-100, 60, 17, device='cuda', dtype=torch.float64)

    # Use V5 initial state for concentrations
    state = v5_model.get_initial_state()

    # Compare current values
    for V in V_test:
        I_v5 = v5_model.compute_current(current_name, V, state)
        I_v51 = v51_func(V, ...).item()

        rel_error = abs(I_v5 - I_v51) / max(abs(I_v5), 1e-12)
        assert rel_error < 1e-6, f"{current_name} at V={V}: rel_error={rel_error}"
```

- [ ] All 15 currents match V5 within 1e-6 relative error
- [ ] I-V curves qualitatively correct
- [ ] Total Iion sums correctly

---

### Stage 1.4: Calcium Handling

**Objective**: Implement SR calcium dynamics and buffering.

#### Compartments

```
[Ca2+]_subspace (cass) ←→ [Ca2+]_cytosol (cai) ←→ [Ca2+]_NSR (cansr) ←→ [Ca2+]_JSR (cajsr)
```

#### SR Release (Jrel)

Calcium-induced calcium release from JSR:

```python
def J_rel(cajsr, cass, Jrelnp, Jrelp, fCaMKp, params):
    """SR calcium release flux."""
    # Release rate depends on JSR load and subspace Ca

    # Non-phosphorylated
    Jrel_inf_np = params.a_rel * (-I_CaL) / (1.0 + (params.cajsr_half / cajsr)**8)
    tau_rel_np = params.bt / (1.0 + 0.0123 / cajsr)
    tau_rel_np = torch.clamp(tau_rel_np, min=0.001)  # Minimum 1 µs

    # Phosphorylated (faster release)
    Jrel_inf_p = params.a_relp * (-I_CaL) / (1.0 + (params.cajsr_half / cajsr)**8)
    tau_rel_p = params.btp / (1.0 + 0.0123 / cajsr)
    tau_rel_p = torch.clamp(tau_rel_p, min=0.001)

    # Update release variables (Rush-Larsen)
    Jrelnp_new = Jrel_inf_np - (Jrel_inf_np - Jrelnp) * torch.exp(-dt / tau_rel_np)
    Jrelp_new = Jrel_inf_p - (Jrel_inf_p - Jrelp) * torch.exp(-dt / tau_rel_p)

    # Total release
    Jrel = (1.0 - fCaMKp) * Jrelnp_new + fCaMKp * Jrelp_new

    return Jrel, Jrelnp_new, Jrelp_new
```

#### SERCA Uptake (Jup)

Calcium uptake into NSR:

```python
def J_up(cai, cansr, fCaMKp, params):
    """SERCA calcium uptake flux."""
    # Michaelis-Menten kinetics
    Jupnp = params.Jup_max * cai / (cai + params.Kup)
    Jupp = params.Jup_max * 1.75 * cai / (cai + params.Kup - 0.00017)

    # Leak from NSR
    Jleak = params.Jleak_max * cansr / params.cansr_max

    Jup = (1.0 - fCaMKp) * Jupnp + fCaMKp * Jupp - Jleak
    return Jup
```

#### Diffusion Fluxes

```python
def J_diff_Ca(cass, cai, tau_diff=0.2):
    """Calcium diffusion from subspace to cytosol."""
    return (cass - cai) / tau_diff

def J_diff_Na(nass, nai, tau_diff=2.0):
    """Sodium diffusion from subspace to cytosol."""
    return (nass - nai) / tau_diff

def J_tr(cansr, cajsr, tau_tr=100.0):
    """Calcium transfer from NSR to JSR."""
    return (cansr - cajsr) / tau_tr
```

#### Buffering

```python
def buffer_factor_cai(cai, params):
    """Cytosolic calcium buffer factor."""
    # CMDN (calmodulin) + TRPN (troponin)
    bcai = 1.0 / (1.0 + params.CMDN_max * params.Km_CMDN / (params.Km_CMDN + cai)**2
                     + params.TRPN_max * params.Km_TRPN / (params.Km_TRPN + cai)**2)
    return bcai

def buffer_factor_cass(cass, params):
    """Subspace calcium buffer factor."""
    # BSR + BSL
    bcass = 1.0 / (1.0 + params.BSR_max * params.Km_BSR / (params.Km_BSR + cass)**2
                      + params.BSL_max * params.Km_BSL / (params.Km_BSL + cass)**2)
    return bcass

def buffer_factor_cajsr(cajsr, params):
    """JSR calcium buffer factor."""
    # CSQN (calsequestrin)
    bcajsr = 1.0 / (1.0 + params.CSQN_max * params.Km_CSQN / (params.Km_CSQN + cajsr)**2)
    return bcajsr
```

#### Concentration Updates

```python
def update_calcium(cai, cass, cansr, cajsr, ICaL, ICab, IpCa, INaCa_i, INaCa_ss,
                   Jrel, Jup, dt, params):
    """Update all calcium concentrations."""

    # Geometry factors
    vcell = params.vcell
    vmyo = 0.68 * vcell    # Cytosol volume
    vnsr = 0.0552 * vcell  # NSR volume
    vjsr = 0.0048 * vcell  # JSR volume
    vss = 0.02 * vcell     # Subspace volume

    # Fluxes
    Jdiff = J_diff_Ca(cass, cai)
    Jtr = J_tr(cansr, cajsr)

    # Buffer factors
    bcai = buffer_factor_cai(cai, params)
    bcass = buffer_factor_cass(cass, params)
    bcajsr = buffer_factor_cajsr(cajsr, params)

    # Concentration derivatives
    Acap = 2.0 * params.Acap  # Membrane area

    dcai = bcai * (-((IpCa + ICab) - 2.0 * INaCa_i) * Acap / (2.0 * F * vmyo)
                   - Jup * vnsr / vmyo + Jdiff * vss / vmyo)

    dcass = bcass * (-(ICaL - 2.0 * INaCa_ss) * Acap / (2.0 * F * vss)
                     + Jrel * vjsr / vss - Jdiff)

    dcansr = Jup - Jtr * vjsr / vnsr

    dcajsr = bcajsr * (Jtr - Jrel)

    # Forward Euler update
    cai_new = cai + dt * dcai
    cass_new = cass + dt * dcass
    cansr_new = cansr + dt * dcansr
    cajsr_new = cajsr + dt * dcajsr

    return cai_new, cass_new, cansr_new, cajsr_new
```

#### Validation 1.4

- [ ] Buffer factors match V5 at test Ca concentrations
- [ ] SR release dynamics match V5 during AP
- [ ] SERCA uptake matches V5
- [ ] Steady-state Ca concentrations match V5 at rest

---

### Stage 1.5: CaMKII Signaling

**Objective**: Implement calcium/calmodulin-dependent protein kinase II.

#### CaMKII Activation

```python
def compute_CaMKa(CaMKt, cass, CaMKo=0.05, KmCaM=0.0015):
    """Compute active CaMKII fraction."""
    CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
    CaMKa = CaMKb + CaMKt
    return CaMKb, CaMKa

def fCaMKp(CaMKa, KmCaMK=0.15):
    """Phosphorylation factor for CaMKII-dependent effects."""
    return 1.0 / (1.0 + KmCaMK / CaMKa)
```

#### CaMKII Trapping Update

```python
def update_CaMKt(CaMKt, CaMKb, CaMKa, dt, aCaMK=0.05, bCaMK=0.00068):
    """Update trapped CaMKII fraction."""
    dCaMKt = aCaMK * CaMKb * CaMKa - bCaMK * CaMKt
    return CaMKt + dt * dCaMKt
```

#### Validation 1.5

- [ ] CaMKa matches V5 at various cass levels
- [ ] CaMKt dynamics match V5 during AP train
- [ ] fCaMKp factor matches V5

---

### Stage 1.6: Rush-Larsen Integration

**Objective**: Implement efficient exponential integration for gating variables.

#### Mathematical Basis

For ODEs of the form:
```
dx/dt = (x_inf - x) / tau
```

The exact solution over timestep dt is:
```
x(t+dt) = x_inf - (x_inf - x(t)) * exp(-dt/tau)
```

This is unconditionally stable for any dt (though accuracy requires dt << tau).

#### Implementation

```python
def rush_larsen_update(x: torch.Tensor, x_inf: torch.Tensor,
                       tau: torch.Tensor, dt: float) -> torch.Tensor:
    """Rush-Larsen exponential update for gating variables."""
    return x_inf - (x_inf - x) * torch.exp(-dt / tau)
```

#### Batched Gate Update

```python
def update_all_gates(states: torch.Tensor, dt: float) -> torch.Tensor:
    """Update all gating variables using Rush-Larsen."""
    V = states[..., StateIndex.V]

    # INa gates
    m_inf = INa_m_inf(V)
    m_tau = INa_m_tau(V)
    states[..., StateIndex.m] = rush_larsen_update(
        states[..., StateIndex.m], m_inf, m_tau, dt
    )

    # ... repeat for all 15+ gating variables

    return states
```

#### Validation 1.6

- [ ] Single gate integration matches analytical solution
- [ ] All gates stable at dt = 0.01 ms
- [ ] Results match V5 after 1000 steps within 1e-6

---

### Stage 1.7: Single Cell Model

**Objective**: Complete single-cell ORd model with all components integrated.

#### Model Class

```python
class ORdModel:
    """O'Hara-Rudy 2011 ventricular myocyte model (PyTorch GPU)."""

    def __init__(self, celltype: CellType = CellType.ENDO, device: str = 'cuda'):
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.celltype = celltype

        # Load parameters for cell type
        self.params = self._get_parameters(celltype)

    def get_initial_state(self) -> torch.Tensor:
        """Return initial state tensor (41 values)."""
        # Standard ORd initial conditions
        state = torch.zeros(41, dtype=self.dtype, device=self.device)
        state[StateIndex.V] = -87.5
        state[StateIndex.nai] = 7.0
        state[StateIndex.nass] = 7.0
        state[StateIndex.ki] = 145.0
        state[StateIndex.kss] = 145.0
        state[StateIndex.cai] = 1e-4
        state[StateIndex.cass] = 1e-4
        state[StateIndex.cansr] = 1.2
        state[StateIndex.cajsr] = 1.2
        # ... gating variables at steady-state for V=-87.5
        return state

    def step(self, state: torch.Tensor, dt: float, Istim: float = 0.0) -> torch.Tensor:
        """Advance model by one timestep."""
        # 1. Compute CaMKII activation
        CaMKb, CaMKa = compute_CaMKa(state[StateIndex.CaMKt], state[StateIndex.cass])
        fCaMKp = compute_fCaMKp(CaMKa)

        # 2. Update gating variables (Rush-Larsen)
        state = update_all_gates(state, dt)

        # 3. Compute all currents
        currents = self.compute_currents(state, CaMKa, fCaMKp)

        # 4. Compute total ionic current
        Iion = sum(currents.values())

        # 5. Update voltage
        dV = -dt * (Iion + Istim) / self.params.Cm
        state[StateIndex.V] = state[StateIndex.V] + dV

        # 6. Update concentrations
        state = self.update_concentrations(state, currents, dt)

        # 7. Update CaMKII
        state[StateIndex.CaMKt] = update_CaMKt(
            state[StateIndex.CaMKt], CaMKb, CaMKa, dt
        )

        return state
```

#### Validation 1.7

Compare single AP against V5:

```python
def validate_single_cell_ap():
    v5_model = V5_ORdModel(celltype=CellType.ENDO)
    v51_model = V51_ORdModel(celltype=CellType.ENDO)

    dt = 0.01  # ms
    t_end = 500.0  # ms
    stim_start = 10.0
    stim_duration = 1.0
    stim_amplitude = -80.0

    # Run both models
    V_v5 = []
    V_v51 = []

    state_v5 = v5_model.get_initial_state()
    state_v51 = v51_model.get_initial_state()

    for t in np.arange(0, t_end, dt):
        Istim = stim_amplitude if stim_start <= t < stim_start + stim_duration else 0.0

        state_v5 = v5_model.step(state_v5, dt, Istim)
        state_v51 = v51_model.step(state_v51, dt, Istim)

        V_v5.append(state_v5[0])
        V_v51.append(state_v51[0].cpu().item())

    # Compare metrics
    V_v5 = np.array(V_v5)
    V_v51 = np.array(V_v51)

    max_diff = np.max(np.abs(V_v5 - V_v51))
    apd90_v5 = compute_apd90(V_v5, dt)
    apd90_v51 = compute_apd90(V_v51, dt)

    print(f"Max voltage difference: {max_diff:.4f} mV")
    print(f"APD90 V5: {apd90_v5:.1f} ms, V5.1: {apd90_v51:.1f} ms")

    assert max_diff < 1.0, f"Voltage difference too large: {max_diff} mV"
    assert abs(apd90_v5 - apd90_v51) < apd90_v5 * 0.01, "APD90 differs by more than 1%"
```

Validation criteria (moderate):
- [ ] Max voltage difference < 1.0 mV
- [ ] APD90 within 1% of V5
- [ ] Resting potential within 0.1 mV
- [ ] Peak voltage within 1 mV
- [ ] dV/dt_max within 5%

---

### Stage 1.8: Batch Cell Operations

**Objective**: Extend single-cell model to operate on tissue arrays.

#### Tissue State Tensor

```python
# Shape: (ny, nx, 41)
# All operations vectorized over first two dimensions

def step_tissue(states: torch.Tensor, dt: float,
                stim_mask: torch.Tensor, stim_amplitude: float) -> torch.Tensor:
    """Advance all cells by one timestep."""

    # Stimulus current (applied where mask is True)
    Istim = torch.where(stim_mask.unsqueeze(-1),
                        torch.tensor(-stim_amplitude, device=states.device),
                        torch.tensor(0.0, device=states.device))

    # All operations are batched over (ny, nx)
    V = states[..., StateIndex.V]

    # CaMKII (batched)
    CaMKb, CaMKa = compute_CaMKa(states[..., StateIndex.CaMKt],
                                  states[..., StateIndex.cass])
    fCaMKp_val = fCaMKp(CaMKa)

    # Update gates (batched Rush-Larsen)
    states = update_all_gates(states, dt)

    # Compute currents (batched)
    Iion = compute_total_current(states, CaMKa, fCaMKp_val)

    # Update voltage
    states[..., StateIndex.V] = V - dt * (Iion + Istim[..., 0]) / Cm

    # Update concentrations (batched)
    states = update_concentrations_batched(states, dt)

    # Update CaMKII (batched)
    states[..., StateIndex.CaMKt] = update_CaMKt(
        states[..., StateIndex.CaMKt], CaMKb, CaMKa, dt
    )

    return states
```

#### Validation 1.8

- [ ] 100x100 tissue initializes correctly
- [ ] All cells at rest match single-cell resting state
- [ ] Single stimulated cell matches single-cell AP
- [ ] Memory usage scales linearly with cell count

---

## Stage 2: Diffusion Operator

### Overview

The diffusion term models electrical coupling between cells:
```
∇·(D·∇V) where D = [D_xx  D_xy]
                   [D_xy  D_yy]
```

### Stage 2.1: FVM Discretization

**Objective**: Implement Finite Volume Method for diffusion.

#### Control Volume Formulation

For cell (i,j), integrate over control volume Ω_ij:
```
∫∫_Ω ∇·(D·∇V) dA = ∮_∂Ω (D·∇V)·n dl
```

Approximate as sum of fluxes through 4 faces:
```
diff[i,j] = (F_e - F_w) / dx + (F_n - F_s) / dy
```

Where fluxes use face-centered gradients:

```python
def compute_diffusion_fvm(V: torch.Tensor, D_xx: float, D_yy: float,
                          D_xy: float, dx: float, dy: float) -> torch.Tensor:
    """Compute div(D·grad(V)) using FVM."""
    ny, nx = V.shape

    # Gradients at cell faces
    # East face: between (i,j) and (i,j+1)
    dV_dx_e = (V[:, 1:] - V[:, :-1]) / dx  # (ny, nx-1)
    dV_dy_e = (V[1:, 1:] + V[1:, :-1] - V[:-1, 1:] - V[:-1, :-1]) / (4 * dy)  # (ny-1, nx-1)

    # Flux through east face
    F_e = D_xx * dV_dx_e + D_xy * dV_dy_e  # (ny, nx-1) approximately

    # ... similar for west, north, south faces

    # Sum fluxes
    diff = torch.zeros_like(V)
    diff[:, 1:-1] += (F_e[:, 1:] - F_e[:, :-1]) / dx
    diff[1:-1, :] += (F_n[1:, :] - F_n[:-1, :]) / dy

    return diff
```

#### Boundary Conditions

No-flux (Neumann) boundaries:
```python
# At boundaries, flux = 0
# Implemented by not adding flux contribution at edges
```

#### Validation 2.1

- [ ] Uniform V → zero diffusion
- [ ] Linear gradient → constant diffusion
- [ ] Quadratic profile → known analytical result
- [ ] Compare against V5 FVM on test cases

---

### Stage 2.2: Isotropic Diffusion

**Objective**: Implement simplified isotropic case (D_xy = 0, D_xx = D_yy = D).

```python
def compute_diffusion_isotropic(V: torch.Tensor, D: float,
                                 dx: float, dy: float) -> torch.Tensor:
    """Isotropic diffusion using 5-point stencil."""
    # Laplacian: (V[i+1,j] + V[i-1,j] - 2*V[i,j]) / dx^2 + (V[i,j+1] + V[i,j-1] - 2*V[i,j]) / dy^2

    # Using conv2d for efficiency
    kernel = torch.tensor([
        [0, 1/dy**2, 0],
        [1/dx**2, -2/dx**2 - 2/dy**2, 1/dx**2],
        [0, 1/dy**2, 0]
    ], dtype=V.dtype, device=V.device)

    # Pad for boundary conditions (replicate = no-flux)
    V_padded = F.pad(V.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='replicate')

    laplacian = F.conv2d(V_padded, kernel.unsqueeze(0).unsqueeze(0))[0, 0]

    return D * laplacian
```

#### Validation 2.2

- [ ] Matches analytical Laplacian for test functions
- [ ] No-flux boundaries preserve total voltage
- [ ] Matches V5 isotropic diffusion within 1e-10

---

### Stage 2.3: Anisotropic Diffusion Tensor

**Objective**: Handle fiber-aligned anisotropy.

#### Tensor Construction

Given fiber angle θ:
```
D = R(θ) · [D_L  0  ] · R(θ)^T
          [0   D_T]

where R(θ) = [cos(θ)  -sin(θ)]
             [sin(θ)   cos(θ)]
```

Resulting in:
```
D_xx = D_L·cos²θ + D_T·sin²θ
D_yy = D_L·sin²θ + D_T·cos²θ
D_xy = (D_L - D_T)·sinθ·cosθ
```

```python
def compute_diffusion_tensor(D_L: float, D_T: float,
                              theta: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Compute diffusion tensor components from fiber angle."""
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    cos2 = cos_t ** 2
    sin2 = sin_t ** 2

    D_xx = D_L * cos2 + D_T * sin2
    D_yy = D_L * sin2 + D_T * cos2
    D_xy = (D_L - D_T) * sin_t * cos_t

    return D_xx, D_yy, D_xy
```

#### Validation 2.3

- [ ] θ=0 gives D_xx=D_L, D_yy=D_T, D_xy=0
- [ ] θ=90° gives D_xx=D_T, D_yy=D_L, D_xy=0
- [ ] θ=45° gives symmetric tensor with D_xy ≠ 0

---

### Stage 2.4: Stability Analysis

**Objective**: Compute stable timestep for explicit integration.

#### Von Neumann Stability

For explicit Euler with 5-point stencil:
```
dt_max = 1 / (2 * D * (1/dx² + 1/dy²))
```

For anisotropic case:
```
dt_max = 1 / (2 * (D_xx/dx² + D_yy/dy² + 2*|D_xy|/(dx*dy)))
```

```python
def get_stability_limit(D_xx: float, D_yy: float, D_xy: float,
                        dx: float, dy: float) -> float:
    """Compute maximum stable timestep."""
    coeff = D_xx / dx**2 + D_yy / dy**2 + 2 * abs(D_xy) / (dx * dy)
    return 1.0 / (2.0 * coeff)
```

#### Validation 2.4

- [ ] dt at 90% of limit is stable for 10000 steps
- [ ] dt at 110% of limit shows instability
- [ ] Matches V5 stability limit computation

---

### Stage 2.5: CV-Based Parameter Calibration

**Objective**: Compute D from target conduction velocity.

#### Empirical Relationship

From V5 calibration:
```
CV = k * sqrt(D)  where k ≈ 1.514 cm^0.5 / ms^0.5
```

Therefore:
```
D = (CV / k)²
```

With mesh-dependent correction:
```python
def compute_D_from_cv(cv: float, dx: float, dx_ref: float = 0.01) -> float:
    """Compute diffusion coefficient from target CV."""
    k = 1.514  # Empirical constant
    D_base = (cv / k) ** 2

    # Mesh correction (coarser mesh → reduce D)
    ratio = dx / dx_ref
    if ratio > 1.0:
        correction = 1.0 / (1.0 + 0.04 * (ratio - 1.0))
    else:
        correction = 1.0 + 0.02 * (1.0 - ratio)

    return D_base * correction
```

#### Validation 2.5

- [ ] CV within 5% of target at dx=100µm
- [ ] CV within 10% of target at dx=50µm and dx=200µm
- [ ] Matches V5 CV measurements

---

### Stage 2.6: Diffusion Operator Class

**Objective**: Encapsulate diffusion in reusable class.

```python
class DiffusionOperator:
    """GPU-accelerated diffusion operator."""

    def __init__(self, ny: int, nx: int, dx: float, dy: float,
                 D_L: float, D_T: float, fiber_angle: float = 0.0):
        self.ny = ny
        self.nx = nx
        self.dx = dx
        self.dy = dy
        self.D_L = D_L
        self.D_T = D_T

        # Precompute tensor components
        self.D_xx, self.D_yy, self.D_xy = compute_diffusion_tensor(
            D_L, D_T, torch.tensor(fiber_angle)
        )

        # Precompute convolution kernel for isotropic case
        if abs(fiber_angle) < 1e-10 and abs(D_L - D_T) < 1e-10:
            self._setup_isotropic_kernel()
        else:
            self._setup_anisotropic()

    def apply(self, V: torch.Tensor) -> torch.Tensor:
        """Compute div(D·grad(V))."""
        # Implementation depends on tensor structure
        ...

    def get_stability_limit(self) -> float:
        """Return maximum stable timestep."""
        return get_stability_limit(
            self.D_xx, self.D_yy, self.D_xy, self.dx, self.dy
        )
```

#### Validation 2.6

- [ ] Full operator matches V5 diffusion output
- [ ] Stability limit matches V5
- [ ] CV measurement matches target

---

## Stage 3: Integration and Optimization

### Stage 3.1: Monodomain Simulation Class

**Objective**: Combine ionic and diffusion into complete simulator.

```python
class MonodomainSimulation:
    """Complete monodomain tissue simulation (PyTorch GPU)."""

    def __init__(self, ny: int, nx: int, dx: float = 0.01, dy: float = 0.01,
                 cv_long: float = 0.06, cv_trans: float = 0.02,
                 fiber_angle: float = 0.0, celltype: CellType = CellType.ENDO):

        self.device = torch.device('cuda')
        self.dtype = torch.float64

        # Grid
        self.ny = ny
        self.nx = nx
        self.dx = dx
        self.dy = dy

        # Diffusion parameters from CV
        self.D_L = compute_D_from_cv(cv_long, dx)
        self.D_T = compute_D_from_cv(cv_trans, dx)

        # Initialize components
        self.ionic = ORdModel(celltype=celltype, device='cuda')
        self.diffusion = DiffusionOperator(ny, nx, dx, dy, self.D_L, self.D_T, fiber_angle)

        # State tensor (ny, nx, 41)
        self.states = self._init_states()

        # Stimulus
        self.stim_sites = []
        self.stim_amplitude = 80.0

        # Time
        self.time = 0.0

    def step(self, dt: float):
        """Advance by one timestep (Godunov splitting)."""
        # Get stimulus mask
        stim_mask = self._get_stimulus_mask(self.time)

        # Ionic step
        self.states = step_tissue(self.states, dt, stim_mask, self.stim_amplitude)

        # Diffusion step
        V = self.states[..., StateIndex.V]
        diff = self.diffusion.apply(V)
        self.states[..., StateIndex.V] = V + dt * diff

        self.time += dt
```

#### Validation 3.1

- [ ] 100x100 simulation runs without errors
- [ ] Planar wave propagates correctly
- [ ] CV matches target within 10%

---

### Stage 3.2: torch.compile Optimization

**Objective**: Enable JIT compilation for performance.

```python
@torch.compile(mode='reduce-overhead')
def step_tissue_compiled(states, dt, stim_mask, stim_amplitude, params):
    """JIT-compiled tissue step."""
    # Same implementation as step_tissue
    ...
```

#### Validation 3.2

- [ ] Compiled version matches non-compiled output
- [ ] Speedup of 2-5x observed
- [ ] No numerical drift over long simulations

---

### Stage 3.3: Memory Optimization

**Objective**: Minimize GPU memory usage.

Strategies:
1. In-place operations where possible
2. Reuse intermediate tensors
3. Avoid unnecessary copies

```python
def step_tissue_optimized(states: torch.Tensor, dt: float, ...):
    """Memory-optimized tissue step."""
    # Use views instead of copies
    V = states[..., StateIndex.V]  # View, not copy

    # In-place updates where safe
    states[..., StateIndex.m].mul_(exp_factor).add_(m_inf_contrib)

    ...
```

#### Validation 3.3

- [ ] Memory usage for 500x500 < 1 GB
- [ ] No memory leaks over 10000 steps
- [ ] Output unchanged from non-optimized version

---

### Stage 3.4: Benchmark and Profiling

**Objective**: Measure and document performance.

```python
def benchmark_simulation():
    """Comprehensive performance benchmark."""
    results = {}

    for size in [(100, 100), (250, 250), (500, 500), (1000, 1000)]:
        ny, nx = size
        sim = MonodomainSimulation(ny, nx)

        # Warmup
        for _ in range(10):
            sim.step(0.01)

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(100):
            sim.step(0.01)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        cells_per_sec = ny * nx * 100 / elapsed
        results[size] = {
            'time_per_step_ms': elapsed * 10,
            'cells_per_sec': cells_per_sec,
            'speedup_vs_v5': cells_per_sec / 8.4e6
        }

    return results
```

Target metrics:
- [ ] 500x500: > 50M cells/sec (6x speedup vs V5)
- [ ] 1000x1000: > 80M cells/sec
- [ ] Step time < 5ms for 500x500

---

## Backward Compatibility

### Loading V5 States

```python
def load_v5_state(v5_state: np.ndarray) -> torch.Tensor:
    """Convert V5 numpy state to V5.1 torch tensor."""
    return torch.tensor(v5_state, dtype=torch.float64, device='cuda')

def save_for_v5(state: torch.Tensor) -> np.ndarray:
    """Convert V5.1 state to numpy for V5 comparison."""
    return state.cpu().numpy()
```

### State Index Compatibility

V5.1 uses identical state indexing to V5, ensuring direct comparison.

---

## Stage 4: MeshBuilder System

### Overview

The MeshBuilder provides a high-level interface for configuring tissue simulations with
pre-tuned diffusion coefficients. It includes:

- **MeshBuilder**: 2D tissue configuration with fluent API
- **CableMesh**: 1D cable for CV tuning and validation

### Pre-tuned Diffusion Coefficients

Diffusion coefficients were empirically calibrated via 1D cable simulations at:
- dx = 0.02 cm (200 µm)
- dt = 0.02 ms
- ORd ionic model

| Target CV | D (cm²/ms) | D/D_min |
|-----------|------------|---------|
| 0.6 m/s (longitudinal) | 0.002161 | 5.87x |
| 0.3 m/s (transverse) | 0.000819 | 2.23x |

### Usage

```python
from tissue import MeshBuilder

# Default: 15x15 cm, dx=0.02, anisotropic
mesh = MeshBuilder.create_default()

# Isotropic (CV = 0.6 m/s all directions) - ideal for spiral waves
mesh = MeshBuilder.create_default(anisotropic=False)

# Or use shortcut
mesh = MeshBuilder.create_isotropic()

# Create simulation from mesh
sim = mesh.create_simulation(celltype=CellType.ENDO, device='cuda')
```

### Anisotropic vs Isotropic Mode

| Mode | D_L | D_T | Use Case |
|------|-----|-----|----------|
| Anisotropic (default) | 0.002161 | 0.000819 | Fiber-aligned propagation |
| Isotropic | 0.002161 | 0.002161 | Spiral waves, spherical wavefronts |

---

## File Structure

```
Engine_V5.1/
├── README.md
├── IMPLEMENTATION.md
├── ionic/
│   ├── __init__.py
│   ├── model.py          # ORdModel class
│   ├── gating.py         # Gating functions
│   ├── currents.py       # Ion current functions
│   ├── calcium.py        # Calcium handling
│   ├── camkii.py         # CaMKII signaling
│   └── parameters.py     # Model parameters
├── tissue/
│   ├── __init__.py
│   ├── mesh.py           # MeshBuilder & CableMesh
│   ├── diffusion.py      # DiffusionOperator
│   └── simulation.py     # MonodomainSimulation
├── utils/
│   ├── __init__.py
│   ├── device.py         # DeviceManager
│   └── validation.py     # Validation utilities
├── examples/
│   ├── single_cell.py
│   ├── spiral_wave_s1s2.py  # S1-S2 spiral induction
│   └── tissue_animation.py
└── tests/
    ├── test_mesh_builder.py
    ├── test_gating.py
    ├── test_currents.py
    ├── test_ionic.py
    ├── test_diffusion.py
    └── test_simulation.py
```

---

## Validation Summary

### Stage 1 Checkpoints

| Substage | Validation Criteria | Pass Condition |
|----------|---------------------|----------------|
| 1.1 | Tensor infrastructure | CUDA available, float64 confirmed |
| 1.2 | Gating functions | All < 1e-10 error vs V5 |
| 1.3 | Ion currents | All < 1e-6 relative error vs V5 |
| 1.4 | Calcium handling | Buffer factors match, SR dynamics match |
| 1.5 | CaMKII | CaMKa, fCaMKp match V5 |
| 1.6 | Rush-Larsen | Stable at dt=0.01ms, matches analytical |
| 1.7 | Single cell AP | Max error < 1mV, APD90 < 1% diff |
| 1.8 | Batch operations | Memory scales linearly, results match |

### Stage 2 Checkpoints

| Substage | Validation Criteria | Pass Condition |
|----------|---------------------|----------------|
| 2.1 | FVM discretization | Matches analytical for test cases |
| 2.2 | Isotropic diffusion | < 1e-10 error vs V5 |
| 2.3 | Anisotropic tensor | Correct at θ=0°, 45°, 90° |
| 2.4 | Stability | Stable at 90% limit, unstable at 110% |
| 2.5 | CV calibration | CV within 5% of target |
| 2.6 | Full operator | Matches V5 diffusion output |

### Stage 3 Checkpoints

| Substage | Validation Criteria | Pass Condition |
|----------|---------------------|----------------|
| 3.1 | Integration | Wave propagates, CV correct |
| 3.2 | torch.compile | Output matches, 2-5x speedup |
| 3.3 | Memory | < 1GB for 500x500, no leaks |
| 3.4 | Benchmark | > 50M cells/sec for 500x500 |

---

## References

1. O'Hara T, et al. (2011). "Simulation of the Undiseased Human Cardiac Ventricular Action Potential: Model Formulation and Experimental Validation." PLoS Comput Biol.

2. Rush S, Larsen H (1978). "A practical algorithm for solving dynamic membrane equations." IEEE Trans Biomed Eng.

3. Sundnes J, et al. (2006). "Computing the Electrical Activity in the Heart." Springer.
