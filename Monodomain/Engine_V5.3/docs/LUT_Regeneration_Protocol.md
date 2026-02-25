# LUT Regeneration Protocol

## Purpose

This document defines when and how to regenerate Lookup Tables (LUTs) for ionic models when parameters change. Proper LUT management ensures simulation accuracy while maintaining performance benefits.

---

## 1. Parameter Classification

### 1.1 Parameters That DO NOT Affect LUT

These parameters multiply currents or concentrations but don't change voltage-dependent gating kinetics:

| Category | Parameters | Reason |
|----------|------------|--------|
| Conductances | GNa, GNaL, Gto, GKr, GKs, GK1, GKb, GpCa | Multiply I_ion, not gating |
| Permeabilities | PCa, PNab, PCab | Multiply I_ion, not gating |
| Scaling factors | GNaL_scale, Gto_scale, GKr_scale, GKs_scale, etc. | Multiply I_ion, not gating |
| Extracellular ions | nao, cao | Don't affect gating steady-states |
| SR parameters | Jup_max, a_rel, bt, cajsr_half | Calcium handling, not V-gating |
| CaMKII | CaMKo, KmCaM, KmCaMK, aCaMK, bCaMK | Modulates phosphorylation fraction |
| Diffusion | tau_diff_Na, tau_diff_K, tau_diff_Ca, tau_tr | Concentration diffusion |
| Geometry | L, rad, Cm | Cell geometry |

**No LUT regeneration needed** when these change.

### 1.2 Parameters That AFFECT LUT

These parameters directly modify voltage-dependent gating functions:

| Parameter | Affects | Location in LUT |
|-----------|---------|-----------------|
| `tau_hs_scale` | INa hs time constant | `hs_tau` table |
| `tau_hsp_scale` | INa hsp time constant | `hsp_tau` table |
| `ko` (extracellular K+) | IK1 xk1_inf | `xk1_inf` table (if included) |
| Cell type (ENDO/EPI/M) | Ito delta_epi | `delta_epi` table |

**LUT regeneration REQUIRED** when these change.

### 1.3 Future-Proofing

Any new parameter that modifies:
- Steady-state functions (x_inf)
- Time constant functions (tau_x)
- Voltage-dependent rectification factors

Must be added to Section 1.2 and trigger LUT regeneration.

---

## 2. LUT Regeneration Triggers

### 2.1 Automatic Triggers

The LUT should be regenerated automatically when:

```python
# Trigger 1: Model initialization with custom parameters
model = ORdModel(celltype=CellType.EPI, use_lut=True,
                 params_override={'tau_hs_scale': 0.8})  # Triggers rebuild

# Trigger 2: Cell type change (if model supports runtime change)
model.set_celltype(CellType.M_CELL)  # Triggers rebuild for delta_epi

# Trigger 3: Explicit parameter modification of LUT-affecting params
model.params.tau_hs_scale = 1.2  # Should trigger rebuild
```

### 2.2 Manual Trigger

Provide explicit method for forced regeneration:

```python
# Force LUT rebuild (e.g., after batch parameter changes)
model.rebuild_lut()
```

---

## 3. Implementation: LUT Cache Key System

### 3.1 Cache Key Structure

Create a unique key based on LUT-affecting parameters:

```python
@dataclass
class LUTCacheKey:
    """Unique identifier for LUT configuration."""
    model_name: str           # 'ORd' or 'TTP06'
    celltype: CellType        # ENDO, EPI, M_CELL
    device: str               # 'cuda' or 'cpu'
    dtype: str                # 'float64' or 'float32'

    # LUT-affecting parameters (frozen for hashing)
    tau_hs_scale: float = 1.0
    tau_hsp_scale: float = 1.0
    ko: float = 5.4           # Only if xk1_inf in LUT

    def __hash__(self):
        return hash((self.model_name, self.celltype, self.device,
                     self.dtype, self.tau_hs_scale, self.tau_hsp_scale))
```

### 3.2 Global Cache with Key Lookup

```python
# Global LUT cache
_lut_cache: Dict[LUTCacheKey, LookupTable] = {}

def get_ord_lut(celltype: CellType,
                device: torch.device,
                params: ORdParameters) -> ORdLUT:
    """
    Get or create cached ORd LUT for given configuration.

    Automatically rebuilds if parameters differ from cached version.
    """
    key = LUTCacheKey(
        model_name='ORd',
        celltype=celltype,
        device=str(device),
        dtype='float64',
        tau_hs_scale=params.tau_hs_scale,
        tau_hsp_scale=params.tau_hsp_scale,
    )

    if key not in _lut_cache:
        # Build new LUT with current parameters
        _lut_cache[key] = ORdLUT(
            celltype=celltype,
            device=device,
            tau_hs_scale=params.tau_hs_scale,
            tau_hsp_scale=params.tau_hsp_scale,
        )

    return _lut_cache[key]

def clear_lut_cache():
    """Clear all cached LUTs (useful for memory management)."""
    _lut_cache.clear()

def invalidate_lut(model_name: str = None):
    """Invalidate LUTs for specific model or all models."""
    if model_name is None:
        _lut_cache.clear()
    else:
        keys_to_remove = [k for k in _lut_cache if k.model_name == model_name]
        for k in keys_to_remove:
            del _lut_cache[k]
```

---

## 4. ORdLUT Class with Parameter Support

### 4.1 Constructor with Scaling Parameters

```python
class ORdLUT(LookupTable):
    """
    Precomputed lookup tables for ORd model.

    Parameters
    ----------
    celltype : CellType
        Cell type (affects delta_epi table)
    device : torch.device
        Computation device
    tau_hs_scale : float
        Scaling factor for INa hs time constant (default 1.0)
    tau_hsp_scale : float
        Scaling factor for INa hsp time constant (default 1.0)
    """

    def __init__(self,
                 celltype: CellType = CellType.ENDO,
                 device: torch.device = None,
                 dtype: torch.dtype = torch.float64,
                 tau_hs_scale: float = 1.0,
                 tau_hsp_scale: float = 1.0):

        config = LUTConfig(
            V_min=-100.0,
            V_max=80.0,
            n_points=2001,
            device=device,
            dtype=dtype
        )
        super().__init__(config)

        # Store parameters for cache key generation
        self.celltype = celltype
        self.tau_hs_scale = tau_hs_scale
        self.tau_hsp_scale = tau_hsp_scale

        # Build all tables
        self._build_tables()

    def _build_tables(self):
        """Build all voltage-dependent lookup tables."""
        from ionic.ord.gating import (
            INa_m_inf, INa_m_tau, INa_h_inf, INa_hf_tau, INa_hs_tau,
            INa_j_tau, INa_hsp_inf, INa_jp_tau,
            # ... other imports
        )

        # INa gates
        self.add_function('m_inf', INa_m_inf)
        self.add_function('m_tau', INa_m_tau)
        self.add_function('h_inf', INa_h_inf)
        self.add_function('hf_tau', INa_hf_tau)

        # INa hs with scaling parameter
        self.add_function('hs_tau',
            lambda V: INa_hs_tau(V, scale=self.tau_hs_scale))

        self.add_function('j_tau', INa_j_tau)
        self.add_function('hsp_inf', INa_hsp_inf)

        # INa hsp with scaling parameter
        self.add_function('hsp_tau',
            lambda V: INa_hsp_tau(V, scale=self.tau_hsp_scale))

        self.add_function('jp_tau', INa_jp_tau)

        # ... build remaining tables

        # Cell-type specific: delta_epi (only meaningful for EPI)
        if self.celltype == CellType.EPI:
            from ionic.ord.gating import Ito_delta_epi
            self.add_function('delta_epi', Ito_delta_epi)
```

---

## 5. Model Integration

### 5.1 ORdModel with LUT Parameter Tracking

```python
class ORdModel(IonicModel):
    def __init__(self, celltype: CellType = CellType.ENDO,
                 device: str = 'cuda',
                 use_lut: bool = False,
                 params_override: Optional[Dict[str, float]] = None):

        super().__init__(device)
        self.celltype = celltype
        self.use_lut = use_lut

        # Get parameters (with overrides)
        self.params = get_celltype_parameters(celltype)
        if params_override:
            for name, value in params_override.items():
                setattr(self.params, name, value)

        # Initialize LUT if requested
        self._lut = None
        if use_lut:
            self._init_lut()

    def _init_lut(self):
        """Initialize or retrieve cached LUT."""
        self._lut = get_ord_lut(
            celltype=self.celltype,
            device=self.device,
            params=self.params
        )

    def rebuild_lut(self):
        """Force LUT rebuild (call after modifying LUT-affecting params)."""
        if self.use_lut:
            # Invalidate cache for this configuration
            invalidate_lut('ORd')
            self._init_lut()

    @property
    def lut(self) -> Optional[ORdLUT]:
        """Current LUT instance (None if LUT disabled)."""
        return self._lut
```

### 5.2 Parameter Modification with Auto-Rebuild

```python
class ORdModel(IonicModel):
    # ... existing code ...

    def set_parameter(self, name: str, value: float, rebuild_lut: bool = True):
        """
        Set a model parameter.

        Parameters
        ----------
        name : str
            Parameter name
        value : float
            New value
        rebuild_lut : bool
            If True, rebuild LUT if parameter affects gating kinetics
        """
        # Parameters that require LUT rebuild
        LUT_AFFECTING_PARAMS = {'tau_hs_scale', 'tau_hsp_scale'}

        if not hasattr(self.params, name):
            raise ValueError(f"Unknown parameter: {name}")

        setattr(self.params, name, value)

        if rebuild_lut and self.use_lut and name in LUT_AFFECTING_PARAMS:
            self.rebuild_lut()
```

---

## 6. Validation Protocol

After LUT regeneration, verify accuracy:

```python
def validate_lut_accuracy(model: ORdModel, tolerance: float = 0.001):
    """
    Validate LUT accuracy against direct computation.

    Parameters
    ----------
    model : ORdModel
        Model with LUT enabled
    tolerance : float
        Maximum allowed relative error

    Returns
    -------
    bool
        True if all tables pass validation
    """
    if model.lut is None:
        return True  # No LUT to validate

    # Test voltages spanning physiological range
    V_test = torch.linspace(-100, 80, 1000,
                            device=model.device, dtype=model.dtype)

    from ionic.ord.gating import INa_m_inf, INa_m_tau  # etc.

    test_cases = [
        ('m_inf', INa_m_inf),
        ('m_tau', INa_m_tau),
        # ... all functions
    ]

    all_passed = True
    for name, func in test_cases:
        lut_vals = model.lut.lookup(name, V_test)
        direct_vals = func(V_test)

        rel_error = (lut_vals - direct_vals).abs() / (direct_vals.abs() + 1e-10)
        max_error = rel_error.max().item()

        if max_error > tolerance:
            print(f"FAIL: {name} max error = {max_error:.6f}")
            all_passed = False

    return all_passed
```

---

## 7. Summary Flowchart

```
Parameter Change Request
         │
         ▼
┌─────────────────────────────┐
│ Is parameter in             │
│ LUT_AFFECTING_PARAMS?       │
└─────────────────────────────┘
         │
    ┌────┴────┐
    │         │
   YES        NO
    │         │
    ▼         ▼
┌─────────┐  ┌─────────────────┐
│ Rebuild │  │ Update param    │
│ LUT     │  │ No LUT change   │
└─────────┘  └─────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Generate new cache key      │
│ Build new LUT tables        │
│ Store in global cache       │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Validate LUT accuracy       │
│ (optional, debug mode)      │
└─────────────────────────────┘
```

---

## 8. Best Practices

1. **Batch parameter changes**: Modify all parameters first, then call `rebuild_lut()` once
2. **Cache management**: Call `clear_lut_cache()` when switching between many configurations
3. **Memory**: Each LUT uses ~1-2 MB; cache grows with unique configurations
4. **Validation**: Run `validate_lut_accuracy()` after custom parameter modifications
5. **Cell type**: Changing celltype always requires LUT rebuild (delta_epi differs)

---

## 9. Error Handling

```python
class LUTError(Exception):
    """Base exception for LUT-related errors."""
    pass

class LUTRegenerationRequired(LUTError):
    """Raised when LUT is stale and needs regeneration."""
    pass

class LUTValidationError(LUTError):
    """Raised when LUT fails accuracy validation."""
    pass
```
