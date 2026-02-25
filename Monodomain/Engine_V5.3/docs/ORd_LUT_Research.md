# ORd Lookup Table (LUT) Implementation Research

**Date**: January 3, 2025
**Status**: Research Complete
**Related Documents**:
- [LUT Regeneration Protocol](./LUT_Regeneration_Protocol.md)

---

## Executive Summary

Engine V5.3 currently has LUT acceleration for TTP06 model but **not for ORd model**.
This document summarizes research on best practices for implementing ORd LUT based on:
1. OpenCarp's EasyML LUT approach
2. Our existing TTP06LUT implementation
3. Academic literature on LUT optimization for cardiac models

### Quick Reference

| Aspect | Recommendation |
|--------|----------------|
| Voltage range | -100 to +80 mV |
| Resolution | 0.09 mV (2001 points) |
| Tables needed | ~34 voltage-dependent functions |
| Expected speedup | 2-4x |
| Interpolation | Linear |
| Singularities | None in ORd (clean exponential forms) |

---

## 1. OpenCarp LUT Implementation

### 1.1 EasyML Syntax
OpenCarp uses a domain-specific language (EasyML) with LUT markup:

```
V_m; .lookup(-100, 100, 0.05);
```

Parameters:
- **min**: -100 mV (lower voltage bound)
- **max**: 100 mV (upper voltage bound)
- **step**: 0.05 mV (discretization resolution)

This gives **4001 table entries** for voltage-dependent functions.

### 1.2 Key Design Principles from OpenCarp

1. **Minimize function invocations**: LUT replaces expensive `exp()`, `log()`, divisions
2. **Linear interpolation**: For voltages between grid points
3. **Handle singularities**: Some gating functions have discontinuities (e.g., tau_d at -5 mV)
4. **L'Hôpital's rule**: Used to resolve 0/0 singularities analytically

### 1.3 Singularity Example (from OpenCarp docs)
For ICaL d-gate time constant with singularity at V = -5 mV:
```c
tau_d = (V_m == -5.0) ?
    (-d_inf / 6.0 / 0.035) :
    ((1.0 * d_inf * (1.0 - exp(-(V_m + 5.0) / 6.0))) / (0.035 * (V_m + 5.0)));
```

---

## 2. Literature: LUT with Rush-Larsen (Sherwin et al. 2022)

Reference: [Frontiers in Physiology 2022](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.904648/full)

### 2.1 Optimized Rush-Larsen Form

Standard Rush-Larsen:
```
w(t+Δt) = w_inf + (w(t) - w_inf) * exp(-dt/tau)
```

LUT-optimized form (precompute a(V) and b(V)):
```
w(t+Δt) = a(V) * w(t) + b(V)

where:
  a(V) = exp(-dt/tau(V))
  b(V) = w_inf(V) * (1 - exp(-dt/tau(V)))
```

**Key insight**: For fixed dt, precompute both `a` and `b` terms, reducing runtime to 1 multiply + 1 add per gate.

### 2.2 LUT Configuration (TP06 model)

| Parameter | Value |
|-----------|-------|
| V_min | -100 mV |
| V_max | 50 mV |
| V_step | configurable |
| Tables | 17 expressions for V, 1 for Ca_ss |

### 2.3 Performance Results

| Implementation | Speedup vs Naive |
|----------------|------------------|
| LUT only | 3.5-4.7x |
| SIMD + LUT | Hardware-dependent |

---

## 3. Our Current TTP06LUT Implementation

Located in: `ionic/lut.py`

### 3.1 Configuration
```python
class LUTConfig:
    V_min: float = -100.0      # Minimum voltage (mV)
    V_max: float = 80.0        # Maximum voltage (mV)
    n_points: int = 2001       # Resolution: 0.09 mV step
```

### 3.2 Functions Precomputed (22 tables)

**INa gates (6)**: m_inf, m_tau, h_inf, h_tau, j_inf, j_tau

**Ito gates (6)**: r_inf, r_tau, s_inf_endo, s_inf_epi, s_tau_endo, s_tau_epi

**ICaL gates (6)**: d_inf, d_tau, f_inf, f_tau, f2_inf, f2_tau

**IKr gates (4)**: Xr1_inf, Xr1_tau, Xr2_inf, Xr2_tau

**IKs gates (2)**: Xs_inf, Xs_tau

### 3.3 Interpolation Method
```python
def lookup(self, name: str, V: torch.Tensor) -> torch.Tensor:
    idx_float = (V - self.V_min) * self.inv_dV
    idx_float = torch.clamp(idx_float, 0, self.n_points - 1.0001)

    idx = idx_float.long()
    frac = idx_float - idx.float()

    # Linear interpolation
    return table[idx] * (1.0 - frac) + table[idx + 1] * frac
```

---

## 4. ORd Model Gating Analysis

### 4.1 ORd State Variables (41 total)

**Voltage-dependent gates that benefit from LUT (27 gates)**:

| Current | Gates | Functions Needed |
|---------|-------|-----------------|
| INa | m, hf, hs, j, hsp, jp | 8 (m_inf, m_tau, h_inf, hf_tau, hs_tau, j_tau, hsp_inf, jp_tau) |
| INaL | mL, hL, hLp | 4 (mL_inf, hL_inf, hL_tau, hLp_inf) |
| Ito | a, iF, iS, ap, iFp, iSp | 5 (a_inf, a_tau, i_inf, iF_tau, iS_tau, ap_inf, delta_epi) |
| ICaL | d, ff, fs, fcaf, fcas, jca, ffp, fcafp | 8 (d_inf, d_tau, f_inf, ff_tau, fs_tau, fcaf_tau, fcas_tau, ffp_tau, fcafp_tau) |
| IKr | xrf, xrs | 4 (xr_inf, xrf_tau, xrs_tau, Axrf, RKr) |
| IKs | xs1, xs2 | 3 (xs1_inf, xs1_tau, xs2_tau) |
| IK1 | xk1 | 2 (xk1_tau, rk1) - Note: xk1_inf depends on ko |

**Total: ~34 voltage-dependent functions to precompute**

### 4.2 Special Cases in ORd

1. **IK1 xk1_inf**: Depends on both V and ko (extracellular K+)
   - Solution: Either exclude from LUT or create ko-parameterized tables

2. **Phosphorylated variants**: Many share base calculations
   - hsp_tau = 3.0 * hs_tau (can derive from hs_tau table)
   - jp_tau = 1.46 * j_tau (can derive from j_tau table)

3. **CaMKII-dependent terms**: dti_develop, dti_recover for Ito phosphorylated gates
   - These are voltage-dependent and should be in LUT

4. **Ito delta_epi**: Only used for EPI cells
   - Include in LUT, apply conditionally

---

## 5. Implementation Plan

### 5.1 Create ORdLUT Class

```python
class ORdLUT(LookupTable):
    """
    Precomputed lookup tables for ORd model gating functions.

    Supports ENDO, EPI, and M_CELL types with cell-type specific
    tables where needed (e.g., delta_epi for Ito).
    """

    def __init__(self, device=None, dtype=torch.float64):
        config = LUTConfig(
            V_min=-100.0,
            V_max=80.0,
            n_points=2001,  # 0.09 mV resolution
            device=device,
            dtype=dtype
        )
        super().__init__(config)
        self._build_tables()
```

### 5.2 Tables to Precompute

**Phase 1: Core gating (steady-state and time constants)**
- INa: m_inf, m_tau, h_inf, hf_tau, hs_tau, j_tau
- INa phosphorylated: hsp_inf, jp_tau (= 1.46 * j_tau)
- INaL: mL_inf, mL_tau, hL_inf, hL_tau, hLp_inf, hLp_tau
- Ito: a_inf, a_tau, i_inf, iF_tau, iS_tau, ap_inf, delta_epi
- ICaL: d_inf, d_tau, f_inf, ff_tau, fs_tau, fcaf_tau, fcas_tau
- ICaL phosphorylated: ffp_tau, fcafp_tau
- IKr: xr_inf, xrf_tau, xrs_tau, Axrf, RKr
- IKs: xs1_inf, xs1_tau, xs2_tau
- IK1: xk1_tau (xk1_inf excluded - depends on ko)

**Phase 2: CaMKII modulation terms**
- dti_develop, dti_recover (for phosphorylated Ito)

**Phase 3: Rush-Larsen optimized (optional, for fixed dt)**
- Precompute a(V) = exp(-dt/tau) and b(V) = x_inf * (1 - a)
- Requires knowing dt at table build time

### 5.3 Modify ORdModel

```python
class ORdModel(IonicModel):
    def __init__(self, celltype=CellType.ENDO, device='cuda',
                 use_lut: bool = False):  # Add LUT option
        ...
        if use_lut:
            self.lut = get_ord_lut(device)
        else:
            self.lut = None

    def _update_gates(self, states, dt):
        if self.lut is not None:
            return self._update_gates_lut(states, dt)
        else:
            return self._update_gates_direct(states, dt)
```

### 5.4 Test Coverage

1. **Interpolation accuracy**: LUT vs direct < 0.1% relative error
2. **Single cell AP match**: LUT vs direct V traces < 0.5 mV difference
3. **Tissue CV match**: LUT vs direct conduction velocity < 1% difference
4. **Speedup measurement**: Expect 2-4x improvement
5. **Edge cases**: V at bounds, beyond bounds

---

## 6. References

1. **OpenCarp Documentation**: [EasyML LUT syntax](https://opencarp.org/documentation/examples/01_ep_single_cell/05_easyml)
2. **Sherwin et al. (2022)**: [Resource-Efficient LUT + Rush-Larsen](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.904648/full)
3. **O'Hara et al. (2011)**: [ORd model paper](https://pubmed.ncbi.nlm.nih.gov/21637795/)
4. **CellML ORd**: [Physiome Model Repository](https://models.cellml.org/e/71)
5. **OpenCarp LIMPET**: [GitLab repository](https://git.opencarp.org/openCARP/openCARP/-/tree/master/physics/limpet/models)

---

## 7. Appendix A: Complete ORd Gating Functions for LUT

From `ionic/ord/gating.py` - all functions use clean exponential forms with **no singularities**:

### INa (Fast Sodium Current)
```python
# Steady-states
INa_m_inf(V) = 1 / (1 + exp(-(V + 39.57) / 9.871))
INa_h_inf(V) = 1 / (1 + exp((V + 82.90) / 6.086))      # shared by hf, hs, j
INa_hsp_inf(V) = 1 / (1 + exp((V + 89.1) / 6.086))     # phosphorylated

# Time constants
INa_m_tau(V) = 1 / (6.765·exp((V+11.64)/34.77) + 8.552·exp(-(V+77.42)/5.955))
INa_hf_tau(V) = 1 / (1.432e-5·exp(-(V+1.196)/6.285) + 6.149·exp((V+0.5096)/20.27))
INa_hs_tau(V, scale) = scale / (0.009794·exp(-(V+17.95)/28.05) + 0.3343·exp((V+5.730)/56.66))
INa_j_tau(V) = 2.038 + 1 / (0.02136·exp(-(V+100.6)/8.281) + 0.3052·exp((V+0.9941)/38.45))
INa_hsp_tau(V, scale) = 3.0 · INa_hs_tau(V, scale)
INa_jp_tau(V) = 1.46 · INa_j_tau(V)
```

### INaL (Late Sodium Current)
```python
INaL_mL_inf(V) = 1 / (1 + exp(-(V + 42.85) / 5.264))
INaL_mL_tau(V) = INa_m_tau(V)  # Same as INa
INaL_hL_inf(V) = 1 / (1 + exp((V + 87.61) / 7.488))
INaL_hL_tau(V) = 200.0  # Constant
INaL_hLp_inf(V) = 1 / (1 + exp((V + 93.81) / 7.488))
INaL_hLp_tau(V) = 600.0  # 3 × hL_tau
```

### Ito (Transient Outward K+)
```python
Ito_a_inf(V) = 1 / (1 + exp(-(V - 14.34) / 14.82))
Ito_a_tau(V) = 1.0515 / (1/(1.2089·(1+exp(-(V-18.4099)/29.3814))) + 3.5/(1+exp((V+100)/29.3814)))
Ito_i_inf(V) = 1 / (1 + exp((V + 43.94) / 5.711))  # shared by iF, iS
Ito_iF_tau(V) = 4.562 + 1 / (0.3933·exp(-(V+100)/100) + 0.08004·exp((V+50)/16.59))
Ito_iS_tau(V) = 23.62 + 1 / (0.001416·exp(-(V+96.52)/59.05) + 1.780e-8·exp((V+114.1)/8.079))
Ito_ap_inf(V) = 1 / (1 + exp(-(V - 24.34) / 14.82))  # Phosphorylated
Ito_delta_epi(V) = 1 - 0.95 / (1 + exp((V + 70) / 5))  # EPI cells only
```

### ICaL (L-type Calcium)
```python
ICaL_d_inf(V) = 1 / (1 + exp(-(V + 3.94) / 4.23))
ICaL_d_tau(V) = 0.6 + 1 / (exp(-0.05·(V+6)) + exp(0.09·(V+14)))
ICaL_f_inf(V) = 1 / (1 + exp((V + 19.58) / 3.696))  # shared by ff, fs
ICaL_ff_tau(V) = 7 + 1 / (0.0045·exp(-(V+20)/10) + 0.0045·exp((V+20)/10))
ICaL_fs_tau(V) = 1000 + 1 / (0.000035·exp(-(V+5)/4) + 0.000035·exp((V+5)/6))
ICaL_fcaf_tau(V) = 7 + 1 / (0.04·exp(-(V-4)/7) + 0.04·exp((V-4)/7))
ICaL_fcas_tau(V) = 100 + 1 / (0.00012·exp(-V/3) + 0.00012·exp(V/7))
ICaL_jca_tau(V) = 75.0  # Constant
ICaL_ffp_tau(V) = 2.5 · ICaL_ff_tau(V)
ICaL_fcafp_tau(V) = 2.5 · ICaL_fcaf_tau(V)
```

### IKr (Rapid Delayed Rectifier)
```python
IKr_xr_inf(V) = 1 / (1 + exp(-(V + 8.337) / 6.789))  # shared by xrf, xrs
IKr_xrf_tau(V) = 12.98 + 1 / (0.3652·exp((V-31.66)/3.869) + 4.123e-5·exp(-(V-47.78)/20.38))
IKr_xrs_tau(V) = 1.865 + 1 / (0.06629·exp((V-34.70)/7.355) + 1.128e-5·exp(-(V-29.74)/25.94))
IKr_Axrf(V) = 1 / (1 + exp((V + 54.81) / 38.21))  # Fast/slow fraction
IKr_RKr(V) = 1/(1+exp((V+55)/75)) · 1/(1+exp((V-10)/30))  # Rectification
```

### IKs (Slow Delayed Rectifier)
```python
IKs_xs1_inf(V) = 1 / (1 + exp(-(V + 11.60) / 8.932))  # shared by xs1, xs2
IKs_xs1_tau(V) = 817.3 + 1 / (2.326e-4·exp((V+48.28)/17.80) + 0.001292·exp(-(V+210)/230))
IKs_xs2_tau(V) = 1 / (0.01·exp((V-50)/20) + 0.0193·exp(-(V+66.54)/31))
```

### IK1 (Inward Rectifier)
```python
IK1_xk1_inf(V, ko) = 1 / (1 + exp(-(V + 2.5538·ko + 144.59) / (1.5692·ko + 3.8115)))  # ko-dependent!
IK1_xk1_tau(V) = 122.2 / (exp(-(V+127.2)/20.36) + exp((V+236.8)/69.33))
IK1_rk1(V, ko) = 1 / (1 + exp((V + 105.8 - 2.6·ko) / 9.493))  # ko-dependent!
```

### CaMKII Modulation Terms
```python
dti_develop(V) = 1.354 + 1e-4 / (exp((V-167.4)/15.89) + exp(-(V-12.23)/0.2154))
dti_recover(V) = 1 - 0.5 / (1 + exp((V + 70) / 20))
```

---

## 8. Appendix B: LUT Table Summary

| Table Name | Function | Include in LUT | Notes |
|------------|----------|----------------|-------|
| m_inf | INa_m_inf | Yes | |
| m_tau | INa_m_tau | Yes | |
| h_inf | INa_h_inf | Yes | Shared by hf, hs, j |
| hf_tau | INa_hf_tau | Yes | |
| hs_tau | INa_hs_tau | Yes | Uses tau_hs_scale param |
| j_tau | INa_j_tau | Yes | |
| hsp_inf | INa_hsp_inf | Yes | |
| hsp_tau | INa_hsp_tau | Yes | Uses tau_hsp_scale param |
| jp_tau | INa_jp_tau | Yes | = 1.46 × j_tau |
| mL_inf | INaL_mL_inf | Yes | |
| mL_tau | INaL_mL_tau | Yes | Same as m_tau |
| hL_inf | INaL_hL_inf | Yes | |
| hLp_inf | INaL_hLp_inf | Yes | |
| a_inf | Ito_a_inf | Yes | |
| a_tau | Ito_a_tau | Yes | |
| i_inf | Ito_i_inf | Yes | Shared by iF, iS |
| iF_tau | Ito_iF_tau | Yes | |
| iS_tau | Ito_iS_tau | Yes | |
| ap_inf | Ito_ap_inf | Yes | |
| delta_epi | Ito_delta_epi | Yes | EPI cells only |
| dti_develop | CaMKII | Yes | |
| dti_recover | CaMKII | Yes | |
| d_inf | ICaL_d_inf | Yes | |
| d_tau | ICaL_d_tau | Yes | |
| f_inf | ICaL_f_inf | Yes | Shared by ff, fs, fcaf, fcas |
| ff_tau | ICaL_ff_tau | Yes | |
| fs_tau | ICaL_fs_tau | Yes | |
| fcaf_tau | ICaL_fcaf_tau | Yes | |
| fcas_tau | ICaL_fcas_tau | Yes | |
| ffp_tau | ICaL_ffp_tau | Yes | = 2.5 × ff_tau |
| fcafp_tau | ICaL_fcafp_tau | Yes | = 2.5 × fcaf_tau |
| xr_inf | IKr_xr_inf | Yes | Shared by xrf, xrs |
| xrf_tau | IKr_xrf_tau | Yes | |
| xrs_tau | IKr_xrs_tau | Yes | |
| Axrf | IKr_Axrf | Yes | |
| RKr | IKr_RKr | Yes | |
| xs1_inf | IKs_xs1_inf | Yes | Shared by xs1, xs2 |
| xs1_tau | IKs_xs1_tau | Yes | |
| xs2_tau | IKs_xs2_tau | Yes | |
| xk1_tau | IK1_xk1_tau | Yes | |
| xk1_inf | IK1_xk1_inf | **No** | ko-dependent |
| rk1 | IK1_rk1 | **No** | ko-dependent |

**Total LUT tables: 38** (excluding ko-dependent functions)
