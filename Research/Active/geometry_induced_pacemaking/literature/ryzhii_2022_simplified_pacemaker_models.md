# Ryzhii & Ryzhii 2022 — Simplified Pacemaker Cell Models (pAP, pCN)

**Citation**: Ryzhii E, Ryzhii M. "A two-variable model robust to pacemaker behaviour." *PLoS ONE* 17(9):e0257935, 2022.
**Code**: github.com/mryzhii/Simplified-pacemaker-cell-models (MATLAB + CellML)

## Key Contribution
Derived minimal 2-variable pacemaker models from the excitable Aliev-Panfilov (AP) and Corrado-Niederer (CN) models by converting a single parameter, enabling Hopf bifurcation from excitable to oscillatory behavior.

## Pacemaker Aliev-Panfilov (pAP)
- **Variables**: 2 (u, v) — identical form to excitable AP model
- **Pacemaker switch**: Set b_AP = -a (shifts nullcline intersection past Hopf bifurcation)
- **Frequency range**: 0.007-7.6 Hz (0.4-450 BPM)
- **Coupling to excitable cells**: d(u_p)/dt = ... + D∇²u_p at pacemaker-excitable boundary

## Pacemaker Corrado-Niederer (pCN)
- **Variables**: 2 (u, h)
- **Pacemaker switch**: b_CN parameter shift
- **Frequency range**: 0.14-14 Hz (8.4-840 BPM)
- **Advantage over pAP**: Wider synchronization area, lower sensitivity to coupling strength, more rectangular AP shape

## Tissue Simulation Results
- **2D SAN model**: 10 mm × 10 mm, 200×200 mesh, pacemaker center + excitable periphery
- **3D intestine tube**: Pacemaker ring driving excitable tube
- **1D strand**: Pacemaker cells coupled to excitable strand

### Coupling-Dependent Behavior:
| Coupling strength | Result |
|---|---|
| Weak | 2:5 or other sub-harmonic block |
| Intermediate | Successful 1:1 pace-and-drive |
| Strong | Complete quiescence (pacemaker suppressed) |

## Why These Models Matter for Us
1. **Trivially cheap**: 2 ODEs per cell. Can sweep geometry parameters (tip angle, node size, exit width, coupling D) in minutes on GPU.
2. **Proven pacemaker-excitable coupling**: Already demonstrated the exact phenomenon we want to study.
3. **Natural progression**: pAP/pCN for geometry exploration → PHAS13 for physiological validation.
4. **Implementation**: Could be added to V5.4 as a new IonicModel in a few hours.

## Relevance to Our Work
Ideal Tier 1 model for rapid prototyping. Implement pAP/pCN first, identify which geometries produce organized pacing, then repeat the key cases with PHAS13.
