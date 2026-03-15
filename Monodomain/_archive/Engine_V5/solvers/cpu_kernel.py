"""
Parallel CPU Kernel for ORd Model Tissue Simulation

This module provides a fully JIT-compiled, parallelized ionic step kernel
that can process thousands of cells in parallel using Numba's prange.

The kernel combines all gating, current, and calcium calculations into
a single @njit function to eliminate Python overhead.
"""

import numpy as np
from numba import njit, prange
import math


# =============================================================================
# Physical Constants and State Indices
# =============================================================================

# State indices (must match StateIndex in parameters.py)
V_IDX = 0
NAI_IDX = 1
KI_IDX = 2
CAI_IDX = 3
CANSR_IDX = 4
NASS_IDX = 5
KSS_IDX = 6
CASS_IDX = 7
CAJSR_IDX = 8
M_IDX = 9
HF_IDX = 10
HS_IDX = 11
J_IDX = 12
HSP_IDX = 13
JP_IDX = 14
ML_IDX = 15
HL_IDX = 16
HLP_IDX = 17
A_IDX = 18
IF_IDX = 19
IS_IDX = 20
AP_IDX = 21
IFP_IDX = 22
ISP_IDX = 23
D_IDX = 24
FF_IDX = 25
FS_IDX = 26
FCAF_IDX = 27
FCAS_IDX = 28
JCA_IDX = 29
NCA_IDX = 30
FFP_IDX = 31
FCAFP_IDX = 32
XRF_IDX = 33
XRS_IDX = 34
XS1_IDX = 35
XS2_IDX = 36
XK1_IDX = 37
JRELNP_IDX = 38
JRELP_IDX = 39
CAMKT_IDX = 40
N_STATES = 41


# =============================================================================
# Helper Functions (inlined into kernel)
# =============================================================================

@njit(cache=True, fastmath=True)
def safe_exp(x):
    """Clipped exponential to prevent overflow."""
    return math.exp(max(-80.0, min(80.0, x)))


@njit(cache=True, fastmath=True)
def rush_larsen(gate, gate_inf, tau, dt):
    """Rush-Larsen exponential integration for gating variables."""
    return gate_inf - (gate_inf - gate) * safe_exp(-dt / tau)


# =============================================================================
# Single Cell Ionic Step Kernel
# =============================================================================

@njit(cache=True, fastmath=True)
def ionic_step_single_cell(
    y, dt, Istim,
    # Cell type (0=endo, 1=epi, 2=M)
    celltype,
    # Physical constants
    F, R, T, RTF,
    # Extracellular concentrations
    nao, ko, cao,
    # Conductances (scaled by celltype)
    GNa, GNaL, Gto, PCa, GKr, GKs, GK1, Gncx, Pnak, GKb, PNab, PCab, GpCa,
    # Other parameters
    thL, Acap, vmyo, vnsr, vjsr, vss,
    cmdnmax, kmcmdn, trpnmax, kmtrpn,
    BSRmax, KmBSR, BSLmax, KmBSL, csqnmax, kmcsqn,
    bt, CaMKo, KmCaM, KmCaMK, aCaMK, bCaMK
):
    """
    Advance ORd model by one time step for a single cell.

    All calculations are inlined for maximum JIT optimization.

    Returns: Updated state vector (41 elements)
    """
    y_new = np.empty(N_STATES)

    # Extract state variables
    V = y[V_IDX]
    nai = y[NAI_IDX]
    ki = y[KI_IDX]
    cai = y[CAI_IDX]
    cansr = y[CANSR_IDX]
    nass = y[NASS_IDX]
    kss = y[KSS_IDX]
    cass = y[CASS_IDX]
    cajsr = y[CAJSR_IDX]
    CaMKt = y[CAMKT_IDX]

    # =========================================================================
    # CaMKII Signaling
    # =========================================================================
    CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
    CaMKa = CaMKb + CaMKt
    fCaMKp = 1.0 / (1.0 + KmCaMK / CaMKa)

    dCaMKt = aCaMK * CaMKb * (CaMKb + CaMKt) - bCaMK * CaMKt
    y_new[CAMKT_IDX] = CaMKt + dt * dCaMKt

    # =========================================================================
    # Reversal Potentials
    # =========================================================================
    ENa = RTF * math.log(nao / nai)
    EK = RTF * math.log(ko / ki)
    EKs = RTF * math.log((ko + 0.01833 * nao) / (ki + 0.01833 * nai))

    # =========================================================================
    # INa Gating
    # =========================================================================
    # m gate
    m_inf = 1.0 / (1.0 + safe_exp(-(V + 39.57) / 9.871))
    m_tau = 1.0 / (6.765 * safe_exp((V + 11.64) / 34.77) +
                   8.552 * safe_exp(-(V + 77.42) / 5.955))
    y_new[M_IDX] = rush_larsen(y[M_IDX], m_inf, m_tau, dt)

    # h gates
    h_inf = 1.0 / (1.0 + safe_exp((V + 82.90) / 6.086))
    hf_tau = 1.0 / (1.432e-5 * safe_exp(-(V + 1.196) / 6.285) +
                    6.149 * safe_exp((V + 0.5096) / 20.27))
    hs_tau = 1.0 / (0.009794 * safe_exp(-(V + 17.95) / 28.05) +
                    0.3343 * safe_exp((V + 5.730) / 56.66))
    j_tau = 2.038 + 1.0 / (0.02136 * safe_exp(-(V + 100.6) / 8.281) +
                           0.3052 * safe_exp((V + 0.9941) / 38.45))

    y_new[HF_IDX] = rush_larsen(y[HF_IDX], h_inf, hf_tau, dt)
    y_new[HS_IDX] = rush_larsen(y[HS_IDX], h_inf, hs_tau, dt)
    y_new[J_IDX] = rush_larsen(y[J_IDX], h_inf, j_tau, dt)

    # Phosphorylated h gates
    hsp_inf = 1.0 / (1.0 + safe_exp((V + 89.1) / 6.086))
    y_new[HSP_IDX] = rush_larsen(y[HSP_IDX], hsp_inf, 3.0 * hs_tau, dt)
    y_new[JP_IDX] = rush_larsen(y[JP_IDX], h_inf, 1.46 * j_tau, dt)

    # =========================================================================
    # INaL Gating
    # =========================================================================
    mL_inf = 1.0 / (1.0 + safe_exp(-(V + 42.85) / 5.264))
    hL_inf = 1.0 / (1.0 + safe_exp((V + 87.61) / 7.488))
    hLp_inf = 1.0 / (1.0 + safe_exp((V + 93.81) / 7.488))

    y_new[ML_IDX] = rush_larsen(y[ML_IDX], mL_inf, m_tau, dt)
    y_new[HL_IDX] = rush_larsen(y[HL_IDX], hL_inf, thL, dt)
    y_new[HLP_IDX] = rush_larsen(y[HLP_IDX], hLp_inf, 3.0 * thL, dt)

    # =========================================================================
    # Ito Gating
    # =========================================================================
    a_inf = 1.0 / (1.0 + safe_exp(-(V - 14.34) / 14.82))
    a_tau = 1.0515 / (1.0 / (1.2089 * (1.0 + safe_exp(-(V - 18.4099) / 29.3814))) +
                      3.5 / (1.0 + safe_exp((V + 100.0) / 29.3814)))

    i_inf = 1.0 / (1.0 + safe_exp((V + 43.94) / 5.711))

    # Cell-type dependent time constants
    if celltype == 1:  # Epi
        delta_epi = 1.0 - 0.95 / (1.0 + safe_exp((V + 70.0) / 5.0))
    else:
        delta_epi = 1.0

    iF_tau = 4.562 + 1.0 / (0.3933 * safe_exp(-(V + 100.0) / 100.0) +
                            0.08004 * safe_exp((V + 50.0) / 16.59))
    iS_tau = 23.62 + 1.0 / (0.001416 * safe_exp(-(V + 96.52) / 59.05) +
                            1.780e-8 * safe_exp((V + 114.1) / 8.079))
    iF_tau *= delta_epi
    iS_tau *= delta_epi

    y_new[A_IDX] = rush_larsen(y[A_IDX], a_inf, a_tau, dt)
    y_new[IF_IDX] = rush_larsen(y[IF_IDX], i_inf, iF_tau, dt)
    y_new[IS_IDX] = rush_larsen(y[IS_IDX], i_inf, iS_tau, dt)

    ap_inf = 1.0 / (1.0 + safe_exp(-(V - 24.34) / 14.82))
    y_new[AP_IDX] = rush_larsen(y[AP_IDX], ap_inf, a_tau, dt)

    # Phosphorylated inactivation
    dti_develop = 1.354 + 1.0e-4 / (safe_exp((V - 167.4) / 15.89) +
                                     safe_exp(-(V - 12.23) / 0.2154))
    dti_recover = 1.0 - 0.5 / (1.0 + safe_exp((V + 70.0) / 20.0))
    tiFp = dti_develop * dti_recover * iF_tau
    tiSp = dti_develop * dti_recover * iS_tau

    y_new[IFP_IDX] = rush_larsen(y[IFP_IDX], i_inf, tiFp, dt)
    y_new[ISP_IDX] = rush_larsen(y[ISP_IDX], i_inf, tiSp, dt)

    # =========================================================================
    # ICaL Gating
    # =========================================================================
    d_inf = 1.0 / (1.0 + safe_exp(-(V + 3.940) / 4.230))
    d_tau = 0.6 + 1.0 / (safe_exp(-0.05 * (V + 6.0)) + safe_exp(0.09 * (V + 14.0)))

    f_inf = 1.0 / (1.0 + safe_exp((V + 19.58) / 3.696))
    ff_tau = 7.0 + 1.0 / (0.0045 * safe_exp(-(V + 20.0) / 10.0) +
                          0.0045 * safe_exp((V + 20.0) / 10.0))
    fs_tau = 1000.0 + 1.0 / (0.000035 * safe_exp(-(V + 5.0) / 4.0) +
                              0.000035 * safe_exp((V + 5.0) / 6.0))

    fcaf_tau = 7.0 + 1.0 / (0.04 * safe_exp(-(V - 4.0) / 7.0) +
                            0.04 * safe_exp((V - 4.0) / 7.0))
    fcas_tau = 100.0 + 1.0 / (0.00012 * safe_exp(-V / 3.0) +
                               0.00012 * safe_exp(V / 7.0))

    y_new[D_IDX] = rush_larsen(y[D_IDX], d_inf, d_tau, dt)
    y_new[FF_IDX] = rush_larsen(y[FF_IDX], f_inf, ff_tau, dt)
    y_new[FS_IDX] = rush_larsen(y[FS_IDX], f_inf, fs_tau, dt)
    y_new[FCAF_IDX] = rush_larsen(y[FCAF_IDX], f_inf, fcaf_tau, dt)
    y_new[FCAS_IDX] = rush_larsen(y[FCAS_IDX], f_inf, fcas_tau, dt)
    y_new[JCA_IDX] = rush_larsen(y[JCA_IDX], f_inf, 75.0, dt)
    y_new[FFP_IDX] = rush_larsen(y[FFP_IDX], f_inf, 2.5 * ff_tau, dt)
    y_new[FCAFP_IDX] = rush_larsen(y[FCAFP_IDX], f_inf, 2.5 * fcaf_tau, dt)

    # nca (Ca/calmodulin binding)
    Kmn = 0.002
    k2n = 1000.0
    km2n = y[JCA_IDX] * 1.0
    anca = 1.0 / (k2n / km2n + (1.0 + Kmn / cass) ** 4.0)
    dnca = anca * k2n - y[NCA_IDX] * km2n
    y_new[NCA_IDX] = y[NCA_IDX] + dt * dnca

    # =========================================================================
    # IKr Gating
    # =========================================================================
    xr_inf = 1.0 / (1.0 + safe_exp(-(V + 8.337) / 6.789))
    xrf_tau = 12.98 + 1.0 / (0.3652 * safe_exp((V - 31.66) / 3.869) +
                             4.123e-5 * safe_exp(-(V - 47.78) / 20.38))
    xrs_tau = 1.865 + 1.0 / (0.06629 * safe_exp((V - 34.70) / 7.355) +
                             1.128e-5 * safe_exp(-(V - 29.74) / 25.94))

    y_new[XRF_IDX] = rush_larsen(y[XRF_IDX], xr_inf, xrf_tau, dt)
    y_new[XRS_IDX] = rush_larsen(y[XRS_IDX], xr_inf, xrs_tau, dt)

    # =========================================================================
    # IKs Gating
    # =========================================================================
    xs1_inf = 1.0 / (1.0 + safe_exp(-(V + 11.60) / 8.932))
    xs1_tau = 817.3 + 1.0 / (2.326e-4 * safe_exp((V + 48.28) / 17.80) +
                              0.001292 * safe_exp(-(V + 210.0) / 230.0))
    xs2_tau = 1.0 / (0.01 * safe_exp((V - 50.0) / 20.0) +
                     0.0193 * safe_exp(-(V + 66.54) / 31.0))

    y_new[XS1_IDX] = rush_larsen(y[XS1_IDX], xs1_inf, xs1_tau, dt)
    y_new[XS2_IDX] = rush_larsen(y[XS2_IDX], xs1_inf, xs2_tau, dt)

    # =========================================================================
    # IK1 Gating
    # =========================================================================
    xk1_inf = 1.0 / (1.0 + safe_exp(-(V + 2.5538 * ko + 144.59) / (1.5692 * ko + 3.8115)))
    xk1_tau = 122.2 / (safe_exp(-(V + 127.2) / 20.36) + safe_exp((V + 236.8) / 69.33))

    y_new[XK1_IDX] = rush_larsen(y[XK1_IDX], xk1_inf, xk1_tau, dt)

    # =========================================================================
    # Compute Ionic Currents
    # =========================================================================

    # INa
    Ahf = 0.99
    Ahs = 1.0 - Ahf
    h = Ahf * y_new[HF_IDX] + Ahs * y_new[HS_IDX]
    hp = Ahf * y_new[HF_IDX] + Ahs * y_new[HSP_IDX]
    fINap = 1.0 / (1.0 + KmCaMK / CaMKa)
    INa = GNa * (V - ENa) * y_new[M_IDX]**3 * ((1.0 - fINap) * h * y_new[J_IDX] +
                                                 fINap * hp * y_new[JP_IDX])

    # INaL
    fINaLp = fINap
    INaL = GNaL * (V - ENa) * y_new[ML_IDX] * ((1.0 - fINaLp) * y_new[HL_IDX] +
                                                 fINaLp * y_new[HLP_IDX])

    # Ito
    AiF = 1.0 / (1.0 + safe_exp((V - 213.6) / 151.2))
    AiS = 1.0 - AiF
    i_gate = AiF * y_new[IF_IDX] + AiS * y_new[IS_IDX]
    ip = AiF * y_new[IFP_IDX] + AiS * y_new[ISP_IDX]
    fItop = fINap
    Ito = Gto * (V - EK) * ((1.0 - fItop) * y_new[A_IDX] * i_gate +
                            fItop * y_new[AP_IDX] * ip)

    # ICaL (GHK formulation)
    vffrt = V * F * F / (R * T)
    vfrt = V * F / (R * T)

    # Avoid division by zero near V=0
    if abs(V) < 1e-6:
        PhiCaL = 4.0 * F * (cass - 0.341 * cao)
        PhiCaNa = F * (0.75 * nass - 0.75 * nao)
        PhiCaK = F * (0.75 * kss - 0.75 * ko)
    else:
        PhiCaL = 4.0 * vffrt * (cass * safe_exp(2.0 * vfrt) - 0.341 * cao) / \
                 (safe_exp(2.0 * vfrt) - 1.0)
        PhiCaNa = vffrt * (0.75 * nass * safe_exp(vfrt) - 0.75 * nao) / \
                  (safe_exp(vfrt) - 1.0)
        PhiCaK = vffrt * (0.75 * kss * safe_exp(vfrt) - 0.75 * ko) / \
                 (safe_exp(vfrt) - 1.0)

    # ICaL gating
    Afcaf = 0.3 + 0.6 / (1.0 + safe_exp((V - 10.0) / 10.0))
    Afcas = 1.0 - Afcaf
    fca = Afcaf * y_new[FCAF_IDX] + Afcas * y_new[FCAS_IDX]
    fcap = Afcaf * y_new[FCAFP_IDX] + Afcas * y_new[FCAS_IDX]

    Aff = 0.6
    Afs = 1.0 - Aff
    f_gate = Aff * y_new[FF_IDX] + Afs * y_new[FS_IDX]
    fp = Aff * y_new[FFP_IDX] + Afs * y_new[FS_IDX]

    fICaLp = fINap
    d_gate = y_new[D_IDX]

    ICaL = PCa * PhiCaL * d_gate * ((1.0 - fICaLp) * (f_gate * (1.0 - y_new[NCA_IDX]) +
           y_new[JCA_IDX] * fca * y_new[NCA_IDX]) + fICaLp * (fp * (1.0 - y_new[NCA_IDX]) +
           y_new[JCA_IDX] * fcap * y_new[NCA_IDX]))

    ICaNa = 0.00125 * PCa * PhiCaNa * d_gate * ((1.0 - fICaLp) * (f_gate * (1.0 - y_new[NCA_IDX]) +
            y_new[JCA_IDX] * fca * y_new[NCA_IDX]) + fICaLp * (fp * (1.0 - y_new[NCA_IDX]) +
            y_new[JCA_IDX] * fcap * y_new[NCA_IDX]))

    ICaK = 0.0003574 * PCa * PhiCaK * d_gate * ((1.0 - fICaLp) * (f_gate * (1.0 - y_new[NCA_IDX]) +
           y_new[JCA_IDX] * fca * y_new[NCA_IDX]) + fICaLp * (fp * (1.0 - y_new[NCA_IDX]) +
           y_new[JCA_IDX] * fcap * y_new[NCA_IDX]))

    # IKr
    Axrf = 1.0 / (1.0 + safe_exp((V + 54.81) / 38.21))
    Axrs = 1.0 - Axrf
    xr = Axrf * y_new[XRF_IDX] + Axrs * y_new[XRS_IDX]
    rkr = 1.0 / (1.0 + safe_exp((V + 55.0) / 75.0)) * \
          1.0 / (1.0 + safe_exp((V - 10.0) / 30.0))
    IKr = GKr * math.sqrt(ko / 5.4) * xr * rkr * (V - EK)

    # IKs
    KsCa = 1.0 + 0.6 / (1.0 + (3.8e-5 / cai) ** 1.4)
    IKs = GKs * KsCa * y_new[XS1_IDX] * y_new[XS2_IDX] * (V - EKs)

    # IK1
    rk1 = 1.0 / (1.0 + safe_exp((V + 105.8 - 2.6 * ko) / 9.493))
    IK1 = GK1 * math.sqrt(ko) * rk1 * y_new[XK1_IDX] * (V - EK)

    # INaCa (simplified 3-state model)
    kna1 = 15.0
    kna2 = 5.0
    kna3 = 88.12
    kasymm = 12.5
    wna = 6.0e4
    wca = 6.0e4
    wnaca = 5.0e3
    kcaon = 1.5e6
    kcaoff = 5.0e3
    qna = 0.5224
    qca = 0.1670

    hca = safe_exp(qca * V * F / (R * T))
    hna = safe_exp(qna * V * F / (R * T))

    # INaCa_i (80%)
    h1_i = 1.0 + nai / kna3 * (1.0 + hna)
    h2_i = nai * hna / (kna3 * h1_i)
    h3_i = 1.0 / h1_i
    h4_i = 1.0 + nai / kna1 * (1.0 + nai / kna2)
    h5_i = nai * nai / (h4_i * kna1 * kna2)
    h6_i = 1.0 / h4_i
    h7_i = 1.0 + nao / kna3 * (1.0 + 1.0 / hna)
    h8_i = nao / (kna3 * hna * h7_i)
    h9_i = 1.0 / h7_i
    h10_i = kasymm + 1.0 + nao / kna1 * (1.0 + nao / kna2)
    h11_i = nao * nao / (h10_i * kna1 * kna2)
    h12_i = 1.0 / h10_i

    k1_i = h12_i * cao * kcaon
    k2_i = kcaoff
    k3p_i = h9_i * wca
    k3pp_i = h8_i * wnaca
    k3_i = k3p_i + k3pp_i
    k4p_i = h3_i * wca / hca
    k4pp_i = h2_i * wnaca
    k4_i = k4p_i + k4pp_i
    k5_i = kcaoff
    k6_i = h6_i * cai * kcaon
    k7_i = h5_i * h2_i * wna
    k8_i = h8_i * h11_i * wna

    x1_i = k2_i * k4_i * (k7_i + k6_i) + k5_i * k7_i * (k2_i + k3_i)
    x2_i = k1_i * k7_i * (k4_i + k5_i) + k4_i * k6_i * (k1_i + k8_i)
    x3_i = k1_i * k3_i * (k7_i + k6_i) + k8_i * k6_i * (k2_i + k3_i)
    x4_i = k2_i * k8_i * (k4_i + k5_i) + k3_i * k5_i * (k1_i + k8_i)

    E1_i = x1_i / (x1_i + x2_i + x3_i + x4_i)
    E2_i = x2_i / (x1_i + x2_i + x3_i + x4_i)
    E3_i = x3_i / (x1_i + x2_i + x3_i + x4_i)
    E4_i = x4_i / (x1_i + x2_i + x3_i + x4_i)

    allo_i = 1.0 / (1.0 + (0.0015 / cai) ** 2)
    JncxNa_i = 3.0 * (E4_i * k7_i - E1_i * k8_i) + E3_i * k4pp_i - E2_i * k3pp_i
    JncxCa_i = E2_i * k2_i - E1_i * k1_i
    INaCa_i = 0.8 * Gncx * allo_i * (JncxNa_i + 2.0 * JncxCa_i)

    # INaCa_ss (20%)
    h1_ss = 1.0 + nass / kna3 * (1.0 + hna)
    h2_ss = nass * hna / (kna3 * h1_ss)
    h3_ss = 1.0 / h1_ss
    h4_ss = 1.0 + nass / kna1 * (1.0 + nass / kna2)
    h5_ss = nass * nass / (h4_ss * kna1 * kna2)
    h6_ss = 1.0 / h4_ss

    k6_ss = h6_ss * cass * kcaon
    k7_ss = h5_ss * h2_ss * wna

    x1_ss = k2_i * k4_i * (k7_ss + k6_ss) + k5_i * k7_ss * (k2_i + k3_i)
    x2_ss = k1_i * k7_ss * (k4_i + k5_i) + k4_i * k6_ss * (k1_i + k8_i)
    x3_ss = k1_i * k3_i * (k7_ss + k6_ss) + k8_i * k6_ss * (k2_i + k3_i)
    x4_ss = k2_i * k8_i * (k4_i + k5_i) + k3_i * k5_i * (k1_i + k8_i)

    E1_ss = x1_ss / (x1_ss + x2_ss + x3_ss + x4_ss)
    E2_ss = x2_ss / (x1_ss + x2_ss + x3_ss + x4_ss)
    E3_ss = x3_ss / (x1_ss + x2_ss + x3_ss + x4_ss)
    E4_ss = x4_ss / (x1_ss + x2_ss + x3_ss + x4_ss)

    allo_ss = 1.0 / (1.0 + (0.0015 / cass) ** 2)
    JncxNa_ss = 3.0 * (E4_ss * k7_ss - E1_ss * k8_i) + E3_ss * k4pp_i - E2_ss * k3pp_i
    JncxCa_ss = E2_ss * k2_i - E1_ss * k1_i
    INaCa_ss = 0.2 * Gncx * allo_ss * (JncxNa_ss + 2.0 * JncxCa_ss)

    INaCa = INaCa_i + INaCa_ss

    # INaK (simplified)
    k1p = 949.5
    k1m = 182.4
    k2p = 687.2
    k2m = 39.4
    k3p = 1899.0
    k3m = 79300.0
    k4p = 639.0
    k4m = 40.0
    Knai0 = 9.073
    Knao0 = 27.78
    Kki = 0.5
    Kko = 0.3582
    MgADP = 0.05
    MgATP = 9.8
    Kmgatp = 1.698e-7
    H = 1.0e-7
    eP = 4.2
    Khp = 1.698e-7
    Knap = 224.0
    Kxkur = 292.0

    delta_nak = -0.155
    Knai = Knai0 * safe_exp(delta_nak * vfrt / 3.0)
    Knao = Knao0 * safe_exp((1.0 - delta_nak) * vfrt / 3.0)
    P = eP / (1.0 + H / Khp + nai / Knap + ki / Kxkur)

    a1 = k1p * (nai / Knai) ** 3 / ((1.0 + nai / Knai) ** 3 + (1.0 + ki / Kki) ** 2 - 1.0)
    b1 = k1m * MgADP
    a2 = k2p
    b2 = k2m * (nao / Knao) ** 3 / ((1.0 + nao / Knao) ** 3 + (1.0 + ko / Kko) ** 2 - 1.0)
    a3 = k3p * (ko / Kko) ** 2 / ((1.0 + nao / Knao) ** 3 + (1.0 + ko / Kko) ** 2 - 1.0)
    b3 = k3m * P * H / (1.0 + MgATP / Kmgatp)
    a4 = k4p * MgATP / Kmgatp / (1.0 + MgATP / Kmgatp)
    b4 = k4m * (ki / Kki) ** 2 / ((1.0 + nai / Knai) ** 3 + (1.0 + ki / Kki) ** 2 - 1.0)

    x1_nak = a4 * a1 * a2 + b2 * b4 * b3 + a2 * b4 * b3 + b3 * a1 * a2
    x2_nak = b2 * b1 * b4 + a1 * a2 * a3 + a3 * b1 * b4 + a2 * a3 * b4
    x3_nak = a2 * a3 * a4 + b3 * b2 * b1 + b2 * b1 * a4 + a3 * a4 * b1
    x4_nak = b4 * b3 * b2 + a3 * a4 * a1 + b2 * a4 * a1 + b3 * b2 * a1

    E1_nak = x1_nak / (x1_nak + x2_nak + x3_nak + x4_nak)
    E2_nak = x2_nak / (x1_nak + x2_nak + x3_nak + x4_nak)
    E3_nak = x3_nak / (x1_nak + x2_nak + x3_nak + x4_nak)
    E4_nak = x4_nak / (x1_nak + x2_nak + x3_nak + x4_nak)

    JnakNa = 3.0 * (E1_nak * a3 - E2_nak * b3)
    JnakK = 2.0 * (E4_nak * b1 - E3_nak * a1)
    INaK = Pnak * (JnakNa + JnakK)

    # Background currents
    IKb = GKb * (V - EK)
    INab = PNab * vffrt * (nai * safe_exp(vfrt) - nao) / (safe_exp(vfrt) - 1.0) if abs(V) > 1e-6 else PNab * F * (nai - nao)
    ICab = PCab * 4.0 * vffrt * (cai * safe_exp(2.0 * vfrt) - 0.341 * cao) / (safe_exp(2.0 * vfrt) - 1.0) if abs(V) > 1e-6 else PCab * 4.0 * F * (cai - 0.341 * cao)
    IpCa = GpCa * cai / (0.0005 + cai)

    # =========================================================================
    # Calcium Handling
    # =========================================================================

    # Diffusion fluxes
    JdiffNa = (nass - nai) / 2.0
    JdiffK = (kss - ki) / 2.0
    Jdiff = (cass - cai) / 0.2

    # SR release (RyR)
    a_rel = 0.5 * bt
    Jrel_inf_tmp = a_rel * (-ICaL) / (1.0 + (1.5 / cajsr) ** 8)
    if celltype == 2:  # M-cell
        Jrel_inf = Jrel_inf_tmp * 1.7
    else:
        Jrel_inf = Jrel_inf_tmp

    tau_rel_tmp = bt / (1.0 + 0.0123 / cajsr)
    tau_rel = max(0.001, tau_rel_tmp)

    Jrelnp = y[JRELNP_IDX]
    Jrelp = y[JRELP_IDX]

    Jrelnp_new = Jrel_inf - (Jrel_inf - Jrelnp) * safe_exp(-dt / tau_rel)

    btp = 1.25 * bt
    a_relp = 0.5 * btp
    Jrel_infp_tmp = a_relp * (-ICaL) / (1.0 + (1.5 / cajsr) ** 8)
    if celltype == 2:
        Jrel_infp = Jrel_infp_tmp * 1.7
    else:
        Jrel_infp = Jrel_infp_tmp

    tau_relp_tmp = btp / (1.0 + 0.0123 / cajsr)
    tau_relp = max(0.001, tau_relp_tmp)

    Jrelp_new = Jrel_infp - (Jrel_infp - Jrelp) * safe_exp(-dt / tau_relp)

    fJrelp = fCaMKp
    Jrel = (1.0 - fJrelp) * Jrelnp_new + fJrelp * Jrelp_new

    y_new[JRELNP_IDX] = Jrelnp_new
    y_new[JRELP_IDX] = Jrelp_new

    # SERCA uptake
    upScale = 1.0
    if celltype == 1:  # Epi
        upScale = 1.3

    Jupnp = upScale * 0.004375 * cai / (cai + 0.00092)
    Jupp = upScale * 2.75 * 0.004375 * cai / (cai + 0.00092 - 0.00017)
    fJupp = fCaMKp
    Jup = (1.0 - fJupp) * Jupnp + fJupp * Jupp - 0.0039375 * cansr / 15.0

    # SR transfer
    Jtr = (cansr - cajsr) / 100.0

    # Buffering
    Bcai = 1.0 / (1.0 + cmdnmax * kmcmdn / (kmcmdn + cai) ** 2 +
                  trpnmax * kmtrpn / (kmtrpn + cai) ** 2)
    Bcass = 1.0 / (1.0 + BSRmax * KmBSR / (KmBSR + cass) ** 2 +
                   BSLmax * KmBSL / (KmBSL + cass) ** 2)
    Bcajsr = 1.0 / (1.0 + csqnmax * kmcsqn / (kmcsqn + cajsr) ** 2)

    # =========================================================================
    # Update Concentrations
    # =========================================================================

    # Na+
    dnai = -(INa + INaL + 3.0 * INaCa_i + 3.0 * INaK + INab) * Acap / (F * vmyo) + JdiffNa * vss / vmyo
    dnass = -(ICaNa + 3.0 * INaCa_ss) * Acap / (F * vss) - JdiffNa
    y_new[NAI_IDX] = nai + dt * dnai
    y_new[NASS_IDX] = nass + dt * dnass

    # K+
    dki = -(Ito + IKr + IKs + IK1 + IKb + Istim - 2.0 * INaK) * Acap / (F * vmyo) + JdiffK * vss / vmyo
    dkss = -ICaK * Acap / (F * vss) - JdiffK
    y_new[KI_IDX] = ki + dt * dki
    y_new[KSS_IDX] = kss + dt * dkss

    # Ca2+
    dcai = Bcai * (-(IpCa + ICab - 2.0 * INaCa_i) * Acap / (2.0 * F * vmyo) -
                   Jup * vnsr / vmyo + Jdiff * vss / vmyo)
    dcass = Bcass * (-(ICaL - 2.0 * INaCa_ss) * Acap / (2.0 * F * vss) +
                     Jrel * vjsr / vss - Jdiff)
    dcansr = Jup - Jtr * vjsr / vnsr
    dcajsr = Bcajsr * (Jtr - Jrel)

    y_new[CAI_IDX] = max(1e-8, cai + dt * dcai)
    y_new[CASS_IDX] = max(1e-8, cass + dt * dcass)
    y_new[CANSR_IDX] = max(1e-8, cansr + dt * dcansr)
    y_new[CAJSR_IDX] = max(1e-8, cajsr + dt * dcajsr)

    # =========================================================================
    # Update Membrane Potential
    # =========================================================================
    Iion = INa + INaL + Ito + ICaL + ICaNa + ICaK + IKr + IKs + IK1 + \
           INaCa + INaK + INab + IKb + IpCa + ICab + Istim

    y_new[V_IDX] = V - dt * Iion

    return y_new


# =============================================================================
# Parallel Tissue Kernel
# =============================================================================

@njit(parallel=True, cache=True)
def ionic_step_tissue(
    states, dt, stim_mask, stim_amplitude,
    celltype,
    F, R, T, RTF,
    nao, ko, cao,
    GNa, GNaL, Gto, PCa, GKr, GKs, GK1, Gncx, Pnak, GKb, PNab, PCab, GpCa,
    thL, Acap, vmyo, vnsr, vjsr, vss,
    cmdnmax, kmcmdn, trpnmax, kmtrpn,
    BSRmax, KmBSR, BSLmax, KmBSL, csqnmax, kmcsqn,
    bt, CaMKo, KmCaM, KmCaMK, aCaMK, bCaMK
):
    """
    Parallel ionic step for 2D tissue.

    Parameters
    ----------
    states : ndarray (ny, nx, 41)
        State array for all cells
    dt : float
        Time step (ms)
    stim_mask : ndarray (ny, nx)
        Boolean mask for stimulated cells
    stim_amplitude : float
        Stimulus current magnitude (uA/uF)
    ... : model parameters

    Returns
    -------
    states_new : ndarray (ny, nx, 41)
        Updated state array
    """
    ny, nx, n_states = states.shape
    states_new = np.empty_like(states)

    for i in prange(ny):
        for j in range(nx):
            Istim = -stim_amplitude if stim_mask[i, j] else 0.0

            states_new[i, j, :] = ionic_step_single_cell(
                states[i, j, :], dt, Istim,
                celltype,
                F, R, T, RTF,
                nao, ko, cao,
                GNa, GNaL, Gto, PCa, GKr, GKs, GK1, Gncx, Pnak, GKb, PNab, PCab, GpCa,
                thL, Acap, vmyo, vnsr, vjsr, vss,
                cmdnmax, kmcmdn, trpnmax, kmtrpn,
                BSRmax, KmBSR, BSLmax, KmBSL, csqnmax, kmcsqn,
                bt, CaMKo, KmCaM, KmCaMK, aCaMK, bCaMK
            )

    return states_new


# =============================================================================
# Parameter Pack Helper
# =============================================================================

def get_kernel_params(model):
    """
    Extract parameters from ORdModel for kernel calls.

    Returns tuple of all parameters needed by ionic_step_tissue.
    """
    p = model.params

    return (
        int(model.celltype),
        p.F, p.R, p.T, p.RTF,
        p.nao, p.ko, p.cao,
        p.GNa, model.GNaL, model.Gto, model.PCa, model.GKr, model.GKs,
        model.GK1, model.Gncx, model.Pnak, model.GKb, p.PNab, p.PCab, p.GpCa,
        p.thL, p.Acap, p.vmyo, p.vnsr, p.vjsr, p.vss,
        model.cmdnmax, p.kmcmdn, p.trpnmax, p.kmtrpn,
        p.BSRmax, p.KmBSR, p.BSLmax, p.KmBSL, p.csqnmax, p.kmcsqn,
        p.bt, p.CaMKo, p.KmCaM, p.KmCaMK, p.aCaMK, p.bCaMK
    )
