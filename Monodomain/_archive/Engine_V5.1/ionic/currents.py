"""
Ion Current Calculations for ORd Model

Contains all 15 ionic currents:
- INa: Fast sodium
- INaL: Late sodium
- Ito: Transient outward potassium
- ICaL, ICaNa, ICaK: L-type calcium (Ca, Na, K components)
- IKr: Rapid delayed rectifier potassium
- IKs: Slow delayed rectifier potassium
- IK1: Inward rectifier potassium
- INaCa_i, INaCa_ss: Sodium-calcium exchanger (cytosol, subspace)
- INaK: Sodium-potassium pump
- INab: Background sodium
- ICab: Background calcium
- IpCa: Sarcolemmal calcium pump
- IKb: Background potassium

All functions are vectorized for batch operation over tissue tensors.
"""

import torch
from typing import Tuple
from ionic.gating import safe_exp


# =============================================================================
# Physical Constants and Reversal Potentials
# =============================================================================

# Constants at 37°C
R = 8314.0      # Gas constant (mJ/(mol·K))
T = 310.0       # Temperature (K)
F = 96485.0     # Faraday constant (C/mol)
RTF = R * T / F  # ~26.71 mV


def E_Na(nai: torch.Tensor, nao: float = 140.0) -> torch.Tensor:
    """Sodium reversal potential (mV)."""
    return RTF * torch.log(nao / nai)


def E_K(ki: torch.Tensor, ko: float = 5.4) -> torch.Tensor:
    """Potassium reversal potential (mV)."""
    return RTF * torch.log(ko / ki)


def E_Ca(cai: torch.Tensor, cao: float = 1.8) -> torch.Tensor:
    """Calcium reversal potential (mV)."""
    return 0.5 * RTF * torch.log(cao / cai)


def E_Ks(ki: torch.Tensor, nai: torch.Tensor,
         ko: float = 5.4, nao: float = 140.0, PKNa: float = 0.01833) -> torch.Tensor:
    """IKs reversal potential with Na permeability."""
    return RTF * torch.log((ko + PKNa * nao) / (ki + PKNa * nai))


# =============================================================================
# INa (Fast Sodium Current)
# =============================================================================

def I_Na(V: torch.Tensor, m: torch.Tensor,
         hf: torch.Tensor, hs: torch.Tensor, j: torch.Tensor,
         hsp: torch.Tensor, jp: torch.Tensor,
         nai: torch.Tensor, fCaMKp: torch.Tensor,
         GNa: float = 75.0, nao: float = 140.0) -> torch.Tensor:
    """
    Fast sodium current.

    Parameters
    ----------
    V : Membrane potential (mV)
    m : Activation gate
    hf, hs : Fast and slow inactivation gates
    j : Recovery gate
    hsp, jp : Phosphorylated inactivation and recovery gates
    nai : Intracellular Na+ (mM)
    fCaMKp : CaMKII phosphorylation factor
    GNa : Maximum conductance (mS/µF)
    nao : Extracellular Na+ (mM)

    Returns
    -------
    INa : Current density (µA/µF)
    """
    ENa = E_Na(nai, nao)

    # Non-phosphorylated pathway: fast hf + slow hs
    h = 0.99 * hf + 0.01 * hs
    INa_np = GNa * (m ** 3) * h * j * (V - ENa)

    # Phosphorylated pathway: fast hf + phosphorylated slow hsp
    hp = 0.99 * hf + 0.01 * hsp
    INa_p = GNa * (m ** 3) * hp * jp * (V - ENa)

    # Weighted by CaMKII phosphorylation
    return (1.0 - fCaMKp) * INa_np + fCaMKp * INa_p


# =============================================================================
# INaL (Late Sodium Current)
# =============================================================================

def I_NaL(V: torch.Tensor, mL: torch.Tensor,
          hL: torch.Tensor, hLp: torch.Tensor,
          nai: torch.Tensor, fCaMKp: torch.Tensor,
          GNaL: float = 0.0075, nao: float = 140.0,
          GNaL_scale: float = 1.0) -> torch.Tensor:
    """
    Late sodium current.

    Parameters
    ----------
    V : Membrane potential (mV)
    mL : Activation gate
    hL, hLp : Inactivation gates (normal and phosphorylated)
    nai : Intracellular Na+ (mM)
    fCaMKp : CaMKII phosphorylation factor
    GNaL : Maximum conductance (mS/µF)
    GNaL_scale : Cell-type scaling factor

    Returns
    -------
    INaL : Current density (µA/µF)
    """
    ENa = E_Na(nai, nao)
    GNaL_eff = GNaL * GNaL_scale

    # Non-phosphorylated
    INaL_np = GNaL_eff * mL * hL * (V - ENa)

    # Phosphorylated
    INaL_p = GNaL_eff * mL * hLp * (V - ENa)

    return (1.0 - fCaMKp) * INaL_np + fCaMKp * INaL_p


# =============================================================================
# Ito (Transient Outward Potassium Current)
# =============================================================================

def I_to(V: torch.Tensor, a: torch.Tensor,
         iF: torch.Tensor, iS: torch.Tensor,
         ap: torch.Tensor, iFp: torch.Tensor, iSp: torch.Tensor,
         ki: torch.Tensor, fCaMKp: torch.Tensor,
         Gto: float = 0.02, ko: float = 5.4,
         Gto_scale: float = 1.0, delta_epi: torch.Tensor = None) -> torch.Tensor:
    """
    Transient outward potassium current.

    Parameters
    ----------
    V : Membrane potential (mV)
    a : Activation gate
    iF, iS : Fast and slow inactivation gates
    ap, iFp, iSp : Phosphorylated gates
    ki : Intracellular K+ (mM)
    fCaMKp : CaMKII phosphorylation factor
    Gto : Maximum conductance (mS/µF)
    Gto_scale : Cell-type scaling factor
    delta_epi : Epicardial fraction (for EPI/M_CELL)

    Returns
    -------
    Ito : Current density (µA/µF)
    """
    EK = E_K(ki, ko)
    Gto_eff = Gto * Gto_scale

    # Fast/slow inactivation ratio (voltage-dependent)
    AiF = 1.0 / (1.0 + safe_exp((V - 213.6) / 151.2))
    AiS = 1.0 - AiF

    # Inactivation (weighted sum, same for all cell types)
    i = AiF * iF + AiS * iS
    ip = AiF * iFp + AiS * iSp

    # Non-phosphorylated
    Ito_np = Gto_eff * a * i * (V - EK)

    # Phosphorylated
    Ito_p = Gto_eff * ap * ip * (V - EK)

    return (1.0 - fCaMKp) * Ito_np + fCaMKp * Ito_p


# =============================================================================
# ICaL (L-type Calcium Current)
# =============================================================================

def I_CaL(V: torch.Tensor, d: torch.Tensor,
          ff: torch.Tensor, fs: torch.Tensor,
          fcaf: torch.Tensor, fcas: torch.Tensor,
          jca: torch.Tensor, nca: torch.Tensor,
          ffp: torch.Tensor, fcafp: torch.Tensor,
          cass: torch.Tensor, nass: torch.Tensor, kss: torch.Tensor,
          fCaMKp: torch.Tensor,
          PCa: float = 0.0001, cao: float = 1.8, nao: float = 140.0, ko: float = 5.4,
          PCa_scale: float = 1.0,
          PCaNa_frac: float = 0.00125, PCaK_frac: float = 3.574e-4
          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L-type calcium current with Na and K components.

    Uses Goldman-Hodgkin-Katz formulation for permeation.

    Parameters
    ----------
    V : Membrane potential (mV)
    d : Activation gate
    ff, fs : Fast and slow voltage inactivation
    fcaf, fcas : Fast and slow Ca-dependent inactivation
    jca : Ca-dependent recovery
    nca : Ca-dependent factor
    ffp, fcafp : Phosphorylated gates
    cass, nass, kss : Subspace concentrations
    fCaMKp : CaMKII phosphorylation factor
    PCa : Ca permeability (cm/s)

    Returns
    -------
    ICaL : Ca component (µA/µF)
    ICaNa : Na component (µA/µF)
    ICaK : K component (µA/µF)
    """
    # GHK driving force calculations
    vfrt = V * F / (R * T)
    vffrt = V * F * F / (R * T)  # V * F^2 / RT

    # Permeabilities (scaled by cell type)
    PCa_eff = PCa * PCa_scale
    PCaNa = PCa_eff * PCaNa_frac
    PCaK = PCa_eff * PCaK_frac

    # GHK for Ca2+ (z=2)
    exp2vfrt = safe_exp(2.0 * vfrt)
    PhiCa_L = 4.0 * vffrt * (cass * exp2vfrt - 0.341 * cao) / (exp2vfrt - 1.0)

    # GHK for Na+ (z=1)
    expvfrt = safe_exp(vfrt)
    PhiCaNa = vffrt * (0.75 * nass * expvfrt - 0.75 * nao) / (expvfrt - 1.0)

    # GHK for K+ (z=1)
    PhiCaK = vffrt * (0.75 * kss * expvfrt - 0.75 * ko) / (expvfrt - 1.0)

    # Handle V near 0 (L'Hopital's rule limit)
    PhiCa_L = torch.where(torch.abs(V) < 1e-7,
                          4.0 * F * (cass - 0.341 * cao),
                          PhiCa_L)
    PhiCaNa = torch.where(torch.abs(V) < 1e-7,
                          F * 0.75 * (nass - nao),
                          PhiCaNa)
    PhiCaK = torch.where(torch.abs(V) < 1e-7,
                         F * 0.75 * (kss - ko),
                         PhiCaK)

    # Gate combinations (from ORd C++ code)
    # Voltage inactivation: Aff=0.6, Afs=0.4
    Aff = 0.6
    Afs = 1.0 - Aff
    fv = Aff * ff + Afs * fs
    fvp = Aff * ffp + Afs * fs

    # Ca-dependent inactivation fraction (voltage-dependent)
    Afcaf = 0.3 + 0.6 / (1.0 + safe_exp((V - 10.0) / 10.0))
    Afcas = 1.0 - Afcaf
    fca = Afcaf * fcaf + Afcas * fcas
    fcap = Afcaf * fcafp + Afcas * fcas

    # Non-phosphorylated pathway
    # nca modulates between voltage-only and Ca-dependent inactivation
    ICaL_np = PCa_eff * PhiCa_L * d * (fv * (1.0 - nca) + jca * fca * nca)
    ICaNa_np = PCaNa * PhiCaNa * d * (fv * (1.0 - nca) + jca * fca * nca)
    ICaK_np = PCaK * PhiCaK * d * (fv * (1.0 - nca) + jca * fca * nca)

    # Phosphorylated pathway (1.1x PCa)
    PCap = 1.1 * PCa_eff
    PCaNap = 1.1 * PCaNa
    PCaKp = 1.1 * PCaK
    ICaL_p = PCap * PhiCa_L * d * (fvp * (1.0 - nca) + jca * fcap * nca)
    ICaNa_p = PCaNap * PhiCaNa * d * (fvp * (1.0 - nca) + jca * fcap * nca)
    ICaK_p = PCaKp * PhiCaK * d * (fvp * (1.0 - nca) + jca * fcap * nca)

    # Weighted by CaMKII
    ICaL = (1.0 - fCaMKp) * ICaL_np + fCaMKp * ICaL_p
    ICaNa = (1.0 - fCaMKp) * ICaNa_np + fCaMKp * ICaNa_p
    ICaK = (1.0 - fCaMKp) * ICaK_np + fCaMKp * ICaK_p

    return ICaL, ICaNa, ICaK


# =============================================================================
# IKr (Rapid Delayed Rectifier Potassium Current)
# =============================================================================

def I_Kr(V: torch.Tensor, xrf: torch.Tensor, xrs: torch.Tensor,
         ki: torch.Tensor,
         GKr: float = 0.046, ko: float = 5.4,
         GKr_scale: float = 1.0) -> torch.Tensor:
    """
    Rapid delayed rectifier potassium current.

    Parameters
    ----------
    V : Membrane potential (mV)
    xrf, xrs : Fast and slow activation gates
    ki : Intracellular K+ (mM)
    GKr : Maximum conductance (mS/µF)
    GKr_scale : Cell-type scaling factor

    Returns
    -------
    IKr : Current density (µA/µF)
    """
    EK = E_K(ki, ko)
    GKr_eff = GKr * GKr_scale * (ko / 5.4) ** 0.5  # Ko dependence

    # Fast/slow activation fraction
    Axrf = 1.0 / (1.0 + safe_exp((V + 54.81) / 38.21))
    xr = Axrf * xrf + (1.0 - Axrf) * xrs

    # Rectification
    rKr = 1.0 / (1.0 + safe_exp((V + 55.0) / 75.0)) * \
          1.0 / (1.0 + safe_exp((V - 10.0) / 30.0))

    return GKr_eff * xr * rKr * (V - EK)


# =============================================================================
# IKs (Slow Delayed Rectifier Potassium Current)
# =============================================================================

def I_Ks(V: torch.Tensor, xs1: torch.Tensor, xs2: torch.Tensor,
         ki: torch.Tensor, nai: torch.Tensor, cai: torch.Tensor,
         GKs: float = 0.0034, ko: float = 5.4, nao: float = 140.0,
         GKs_scale: float = 1.0) -> torch.Tensor:
    """
    Slow delayed rectifier potassium current.

    Parameters
    ----------
    V : Membrane potential (mV)
    xs1, xs2 : Activation gates
    ki : Intracellular K+ (mM)
    nai : Intracellular Na+ (mM)
    cai : Intracellular Ca2+ (mM) for Ca-dependent conductance
    GKs : Maximum conductance (mS/µF)
    GKs_scale : Cell-type scaling factor

    Returns
    -------
    IKs : Current density (µA/µF)
    """
    EKs = E_Ks(ki, nai, ko, nao)
    GKs_eff = GKs * GKs_scale

    # Ca-dependent conductance factor (increases IKs at higher Ca)
    KsCa = 1.0 + 0.6 / (1.0 + (3.8e-5 / (cai + 1e-10)) ** 1.4)

    return GKs_eff * KsCa * xs1 * xs2 * (V - EKs)


# =============================================================================
# IK1 (Inward Rectifier Potassium Current)
# =============================================================================

def I_K1(V: torch.Tensor, xk1: torch.Tensor,
         ki: torch.Tensor,
         GK1: float = 0.1908, ko: float = 5.4,
         GK1_scale: float = 1.0) -> torch.Tensor:
    """
    Inward rectifier potassium current.

    Parameters
    ----------
    V : Membrane potential (mV)
    xk1 : Activation gate
    ki : Intracellular K+ (mM)
    GK1 : Maximum conductance (mS/µF)
    GK1_scale : Cell-type scaling factor

    Returns
    -------
    IK1 : Current density (µA/µF)
    """
    EK = E_K(ki, ko)
    # Ko-dependence: sqrt(ko), NOT (ko/5.4)**0.5
    GK1_eff = GK1 * GK1_scale * (ko ** 0.5)

    # Rectification (from ORd C++ code)
    rk1 = 1.0 / (1.0 + safe_exp((V + 105.8 - 2.6 * ko) / 9.493))

    return GK1_eff * xk1 * rk1 * (V - EK)


# =============================================================================
# INaCa (Sodium-Calcium Exchanger)
# =============================================================================

def I_NaCa(V: torch.Tensor,
           nai: torch.Tensor, nass: torch.Tensor,
           cai: torch.Tensor, cass: torch.Tensor,
           Gncx: float = 0.0008, cao: float = 1.8, nao: float = 140.0,
           Gncx_scale: float = 1.0,
           KmCaAct: float = 150e-6, kna1: float = 15.0, kna2: float = 5.0,
           kna3: float = 88.12, kasymm: float = 12.5,
           wna: float = 6e4, wca: float = 6e4, wnaca: float = 5e3,
           kcaon: float = 1.5e6, kcaoff: float = 5e3,
           qna: float = 0.5224, qca: float = 0.167
           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sodium-calcium exchanger current (cytosolic and subspace components).

    Uses allosteric model with 3 Na+ and 1 Ca2+ binding sites.

    Parameters
    ----------
    V : Membrane potential (mV)
    nai, nass : Cytosolic and subspace Na+ (mM)
    cai, cass : Cytosolic and subspace Ca2+ (mM)
    Gncx : Maximum exchange rate
    Gncx_scale : Cell-type scaling factor

    Returns
    -------
    INaCa_i : Cytosolic component (µA/µF)
    INaCa_ss : Subspace component (µA/µF)
    """
    Gncx_eff = Gncx * Gncx_scale

    # Voltage-dependent factors
    hna = safe_exp(qna * V * F / (R * T))
    hca = safe_exp(qca * V * F / (R * T))

    # Cytosolic component
    h1_i = 1.0 + nai / kna3 * (1.0 + hna)
    h2_i = (nai * hna) / (kna3 * h1_i)
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

    allo_i = 1.0 / (1.0 + (KmCaAct / cai) ** 2)
    JncxNa_i = 3.0 * (E4_i * k7_i - E1_i * k8_i) + E3_i * k4pp_i - E2_i * k3pp_i
    JncxCa_i = E2_i * k2_i - E1_i * k1_i

    INaCa_i = 0.8 * Gncx_eff * allo_i * (JncxNa_i + 2.0 * JncxCa_i)

    # Subspace component (similar calculations with ss concentrations)
    h1_ss = 1.0 + nass / kna3 * (1.0 + hna)
    h2_ss = (nass * hna) / (kna3 * h1_ss)
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

    allo_ss = 1.0 / (1.0 + (KmCaAct / cass) ** 2)
    JncxNa_ss = 3.0 * (E4_ss * k7_ss - E1_ss * k8_i) + E3_ss * k4pp_i - E2_ss * k3pp_i
    JncxCa_ss = E2_ss * k2_i - E1_ss * k1_i

    INaCa_ss = 0.2 * Gncx_eff * allo_ss * (JncxNa_ss + 2.0 * JncxCa_ss)

    return INaCa_i, INaCa_ss


# =============================================================================
# INaK (Sodium-Potassium Pump)
# =============================================================================

def I_NaK(V: torch.Tensor,
          nai: torch.Tensor, nass: torch.Tensor, ki: torch.Tensor,
          Pnak: float = 30.0, ko: float = 5.4, nao: float = 140.0,
          Pnak_scale: float = 1.0,
          k1p: float = 949.5, k1m: float = 182.4,
          k2p: float = 687.2, k2m: float = 39.4,
          k3p: float = 1899.0, k3m: float = 79300.0,
          k4p: float = 639.0, k4m: float = 40.0,
          Knai0: float = 9.073, Knao0: float = 27.78,
          delta_eNa: float = -0.1550,
          Kki: float = 0.5, Kko: float = 0.3582,
          MgADP: float = 0.05, MgATP: float = 9.8,
          Kmgatp: float = 1.698e-7, H: float = 1e-7, eP: float = 4.2,
          Khp: float = 1.698e-7, Knap: float = 224.0, Kxkur: float = 292.0
          ) -> torch.Tensor:
    """
    Sodium-potassium pump current.

    Uses Albers-Post kinetic model.

    Parameters
    ----------
    V : Membrane potential (mV)
    nai, nass : Cytosolic and subspace Na+ (mM)
    ki : Intracellular K+ (mM)
    Pnak : Maximum pump rate

    Returns
    -------
    INaK : Current density (µA/µF)
    """
    # Voltage-dependent Na affinity
    Knai = Knai0 * safe_exp(delta_eNa * V * F / (3.0 * R * T))
    Knao = Knao0 * safe_exp((1.0 - delta_eNa) * V * F / (3.0 * R * T))

    # Phosphorylation state
    P = eP / (1.0 + H / Khp + nai / Knap + ki / Kxkur)

    # State variables
    a1 = k1p * (nai / Knai) ** 3 / ((1.0 + nai / Knai) ** 3 + (1.0 + ki / Kki) ** 2 - 1.0)
    b1 = k1m * MgADP
    a2 = k2p
    b2 = k2m * (nao / Knao) ** 3 / ((1.0 + nao / Knao) ** 3 + (1.0 + ko / Kko) ** 2 - 1.0)
    a3 = k3p * (ko / Kko) ** 2 / ((1.0 + nao / Knao) ** 3 + (1.0 + ko / Kko) ** 2 - 1.0)
    b3 = k3m * P * H / (1.0 + MgATP / Kmgatp)
    a4 = k4p * MgATP / Kmgatp / (1.0 + MgATP / Kmgatp)
    b4 = k4m * (ki / Kki) ** 2 / ((1.0 + nai / Knai) ** 3 + (1.0 + ki / Kki) ** 2 - 1.0)

    x1 = a4 * a1 * a2 + b2 * b4 * b3 + a2 * b4 * b3 + b3 * a1 * a2
    x2 = b2 * b1 * b4 + a1 * a2 * a3 + a3 * b1 * b4 + a2 * a3 * b4
    x3 = a2 * a3 * a4 + b3 * b2 * b1 + b2 * b1 * a4 + a3 * a4 * b1
    x4 = b4 * b3 * b2 + a3 * a4 * a1 + b2 * a4 * a1 + b3 * b2 * a1

    E1 = x1 / (x1 + x2 + x3 + x4)
    E2 = x2 / (x1 + x2 + x3 + x4)
    E3 = x3 / (x1 + x2 + x3 + x4)
    E4 = x4 / (x1 + x2 + x3 + x4)

    JnakNa = 3.0 * (E1 * a3 - E2 * b3)
    JnakK = 2.0 * (E4 * b1 - E3 * a1)

    return Pnak * Pnak_scale * (JnakNa + JnakK)


# =============================================================================
# Background Currents
# =============================================================================

def I_Nab(V: torch.Tensor, nai: torch.Tensor,
          PNab: float = 3.75e-10, nao: float = 140.0) -> torch.Tensor:
    """Background sodium current (GHK)."""
    vfrt = V * F / (R * T)

    INab = PNab * F * vfrt * (nai * safe_exp(vfrt) - nao) / (safe_exp(vfrt) - 1.0)

    # Handle V near 0
    INab = torch.where(torch.abs(V) < 1e-6,
                       PNab * F * F / (R * T) * (nai - nao),
                       INab)
    return INab


def I_Cab(V: torch.Tensor, cai: torch.Tensor,
          PCab: float = 2.5e-8, cao: float = 1.8) -> torch.Tensor:
    """Background calcium current (GHK)."""
    vfrt = V * F / (R * T)

    ICab = PCab * 4.0 * F * vfrt * (cai * safe_exp(2.0 * vfrt) - 0.341 * cao) / \
           (safe_exp(2.0 * vfrt) - 1.0)

    # Handle V near 0
    ICab = torch.where(torch.abs(V) < 1e-6,
                       PCab * 4.0 * F * F / (R * T) * (cai - 0.341 * cao),
                       ICab)
    return ICab


def I_Kb(V: torch.Tensor, ki: torch.Tensor,
         GKb: float = 0.003, ko: float = 5.4,
         GKb_scale: float = 1.0) -> torch.Tensor:
    """Background potassium current."""
    EK = E_K(ki, ko)
    xkb = 1.0 / (1.0 + safe_exp(-(V - 14.48) / 18.34))
    return GKb * GKb_scale * xkb * (V - EK)


def I_pCa(cai: torch.Tensor, GpCa: float = 0.0005) -> torch.Tensor:
    """Sarcolemmal calcium pump current."""
    return GpCa * cai / (0.0005 + cai)
