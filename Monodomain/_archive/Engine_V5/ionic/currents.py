"""
O'Hara-Rudy (ORd 2011) Ionic Currents

All 15 ionic currents in the ORd model with CaMKII modulation.
Currents are in units of uA/uF.

Reference: O'Hara T, Virag L, Varro A, Rudy Y. PLoS Comput Biol. 2011;7(5):e1002061.

Currents:
1.  INa     - Fast sodium
2.  INaL    - Late sodium
3.  Ito     - Transient outward K+
4.  ICaL    - L-type calcium (GHK)
5.  ICaNa   - L-type Na component (GHK)
6.  ICaK    - L-type K component (GHK)
7.  IKr     - Rapid delayed rectifier K+
8.  IKs     - Slow delayed rectifier K+
9.  IK1     - Inward rectifier K+
10. INaCa_i - Na/Ca exchanger (bulk)
11. INaCa_ss - Na/Ca exchanger (subspace)
12. INaK    - Na/K pump
13. IKb     - Background K+
14. INab    - Background Na+ (GHK)
15. ICab    - Background Ca2+ (GHK)
16. IpCa    - Sarcolemmal Ca pump
"""

import numpy as np
from numba import njit


# =============================================================================
# Physical Constants (for GHK)
# =============================================================================
F = 96485.0   # Faraday constant (C/mol)
R = 8314.0    # Gas constant (mJ/(mol·K))
T = 310.0     # Temperature (K)


# =============================================================================
# INa - Fast Sodium Current
# =============================================================================

@njit(cache=True)
def I_Na(V: float, m: float, hf: float, hs: float, j: float,
         hsp: float, jp: float, nai: float, nao: float,
         fINap: float, GNa: float = 75.0) -> float:
    """
    Fast sodium current with CaMKII modulation.

    From C++:
        h = Ahf*hf + Ahs*hs
        hp = Ahf*hf + Ahs*hsp
        INa = GNa*(v-ENa)*m^3*((1-fINap)*h*j + fINap*hp*jp)

    Args:
        V: Membrane potential (mV)
        m, hf, hs, j: Standard gates
        hsp, jp: Phosphorylated gates
        nai, nao: Na+ concentrations (mM)
        fINap: CaMKII phosphorylation fraction
        GNa: Max conductance (mS/uF)

    Returns:
        INa current (uA/uF)
    """
    ENa = (R * T / F) * np.log(nao / nai)

    Ahf = 0.99
    Ahs = 1.0 - Ahf
    h = Ahf * hf + Ahs * hs
    hp = Ahf * hf + Ahs * hsp

    return GNa * (V - ENa) * m * m * m * ((1.0 - fINap) * h * j + fINap * hp * jp)


# =============================================================================
# INaL - Late Sodium Current
# =============================================================================

@njit(cache=True)
def I_NaL(V: float, mL: float, hL: float, hLp: float,
          nai: float, nao: float, fINaLp: float,
          GNaL: float = 0.0075) -> float:
    """
    Late sodium current with CaMKII modulation.

    Args:
        V: Membrane potential (mV)
        mL, hL: Standard gates
        hLp: Phosphorylated gate
        nai, nao: Na+ concentrations (mM)
        fINaLp: CaMKII phosphorylation fraction
        GNaL: Max conductance (mS/uF)

    Returns:
        INaL current (uA/uF)
    """
    ENa = (R * T / F) * np.log(nao / nai)
    return GNaL * (V - ENa) * mL * ((1.0 - fINaLp) * hL + fINaLp * hLp)


# =============================================================================
# Ito - Transient Outward K+ Current
# =============================================================================

@njit(cache=True)
def I_to(V: float, a: float, iF: float, iS: float,
         ap: float, iFp: float, iSp: float,
         ki: float, ko: float, fItop: float,
         Gto: float = 0.02) -> float:
    """
    Transient outward K+ current with CaMKII modulation.

    Args:
        V: Membrane potential (mV)
        a, iF, iS: Standard gates
        ap, iFp, iSp: Phosphorylated gates
        ki, ko: K+ concentrations (mM)
        fItop: CaMKII phosphorylation fraction
        Gto: Max conductance (mS/uF)

    Returns:
        Ito current (uA/uF)
    """
    EK = (R * T / F) * np.log(ko / ki)

    AiF = 1.0 / (1.0 + np.exp((V - 213.6) / 151.2))
    AiS = 1.0 - AiF
    i = AiF * iF + AiS * iS
    ip = AiF * iFp + AiS * iSp

    return Gto * (V - EK) * ((1.0 - fItop) * a * i + fItop * ap * ip)


# =============================================================================
# ICaL - L-type Calcium Current (GHK formulation)
# =============================================================================

@njit(cache=True)
def I_CaL(V: float, d: float, ff: float, fs: float,
          fcaf: float, fcas: float, jca: float, nca: float,
          ffp: float, fcafp: float,
          cass: float, nass: float, kss: float,
          cao: float, nao: float, ko: float,
          fICaLp: float, PCa: float = 0.0001) -> tuple:
    """
    L-type calcium current with GHK formulation and CaMKII modulation.

    Returns ICaL, ICaNa, ICaK components.

    Args:
        V: Membrane potential (mV)
        d, ff, fs, fcaf, fcas, jca, nca: Standard gates
        ffp, fcafp: Phosphorylated gates
        cass, nass, kss: Subspace concentrations (mM)
        cao, nao, ko: Extracellular concentrations (mM)
        fICaLp: CaMKII phosphorylation fraction
        PCa: Ca permeability (cm/s)

    Returns:
        Tuple of (ICaL, ICaNa, ICaK) in uA/uF
    """
    vffrt = V * F * F / (R * T)
    vfrt = V * F / (R * T)

    # Voltage inactivation
    Aff = 0.6
    Afs = 1.0 - Aff
    f = Aff * ff + Afs * fs
    fp = Aff * ffp + Afs * fs

    # Ca inactivation
    Afcaf = 0.3 + 0.6 / (1.0 + np.exp((V - 10.0) / 10.0))
    Afcas = 1.0 - Afcaf
    fca = Afcaf * fcaf + Afcas * fcas
    fcap = Afcaf * fcafp + Afcas * fcas

    # GHK driving forces
    if abs(V) < 1e-7:
        # Limit for V→0
        PhiCaL = 4.0 * F * (cass - 0.341 * cao)
        PhiCaNa = F * (0.75 * nass - 0.75 * nao)
        PhiCaK = F * (0.75 * kss - 0.75 * ko)
    else:
        exp2vfrt = np.exp(2.0 * vfrt)
        expvfrt = np.exp(vfrt)
        PhiCaL = 4.0 * vffrt * (cass * exp2vfrt - 0.341 * cao) / (exp2vfrt - 1.0)
        PhiCaNa = vffrt * (0.75 * nass * expvfrt - 0.75 * nao) / (expvfrt - 1.0)
        PhiCaK = vffrt * (0.75 * kss * expvfrt - 0.75 * ko) / (expvfrt - 1.0)

    # Permeabilities
    PCap = 1.1 * PCa
    PCaNa = 0.00125 * PCa
    PCaK = 3.574e-4 * PCa
    PCaNap = 0.00125 * PCap
    PCaKp = 3.574e-4 * PCap

    # Current components
    ICaL = (1.0 - fICaLp) * PCa * PhiCaL * d * (f * (1.0 - nca) + jca * fca * nca) + \
           fICaLp * PCap * PhiCaL * d * (fp * (1.0 - nca) + jca * fcap * nca)

    ICaNa = (1.0 - fICaLp) * PCaNa * PhiCaNa * d * (f * (1.0 - nca) + jca * fca * nca) + \
            fICaLp * PCaNap * PhiCaNa * d * (fp * (1.0 - nca) + jca * fcap * nca)

    ICaK = (1.0 - fICaLp) * PCaK * PhiCaK * d * (f * (1.0 - nca) + jca * fca * nca) + \
           fICaLp * PCaKp * PhiCaK * d * (fp * (1.0 - nca) + jca * fcap * nca)

    return ICaL, ICaNa, ICaK


# =============================================================================
# IKr - Rapid Delayed Rectifier K+ Current
# =============================================================================

@njit(cache=True)
def I_Kr(V: float, xrf: float, xrs: float,
         ki: float, ko: float, GKr: float = 0.046) -> float:
    """
    Rapid delayed rectifier K+ current.

    Args:
        V: Membrane potential (mV)
        xrf, xrs: Fast and slow activation gates
        ki, ko: K+ concentrations (mM)
        GKr: Max conductance (mS/uF)

    Returns:
        IKr current (uA/uF)
    """
    EK = (R * T / F) * np.log(ko / ki)

    Axrf = 1.0 / (1.0 + np.exp((V + 54.81) / 38.21))
    Axrs = 1.0 - Axrf
    xr = Axrf * xrf + Axrs * xrs

    rkr = 1.0 / (1.0 + np.exp((V + 55.0) / 75.0)) * \
          1.0 / (1.0 + np.exp((V - 10.0) / 30.0))

    return GKr * np.sqrt(ko / 5.4) * xr * rkr * (V - EK)


# =============================================================================
# IKs - Slow Delayed Rectifier K+ Current
# =============================================================================

@njit(cache=True)
def I_Ks(V: float, xs1: float, xs2: float,
         ki: float, nai: float, ko: float, nao: float,
         cai: float, GKs: float = 0.0034) -> float:
    """
    Slow delayed rectifier K+ current (Ca-dependent).

    Args:
        V: Membrane potential (mV)
        xs1, xs2: Activation gates
        ki, nai, ko, nao: Ion concentrations (mM)
        cai: Intracellular Ca2+ for Ca-dependent conductance
        GKs: Max conductance (mS/uF)

    Returns:
        IKs current (uA/uF)
    """
    EKs = (R * T / F) * np.log((ko + 0.01833 * nao) / (ki + 0.01833 * nai))
    KsCa = 1.0 + 0.6 / (1.0 + (3.8e-5 / cai) ** 1.4)

    return GKs * KsCa * xs1 * xs2 * (V - EKs)


# =============================================================================
# IK1 - Inward Rectifier K+ Current
# =============================================================================

@njit(cache=True)
def I_K1(V: float, xk1: float, ki: float, ko: float,
         GK1: float = 0.1908) -> float:
    """
    Inward rectifier K+ current.

    Args:
        V: Membrane potential (mV)
        xk1: Activation gate
        ki, ko: K+ concentrations (mM)
        GK1: Max conductance (mS/uF)

    Returns:
        IK1 current (uA/uF)
    """
    EK = (R * T / F) * np.log(ko / ki)
    rk1 = 1.0 / (1.0 + np.exp((V + 105.8 - 2.6 * ko) / 9.493))

    return GK1 * np.sqrt(ko) * rk1 * xk1 * (V - EK)


# =============================================================================
# INaCa - Na/Ca Exchanger (split into bulk and subspace)
# =============================================================================

@njit(cache=True)
def I_NaCa_i(V: float, nai: float, cai: float,
             nao: float, cao: float, Gncx: float = 0.0008) -> float:
    """
    Na/Ca exchanger current (bulk cytosol, 80% of total).

    Uses detailed 15-state model from ORd.

    Args:
        V: Membrane potential (mV)
        nai, cai: Intracellular concentrations (mM)
        nao, cao: Extracellular concentrations (mM)
        Gncx: Scaling factor

    Returns:
        INaCa_i current (uA/uF)
    """
    # NCX parameters
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
    zca = 2.0
    zna = 1.0

    hca = np.exp(qca * V * F / (R * T))
    hna = np.exp(qna * V * F / (R * T))

    # State calculations
    h1 = 1.0 + nai / kna3 * (1.0 + hna)
    h2 = (nai * hna) / (kna3 * h1)
    h3 = 1.0 / h1
    h4 = 1.0 + nai / kna1 * (1.0 + nai / kna2)
    h5 = nai * nai / (h4 * kna1 * kna2)
    h6 = 1.0 / h4
    h7 = 1.0 + nao / kna3 * (1.0 + 1.0 / hna)
    h8 = nao / (kna3 * hna * h7)
    h9 = 1.0 / h7
    h10 = kasymm + 1.0 + nao / kna1 * (1.0 + nao / kna2)
    h11 = nao * nao / (h10 * kna1 * kna2)
    h12 = 1.0 / h10

    k1 = h12 * cao * kcaon
    k2 = kcaoff
    k3p = h9 * wca
    k3pp = h8 * wnaca
    k3 = k3p + k3pp
    k4p = h3 * wca / hca
    k4pp = h2 * wnaca
    k4 = k4p + k4pp
    k5 = kcaoff
    k6 = h6 * cai * kcaon
    k7 = h5 * h2 * wna
    k8 = h8 * h11 * wna

    x1 = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3)
    x2 = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
    x3 = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3)
    x4 = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)

    E1 = x1 / (x1 + x2 + x3 + x4)
    E2 = x2 / (x1 + x2 + x3 + x4)
    E3 = x3 / (x1 + x2 + x3 + x4)
    E4 = x4 / (x1 + x2 + x3 + x4)

    KmCaAct = 150.0e-6
    allo = 1.0 / (1.0 + (KmCaAct / cai) ** 2.0)

    JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp
    JncxCa = E2 * k2 - E1 * k1

    return 0.8 * Gncx * allo * (zna * JncxNa + zca * JncxCa)


@njit(cache=True)
def I_NaCa_ss(V: float, nass: float, cass: float,
              nao: float, cao: float, Gncx: float = 0.0008) -> float:
    """
    Na/Ca exchanger current (subspace, 20% of total).

    Same formulation as I_NaCa_i but uses subspace concentrations.

    Args:
        V: Membrane potential (mV)
        nass, cass: Subspace concentrations (mM)
        nao, cao: Extracellular concentrations (mM)
        Gncx: Scaling factor

    Returns:
        INaCa_ss current (uA/uF)
    """
    # NCX parameters (same as bulk)
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
    zca = 2.0
    zna = 1.0

    hca = np.exp(qca * V * F / (R * T))
    hna = np.exp(qna * V * F / (R * T))

    # Use subspace concentrations
    h1 = 1.0 + nass / kna3 * (1.0 + hna)
    h2 = (nass * hna) / (kna3 * h1)
    h3 = 1.0 / h1
    h4 = 1.0 + nass / kna1 * (1.0 + nass / kna2)
    h5 = nass * nass / (h4 * kna1 * kna2)
    h6 = 1.0 / h4
    h7 = 1.0 + nao / kna3 * (1.0 + 1.0 / hna)
    h8 = nao / (kna3 * hna * h7)
    h9 = 1.0 / h7
    h10 = kasymm + 1.0 + nao / kna1 * (1.0 + nao / kna2)
    h11 = nao * nao / (h10 * kna1 * kna2)
    h12 = 1.0 / h10

    k1 = h12 * cao * kcaon
    k2 = kcaoff
    k3p = h9 * wca
    k3pp = h8 * wnaca
    k3 = k3p + k3pp
    k4p = h3 * wca / hca
    k4pp = h2 * wnaca
    k4 = k4p + k4pp
    k5 = kcaoff
    k6 = h6 * cass * kcaon
    k7 = h5 * h2 * wna
    k8 = h8 * h11 * wna

    x1 = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3)
    x2 = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
    x3 = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3)
    x4 = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)

    E1 = x1 / (x1 + x2 + x3 + x4)
    E2 = x2 / (x1 + x2 + x3 + x4)
    E3 = x3 / (x1 + x2 + x3 + x4)
    E4 = x4 / (x1 + x2 + x3 + x4)

    KmCaAct = 150.0e-6
    allo = 1.0 / (1.0 + (KmCaAct / cass) ** 2.0)

    JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp
    JncxCa = E2 * k2 - E1 * k1

    return 0.2 * Gncx * allo * (zna * JncxNa + zca * JncxCa)


# =============================================================================
# INaK - Na/K Pump
# =============================================================================

@njit(cache=True)
def I_NaK(V: float, nai: float, ki: float,
          nao: float, ko: float, Pnak: float = 30.0) -> float:
    """
    Na/K pump current.

    Uses detailed 15-state model from ORd.

    Args:
        V: Membrane potential (mV)
        nai, ki: Intracellular concentrations (mM)
        nao, ko: Extracellular concentrations (mM)
        Pnak: Max pump rate (uA/uF)

    Returns:
        INaK current (uA/uF)
    """
    # NaK parameters
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
    delta_nak = -0.1550
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
    zna = 1.0
    zk = 1.0

    Knai = Knai0 * np.exp(delta_nak * V * F / (3.0 * R * T))
    Knao = Knao0 * np.exp((1.0 - delta_nak) * V * F / (3.0 * R * T))

    P = eP / (1.0 + H / Khp + nai / Knap + ki / Kxkur)

    a1 = (k1p * (nai / Knai) ** 3.0) / ((1.0 + nai / Knai) ** 3.0 + (1.0 + ki / Kki) ** 2.0 - 1.0)
    b1 = k1m * MgADP
    a2 = k2p
    b2 = (k2m * (nao / Knao) ** 3.0) / ((1.0 + nao / Knao) ** 3.0 + (1.0 + ko / Kko) ** 2.0 - 1.0)
    a3 = (k3p * (ko / Kko) ** 2.0) / ((1.0 + nao / Knao) ** 3.0 + (1.0 + ko / Kko) ** 2.0 - 1.0)
    b3 = (k3m * P * H) / (1.0 + MgATP / Kmgatp)
    a4 = (k4p * MgATP / Kmgatp) / (1.0 + MgATP / Kmgatp)
    b4 = (k4m * (ki / Kki) ** 2.0) / ((1.0 + nai / Knai) ** 3.0 + (1.0 + ki / Kki) ** 2.0 - 1.0)

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

    return Pnak * (zna * JnakNa + zk * JnakK)


# =============================================================================
# Background Currents
# =============================================================================

@njit(cache=True)
def I_Kb(V: float, ki: float, ko: float, GKb: float = 0.003) -> float:
    """Background K+ current."""
    EK = (R * T / F) * np.log(ko / ki)
    xkb = 1.0 / (1.0 + np.exp(-(V - 14.48) / 18.34))
    return GKb * xkb * (V - EK)


@njit(cache=True)
def I_Nab(V: float, nai: float, nao: float, PNab: float = 3.75e-10) -> float:
    """Background Na+ current (GHK)."""
    vffrt = V * F * F / (R * T)
    vfrt = V * F / (R * T)
    if abs(V) < 1e-7:
        return PNab * F * (nai - nao)
    return PNab * vffrt * (nai * np.exp(vfrt) - nao) / (np.exp(vfrt) - 1.0)


@njit(cache=True)
def I_Cab(V: float, cai: float, cao: float, PCab: float = 2.5e-8) -> float:
    """Background Ca2+ current (GHK)."""
    vffrt = V * F * F / (R * T)
    vfrt = V * F / (R * T)
    if abs(V) < 1e-7:
        return PCab * 4.0 * F * (cai - 0.341 * cao)
    exp2vfrt = np.exp(2.0 * vfrt)
    return PCab * 4.0 * vffrt * (cai * exp2vfrt - 0.341 * cao) / (exp2vfrt - 1.0)


@njit(cache=True)
def I_pCa(cai: float, GpCa: float = 0.0005) -> float:
    """Sarcolemmal Ca2+ pump."""
    return GpCa * cai / (0.0005 + cai)
