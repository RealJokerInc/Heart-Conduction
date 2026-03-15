"""
O'Hara-Rudy (ORd 2011) Calcium Handling

SR calcium fluxes, subspace diffusion, and buffering for the ORd model.

Reference: O'Hara T, Virag L, Varro A, Rudy Y. PLoS Comput Biol. 2011;7(5):e1002061.

Components:
- Diffusion fluxes (subspace ↔ bulk)
- SR release (RyR)
- SERCA uptake
- SR transfer and leak
- Calcium buffering (rapid equilibrium)
"""

import numpy as np
from numba import njit


# =============================================================================
# Diffusion Fluxes (Subspace ↔ Bulk)
# =============================================================================

@njit(cache=True)
def J_diffNa(nass: float, nai: float) -> float:
    """
    Na+ diffusion from subspace to bulk.

    From C++: JdiffNa = (nass - nai) / 2.0

    Args:
        nass: Subspace Na+ (mM)
        nai: Bulk Na+ (mM)

    Returns:
        Diffusion flux (mM/ms)
    """
    return (nass - nai) / 2.0


@njit(cache=True)
def J_diffK(kss: float, ki: float) -> float:
    """
    K+ diffusion from subspace to bulk.

    From C++: JdiffK = (kss - ki) / 2.0

    Args:
        kss: Subspace K+ (mM)
        ki: Bulk K+ (mM)

    Returns:
        Diffusion flux (mM/ms)
    """
    return (kss - ki) / 2.0


@njit(cache=True)
def J_diffCa(cass: float, cai: float) -> float:
    """
    Ca2+ diffusion from subspace to bulk.

    From C++: Jdiff = (cass - cai) / 0.2

    Args:
        cass: Subspace Ca2+ (mM)
        cai: Bulk Ca2+ (mM)

    Returns:
        Diffusion flux (mM/ms)
    """
    return (cass - cai) / 0.2


# =============================================================================
# SR Release (RyR - Ryanodine Receptor)
# =============================================================================

@njit(cache=True)
def Jrel_inf(ICaL: float, cajsr: float, bt: float = 4.75) -> float:
    """
    Steady-state SR release flux.

    Release is proportional to ICaL and depends steeply on JSR load.

    From C++:
        a_rel = 0.5 * bt
        Jrel_inf = a_rel * (-ICaL) / (1 + (1.5/cajsr)^8)

    Args:
        ICaL: L-type Ca current (uA/uF) - should be negative during Ca entry
        cajsr: JSR Ca2+ (mM)
        bt: Base time constant (ms)

    Returns:
        Steady-state release (mM/ms)
    """
    a_rel = 0.5 * bt

    # Protect against division by zero
    if cajsr < 1e-10:
        cajsr = 1e-10

    return a_rel * (-ICaL) / (1.0 + (1.5 / cajsr) ** 8.0)


@njit(cache=True)
def tau_rel(cajsr: float, bt: float = 4.75) -> float:
    """
    SR release time constant.

    From C++:
        tau_rel = bt / (1 + 0.0123/cajsr)
        if (tau_rel < 0.005) tau_rel = 0.005

    Args:
        cajsr: JSR Ca2+ (mM)
        bt: Base time constant (ms)

    Returns:
        Release time constant (ms)
    """
    if cajsr < 1e-10:
        cajsr = 1e-10

    tau = bt / (1.0 + 0.0123 / cajsr)

    # Minimum time constant
    if tau < 0.005:
        tau = 0.005

    return tau


@njit(cache=True)
def compute_Jrel(Jrelnp: float, Jrelp: float, fJrelp: float,
                 ICaL: float, cajsr: float, dt: float,
                 bt: float = 4.75, celltype: int = 0) -> tuple:
    """
    Compute SR release flux with CaMKII modulation.

    Updates both non-phosphorylated and phosphorylated release states.

    Args:
        Jrelnp: Non-phosphorylated release state
        Jrelp: Phosphorylated release state
        fJrelp: CaMKII phosphorylation fraction
        ICaL: L-type Ca current (uA/uF)
        cajsr: JSR Ca2+ (mM)
        dt: Time step (ms)
        bt: Base time constant (ms)
        celltype: Cell type (0=endo, 1=epi, 2=M)

    Returns:
        Tuple of (Jrel, Jrelnp_new, Jrelp_new)
    """
    # Non-phosphorylated
    Jrel_inf_np = Jrel_inf(ICaL, cajsr, bt)
    if celltype == 2:  # M-cell
        Jrel_inf_np *= 1.7
    tau_rel_np = tau_rel(cajsr, bt)
    Jrelnp_new = Jrel_inf_np - (Jrel_inf_np - Jrelnp) * np.exp(-dt / tau_rel_np)

    # Phosphorylated (larger time constant base)
    btp = 1.25 * bt
    Jrel_inf_p = Jrel_inf(ICaL, cajsr, btp)
    if celltype == 2:  # M-cell
        Jrel_inf_p *= 1.7
    tau_rel_p = tau_rel(cajsr, btp)
    Jrelp_new = Jrel_inf_p - (Jrel_inf_p - Jrelp) * np.exp(-dt / tau_rel_p)

    # Total release (weighted by phosphorylation)
    Jrel = (1.0 - fJrelp) * Jrelnp_new + fJrelp * Jrelp_new

    return Jrel, Jrelnp_new, Jrelp_new


# =============================================================================
# SERCA Uptake
# =============================================================================

@njit(cache=True)
def compute_Jup(cai: float, cansr: float, fJupp: float,
                celltype: int = 0) -> tuple:
    """
    Compute SERCA uptake flux with CaMKII modulation.

    From C++:
        Jupnp = 0.004375 * cai / (cai + 0.00092)
        Jupp = 2.75 * 0.004375 * cai / (cai + 0.00092 - 0.00017)
        Jleak = 0.0039375 * cansr / 15.0
        Jup = (1-fJupp)*Jupnp + fJupp*Jupp - Jleak

    Args:
        cai: Bulk Ca2+ (mM)
        cansr: NSR Ca2+ (mM)
        fJupp: CaMKII phosphorylation fraction
        celltype: Cell type (0=endo, 1=epi, 2=M)

    Returns:
        Tuple of (Jup, Jleak)
    """
    # Non-phosphorylated SERCA
    Jupnp = 0.004375 * cai / (cai + 0.00092)

    # Phosphorylated SERCA (enhanced)
    Jupp = 2.75 * 0.004375 * cai / (cai + 0.00092 - 0.00017)

    # Epicardial scaling
    if celltype == 1:
        Jupnp *= 1.3
        Jupp *= 1.3

    # NSR leak
    Jleak = 0.0039375 * cansr / 15.0

    # Total uptake
    Jup = (1.0 - fJupp) * Jupnp + fJupp * Jupp - Jleak

    return Jup, Jleak


# =============================================================================
# SR Transfer
# =============================================================================

@njit(cache=True)
def J_tr(cansr: float, cajsr: float) -> float:
    """
    NSR → JSR transfer flux.

    From C++: Jtr = (cansr - cajsr) / 100.0

    Args:
        cansr: NSR Ca2+ (mM)
        cajsr: JSR Ca2+ (mM)

    Returns:
        Transfer flux (mM/ms)
    """
    return (cansr - cajsr) / 100.0


# =============================================================================
# Calcium Buffering (Rapid Equilibrium)
# =============================================================================

@njit(cache=True)
def beta_cai(cai: float, cmdnmax: float = 0.05, kmcmdn: float = 0.00238,
             trpnmax: float = 0.07, kmtrpn: float = 0.0005) -> float:
    """
    Cytoplasmic Ca2+ buffering factor.

    From C++:
        Bcai = 1/(1 + cmdnmax*kmcmdn/(kmcmdn+cai)^2 + trpnmax*kmtrpn/(kmtrpn+cai)^2)

    Args:
        cai: Bulk Ca2+ (mM)
        cmdnmax: Calmodulin max (mM)
        kmcmdn: Calmodulin Kd (mM)
        trpnmax: Troponin max (mM)
        kmtrpn: Troponin Kd (mM)

    Returns:
        Buffering factor (0-1)
    """
    denom = 1.0 + cmdnmax * kmcmdn / (kmcmdn + cai) ** 2.0 + \
            trpnmax * kmtrpn / (kmtrpn + cai) ** 2.0
    return 1.0 / denom


@njit(cache=True)
def beta_cass(cass: float, BSRmax: float = 0.047, KmBSR: float = 0.00087,
              BSLmax: float = 1.124, KmBSL: float = 0.0087) -> float:
    """
    Subspace Ca2+ buffering factor.

    From C++:
        Bcass = 1/(1 + BSRmax*KmBSR/(KmBSR+cass)^2 + BSLmax*KmBSL/(KmBSL+cass)^2)

    Args:
        cass: Subspace Ca2+ (mM)
        BSRmax: SR binding sites max (mM)
        KmBSR: SR binding Kd (mM)
        BSLmax: Sarcolemmal binding max (mM)
        KmBSL: Sarcolemmal binding Kd (mM)

    Returns:
        Buffering factor (0-1)
    """
    denom = 1.0 + BSRmax * KmBSR / (KmBSR + cass) ** 2.0 + \
            BSLmax * KmBSL / (KmBSL + cass) ** 2.0
    return 1.0 / denom


@njit(cache=True)
def beta_cajsr(cajsr: float, csqnmax: float = 10.0, kmcsqn: float = 0.8) -> float:
    """
    JSR Ca2+ buffering factor (calsequestrin).

    From C++:
        Bcajsr = 1/(1 + csqnmax*kmcsqn/(kmcsqn+cajsr)^2)

    Args:
        cajsr: JSR Ca2+ (mM)
        csqnmax: Calsequestrin max (mM)
        kmcsqn: Calsequestrin Kd (mM)

    Returns:
        Buffering factor (0-1)
    """
    denom = 1.0 + csqnmax * kmcsqn / (kmcsqn + cajsr) ** 2.0
    return 1.0 / denom


# =============================================================================
# Concentration Derivatives
# =============================================================================

@njit(cache=True)
def dcai_dt(ICaL: float, ICab: float, IpCa: float, INaCa_i: float,
            Jdiff: float, Jup: float,
            Bcai: float, vmyo: float, vnsr: float, vss: float,
            Acap: float, F: float) -> float:
    """
    Bulk Ca2+ derivative.

    From C++:
        cai += dt * Bcai * (-(IpCa+ICab-2*INaCa_i)*Acap/(2*F*vmyo) -
                            Jup*vnsr/vmyo + Jdiff*vss/vmyo)

    Returns:
        dcai/dt (mM/ms)
    """
    current_term = -(IpCa + ICab - 2.0 * INaCa_i) * Acap / (2.0 * F * vmyo)
    sr_term = -Jup * vnsr / vmyo
    diff_term = Jdiff * vss / vmyo

    return Bcai * (current_term + sr_term + diff_term)


@njit(cache=True)
def dcass_dt(ICaL: float, INaCa_ss: float, Jrel: float, Jdiff: float,
             Bcass: float, vss: float, vjsr: float,
             Acap: float, F: float) -> float:
    """
    Subspace Ca2+ derivative.

    From C++:
        cass += dt * Bcass * (-(ICaL-2*INaCa_ss)*Acap/(2*F*vss) +
                              Jrel*vjsr/vss - Jdiff)

    Returns:
        dcass/dt (mM/ms)
    """
    current_term = -(ICaL - 2.0 * INaCa_ss) * Acap / (2.0 * F * vss)
    rel_term = Jrel * vjsr / vss
    diff_term = -Jdiff

    return Bcass * (current_term + rel_term + diff_term)


@njit(cache=True)
def dcansr_dt(Jup: float, Jtr: float, vjsr: float, vnsr: float) -> float:
    """
    NSR Ca2+ derivative.

    From C++: cansr += dt * (Jup - Jtr*vjsr/vnsr)

    Returns:
        dcansr/dt (mM/ms)
    """
    return Jup - Jtr * vjsr / vnsr


@njit(cache=True)
def dcajsr_dt(Jtr: float, Jrel: float, Bcajsr: float) -> float:
    """
    JSR Ca2+ derivative.

    From C++: cajsr += dt * Bcajsr * (Jtr - Jrel)

    Returns:
        dcajsr/dt (mM/ms)
    """
    return Bcajsr * (Jtr - Jrel)
