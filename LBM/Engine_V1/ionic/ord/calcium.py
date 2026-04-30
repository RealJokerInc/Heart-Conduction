"""
Calcium Handling for ORd Model

Contains:
- SR calcium release (Jrel) - ryanodine receptor
- SERCA calcium uptake (Jup)
- Calcium diffusion between compartments
- Calcium buffering in cytosol, subspace, and SR

Compartments:
- cai: Cytosolic Ca2+ (bulk)
- cass: Subspace Ca2+ (near L-type channels and RyR)
- cansr: Network SR Ca2+ (free)
- cajsr: Junctional SR Ca2+ (release site)
"""

import torch
from typing import Tuple


# =============================================================================
# SR Calcium Release (Jrel)
# =============================================================================

def J_rel(cajsr: torch.Tensor, ICaL: torch.Tensor,
          Jrelnp: torch.Tensor, Jrelp: torch.Tensor,
          fCaMKp: torch.Tensor, dt: float,
          a_rel: float = 0.5, bt: float = 4.75,
          cajsr_half: float = 1.5,
          Jrel_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SR calcium release via ryanodine receptors.

    Release is triggered by ICaL (calcium-induced calcium release, CICR).
    Both non-phosphorylated and phosphorylated pathways exist.

    Parameters
    ----------
    cajsr : JSR Ca2+ concentration (mM)
    ICaL : L-type Ca current (uA/uF) - negative inward current
    Jrelnp : Non-phosphorylated release state variable
    Jrelp : Phosphorylated release state variable
    fCaMKp : CaMKII phosphorylation factor
    dt : Time step (ms)
    a_rel : Base release amplitude factor
    bt : Time constant base (ms)
    cajsr_half : JSR Ca for half-max release (mM)

    Returns
    -------
    Jrel : Total SR release flux (mM/ms)
    Jrelnp_new : Updated non-phosphorylated state
    Jrelp_new : Updated phosphorylated state
    """
    # Protect against division by zero
    cajsr_safe = torch.clamp(cajsr, min=1e-10)

    # JSR Ca-dependent release availability
    frel_inf = 1.0 / (1.0 + (cajsr_half / cajsr_safe) ** 8)

    # Non-phosphorylated pathway
    a_rel_np = a_rel * Jrel_scale * bt
    Jrel_inf_np = a_rel_np * torch.clamp(-ICaL, min=0.0) / (1.0 + (cajsr_half / cajsr_safe) ** 8)

    tau_rel_np = bt / (1.0 + 0.0123 / cajsr_safe)
    tau_rel_np = torch.clamp(tau_rel_np, min=0.005)  # Minimum 5 us

    # Rush-Larsen update
    Jrelnp_new = Jrel_inf_np - (Jrel_inf_np - Jrelnp) * torch.exp(-dt / tau_rel_np)

    # Phosphorylated pathway (uses btp = 1.25 * bt)
    btp = 1.25 * bt
    a_rel_p = a_rel * Jrel_scale * btp
    Jrel_inf_p = a_rel_p * torch.clamp(-ICaL, min=0.0) / (1.0 + (cajsr_half / cajsr_safe) ** 8)

    tau_rel_p = btp / (1.0 + 0.0123 / cajsr_safe)
    tau_rel_p = torch.clamp(tau_rel_p, min=0.005)

    Jrelp_new = Jrel_inf_p - (Jrel_inf_p - Jrelp) * torch.exp(-dt / tau_rel_p)

    # Total release (weighted by phosphorylation)
    Jrel = (1.0 - fCaMKp) * Jrelnp_new + fCaMKp * Jrelp_new

    return Jrel, Jrelnp_new, Jrelp_new


# =============================================================================
# SERCA Calcium Uptake (Jup)
# =============================================================================

def J_up(cai: torch.Tensor, cansr: torch.Tensor, fCaMKp: torch.Tensor,
         Jup_max: float = 0.004375, Kmup: float = 0.00092,
         nsrbar: float = 15.0, Jup_scale: float = 1.0) -> torch.Tensor:
    """
    SERCA pump calcium uptake into SR.

    Uses Michaelis-Menten kinetics. Phosphorylation increases
    both Vmax and affinity.

    Parameters
    ----------
    cai : Cytosolic Ca2+ (mM)
    cansr : NSR Ca2+ (mM)
    fCaMKp : CaMKII phosphorylation factor
    Jup_max : Maximum uptake rate (mM/ms)
    Kmup : Half-saturation constant (mM)
    nsrbar : Maximum NSR Ca2+ for leak calculation (mM)
    Jup_scale : Cell-type scaling factor

    Returns
    -------
    Jup : Net uptake flux (mM/ms), positive = uptake into SR
    """
    Jup_max_eff = Jup_max * Jup_scale

    # Non-phosphorylated uptake
    Jupnp = Jup_max_eff * cai / (cai + Kmup)

    # Phosphorylated uptake (175% higher Vmax = 2.75x, higher affinity)
    Jupp = Jup_max_eff * 2.75 * cai / (cai + Kmup - 0.00017)

    # Weighted by phosphorylation
    Jup_active = (1.0 - fCaMKp) * Jupnp + fCaMKp * Jupp

    # Passive leak from NSR (proportional to load)
    Jleak = 0.0039375 * cansr / nsrbar

    # Net uptake
    return Jup_active - Jleak


# =============================================================================
# Calcium Diffusion
# =============================================================================

def J_diff_Ca(cass: torch.Tensor, cai: torch.Tensor,
              tau_diff: float = 0.2) -> torch.Tensor:
    """
    Calcium diffusion from subspace to cytosol.

    Parameters
    ----------
    cass : Subspace Ca2+ (mM)
    cai : Cytosolic Ca2+ (mM)
    tau_diff : Diffusion time constant (ms)

    Returns
    -------
    Jdiff : Diffusion flux (mM/ms), positive = from ss to cytosol
    """
    return (cass - cai) / tau_diff


def J_diff_Na(nass: torch.Tensor, nai: torch.Tensor,
              tau_diff: float = 2.0) -> torch.Tensor:
    """
    Sodium diffusion from subspace to cytosol.

    Parameters
    ----------
    nass : Subspace Na+ (mM)
    nai : Cytosolic Na+ (mM)
    tau_diff : Diffusion time constant (ms)

    Returns
    -------
    JdiffNa : Diffusion flux (mM/ms)
    """
    return (nass - nai) / tau_diff


def J_diff_K(kss: torch.Tensor, ki: torch.Tensor,
             tau_diff: float = 2.0) -> torch.Tensor:
    """
    Potassium diffusion from subspace to cytosol.

    Parameters
    ----------
    kss : Subspace K+ (mM)
    ki : Cytosolic K+ (mM)
    tau_diff : Diffusion time constant (ms)

    Returns
    -------
    JdiffK : Diffusion flux (mM/ms)
    """
    return (kss - ki) / tau_diff


def J_tr(cansr: torch.Tensor, cajsr: torch.Tensor,
         tau_tr: float = 100.0) -> torch.Tensor:
    """
    Calcium transfer from NSR to JSR.

    Parameters
    ----------
    cansr : NSR Ca2+ (mM)
    cajsr : JSR Ca2+ (mM)
    tau_tr : Transfer time constant (ms)

    Returns
    -------
    Jtr : Transfer flux (mM/ms), positive = from NSR to JSR
    """
    return (cansr - cajsr) / tau_tr


# =============================================================================
# Calcium Buffering
# =============================================================================

def buffer_factor_cai(cai: torch.Tensor,
                      cmdnmax: float = 0.05, kmcmdn: float = 0.00238,
                      trpnmax: float = 0.07, kmtrpn: float = 0.0005) -> torch.Tensor:
    """
    Cytosolic calcium buffer factor.

    Accounts for calmodulin (CMDN) and troponin (TRPN) buffering.

    Parameters
    ----------
    cai : Cytosolic Ca2+ (mM)
    cmdnmax : Total calmodulin concentration (mM)
    kmcmdn : Calmodulin Ca2+ affinity (mM)
    trpnmax : Total troponin concentration (mM)
    kmtrpn : Troponin Ca2+ affinity (mM)

    Returns
    -------
    bcai : Buffer factor (dimensionless, 0 < bcai < 1)
    """
    bcmdn = cmdnmax * kmcmdn / (kmcmdn + cai) ** 2
    btrpn = trpnmax * kmtrpn / (kmtrpn + cai) ** 2
    return 1.0 / (1.0 + bcmdn + btrpn)


def buffer_factor_cass(cass: torch.Tensor,
                       BSRmax: float = 0.047, KmBSR: float = 0.00087,
                       BSLmax: float = 1.124, KmBSL: float = 0.0087) -> torch.Tensor:
    """
    Subspace calcium buffer factor.

    Accounts for SR membrane (BSR) and sarcolemmal (BSL) buffers.

    Parameters
    ----------
    cass : Subspace Ca2+ (mM)
    BSRmax : SR buffer capacity (mM)
    KmBSR : SR buffer affinity (mM)
    BSLmax : SL buffer capacity (mM)
    KmBSL : SL buffer affinity (mM)

    Returns
    -------
    bcass : Buffer factor (dimensionless)
    """
    bsr = BSRmax * KmBSR / (KmBSR + cass) ** 2
    bsl = BSLmax * KmBSL / (KmBSL + cass) ** 2
    return 1.0 / (1.0 + bsr + bsl)


def buffer_factor_cajsr(cajsr: torch.Tensor,
                        csqnmax: float = 10.0, kmcsqn: float = 0.8) -> torch.Tensor:
    """
    JSR calcium buffer factor.

    Accounts for calsequestrin (CSQN) buffering.

    Parameters
    ----------
    cajsr : JSR Ca2+ (mM)
    csqnmax : Total calsequestrin concentration (mM)
    kmcsqn : Calsequestrin Ca2+ affinity (mM)

    Returns
    -------
    bcajsr : Buffer factor (dimensionless)
    """
    bcsqn = csqnmax * kmcsqn / (kmcsqn + cajsr) ** 2
    return 1.0 / (1.0 + bcsqn)


# =============================================================================
# Concentration Updates
# =============================================================================

def update_concentrations(
    # Current concentrations
    nai: torch.Tensor, nass: torch.Tensor,
    ki: torch.Tensor, kss: torch.Tensor,
    cai: torch.Tensor, cass: torch.Tensor,
    cansr: torch.Tensor, cajsr: torch.Tensor,
    # Currents (all in uA/uF)
    INa: torch.Tensor, INaL: torch.Tensor,
    ICaL: torch.Tensor, ICaNa: torch.Tensor, ICaK: torch.Tensor,
    ICab: torch.Tensor, INab: torch.Tensor, IpCa: torch.Tensor,
    INaCa_i: torch.Tensor, INaCa_ss: torch.Tensor,
    INaK: torch.Tensor,
    IKr: torch.Tensor, IKs: torch.Tensor, IK1: torch.Tensor,
    Ito: torch.Tensor, IKb: torch.Tensor,
    Istim: torch.Tensor,
    # Fluxes
    Jrel: torch.Tensor, Jup: torch.Tensor,
    # Time step
    dt: float,
    # Geometry (from params)
    Acap: float, vmyo: float, vnsr: float, vjsr: float, vss: float,
    # Diffusion time constants
    tau_diff_Na: float = 2.0, tau_diff_K: float = 2.0, tau_diff_Ca: float = 0.2,
    tau_tr: float = 100.0,
    # Buffer parameters
    cmdnmax: float = 0.05, kmcmdn: float = 0.00238,
    trpnmax: float = 0.07, kmtrpn: float = 0.0005,
    BSRmax: float = 0.047, KmBSR: float = 0.00087,
    BSLmax: float = 1.124, KmBSL: float = 0.0087,
    csqnmax: float = 10.0, kmcsqn: float = 0.8,
    cmdnmax_scale: float = 1.0  # Cell-type scale (1.3 for EPI)
) -> Tuple[torch.Tensor, ...]:
    """
    Update all ion concentrations for one time step.

    Uses Forward Euler integration with buffering factors.

    Parameters
    ----------
    Currents : All membrane currents (uA/uF)
    Fluxes : SR fluxes (mM/ms)
    dt : Time step (ms)

    Returns
    -------
    Updated concentrations: nai, nass, ki, kss, cai, cass, cansr, cajsr
    """
    F = 96485.0  # Faraday constant

    # Conversion factor: current (uA/uF) to flux (mM/ms)
    cm2_to_uL = Acap / F

    # =========================================================================
    # Sodium
    # =========================================================================
    # Subspace Na+
    INa_ss = ICaNa + 3.0 * INaCa_ss
    JdiffNa = J_diff_Na(nass, nai, tau_diff_Na)

    dnass = -INa_ss * cm2_to_uL / vss - JdiffNa
    nass_new = nass + dt * dnass

    # Cytosolic Na+
    INa_i = INa + INaL + INab + 3.0 * INaCa_i + 3.0 * INaK
    dnai = -INa_i * cm2_to_uL / vmyo + JdiffNa * vss / vmyo
    nai_new = nai + dt * dnai

    # =========================================================================
    # Potassium
    # =========================================================================
    # Subspace K+
    IK_ss = ICaK
    JdiffK = J_diff_K(kss, ki, tau_diff_K)

    dkss = -IK_ss * cm2_to_uL / vss - JdiffK
    kss_new = kss + dt * dkss

    # Cytosolic K+
    IK_i = Ito + IKr + IKs + IK1 + IKb - 2.0 * INaK + Istim
    dki = -IK_i * cm2_to_uL / vmyo + JdiffK * vss / vmyo
    ki_new = ki + dt * dki

    # =========================================================================
    # Calcium
    # =========================================================================
    # Buffer factors (cmdnmax scaled by cell type)
    bcai = buffer_factor_cai(cai, cmdnmax * cmdnmax_scale, kmcmdn, trpnmax, kmtrpn)
    bcass = buffer_factor_cass(cass, BSRmax, KmBSR, BSLmax, KmBSL)
    bcajsr = buffer_factor_cajsr(cajsr, csqnmax, kmcsqn)

    # Diffusion and transfer
    JdiffCa = J_diff_Ca(cass, cai, tau_diff_Ca)
    Jtr = J_tr(cansr, cajsr, tau_tr)

    # Subspace Ca2+
    ICa_ss = ICaL - 2.0 * INaCa_ss
    dcass = bcass * (-ICa_ss * cm2_to_uL / (2.0 * vss) +
                     Jrel * vjsr / vss - JdiffCa)
    cass_new = cass + dt * dcass

    # Cytosolic Ca2+
    ICa_i = IpCa + ICab - 2.0 * INaCa_i
    dcai = bcai * (-ICa_i * cm2_to_uL / (2.0 * vmyo) -
                   Jup * vnsr / vmyo + JdiffCa * vss / vmyo)
    cai_new = cai + dt * dcai

    # NSR Ca2+ (no buffering)
    dcansr = Jup - Jtr * vjsr / vnsr
    cansr_new = cansr + dt * dcansr

    # JSR Ca2+ (with CSQN buffering)
    dcajsr = bcajsr * (Jtr - Jrel)
    cajsr_new = cajsr + dt * dcajsr

    # Ensure positivity
    nai_new = torch.clamp(nai_new, min=1e-6)
    nass_new = torch.clamp(nass_new, min=1e-6)
    ki_new = torch.clamp(ki_new, min=1e-6)
    kss_new = torch.clamp(kss_new, min=1e-6)
    cai_new = torch.clamp(cai_new, min=1e-9)
    cass_new = torch.clamp(cass_new, min=1e-9)
    cansr_new = torch.clamp(cansr_new, min=1e-9)
    cajsr_new = torch.clamp(cajsr_new, min=1e-9)

    return nai_new, nass_new, ki_new, kss_new, cai_new, cass_new, cansr_new, cajsr_new
