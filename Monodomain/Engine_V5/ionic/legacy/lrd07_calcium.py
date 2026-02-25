"""
LRd07 Calcium Handling

This module implements:
1. SR calcium fluxes (Iup, Itr, Ileak, Irel)
2. Analytical calcium buffering (cubic and quadratic solutions)

The analytical buffering approach solves for free Ca2+ from total Ca2+
rather than using quasi-steady-state approximations.

Reference: Livshitz LM, Rudy Y. Am J Physiol Heart Circ Physiol. 2007;292(6):H2854-66.
"""

import numpy as np
from numba import njit


# =============================================================================
# SR Calcium Fluxes
# =============================================================================

@njit(cache=True)
def I_up(Ca_i: float, I_up_max: float, Km_up: float) -> float:
    """
    SERCA pump uptake (cytosol → NSR).

    From MATLAB:
    iup = data.iupbar.*ca_i/(ca_i+data.kmup)

    Args:
        Ca_i: Free intracellular Ca2+ (mM)
        I_up_max: Max uptake rate (mM/ms)
        Km_up: Half-saturation (mM)

    Returns:
        Uptake flux (mM/ms)
    """
    return I_up_max * Ca_i / (Ca_i + Km_up)


@njit(cache=True)
def I_leak(nsr: float, I_up_max: float, NSR_max: float) -> float:
    """
    NSR leak (NSR → cytosol).

    From MATLAB:
    ileak = data.iupbar / data.nsrbar * nsr

    This maintains a balance with SERCA at rest.

    Args:
        nsr: NSR Ca2+ concentration (mM)
        I_up_max: Max uptake rate (mM/ms) - same as SERCA
        NSR_max: Max NSR concentration (mM)

    Returns:
        Leak flux (mM/ms)
    """
    return I_up_max / NSR_max * nsr


@njit(cache=True)
def I_tr(nsr: float, jsr: float, tau_tr: float) -> float:
    """
    NSR → JSR transfer flux.

    From MATLAB:
    itr = (nsr-jsr)./data.tautr

    Args:
        nsr: NSR Ca2+ concentration (mM)
        jsr: Free JSR Ca2+ concentration (mM)
        tau_tr: Transfer time constant (ms)

    Returns:
        Transfer flux (mM/ms)
    """
    return (nsr - jsr) / tau_tr


@njit(cache=True)
def Rel_infinity(I_CaL: float, jsr: float,
                  alpha_Rel: float, K_Rel_ss: float, qn: float) -> float:
    """
    Steady-state SR release flux.

    From MATLAB:
    Rel_ss = ilca.*data.alpha_Rel/(1+(data.K_Relss./jsr).^data.qn)

    The release is graded by ICaL magnitude and JSR load.
    Low JSR → reduced release (steep dependence via qn=9).

    Args:
        I_CaL: L-type Ca current, Ca component only (uA/uF)
        jsr: Free JSR Ca2+ (mM)
        alpha_Rel: Release rate constant (0.59375)
        K_Rel_ss: Half-sat for JSR dependence (mM)
        qn: Hill coefficient (9)

    Returns:
        Steady-state release (mM/ms)
    """
    # Protect against division by zero
    if jsr < 1e-10:
        jsr = 1e-10

    jsr_factor = 1.0 / (1.0 + (K_Rel_ss / jsr)**qn)
    return I_CaL * alpha_Rel * jsr_factor


@njit(cache=True)
def tau_Rel_func(jsr: float, tau_base: float) -> float:
    """
    JSR-dependent release time constant.

    From MATLAB:
    tau_Rel = data.tau./(1+0.0123./jsr)

    Lower JSR → faster time constant (quicker shutoff).

    Args:
        jsr: Free JSR Ca2+ (mM)
        tau_base: Base time constant (4.75 ms)

    Returns:
        Effective time constant (ms)
    """
    if jsr < 1e-10:
        jsr = 1e-10
    return tau_base / (1.0 + 0.0123 / jsr)


@njit(cache=True)
def dRel_dt(Rel: float, Rel_ss: float, tau_Rel: float) -> float:
    """
    SR release derivative.

    From MATLAB:
    dRel = -(Rel_ss + Rel)./tau_Rel

    Note the sign: Rel is negative (Ca2+ leaving JSR).

    Args:
        Rel: Current release flux (mM/ms)
        Rel_ss: Steady-state release (mM/ms)
        tau_Rel: Time constant (ms)

    Returns:
        dRel/dt (mM/ms^2)
    """
    return -(Rel_ss + Rel) / tau_Rel


# =============================================================================
# Analytical Calcium Buffering
# =============================================================================

@njit(cache=True)
def solve_Ca_i(ca_T: float, CMDN_max: float, TRPN_max: float,
               Km_CMDN: float, Km_TRPN: float) -> float:
    """
    Solve for free cytosolic Ca2+ from total Ca2+ using cubic equation.

    The total Ca includes free Ca plus Ca bound to calmodulin and troponin:
    ca_T = Ca_i + CMDN_max * Ca_i/(Km_CMDN + Ca_i) + TRPN_max * Ca_i/(Km_TRPN + Ca_i)

    This is rearranged into a cubic equation: a*Ca^3 + b*Ca^2 + c*Ca + d = 0

    From MATLAB conc_cai():
    bmyo = cmdnbar+trpnbar-ca_t+kmtrpn+kmcmdn
    cmyo = kmcmdn*kmtrpn -ca_t*(kmtrpn+kmcmdn)+trpnbar*kmcmdn+cmdnbar*kmtrpn
    dmyo = -kmtrpn*kmcmdn*ca_t
    cai = (2*(bmyo^2-3*cmyo)^(1/2)/3)*cos(acos((9*bmyo*cmyo-2*bmyo^3-27*dmyo)/
          (2*(bmyo^2-3*cmyo)^1.5))/3)-(bmyo/3)

    Args:
        ca_T: Total cytosolic Ca2+ (mM)
        CMDN_max: Total calmodulin (mM)
        TRPN_max: Total troponin (mM)
        Km_CMDN: Calmodulin affinity (mM)
        Km_TRPN: Troponin affinity (mM)

    Returns:
        Free cytosolic Ca2+ (mM)
    """
    # Coefficients for depressed cubic
    # After substitution x = Ca + b/3, the cubic becomes: x^3 + px + q = 0
    # where p = c/a - b^2/(3a^2), q = 2b^3/(27a^3) - bc/(3a^2) + d/a
    # For this specific form with a=1, the coefficients are:
    b = CMDN_max + TRPN_max - ca_T + Km_TRPN + Km_CMDN
    c = Km_CMDN * Km_TRPN - ca_T * (Km_TRPN + Km_CMDN) + \
        TRPN_max * Km_CMDN + CMDN_max * Km_TRPN
    d = -Km_TRPN * Km_CMDN * ca_T

    # Use trigonometric solution for depressed cubic (Cardano)
    # p = c - b^2/3, q = 2b^3/27 - bc/3 + d
    p = c - b * b / 3.0
    q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d

    # Discriminant-like term
    # For 3 real roots: q^2/4 + p^3/27 < 0
    disc = b * b - 3.0 * c

    if disc < 1e-20:
        # Degenerate case - use simple approximation
        if ca_T < 1e-12:
            return 1e-12
        return ca_T / (1.0 + CMDN_max / Km_CMDN + TRPN_max / Km_TRPN)

    sqrt_disc = np.sqrt(disc)

    # The MATLAB formula uses the trigonometric form directly
    # Ca_i = (2/3)*sqrt(b^2-3c)*cos(theta/3) - b/3
    # where cos(3*theta) = (9bc - 2b^3 - 27d) / (2*(b^2-3c)^1.5)

    arg = (9.0 * b * c - 2.0 * b * b * b - 27.0 * d) / (2.0 * disc * sqrt_disc)

    # Clamp to valid range for acos
    if arg > 1.0:
        arg = 1.0
    elif arg < -1.0:
        arg = -1.0

    theta = np.arccos(arg)
    Ca_i = (2.0 * sqrt_disc / 3.0) * np.cos(theta / 3.0) - b / 3.0

    # Ensure positive result
    if Ca_i < 1e-12:
        Ca_i = 1e-12

    return Ca_i


@njit(cache=True)
def solve_Ca_JSR(jsr_T: float, CSQN_max: float, Km_CSQN: float) -> float:
    """
    Solve for free JSR Ca2+ from total JSR Ca2+ using quadratic equation.

    The total JSR Ca includes free Ca plus Ca bound to calsequestrin:
    jsr_T = Ca_JSR + CSQN_max * Ca_JSR/(Km_CSQN + Ca_JSR)

    This is rearranged into: Ca^2 + b*Ca - c = 0
    where b = CSQN_max + Km_CSQN - jsr_T
          c = jsr_T * Km_CSQN

    From MATLAB conc_jsr():
    b = csqnbar + kmcsqn - ca_t
    c = ca_t * kmcsqn
    cajsr = -b/2 + sqrt(b^2 + 4*c)/2

    Args:
        jsr_T: Total JSR Ca2+ (mM)
        CSQN_max: Total calsequestrin (mM)
        Km_CSQN: Calsequestrin affinity (mM)

    Returns:
        Free JSR Ca2+ (mM)
    """
    b = CSQN_max + Km_CSQN - jsr_T
    c = jsr_T * Km_CSQN

    disc = b * b + 4.0 * c
    if disc < 0:
        disc = 0

    Ca_JSR = -b / 2.0 + np.sqrt(disc) / 2.0

    # Ensure positive result
    if Ca_JSR < 1e-12:
        Ca_JSR = 1e-12

    return Ca_JSR


# =============================================================================
# Concentration Derivatives
# =============================================================================

@njit(cache=True)
def dCa_T_dt(I_Ca_ion: float, I_leak: float, I_up: float, Rel: float, Over: float,
             AF: float, vmyo: float, vnsr: float, vjsr: float) -> float:
    """
    Total cytosolic Ca2+ derivative.

    From MATLAB:
    dcai = -caiont*data.AF/(data.vmyo*2) +
           (ileak-iup)*data.vnsr/data.vmyo +
           (Over+Rel)*data.vjsr/data.vmyo

    Args:
        I_Ca_ion: Total Ca2+ current (uA/uF)
        I_leak: NSR leak flux (mM/ms)
        I_up: SERCA uptake flux (mM/ms)
        Rel: SR release flux (mM/ms)
        Over: Overload flux (mM/ms), typically 0
        AF: Acap/F conversion factor
        vmyo: Myoplasm volume (uL)
        vnsr: NSR volume (uL)
        vjsr: JSR volume (uL)

    Returns:
        dca_T/dt (mM/ms)
    """
    # Current contribution (factor of 2 for Ca2+)
    current_term = -I_Ca_ion * AF / (vmyo * 2.0)

    # SR leak and uptake
    sr_exchange = (I_leak - I_up) * vnsr / vmyo

    # SR release
    release_term = (Over + Rel) * vjsr / vmyo

    return current_term + sr_exchange + release_term


@njit(cache=True)
def dNSR_dt(I_up: float, I_tr: float, I_leak: float,
            vjsr: float, vnsr: float) -> float:
    """
    NSR Ca2+ derivative.

    From MATLAB:
    dnsr = iup - itr*data.vjsr./data.vnsr - ileak

    Args:
        I_up: SERCA uptake flux (mM/ms)
        I_tr: NSR→JSR transfer flux (mM/ms)
        I_leak: NSR leak flux (mM/ms)
        vjsr: JSR volume (uL)
        vnsr: NSR volume (uL)

    Returns:
        dnsr/dt (mM/ms)
    """
    return I_up - I_tr * vjsr / vnsr - I_leak


@njit(cache=True)
def dJSR_T_dt(I_tr: float, Rel: float) -> float:
    """
    Total JSR Ca2+ derivative.

    From MATLAB:
    djsr = itr - (Rel)

    Note: Rel is the release flux, negative for Ca leaving JSR.

    Args:
        I_tr: NSR→JSR transfer flux (mM/ms)
        Rel: SR release flux (mM/ms)

    Returns:
        djsr_T/dt (mM/ms)
    """
    return I_tr - Rel


@njit(cache=True)
def dNa_i_dt(I_Na_ion: float, AF: float, vmyo: float) -> float:
    """
    Intracellular Na+ derivative.

    From MATLAB:
    dnai = -naiont*data.AF/(data.vmyo)

    Args:
        I_Na_ion: Total Na+ current (uA/uF)
        AF: Acap/F conversion factor
        vmyo: Myoplasm volume (uL)

    Returns:
        dNa_i/dt (mM/ms)
    """
    return -I_Na_ion * AF / vmyo


@njit(cache=True)
def dK_i_dt(I_K_ion: float, AF: float, vmyo: float) -> float:
    """
    Intracellular K+ derivative.

    From MATLAB:
    dki = -kiont*data.AF/(data.vmyo)

    Args:
        I_K_ion: Total K+ current (uA/uF)
        AF: Acap/F conversion factor
        vmyo: Myoplasm volume (uL)

    Returns:
        dK_i/dt (mM/ms)
    """
    return -I_K_ion * AF / vmyo
