"""
Luo-Rudy 1994 (LRd) Ionic Currents
==================================

All 10 ionic currents for the LRd model:

Voltage-Gated Channels:
- I_Na:  Fast sodium current (Ohmic)
- I_CaL: L-type calcium current (GHK)
- I_K:   Time-dependent potassium current (Ohmic)
- I_K1:  Inward rectifier potassium current (Ohmic)
- I_Kp:  Plateau potassium current (Ohmic)

Pumps & Exchangers:
- I_NaCa: Na+/Ca2+ exchanger
- I_NaK:  Na+/K+ ATPase pump
- I_pCa:  Sarcolemmal Ca2+ pump

Background Leak:
- I_Na_b: Background sodium current
- I_Ca_b: Background calcium current

Key Feature: ICaL uses Goldman-Hodgkin-Katz (GHK) equation for accurate
calcium flux at large concentration ratios.

Reference:
    Luo CH, Rudy Y. Circ Res. 1994;74(6):1071-1096.

Author: Generated with Claude Code
Date: 2024-12
"""

from __future__ import annotations
import numpy as np
import numba
from typing import Tuple, NamedTuple

from gating_lrd import (
    m_inf, h_inf, j_inf,
    d_inf, f_inf, f_Ca_inf,
    X_inf, Xi, K1_inf, Kp,
    safe_exp
)


# =============================================================================
# Physical Constants (duplicated for Numba compatibility)
# =============================================================================

# These are duplicated here to allow Numba to compile without importing
R = 8314.0      # Gas constant [mJ/(mol·K)]
T = 310.0       # Temperature [K]
F = 96485.0     # Faraday constant [C/mol]
RTF = R * T / F  # ~26.7 mV
FRT = F / (R * T)  # ~0.0374 1/mV


# =============================================================================
# Reversal Potentials (Nernst Equation)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def E_Na(Na_i: float, Na_o: float = 140.0) -> float:
    """
    Sodium reversal potential (Nernst).

    E_Na = (RT/F) * ln(Na_o / Na_i)

    Parameters
    ----------
    Na_i : float
        Intracellular Na+ [mM]
    Na_o : float
        Extracellular Na+ [mM]

    Returns
    -------
    E_Na : float
        Reversal potential [mV]
    """
    return RTF * np.log(Na_o / Na_i)


@numba.jit(nopython=True, cache=True)
def E_K(K_i: float, K_o: float = 5.4) -> float:
    """
    Potassium reversal potential (Nernst).

    E_K = (RT/F) * ln(K_o / K_i)
    """
    return RTF * np.log(K_o / K_i)


@numba.jit(nopython=True, cache=True)
def E_Ca(Ca_i: float, Ca_o: float = 1.8) -> float:
    """
    Calcium reversal potential (Nernst).

    E_Ca = (RT/2F) * ln(Ca_o / Ca_i)

    Note: Factor of 2 for divalent ion.
    """
    return 0.5 * RTF * np.log(Ca_o / Ca_i)


@numba.jit(nopython=True, cache=True)
def E_Ks(K_i: float, Na_i: float, K_o: float = 5.4, Na_o: float = 140.0,
         PR_NaK: float = 0.01833) -> float:
    """
    Reversal potential for IK with Na+ permeability.

    E_Ks = (RT/F) * ln((K_o + PR_NaK * Na_o) / (K_i + PR_NaK * Na_i))
    """
    return RTF * np.log((K_o + PR_NaK * Na_o) / (K_i + PR_NaK * Na_i))


# =============================================================================
# Goldman-Hodgkin-Katz (GHK) Flux Equation
# =============================================================================

@numba.jit(nopython=True, cache=True)
def ghk_flux(V: float, z: float, C_i: float, C_o: float,
             gamma_i: float = 1.0, gamma_o: float = 1.0) -> float:
    """
    Goldman-Hodgkin-Katz flux equation.

    Φ = z² * F² * V / (RT) * (γ_i * C_i * exp(zFV/RT) - γ_o * C_o) / (exp(zFV/RT) - 1)

    This gives the driving force for ion flux through a channel,
    accounting for concentration gradients and electric field.

    Parameters
    ----------
    V : float
        Membrane potential [mV]
    z : float
        Ion valence (+1 for Na+/K+, +2 for Ca2+)
    C_i : float
        Intracellular concentration [mM]
    C_o : float
        Extracellular concentration [mM]
    gamma_i : float
        Intracellular activity coefficient
    gamma_o : float
        Extracellular activity coefficient

    Returns
    -------
    flux : float
        GHK flux term [mM * mV] (multiply by permeability to get current)

    Notes
    -----
    Uses L'Hopital's rule near V = 0 to avoid division by zero.
    The result has units that when multiplied by permeability [cm/s]
    and appropriate conversion factors gives current density [µA/cm²].
    """
    zFV_RT = z * F * V / (R * T)  # Dimensionless

    # L'Hopital limit for V → 0
    if abs(zFV_RT) < 1e-6:
        # Φ → z * F * (γ_i * C_i - γ_o * C_o)
        return z * F * (gamma_i * C_i - gamma_o * C_o)

    exp_term = safe_exp(zFV_RT)

    # Full GHK equation
    # Units: z² * (C/mol)² * mV / (mJ/mol) * mM = C²·mV·mM / (mol·mJ)
    #      = C²·mV·mM / (mol·mJ) = ... → need conversion factor
    numerator = gamma_i * C_i * exp_term - gamma_o * C_o
    denominator = exp_term - 1.0

    return (z * z * F * F * V / (R * T)) * (numerator / denominator)


@numba.jit(nopython=True, cache=True)
def ghk_current(P: float, V: float, z: float, C_i: float, C_o: float,
                gamma_i: float = 1.0, gamma_o: float = 1.0) -> float:
    """
    GHK current density.

    I = P * Φ_GHK

    Parameters
    ----------
    P : float
        Permeability [cm/s]
    V : float
        Membrane potential [mV]
    z : float
        Ion valence
    C_i, C_o : float
        Ion concentrations [mM]
    gamma_i, gamma_o : float
        Activity coefficients

    Returns
    -------
    I : float
        Current density [µA/cm²]

    Notes
    -----
    Conversion: P [cm/s] * F [C/mol] * conc [mM] = P * F * conc * 1e-3 [A/cm²]
                = P * F * conc * 1e-3 * 1e6 [µA/cm²] = P * F * conc * 1e3 [µA/cm²]
    But the GHK equation already includes F², so units work out.
    """
    return P * ghk_flux(V, z, C_i, C_o, gamma_i, gamma_o)


# =============================================================================
# Fast Sodium Current (I_Na)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def I_Na(V: float, m: float, h: float, j: float,
         Na_i: float, Na_o: float = 140.0,
         g_Na: float = 16.0) -> float:
    """
    Fast sodium current (Ohmic formulation).

    I_Na = g_Na * m³ * h * j * (V - E_Na)

    Parameters
    ----------
    V : float
        Membrane potential [mV]
    m, h, j : float
        Gating variables
    Na_i : float
        Intracellular Na+ [mM]
    Na_o : float
        Extracellular Na+ [mM]
    g_Na : float
        Maximum conductance [mS/cm²]

    Returns
    -------
    I_Na : float
        Current density [µA/cm²]
    """
    E = E_Na(Na_i, Na_o)
    return g_Na * (m ** 3) * h * j * (V - E)


# =============================================================================
# L-Type Calcium Current (I_CaL) - GHK Formulation
# =============================================================================

@numba.jit(nopython=True, cache=True)
def I_CaL(V: float, d: float, f: float, f_Ca: float,
          Ca_i: float, Na_i: float, K_i: float,
          Ca_o: float = 1.8, Na_o: float = 140.0, K_o: float = 5.4,
          P_Ca: float = 5.4e-4, P_Na: float = 6.75e-7, P_K: float = 1.93e-7,
          gamma_Cai: float = 1.0, gamma_Cao: float = 0.341,
          gamma_Nai: float = 0.75, gamma_Nao: float = 0.75,
          gamma_Ki: float = 0.75, gamma_Ko: float = 0.75) -> float:
    """
    L-type calcium current using GHK equation.

    I_CaL = d * f * f_Ca * (I_CaCa + I_CaNa + I_CaK)

    Where each component uses GHK flux:
    - I_CaCa: Ca2+ flux through L-type channel
    - I_CaNa: Na+ flux through L-type channel
    - I_CaK:  K+ flux through L-type channel

    Parameters
    ----------
    V : float
        Membrane potential [mV]
    d, f, f_Ca : float
        Gating variables (activation, voltage inactivation, Ca inactivation)
    Ca_i, Na_i, K_i : float
        Intracellular concentrations [mM]
    Ca_o, Na_o, K_o : float
        Extracellular concentrations [mM]
    P_Ca, P_Na, P_K : float
        Permeabilities [cm/s]
    gamma_* : float
        Activity coefficients

    Returns
    -------
    I_CaL : float
        Total L-type calcium current [µA/cm²]
    """
    # Calcium component (z = +2)
    I_CaCa = ghk_current(P_Ca, V, 2.0, Ca_i, Ca_o, gamma_Cai, gamma_Cao)

    # Sodium component (z = +1)
    I_CaNa = ghk_current(P_Na, V, 1.0, Na_i, Na_o, gamma_Nai, gamma_Nao)

    # Potassium component (z = +1)
    I_CaK = ghk_current(P_K, V, 1.0, K_i, K_o, gamma_Ki, gamma_Ko)

    # Total with gating
    return d * f * f_Ca * (I_CaCa + I_CaNa + I_CaK)


@numba.jit(nopython=True, cache=True)
def I_CaL_Ca_component(V: float, d: float, f: float, f_Ca: float,
                       Ca_i: float, Ca_o: float = 1.8,
                       P_Ca: float = 5.4e-4,
                       gamma_Cai: float = 1.0, gamma_Cao: float = 0.341) -> float:
    """
    Calcium component of I_CaL only.

    Useful for calculating Ca2+ entry for concentration updates.
    """
    I_CaCa = ghk_current(P_Ca, V, 2.0, Ca_i, Ca_o, gamma_Cai, gamma_Cao)
    return d * f * f_Ca * I_CaCa


# =============================================================================
# Time-Dependent Potassium Current (I_K)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def I_K(V: float, X: float, K_i: float, Na_i: float,
        K_o: float = 5.4, Na_o: float = 140.0,
        g_K_max: float = 0.282, PR_NaK: float = 0.01833) -> float:
    """
    Time-dependent potassium current.

    I_K = g_K * X² * Xi * (V - E_Ks)

    Where:
    - g_K = g_K_max * sqrt(K_o / 5.4)  (K_o dependence)
    - Xi is voltage-dependent rectification factor
    - E_Ks includes Na+ permeability

    Parameters
    ----------
    V : float
        Membrane potential [mV]
    X : float
        Activation gating variable
    K_i, Na_i : float
        Intracellular concentrations [mM]
    K_o, Na_o : float
        Extracellular concentrations [mM]
    g_K_max : float
        Maximum conductance [mS/cm²]
    PR_NaK : float
        Na/K permeability ratio

    Returns
    -------
    I_K : float
        Current density [µA/cm²]
    """
    # Conductance with K_o dependence
    g_K = g_K_max * np.sqrt(K_o / 5.4)

    # Reversal potential with Na+ permeability
    E = E_Ks(K_i, Na_i, K_o, Na_o, PR_NaK)

    # Rectification factor
    xi = Xi(V)

    return g_K * (X ** 2) * xi * (V - E)


# =============================================================================
# Inward Rectifier Potassium Current (I_K1)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def I_K1(V: float, K_i: float, K_o: float = 5.4,
         g_K1_max: float = 0.6047) -> float:
    """
    Inward rectifier potassium current.

    I_K1 = g_K1 * K1_inf * (V - E_K)

    Where:
    - g_K1 = g_K1_max * sqrt(K_o / 5.4)
    - K1_inf is instantaneous voltage-dependent gating

    Parameters
    ----------
    V : float
        Membrane potential [mV]
    K_i : float
        Intracellular K+ [mM]
    K_o : float
        Extracellular K+ [mM]
    g_K1_max : float
        Maximum conductance [mS/cm²]

    Returns
    -------
    I_K1 : float
        Current density [µA/cm²]
    """
    # Conductance with K_o dependence
    g_K1 = g_K1_max * np.sqrt(K_o / 5.4)

    # Reversal potential
    E = E_K(K_i, K_o)

    # Instantaneous gating
    k1_inf = K1_inf(V, E)

    return g_K1 * k1_inf * (V - E)


# =============================================================================
# Plateau Potassium Current (I_Kp)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def I_Kp(V: float, K_i: float, K_o: float = 5.4,
         g_Kp: float = 0.0183) -> float:
    """
    Plateau potassium current.

    I_Kp = g_Kp * Kp * (V - E_K)

    Parameters
    ----------
    V : float
        Membrane potential [mV]
    K_i : float
        Intracellular K+ [mM]
    K_o : float
        Extracellular K+ [mM]
    g_Kp : float
        Conductance [mS/cm²]

    Returns
    -------
    I_Kp : float
        Current density [µA/cm²]
    """
    E = E_K(K_i, K_o)
    kp = Kp(V)
    return g_Kp * kp * (V - E)


# =============================================================================
# Na+/Ca2+ Exchanger (I_NaCa)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def I_NaCa(V: float, Na_i: float, Ca_i: float,
           Na_o: float = 140.0, Ca_o: float = 1.8,
           k_NaCa: float = 2000.0, Km_Na: float = 87.5, Km_Ca: float = 1.38,
           k_sat: float = 0.1, eta: float = 0.35) -> float:
    """
    Na+/Ca2+ exchanger current.

    Exchanges 3 Na+ for 1 Ca2+ (electrogenic).

    I_NaCa = k_NaCa * (Na_i³ * Ca_o * exp(η*V*F/RT) - Na_o³ * Ca_i * exp((η-1)*V*F/RT))
             / ((Km_Na³ + Na_o³) * (Km_Ca + Ca_o) * (1 + k_sat * exp((η-1)*V*F/RT)))

    Parameters
    ----------
    V : float
        Membrane potential [mV]
    Na_i, Ca_i : float
        Intracellular concentrations [mM]
    Na_o, Ca_o : float
        Extracellular concentrations [mM]
    k_NaCa : float
        Scaling factor [µA/cm²]
    Km_Na, Km_Ca : float
        Half-saturation constants [mM]
    k_sat : float
        Saturation factor at negative potentials
    eta : float
        Position of energy barrier (0-1)

    Returns
    -------
    I_NaCa : float
        Current density [µA/cm²]

    Notes
    -----
    Positive I_NaCa = Ca2+ entry (forward mode)
    Negative I_NaCa = Ca2+ exit (reverse mode)
    """
    # Exponential terms
    exp_eta = safe_exp(eta * V * FRT)
    exp_eta_1 = safe_exp((eta - 1.0) * V * FRT)

    # Numerator: forward - reverse flux
    Na_i_cubed = Na_i ** 3
    Na_o_cubed = Na_o ** 3

    numerator = Na_i_cubed * Ca_o * exp_eta - Na_o_cubed * Ca_i * exp_eta_1

    # Denominator: saturation terms
    denominator = ((Km_Na ** 3 + Na_o_cubed) *
                   (Km_Ca + Ca_o) *
                   (1.0 + k_sat * exp_eta_1))

    return k_NaCa * numerator / denominator


# =============================================================================
# Na+/K+ ATPase Pump (I_NaK)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def I_NaK(V: float, Na_i: float, K_o: float = 5.4, Na_o: float = 140.0,
          I_NaK_max: float = 1.5, Km_Nai: float = 10.0, Km_Ko: float = 1.5) -> float:
    """
    Na+/K+ ATPase pump current.

    Pumps 3 Na+ out, 2 K+ in (electrogenic).

    I_NaK = I_NaK_max * f_NaK * (K_o / (K_o + Km_Ko)) * (Na_i / (Na_i + Km_Nai))

    Where f_NaK is a voltage-dependent factor.

    Parameters
    ----------
    V : float
        Membrane potential [mV]
    Na_i : float
        Intracellular Na+ [mM]
    K_o : float
        Extracellular K+ [mM]
    Na_o : float
        Extracellular Na+ [mM]
    I_NaK_max : float
        Maximum pump current [µA/cm²]
    Km_Nai : float
        Na+ half-saturation [mM]
    Km_Ko : float
        K+ half-saturation [mM]

    Returns
    -------
    I_NaK : float
        Current density [µA/cm²]
    """
    # Voltage-dependent factor
    sigma = (np.exp(Na_o / 67.3) - 1.0) / 7.0

    f_NaK = 1.0 / (1.0 + 0.1245 * safe_exp(-0.1 * V * FRT) +
                   0.0365 * sigma * safe_exp(-V * FRT))

    # Concentration-dependent terms
    K_term = K_o / (K_o + Km_Ko)
    Na_term = Na_i / (Na_i + Km_Nai)

    return I_NaK_max * f_NaK * K_term * Na_term


# =============================================================================
# Sarcolemmal Ca2+ Pump (I_pCa)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def I_pCa(Ca_i: float, I_pCa_max: float = 1.15, Km_pCa: float = 0.5e-3) -> float:
    """
    Sarcolemmal Ca2+ pump current (Michaelis-Menten).

    I_pCa = I_pCa_max * Ca_i / (Km_pCa + Ca_i)

    Parameters
    ----------
    Ca_i : float
        Intracellular Ca2+ [mM]
    I_pCa_max : float
        Maximum pump current [µA/cm²]
    Km_pCa : float
        Ca2+ half-saturation [mM]

    Returns
    -------
    I_pCa : float
        Current density [µA/cm²]
    """
    return I_pCa_max * Ca_i / (Km_pCa + Ca_i)


# =============================================================================
# Background Sodium Current (I_Na_b)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def I_Na_b(V: float, Na_i: float, Na_o: float = 140.0,
           g_Na_b: float = 0.001) -> float:
    """
    Background sodium leak current.

    I_Na_b = g_Na_b * (V - E_Na)

    Parameters
    ----------
    V : float
        Membrane potential [mV]
    Na_i : float
        Intracellular Na+ [mM]
    Na_o : float
        Extracellular Na+ [mM]
    g_Na_b : float
        Conductance [mS/cm²]

    Returns
    -------
    I_Na_b : float
        Current density [µA/cm²]
    """
    E = E_Na(Na_i, Na_o)
    return g_Na_b * (V - E)


# =============================================================================
# Background Calcium Current (I_Ca_b)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def I_Ca_b(V: float, Ca_i: float, Ca_o: float = 1.8,
           g_Ca_b: float = 0.003) -> float:
    """
    Background calcium leak current.

    I_Ca_b = g_Ca_b * (V - E_Ca)

    Parameters
    ----------
    V : float
        Membrane potential [mV]
    Ca_i : float
        Intracellular Ca2+ [mM]
    Ca_o : float
        Extracellular Ca2+ [mM]
    g_Ca_b : float
        Conductance [mS/cm²]

    Returns
    -------
    I_Ca_b : float
        Current density [µA/cm²]
    """
    E = E_Ca(Ca_i, Ca_o)
    return g_Ca_b * (V - E)


# =============================================================================
# Total Ionic Current
# =============================================================================

@numba.jit(nopython=True, cache=True)
def I_ion_total(V: float, m: float, h: float, j: float,
                d: float, f: float, f_Ca: float, X: float,
                Na_i: float, K_i: float, Ca_i: float,
                Na_o: float = 140.0, K_o: float = 5.4, Ca_o: float = 1.8) -> float:
    """
    Total ionic current (sum of all 10 currents).

    Parameters
    ----------
    V : float
        Membrane potential [mV]
    m, h, j, d, f, f_Ca, X : float
        Gating variables
    Na_i, K_i, Ca_i : float
        Intracellular concentrations [mM]
    Na_o, K_o, Ca_o : float
        Extracellular concentrations [mM]

    Returns
    -------
    I_ion : float
        Total ionic current density [µA/cm²]
    """
    # Voltage-gated channels
    i_na = I_Na(V, m, h, j, Na_i, Na_o)
    i_cal = I_CaL(V, d, f, f_Ca, Ca_i, Na_i, K_i, Ca_o, Na_o, K_o)
    i_k = I_K(V, X, K_i, Na_i, K_o, Na_o)
    i_k1 = I_K1(V, K_i, K_o)
    i_kp = I_Kp(V, K_i, K_o)

    # Pumps and exchangers
    i_naca = I_NaCa(V, Na_i, Ca_i, Na_o, Ca_o)
    i_nak = I_NaK(V, Na_i, K_o, Na_o)
    i_pca = I_pCa(Ca_i)

    # Background
    i_na_b = I_Na_b(V, Na_i, Na_o)
    i_ca_b = I_Ca_b(V, Ca_i, Ca_o)

    return i_na + i_cal + i_k + i_k1 + i_kp + i_naca + i_nak + i_pca + i_na_b + i_ca_b


# =============================================================================
# Current Components for Concentration Updates
# =============================================================================

@numba.jit(nopython=True, cache=True)
def get_Na_currents(V: float, m: float, h: float, j: float,
                    Na_i: float, Ca_i: float,
                    Na_o: float = 140.0, K_o: float = 5.4, Ca_o: float = 1.8) -> float:
    """
    Total Na+ current for [Na+]i update.

    I_Na_total = I_Na + I_Na_b + 3*I_NaK + 3*I_NaCa

    Note: I_NaK and I_NaCa contribute 3 Na+ each per cycle.
    """
    i_na = I_Na(V, m, h, j, Na_i, Na_o)
    i_na_b = I_Na_b(V, Na_i, Na_o)
    i_nak = I_NaK(V, Na_i, K_o, Na_o)
    i_naca = I_NaCa(V, Na_i, Ca_i, Na_o, Ca_o)

    return i_na + i_na_b + 3.0 * i_nak + 3.0 * i_naca


@numba.jit(nopython=True, cache=True)
def get_K_currents(V: float, X: float, K_i: float, Na_i: float,
                   K_o: float = 5.4, Na_o: float = 140.0) -> float:
    """
    Total K+ current for [K+]i update.

    I_K_total = I_K + I_K1 + I_Kp - 2*I_NaK

    Note: I_NaK brings 2 K+ in per cycle (negative contribution).
    """
    i_k = I_K(V, X, K_i, Na_i, K_o, Na_o)
    i_k1 = I_K1(V, K_i, K_o)
    i_kp = I_Kp(V, K_i, K_o)
    i_nak = I_NaK(V, Na_i, K_o, Na_o)

    return i_k + i_k1 + i_kp - 2.0 * i_nak


@numba.jit(nopython=True, cache=True)
def get_Ca_currents(V: float, d: float, f: float, f_Ca: float,
                    Ca_i: float, Na_i: float, K_i: float,
                    Ca_o: float = 1.8, Na_o: float = 140.0, K_o: float = 5.4) -> float:
    """
    Total Ca2+ current for [Ca2+]i update.

    I_Ca_total = I_CaL + I_Ca_b + I_pCa - 2*I_NaCa

    Note: I_NaCa brings 1 Ca2+ in per cycle in forward mode
          (contributes -2*I_NaCa due to 2+ charge).
    """
    i_cal = I_CaL(V, d, f, f_Ca, Ca_i, Na_i, K_i, Ca_o, Na_o, K_o)
    i_ca_b = I_Ca_b(V, Ca_i, Ca_o)
    i_pca = I_pCa(Ca_i)
    i_naca = I_NaCa(V, Na_i, Ca_i, Na_o, Ca_o)

    return i_cal + i_ca_b + i_pca - 2.0 * i_naca


# =============================================================================
# Test Module
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Testing LRd94 Ionic Currents")
    print("=" * 60)

    # Test conditions
    V_rest = -84.624
    V_range = np.linspace(-120, 60, 500)

    # Resting state values
    m_rest, h_rest, j_rest = 0.00136, 0.9814, 0.9905
    d_rest, f_rest, f_Ca_rest = 3e-6, 1.0, 1.0
    X_rest = 0.0057
    Na_i, K_i, Ca_i = 10.0, 145.0, 0.12e-3  # mM

    # External concentrations
    Na_o, K_o, Ca_o = 140.0, 5.4, 1.8

    print("\n--- Reversal Potentials ---")
    print(f"E_Na = {E_Na(Na_i, Na_o):.2f} mV")
    print(f"E_K  = {E_K(K_i, K_o):.2f} mV")
    print(f"E_Ca = {E_Ca(Ca_i, Ca_o):.2f} mV")
    print(f"E_Ks = {E_Ks(K_i, Na_i, K_o, Na_o):.2f} mV")

    print("\n--- Currents at Rest (V = -84.6 mV) ---")
    print(f"I_Na   = {I_Na(V_rest, m_rest, h_rest, j_rest, Na_i, Na_o):.4f} µA/cm²")
    print(f"I_CaL  = {I_CaL(V_rest, d_rest, f_rest, f_Ca_rest, Ca_i, Na_i, K_i):.4f} µA/cm²")
    print(f"I_K    = {I_K(V_rest, X_rest, K_i, Na_i):.4f} µA/cm²")
    print(f"I_K1   = {I_K1(V_rest, K_i):.4f} µA/cm²")
    print(f"I_Kp   = {I_Kp(V_rest, K_i):.4f} µA/cm²")
    print(f"I_NaCa = {I_NaCa(V_rest, Na_i, Ca_i):.4f} µA/cm²")
    print(f"I_NaK  = {I_NaK(V_rest, Na_i):.4f} µA/cm²")
    print(f"I_pCa  = {I_pCa(Ca_i):.4f} µA/cm²")
    print(f"I_Na_b = {I_Na_b(V_rest, Na_i):.4f} µA/cm²")
    print(f"I_Ca_b = {I_Ca_b(V_rest, Ca_i):.4f} µA/cm²")

    i_total = I_ion_total(V_rest, m_rest, h_rest, j_rest,
                          d_rest, f_rest, f_Ca_rest, X_rest,
                          Na_i, K_i, Ca_i)
    print(f"\nI_ion_total = {i_total:.4f} µA/cm² (should be ~0 at rest)")

    # Compute I-V curves
    print("\n--- Computing I-V Curves ---")

    # For I-V curves, use steady-state gating at each voltage
    I_Na_IV = np.zeros_like(V_range)
    I_CaL_IV = np.zeros_like(V_range)
    I_K_IV = np.zeros_like(V_range)
    I_K1_IV = np.zeros_like(V_range)
    I_NaCa_IV = np.zeros_like(V_range)

    for i, V in enumerate(V_range):
        # Use steady-state gating
        m_ss = m_inf(V)
        h_ss = h_inf(V)
        j_ss = j_inf(V)
        d_ss = d_inf(V)
        f_ss = f_inf(V)
        X_ss = X_inf(V)

        I_Na_IV[i] = I_Na(V, m_ss, h_ss, j_ss, Na_i, Na_o)
        I_CaL_IV[i] = I_CaL(V, d_ss, f_ss, f_Ca_rest, Ca_i, Na_i, K_i)
        I_K_IV[i] = I_K(V, X_ss, K_i, Na_i)
        I_K1_IV[i] = I_K1(V, K_i)
        I_NaCa_IV[i] = I_NaCa(V, Na_i, Ca_i)

    # Test GHK vs Ohmic for ICaL
    print("\n--- GHK Test for ICaL ---")
    # At V = 0, the GHK should show non-linear behavior
    print(f"I_CaL at V=0 mV (GHK):  {I_CaL(0, d_inf(0), f_inf(0), 1.0, Ca_i, Na_i, K_i):.4f} µA/cm²")
    print(f"I_CaL at V=-40 mV:      {I_CaL(-40, d_inf(-40), f_inf(-40), 1.0, Ca_i, Na_i, K_i):.4f} µA/cm²")
    print(f"I_CaL at V=+20 mV:      {I_CaL(20, d_inf(20), f_inf(20), 1.0, Ca_i, Na_i, K_i):.4f} µA/cm²")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # I_Na I-V
    axes[0, 0].plot(V_range, I_Na_IV, 'b-', linewidth=2)
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(x=E_Na(Na_i, Na_o), color='r', linestyle='--', alpha=0.5, label=f'E_Na={E_Na(Na_i, Na_o):.0f}mV')
    axes[0, 0].set_xlabel('V [mV]')
    axes[0, 0].set_ylabel('I_Na [µA/cm²]')
    axes[0, 0].set_title('INa I-V Curve (steady-state gating)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(-120, 60)

    # I_CaL I-V (GHK!)
    axes[0, 1].plot(V_range, I_CaL_IV, 'r-', linewidth=2)
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('V [mV]')
    axes[0, 1].set_ylabel('I_CaL [µA/cm²]')
    axes[0, 1].set_title('ICaL I-V Curve (GHK, steady-state gating)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(-120, 60)

    # I_K I-V
    axes[0, 2].plot(V_range, I_K_IV, 'g-', linewidth=2)
    axes[0, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 2].axvline(x=E_Ks(K_i, Na_i), color='r', linestyle='--', alpha=0.5, label=f'E_Ks={E_Ks(K_i, Na_i):.0f}mV')
    axes[0, 2].set_xlabel('V [mV]')
    axes[0, 2].set_ylabel('I_K [µA/cm²]')
    axes[0, 2].set_title('IK I-V Curve (steady-state gating)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim(-120, 60)

    # I_K1 I-V
    axes[1, 0].plot(V_range, I_K1_IV, 'm-', linewidth=2)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=E_K(K_i), color='r', linestyle='--', alpha=0.5, label=f'E_K={E_K(K_i):.0f}mV')
    axes[1, 0].set_xlabel('V [mV]')
    axes[1, 0].set_ylabel('I_K1 [µA/cm²]')
    axes[1, 0].set_title('IK1 I-V Curve (inward rectifier)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(-120, 60)

    # I_NaCa I-V
    axes[1, 1].plot(V_range, I_NaCa_IV, 'c-', linewidth=2)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('V [mV]')
    axes[1, 1].set_ylabel('I_NaCa [µA/cm²]')
    axes[1, 1].set_title('INaCa I-V Curve (Na/Ca exchanger)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(-120, 60)

    # All currents at steady-state
    I_total_IV = np.zeros_like(V_range)
    for i, V in enumerate(V_range):
        m_ss = m_inf(V)
        h_ss = h_inf(V)
        j_ss = j_inf(V)
        d_ss = d_inf(V)
        f_ss = f_inf(V)
        X_ss = X_inf(V)
        I_total_IV[i] = I_ion_total(V, m_ss, h_ss, j_ss, d_ss, f_ss, f_Ca_rest, X_ss,
                                     Na_i, K_i, Ca_i)

    axes[1, 2].plot(V_range, I_total_IV, 'k-', linewidth=2)
    axes[1, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('V [mV]')
    axes[1, 2].set_ylabel('I_ion [µA/cm²]')
    axes[1, 2].set_title('Total Ionic Current (steady-state)')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlim(-120, 60)

    plt.tight_layout()
    plt.savefig('currents_lrd_test.png', dpi=150)
    print(f"\nPlot saved: currents_lrd_test.png")

    plt.show()

    print("\n" + "=" * 60)
    print("Ionic currents test complete!")
