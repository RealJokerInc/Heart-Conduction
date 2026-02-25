"""
LRd07 Ionic Currents

All current equations are translated directly from cell_web2007.m.
Currents are in units of uA/uF (microamperes per microfarad).

Reference: Livshitz LM, Rudy Y. Am J Physiol Heart Circ Physiol. 2007;292(6):H2854-66.

Currents implemented:
1. INa   - Fast sodium current
2. INab  - Background sodium current
3. ICaL  - L-type calcium current (GHK formulation)
4. ICaT  - T-type calcium current
5. ICab  - Background calcium current
6. IKr   - Rapid delayed rectifier K+ current
7. IKs   - Slow delayed rectifier K+ current (Ca-dependent)
8. IK1   - Inward rectifier K+ current
9. IKp   - Plateau K+ current
10. INaCa - Na/Ca exchanger current
11. INaK  - Na/K pump current
12. IpCa  - Sarcolemmal Ca pump current
"""

import numpy as np
from numba import njit
from .gating import f_Ca, r_Kr


# =============================================================================
# INa - Fast Sodium Current
# =============================================================================

@njit(cache=True)
def I_Na(V: float, m: float, H: float, J: float,
         Na_i: float, Na_o: float,
         G_Na: float, frt: float) -> float:
    """
    Fast sodium current (Ohmic).

    From MATLAB:
    ENa = log(data.na_o./Na_i)/data.frt
    gNa = data.GNa*m*m*m*H*J
    In = gNa.*(V-ENa)

    Args:
        V: Membrane potential (mV)
        m: Activation gate
        H: Fast inactivation gate
        J: Slow inactivation gate
        Na_i: Intracellular Na+ (mM)
        Na_o: Extracellular Na+ (mM)
        G_Na: Max conductance (mS/cm^2)
        frt: F/(R*T)

    Returns:
        INa current (uA/uF)
    """
    E_Na = np.log(Na_o / Na_i) / frt
    g_Na = G_Na * m * m * m * H * J
    return g_Na * (V - E_Na)


@njit(cache=True)
def I_Nab(V: float, Na_i: float, Na_o: float,
          G_Nab: float, frt: float) -> float:
    """
    Background sodium current (Ohmic leak).

    From MATLAB:
    inab = data.GNab*(V-ENa)

    Args:
        V: Membrane potential (mV)
        Na_i: Intracellular Na+ (mM)
        Na_o: Extracellular Na+ (mM)
        G_Nab: Conductance (mS/cm^2)
        frt: F/(R*T)

    Returns:
        INab current (uA/uF)
    """
    E_Na = np.log(Na_o / Na_i) / frt
    return G_Nab * (V - E_Na)


# =============================================================================
# ICaL - L-type Calcium Current (GHK formulation)
# =============================================================================

@njit(cache=True)
def I_CaL(V: float, d: float, f: float, Ca_i: float,
          Na_i: float, K_i: float,
          Na_o: float, K_o: float, Ca_o: float,
          P_Ca: float, P_Na: float, P_K: float,
          gamma_Cai: float, gamma_Cao: float,
          gamma_Nai: float, gamma_Nao: float,
          gamma_Ki: float, gamma_Ko: float,
          Km_Ca: float, F: float, frt: float) -> tuple:
    """
    L-type calcium current with GHK formulation.

    From MATLAB:
    ibarca = pca*4*v*F*frt*((gacai*cai*exp(2*v*frt)-gacao*ca_o)/(exp(2*v*frt)-1))
    ibarna = pna*(v*F*frt)*((ganai*nai*exp(v*frt)-ganao*na_o)/(exp(v*frt)-1))
    ibark = pk*(v*F*frt)*((gaki*ki*exp(v*frt)-gako*k_o)/(exp(v*frt)-1))
    fca = 1./(1+(cai./data.kmca))
    ilca = d.*f.*fca.*ibarca
    ilcana = d.*f*fca*ibarna
    ilcak = d.*f*fca*ibark

    Args:
        V: Membrane potential (mV)
        d: Activation gate
        f: Voltage inactivation gate
        Ca_i: Free intracellular Ca2+ (mM)
        Na_i, K_i: Intracellular ion concentrations (mM)
        Na_o, K_o, Ca_o: Extracellular concentrations (mM)
        P_Ca, P_Na, P_K: Permeabilities (cm/s)
        gamma_*: Activity coefficients
        Km_Ca: Half-sat for fCa (mM)
        F: Faraday constant
        frt: F/(R*T)

    Returns:
        Tuple of (I_CaL_Ca, I_CaL_Na, I_CaL_K) currents (uA/uF)
    """
    # fCa - Ca-dependent inactivation
    fCa = f_Ca(Ca_i, Km_Ca)

    # GHK driving force for Ca (z=2)
    vfrt = V * frt
    vfrt2 = 2.0 * vfrt

    # Handle V ≈ 0 singularity
    if abs(V) < 1e-6:
        # Limit as V→0: GHK reduces to linear form
        I_bar_Ca = P_Ca * 4.0 * F * frt * (gamma_Cai * Ca_i - gamma_Cao * Ca_o)
        I_bar_Na = P_Na * F * frt * (gamma_Nai * Na_i - gamma_Nao * Na_o)
        I_bar_K = P_K * F * frt * (gamma_Ki * K_i - gamma_Ko * K_o)
    else:
        # Standard GHK current equation
        exp_vfrt = np.exp(vfrt)
        exp_vfrt2 = np.exp(vfrt2)

        I_bar_Ca = P_Ca * 4.0 * V * F * frt * (
            (gamma_Cai * Ca_i * exp_vfrt2 - gamma_Cao * Ca_o) /
            (exp_vfrt2 - 1.0)
        )

        I_bar_Na = P_Na * V * F * frt * (
            (gamma_Nai * Na_i * exp_vfrt - gamma_Nao * Na_o) /
            (exp_vfrt - 1.0)
        )

        I_bar_K = P_K * V * F * frt * (
            (gamma_Ki * K_i * exp_vfrt - gamma_Ko * K_o) /
            (exp_vfrt - 1.0)
        )

    # Apply gating
    gate = d * f * fCa
    I_CaL_Ca = gate * I_bar_Ca
    I_CaL_Na = gate * I_bar_Na
    I_CaL_K = gate * I_bar_K

    return I_CaL_Ca, I_CaL_Na, I_CaL_K


@njit(cache=True)
def I_CaL_total(V: float, d: float, f: float, Ca_i: float,
                Na_i: float, K_i: float,
                Na_o: float, K_o: float, Ca_o: float,
                P_Ca: float, P_Na: float, P_K: float,
                gamma_Cai: float, gamma_Cao: float,
                gamma_Nai: float, gamma_Nao: float,
                gamma_Ki: float, gamma_Ko: float,
                Km_Ca: float, F: float, frt: float) -> float:
    """
    Total L-type calcium current (Ca + Na + K components).

    Returns:
        Total ICaL current (uA/uF)
    """
    I_Ca, I_Na, I_K = I_CaL(V, d, f, Ca_i, Na_i, K_i,
                            Na_o, K_o, Ca_o,
                            P_Ca, P_Na, P_K,
                            gamma_Cai, gamma_Cao,
                            gamma_Nai, gamma_Nao,
                            gamma_Ki, gamma_Ko,
                            Km_Ca, F, frt)
    return I_Ca + I_Na + I_K


# =============================================================================
# ICaT - T-type Calcium Current
# =============================================================================

@njit(cache=True)
def I_CaT(V: float, B: float, G: float,
          Ca_i: float, Ca_o: float,
          g_CaT: float, frt: float) -> float:
    """
    T-type calcium current (Ohmic).

    From MATLAB:
    ECa = log(data.ca_o/cai)/2/data.frt
    icat = data.gcat*b*b*g*(V-ECa)

    Args:
        V: Membrane potential (mV)
        B: Activation gate
        G: Inactivation gate
        Ca_i: Free intracellular Ca2+ (mM)
        Ca_o: Extracellular Ca2+ (mM)
        g_CaT: Max conductance (mS/cm^2)
        frt: F/(R*T)

    Returns:
        ICaT current (uA/uF)
    """
    E_Ca = np.log(Ca_o / Ca_i) / (2.0 * frt)
    return g_CaT * B * B * G * (V - E_Ca)


# =============================================================================
# ICab - Background Calcium Current
# =============================================================================

@njit(cache=True)
def I_Cab(V: float, Ca_i: float, Ca_o: float,
          g_Cab: float, frt: float) -> float:
    """
    Background calcium current (Ohmic leak).

    From MATLAB:
    icab = data.gcab*(v - log(data.ca_o/ca_i)/2/data.frt)

    Args:
        V: Membrane potential (mV)
        Ca_i: Free intracellular Ca2+ (mM)
        Ca_o: Extracellular Ca2+ (mM)
        g_Cab: Conductance (mS/cm^2)
        frt: F/(R*T)

    Returns:
        ICab current (uA/uF)
    """
    E_Ca = np.log(Ca_o / Ca_i) / (2.0 * frt)
    return g_Cab * (V - E_Ca)


# =============================================================================
# IKr - Rapid Delayed Rectifier K+ Current
# =============================================================================

@njit(cache=True)
def I_Kr(V: float, xr: float, K_i: float, K_o: float,
         g_Kr_max: float, frt: float) -> float:
    """
    Rapid delayed rectifier K+ current (Ohmic with rectification).

    From MATLAB:
    gkr = data.gkrmax*(data.k_o/5.4).^(1/2)
    ekr = log(data.k_o/ki)/data.frt
    r = 1/(1+exp((v+9)/22.4))
    ikr = gkr*xr*r*(v-ekr)

    Args:
        V: Membrane potential (mV)
        xr: Activation gate
        K_i: Intracellular K+ (mM)
        K_o: Extracellular K+ (mM)
        g_Kr_max: Max conductance (mS/cm^2)
        frt: F/(R*T)

    Returns:
        IKr current (uA/uF)
    """
    g_Kr = g_Kr_max * np.sqrt(K_o / 5.4)
    E_K = np.log(K_o / K_i) / frt
    r = r_Kr(V)
    return g_Kr * xr * r * (V - E_K)


# =============================================================================
# IKs - Slow Delayed Rectifier K+ Current (Ca-dependent)
# =============================================================================

@njit(cache=True)
def I_Ks(V: float, xs: float, xs2: float,
         Ca_i: float, Na_i: float, K_i: float,
         Na_o: float, K_o: float,
         G_Ks_max: float, prnak: float,
         IKs_Ca_max: float, IKs_Ca_Kd: float,
         frt: float) -> float:
    """
    Slow delayed rectifier K+ current (Ohmic, Ca-dependent conductance).

    From MATLAB:
    gks = data.GKsmax*(1+0.6/(1+(3.8e-5/cai)^1.4))
    eks = log((data.k_o+data.prnak*data.na_o)/(ki+data.prnak*nai))/data.frt
    iks = gks*xs1*xs2*(v-eks)

    The Ca-dependence increases IKs at higher Ca_i, contributing to
    APD shortening at fast rates.

    Args:
        V: Membrane potential (mV)
        xs, xs2: Activation gates (two gates for slower kinetics)
        Ca_i: Free intracellular Ca2+ (mM)
        Na_i, K_i: Intracellular concentrations (mM)
        Na_o, K_o: Extracellular concentrations (mM)
        G_Ks_max: Max conductance (mS/cm^2)
        prnak: Na/K permeability ratio
        IKs_Ca_max: Max Ca enhancement factor (0.6)
        IKs_Ca_Kd: Half-sat for Ca enhancement (38e-6 mM = 38 nM)
        frt: F/(R*T)

    Returns:
        IKs current (uA/uF)
    """
    # Ca-dependent conductance
    # gks = GKsmax * (1 + 0.6/(1 + (38e-6/Ca_i)^1.4))
    Ca_factor = 1.0 + IKs_Ca_max / (1.0 + (IKs_Ca_Kd / Ca_i)**1.4)
    g_Ks = G_Ks_max * Ca_factor

    # Reversal potential (includes Na permeability)
    E_Ks = np.log((K_o + prnak * Na_o) / (K_i + prnak * Na_i)) / frt

    return g_Ks * xs * xs2 * (V - E_Ks)


# =============================================================================
# IK1 - Inward Rectifier K+ Current
# =============================================================================

@njit(cache=True)
def I_K1(V: float, K_i: float, K_o: float,
         G_K1_max: float, frt: float) -> float:
    """
    Inward rectifier K+ current (instantaneous, strong rectification).

    From MATLAB:
    GK1_ = data.GK1max*sqrt(data.k_o/5.4)
    EK = log(data.k_o/K_i)/data.frt
    ak1 = 1.02/(1+exp(0.2385*(V-EK-59.215)))
    bk1 = (0.49124*exp(0.08032*(V-EK+5.476))+exp(0.06175*(V-EK-594.31)))/
          (1+exp(-0.5143*(V-EK+4.753)))
    gK1 = GK1_*ak1/(ak1+bk1)
    IK1 = gK1*(V-EK)

    Args:
        V: Membrane potential (mV)
        K_i: Intracellular K+ (mM)
        K_o: Extracellular K+ (mM)
        G_K1_max: Max conductance (mS/cm^2)
        frt: F/(R*T)

    Returns:
        IK1 current (uA/uF)
    """
    G_K1_ = G_K1_max * np.sqrt(K_o / 5.4)
    E_K = np.log(K_o / K_i) / frt

    # Voltage relative to E_K
    vek = V - E_K

    # Rectification (time-independent)
    ak1 = 1.02 / (1.0 + np.exp(0.2385 * (vek - 59.215)))
    bk1 = (0.49124 * np.exp(0.08032 * (vek + 5.476)) +
           np.exp(0.06175 * (vek - 594.31))) / \
          (1.0 + np.exp(-0.5143 * (vek + 4.753)))

    g_K1 = G_K1_ * ak1 / (ak1 + bk1)
    return g_K1 * (V - E_K)


# =============================================================================
# IKp - Plateau K+ Current
# =============================================================================

@njit(cache=True)
def I_Kp(V: float, K_i: float, K_o: float,
         G_Kp_max: float, frt: float) -> float:
    """
    Plateau K+ current (Ohmic, voltage-dependent activation).

    From MATLAB:
    EK = log(data.k_o/K_i)/data.frt
    ikp = data.GKpmax*(V-EK)./(1+exp((7.488-V)./5.98))

    Args:
        V: Membrane potential (mV)
        K_i: Intracellular K+ (mM)
        K_o: Extracellular K+ (mM)
        G_Kp_max: Max conductance (mS/cm^2)
        frt: F/(R*T)

    Returns:
        IKp current (uA/uF)
    """
    E_K = np.log(K_o / K_i) / frt
    Kp = 1.0 / (1.0 + np.exp((7.488 - V) / 5.98))
    return G_Kp_max * Kp * (V - E_K)


# =============================================================================
# INaCa - Na/Ca Exchanger Current
# =============================================================================

@njit(cache=True)
def I_NaCa(V: float, Ca_i: float, Na_i: float,
           Na_o: float, Ca_o: float,
           c1: float, c2: float, gamma: float,
           frt: float) -> float:
    """
    Na/Ca exchanger current (electrogenic, 3Na:1Ca).

    From MATLAB:
    inaca = c1*exp((gammas-1)*v*frt)*((exp(v*frt)*nai^3*ca_o - na_o^3*cai)/
            (1 + c2*exp((gammas-1)*v*frt)*(exp(v*frt)*nai^3*ca_o + na_o^3*cai)))

    Positive current = Ca2+ extrusion (Na+ entry)

    Args:
        V: Membrane potential (mV)
        Ca_i: Free intracellular Ca2+ (mM)
        Na_i: Intracellular Na+ (mM)
        Na_o: Extracellular Na+ (mM)
        Ca_o: Extracellular Ca2+ (mM)
        c1: Scaling factor (0.00025)
        c2: Saturation factor (0.0001)
        gamma: Voltage dependence position (0.15)
        frt: F/(R*T)

    Returns:
        INaCa current (uA/uF)
    """
    vfrt = V * frt
    exp_vfrt = np.exp(vfrt)
    exp_gamma = np.exp((gamma - 1.0) * vfrt)

    Na_i3 = Na_i * Na_i * Na_i
    Na_o3 = Na_o * Na_o * Na_o

    numerator = exp_vfrt * Na_i3 * Ca_o - Na_o3 * Ca_i
    denominator = 1.0 + c2 * exp_gamma * (exp_vfrt * Na_i3 * Ca_o + Na_o3 * Ca_i)

    return c1 * exp_gamma * numerator / denominator


# =============================================================================
# INaK - Na/K Pump Current
# =============================================================================

@njit(cache=True)
def I_NaK(V: float, Na_i: float, Na_o: float, K_o: float,
          I_NaK_max: float, Km_Nai: float, Km_Ko: float,
          frt: float) -> float:
    """
    Na/K pump current (electrogenic, 3Na out: 2K in).

    From MATLAB:
    sigma = (exp(data.na_o/67.3)-1)/7
    fnak = 1/(1+0.1245*exp(-0.1*v*frt) + 0.0365*sigma*exp(-v*frt))
    inak = data.ibarnak*fnak/(1+(data.kmnai/nai)^2)/(1+data.kmko/data.k_o)

    Args:
        V: Membrane potential (mV)
        Na_i: Intracellular Na+ (mM)
        Na_o: Extracellular Na+ (mM)
        K_o: Extracellular K+ (mM)
        I_NaK_max: Max pump current (uA/uF)
        Km_Nai: Half-sat for Na (mM)
        Km_Ko: Half-sat for K (mM)
        frt: F/(R*T)

    Returns:
        INaK current (uA/uF)
    """
    # Voltage dependence
    sigma = (np.exp(Na_o / 67.3) - 1.0) / 7.0
    vfrt = V * frt
    f_NaK = 1.0 / (1.0 + 0.1245 * np.exp(-0.1 * vfrt) +
                   0.0365 * sigma * np.exp(-vfrt))

    # Na and K dependence
    f_Na = 1.0 / (1.0 + (Km_Nai / Na_i)**2)
    f_K = 1.0 / (1.0 + Km_Ko / K_o)

    return I_NaK_max * f_NaK * f_Na * f_K


# =============================================================================
# IpCa - Sarcolemmal Ca Pump Current
# =============================================================================

@njit(cache=True)
def I_pCa(Ca_i: float, I_pCa_max: float, Km_pCa: float) -> float:
    """
    Sarcolemmal Ca pump current (Ca2+ extrusion).

    From MATLAB:
    ipca = (data.ibarpca*ca_i)/(data.kmpca+ca_i)

    Args:
        Ca_i: Free intracellular Ca2+ (mM)
        I_pCa_max: Max pump current (uA/uF)
        Km_pCa: Half-saturation (mM)

    Returns:
        IpCa current (uA/uF)
    """
    return I_pCa_max * Ca_i / (Km_pCa + Ca_i)


# =============================================================================
# Current Sums by Ion Species
# =============================================================================

@njit(cache=True)
def compute_I_ion_totals(I_Na: float, I_Nab: float,
                          I_CaL_Ca: float, I_CaL_Na: float, I_CaL_K: float,
                          I_CaT: float, I_Cab: float, I_pCa: float,
                          I_NaCa: float, I_NaK: float,
                          I_Kr: float, I_Ks: float, I_K1: float, I_Kp: float,
                          I_stim: float) -> tuple:
    """
    Compute total currents by ion species for concentration updates.

    From MATLAB:
    caiont = ilca + icab + ipca - 2*inaca + icat
    naiont = ina + inab + 3*inaca + ilcana + 3*inak
    kiont = ikr + iks + IK1 + ikp + ilcak - 2*inak - In

    Returns:
        Tuple of (I_Ca_total, I_Na_total, I_K_total)
    """
    # Ca current: inward positive for ICaL, outward positive for IpCa
    # INaCa: positive = Ca extrusion, so -2*INaCa adds to Ca influx
    I_Ca_total = I_CaL_Ca + I_Cab + I_pCa - 2.0 * I_NaCa + I_CaT

    # Na current: INa, INab, INaCa (3Na in per Ca out), ICaL Na component, INaK (3Na out)
    I_Na_total = I_Na + I_Nab + 3.0 * I_NaCa + I_CaL_Na + 3.0 * I_NaK

    # K current: IKr, IKs, IK1, IKp, ICaL K component, INaK (2K in)
    I_K_total = I_Kr + I_Ks + I_K1 + I_Kp + I_CaL_K - 2.0 * I_NaK - I_stim

    return I_Ca_total, I_Na_total, I_K_total
