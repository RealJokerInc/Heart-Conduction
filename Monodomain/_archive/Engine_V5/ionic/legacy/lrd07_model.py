"""
LRd07 Model - Main Integration Class

This module provides the main LRd07 cardiac cell model class that integrates
all ionic currents, gating kinetics, and calcium handling.

Reference: Livshitz LM, Rudy Y. Am J Physiol Heart Circ Physiol. 2007;292(6):H2854-66.
"""

import numpy as np
from numba import njit
from typing import Optional, Tuple

from .parameters import LRd07Parameters, StateIndex, DEFAULT_PARAMS
from .gating import (
    alpha_m, beta_m, alpha_h, beta_h, alpha_j, beta_j,
    d_inf, tau_d, f_inf, tau_f,
    xr_inf, tau_xr,
    xs_inf, tau_xs,
    B_inf, tau_B, G_inf, tau_G,
    rush_larsen_step, hh_step, f_Ca
)
from .currents import (
    I_Na, I_Nab, I_CaL, I_CaT, I_Cab,
    I_Kr, I_Ks, I_K1, I_Kp,
    I_NaCa, I_NaK, I_pCa,
    compute_I_ion_totals
)
from .calcium import (
    solve_Ca_i, solve_Ca_JSR,
    I_up, I_leak, I_tr,
    Rel_infinity, tau_Rel_func, dRel_dt,
    dCa_T_dt, dNSR_dt, dJSR_T_dt, dNa_i_dt, dK_i_dt
)


@njit(cache=True)
def compute_derivatives(y: np.ndarray, t: float, I_stim: float,
                        # Physical constants
                        F: float, frt: float,
                        # Extracellular concentrations
                        Na_o: float, K_o: float, Ca_o: float,
                        # Volumes
                        vmyo: float, vnsr: float, vjsr: float, AF: float,
                        # INa parameters
                        G_Na: float, G_Nab: float,
                        # ICaL parameters
                        P_Ca: float, P_Na: float, P_K: float,
                        gamma_Cai: float, gamma_Cao: float,
                        gamma_Nai: float, gamma_Nao: float,
                        gamma_Ki: float, gamma_Ko: float,
                        Km_Ca: float,
                        # ICaT parameters
                        g_CaT: float,
                        # ICab parameters
                        g_Cab: float,
                        # IKr parameters
                        g_Kr_max: float,
                        # IKs parameters
                        G_Ks_max: float, prnak: float,
                        IKs_Ca_max: float, IKs_Ca_Kd: float,
                        # IK1, IKp parameters
                        G_K1_max: float, G_Kp_max: float,
                        # INaCa parameters
                        c1: float, c2: float, gamma_NCX: float,
                        # INaK parameters
                        I_NaK_max: float, Km_Nai: float, Km_Ko: float,
                        # IpCa parameters
                        I_pCa_max: float, Km_pCa: float,
                        # SR parameters
                        I_up_max: float, Km_up: float, NSR_max: float,
                        tau_tr: float,
                        tau_Rel_base: float, alpha_Rel: float,
                        K_Rel_ss: float, qn: float,
                        # Buffering parameters
                        CMDN_max: float, TRPN_max: float,
                        Km_CMDN: float, Km_TRPN: float,
                        CSQN_max: float, Km_CSQN: float) -> np.ndarray:
    """
    Compute derivatives of all 18 state variables.

    This is the core function that computes dy/dt for the ODE system.

    Args:
        y: State vector (18 elements)
        t: Current time (ms) - not used but included for ODE solver compatibility
        I_stim: Applied stimulus current (uA/uF), positive = depolarizing
        ... (all model parameters)

    Returns:
        dy: Derivative vector (18 elements)
    """
    # Unpack state variables
    V = y[0]
    H = y[1]
    m = y[2]
    J = y[3]
    d = y[4]
    f = y[5]
    xr = y[6]
    ca_T = y[7]
    Na_i = y[8]
    K_i = y[9]
    jsr_T = y[10]
    nsr = y[11]
    xs = y[12]
    B = y[13]
    G = y[14]
    xs2 = y[15]
    Rel = y[16]
    Over = y[17]

    # =========================================================================
    # Compute free Ca2+ from buffered totals (analytical solution)
    # =========================================================================
    Ca_i = solve_Ca_i(ca_T, CMDN_max, TRPN_max, Km_CMDN, Km_TRPN)
    Ca_JSR = solve_Ca_JSR(jsr_T, CSQN_max, Km_CSQN)

    # =========================================================================
    # Compute ionic currents
    # =========================================================================
    # INa
    i_Na = I_Na(V, m, H, J, Na_i, Na_o, G_Na, frt)
    i_Nab = I_Nab(V, Na_i, Na_o, G_Nab, frt)

    # ICaL (returns Ca, Na, K components)
    i_CaL_Ca, i_CaL_Na, i_CaL_K = I_CaL(
        V, d, f, Ca_i, Na_i, K_i,
        Na_o, K_o, Ca_o,
        P_Ca, P_Na, P_K,
        gamma_Cai, gamma_Cao,
        gamma_Nai, gamma_Nao,
        gamma_Ki, gamma_Ko,
        Km_Ca, F, frt
    )
    i_CaL = i_CaL_Ca  # Ca component for SR release

    # ICaT
    i_CaT = I_CaT(V, B, G, Ca_i, Ca_o, g_CaT, frt)

    # ICab
    i_Cab = I_Cab(V, Ca_i, Ca_o, g_Cab, frt)

    # IKr
    i_Kr = I_Kr(V, xr, K_i, K_o, g_Kr_max, frt)

    # IKs (Ca-dependent)
    i_Ks = I_Ks(V, xs, xs2, Ca_i, Na_i, K_i, Na_o, K_o,
                G_Ks_max, prnak, IKs_Ca_max, IKs_Ca_Kd, frt)

    # IK1
    i_K1 = I_K1(V, K_i, K_o, G_K1_max, frt)

    # IKp
    i_Kp = I_Kp(V, K_i, K_o, G_Kp_max, frt)

    # INaCa
    i_NaCa = I_NaCa(V, Ca_i, Na_i, Na_o, Ca_o, c1, c2, gamma_NCX, frt)

    # INaK
    i_NaK = I_NaK(V, Na_i, Na_o, K_o, I_NaK_max, Km_Nai, Km_Ko, frt)

    # IpCa
    i_pCa = I_pCa(Ca_i, I_pCa_max, Km_pCa)

    # =========================================================================
    # Compute current sums by ion species
    # =========================================================================
    I_Ca_total, I_Na_total, I_K_total = compute_I_ion_totals(
        i_Na, i_Nab,
        i_CaL_Ca, i_CaL_Na, i_CaL_K,
        i_CaT, i_Cab, i_pCa,
        i_NaCa, i_NaK,
        i_Kr, i_Ks, i_K1, i_Kp,
        I_stim
    )

    # =========================================================================
    # Compute SR calcium fluxes
    # =========================================================================
    i_up = I_up(Ca_i, I_up_max, Km_up)
    i_leak = I_leak(nsr, I_up_max, NSR_max)
    i_tr = I_tr(nsr, Ca_JSR, tau_tr)

    # SR release dynamics
    Rel_ss = Rel_infinity(i_CaL, Ca_JSR, alpha_Rel, K_Rel_ss, qn)
    tau_Rel = tau_Rel_func(Ca_JSR, tau_Rel_base)

    # =========================================================================
    # Compute derivatives
    # =========================================================================
    dy = np.zeros(18)

    # dV/dt = -(sum of all ionic currents)
    dy[0] = -(I_Na_total + I_K_total + I_Ca_total)

    # INa gates (Hodgkin-Huxley formalism)
    am = alpha_m(V)
    bm = beta_m(V)
    ah = alpha_h(V)
    bh = beta_h(V)
    aj = alpha_j(V)
    bj = beta_j(V)

    dy[1] = ah * (1.0 - H) - bh * H  # dH/dt
    dy[2] = am * (1.0 - m) - bm * m  # dm/dt
    dy[3] = aj * (1.0 - J) - bj * J  # dJ/dt

    # ICaL gates (Rush-Larsen formalism, compute as (x_inf - x) / tau)
    dy[4] = (d_inf(V) - d) / tau_d(V)  # dd/dt
    dy[5] = (f_inf(V) - f) / tau_f(V)  # df/dt

    # IKr gate
    dy[6] = (xr_inf(V) - xr) / tau_xr(V)  # dxr/dt

    # Calcium concentrations
    dy[7] = dCa_T_dt(I_Ca_total, i_leak, i_up, Rel, Over, AF, vmyo, vnsr, vjsr)

    # Sodium and potassium concentrations
    dy[8] = dNa_i_dt(I_Na_total, AF, vmyo)  # dNa_i/dt
    dy[9] = dK_i_dt(I_K_total, AF, vmyo)    # dK_i/dt

    # SR calcium
    dy[10] = dJSR_T_dt(i_tr, Rel)           # djsr_T/dt
    dy[11] = dNSR_dt(i_up, i_tr, i_leak, vjsr, vnsr)  # dnsr/dt

    # IKs gates
    tau_xs_val = tau_xs(V)
    xs_inf_val = xs_inf(V)
    dy[12] = (xs_inf_val - xs) / tau_xs_val       # dxs/dt
    dy[15] = (xs_inf_val - xs2) / (tau_xs_val * 4.0)  # dxs2/dt (4x slower)

    # ICaT gates
    dy[13] = (B_inf(V) - B) / tau_B(V)  # dB/dt
    dy[14] = (G_inf(V) - G) / tau_G(V)  # dG/dt

    # SR release
    dy[16] = dRel_dt(Rel, Rel_ss, tau_Rel)  # dRel/dt

    # Overload (unused in standard model)
    dy[17] = 0.0  # dOver/dt

    return dy


@njit(cache=True)
def integrate_step_rush_larsen(y: np.ndarray, t: float, dt: float, I_stim: float,
                                # Physical constants
                                F: float, frt: float,
                                # Extracellular concentrations
                                Na_o: float, K_o: float, Ca_o: float,
                                # Volumes
                                vmyo: float, vnsr: float, vjsr: float, AF: float,
                                # INa parameters
                                G_Na: float, G_Nab: float,
                                # ICaL parameters
                                P_Ca: float, P_Na: float, P_K: float,
                                gamma_Cai: float, gamma_Cao: float,
                                gamma_Nai: float, gamma_Nao: float,
                                gamma_Ki: float, gamma_Ko: float,
                                Km_Ca: float,
                                # ICaT parameters
                                g_CaT: float,
                                # ICab parameters
                                g_Cab: float,
                                # IKr parameters
                                g_Kr_max: float,
                                # IKs parameters
                                G_Ks_max: float, prnak: float,
                                IKs_Ca_max: float, IKs_Ca_Kd: float,
                                # IK1, IKp parameters
                                G_K1_max: float, G_Kp_max: float,
                                # INaCa parameters
                                c1: float, c2: float, gamma_NCX: float,
                                # INaK parameters
                                I_NaK_max: float, Km_Nai: float, Km_Ko: float,
                                # IpCa parameters
                                I_pCa_max: float, Km_pCa: float,
                                # SR parameters
                                I_up_max: float, Km_up: float, NSR_max: float,
                                tau_tr: float,
                                tau_Rel_base: float, alpha_Rel: float,
                                K_Rel_ss: float, qn: float,
                                # Buffering parameters
                                CMDN_max: float, TRPN_max: float,
                                Km_CMDN: float, Km_TRPN: float,
                                CSQN_max: float, Km_CSQN: float) -> np.ndarray:
    """
    Integrate one time step using Rush-Larsen method for gates + Forward Euler for others.

    Rush-Larsen is unconditionally stable for gating variables, making it suitable
    for larger time steps (dt up to ~0.05 ms).

    Args:
        y: State vector (18 elements)
        t: Current time (ms)
        dt: Time step (ms)
        I_stim: Applied stimulus current (uA/uF)
        ... (all model parameters)

    Returns:
        y_new: Updated state vector
    """
    y_new = np.zeros(18)

    # Unpack state variables
    V = y[0]
    H = y[1]
    m = y[2]
    J = y[3]
    d = y[4]
    f = y[5]
    xr = y[6]
    ca_T = y[7]
    Na_i = y[8]
    K_i = y[9]
    jsr_T = y[10]
    nsr = y[11]
    xs = y[12]
    B = y[13]
    G = y[14]
    xs2 = y[15]
    Rel = y[16]
    Over = y[17]

    # =========================================================================
    # Compute free Ca2+ from buffered totals
    # =========================================================================
    Ca_i = solve_Ca_i(ca_T, CMDN_max, TRPN_max, Km_CMDN, Km_TRPN)
    Ca_JSR = solve_Ca_JSR(jsr_T, CSQN_max, Km_CSQN)

    # =========================================================================
    # Compute ionic currents
    # =========================================================================
    i_Na = I_Na(V, m, H, J, Na_i, Na_o, G_Na, frt)
    i_Nab = I_Nab(V, Na_i, Na_o, G_Nab, frt)

    i_CaL_Ca, i_CaL_Na, i_CaL_K = I_CaL(
        V, d, f, Ca_i, Na_i, K_i,
        Na_o, K_o, Ca_o,
        P_Ca, P_Na, P_K,
        gamma_Cai, gamma_Cao,
        gamma_Nai, gamma_Nao,
        gamma_Ki, gamma_Ko,
        Km_Ca, F, frt
    )
    i_CaL = i_CaL_Ca

    i_CaT = I_CaT(V, B, G, Ca_i, Ca_o, g_CaT, frt)
    i_Cab = I_Cab(V, Ca_i, Ca_o, g_Cab, frt)
    i_Kr = I_Kr(V, xr, K_i, K_o, g_Kr_max, frt)
    i_Ks = I_Ks(V, xs, xs2, Ca_i, Na_i, K_i, Na_o, K_o,
                G_Ks_max, prnak, IKs_Ca_max, IKs_Ca_Kd, frt)
    i_K1 = I_K1(V, K_i, K_o, G_K1_max, frt)
    i_Kp = I_Kp(V, K_i, K_o, G_Kp_max, frt)
    i_NaCa = I_NaCa(V, Ca_i, Na_i, Na_o, Ca_o, c1, c2, gamma_NCX, frt)
    i_NaK = I_NaK(V, Na_i, Na_o, K_o, I_NaK_max, Km_Nai, Km_Ko, frt)
    i_pCa = I_pCa(Ca_i, I_pCa_max, Km_pCa)

    # Current sums
    I_Ca_total, I_Na_total, I_K_total = compute_I_ion_totals(
        i_Na, i_Nab, i_CaL_Ca, i_CaL_Na, i_CaL_K,
        i_CaT, i_Cab, i_pCa, i_NaCa, i_NaK,
        i_Kr, i_Ks, i_K1, i_Kp, I_stim
    )

    # SR fluxes
    i_up = I_up(Ca_i, I_up_max, Km_up)
    i_leak = I_leak(nsr, I_up_max, NSR_max)
    i_tr = I_tr(nsr, Ca_JSR, tau_tr)
    Rel_ss = Rel_infinity(i_CaL, Ca_JSR, alpha_Rel, K_Rel_ss, qn)
    tau_Rel = tau_Rel_func(Ca_JSR, tau_Rel_base)

    # =========================================================================
    # Integrate gating variables with Rush-Larsen
    # =========================================================================
    # INa gates - convert alpha/beta to inf/tau form
    am = alpha_m(V)
    bm = beta_m(V)
    m_inf = am / (am + bm)
    tau_m = 1.0 / (am + bm)
    y_new[2] = m_inf + (m - m_inf) * np.exp(-dt / tau_m)

    ah = alpha_h(V)
    bh = beta_h(V)
    h_inf = ah / (ah + bh)
    tau_h = 1.0 / (ah + bh)
    y_new[1] = h_inf + (H - h_inf) * np.exp(-dt / tau_h)

    aj = alpha_j(V)
    bj = beta_j(V)
    j_inf = aj / (aj + bj)
    tau_j = 1.0 / (aj + bj)
    y_new[3] = j_inf + (J - j_inf) * np.exp(-dt / tau_j)

    # ICaL gates
    d_inf_val = d_inf(V)
    tau_d_val = tau_d(V)
    y_new[4] = d_inf_val + (d - d_inf_val) * np.exp(-dt / tau_d_val)

    f_inf_val = f_inf(V)
    tau_f_val = tau_f(V)
    y_new[5] = f_inf_val + (f - f_inf_val) * np.exp(-dt / tau_f_val)

    # IKr gate
    xr_inf_val = xr_inf(V)
    tau_xr_val = tau_xr(V)
    y_new[6] = xr_inf_val + (xr - xr_inf_val) * np.exp(-dt / tau_xr_val)

    # IKs gates
    xs_inf_val = xs_inf(V)
    tau_xs_val = tau_xs(V)
    y_new[12] = xs_inf_val + (xs - xs_inf_val) * np.exp(-dt / tau_xs_val)
    y_new[15] = xs_inf_val + (xs2 - xs_inf_val) * np.exp(-dt / (tau_xs_val * 4.0))

    # ICaT gates
    b_inf_val = B_inf(V)
    tau_b_val = tau_B(V)
    y_new[13] = b_inf_val + (B - b_inf_val) * np.exp(-dt / tau_b_val)

    g_inf_val = G_inf(V)
    tau_g_val = tau_G(V)
    y_new[14] = g_inf_val + (G - g_inf_val) * np.exp(-dt / tau_g_val)

    # =========================================================================
    # Integrate other variables with Forward Euler
    # =========================================================================
    # Voltage
    dV = -(I_Na_total + I_K_total + I_Ca_total)
    y_new[0] = V + dV * dt

    # Concentrations
    y_new[7] = ca_T + dCa_T_dt(I_Ca_total, i_leak, i_up, Rel, Over, AF, vmyo, vnsr, vjsr) * dt
    y_new[8] = Na_i + dNa_i_dt(I_Na_total, AF, vmyo) * dt
    y_new[9] = K_i + dK_i_dt(I_K_total, AF, vmyo) * dt

    # SR calcium
    y_new[10] = jsr_T + dJSR_T_dt(i_tr, Rel) * dt
    y_new[11] = nsr + dNSR_dt(i_up, i_tr, i_leak, vjsr, vnsr) * dt

    # SR release
    y_new[16] = Rel + dRel_dt(Rel, Rel_ss, tau_Rel) * dt

    # Overload (unused)
    y_new[17] = 0.0

    return y_new


class LRd07Model:
    """
    LRd07 Cardiac Cell Model.

    Implements the Livshitz-Rudy 2007 model for guinea pig ventricular myocyte.

    Usage:
        model = LRd07Model()
        y0 = model.params.get_initial_state()
        dy = model.derivatives(y0, 0.0, I_stim=80.0)
    """

    def __init__(self, params: Optional[LRd07Parameters] = None):
        """
        Initialize the model.

        Args:
            params: Model parameters. If None, uses default parameters.
        """
        self.params = params if params is not None else DEFAULT_PARAMS

    def derivatives(self, y: np.ndarray, t: float, I_stim: float = 0.0) -> np.ndarray:
        """
        Compute state variable derivatives.

        Args:
            y: State vector (18 elements)
            t: Current time (ms)
            I_stim: Applied stimulus current (uA/uF)

        Returns:
            dy/dt vector (18 elements)
        """
        p = self.params
        return compute_derivatives(
            y, t, I_stim,
            # Physical constants
            p.F, p.frt,
            # Extracellular concentrations
            p.Na_o, p.K_o, p.Ca_o,
            # Volumes
            p.vmyo, p.vnsr, p.vjsr, p.AF,
            # INa
            p.G_Na, p.G_Nab,
            # ICaL
            p.P_Ca, p.P_Na, p.P_K,
            p.gamma_Cai, p.gamma_Cao,
            p.gamma_Nai, p.gamma_Nao,
            p.gamma_Ki, p.gamma_Ko,
            p.Km_Ca,
            # ICaT
            p.g_CaT,
            # ICab
            p.g_Cab,
            # IKr
            p.g_Kr_max,
            # IKs
            p.G_Ks_max, p.prnak,
            p.IKs_Ca_max, p.IKs_Ca_Kd,
            # IK1, IKp
            p.G_K1_max, p.G_Kp_max,
            # INaCa
            p.c1, p.c2, p.gamma_NCX,
            # INaK
            p.I_NaK_max, p.Km_Nai, p.Km_Ko,
            # IpCa
            p.I_pCa_max, p.Km_pCa,
            # SR
            p.I_up_max, p.Km_up, p.NSR_max,
            p.tau_tr,
            p.tau_Rel, p.alpha_Rel,
            p.K_Rel_ss, p.qn,
            # Buffering
            p.CMDN_max, p.TRPN_max,
            p.Km_CMDN, p.Km_TRPN,
            p.CSQN_max, p.Km_CSQN
        )

    def get_free_Ca(self, y: np.ndarray) -> Tuple[float, float]:
        """
        Get free calcium concentrations from state.

        Args:
            y: State vector

        Returns:
            Tuple of (Ca_i, Ca_JSR) in mM
        """
        p = self.params
        Ca_i = solve_Ca_i(y[StateIndex.ca_T],
                          p.CMDN_max, p.TRPN_max,
                          p.Km_CMDN, p.Km_TRPN)
        Ca_JSR = solve_Ca_JSR(y[StateIndex.jsr_T],
                               p.CSQN_max, p.Km_CSQN)
        return Ca_i, Ca_JSR

    def forward_euler_step(self, y: np.ndarray, t: float, dt: float,
                           I_stim: float = 0.0) -> np.ndarray:
        """
        Single forward Euler integration step.

        WARNING: This method is unstable for dt > 0.005 ms due to fast gating.
        Use rush_larsen_step() instead for stability.

        Args:
            y: Current state
            t: Current time (ms)
            dt: Time step (ms)
            I_stim: Stimulus current (uA/uF)

        Returns:
            Updated state vector
        """
        dy = self.derivatives(y, t, I_stim)
        return y + dy * dt

    def rush_larsen_step(self, y: np.ndarray, t: float, dt: float,
                         I_stim: float = 0.0) -> np.ndarray:
        """
        Single integration step using Rush-Larsen for gating + Forward Euler for others.

        This method is unconditionally stable for gating variables and allows
        time steps up to ~0.05 ms.

        Args:
            y: Current state
            t: Current time (ms)
            dt: Time step (ms)
            I_stim: Stimulus current (uA/uF)

        Returns:
            Updated state vector
        """
        p = self.params
        return integrate_step_rush_larsen(
            y, t, dt, I_stim,
            # Physical constants
            p.F, p.frt,
            # Extracellular concentrations
            p.Na_o, p.K_o, p.Ca_o,
            # Volumes
            p.vmyo, p.vnsr, p.vjsr, p.AF,
            # INa
            p.G_Na, p.G_Nab,
            # ICaL
            p.P_Ca, p.P_Na, p.P_K,
            p.gamma_Cai, p.gamma_Cao,
            p.gamma_Nai, p.gamma_Nao,
            p.gamma_Ki, p.gamma_Ko,
            p.Km_Ca,
            # ICaT
            p.g_CaT,
            # ICab
            p.g_Cab,
            # IKr
            p.g_Kr_max,
            # IKs
            p.G_Ks_max, p.prnak,
            p.IKs_Ca_max, p.IKs_Ca_Kd,
            # IK1, IKp
            p.G_K1_max, p.G_Kp_max,
            # INaCa
            p.c1, p.c2, p.gamma_NCX,
            # INaK
            p.I_NaK_max, p.Km_Nai, p.Km_Ko,
            # IpCa
            p.I_pCa_max, p.Km_pCa,
            # SR
            p.I_up_max, p.Km_up, p.NSR_max,
            p.tau_tr,
            p.tau_Rel, p.alpha_Rel,
            p.K_Rel_ss, p.qn,
            # Buffering
            p.CMDN_max, p.TRPN_max,
            p.Km_CMDN, p.Km_TRPN,
            p.CSQN_max, p.Km_CSQN
        )

    def simulate(self, t_span: Tuple[float, float], y0: Optional[np.ndarray] = None,
                 dt: float = 0.01, bcl: float = 400.0,
                 stim_duration: float = 0.5, stim_amplitude: float = 80.0,
                 stim_start: float = 0.0,
                 method: str = 'rush_larsen') -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a simulation with periodic pacing.

        Args:
            t_span: (t_start, t_end) in ms
            y0: Initial state. If None, uses default.
            dt: Time step (ms)
            bcl: Basic cycle length (ms)
            stim_duration: Stimulus duration (ms)
            stim_amplitude: Stimulus amplitude (uA/uF)
            stim_start: Time of first stimulus in each beat (ms)
            method: Integration method ('rush_larsen' or 'forward_euler')

        Returns:
            Tuple of (t_array, y_array) where y_array has shape (n_steps, 18)
        """
        if y0 is None:
            y0 = self.params.get_initial_state()

        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt) + 1

        t_array = np.linspace(t_start, t_end, n_steps)
        y_array = np.zeros((n_steps, StateIndex.N_STATES))
        y_array[0] = y0

        # Select stepping function
        if method == 'rush_larsen':
            step_func = self.rush_larsen_step
        else:
            step_func = self.forward_euler_step

        y = y0.copy()
        for i in range(1, n_steps):
            t = t_array[i - 1]

            # Determine stimulus
            t_in_beat = t % bcl
            if stim_start <= t_in_beat < stim_start + stim_duration:
                I_stim = stim_amplitude
            else:
                I_stim = 0.0

            y = step_func(y, t, dt, I_stim)
            y_array[i] = y

        return t_array, y_array
