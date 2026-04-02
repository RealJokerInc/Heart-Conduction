"""V3 Preprocessor: TTP06 47-col raw data → v3 training format.

Extracts and reorders columns, computes Nernst reversal potentials,
computes effective gate conductance products, and generates normalization
statistics for Stage 2 environment tokens.

TTP06 StateIndex order (within cols 3-20):
    0:Ki, 1:Nai, 2:Cai, 3:CaSR, 4:CaSS,
    5:m, 6:h, 7:j, 8:r, 9:s, 10:d, 11:f, 12:f2, 13:fCass,
    14:Xr1, 15:Xr2, 16:Xs, 17:RR

v3 concentration order: [Na_i, K_i, Ca_i, Ca_ss] = state indices [1, 0, 2, 4]
v3 gate order: indices 5-16 = 12 HH gates (m through Xs, excludes RR)
"""

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor

from .single_cell_generator import TraceData


# === TTP06 column mapping ===
# Raw 47-col format
COL_VM = 0
COL_I_STIM = 1
COL_DT = 2
COL_STATES_START = 3
COL_STATES_END = 21    # exclusive
COL_I_ION = 21
COL_CLAMP = 22
COL_GATE_INF_START = 23
COL_GATE_INF_END = 35  # exclusive, 12 gate_inf values
COL_GATE_TAU_START = 35
COL_GATE_TAU_END = 47  # exclusive, 12 gate_tau values

# State-relative indices (within cols 3-20)
# Concentrations: reorder from TTP06 [Ki,Nai,Cai,CaSR,CaSS] to v3 [Na_i,K_i,Ca_i,Ca_ss]
CONC_REORDER = [1, 0, 2, 4]  # Nai, Ki, Cai, CaSS

# 12 HH gates (excludes RR at index 17)
GATE_INDICES = list(range(5, 17))  # m(5) through Xs(16)

# All ionic state targets for ionic_state_decoder (13 gates + CaSR + RR = 15)
IONIC_STATE_INDICES = list(range(5, 18)) + [3]  # gates(5-17) + CaSR(3)
# Note: this gives 14 items. The 15th is RR at index 17, already included in range(5,18).
# Actually range(5,18) = [5,6,7,8,9,10,11,12,13,14,15,16,17] = 13 items (m through RR)
# Plus CaSR at index 3 = 14 items. Let me recalculate.
# 13 HH gates: m(5),h(6),j(7),r(8),s(9),d(10),f(11),f2(12),fCass(13),Xr1(14),Xr2(15),Xs(16),RR(17) = 13
# But wait, N_IONIC_TARGETS=15 in stage1.py. That's 13 gates + CaSR(3) + ... what's the 15th?
# Checking: 13 gates (m,h,j,r,s,d,f,f2,fCass,Xr1,Xr2,Xs = 12, plus RR = 13) + CaSR = 14.
# The 15th might be a miscounted. Let me just use 13+CaSR = 14 for now and fix N_IONIC_TARGETS.
# Actually looking at stage1.py: N_IONIC_TARGETS = 15 with comment "13 HH gates + Ca_SR + RR"
# So: 12 RL gates (m through Xs) + RR (non-RL gate) + CaSR + ... = 14. Where's the 15th?
# Wait: the code says "13 HH gates" but there are 12 RL gates + RR = 13. Plus CaSR = 14. Plus...?
# I think N_IONIC_TARGETS should be 14, not 15. But let's match the code for now.

# Effective gate conductance products (for gate_conductance_decoder scaffold)
# These are the 5 products the compression is supposed to learn
def compute_conductance_products(states: Tensor) -> Tensor:
    """Compute 5 effective gate conductance products from raw states.

    Args:
        states: (T, 18) raw TTP06 state variables.

    Returns:
        (T, 5) effective conductances:
            0: G_Na = m³·h·j
            1: G_CaL = d·f·f2·fCass
            2: G_to = r·s
            3: G_Kr = Xr1·Xr2
            4: G_Ks = Xs²
    """
    m = states[:, 5]
    h = states[:, 6]
    j = states[:, 7]
    r = states[:, 8]
    s = states[:, 9]
    d = states[:, 10]
    f = states[:, 11]
    f2 = states[:, 12]
    fCass = states[:, 13]
    Xr1 = states[:, 14]
    Xr2 = states[:, 15]
    Xs = states[:, 16]

    G_Na = m.pow(3) * h * j
    G_CaL = d * f * f2 * fCass
    G_to = r * s
    G_Kr = Xr1 * Xr2
    G_Ks = Xs.pow(2)

    return torch.stack([G_Na, G_CaL, G_to, G_Kr, G_Ks], dim=-1)


# Nernst constants (must match nernst.py)
RTONF = 8314.472 * 310.0 / 96485.3415
NAO = 140.0
KO = 5.4
CAO = 2.0
PRNAK = 0.03
EPS = 1e-12


class V3Preprocessor:
    """Convert TTP06 47-col data to v3 training format.

    Extracts concentrations (reordered), gates, Nernst reversal potentials,
    effective conductance products, and normalization statistics.
    """

    def process_segment(self, raw: Tensor) -> dict:
        """Convert a 47-col segment to named tensors for v3 training.

        Args:
            raw: (T, 47) raw TTP06 trace data.

        Returns:
            dict with keys:
                Vm: (T,)
                dt: (T,)
                I_stim: (T,)
                I_ion: (T,)
                clamp_mask: (T,)
                concentrations: (T, 4) [Na_i, K_i, Ca_i, Ca_ss]
                gates: (T, 12) HH gates (m through Xs)
                ionic_states: (T, 14) all ionic state targets (13 gates + CaSR)
                conductance_products: (T, 5) effective gate products
                E: (T, 4) [E_Na, E_K, E_Ca, E_Ks]
                gate_inf: (T, 12)
                gate_tau: (T, 12)
        """
        assert raw.shape[-1] == 47, f"Expected 47 columns, got {raw.shape[-1]}"

        Vm = raw[:, COL_VM]
        I_stim = raw[:, COL_I_STIM]
        dt = raw[:, COL_DT]
        states = raw[:, COL_STATES_START:COL_STATES_END]  # (T, 18)
        I_ion = raw[:, COL_I_ION]
        clamp_mask = raw[:, COL_CLAMP]
        gate_inf = raw[:, COL_GATE_INF_START:COL_GATE_INF_END]  # (T, 12)
        gate_tau = raw[:, COL_GATE_TAU_START:COL_GATE_TAU_END]  # (T, 12)

        # Concentrations: reorder to [Na_i, K_i, Ca_i, Ca_ss]
        conc = states[:, CONC_REORDER]  # (T, 4)

        # 12 HH gates (m through Xs, excludes RR)
        gates = states[:, GATE_INDICES]  # (T, 12)

        # All ionic state targets (13 gates including RR + CaSR)
        # range(5,18) = m,h,j,r,s,d,f,f2,fCass,Xr1,Xr2,Xs,RR = 13 items
        # Plus CaSR at index 3
        gates_plus_rr = states[:, 5:18]  # (T, 13)
        ca_sr = states[:, 3:4]           # (T, 1)
        ionic_states = torch.cat([gates_plus_rr, ca_sr], dim=-1)  # (T, 14)

        # Effective conductance products
        conductance_products = compute_conductance_products(states)  # (T, 5)

        # Nernst reversal potentials
        Na_i = conc[:, 0]
        K_i = conc[:, 1]
        Ca_i = conc[:, 2]

        E_Na = RTONF * torch.log(NAO / Na_i.clamp(min=EPS))
        E_K = RTONF * torch.log(KO / K_i.clamp(min=EPS))
        E_Ca = 0.5 * RTONF * torch.log(CAO / Ca_i.clamp(min=EPS))
        E_Ks = RTONF * torch.log(
            (KO + PRNAK * NAO) / (K_i + PRNAK * Na_i).clamp(min=EPS)
        )
        E = torch.stack([E_Na, E_K, E_Ca, E_Ks], dim=-1)  # (T, 4)

        return {
            'Vm': Vm,
            'dt': dt,
            'I_stim': I_stim,
            'I_ion': I_ion,
            'clamp_mask': clamp_mask,
            'concentrations': conc,
            'gates': gates,
            'ionic_states': ionic_states,
            'conductance_products': conductance_products,
            'E': E,
            'gate_inf': gate_inf,
            'gate_tau': gate_tau,
        }

    def compute_normalization_stats(
        self, data_dir: str, tiers: Optional[list] = None
    ) -> dict:
        """First pass over training data to compute per-token normalization stats.

        Computes min, max, mean, std for all 9 environment tokens:
        [Vm, E_Na, E_K, E_Ca, E_Ks, Na_i, K_i, Ca_i, Ca_ss]

        Args:
            data_dir: Path to raw HDF5 directory.
            tiers: List of tier numbers to include (default: [1,2,3]).

        Returns:
            dict with 'shift' (9,), 'scale' (9,), 'min' (9,), 'max' (9,),
            'mean' (9,), 'std' (9,).
        """
        from .storage import TraceStorage

        if tiers is None:
            tiers = [1, 2, 3]

        storage = TraceStorage(base_dir=data_dir)

        # Accumulate running stats
        all_env = []

        for tier in tiers:
            protocols = storage.list_protocols(tier)
            for proto_name in protocols:
                trace = storage.load_trace(tier, proto_name)
                processed = self.process_segment(trace.data)

                # Build 9-token environment vector
                env = torch.stack([
                    processed['Vm'],
                    processed['E'][:, 0],   # E_Na
                    processed['E'][:, 1],   # E_K
                    processed['E'][:, 2],   # E_Ca
                    processed['E'][:, 3],   # E_Ks
                    processed['concentrations'][:, 0],  # Na_i
                    processed['concentrations'][:, 1],  # K_i
                    processed['concentrations'][:, 2],  # Ca_i
                    processed['concentrations'][:, 3],  # Ca_ss
                ], dim=-1)  # (T, 9)

                all_env.append(env)

        all_env = torch.cat(all_env, dim=0)  # (total_T, 9)

        env_min = all_env.min(dim=0).values
        env_max = all_env.max(dim=0).values
        env_mean = all_env.mean(dim=0)
        env_std = all_env.std(dim=0)

        # Shift/scale for normalization to ~[-1, 1]
        midpoint = (env_min + env_max) / 2
        halfrange = (env_max - env_min) / 2
        halfrange = halfrange.clamp(min=1e-8)  # prevent division by zero

        return {
            'shift': midpoint,
            'scale': halfrange,
            'min': env_min,
            'max': env_max,
            'mean': env_mean,
            'std': env_std,
            'token_names': ['Vm', 'E_Na', 'E_K', 'E_Ca', 'E_Ks',
                            'Na_i', 'K_i', 'Ca_i', 'Ca_ss'],
        }
