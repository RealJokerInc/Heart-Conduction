"""ORd single-cell data generator.

Generates (T, 101) traces from the O'Hara-Rudy model. Mirrors the TTP06
SingleCellGenerator pattern but with ORd-specific handling:
- 40 state variables (vs 18)
- 28 Rush-Larsen gates for gate_inf/tau (vs 12)
- CaMKII warmup (100 beats at BCL=1000 before recording)
- Istim= kwarg (not I_stim=)
- ORdConcentrationPerturbation (params_override, not TTP06 ConcentrationPerturbation)
"""

import sys
import copy
from pathlib import Path
from typing import Optional, Dict, List

import torch
from torch import Tensor

# Add engine path for ORd model imports
_ENGINE_PATH = str(Path(__file__).resolve().parents[3] / 'Bidomain' / 'Engine_V1')
if _ENGINE_PATH not in sys.path:
    sys.path.insert(0, _ENGINE_PATH)

from cardiac_sim.ionic.ord.model import ORdModel
from cardiac_sim.ionic.ord.parameters import StateIndex, V_REST
from cardiac_sim.ionic.base import CellType

from .ord_trace_data import ORdTraceData
from .augmentation import StitchedProtocol, corrupt_states


# === Hyperparameters ===
WARMUP_BEATS = 100
WARMUP_BCL = 1000.0     # ms
WARMUP_STIM_AMP = -80.0 # uA/uF (same convention as TTP06)
WARMUP_STIM_DUR = 1.0   # ms

_CELLTYPE_MAP = {
    'EPI': CellType.EPI,
    'ENDO': CellType.ENDO,
    'M_CELL': CellType.M_CELL,
}


class ORdConcentrationPerturbation:
    """ORd-specific concentration perturbation via params_override.

    Unlike TTP06's ConcentrationPerturbation which modifies model params via
    deepcopy, ORd perturbations use params_override keys (lowercase: ko, nao, cao).
    """

    def __init__(self, base_protocol, **overrides):
        """
        Args:
            base_protocol: The underlying pacing protocol.
            **overrides: ORd parameter overrides, e.g., ko=8.0, nao=130.0, cao=1.0.
        """
        self.base_protocol = base_protocol
        self.overrides = overrides
        self.name = f"conc_perturb_{base_protocol.name}"
        self.tier = base_protocol.tier if hasattr(base_protocol, 'tier') else 7
        self.duration_ms = base_protocol.duration_ms

    # Delegate timing methods to base_protocol
    def get_I_stim(self, t): return self.base_protocol.get_I_stim(t)
    def get_I_ext(self, t): return self.base_protocol.get_I_ext(t) if hasattr(self.base_protocol, 'get_I_ext') else 0.0
    def get_dt(self, t): return self.base_protocol.get_dt(t)
    def is_clamped(self, t): return self.base_protocol.is_clamped(t) if hasattr(self.base_protocol, 'is_clamped') else False
    def get_clamp_voltage(self, t): return self.base_protocol.get_clamp_voltage(t) if hasattr(self.base_protocol, 'get_clamp_voltage') else -80.0


class ORdSingleCellGenerator:
    """Generate single-cell ORd traces in 101-column format.

    Args:
        cell_type: 'EPI', 'ENDO', or 'M_CELL'.
        device: 'cpu' or 'cuda'.
        warmup_beats: CaMKII warmup beats (default 100).
        params_override: Optional ORd parameter overrides.
    """

    def __init__(self, cell_type: str = 'EPI', device: str = 'cpu',
                 warmup_beats: int = WARMUP_BEATS,
                 params_override: Optional[Dict[str, float]] = None):
        self.cell_type_str = cell_type
        self.cell_type = _CELLTYPE_MAP[cell_type]
        self.device = device
        self.warmup_beats = warmup_beats
        self.model = ORdModel(
            cell_type=self.cell_type,
            device=device,
            params_override=params_override,
        )

    def _warmup(self, V: Tensor, states: Tensor) -> tuple:
        """Silent CaMKII warmup. No recording."""
        dt = 0.01
        model = self.model
        for beat in range(self.warmup_beats):
            n_steps = int(WARMUP_BCL / dt)
            for step in range(n_steps):
                t = step * dt
                I_stim = WARMUP_STIM_AMP if t < WARMUP_STIM_DUR else 0.0
                stim = torch.tensor(I_stim, dtype=torch.float64, device=self.device)
                V, states = model.step(V, states, dt, Istim=stim)
        return V, states

    def run_protocol(self, protocol) -> ORdTraceData:
        """Run a protocol and return 101-column ORd trace."""
        model = self.model
        V = torch.tensor(V_REST, dtype=torch.float64, device=self.device)
        states = model.get_initial_state(n_cells=1).to(self.device)

        # Handle ORdConcentrationPerturbation BEFORE warmup
        # (warmup steady-state depends on extracellular concentrations)
        if isinstance(protocol, ORdConcentrationPerturbation):
            model = self._setup_concentration_perturbation(protocol)
            protocol = protocol.base_protocol

        # Handle initial_states override
        if hasattr(protocol, 'initial_states') and protocol.initial_states is not None:
            states = protocol.initial_states.to(self.device)

        # CaMKII warmup
        if self.warmup_beats > 0:
            V, states = self._warmup(V, states)

        # Handle CorruptionRecovery
        if hasattr(protocol, 'corruption_type'):
            states = corrupt_states(
                states, protocol.corruption_type,
                getattr(protocol, 'severity', 0.5),
                model_type='ord'
            )
            if hasattr(protocol, 'base_protocol'):
                protocol = protocol.base_protocol

        # Handle StitchedProtocol
        if isinstance(protocol, StitchedProtocol):
            return self._run_stitched(protocol, model, V, states)

        return self._run_loop(protocol, model, V, states)

    def _run_loop(self, protocol, model, V, states) -> ORdTraceData:
        """Core simulation loop. Returns (T, 101) ORdTraceData."""
        records = []
        t = 0.0
        while t < protocol.duration_ms:
            dt = protocol.get_dt(t) if hasattr(protocol, 'get_dt') else 0.01
            I_stim = protocol.get_I_stim(t) if hasattr(protocol, 'get_I_stim') else 0.0
            I_ext = protocol.get_I_ext(t) if hasattr(protocol, 'get_I_ext') else 0.0
            clamped = 1.0 if (hasattr(protocol, 'is_clamped') and protocol.is_clamped(t)) else 0.0

            # Compute I_ion from current state
            I_ion = model.compute_Iion(V, states)

            # Record: sign-flip stimulus (positive = depolarizing in our convention)
            recorded_stim = -(I_stim + I_ext)

            # Build record row: [Vm, I_stim, dt, 40 states, I_ion, clamp_mask] = 45 cols
            state_flat = states.squeeze(0) if states.dim() > 1 else states
            record = torch.cat([
                V.reshape(1),
                torch.tensor([recorded_stim, dt], dtype=torch.float64, device=self.device),
                state_flat,                   # 40 states
                I_ion.reshape(1),
                torch.tensor([clamped], dtype=torch.float64, device=self.device),
            ])
            records.append(record)

            # Advance state
            if clamped > 0.5 and hasattr(protocol, 'get_clamp_voltage'):
                V_clamp = torch.tensor(protocol.get_clamp_voltage(t),
                                       dtype=torch.float64, device=self.device)
                _, states = model.step(V_clamp, states, dt, Istim=None)
                V = torch.tensor(protocol.get_clamp_voltage(t + dt),
                                 dtype=torch.float64, device=self.device)
            else:
                total_stim = torch.tensor(I_stim + I_ext,
                                          dtype=torch.float64, device=self.device)
                V_new, states = model.step(V, states, dt, Istim=total_stim)
                # Adaptive dt tracking
                if hasattr(protocol, '_last_dvdt'):
                    protocol._last_dvdt = float((V_new.item() - V.item()) / dt) if dt > 0 else 0.0
                V = V_new

            t += dt

        # Stack core data
        data_core = torch.stack(records)  # (T, 45)

        # Post-hoc: gate_inf (28) and gate_tau (28), vectorized
        Vm_all = data_core[:, ORdTraceData.VM]
        states_all = data_core[:, ORdTraceData.STATES_START:ORdTraceData.STATES_END]
        gate_inf = model.compute_gate_steady_states(Vm_all, states_all)   # (T, 28)
        gate_tau = model.compute_gate_time_constants(Vm_all, states_all)  # (T, 28)
        data = torch.cat([data_core, gate_inf, gate_tau], dim=1)          # (T, 101)

        metadata = {
            'cell_type': self.cell_type_str,
            'model': 'ord',
            'warmup_beats': self.warmup_beats,
            'protocol_name': getattr(protocol, 'name', 'unknown'),
            'tier': getattr(protocol, 'tier', 0),
        }
        return ORdTraceData(data=data, metadata=metadata)

    def _run_stitched(self, protocol, model, V, states) -> ORdTraceData:
        """Run StitchedProtocol: concatenate sub-protocol traces, carry state forward."""
        all_traces = []
        for i, sub_proto in enumerate(protocol.protocols):
            trace = self._run_loop(sub_proto, model, V, states)
            all_traces.append(trace)
            # Carry forward final state using ORd column offsets
            final_row = trace.data[-1]
            V = final_row[ORdTraceData.VM].unsqueeze(0)
            states = final_row[ORdTraceData.STATES_START:ORdTraceData.STATES_END].unsqueeze(0)

            # Rest period (if not last sub-protocol)
            if i < len(protocol.protocols) - 1 and i < len(protocol.rest_durations):
                rest_ms = protocol.rest_durations[i]
                if rest_ms > 0:
                    dt = 0.01
                    n_rest_steps = int(rest_ms / dt)
                    rest_records = []
                    for _ in range(n_rest_steps):
                        I_ion = model.compute_Iion(V, states)
                        state_flat = states.squeeze(0) if states.dim() > 1 else states
                        record = torch.cat([
                            V.reshape(1),
                            torch.tensor([0.0, dt], dtype=torch.float64, device=self.device),
                            state_flat, I_ion.reshape(1),
                            torch.tensor([0.0], dtype=torch.float64, device=self.device),
                        ])
                        rest_records.append(record)
                        V, states = model.step(V, states, dt, Istim=None)
                    if rest_records:
                        rest_core = torch.stack(rest_records)
                        Vm_rest = rest_core[:, ORdTraceData.VM]
                        st_rest = rest_core[:, ORdTraceData.STATES_START:ORdTraceData.STATES_END]
                        gi = model.compute_gate_steady_states(Vm_rest, st_rest)
                        gt = model.compute_gate_time_constants(Vm_rest, st_rest)
                        rest_data = torch.cat([rest_core, gi, gt], dim=1)
                        all_traces.append(ORdTraceData(
                            data=rest_data,
                            metadata={'protocol_name': f'rest_{i}', 'model': 'ord'}
                        ))

        combined_data = torch.cat([t.data for t in all_traces], dim=0)
        metadata = {
            'cell_type': self.cell_type_str,
            'model': 'ord',
            'warmup_beats': self.warmup_beats,
            'protocol_name': 'stitched',
            'stitched': True,
            'n_sub': len(protocol.protocols),
        }
        return ORdTraceData(data=combined_data, metadata=metadata)

    def _setup_concentration_perturbation(self, protocol: ORdConcentrationPerturbation):
        """Create new ORdModel with modified extracellular concentrations."""
        new_model = copy.deepcopy(self.model)
        for key, val in protocol.overrides.items():
            if hasattr(new_model.params, key):
                setattr(new_model.params, key, val)
        return new_model
