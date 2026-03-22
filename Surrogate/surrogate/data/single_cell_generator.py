"""Single-cell TTP06 ODE generator for surrogate training data.

Wraps the Bidomain V1 TTP06 model to execute pacing protocols and record
(Vm, I_stim, dt, 18 ionic states, I_ion, clamp_mask) at every timestep.

Sign convention: TTP06 uses negative I_stim for depolarizing (e.g., -80).
Recorded I_stim is sign-flipped so depolarizing = positive in surrogate convention.
"""

import sys
import copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, Tuple

import torch

# Add Bidomain engine to path for TTP06 imports
_BIDOMAIN_ROOT = Path(__file__).resolve().parents[3] / 'Bidomain' / 'Engine_V1'
if str(_BIDOMAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BIDOMAIN_ROOT))

from cardiac_sim.ionic.ttp06.model import TTP06Model
from cardiac_sim.ionic.ttp06.parameters import (
    StateIndex, get_celltype_parameters, V_REST
)
from cardiac_sim.ionic.ttp06.celltypes.standard import CellTypeConfig
from cardiac_sim.ionic.base import CellType


@dataclass
class TraceData:
    """Container for recorded single-cell trace data.

    Attributes:
        data: (T, 47) float64 tensor with columns:
            0: Vm (mV)
            1: I_stim (sign-flipped: positive = depolarizing)
            2: dt (ms)
            3-20: 18 ionic states [Ki, Nai, Cai, CaSR, CaSS, m, h, j, r, s, d, f, f2, fCass, Xr1, Xr2, Xs, RR]
            21: I_ion (pure ionic current from compute_Iion, no stimulus)
            22: clamp_mask (0.0 = free-running, 1.0 = voltage clamped)
            23-34: 12 gate_inf values (steady-state at current Vm, gate_indices order)
            35-46: 12 gate_tau values (time constants at current Vm, gate_indices order, ms)
        metadata: dict with protocol info, cell_type, conductances, etc.
    """
    data: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Column indices
    VM = 0
    I_STIM = 1
    DT = 2
    STATES_START = 3
    STATES_END = 21    # exclusive end (Python convention): states are columns 3..20
    I_ION = 21
    CLAMP_MASK = 22
    GATE_INF_START = 23
    GATE_INF_END = 35   # exclusive: 12 gates in gate_indices order
    GATE_TAU_START = 35
    GATE_TAU_END = 47   # exclusive: 12 gate taus in gate_indices order
    N_COLUMNS = 47


class SingleCellGenerator:
    """Runs TTP06 single-cell ODE for any protocol, recording full state.

    Args:
        cell_type: 'EPI', 'ENDO', or 'M_CELL'
        device: torch device string
        conductance_scaling: optional {conductance_name: scale_factor}.
            Scale factors are relative (0.5 = half, 2.0 = double).
            Converted to absolute values via base conductance lookup,
            then applied via CellTypeConfig + TTP06Model.from_config().
    """

    _CELLTYPE_MAP = {
        'EPI': CellType.EPI,
        'ENDO': CellType.ENDO,
        'M_CELL': CellType.M_CELL,
    }

    def __init__(self, cell_type: str = 'EPI', device: str = 'cuda',
                 conductance_scaling: Optional[Dict[str, float]] = None):
        if cell_type not in self._CELLTYPE_MAP:
            raise ValueError(f"Unknown cell_type '{cell_type}'. "
                             f"Valid: {list(self._CELLTYPE_MAP.keys())}")

        self.cell_type_str = cell_type
        self.cell_type_enum = self._CELLTYPE_MAP[cell_type]
        self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu'
                                   else 'cpu')
        self.conductance_scaling = conductance_scaling

        if conductance_scaling:
            config = self._make_scaled_config(conductance_scaling)
            self.model = TTP06Model.from_config(
                config, base_cell_type=self.cell_type_enum, device=self.device
            )
        else:
            self.model = TTP06Model(cell_type=self.cell_type_enum, device=self.device)

    def _get_base_conductances(self) -> Dict[str, float]:
        """Look up default conductance values for this cell type."""
        base_params = get_celltype_parameters(self.cell_type_enum)
        return {
            'GNa': base_params.GNa, 'GK1': base_params.GK1,
            'Gto': base_params.Gto, 'GKr': base_params.GKr,
            'GKs': base_params.GKs, 'GCaL': base_params.PCa,
            'GpCa': base_params.GpCa, 'GpK': base_params.GpK,
            'GbNa': base_params.GbNa, 'GbCa': base_params.GbCa,
            'PCa': base_params.PCa, 'KNaCa': base_params.KNaCa,
            'PNaK': base_params.PNaK,
        }

    def _make_scaled_config(self, scaling: Dict[str, float]) -> CellTypeConfig:
        """Convert scale factors to absolute conductances and build CellTypeConfig."""
        base = self._get_base_conductances()
        absolute = {}
        for name, scale in scaling.items():
            if name not in base:
                raise ValueError(f"Unknown conductance '{name}'. "
                                 f"Valid: {list(base.keys())}")
            absolute[name] = base[name] * scale

        return CellTypeConfig(
            name=f'scaled_{self.cell_type_str}',
            use_epi_ito_kinetics=(self.cell_type_enum != CellType.ENDO),
            **absolute,
        )

    def run_protocol(self, protocol) -> TraceData:
        """Execute a protocol and return full trace data.

        Handles free-running, voltage clamp, and partial clamp modes.
        For ConcentrationPerturbation protocols, re-instantiates the model
        with modified TTP06Parameters.

        Returns:
            TraceData with (T, 23) tensor and metadata dict.
        """
        # Determine model and initial state
        model = self.model
        V = torch.tensor(V_REST, dtype=torch.float64, device=self.device)
        states = model.get_initial_state(n_cells=1).to(self.device)

        # Handle concentration perturbation (modifies model params, not CellTypeConfig)
        from .protocols import ConcentrationPerturbation
        if isinstance(protocol, ConcentrationPerturbation):
            model, states = self._setup_concentration_perturbation(protocol, states)
            protocol = protocol.base_protocol

        # Handle StitchedProtocol
        from .augmentation import StitchedProtocol
        if isinstance(protocol, StitchedProtocol):
            return self._run_stitched(protocol, model, V, states)

        # Handle corruption recovery (modifies initial states)
        from .protocols import CorruptionRecovery
        if isinstance(protocol, CorruptionRecovery):
            from .augmentation import corrupt_states
            states = corrupt_states(states, protocol.corruption_type, protocol.severity)
            protocol = protocol.base_protocol

        # Override initial states if protocol specifies them
        if hasattr(protocol, 'initial_states') and protocol.initial_states is not None:
            states = protocol.initial_states.clone().to(self.device)

        return self._run_loop(protocol, model, V, states)

    def _run_loop(self, protocol, model, V, states) -> TraceData:
        """Core simulation loop."""
        records = []
        t = 0.0

        while t < protocol.duration_ms:
            dt = protocol.get_dt(t)
            I_stim = protocol.get_I_stim(t)
            I_ext = protocol.get_I_ext(t)
            clamped = 1.0 if protocol.is_clamped(t) else 0.0

            # Pure ionic current (no stimulus)
            I_ion = model.compute_Iion(V, states)

            # Gate steady-states and time constants (12 each, in gate_indices order)
            gate_inf = model.compute_gate_steady_states(V, states)  # (1, 12) or (12,)
            gate_tau = model.compute_gate_time_constants(V, states)  # (1, 12) or (12,)
            gate_inf_flat = gate_inf.squeeze(0) if gate_inf.dim() > 1 else gate_inf
            gate_tau_flat = gate_tau.squeeze(0) if gate_tau.dim() > 1 else gate_tau

            # Sign flip for surrogate convention: negate TTP06 stimulus
            recorded_stim = -(I_stim + I_ext)

            # Record: [Vm, recorded_stim, dt, 18 states, I_ion, clamp_mask, 12 gate_inf, 12 gate_tau]
            state_flat = states.squeeze(0) if states.dim() > 1 else states
            record = torch.cat([
                V.reshape(1),
                torch.tensor([recorded_stim, dt], dtype=torch.float64, device=self.device),
                state_flat,
                I_ion.reshape(1),
                torch.tensor([clamped], dtype=torch.float64, device=self.device),
                gate_inf_flat,
                gate_tau_flat,
            ])
            records.append(record)

            # Advance state
            if protocol.is_clamped(t):
                if hasattr(protocol, 'alpha'):
                    # Partial clamp: blend V_cmd with V_free
                    V_cmd = torch.tensor(protocol.get_clamp_voltage(t),
                                         dtype=torch.float64, device=self.device)
                    V_free, states = model.step(V, states, dt, I_stim=None)
                    V = protocol.alpha * V_cmd + (1 - protocol.alpha) * V_free
                else:
                    # Full clamp: override Vm, still update gates
                    V_clamp = torch.tensor(protocol.get_clamp_voltage(t),
                                           dtype=torch.float64, device=self.device)
                    _, states = model.step(V_clamp, states, dt, I_stim=None)
                    V = torch.tensor(protocol.get_clamp_voltage(t + dt),
                                     dtype=torch.float64, device=self.device)
            else:
                # Free-running: TTP06 step with stimulus
                total_stim = torch.tensor(I_stim + I_ext,
                                          dtype=torch.float64, device=self.device)
                V_new, states = model.step(V, states, dt, I_stim=total_stim)
                V = V_new

            t += dt

            # Update adaptive dt if protocol tracks dVm/dt
            if hasattr(protocol, '_last_dvdt'):
                protocol._last_dvdt = float((V - records[-1][0]) / dt)

        data = torch.stack(records)  # (T, 23)
        metadata = {
            'protocol_name': protocol.name,
            'protocol_tier': protocol.tier,
            'cell_type': self.cell_type_str,
            'duration_ms': protocol.duration_ms,
            'dt_default': protocol.dt_default,
            'conductance_scaling': self.conductance_scaling,
            'n_timesteps': len(records),
        }
        return TraceData(data=data, metadata=metadata)

    def _setup_concentration_perturbation(self, protocol, states):
        """Create modified model for concentration perturbation.

        K_o/Na_o/Ca_o are TTP06Parameters fields, NOT CellTypeConfig fields.
        Must deepcopy model.params and modify directly.
        """
        model = copy.deepcopy(self.model)
        if protocol.Ko is not None:
            model.params.Ko = protocol.Ko
        if protocol.Nai_init is not None:
            if states.dim() > 1:
                states[:, StateIndex.Nai] = protocol.Nai_init
            else:
                states[StateIndex.Nai] = protocol.Nai_init
        if protocol.Cai_scale != 1.0:
            if states.dim() > 1:
                states[:, StateIndex.Cai] *= protocol.Cai_scale
            else:
                states[StateIndex.Cai] *= protocol.Cai_scale
        return model, states

    def _run_stitched(self, stitched, model, V, states) -> TraceData:
        """Run a stitched protocol: sequential sub-protocols with rest breaks."""
        all_records = []
        for i, sub_proto in enumerate(stitched.protocols):
            trace = self._run_loop(sub_proto, model, V, states)
            all_records.append(trace.data)
            # Carry forward final state
            V = trace.data[-1, TraceData.VM].clone().unsqueeze(0)
            states = trace.data[-1, TraceData.STATES_START:TraceData.STATES_END].clone().unsqueeze(0)

            # Insert rest period
            if i < len(stitched.rest_durations):
                rest_ms = stitched.rest_durations[i]
                from .protocols import QuiescentProtocol
                rest_proto = QuiescentProtocol(duration_ms=rest_ms,
                                               dt_default=sub_proto.dt_default)
                rest_trace = self._run_loop(rest_proto, model, V, states)
                all_records.append(rest_trace.data)
                V = rest_trace.data[-1, TraceData.VM].clone().unsqueeze(0)
                states = rest_trace.data[-1, TraceData.STATES_START:TraceData.STATES_END].clone().unsqueeze(0)

        data = torch.cat(all_records, dim=0)
        metadata = {
            'protocol_name': 'stitched',
            'protocol_tier': 11,
            'cell_type': self.cell_type_str,
            'n_sub_protocols': len(stitched.protocols),
            'n_timesteps': len(data),
        }
        return TraceData(data=data, metadata=metadata)

    def run_pacing(self, bcl: float, n_beats: int, dt: float = 0.01,
                   stim_amplitude: float = -80.0, stim_duration: float = 1.0,
                   I_ext: Optional[Callable] = None) -> TraceData:
        """Convenience for simple pacing protocols."""
        from .protocols import SteadyStatePacing
        proto = SteadyStatePacing(bcl=bcl, n_beats=n_beats,
                                   stim_amplitude=stim_amplitude,
                                   stim_duration=stim_duration,
                                   dt_default=dt)
        return self.run_protocol(proto)
