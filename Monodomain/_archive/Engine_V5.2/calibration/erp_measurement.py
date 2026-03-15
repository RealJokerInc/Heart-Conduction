"""
ERP (Effective Refractory Period) Measurement.

Implements S1-S2 protocol for measuring:
- Single-cell ERP
- Tissue ERP (1D cable)

The S1-S2 protocol:
1. Apply S1 stimulus to establish baseline beat
2. Wait for repolarization
3. Apply S2 stimulus at decreasing intervals
4. ERP = minimum S1-S2 interval that produces propagating AP
"""

import torch
import numpy as np
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ionic.model import ORdModel
from ionic.parameters import CellType
from .cable_1d import Cable1D, CableConfig


@dataclass
class S1S2Config:
    """Configuration for S1-S2 protocol."""

    # S1 parameters
    s1_bcl: float = 1000.0      # S1 basic cycle length (ms)
    s1_count: int = 10          # Number of S1 beats to establish steady state

    # S2 parameters
    s2_start: float = 400.0     # Initial S1-S2 interval (ms)
    s2_end: float = 200.0       # Final S1-S2 interval (ms)
    s2_step: float = 10.0       # Step size for S2 interval (ms)
    s2_fine_step: float = 2.0   # Fine step near ERP (ms)

    # Stimulus
    stim_amplitude: float = -80.0  # Stimulus current (uA/uF)
    stim_duration: float = 1.0     # Stimulus duration (ms)

    # Detection
    ap_threshold: float = 0.0       # Threshold for AP detection (mV)
    min_ap_amplitude: float = 80.0  # Minimum amplitude for valid AP (mV)


class ERPResult(NamedTuple):
    """Results from ERP measurement."""
    erp: float              # Effective refractory period (ms)
    apd90: float            # APD90 from S1 beat (ms)
    last_successful_s2: float   # Last S1-S2 that produced AP
    success: bool           # Whether measurement succeeded


class SingleCellERP:
    """
    Measures ERP in isolated single cell using S1-S2 protocol.

    Single-cell ERP provides the intrinsic refractory period without
    source-sink effects from tissue coupling.
    """

    def __init__(
        self,
        config: S1S2Config,
        dt: float = 0.02,
        cell_type: int = 0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize single-cell ERP measurement.

        Args:
            config: S1-S2 protocol configuration
            dt: Time step (ms)
            cell_type: Cell type (0=ENDO, 1=EPI, 2=M_CELL)
            device: Torch device
        """
        self.config = config
        self.dt = dt
        self.cell_type = cell_type
        self.device = device or torch.device('cpu')

        # Map int cell type to CellType enum
        cell_type_map = {0: CellType.ENDO, 1: CellType.EPI, 2: CellType.M_CELL}
        self.cell_type_enum = cell_type_map.get(cell_type, CellType.ENDO)

        # Initialize ionic model
        device_str = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.ionic = ORdModel(celltype=self.cell_type_enum, device=device_str)

    def measure(self, verbose: bool = False) -> ERPResult:
        """
        Run S1-S2 protocol to measure single-cell ERP.

        Returns:
            ERPResult with ERP, APD90, etc.
        """
        cfg = self.config

        # Initialize state - ORdModel.get_initial_state() returns (41,) tensor
        state = self.ionic.get_initial_state()
        state = state.unsqueeze(0)  # (1, n_states) for batch dimension

        # === Phase 1: S1 pacing to steady state ===
        if verbose:
            print("Phase 1: S1 pacing...")

        s1_apd90 = 0.0
        for beat in range(cfg.s1_count):
            state, apd = self._run_beat(
                state,
                bcl=cfg.s1_bcl,
                stim_amp=cfg.stim_amplitude,
                stim_dur=cfg.stim_duration
            )
            s1_apd90 = apd

            if verbose and (beat == 0 or beat == cfg.s1_count - 1):
                print(f"  Beat {beat + 1}: APD90 = {apd:.1f} ms")

        # === Phase 2: S1-S2 protocol ===
        if verbose:
            print("\nPhase 2: S1-S2 protocol...")

        # Store steady-state for reset
        steady_state = state.clone()

        last_successful = cfg.s2_start

        # Coarse search first
        s2_interval = cfg.s2_start
        while s2_interval >= cfg.s2_end:
            state = steady_state.clone()

            # Apply S1
            state = self._apply_stimulus(state, cfg.stim_duration)

            # Wait for S2 interval
            n_wait = int((s2_interval - cfg.stim_duration) / self.dt)
            for _ in range(n_wait):
                state = self.ionic.step(state, self.dt, None)

            # Apply S2
            state = self._apply_stimulus(state, cfg.stim_duration)

            # Check if AP was triggered
            ap_triggered = self._check_ap(state)

            if verbose:
                status = "AP" if ap_triggered else "no AP"
                print(f"  S2 = {s2_interval:.1f} ms: {status}")

            if ap_triggered:
                last_successful = s2_interval
                s2_interval -= cfg.s2_step
            else:
                # Found approximate ERP, do fine search
                break

        # Fine search
        erp = last_successful
        s2_interval = last_successful
        while s2_interval >= max(cfg.s2_end, last_successful - cfg.s2_step):
            state = steady_state.clone()

            # Apply S1
            state = self._apply_stimulus(state, cfg.stim_duration)

            # Wait for S2 interval
            n_wait = int((s2_interval - cfg.stim_duration) / self.dt)
            for _ in range(n_wait):
                state = self.ionic.step(state, self.dt, None)

            # Apply S2
            state = self._apply_stimulus(state, cfg.stim_duration)

            # Check if AP was triggered
            ap_triggered = self._check_ap(state)

            if ap_triggered:
                erp = s2_interval

            s2_interval -= cfg.s2_fine_step

        if verbose:
            print(f"\nERP (single cell) = {erp:.1f} ms")
            print(f"APD90 = {s1_apd90:.1f} ms")
            if s1_apd90 > 0:
                print(f"ERP/APD ratio = {erp / s1_apd90:.2f}")

        return ERPResult(
            erp=erp,
            apd90=s1_apd90,
            last_successful_s2=last_successful,
            success=True
        )

    def _apply_stimulus(
        self,
        state: torch.Tensor,
        duration: float
    ) -> torch.Tensor:
        """Apply stimulus for specified duration."""
        I_stim = torch.full((1,), self.config.stim_amplitude,
                            device=self.device, dtype=torch.float64)
        n_steps = int(duration / self.dt)

        for _ in range(n_steps):
            state = self.ionic.step(state, self.dt, I_stim)

        return state

    def _run_beat(
        self,
        state: torch.Tensor,
        bcl: float,
        stim_amp: float,
        stim_dur: float
    ) -> Tuple[torch.Tensor, float]:
        """
        Run one beat and measure APD90.

        Returns:
            (final_state, APD90)
        """
        # Apply stimulus
        I_stim = torch.full((1,), stim_amp, device=self.device, dtype=torch.float64)
        n_stim = int(stim_dur / self.dt)

        v_trace = []

        for _ in range(n_stim):
            state = self.ionic.step(state, self.dt, I_stim)
            v_trace.append(state[0, 0].item())

        # Continue without stimulus
        n_rest = int((bcl - stim_dur) / self.dt)
        for _ in range(n_rest):
            state = self.ionic.step(state, self.dt, None)
            v_trace.append(state[0, 0].item())

        # Compute APD90
        v_trace = np.array(v_trace)
        i_peak = np.argmax(v_trace)
        v_peak = v_trace[i_peak]
        v_rest = v_trace[-1]

        amplitude = v_peak - v_rest
        if amplitude < 50:
            return state, 0.0

        v_90 = v_rest + 0.1 * amplitude
        v_after = v_trace[i_peak:]

        idx = np.where(v_after < v_90)[0]
        if len(idx) > 0:
            apd90 = idx[0] * self.dt
        else:
            apd90 = 0.0

        return state, apd90

    def _check_ap(
        self,
        state: torch.Tensor
    ) -> bool:
        """
        Check if S2 stimulus triggered a valid AP.

        Runs simulation for 100ms after S2 and checks for AP.
        """
        duration = 100.0
        n_steps = int(duration / self.dt)

        v_max = state[0, 0].item()
        v_start = v_max

        for _ in range(n_steps):
            state = self.ionic.step(state, self.dt, None)
            v_max = max(v_max, state[0, 0].item())

        amplitude = v_max - v_start
        return amplitude > self.config.min_ap_amplitude


class TissueERP:
    """
    Measures ERP in 1D cable using S1-S2 protocol.

    Tissue ERP is longer than single-cell ERP due to source-sink effects:
    - During early repolarization, the stimulus must overcome the
      electrotonic load from still-refractory neighboring cells
    - Typical ratio: ERP_tissue / APD ≈ 1.17

    The tissue ERP determines the minimum wavelength for reentry:
        λ = CV × ERP_tissue
    """

    def __init__(
        self,
        config: S1S2Config,
        cable_config: CableConfig,
        D: float,
        cell_type: int = 0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize tissue ERP measurement.

        Args:
            config: S1-S2 protocol configuration
            cable_config: Cable geometry configuration
            D: Diffusion coefficient (cm²/ms)
            cell_type: Cell type (0=ENDO, 1=EPI, 2=M_CELL)
            device: Torch device
        """
        self.config = config
        self.cable_config = cable_config
        self.D = D
        self.cell_type = cell_type
        self.device = device or torch.device('cpu')

    def measure(self, verbose: bool = False) -> ERPResult:
        """
        Run S1-S2 protocol on 1D cable to measure tissue ERP.

        The S2 stimulus is applied at the same end as S1.
        ERP is the minimum interval where S2 can initiate propagating wave.

        Returns:
            ERPResult with tissue ERP, APD90, etc.
        """
        cfg = self.config
        cable_cfg = self.cable_config

        # === Phase 1: S1 pacing to steady state ===
        if verbose:
            print("Phase 1: S1 pacing on cable...")

        cable = Cable1D(cable_cfg, D=self.D, cell_type=self.cell_type, device=self.device)
        s1_apd90 = 0.0

        for beat in range(cfg.s1_count):
            result = cable.run(duration=cfg.s1_bcl, verbose=False)
            s1_apd90 = result.apd90
            cable.time = 0.0  # Reset time but keep state

            if verbose and (beat == 0 or beat == cfg.s1_count - 1):
                print(f"  Beat {beat + 1}: APD90 = {result.apd90:.1f} ms, CV = {result.cv:.4f} cm/ms")

        # Store steady-state
        steady_states = cable.states.clone()
        cv = result.cv

        # === Phase 2: S1-S2 protocol ===
        if verbose:
            print("\nPhase 2: S1-S2 protocol on cable...")

        last_successful = cfg.s2_start
        erp = cfg.s2_start

        # Coarse search
        s2_interval = cfg.s2_start
        while s2_interval >= cfg.s2_end:
            # Reset to steady state
            cable = Cable1D(cable_cfg, D=self.D, cell_type=self.cell_type, device=self.device)
            cable.states = steady_states.clone()

            # Apply S1 and wait
            self._run_s1_s2(cable, s2_interval)

            # Check if propagation occurred
            propagated = self._check_propagation(cable)

            if verbose:
                status = "propagated" if propagated else "blocked"
                print(f"  S2 = {s2_interval:.1f} ms: {status}")

            if propagated:
                last_successful = s2_interval
                s2_interval -= cfg.s2_step
            else:
                break

        # Fine search
        s2_interval = last_successful
        while s2_interval >= max(cfg.s2_end, last_successful - cfg.s2_step):
            cable = Cable1D(cable_cfg, D=self.D, cell_type=self.cell_type, device=self.device)
            cable.states = steady_states.clone()

            self._run_s1_s2(cable, s2_interval)
            propagated = self._check_propagation(cable)

            if propagated:
                erp = s2_interval

            s2_interval -= cfg.s2_fine_step

        if verbose:
            print(f"\nERP (tissue) = {erp:.1f} ms")
            print(f"APD90 = {s1_apd90:.1f} ms")
            print(f"ERP/APD ratio = {erp / s1_apd90:.2f}")
            print(f"CV = {cv:.4f} cm/ms")
            wavelength = cv * erp
            print(f"Wavelength = {wavelength:.2f} cm")

        return ERPResult(
            erp=erp,
            apd90=s1_apd90,
            last_successful_s2=last_successful,
            success=True
        )

    def _run_s1_s2(self, cable: Cable1D, s2_interval: float):
        """Apply S1, wait, then apply S2 stimulus."""
        cfg = self.config

        # Apply S1 (using cable's built-in stimulus)
        n_s1 = int(cfg.stim_duration / cable.config.dt)
        for _ in range(n_s1):
            cable.step()

        # Wait for S2 interval (minus S1 duration)
        n_wait = int((s2_interval - cfg.stim_duration) / cable.config.dt)
        for _ in range(n_wait):
            cable.step()

        # Reset activation tracking for S2
        cable.activated = torch.zeros(cable.n_cells, dtype=torch.bool, device=cable.device)
        cable.activation_times = torch.full(
            (cable.n_cells,), float('inf'), device=cable.device, dtype=torch.float64
        )

        # Apply S2 (same location as S1)
        # Manually apply stimulus current
        for _ in range(n_s1):
            I_stim = torch.zeros(cable.n_cells, device=cable.device, dtype=torch.float64)
            I_stim[:cable.config.stim_width] = cfg.stim_amplitude

            # Ionic step with stimulus (ORdModel.step(states, dt, Istim))
            cable.states = cable.ionic.step(
                cable.states,
                cable.config.dt,
                I_stim
            )

            # Diffusion step
            V = cable.states[:, 0].clone()
            V_new = cable._solve_diffusion(V)
            cable.states[:, 0] = V_new

            # Track activation after S2
            V_current = cable.states[:, 0]
            newly_activated = (
                (V_current > cable.config.activation_threshold) &
                (~cable.activated)
            )
            cable.activation_times[newly_activated] = cable.time
            cable.activated |= newly_activated

            cable.time += cable.config.dt

        # Run for 200ms to allow propagation
        n_run = int(200.0 / cable.config.dt)
        for _ in range(n_run):
            cable.step()

    def _check_propagation(self, cable: Cable1D) -> bool:
        """
        Check if S2 stimulus initiated propagating wave.

        Criterion: Far end of cable must activate.
        """
        # Check if distal end activated
        far_idx = cable.n_cells - 10  # 10 cells from far end
        activation_time = cable.activation_times[far_idx].item()

        return not np.isinf(activation_time)


def measure_erp_single_cell(
    dt: float = 0.02,
    cell_type: int = 0,
    s1_bcl: float = 1000.0,
    s1_count: int = 10,
    verbose: bool = False,
    device: Optional[torch.device] = None
) -> ERPResult:
    """
    Convenience function to measure single-cell ERP.

    Args:
        dt: Time step (ms)
        cell_type: Cell type (0=ENDO, 1=EPI, 2=M_CELL)
        s1_bcl: S1 pacing rate (ms)
        s1_count: Number of S1 beats
        verbose: Print progress
        device: Torch device

    Returns:
        ERPResult with single-cell ERP
    """
    config = S1S2Config(s1_bcl=s1_bcl, s1_count=s1_count)
    erp_measure = SingleCellERP(config, dt=dt, cell_type=cell_type, device=device)
    return erp_measure.measure(verbose=verbose)


def measure_erp_tissue(
    D: float,
    dx: float = 0.01,
    dt: float = 0.02,
    cable_length: float = 2.0,
    cell_type: int = 0,
    s1_bcl: float = 1000.0,
    s1_count: int = 10,
    verbose: bool = False,
    device: Optional[torch.device] = None
) -> ERPResult:
    """
    Convenience function to measure tissue ERP.

    Args:
        D: Diffusion coefficient (cm²/ms)
        dx: Mesh spacing (cm)
        dt: Time step (ms)
        cable_length: Cable length (cm)
        cell_type: Cell type (0=ENDO, 1=EPI, 2=M_CELL)
        s1_bcl: S1 pacing rate (ms)
        s1_count: Number of S1 beats
        verbose: Print progress
        device: Torch device

    Returns:
        ERPResult with tissue ERP
    """
    s1s2_config = S1S2Config(s1_bcl=s1_bcl, s1_count=s1_count)
    cable_config = CableConfig(length=cable_length, dx=dx, dt=dt)

    erp_measure = TissueERP(
        s1s2_config, cable_config, D=D,
        cell_type=cell_type, device=device
    )
    return erp_measure.measure(verbose=verbose)


if __name__ == "__main__":
    print("=" * 60)
    print("ERP Measurement Test")
    print("=" * 60)

    # Single-cell ERP
    print("\n--- Single-Cell ERP ---")
    result_sc = measure_erp_single_cell(verbose=True)

    # Tissue ERP
    print("\n--- Tissue ERP ---")
    D = 0.001  # cm²/ms
    result_tissue = measure_erp_tissue(D=D, verbose=True)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Single-cell ERP = {result_sc.erp:.1f} ms")
    print(f"  Tissue ERP = {result_tissue.erp:.1f} ms")
    print(f"  Ratio (tissue/single) = {result_tissue.erp / result_sc.erp:.2f}")
