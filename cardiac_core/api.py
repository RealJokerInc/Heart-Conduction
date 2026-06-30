"""
Simplified Cardiac Simulation API.

Provides monodomain(), bidomain(), and lbm() functions that accept either
a file path or CardiacMeshData and return a CardiacSimulation wrapper
with a uniform generator interface.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import numpy as np
import torch

from .file_format import CardiacMeshData, load_cardiac_mesh
from .grid import Grid
from .conductivity import ConductivityConfig

# cardiac_core is now self-contained: the three engines are vendored under cardiac_core/_monodomain,
# _bidomain, _lbm — no sys.modules engine-path hack, no cross-folder imports. (The original engine
# folders remain on disk, frozen; cardiac_core is the centralized home going forward.)


@dataclass
class Distribution:
    """Per-node stochastic parameter specification.

    Used anywhere a scalar parameter can instead be drawn per-node from a
    probability distribution. The simulation generates (Nx, Ny) samples at
    apply time and stores them as a frozen field.

    Parameters
    ----------
    kind : str
        Distribution type: 'constant', 'gaussian', 'uniform', 'lognormal'.
    kwargs : dict
        Distribution parameters:
        - constant:  {value: float}
        - gaussian:  {mean: float, sigma: float}
        - uniform:   {lower: float, upper: float}
        - lognormal: {mean: float, sigma: float}  (of underlying normal)

    Examples
    --------
    >>> Distribution('gaussian', mean=1.0, sigma=0.1)
    >>> Distribution('uniform', lower=0.0005, upper=0.0015)
    >>> Distribution('constant', value=0.001)
    >>> Distribution('lognormal', mean=0.0, sigma=0.3)
    """
    kind: str
    kwargs: dict

    def __init__(self, kind: str, **kwargs):
        self.kind = kind
        self.kwargs = kwargs

    def sample(self, shape: tuple, device: str = 'cpu', dtype=torch.float64) -> torch.Tensor:
        """Draw samples of given shape.

        Parameters
        ----------
        shape : tuple
            Output shape, typically (Nx, Ny).
        device : str
            Torch device.
        dtype : torch.dtype
            Output dtype.

        Returns
        -------
        torch.Tensor
            Sampled values.
        """
        if self.kind == 'constant':
            return torch.full(shape, self.kwargs['value'], device=device, dtype=dtype)
        elif self.kind == 'gaussian':
            mean = self.kwargs['mean']
            sigma = self.kwargs['sigma']
            return torch.normal(mean, sigma, size=shape, device=device, dtype=dtype)
        elif self.kind == 'uniform':
            lo = self.kwargs['lower']
            hi = self.kwargs['upper']
            return torch.empty(shape, device=device, dtype=dtype).uniform_(lo, hi)
        elif self.kind == 'lognormal':
            mean = self.kwargs['mean']
            sigma = self.kwargs['sigma']
            normal = torch.normal(mean, sigma, size=shape, device=device, dtype=dtype)
            return normal.exp()
        else:
            raise ValueError(f"Unknown distribution: {self.kind}")


@dataclass
class SimulationSnapshot:
    """Uniform return type from all simulation engines.

    Attributes
    ----------
    t : float
        Current simulation time (ms).
    Vm : torch.Tensor
        Membrane potential, shape (Nx, Ny). (Canonical name; ``.V`` is a read-only alias.)
    phi_e : torch.Tensor | None
        Extracellular potential (Nx, Ny), bidomain only.
    Nx, Ny : int
        Grid dimensions.
    dx, dy : float
        Grid spacing (cm).
    """
    t: float
    Vm: torch.Tensor
    phi_e: Optional[torch.Tensor]
    Nx: int
    Ny: int
    dx: float
    dy: float
    ionic_states: Optional[torch.Tensor] = None   # (n_states, Nx, Ny), opt-in via record=

    @property
    def V(self) -> torch.Tensor:
        """Read-only deprecated alias for :attr:`Vm`."""
        return self.Vm


class CardiacSimulation:
    """Uniform wrapper around all cardiac simulation engines.

    Parameters
    ----------
    engine : object
        The underlying engine (MonodomainSimulation, BidomainSimulation, or LBMSimulation).
    engine_type : str
        'monodomain', 'bidomain', or 'lbm'.
    grid : object
        StructuredGrid (for monodomain/bidomain) or None (for LBM).
    data : CardiacMeshData
        The mesh data used to construct this simulation.
    """

    def __init__(self, engine, engine_type: str, grid, data: CardiacMeshData,
                 build_kwargs: Optional[dict] = None):
        self._engine = engine
        self._engine_type = engine_type
        self._grid = grid
        self._data = data
        # Engine-construction knobs (dt/splitting/solver/device/...) — enough to replay the
        # factory with `mesh=self._data` for reset()/with_()/stimulate(). The geometry,
        # conductivity, and stimulus are already baked into `data`.
        self._build_kwargs = dict(build_kwargs or {})
        self._Nx = data.mask.shape[0]
        self._Ny = data.mask.shape[1]
        self._probes: dict[str, dict] = {}   # name → {x, y, ix, iy, t[], V[]}
        self._clamp_mask: Optional[torch.Tensor] = None  # (Nx, Ny) bool
        self._clamp_voltage: Optional[float] = None

    # ========================================================================
    # Core simulation control
    # ========================================================================

    def run(self, t_end: float, save_every: float = 1.0, *, batch: Optional[int] = None,
            record=("Vm",), callback=None) -> "SimulationResult | Iterator[SimulationResult]":
        """Run to ``t_end`` and return a :class:`~cardiac_core.run.SimulationResult`.

        Eager by default (``batch=None``): drains the run, returns ONE ``SimulationResult`` with
        ``Vm (T,Nx,Ny)`` (+ ``phi_e`` for bidomain, ``ionic_states`` if requested). With ``batch=k``,
        returns an ``Iterator[SimulationResult]`` yielding chunks of ≤k save-points (k=1 = frame-by-frame).

        Parameters
        ----------
        t_end : float
            End time (ms).
        save_every : float
            Save interval (ms).
        batch : int, optional
            Chunk size for streaming. ``None`` = eager single result.
        record : tuple[str, ...]
            Fields to collect. ``"Vm"`` always; ``"ionic_states"`` opt-in (NotImplemented for LBM).
        callback : callable, optional
            ``callback(snapshot)`` per save-point; returning ``False`` stops early (eager mode).

        Returns
        -------
        SimulationResult | Iterator[SimulationResult]
        """
        it = self._iter_snapshots(t_end, save_every, record=record, callback=callback)
        if batch is None:
            return _result_from(list(it), record, self.dx, self.dy)

        def _gen():
            buf = []
            for snap in it:
                buf.append(snap)
                if len(buf) == batch:
                    yield _result_from(buf, record, self.dx, self.dy)
                    buf = []
            if buf:
                yield _result_from(buf, record, self.dx, self.dy)
        return _gen()

    def snapshots(self, t_end: float, save_every: float = 1.0, *, record=("Vm",),
                  callback=None) -> Iterator[SimulationSnapshot]:
        """Frame-by-frame generator of :class:`SimulationSnapshot` (the pre-Phase-5 ``run()``).

        Use this where you want to iterate snapshots lazily; ``run()`` is now eager and returns a
        ``SimulationResult``.
        """
        return self._iter_snapshots(t_end, save_every, record=record, callback=callback)

    def _iter_snapshots(self, t_end, save_every, *, record=("Vm",), callback=None):
        if self._engine_type == 'monodomain':
            gen = self._run_monodomain(t_end, save_every, record)
        elif self._engine_type == 'bidomain':
            gen = self._run_bidomain(t_end, save_every, record)
        elif self._engine_type == 'lbm':
            gen = self._run_lbm(t_end, save_every, record)
        else:
            raise ValueError(f"unknown engine {self._engine_type!r}")
        for snap in gen:
            yield snap
            if callback is not None and callback(snap) is False:
                break

    def step(self):
        """Advance simulation by one time step."""
        if self._engine_type == 'lbm':
            self._engine.step()
        else:
            self._engine.step()

    def reset(self):
        """Reset to t=0 by rebuilding the engine from the stored construction record.

        Works for BOTH the declarative and legacy ``mesh=`` paths: the engine is replayed from
        ``self._data`` (which carries geometry + conductivity + the current ``stimuli``).
        """
        fresh = _factory_for(self._engine_type)(mesh=self._data, **self._build_kwargs)
        self._engine = fresh._engine
        self._grid = fresh._grid

    def with_(self, **overrides) -> 'CardiacSimulation':
        """Functional CHANGE: return a NEW simulation with ``overrides`` applied; ``self`` untouched.

        Overrides are factory keyword args (e.g. ``dt``, ``splitting``, ``ionic_model``, ``device``).
        Immutable / sweep-safe — no mutation of the receiver.
        """
        kwargs = {**self._build_kwargs, **overrides}
        return _factory_for(self._engine_type)(mesh=self._data, **kwargs)

    # ========================================================================
    # State access — read
    # ========================================================================

    @property
    def Vm(self) -> torch.Tensor:
        """Current membrane potential as (Nx, Ny) grid."""
        if self._engine_type == 'lbm':
            return self._engine.V
        V_flat = self._engine.state.V
        return self._grid.flat_to_grid(V_flat)

    @property
    def V(self) -> torch.Tensor:
        """Read-only deprecated alias for :attr:`Vm`."""
        return self.Vm

    @property
    def phi_e(self) -> Optional[torch.Tensor]:
        """Current extracellular potential as (Nx, Ny) grid. Bidomain only."""
        if self._engine_type == 'bidomain':
            return self._grid.flat_to_grid(self._engine.state.phi_e)
        return None

    @property
    def t(self) -> float:
        """Current simulation time (ms)."""
        if self._engine_type == 'lbm':
            return self._engine.t
        return self._engine.state.t

    @property
    def ionic_states(self) -> torch.Tensor:
        """All ionic state variables as (Nx, Ny, n_states) grid."""
        raise NotImplementedError

    def get_state(self, name: str) -> torch.Tensor:
        """Get a single ionic state variable by name as (Nx, Ny) grid.

        Parameters
        ----------
        name : str
            State variable name (e.g. 'Ca_i', 'Na_i', 'm', 'h', 'j').
        """
        raise NotImplementedError

    @property
    def state_names(self) -> list[str]:
        """Names of all ionic state variables."""
        raise NotImplementedError

    # ========================================================================
    # State access — write
    # ========================================================================

    def set_voltage(self, V: torch.Tensor):
        """Override membrane potential everywhere.

        Parameters
        ----------
        V : torch.Tensor
            (Nx, Ny) voltage field.
        """
        raise NotImplementedError

    def set_state(self, name: str, values: torch.Tensor):
        """Override a single ionic state variable.

        Parameters
        ----------
        name : str
            State variable name.
        values : torch.Tensor
            (Nx, Ny) field.
        """
        raise NotImplementedError

    # ========================================================================
    # Stimulus / current injection
    # ========================================================================

    def _grid_coords(self):
        """Grid-shaped ``(x, y)`` coords ``(Nx, Ny)`` matching the engine ``ij`` convention."""
        x1d = torch.linspace(0.0, self._data.dx * (self._Nx - 1), self._Nx, dtype=torch.float64)
        y1d = torch.linspace(0.0, self._data.dy * (self._Ny - 1), self._Ny, dtype=torch.float64)
        return torch.meshgrid(x1d, y1d, indexing='ij')

    def stimulate(self, region, start_time: float = 0.0, duration: float = 1.0,
                  amplitude: float = -52.0):
        """STIMULATE idiom: append a stimulus and rebuild (so it takes effect from t=0).

        ``region`` is a callable ``(x, y) -> bool mask`` or an ``(Nx, Ny)`` mask. Stored on
        ``self._data.stimuli`` (the path both declarative and legacy construction share), so this
        works identically regardless of how the sim was built.
        """
        entry = _normalize_stimulus(
            {'region': region, 'start_time': start_time, 'duration': duration,
             'amplitude': amplitude},
            self._grid_coords(),
        )[0]
        entry['mask'] = entry['mask'] & self._data.mask
        self._data.stimuli.append(entry)
        self.reset()

    def add_stimulus(
        self,
        mask: 'np.ndarray | torch.Tensor',
        start_time: float,
        duration: float,
        amplitude: float = -80.0,
    ):
        """Add a stimulus region. Thin alias over :meth:`stimulate` (mask form)."""
        self.stimulate(mask, start_time=start_time, duration=duration, amplitude=amplitude)

    def add_pacing(
        self,
        mask: 'np.ndarray | torch.Tensor',
        bcl: float = 1000.0,
        n_beats: int = 10,
        start_time: float = 0.0,
        duration: float = 2.0,
        amplitude: float = -80.0,
    ):
        """Add a regular pacing protocol.

        Parameters
        ----------
        mask : (Nx, Ny) bool
            Pacing site.
        bcl : float
            Basic cycle length (ms).
        n_beats : int
            Number of beats.
        start_time : float
            First beat time (ms).
        duration : float
            Each stimulus duration (ms).
        amplitude : float
            Stimulus current (µA/µF).
        """
        raise NotImplementedError

    def inject_current(self, mask: 'np.ndarray | torch.Tensor', amplitude: float):
        """Inject current for one time step at the masked nodes.

        Applied immediately at the next step(), then removed.

        Parameters
        ----------
        mask : (Nx, Ny) bool
            Where to inject.
        amplitude : float
            Current (µA/µF).
        """
        raise NotImplementedError

    # ========================================================================
    # Voltage clamp
    # ========================================================================

    def clamp_voltage(
        self,
        mask: 'np.ndarray | torch.Tensor',
        voltage: float,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
    ):
        """Hold nodes at fixed voltage. Ionic currents still computed.

        Clamped nodes have their V reset to `voltage` after every step.
        Use for patch-clamp-like experiments (study ionic currents in isolation).

        Parameters
        ----------
        mask : (Nx, Ny) bool
            Nodes to clamp.
        voltage : float
            Clamp voltage (mV).
        start_time : float, optional
            When clamp activates (ms). None = immediately.
        duration : float, optional
            How long clamp lasts (ms). None = until release_clamp().
        """
        raise NotImplementedError

    def add_clamp_protocol(
        self,
        mask: 'np.ndarray | torch.Tensor',
        steps: list[tuple[float, float]],
        start_time: float = 0.0,
    ):
        """Add a multi-step voltage clamp protocol.

        Applies a sequence of (voltage, duration) steps. Classic
        electrophysiology protocol for studying ionic current kinetics.

        Parameters
        ----------
        mask : (Nx, Ny) bool
            Nodes to clamp.
        steps : list of (voltage_mV, duration_ms)
            Sequential clamp steps. e.g.:
            [(-80, 500), (-40, 200), (-80, 500)]  # hold → step → hold
        start_time : float
            When the first step begins (ms).

        Example
        -------
        >>> # Classic INa activation protocol
        >>> sim.add_clamp_protocol(
        ...     mask, steps=[(-80, 500), (-20, 50), (-80, 500)], start_time=0
        ... )
        """
        raise NotImplementedError

    def release_clamp(self):
        """Release all voltage clamps. Nodes resume normal dynamics."""
        raise NotImplementedError

    # ========================================================================
    # Conductivity / drug block
    # ========================================================================

    def scale_conductance(
        self,
        current_name: str,
        factor: 'float | Distribution',
        mask: 'np.ndarray | torch.Tensor | None' = None,
    ):
        """Scale an ionic current's maximal conductance.

        Simulates drug block (factor < 1.0) or upregulation (factor > 1.0).
        Accepts a Distribution for per-node stochastic scaling (cell-to-cell
        variability in ion channel expression).

        Parameters
        ----------
        current_name : str
            Ionic current name (e.g. 'IKr', 'INa', 'ICaL', 'IK1').
        factor : float or Distribution
            Multiplicative factor. float → uniform everywhere.
            Distribution → per-node sampling (e.g. gaussian(1.0, 0.1)).
        mask : (Nx, Ny) bool, optional
            Apply only in region. None = everywhere.

        Examples
        --------
        >>> sim.scale_conductance('IKr', 0.5)                         # 50% block everywhere
        >>> sim.scale_conductance('INa', Distribution('gaussian', mean=1.0, sigma=0.1))  # ±10% per node
        >>> sim.scale_conductance('ICaL', 0.0, mask=scar_mask)        # full block in scar only
        """
        raise NotImplementedError

    def set_conductivity(
        self,
        mask: 'np.ndarray | torch.Tensor',
        D: 'float | Distribution',
    ):
        """Set diffusion coefficient in a region.

        Creates scar (D=0), slow conduction zones, or heterogeneous tissue.

        Parameters
        ----------
        mask : (Nx, Ny) bool
            Region to modify.
        D : float or Distribution
            New diffusion coefficient (cm²/ms). 0 = no conduction.
            Distribution → per-node sampling.

        Examples
        --------
        >>> sim.set_conductivity(scar_mask, D=0.0)                          # scar
        >>> sim.set_conductivity(border_mask, Distribution('uniform', lower=0.0003, upper=0.0008))
        """
        raise NotImplementedError

    def scale_conductivity(
        self,
        mask: 'np.ndarray | torch.Tensor',
        factor: 'float | Distribution',
    ):
        """Scale diffusion coefficient in a region.

        Parameters
        ----------
        mask : (Nx, Ny) bool
            Region to modify.
        factor : float or Distribution
            Multiplicative factor.

        Examples
        --------
        >>> sim.scale_conductivity(fibrotic_mask, Distribution('lognormal', mean=0.0, sigma=0.5))
        """
        raise NotImplementedError

    # ========================================================================
    # Per-node stochastic parameters
    # ========================================================================

    def set_parameter(
        self,
        name: str,
        value: 'float | Distribution',
        mask: 'np.ndarray | torch.Tensor | None' = None,
    ):
        """Set any named model parameter, optionally with per-node distribution.

        This is the general-purpose interface for introducing heterogeneity.
        Specific methods (scale_conductance, set_conductivity) are convenience
        wrappers around this.

        Parameters
        ----------
        name : str
            Parameter name. Engine-specific, but common ones include:
            - Ionic conductances: 'GNa', 'GKr', 'GCaL', 'GK1', 'Gto', ...
            - Ionic concentrations: 'Na_o', 'K_o', 'Ca_o'
            - Tissue: 'D', 'chi', 'Cm'
        value : float or Distribution
            Scalar → uniform. Distribution → per-node sampling.
        mask : (Nx, Ny) bool, optional
            Apply only in region. None = everywhere.

        Examples
        --------
        >>> # Heterogeneous GNa (±15% per cell)
        >>> sim.set_parameter('GNa', Distribution('gaussian', mean=14.838, sigma=2.2))
        >>> # Elevated extracellular potassium in ischemic zone
        >>> sim.set_parameter('K_o', 8.0, mask=ischemia_mask)
        >>> # Random Cm variation
        >>> sim.set_parameter('Cm', Distribution('uniform', lower=0.8, upper=1.2))
        """
        raise NotImplementedError

    def get_parameter_field(self, name: str) -> torch.Tensor:
        """Get the current per-node values of a parameter as (Nx, Ny) grid.

        After set_parameter with a Distribution, this returns the frozen
        sampled values (what each node actually got).

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        torch.Tensor
            (Nx, Ny) current values.
        """
        raise NotImplementedError

    # ========================================================================
    # Probes / recording
    # ========================================================================

    def add_probe(self, name: str, x: float, y: float):
        """Register a probe point for time-series recording.

        Probes record V (and phi_e for bidomain) at every step during run().

        Parameters
        ----------
        name : str
            Probe identifier (e.g. 'apex', 'base', 'scar_border').
        x, y : float
            Physical coordinates (cm).
        """
        raise NotImplementedError

    def get_traces(self) -> dict:
        """Return recorded probe traces.

        Returns
        -------
        dict
            {name: {'t': np.ndarray, 'V': np.ndarray, 'phi_e': np.ndarray | None}}
        """
        raise NotImplementedError

    def clear_traces(self):
        """Clear all recorded probe data (keeps probe locations)."""
        raise NotImplementedError

    # ========================================================================
    # Analysis
    # ========================================================================

    def compute_activation_time(self, threshold: float = -20.0) -> torch.Tensor:
        """Compute activation time map from last run().

        Parameters
        ----------
        threshold : float
            Voltage threshold for activation (mV).

        Returns
        -------
        torch.Tensor
            (Nx, Ny) activation times (ms). NaN where not activated.
        """
        raise NotImplementedError

    def compute_cv(
        self,
        x1: float, x2: float, y: float,
        threshold: float = -20.0,
    ) -> float:
        """Measure conduction velocity between two x-positions.

        Parameters
        ----------
        x1, x2 : float
            Measurement points (cm). x2 > x1.
        y : float
            Row position (cm).
        threshold : float
            Activation threshold (mV).

        Returns
        -------
        float
            Conduction velocity (cm/s). NaN if activation not detected.
        """
        raise NotImplementedError

    def compute_apd(
        self,
        x: float, y: float,
        repol: float = 0.9,
    ) -> float:
        """Compute action potential duration at a point.

        Parameters
        ----------
        x, y : float
            Measurement point (cm).
        repol : float
            Repolarization fraction (0.9 = APD90, 0.5 = APD50).

        Returns
        -------
        float
            APD (ms). NaN if no complete AP detected.
        """
        raise NotImplementedError

    # ========================================================================
    # Metadata
    # ========================================================================

    @property
    def Nx(self) -> int:
        """Grid dimension in x."""
        return self._Nx

    @property
    def Ny(self) -> int:
        """Grid dimension in y."""
        return self._Ny

    @property
    def dx(self) -> float:
        """Grid spacing in x (cm)."""
        return self._data.dx

    @property
    def dy(self) -> float:
        """Grid spacing in y (cm)."""
        return self._data.dy

    @property
    def mask(self) -> np.ndarray:
        """Domain mask (Nx, Ny) bool."""
        return self._data.mask

    @property
    def engine_type(self) -> str:
        """Engine type: 'monodomain', 'bidomain', or 'lbm'."""
        return self._engine_type

    @property
    def dt(self) -> float:
        """Time step (ms)."""
        return getattr(self._engine, 'dt', None) or self._data.dt

    @property
    def Cm(self) -> float:
        """Membrane capacitance (µF/cm²)."""
        return self._data.Cm

    @property
    def ionic_model(self) -> str:
        """Ionic model name (e.g. 'ttp06')."""
        return self._data.ionic_model

    # --- Private generators ---

    def _grid_ionic(self, state):
        """Grid the engine's flat ionic_states (n_dof, n_states) → (n_states, Nx, Ny)."""
        S = state.ionic_states
        return torch.stack([self._grid.flat_to_grid(S[:, k]) for k in range(S.shape[1])])

    def _run_monodomain(self, t_end, save_every, record=("Vm",)):
        want_ionic = "ionic_states" in record
        for state in self._engine.run(t_end, save_every):
            V_grid = self._grid.flat_to_grid(state.V)
            yield SimulationSnapshot(
                t=state.t,
                Vm=V_grid,
                phi_e=None,
                Nx=self._Nx,
                Ny=self._Ny,
                dx=self._data.dx,
                dy=self._data.dy,
                ionic_states=self._grid_ionic(state) if want_ionic else None,
            )

    def _run_bidomain(self, t_end, save_every, record=("Vm",)):
        want_ionic = "ionic_states" in record
        for state in self._engine.run(t_end, save_every):
            V_grid = self._grid.flat_to_grid(state.Vm)
            phi_e_grid = self._grid.flat_to_grid(state.phi_e)
            yield SimulationSnapshot(
                t=state.t,
                Vm=V_grid,
                phi_e=phi_e_grid,
                Nx=self._Nx,
                Ny=self._Ny,
                dx=self._data.dx,
                dy=self._data.dy,
                ionic_states=self._grid_ionic(state) if want_ionic else None,
            )

    def _run_lbm(self, t_end, save_every, record=("Vm",)):
        if "ionic_states" in record:
            raise NotImplementedError(
                "ionic_states recording is not supported for the LBM engine "
                "(gates live on the sim object, no uniform per-node container)."
            )
        dt = self._engine.dt
        save_interval = max(1, int(round(save_every / dt)))
        step_count = 0
        while self._engine.t < t_end - 1e-12:
            self._engine.step()
            step_count += 1
            if step_count % save_interval == 0:
                yield SimulationSnapshot(
                    t=self._engine.t,
                    Vm=self._engine.V.clone(),
                    phi_e=None,
                    Nx=self._Nx,
                    Ny=self._Ny,
                    dx=self._data.dx,
                    dy=self._data.dy,
                )


def _resolve_mesh(mesh: Union[str, CardiacMeshData]) -> CardiacMeshData:
    """Accept path or CardiacMeshData, return CardiacMeshData."""
    if isinstance(mesh, (str, Path)):
        return load_cardiac_mesh(str(mesh))
    return mesh


def _normalize_stimulus(stimulus, coords) -> list:
    """Convert a declarative ``stimulus`` arg into CardiacMeshData ``stimuli`` dicts.

    Accepts ``None``, a single dict, or a list of dicts. Each dict carries a ``region``
    (a callable ``(x, y) -> bool mask`` evaluated on the grid coordinates, OR an ``(Nx, Ny)``
    array/mask) plus optional ``start_time``/``duration``/``amplitude``/``label``/``bcl``/``num_pulses``.
    """
    if stimulus is None:
        return []
    if isinstance(stimulus, dict):
        stimulus = [stimulus]
    out = []
    for s in stimulus:
        region = s.get('region', s.get('mask'))
        if region is None:
            raise ValueError("each stimulus needs a 'region' (callable or (Nx,Ny) mask)")
        if callable(region):
            x, y = coords
            mask = region(x, y)
        else:
            mask = region
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        mask = np.asarray(mask).astype(bool)
        out.append({
            'mask': mask,
            'label': s.get('label', 'stim'),
            'amplitude': s.get('amplitude', -52.0),
            'duration': s.get('duration', 1.0),
            'start_time': s.get('start_time', 0.0),
            'bcl': s.get('bcl', 0.0),
            'num_pulses': s.get('num_pulses', 1),
        })
    return out


def _build_mesh_data(geometry, ionic_model, conductivity, stimulus, dt, engine: str) -> CardiacMeshData:
    """Assemble a CardiacMeshData from (Grid, ionic_model, ConductivityConfig, stimulus).

    Conductivity is mapped per the target engine (see Step 4.0):
    - monodomain: ``for_monodomain()`` -> D=sigma_eff/chi, engine chi=1, real Cm (Form A).
    - bidomain:   RAW sigma tuples ``(σ,σ,0)`` + real chi/Cm (the factory does σ→D internally; Form B).
    - lbm:        ``for_lbm()`` -> D=D_eff (fully scaled), real Cm (Form B).
    """
    if not isinstance(geometry, Grid):
        raise TypeError(
            "geometry must be a Grid (structured-only). Pass a CardiacMeshData/path as `mesh=` "
            "for the legacy path."
        )
    if conductivity is None or not isinstance(conductivity, ConductivityConfig):
        raise ValueError("declarative construction requires conductivity=ConductivityConfig(...)")

    Nx, Ny = geometry.Nx, geometry.Ny
    dx, dy = geometry.dx, geometry.dy
    if geometry.mask is not None:
        mask = geometry.mask.cpu().numpy().astype(bool)
    else:
        mask = np.ones((Nx, Ny), dtype=bool)

    sigma_i = sigma_e = None
    if engine == 'monodomain':
        emit = conductivity.for_monodomain()
        D, chi, Cm = emit['D'], emit['chi'], emit['Cm']
    elif engine == 'lbm':
        emit = conductivity.for_lbm()
        D, chi, Cm = emit['D'], conductivity.chi, emit['Cm']
    elif engine == 'bidomain':
        if conductivity.sigma_i is None or conductivity.sigma_e is None:
            raise ValueError(
                "bidomain construction needs ConductivityConfig.bidomain(sigma_i, sigma_e, ...)"
            )
        # sigma_* are (xx, yy, xy) FIELDS of shape (Nx, Ny) — the bidomain FDM indexes them [i,j].
        si = float(conductivity.sigma_i)
        se = float(conductivity.sigma_e)
        zeros = np.zeros((Nx, Ny), dtype=np.float64)
        sigma_i = (np.full((Nx, Ny), si), np.full((Nx, Ny), si), zeros)
        sigma_e = (np.full((Nx, Ny), se), np.full((Nx, Ny), se), zeros.copy())
        D, chi, Cm = conductivity.D_eff, conductivity.chi, conductivity.Cm
    else:
        raise ValueError(f"unknown engine {engine!r}")

    if isinstance(D, tuple):
        Dxx, Dyy, Dxy = D
    else:
        Dxx = Dyy = D
        Dxy = 0.0
    D_xx = np.full((Nx, Ny), Dxx, dtype=np.float64)
    D_yy = np.full((Nx, Ny), Dyy, dtype=np.float64)
    D_xy = np.full((Nx, Ny), Dxy, dtype=np.float64)

    stimuli = _normalize_stimulus(stimulus, geometry.coordinates)
    for s in stimuli:
        s['mask'] = s['mask'] & mask  # intersect with tissue

    return CardiacMeshData(
        dx=dx, dy=dy, mask=mask,
        D_xx=D_xx, D_yy=D_yy, D_xy=D_xy,
        chi=chi, Cm=Cm,
        ionic_model=ionic_model or 'ttp06',
        dt=dt if dt is not None else 0.02,
        stimuli=stimuli,
        sigma_i=sigma_i, sigma_e=sigma_e,
    )


def _factory_for(engine_type: str):
    """Return the public factory function for an engine type (for reset/with_ replay)."""
    return {'monodomain': monodomain, 'bidomain': bidomain, 'lbm': lbm}[engine_type]


def _result_from(snaps, record, dx, dy):
    """Stack a list of SimulationSnapshot into a single SimulationResult."""
    from .run import SimulationResult  # local import avoids api<->run circular import
    if not snaps:
        empty = torch.empty(0)
        return SimulationResult(times=empty, Vm=empty, phi_e=None, dx=dx, dy=dy)
    times = torch.tensor([s.t for s in snaps], dtype=torch.float64)
    Vm = torch.stack([s.Vm for s in snaps])
    phi_e = torch.stack([s.phi_e for s in snaps]) if snaps[0].phi_e is not None else None
    ionic = None
    if "ionic_states" in record and snaps[0].ionic_states is not None:
        ionic = torch.stack([s.ionic_states for s in snaps])
    return SimulationResult(times=times, Vm=Vm, phi_e=phi_e, dx=dx, dy=dy, ionic_states=ionic)


def _build_stimulus_protocol_v54(data: CardiacMeshData, grid, device, dtype):
    """Build the monodomain StimulusProtocol from mesh data stimuli (shared cardiac_core.stimulus)."""
    from .stimulus.protocol import StimulusProtocol

    protocol = StimulusProtocol()
    mask_np = data.mask  # (Nx, Ny) bool

    for stim in data.stimuli:
        # Intersect stimulus mask with tissue mask, flatten to active DOFs
        stim_grid = stim['mask'] & mask_np
        stim_flat = torch.tensor(
            stim_grid[mask_np] if grid.domain_mask is not None else stim_grid.flatten(),
            dtype=torch.bool,
            device=device,
        )

        bcl = stim.get('bcl', 0.0)
        num_pulses = stim.get('num_pulses', 1)

        if bcl > 0 and num_pulses > 1:
            # Multiple pulses → add each as separate stimulus
            for p in range(num_pulses):
                t_start = stim['start_time'] + p * bcl
                protocol.add_stimulus(
                    region=stim_flat,
                    start_time=t_start,
                    duration=stim['duration'],
                    amplitude=stim['amplitude'],
                )
        else:
            protocol.add_stimulus(
                region=stim_flat,
                start_time=stim['start_time'],
                duration=stim['duration'],
                amplitude=stim['amplitude'],
            )

    return protocol


def _build_stimulus_protocol_bidomain(data: CardiacMeshData, grid, device, dtype):
    """Build the bidomain StimulusProtocol from mesh data stimuli (shared cardiac_core.stimulus)."""
    from .stimulus.protocol import StimulusProtocol

    protocol = StimulusProtocol()
    mask_np = data.mask

    for stim in data.stimuli:
        stim_grid = stim['mask'] & mask_np
        stim_flat = torch.tensor(
            stim_grid[mask_np] if grid.domain_mask is not None else stim_grid.flatten(),
            dtype=torch.bool,
            device=device,
        )

        bcl = stim.get('bcl', 0.0)
        num_pulses = stim.get('num_pulses', 1)

        if bcl > 0 and num_pulses > 1:
            for p in range(num_pulses):
                t_start = stim['start_time'] + p * bcl
                protocol.add_stimulus(
                    region=stim_flat,
                    start_time=t_start,
                    duration=stim['duration'],
                    amplitude=stim['amplitude'],
                )
        else:
            protocol.add_stimulus(
                region=stim_flat,
                start_time=stim['start_time'],
                duration=stim['duration'],
                amplitude=stim['amplitude'],
            )

    return protocol


# ============================================================================
# Public API functions
# ============================================================================


def monodomain(
    geometry=None,
    ionic_model: Optional[str] = None,
    conductivity: Optional[ConductivityConfig] = None,
    stimulus=None,
    *,
    mesh: Union[str, CardiacMeshData, None] = None,
    dt: Optional[float] = None,
    splitting: str = 'strang',
    diffusion_solver: str = 'crank_nicolson',
    linear_solver: str = 'pcg',
    device: str = 'cpu',
) -> CardiacSimulation:
    """Create a monodomain simulation.

    Two construction idioms:
    - **Declarative**: ``monodomain(Grid(...), 'ttp06', ConductivityConfig.bidomain(...), stimulus)``.
    - **Legacy mesh**: ``monodomain(mesh)`` where ``mesh`` is a path or ``CardiacMeshData`` (a
      positional ``CardiacMeshData``/``str`` is auto-detected and treated as ``mesh=``).

    Parameters
    ----------
    geometry : Grid, optional
        Structured grid for declarative construction.
    ionic_model : str, optional
        Ionic model name (declarative), or override the mesh's model (legacy).
    conductivity : ConductivityConfig, optional
        Conductivity/chi/Cm for declarative construction.
    stimulus : dict or list[dict], optional
        Declarative stimulus region(s).
    mesh : str or CardiacMeshData, optional
        Legacy path/data construction.
    dt, splitting, diffusion_solver, linear_solver, device
        Solver / runtime knobs.

    Returns
    -------
    CardiacSimulation
        Wrapper with .run() generator interface.
    """
    # Back-compat type-sniff: a positional CardiacMeshData/str/path is the legacy `mesh`.
    if isinstance(geometry, (str, Path, CardiacMeshData)):
        mesh, geometry = geometry, None
    if mesh is not None:
        data = _resolve_mesh(mesh)
    else:
        data = _build_mesh_data(geometry, ionic_model, conductivity, stimulus, dt, 'monodomain')
    ionic = ionic_model or data.ionic_model
    timestep = dt or data.dt

    # Construct from the vendored monodomain solver + shared mesh (self-contained; no _prepare_engine).
    # Private package _monodomain (underscore) so it doesn't shadow the public monodomain() factory.
    from .mesh.structured import StructuredGrid
    from ._monodomain import FDMDiscretization, MonodomainSimulation

    # Build grid
    mask_tensor = torch.tensor(data.mask, dtype=torch.bool)
    grid = StructuredGrid.from_mask(mask_tensor, data.dx, data.dy, device=device)

    # Build FDM discretization
    is_isotropic = (
        np.allclose(data.D_xx, data.D_xx.flat[0])
        and np.allclose(data.D_yy, data.D_yy.flat[0])
        and np.allclose(data.D_xy, 0.0)
        and np.isclose(data.D_xx.flat[0], data.D_yy.flat[0])
    )

    if is_isotropic:
        spatial = FDMDiscretization(
            grid=grid,
            D=float(data.D_xx.flat[0]),
            chi=data.chi,
            Cm=data.Cm,
        )
    else:
        dev = torch.device(device)
        D_field = (
            torch.tensor(data.D_xx, dtype=torch.float64, device=dev),
            torch.tensor(data.D_xy, dtype=torch.float64, device=dev),
            torch.tensor(data.D_yy, dtype=torch.float64, device=dev),
        )
        spatial = FDMDiscretization(
            grid=grid,
            chi=data.chi,
            Cm=data.Cm,
            D_field=D_field,
        )

    # Build stimulus
    dev = torch.device(device)
    stimulus = _build_stimulus_protocol_v54(data, grid, dev, torch.float64)

    # Cell type from first group
    cell_type = data.group_cell_types[0] if data.group_cell_types else 'ENDO'

    # Build simulation
    sim = MonodomainSimulation(
        spatial=spatial,
        ionic_model=ionic,
        stimulus=stimulus,
        dt=timestep,
        splitting=splitting,
        diffusion_solver=diffusion_solver,
        linear_solver=linear_solver,
        cell_type=cell_type,
    )

    build_kwargs = dict(dt=timestep, splitting=splitting, diffusion_solver=diffusion_solver,
                        linear_solver=linear_solver, device=device)
    return CardiacSimulation(sim, 'monodomain', grid, data, build_kwargs)


def bidomain(
    geometry=None,
    ionic_model: Optional[str] = None,
    conductivity: Optional[ConductivityConfig] = None,
    stimulus=None,
    *,
    mesh: Union[str, CardiacMeshData, None] = None,
    dt: Optional[float] = None,
    sigma_ratio: float = 3.59,
    boundary: Optional[str] = None,
    elliptic_solver: str = 'auto',
    theta: float = 0.5,
    device: str = 'cpu',
) -> CardiacSimulation:
    """Create a bidomain simulation.

    Declarative: ``bidomain(Grid(...), 'ttp06', ConductivityConfig.bidomain(σ_i, σ_e), stimulus)``.
    Legacy: ``bidomain(mesh)`` (positional ``CardiacMeshData``/``str`` auto-detected as ``mesh=``).

    Parameters
    ----------
    geometry : Grid, optional
        Structured grid for declarative construction.
    ionic_model : str, optional
        Ionic model name (declarative) or override (legacy).
    conductivity : ConductivityConfig, optional
        Must carry sigma_i/sigma_e (use ``ConductivityConfig.bidomain(...)``).
    stimulus : dict or list[dict], optional
        Declarative stimulus region(s).
    mesh : str or CardiacMeshData, optional
        Legacy construction.
    sigma_ratio : float
        Ratio sigma_e/sigma_i for deriving D_i/D_e from effective D (legacy, when sigma_i/e absent).
    boundary : str, optional
        'insulated' or 'bath'.
    elliptic_solver, theta, dt, device
        Solver / runtime knobs.

    Returns
    -------
    CardiacSimulation
        Wrapper with .run() generator interface.
    """
    if isinstance(geometry, (str, Path, CardiacMeshData)):
        mesh, geometry = geometry, None
    if mesh is not None:
        data = _resolve_mesh(mesh)
    else:
        data = _build_mesh_data(geometry, ionic_model, conductivity, stimulus, dt, 'bidomain')
    ionic = ionic_model or data.ionic_model
    timestep = dt or data.dt
    bc_type = boundary or data.boundary

    # Construct from the vendored bidomain solver + shared mesh (self-contained; no _prepare_engine).
    from ._bidomain import BidomainSimulation, BidomainFDMDiscretization, BidomainConductivity
    from .mesh.structured import StructuredGrid
    from .mesh.boundary import BoundarySpec

    # Build grid with boundary spec
    mask_tensor = torch.tensor(data.mask, dtype=torch.bool)
    grid = StructuredGrid.from_mask(mask_tensor, data.dx, data.dy, device=device)

    if bc_type == 'bath':
        grid.boundary_spec = BoundarySpec.bath_coupled()
    else:
        grid.boundary_spec = BoundarySpec.insulated()

    # Build conductivity
    if data.sigma_i is not None and data.sigma_e is not None:
        # Direct sigma → D conversion
        chi_Cm = data.chi * data.Cm
        D_i_xx = data.sigma_i[0] / chi_Cm
        D_i_yy = data.sigma_i[1] / chi_Cm
        D_i_xy = data.sigma_i[2] / chi_Cm
        D_e_xx = data.sigma_e[0] / chi_Cm
        D_e_yy = data.sigma_e[1] / chi_Cm
        D_e_xy = data.sigma_e[2] / chi_Cm

        dev = torch.device(device)
        cond = BidomainConductivity(
            D_i_field=(
                torch.tensor(D_i_xx, dtype=torch.float64, device=dev),
                torch.tensor(D_i_xy, dtype=torch.float64, device=dev),
                torch.tensor(D_i_yy, dtype=torch.float64, device=dev),
            ),
            D_e_field=(
                torch.tensor(D_e_xx, dtype=torch.float64, device=dev),
                torch.tensor(D_e_xy, dtype=torch.float64, device=dev),
                torch.tensor(D_e_yy, dtype=torch.float64, device=dev),
            ),
        )
    else:
        # Derive from effective D and sigma_ratio
        # D_eff = D_i * D_e / (D_i + D_e)
        # With ratio r = D_e/D_i: D_i = D_eff * (1 + r) / r, D_e = D_eff * (1 + r)
        is_isotropic = (
            np.allclose(data.D_xx, data.D_xx.flat[0])
            and np.allclose(data.D_yy, data.D_yy.flat[0])
            and np.allclose(data.D_xy, 0.0)
            and np.isclose(data.D_xx.flat[0], data.D_yy.flat[0])
        )

        if is_isotropic:
            D_eff = float(data.D_xx.flat[0])
            r = sigma_ratio
            D_i = D_eff * (1 + r) / r
            D_e = D_eff * (1 + r)
            cond = BidomainConductivity(D_i=D_i, D_e=D_e)
        else:
            # Anisotropic: apply ratio to each component
            r = sigma_ratio
            dev = torch.device(device)
            D_i_xx = data.D_xx * (1 + r) / r
            D_i_yy = data.D_yy * (1 + r) / r
            D_i_xy = data.D_xy * (1 + r) / r
            D_e_xx = data.D_xx * (1 + r)
            D_e_yy = data.D_yy * (1 + r)
            D_e_xy = data.D_xy * (1 + r)

            cond = BidomainConductivity(
                D_i_field=(
                    torch.tensor(D_i_xx, dtype=torch.float64, device=dev),
                    torch.tensor(D_i_xy, dtype=torch.float64, device=dev),
                    torch.tensor(D_i_yy, dtype=torch.float64, device=dev),
                ),
                D_e_field=(
                    torch.tensor(D_e_xx, dtype=torch.float64, device=dev),
                    torch.tensor(D_e_xy, dtype=torch.float64, device=dev),
                    torch.tensor(D_e_yy, dtype=torch.float64, device=dev),
                ),
            )

    # Build FDM discretization
    spatial = BidomainFDMDiscretization(grid, cond, Cm=data.Cm)

    # Build stimulus
    dev = torch.device(device)
    stimulus = _build_stimulus_protocol_bidomain(data, grid, dev, torch.float64)

    # Build simulation
    sim = BidomainSimulation(
        spatial=spatial,
        ionic_model=ionic,
        stimulus=stimulus,
        dt=timestep,
        elliptic_solver=elliptic_solver,
        theta=theta,
        device=device,
    )

    build_kwargs = dict(dt=timestep, sigma_ratio=sigma_ratio, boundary=boundary,
                        elliptic_solver=elliptic_solver, theta=theta, device=device)
    return CardiacSimulation(sim, 'bidomain', grid, data, build_kwargs)


def lbm(
    geometry=None,
    ionic_model: Optional[str] = None,
    conductivity: Optional[ConductivityConfig] = None,
    stimulus=None,
    *,
    mesh: Union[str, CardiacMeshData, None] = None,
    dt: Optional[float] = None,
    lattice: str = 'd2q5',
    device: str = 'cpu',
) -> CardiacSimulation:
    """Create an LBM simulation.

    Declarative: ``lbm(Grid(...), 'ttp06', ConductivityConfig.isotropic(σ), stimulus)``.
    Legacy: ``lbm(mesh)`` (positional ``CardiacMeshData``/``str`` auto-detected as ``mesh=``).

    Parameters
    ----------
    geometry : Grid, optional
        Structured grid for declarative construction.
    ionic_model : str, optional
        Ionic model name (declarative) or override (legacy).
    conductivity : ConductivityConfig, optional
        Conductivity (Form B: ``D_eff`` is fed straight to LBM).
    stimulus : dict or list[dict], optional
        Declarative stimulus region(s).
    mesh : str or CardiacMeshData, optional
        Legacy construction.
    dt : float, optional
        Time step (LBM typically uses smaller dt, e.g. 0.005).
    lattice : str
        Lattice type ('d2q5' or 'd2q9').
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    CardiacSimulation
        Wrapper with .run() generator interface.
    """
    if isinstance(geometry, (str, Path, CardiacMeshData)):
        mesh, geometry = geometry, None
    if mesh is not None:
        data = _resolve_mesh(mesh)
    else:
        data = _build_mesh_data(geometry, ionic_model, conductivity, stimulus, dt, 'lbm')
    ionic_name = ionic_model or data.ionic_model
    timestep = dt or data.dt

    # Construct from the vendored LBM solver + shared ionic (self-contained; no _prepare_engine).
    from ._lbm.simulation import LBMSimulation
    from .ionic import TTP06Model, ORdModel, PHAS13Model, MHAS13Model

    # Instantiate ionic model
    model_map = {
        'ttp06': TTP06Model,
        'ord': ORdModel,
        'phas13': PHAS13Model,
        'paci': PHAS13Model,
        'mhas13': MHAS13Model,
    }
    if ionic_model is not None and not isinstance(ionic_model, str):
        ionic_instance = ionic_model            # pre-built (e.g. tuner-scaled) IonicModel — use as-is
    else:
        model_cls = model_map.get(ionic_name.lower())
        if model_cls is None:
            raise ValueError(f"Unknown ionic model: {ionic_name}")
        ionic_instance = model_cls(device=device)

    # Determine D. LBM needs spatially-uniform diagonal D (D_xx, D_yy constant;
    # D_xy = 0). Isotropic (D_xx == D_yy) -> BGK on the requested lattice;
    # anisotropic (D_xx != D_yy) -> D2Q9-MRT (BGK is single-relaxation/isotropic).
    uniform = (
        np.allclose(data.D_xx, data.D_xx.flat[0])
        and np.allclose(data.D_yy, data.D_yy.flat[0])
        and np.allclose(data.D_xy, 0.0)
    )
    if not uniform:
        raise ValueError(
            "LBM supports spatially-uniform diagonal D only "
            "(D_xx, D_yy constant; D_xy = 0). "
            "Oblique fibers (D_xy != 0) need the moment-space rotation (out of scope)."
        )
    D_xx = float(data.D_xx.flat[0])
    D_yy = float(data.D_yy.flat[0])
    anisotropic = not np.isclose(D_xx, D_yy)

    Nx, Ny = data.mask.shape
    if anisotropic:
        sim = LBMSimulation(
            Nx=Nx, Ny=Ny,
            dx=data.dx, dt=timestep,
            D=D_xx, D_yy=D_yy, ionic_model=ionic_instance,
            Cm=data.Cm,
            lattice='d2q9', collision='mrt',
        )
    else:
        sim = LBMSimulation(
            Nx=Nx, Ny=Ny,
            dx=data.dx, dt=timestep,
            D=D_xx, ionic_model=ionic_instance,
            Cm=data.Cm,
            lattice=lattice,
        )

    # Add stimuli as (Nx, Ny) bool tensor masks
    dev = torch.device(device)
    for stim in data.stimuli:
        stim_mask = torch.tensor(
            stim['mask'] & data.mask,
            dtype=torch.bool,
            device=dev,
        )

        bcl = stim.get('bcl', 0.0)
        num_pulses = stim.get('num_pulses', 1)

        if bcl > 0 and num_pulses > 1:
            for p in range(num_pulses):
                t_start = stim['start_time'] + p * bcl
                sim.add_stimulus(stim_mask, t_start, stim['duration'], stim['amplitude'])
        else:
            sim.add_stimulus(stim_mask, stim['start_time'], stim['duration'], stim['amplitude'])

    build_kwargs = dict(dt=timestep, lattice=lattice, device=device)
    return CardiacSimulation(sim, 'lbm', None, data, build_kwargs)
