"""
Simplified Cardiac Simulation API.

Provides monodomain(), bidomain(), and lbm() functions that accept either
a file path or CardiacMeshData and return a CardiacSimulation wrapper
with a uniform generator interface.
"""

import copy
import warnings
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

# Fields ``run()``/``snapshots()`` know how to collect. "Vm" is always saved; "ionic_states" is opt-in.
_SUPPORTED_RECORD = frozenset({"Vm", "ionic_states"})


def _validate_record(record) -> None:
    """Reject unknown ``record=`` keys so a typo/unsupported field raises instead of
    silently recording nothing (B7). e.g. ``record=("Vm", "I_Kr")`` -> ValueError."""
    if isinstance(record, str):
        record = (record,)
    unknown = [k for k in record if k not in _SUPPORTED_RECORD]
    if unknown:
        raise ValueError(
            f"unknown record key(s): {unknown}; "
            f"supported: {sorted(_SUPPORTED_RECORD)}"
        )


def _scale_ionic_conductances(model, scalings):
    """Deep-copy ``model`` and multiply the named conductances on the copy.

    Uniform across models (TTP06/ORd/PHAS13/…): all expose their maximal conductances/
    permeabilities as attributes on ``self.params``. Operating on a deep copy of the LIVE
    engine model (not a freshly-named build) keeps the cell type and any prior scalings
    consistent across engines — the bidomain/LBM factories build ENDO by default, so
    re-deriving from name+mesh-cell_type would silently flip cell type. Raises
    ``ValueError`` on an unknown conductance name.
    """
    import copy as _copy
    model = _copy.deepcopy(model)
    params = model.params
    # Scalable = a true maximal conductance / permeability / transporter rate: an ohmic
    # `G*` (TTP06/ORd) or `g_*` (hiPSC paci/phas13/mhas13, LOWERCASE) or a `P*`
    # permeability/pump. EXCLUDES the `*_scale` tuning factors and the dimensionless
    # parameters that merely start with g/p — `gamma_ncx` (NCX voltage-partition) and
    # `PkNa` (the IKs permeability RATIO in the Nernst term) — which scale a shape/reversal,
    # not a magnitude, and which a bare `hasattr`/first-letter check would silently corrupt.
    _NON_CONDUCTANCE = {'gamma_ncx', 'pkna'}
    conductances = {
        a for a in vars(params)
        if a[:1].lower() in ('g', 'p')
        and not a.endswith('_scale')
        and a.lower() not in _NON_CONDUCTANCE
    }
    for name, factor in scalings.items():
        if name not in conductances:
            raise ValueError(
                f"{name!r} is not a scalable conductance of {type(model).__name__}; "
                f"available conductances: {sorted(conductances)}"
            )
        setattr(params, name, getattr(params, name) * float(factor))
    return model


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

    Conductance/conductivity knobs ARE shipped: ``scale_conductance`` (ionic drug block /
    upregulation), ``set_conductivity`` (scar / heterogeneous D), ``scale_conductivity``
    (slow-conduction zone) — each rebuilds the sim from t=0. Other convenience methods are
    still PLANNED and raise an INFORMATIVE NotImplementedError naming the real route (state
    probes get_state/set_state/set_voltage/state_names/ionic_states; voltage clamp; pacing/
    injection helpers; general set_parameter; probes; on-object analysis compute_cv/apd/
    activation_time). For analysis use ``cardiac_core.analysis`` or the ``result`` hooks
    (``result.cv()/apd()/lat()``) on a recorded ``run()`` (Audit #7/#12).

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
                 build_kwargs: Optional[dict] = None, *, boundary_mode: str = 'face_mirror'):
        self._engine = engine
        self._engine_type = engine_type
        self._grid = grid
        self._data = data
        # Engine-construction knobs (dt/splitting/solver/device/...) — enough to replay the
        # factory with `mesh=self._data` for reset()/with_()/stimulate(). The geometry,
        # conductivity, and stimulus are already baked into `data`.
        self._build_kwargs = dict(build_kwargs or {})
        # The ghost/mirror edge rule the analysis field ops should apply (Phase-1 fields).
        # Persisted here because the monodomain factory takes `boundary_mode` but otherwise
        # only forwards it to the discretization; bidomain/LBM pass 'face_mirror' (the
        # no-flux tissue-Vm edge rule) since their edge concepts (BoundarySpec bath/insulated,
        # LBM wall modes) are not the analysis-stencil ghost rule.
        self._boundary_mode = boundary_mode
        self._Nx = data.mask.shape[0]
        self._Ny = data.mask.shape[1]
        self._probes: dict[str, dict] = {}   # name → {x, y, ix, iy, t[], V[]}
        self._clamp_mask: Optional[torch.Tensor] = None   # flat (n_dof,) bool (mono/bidomain)
        self._clamp_value = None                           # scalar | (Nx,Ny) field | callable(t)
        self._clamp_start: Optional[float] = None
        self._clamp_end: Optional[float] = None
        # LBM native clamp record (mask_t, value, start, end) — re-pushed to a fresh engine on reset().
        self._lbm_clamp = None

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
        _validate_record(record)
        it = self._iter_snapshots(t_end, save_every, record=record, callback=callback)
        _shape = (self._Nx, self._Ny)
        if batch is None:
            return _result_from(list(it), record, self.dx, self.dy, shape=_shape, sim=self)

        def _gen():
            buf = []
            for snap in it:
                buf.append(snap)
                if len(buf) == batch:
                    yield _result_from(buf, record, self.dx, self.dy, shape=_shape, sim=self)
                    buf = []
            if buf:
                yield _result_from(buf, record, self.dx, self.dy, shape=_shape, sim=self)
        return _gen()

    def snapshots(self, t_end: float, save_every: float = 1.0, *, record=("Vm",),
                  callback=None) -> Iterator[SimulationSnapshot]:
        """Frame-by-frame generator of :class:`SimulationSnapshot` (the pre-Phase-5 ``run()``).

        Use this where you want to iterate snapshots lazily; ``run()`` is now eager and returns a
        ``SimulationResult``.
        """
        _validate_record(record)
        return self._iter_snapshots(t_end, save_every, record=record, callback=callback)

    def _reset_solver_diagnostics(self):
        """Reset per-run solver diagnostics at the start of each run().

        The non-convergence warning is throttled to once per solver instance (anti-flood).
        Without a per-run reset, a solver REUSED across runs (restitution / S1-S2 / sigma
        sweeps) would warn on run 1 then stay silent forever — re-silencing a later, worse
        under-solve and defeating `filterwarnings('error', SolverConvergenceWarning)` on
        subsequent runs. Resetting here keeps flood-control WITHIN a run while restoring the
        honest signal (and escalation) on every new run.
        """
        ds = getattr(getattr(self._engine, 'splitting', None), 'diffusion_solver', None)
        if ds is None:
            return
        for attr in ('linear_solver', 'parabolic_solver', 'elliptic_solver'):
            s = getattr(ds, attr, None)
            if s is not None:
                s._nonconv_warned = False

    def _iter_snapshots(self, t_end, save_every, *, record=("Vm",), callback=None):
        self._reset_solver_diagnostics()
        if self._clamp_mask is not None and self._engine_type != 'lbm':
            # A mid-run clamp needs per-step enforcement -> wrapper-driven stepping.
            gen = self._stepping_run(t_end, save_every, record)
        elif self._engine_type == 'monodomain':
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
        # A native LBM clamp lives on the engine, which we just rebuilt — re-push it.
        if self._engine_type == 'lbm' and self._lbm_clamp is not None:
            self._engine.set_clamp(*self._lbm_clamp)

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
        """All ionic state variables (not implemented as a live property)."""
        raise NotImplementedError(
            "ionic_states is not a live property; record the history with "
            "run(record=('Vm', 'ionic_states')) and read result.ionic_states."
        )

    def get_state(self, name: str) -> torch.Tensor:
        """Get a single ionic state variable by name as an ``(Nx, Ny)`` grid.

        ``name`` is one of :attr:`state_names` (e.g. ``'Cai'``, ``'Nai'``, ``'m'``, ``'h'``).
        Masked-out nodes read back as NaN (see ``flat_to_grid``).
        """
        col = self._state_col(name)
        return self._grid.flat_to_grid(self._ionic_states_ref()[:, col])

    @property
    def state_names(self) -> list[str]:
        """Names of the ionic state variables, in ``ionic_states`` column order.

        These name the columns of the per-node ionic state (V is separate — use
        :attr:`Vm` / :meth:`set_voltage`). Valid keys for :meth:`get_state`/:meth:`set_state`.
        """
        return list(self._live_ionic_model().state_names)

    # ========================================================================
    # State access — write
    # ========================================================================

    def set_voltage(self, V: torch.Tensor):
        """Overwrite the membrane potential mid-run with an ``(Nx, Ny)`` field.

        Writes directly into the live engine state (on its device/dtype). Use it to inject
        a rotor-seeding phase distribution, a plateau/resting field, or a checkpoint restart.
        LBM is unsupported (V is a moment of the lattice populations, not a stored field).
        """
        v_flat = self._field_to_flat(V)
        self._voltage_ref().copy_(v_flat)

    def set_state(self, name: str, values: torch.Tensor):
        """Overwrite one ionic state variable mid-run with an ``(Nx, Ny)`` field.

        ``name`` is one of :attr:`state_names`. Writes into the live per-node ionic state.
        """
        col = self._state_col(name)
        self._ionic_states_ref()[:, col] = self._field_to_flat(values)

    # ========================================================================
    # Stimulus / current injection
    # ========================================================================

    def _grid_coords(self):
        """Grid-shaped ``(x, y)`` coords ``(Nx, Ny)`` matching the engine ``ij`` convention."""
        x1d = torch.linspace(0.0, self._data.dx * (self._Nx - 1), self._Nx, dtype=torch.float64)
        y1d = torch.linspace(0.0, self._data.dy * (self._Ny - 1), self._Ny, dtype=torch.float64)
        return torch.meshgrid(x1d, y1d, indexing='ij')

    def _as_grid_mask(self, mask):
        """Normalize a region (callable ``(x,y)->bool`` / array / tensor) to an
        ``(Nx, Ny)`` bool ndarray, validating the shape."""
        if callable(mask):
            xx, yy = self._grid_coords()
            mask = mask(xx, yy)
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        m = np.asarray(mask).astype(bool)
        if m.shape != (self._Nx, self._Ny):
            raise ValueError(f"mask shape {m.shape} != (Nx, Ny) = ({self._Nx}, {self._Ny})")
        return m

    def _live_ionic_model(self):
        """The IonicModel instance the running engine actually uses (engine-agnostic).

        Used so ``scale_conductance`` scales the model the sim is really running with —
        preserving its cell type (bidomain/LBM default ENDO regardless of the mesh) and
        any prior scalings, rather than re-deriving from a name + a possibly-mismatched
        cell type.
        """
        eng = self._engine
        if self._engine_type == 'lbm':
            return eng.ionic_model
        if self._engine_type == 'monodomain':
            return eng._ionic_model
        # bidomain: splitting -> ionic_solver -> ionic_model
        return eng.splitting.ionic_solver.ionic_model

    # --- Live-state access helpers (mid-run clamp / injection) ---

    def _require_stateful(self, what: str):
        if self._engine_type == 'lbm':
            raise NotImplementedError(
                f"{what} is not supported for the LBM engine: V and gates are moments of the "
                f"lattice populations, not a stored per-node state. Use a monodomain/bidomain sim."
            )

    def _voltage_ref(self) -> torch.Tensor:
        """The live flat ``(n_dof,)`` membrane-potential tensor (in-place writable)."""
        self._require_stateful("mid-run voltage access")
        st = self._engine.state
        return st.V if self._engine_type == 'monodomain' else st.Vm

    def _ionic_states_ref(self) -> torch.Tensor:
        """The live ``(n_dof, n_states)`` ionic-state tensor (in-place writable)."""
        self._require_stateful("ionic-state access")
        return self._engine.state.ionic_states

    def _state_col(self, name: str) -> int:
        names = list(self._live_ionic_model().state_names)
        if name not in names:
            raise ValueError(f"unknown ionic state {name!r}; valid names: {names}")
        return names.index(name)

    def _field_to_flat(self, field) -> torch.Tensor:
        """Normalize an ``(Nx, Ny)`` field (tensor/array/scalar) to a flat ``(n_dof,)`` tensor
        on the engine device/dtype."""
        ref = self._voltage_ref()
        if not torch.is_tensor(field):
            field = torch.as_tensor(np.asarray(field))
        field = field.to(device=ref.device, dtype=ref.dtype)
        if field.shape == (self._Nx, self._Ny):
            return self._grid.grid_to_flat(field)
        flat = field.reshape(-1)
        if flat.shape == ref.shape:
            return flat
        raise ValueError(f"field shape {tuple(field.shape)} != (Nx, Ny)=({self._Nx}, {self._Ny}) "
                         f"or flat (n_dof,)=({ref.shape[0]},)")

    def _rebuild_with_conductivity(self, mask, *, set_value=None, scale=None):
        """Apply an absolute ``set_value`` or a multiplicative ``scale`` to the tissue
        conductivity on ``mask`` and rebuild the sim from t=0.

        Covers BOTH representations: the ``D_xx/D_yy/D_xy`` diffusivity fields (monodomain,
        LBM, and the legacy D-based bidomain path) AND the ``sigma_i/sigma_e`` conductivity
        fields that a declaratively-built bidomain actually uses (its factory ignores
        ``D_xx`` when sigma fields are present — mutating only ``D_xx`` would be a silent
        no-op). An absolute nonzero ``set_value`` has no unambiguous sigma meaning, so it
        raises for sigma-parameterized bidomain (a D=0 scar is fine: zero everything).
        """
        from dataclasses import replace
        m = self._as_grid_mask(mask)
        if self._engine_type == 'lbm' and not m.all():
            raise NotImplementedError(
                "regional set_conductivity/scale_conductivity is not supported on an LBM "
                "sim — the LBM path requires spatially-uniform diagonal D. Apply a "
                "full-domain change, or use monodomain/bidomain for a scar / heterogeneity."
            )
        data = self._data

        def _op(arr):
            out = arr.copy()
            if set_value is not None:
                out[m] = float(set_value)
            else:
                out[m] *= float(scale)
            return out

        kw = dict(D_xx=_op(data.D_xx), D_yy=_op(data.D_yy))
        # D_xy: scar/absolute-set zeroes the cross term; scaling multiplies it.
        D_xy = data.D_xy.copy()
        if set_value is not None:
            D_xy[m] = 0.0
        else:
            D_xy[m] *= float(scale)
        kw['D_xy'] = D_xy

        if data.sigma_i is not None and data.sigma_e is not None:
            if set_value is not None and float(set_value) != 0.0:
                raise NotImplementedError(
                    "set_conductivity with a nonzero absolute D is not supported on a "
                    "bidomain built from ConductivityConfig (it stores conductivities, "
                    "not a diffusivity). Use scale_conductivity(mask, factor), or build "
                    "the mesh via create_cardiac_mesh (D-based)."
                )

            def _op_sigma(sig):  # sig = (xx, yy, xy) tuple of (Nx,Ny) arrays
                return tuple(_op(c) for c in sig)

            kw['sigma_i'] = _op_sigma(data.sigma_i)
            kw['sigma_e'] = _op_sigma(data.sigma_e)

        # Transactional: commit the new mesh, then rebuild; if the rebuild raises (e.g. a
        # dct sim whose new non-uniform D trips the spectral gate), restore the old mesh so
        # _data and _engine stay consistent rather than describing a scar with a stale engine.
        old_data = self._data
        self._data = replace(data, **kw)
        try:
            self.reset()
        except Exception:
            self._data = old_data
            raise

    def stimulate(self, region, start_time: float = 0.0, duration: float = 1.0,
                  amplitude: float = -52.0):
        """STIMULATE idiom: append a stimulus and rebuild (so it takes effect from t=0).

        ``region`` is a callable ``(x, y) -> bool mask``, an ``(Nx, Ny)`` mask, or a
        :class:`~cardiac_core.stimulus.stim.Stim`. A current ``Stim`` carries its own
        ``start_time``/``duration``/``amplitude`` (the positional kwargs are ignored for it); a
        clamp ``Stim`` is routed to :meth:`clamp_voltage`. Stored on ``self._data.stimuli`` (the path
        both declarative and legacy construction share), so this works regardless of how the sim was
        built.
        """
        from .stimulus.stim import Stim  # local import — avoids an api<->stim module cycle at load
        if isinstance(region, Stim):
            if region.mode == "clamp":
                self.clamp_voltage(region.mask, region.clamp,
                                   start_time=region.start_time, duration=region.duration)
                return
            stim = region
        else:
            # Build a current Stim internally (NOT a raw dict) so the dict-path deprecation
            # warning never fires on this kept API. Resolve a callable region against the grid.
            x, y = self._grid_coords()
            mask = region(x, y) if callable(region) else region
            stim = Stim(mask, amplitude=amplitude, start_time=start_time, duration=duration)
        entry = stim.to_dict()
        entry['mask'] = entry['mask'] & self._data.mask
        if not entry['mask'].any():
            warnings.warn(
                "stimulate: the stimulus region selects 0 tissue nodes — it will have "
                "no effect (check the region shape / coordinate units in cm).",
                stacklevel=2,
            )
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
        raise NotImplementedError(
            "add_pacing is not implemented; issue repeated stimulate()/add_stimulus() "
            "calls at start_time + k*bcl (k=0..n_beats-1) to build a pacing train."
        )

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
        raise NotImplementedError(
            "inject_current is not implemented; use stimulate()/add_stimulus() with a "
            "short duration for a timed current injection."
        )

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
        """Hold ``mask`` nodes near a fixed voltage during ``run()``; gates keep integrating.

        Clamped nodes have their V re-imposed after every internal step (not just each save
        point), so the gates track the clamped potential to O(dt): within a step V drifts from
        the reaction+diffusion update before being re-snapped, so this is a per-step-reimposed
        clamp (dt-accurate), not an exact within-step voltage clamp. Use a small ``dt`` for a
        tight clamp. Also usable as a fixed-potential pacing/boundary region.

        Parameters
        ----------
        mask : (Nx, Ny) bool
            Nodes to clamp (callable ``(x,y)->bool`` / array / tensor also accepted).
        voltage : float or (Nx, Ny) field or callable(t)
            Clamp value (mV). A callable ``value(t)`` gives a time-varying VC step protocol.
        start_time : float, optional
            When the clamp activates (ms). None = from t=0.
        duration : float, optional
            How long it lasts (ms). None = until :meth:`release_clamp`.

        Notes
        -----
        Activating a clamp switches ``run()`` to wrapper-driven stepping (monodomain/bidomain) so
        the clamp can be enforced every step; the unclamped path stays on the fast engine loop.
        For LBM the clamp is native (V is a lattice-population moment, ``Σf``): the wrapper pushes an
        additive, flux-preserving clamp into the engine (:meth:`LBMSimulation.set_clamp`), enforced
        inside the fast engine loop — no wrapper-driven stepping.
        """
        if self._engine_type == 'lbm':
            # Native LBM clamp: cast the mask to a torch bool tensor on the ENGINE's device
            # (numpy-indexing a CUDA tensor would crash), store it so reset() can re-push it,
            # then hand it to the engine.
            m = self._as_grid_mask(mask)   # (Nx, Ny) bool ndarray
            mask_t = torch.as_tensor(m, dtype=torch.bool, device=self._engine.device)
            end = None if duration is None else ((start_time or 0.0) + duration)
            self._lbm_clamp = (mask_t, voltage, start_time, end)
            self._engine.set_clamp(*self._lbm_clamp)
            return
        self._require_stateful("voltage clamp")
        m = self._as_grid_mask(mask)   # (Nx, Ny) bool ndarray
        mask_t = torch.as_tensor(m, dtype=torch.bool, device=self._voltage_ref().device)
        self._clamp_mask = self._grid.grid_to_flat(mask_t)   # flat (n_dof,) bool
        self._clamp_value = voltage
        self._clamp_start = start_time
        self._clamp_end = None if duration is None else ((start_time or 0.0) + duration)

    def release_clamp(self):
        """Remove any active voltage clamp (``run()`` returns to the fast engine loop)."""
        self._clamp_mask = None
        self._clamp_value = None
        self._clamp_start = None
        self._clamp_end = None
        if self._lbm_clamp is not None:
            self._lbm_clamp = None
            if self._engine_type == 'lbm':
                self._engine.release_clamp()

    def _clamp_active_at(self, t: float) -> bool:
        if self._clamp_mask is None:
            return False
        if self._clamp_start is not None and t < self._clamp_start - 1e-9:
            return False
        if self._clamp_end is not None and t >= self._clamp_end - 1e-9:
            return False
        return True

    def _apply_clamp(self):
        """Re-impose the clamp on the live voltage (called after every step)."""
        if not self._clamp_active_at(self.t):
            return
        val = self._clamp_value
        if callable(val):
            val = val(self.t)
        v = self._voltage_ref()
        fm = self._clamp_mask
        if isinstance(val, (int, float)):
            v[fm] = float(val)
        else:
            v[fm] = self._field_to_flat(val)[fm]

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
        """
        self._require_stateful("voltage clamp")
        bounds = []
        t0 = start_time
        for v, d in steps:
            bounds.append((t0, t0 + d, float(v)))
            t0 += d
        if not bounds:
            raise ValueError("add_clamp_protocol: `steps` is empty")

        def protocol(t):
            for lo, hi, v in bounds:
                if lo - 1e-9 <= t < hi - 1e-9:
                    return v
            return bounds[-1][2]  # hold the last level past the end

        self.clamp_voltage(mask, protocol, start_time=start_time,
                           duration=t0 - start_time)

    # ========================================================================
    # Conductivity / drug block
    # ========================================================================

    def scale_conductance(
        self,
        current_name: str,
        factor: 'float | Distribution',
        mask: 'np.ndarray | torch.Tensor | None' = None,
    ):
        """Scale an ionic model conductance (drug block / upregulation).

        Rebuilds the simulation from t=0 with the named maximal conductance on the
        ionic model multiplied by ``factor`` (< 1 = block, > 1 = upregulation). Repeated
        calls compound (each is relative to the current value).

        ``current_name`` is the model PARAMETER name, not the current name — e.g. TTP06
        'GNa'/'GKr'/'GKs'/'Gto'; ORd 'GNa'/'GKr'/'GKs'/'GK1'/'PCa' (ICaL). An unknown
        name raises with the model's available conductances listed.

        Parameters
        ----------
        current_name : str
            Model conductance parameter (validated against the ionic model instance).
        factor : float
            Multiplicative factor applied uniformly.
        mask : optional
            NOT IMPLEMENTED — per-node conductance heterogeneity is future work.

        Notes
        -----
        Global scalar only: a per-node ``mask`` or a ``Distribution`` ``factor`` raises
        NotImplementedError (the ionic params are uniform scalars, applied by rebuild).
        """
        if mask is not None or isinstance(factor, Distribution):
            raise NotImplementedError(
                "scale_conductance supports a single global scalar factor (rebuild "
                "from t=0); per-node conductance heterogeneity (mask= / Distribution) "
                "is not implemented — pass a uniform float factor."
            )
        # Scale a deep copy of the LIVE engine model — preserves cell type across
        # engines (bidomain/LBM default ENDO) and any prior scalings; store it so the
        # rebuild (and future reset()/with_()) use it.
        model = _scale_ionic_conductances(self._live_ionic_model(), {current_name: factor})
        self._build_kwargs['ionic_model'] = model
        self.reset()

    def set_conductivity(
        self,
        mask: 'np.ndarray | torch.Tensor',
        D: 'float | Distribution',
    ):
        """Set the (raw) diffusion coefficient in a region and rebuild from t=0.

        ``D=0.0`` makes an inexcitable scar: the FDM uses a harmonic-mean interface
        conductivity, so a D=0 cell has zero flux and the wave routes around it. ``D``
        is RAW (like ``create_cardiac_mesh``'s ``D``); the membrane-effective diffusivity
        is ``D/(chi*Cm)``. Making the mesh non-uniform switches the engine to its
        per-node D_field path automatically.

        Parameters
        ----------
        mask : (Nx, Ny) bool, or callable ``(x, y) -> bool``
            Region to modify.
        D : float
            New raw diffusion coefficient. 0 = no conduction (scar).

        Notes
        -----
        A ``Distribution`` ``D`` (per-node stochastic conductivity) is not implemented.
        """
        if isinstance(D, Distribution):
            raise NotImplementedError(
                "set_conductivity supports a scalar D (per-node Distribution not "
                "implemented). Pass a float; D=0.0 makes an inexcitable scar."
            )
        self._rebuild_with_conductivity(mask, set_value=float(D))

    def scale_conductivity(
        self,
        mask: 'np.ndarray | torch.Tensor',
        factor: 'float | Distribution',
    ):
        """Multiply the diffusion coefficient in a region by ``factor`` (rebuild from t=0).

        Slow-conduction zones (0 < factor < 1) or faster tracts (factor > 1). Shares the
        machinery of :meth:`set_conductivity`; use that for an absolute value / scar.

        Parameters
        ----------
        mask : (Nx, Ny) bool, or callable ``(x, y) -> bool``
            Region to modify.
        factor : float
            Multiplicative factor on the current per-node D.

        Notes
        -----
        A ``Distribution`` ``factor`` (per-node stochastic scaling) is not implemented.
        """
        if isinstance(factor, Distribution):
            raise NotImplementedError(
                "scale_conductivity supports a scalar factor (per-node Distribution "
                "not implemented)."
            )
        self._rebuild_with_conductivity(mask, scale=float(factor))

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

        """
        raise NotImplementedError(
            "set_parameter (general per-node parameter/heterogeneity) is not "
            "implemented; for a uniform ionic conductance use "
            "scale_conductance(name, factor), and for tissue diffusivity use "
            "set_conductivity(mask, D) / scale_conductivity(mask, factor)."
        )

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
        raise NotImplementedError(
            "get_parameter_field is not implemented (no per-node parameter fields)."
        )

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
        raise NotImplementedError(
            "add_probe is not implemented; record the full field with run() and index "
            "result.Vm[:, ix, iy] at the node(s) you want."
        )

    def get_traces(self) -> dict:
        """Return recorded probe traces.

        Returns
        -------
        dict
            {name: {'t': np.ndarray, 'V': np.ndarray, 'phi_e': np.ndarray | None}}
        """
        raise NotImplementedError(
            "get_traces is not implemented; use run() and slice result.Vm (and "
            "result.phi_e for bidomain) at the node(s) you care about."
        )

    def clear_traces(self):
        """Clear all recorded probe data (not implemented — no probe API)."""
        raise NotImplementedError("clear_traces is not implemented (no probe API).")

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
        raise NotImplementedError(
            "compute_activation_time is not a sim method; call result = sim.run(...) "
            "then result.lat() (or cardiac_core.analysis.activation_time)."
        )

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
        raise NotImplementedError(
            "compute_cv is not a sim method; call result = sim.run(...) then "
            "result.cv(x1, x2, y) (integer node indices)."
        )

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
        raise NotImplementedError(
            "compute_apd is not a sim method; call result = sim.run(...) then "
            "result.apd() (or cardiac_core.analysis.apd_at for a single node)."
        )

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

    @property
    def boundary_mode(self) -> str:
        """Ghost/mirror edge rule the analysis field ops apply (``'face_mirror'`` = no-flux)."""
        return self._boundary_mode

    # --- Private generators ---

    def _grid_ionic(self, state):
        """Grid the engine's flat ionic_states (n_dof, n_states) → (n_states, Nx, Ny)."""
        S = state.ionic_states
        return torch.stack([self._grid.flat_to_grid(S[:, k]) for k in range(S.shape[1])])

    def _snapshot_current(self, record=("Vm",)):
        """Build a SimulationSnapshot from the CURRENT (mono/bidomain) engine state."""
        want_ionic = "ionic_states" in record
        st = self._engine.state
        if self._engine_type == 'monodomain':
            Vm = self._grid.flat_to_grid(st.V)
            phi = None
        else:  # bidomain
            Vm = self._grid.flat_to_grid(st.Vm)
            phi = self._grid.flat_to_grid(st.phi_e)
        return SimulationSnapshot(
            t=st.t, Vm=Vm, phi_e=phi, Nx=self._Nx, Ny=self._Ny,
            dx=self._data.dx, dy=self._data.dy,
            ionic_states=self._grid_ionic(st) if want_ionic else None,
        )

    def _stepping_run(self, t_end, save_every, record=("Vm",)):
        """Wrapper-driven stepping that re-imposes the voltage clamp after every step.

        Mirrors the engine's own run() save cadence (step, t+=dt, save when t crosses the
        next save point) but injects _apply_clamp() each step. Used only when a clamp is
        active; the unclamped path stays on the fast engine.run() loop (goldens untouched).

        The loop/save tolerances MUST match the underlying engine's run() exactly, or a
        clamped run drifts from its unclamped control by a trailing frame. Monodomain.run
        uses `t < t_end` / `t >= next_save - 1e-9`; bidomain.run uses the stricter
        `t < t_end - 1e-12` / `t >= next_save - 1e-12`.
        """
        if self._engine_type == 'bidomain':
            end_eps, save_eps = 1e-12, 1e-12
        else:  # monodomain
            end_eps, save_eps = 0.0, 1e-9
        next_save = save_every
        self._apply_clamp()   # enforce at t=0 as well
        while self.t < t_end - end_eps:
            self._engine.step()
            self._apply_clamp()
            if self.t >= next_save - save_eps:
                next_save += save_every
                yield self._snapshot_current(record)

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
    """Accept path or CardiacMeshData, return CardiacMeshData.

    DEEP-COPY the in-memory branch (I2): _resolve_mesh is the single choke point for every
    construction path (factory, with_, reset), so copying here makes each sim OWN its mesh —
    stimulate()/with_() then can't mutate the caller's object or a sibling sim. _data holds
    only numpy arrays + scalars + a stimuli list of dicts (no torch/CUDA tensors, no ionic
    instance — that lives in _build_kwargs), so the copy is cheap and bit-identical (goldens
    unaffected). The str/path branch already loads fresh from disk.
    """
    if isinstance(mesh, (str, Path)):
        return load_cardiac_mesh(str(mesh))
    return copy.deepcopy(mesh)


def _lbm_bounce_masks(data, lattice_name, anisotropic, device):
    """Per-direction LBM bounce-back masks for a MASKED interior geometry (I1).

    UNION of (a) the interior-hole rim from ``precompute_bounce_masks`` — which uses a
    periodic ``torch.roll`` and so flags ONLY the hole rim, NOT the outer array walls —
    and (b) the outer rectangular edges. Returns None for a full (all-True) mask so the
    engine's own ``_make_rect_masks`` is used unchanged (golden-safe).
    """
    if bool(data.mask.all()):
        return None
    from ._lbm.boundary.masks import precompute_bounce_masks
    from ._lbm.lattice import D2Q9, D2Q5
    lat = D2Q9() if (anisotropic or lattice_name == 'd2q9') else D2Q5()
    dev = torch.device(device)
    mask_t = torch.tensor(data.mask, dtype=torch.bool, device=dev)
    hole = precompute_bounce_masks(mask_t, lat)
    out = {}
    for a in range(1, lat.Q):
        m = hole[a].clone()
        ex, ey = lat.e[a]
        if ex == 1:   m[-1, :] = True
        if ex == -1:  m[0, :] = True
        if ey == 1:   m[:, -1] = True
        if ey == -1:  m[:, 0] = True
        out[a] = m
    return out


def _normalize_stimulus(stimulus, coords) -> list:
    """Convert a declarative ``stimulus`` arg into CardiacMeshData ``stimuli`` dicts.

    Accepts ``None``, a single ``dict``/:class:`~cardiac_core.stimulus.stim.Stim`, or a list of
    either. A dict carries a ``region`` (a callable ``(x, y) -> bool mask`` evaluated on the grid
    coordinates, OR an ``(Nx, Ny)`` array/mask) plus optional ``start_time``/``duration``/
    ``amplitude``/``label``/``bcl``/``num_pulses``. A CURRENT-mode ``Stim`` lowers via ``to_dict()``;
    a CLAMP-mode ``Stim`` is NOT a current stimulus — it raises here (it must be routed to
    ``clamp_voltage`` by the factory-level ``_partition_stimulus``, never reach ``data.stimuli``).
    """
    from .stimulus.stim import Stim  # local import — avoids an api<->stim module cycle at load
    if stimulus is None:
        return []
    if isinstance(stimulus, (dict, Stim)):
        stimulus = [stimulus]
    out = []
    for s in stimulus:
        if isinstance(s, Stim):
            if s.mode == "clamp":
                raise ValueError(
                    "a clamp-mode Stim cannot be lowered as a current stimulus — it must be applied "
                    "via clamp_voltage (the factory routes clamp Stims there). Reaching "
                    "_normalize_stimulus with a clamp Stim is an internal routing error.")
            out.append(s.to_dict())
            continue
        # Soft-deprecation: a raw stimulus dict is the legacy form. It still works (coexistence),
        # but steer users to cardiac_core.Stim. Internal callers (stimulate/protocols) build Stims,
        # so this fires only where a USER passes a dict to a public factory.
        # stacklevel=4 targets the user's factory call (user → factory → _build_mesh_data →
        # _normalize_stimulus → warn); _build_mesh_data is the only production caller.
        warnings.warn(
            "stimulus dicts are deprecated; use cardiac_core.Stim (e.g. Stim.boundary(grid, 'left') "
            "or Stim.from_region(grid, region, amplitude=...)) — dicts still work for now.",
            DeprecationWarning, stacklevel=4)
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


def _partition_stimulus(stimulus):
    """Split a factory ``stimulus`` arg into ``(current, clamp_stims)``.

    A clamp-mode :class:`~cardiac_core.stimulus.stim.Stim` is NOT a current stimulus — it must be
    applied post-build via :meth:`CardiacSimulation.clamp_voltage`, never serialized into
    ``data.stimuli`` (it is a hard voltage override, not an Istim). This routes clamp Stims out so
    the CURRENT half flows through ``_build_mesh_data``/``_normalize_stimulus`` unchanged (dicts and
    current Stims coexist byte-identically), and returns the clamp Stims for the factory to apply.

    ``current`` is the (possibly empty→``None``) remainder passed to ``_build_mesh_data``; anything
    that is not a clamp Stim (dicts, current Stims) stays in it.
    """
    from .stimulus.stim import Stim  # local import — avoids an api<->stim module cycle at load
    if stimulus is None:
        return None, []
    items = list(stimulus) if isinstance(stimulus, (list, tuple)) else [stimulus]
    clamp = [s for s in items if isinstance(s, Stim) and s.mode == "clamp"]
    current = [s for s in items if not (isinstance(s, Stim) and s.mode == "clamp")]
    return (current or None), clamp


def _apply_clamp_stims(sim: 'CardiacSimulation', clamp_stims) -> 'CardiacSimulation':
    """Apply any clamp-mode Stims to a freshly-built sim via ``clamp_voltage`` (engine-agnostic)."""
    for cs in clamp_stims:
        sim.clamp_voltage(cs.mask, cs.clamp, start_time=cs.start_time, duration=cs.duration)
    return sim


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
        # for_lbm() emits EFFECTIVE D. Store RAW (× χ·Cm) so the lbm() factory's
        # χ·Cm division recovers it — keeps D_xx meaning consistent with the
        # create_cardiac_mesh path (Audit #21, round-2). Cm-safe: real Cm below.
        chi, Cm = conductivity.chi, emit['Cm']
        _eff = emit['D']
        if isinstance(_eff, tuple):
            D = tuple(d * (chi * Cm) for d in _eff)
        else:
            D = _eff * (chi * Cm)
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
        if not s['mask'].any():
            warnings.warn(
                "stimulus region selects 0 tissue nodes — it will have no effect "
                "(check the region shape / coordinate units in cm).",
                stacklevel=2,
            )

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


def _result_from(snaps, record, dx, dy, shape=None, sim=None):
    """Stack a list of SimulationSnapshot into a single SimulationResult.

    ``shape`` = ``(Nx, Ny)`` lets the zero-snapshot case return a rank-3 ``(0, Nx, Ny)``
    empty ``Vm`` so the analysis hooks degrade to NaN maps instead of crashing on a
    rank-1 ``(0,)`` tensor (F1, 2026-07-15).

    ``sim`` is the live ``CardiacSimulation`` — passed so this ``.run()`` path populates the
    Phase-1 analysis context (mask/boundary_mode/Cm/chi/conductivity/model identity)
    IDENTICALLY to the ``simulate()`` path (``run._collect``). Without it, ``.run()``-path
    results are silently mask/conductivity-unaware (a silent-wrong on scar domains, not a
    crash, since the fields default to ``None``).
    """
    from .run import SimulationResult  # local import avoids api<->run circular import
    from ._result_context import build_result_context
    if not snaps:
        empty_t = torch.empty(0, dtype=torch.float64)
        empty_v = (torch.empty(0, *shape, dtype=torch.float64)
                   if shape is not None else torch.empty(0))
        ctx = build_result_context(sim, empty_v.device)
        return SimulationResult(times=empty_t, Vm=empty_v, phi_e=None, dx=dx, dy=dy, **ctx)
    # B1: build ``times`` on the same device as the snapshots' Vm — on cuda the
    # snapshots carry Vm on GPU, so a CPU ``times`` would make every downstream
    # analysis/viz call (which indexes times by a GPU index) raise a device mismatch.
    times = torch.tensor([s.t for s in snaps], dtype=torch.float64,
                         device=snaps[0].Vm.device)
    Vm = torch.stack([s.Vm for s in snaps])
    phi_e = torch.stack([s.phi_e for s in snaps]) if snaps[0].phi_e is not None else None
    ionic = None
    if "ionic_states" in record and snaps[0].ionic_states is not None:
        ionic = torch.stack([s.ionic_states for s in snaps])
    ctx = build_result_context(sim, Vm.device)
    return SimulationResult(times=times, Vm=Vm, phi_e=phi_e, dx=dx, dy=dy,
                            ionic_states=ionic, **ctx)


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
    stencil: str = 'cardinal4',
    boundary_mode: str = 'face_mirror',
    theta: Optional[float] = None,
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
        if mesh is not None:
            raise TypeError("pass a mesh positionally OR as mesh=, not both (Audit #17)")
        mesh, geometry = geometry, None
    if mesh is not None:
        _clamp_stims = []          # mesh= path drops stimulus= (both current AND clamp) — pre-existing
        data = _resolve_mesh(mesh)
    else:
        # Route clamp-mode Stims out of the current stimulus (applied post-build via clamp_voltage).
        stimulus, _clamp_stims = _partition_stimulus(stimulus)
        data = _build_mesh_data(geometry, ionic_model, conductivity, stimulus, dt, 'monodomain')
    ionic = ionic_model or data.ionic_model
    timestep = dt or data.dt
    if theta is not None:   # S4 add-and-reject: theta is a bidomain-only knob
        raise ValueError("theta is bidomain-only (the θ-rule elliptic weighting); "
                         "monodomain uses diffusion_solver/splitting instead")

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
            stencil=stencil,
            boundary_mode=boundary_mode,
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
            stencil=stencil,
            boundary_mode=boundary_mode,
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
                        linear_solver=linear_solver, stencil=stencil, boundary_mode=boundary_mode,
                        device=device, ionic_model=ionic)
    return _apply_clamp_stims(
        CardiacSimulation(sim, 'monodomain', grid, data, build_kwargs,
                          boundary_mode=boundary_mode),
        _clamp_stims)


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
    splitting: str = 'strang',
    stencil: str = '5pt',
    diffusion_solver: Optional[str] = None,
    linear_solver: Optional[str] = None,
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
        if mesh is not None:
            raise TypeError("pass a mesh positionally OR as mesh=, not both (Audit #17)")
        mesh, geometry = geometry, None
    if mesh is not None:
        _clamp_stims = []          # mesh= path drops stimulus= (both current AND clamp) — pre-existing
        data = _resolve_mesh(mesh)
    else:
        # Route clamp-mode Stims out of the current stimulus (applied post-build via clamp_voltage).
        stimulus, _clamp_stims = _partition_stimulus(stimulus)
        data = _build_mesh_data(geometry, ionic_model, conductivity, stimulus, dt, 'bidomain')
    ionic = ionic_model or data.ionic_model
    timestep = dt or data.dt
    if boundary is not None and boundary not in ('bath', 'insulated'):
        raise ValueError(
            f"boundary must be 'bath', 'insulated', or None, got {boundary!r} "
            "(bidomain uses bath/insulated; LBM wall modes like 'ncs'/'scs' are LBM-only)"
        )
    for _knob, _val in (('diffusion_solver', diffusion_solver), ('linear_solver', linear_solver)):
        if _val is not None:   # S4 add-and-reject: these are monodomain-only knobs
            raise ValueError(f"{_knob} is monodomain-only; bidomain uses the parabolic/elliptic "
                             "split (elliptic_solver, theta) instead")
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
        if sigma_ratio != 3.59:   # non-default sigma_ratio is ignored here (S3)
            warnings.warn(
                f"sigma_ratio={sigma_ratio} is ignored when sigma_i/sigma_e are provided "
                "(the explicit conductivity wins on the declarative path)", UserWarning)
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

        # D_xx is RAW; effective diffusivity = D/(χ·Cm) (Audit #2/#8/#21). This
        # D_eff branch is reached only by the legacy create_cardiac_mesh→bidomain
        # path (the declarative path sets sigma_i/sigma_e → sigma branch above).
        chi_Cm = data.chi * data.Cm
        if is_isotropic:
            D_eff = float(data.D_xx.flat[0]) / chi_Cm
            r = sigma_ratio
            D_i = D_eff * (1 + r) / r
            D_e = D_eff * (1 + r)
            cond = BidomainConductivity(D_i=D_i, D_e=D_e)
        else:
            # Anisotropic: divide by χ·Cm, then apply ratio to each component
            r = sigma_ratio
            dev = torch.device(device)
            D_i_xx = (data.D_xx / chi_Cm) * (1 + r) / r
            D_i_yy = (data.D_yy / chi_Cm) * (1 + r) / r
            D_i_xy = (data.D_xy / chi_Cm) * (1 + r) / r
            D_e_xx = (data.D_xx / chi_Cm) * (1 + r)
            D_e_yy = (data.D_yy / chi_Cm) * (1 + r)
            D_e_xy = (data.D_xy / chi_Cm) * (1 + r)

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
    spatial = BidomainFDMDiscretization(grid, cond, Cm=data.Cm, stencil=stencil)

    # Build stimulus
    dev = torch.device(device)
    stimulus = _build_stimulus_protocol_bidomain(data, grid, dev, torch.float64)

    # Build simulation
    sim = BidomainSimulation(
        spatial=spatial,
        ionic_model=ionic,
        stimulus=stimulus,
        dt=timestep,
        splitting=splitting,
        elliptic_solver=elliptic_solver,
        theta=theta,
        device=device,
    )

    build_kwargs = dict(dt=timestep, sigma_ratio=sigma_ratio, boundary=boundary,
                        elliptic_solver=elliptic_solver, theta=theta, splitting=splitting,
                        stencil=stencil, device=device, ionic_model=ionic)
    return _apply_clamp_stims(
        CardiacSimulation(sim, 'bidomain', grid, data, build_kwargs), _clamp_stims)


def lbm(
    geometry=None,
    ionic_model: Optional[str] = None,
    conductivity: Optional[ConductivityConfig] = None,
    stimulus=None,
    *,
    mesh: Union[str, CardiacMeshData, None] = None,
    dt: Optional[float] = None,
    lattice: str = 'd2q5',
    weights_mode: str = 'canonical',
    boundary: Optional[str] = None,
    alpha: float = 1.0,
    device: str = 'cpu',
) -> CardiacSimulation:
    """Create an LBM simulation.

    ``boundary`` selects the flat top/bottom wall mode (boundary_conduction_speedup). The
    default (``None``) is lattice-aware: generic 'neumann' bounce-back on d2q5, and the 'hbb'
    flat-wall baseline on d2q9. The D2Q9-ONLY flat-wall family is 'hbb' (the specular baseline),
    'specular_nextcell' (a.k.a. 'ncs' — next-cell specular, zero bias), 'specular_samecell'
    (a.k.a. 'scs' — same-cell specular, inverse crescent), and 'combined' (HBB↔same-cell blend
    via ``alpha``: 1=HBB … 0=same-cell specular — the β-controlled curvature knob). Requesting
    any of these on lattice='d2q5' raises. Corners + east/west stay HBB.

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
        if mesh is not None:
            raise TypeError("pass a mesh positionally OR as mesh=, not both (Audit #17)")
        mesh, geometry = geometry, None
    if mesh is not None:
        _clamp_stims = []          # mesh= path drops stimulus= (both current AND clamp) — pre-existing
        data = _resolve_mesh(mesh)
    else:
        # Route clamp-mode Stims out of the current stimulus (applied post-build via clamp_voltage).
        stimulus, _clamp_stims = _partition_stimulus(stimulus)
        data = _build_mesh_data(geometry, ionic_model, conductivity, stimulus, dt, 'lbm')
    ionic_name = ionic_model or data.ionic_model
    timestep = dt or data.dt
    # Lattice-aware default boundary: d2q9 defaults to the HBB flat-wall baseline; every
    # other lattice defaults to generic neumann bounce-back (user 2026-07-15). hbb is now
    # D2Q9-only (wall_modes.D2Q9_ONLY), so an explicit hbb on d2q5 is rejected downstream.
    if boundary is None:
        boundary = 'hbb' if lattice == 'd2q9' else 'neumann'
    if alpha != 1.0 and boundary in ('neumann', 'hbb'):
        warnings.warn(
            f"alpha={alpha} is inert for boundary={boundary!r} "
            "(alpha only affects the 'combined' wall mode)", UserWarning)

    # Construct from the vendored LBM solver + shared ionic (self-contained; no _prepare_engine).
    from ._lbm.simulation import LBMSimulation
    from .ionic.registry import build_ionic_model

    # Instantiate ionic model via the shared registry (C3); LBM passes no cell_type → ENDO.
    if ionic_model is not None and not isinstance(ionic_model, str):
        ionic_instance = ionic_model            # pre-built (e.g. tuner-scaled) IonicModel — use as-is
    else:
        ionic_instance = build_ionic_model(ionic_name, device=device)

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
            "Oblique fibers (D_xy != 0) are not yet wired — they need the "
            "moment-space rotation of s_jx/s_jy (Audit #46)."
        )
    # D_xx is RAW; the membrane-effective diffusivity is D/(χ·Cm) (Audit #2/#8/#21).
    _chi_Cm = data.chi * data.Cm
    D_xx = float(data.D_xx.flat[0]) / _chi_Cm
    D_yy = float(data.D_yy.flat[0]) / _chi_Cm
    anisotropic = not np.isclose(D_xx, D_yy)
    if anisotropic and lattice != 'd2q9':
        warnings.warn(
            f"anisotropic D (D_xx != D_yy) forces lattice='d2q9'+MRT; the requested "
            f"lattice={lattice!r} is overridden", UserWarning)

    Nx, Ny = data.mask.shape
    # I1: a masked interior hole needs bounce-back on its rim (union with the outer
    # rect edges); None for a full mask → engine default rect masks (golden-safe).
    bounce = _lbm_bounce_masks(data, lattice, anisotropic, device)
    if anisotropic:
        sim = LBMSimulation(
            Nx=Nx, Ny=Ny,
            dx=data.dx, dt=timestep,
            D=D_xx, D_yy=D_yy, ionic_model=ionic_instance,
            Cm=data.Cm,
            lattice='d2q9', collision='mrt', weights_mode=weights_mode,
            boundary=boundary, alpha=alpha,
            bounce_masks=bounce,
        )
    else:
        sim = LBMSimulation(
            Nx=Nx, Ny=Ny,
            dx=data.dx, dt=timestep,
            D=D_xx, ionic_model=ionic_instance,
            Cm=data.Cm,
            lattice=lattice, weights_mode=weights_mode,
            boundary=boundary, alpha=alpha,
            bounce_masks=bounce,
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

    build_kwargs = dict(dt=timestep, lattice=lattice, weights_mode=weights_mode,
                        device=device, ionic_model=ionic_name,
                        boundary=boundary, alpha=alpha)
    return _apply_clamp_stims(
        CardiacSimulation(sim, 'lbm', None, data, build_kwargs), _clamp_stims)
