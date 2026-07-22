"""
One-shot simulation functions — run and return plain tensors.

No generators, no loops. Call once, get (times, V) back.

    times, V = run_monodomain(mesh, t_end=100.0)
    # times: (n_saves,)  V: (n_saves, Nx, Ny)
"""

import functools
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from .file_format import CardiacMeshData
from .api import monodomain, bidomain, lbm


@dataclass
class SimulationResult:
    """Plain tensor output from a one-shot simulation run.

    Attributes
    ----------
    times : torch.Tensor
        Save times (n_saves,) in ms.
    Vm : torch.Tensor
        Membrane potential history (n_saves, Nx, Ny). (Canonical name; ``.V`` is a read-only alias.)
    phi_e : torch.Tensor | None
        Extracellular potential history (n_saves, Nx, Ny). Bidomain only.
    dx, dy : float | None
        Grid spacing (cm) — set by the run helpers so analysis hooks have it.
    ionic_states : torch.Tensor | None
        Recorded ionic-state history (opt-in via ``record=``). None unless requested.

    Analysis context (Phase-1 ``analysis.fields``) — the edge/mask/conductivity/model the
    solver used, so the field ops honour them instead of guessing:

    domain_mask : torch.Tensor | None
        ``(Nx, Ny)`` bool active-tissue mask; ``None`` when the domain is full-rectangle.
    boundary_mode : str
        Ghost/mirror edge rule the field ops apply at the domain boundary (``"face_mirror"``
        ≙ scipy ``reflect`` no-flux). Matches the solver's boundary treatment.
    Cm, chi : float | None
        Membrane capacitance (µF/cm²) and surface-to-volume ratio (cm⁻¹) — the solver's
        ``chi*Cm`` mass factor, needed for ``D_eff`` and the safety-factor numerator.
    conductivity : cardiac_core._result_context.Conductivity | None
        Resolved effective diffusivity ``D_eff = D_raw/(chi*Cm)`` (+ raw tensor + σ tuples
        for bidomain) the ``voltage_flux``/``source_sink``/``current_flux`` fields need.
    ionic_model, cell_type : str | None
        Resolved model identity (name actually run + cell type), so Phase 3/7 can rebuild
        the model and re-evaluate ``I_ion`` for the reaction-identity / safety-factor.
    """
    times: torch.Tensor
    Vm: torch.Tensor
    phi_e: Optional[torch.Tensor] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    ionic_states: Optional[torch.Tensor] = None
    # --- analysis context (Phase 1) ---
    domain_mask: Optional[torch.Tensor] = None
    boundary_mode: str = "face_mirror"
    Cm: Optional[float] = None
    chi: Optional[float] = None
    conductivity: Optional[Any] = None
    ionic_model: Optional[str] = None
    cell_type: Optional[str] = None

    @property
    def V(self) -> torch.Tensor:
        """Read-only deprecated alias for :attr:`Vm`."""
        return self.Vm

    @functools.cached_property
    def fields(self):
        """Named physical fields + operator toolkit (``r.fields.source_sink``,
        ``r.fields.voltage_gradient``, ``r.fields.derivatives.grad(...)``, …).

        Lazily computed and cached for the result's lifetime (the result is an immutable post-run
        snapshot). See :class:`cardiac_core.fields.Fields`."""
        from .fields import Fields
        return Fields(self)

    # --- analysis hooks (thin delegators to cardiac_core.analysis) ---
    def cv(self, x1: int, x2: int, y: int, **kw) -> float:
        """Conduction velocity (cm/s) between x-indices ``x1`` and ``x2`` at row ``y``."""
        from . import analysis
        return analysis.conduction_velocity(self.Vm, self.times, self.dx, x1, x2, y, **kw)

    def apd(self, **kw) -> torch.Tensor:
        """APD map ``(Nx, Ny)`` (default APD90)."""
        from . import analysis
        return analysis.apd_map(self.Vm, self.times, **kw)

    def lat(self, **kw) -> torch.Tensor:
        """Local activation-time map ``(Nx, Ny)``."""
        from . import analysis
        return analysis.activation_time(self.Vm, self.times, **kw)

    def restitution(self, ix: int, iy: int, **kw):
        """APD restitution curve at node ``(ix, iy)`` from a multi-beat recording."""
        from . import analysis
        return analysis.restitution_curve(self.Vm, self.times, ix, iy, **kw)

    # --- P2 aggregate / per-beat / axis hooks ---
    def df_map(self) -> torch.Tensor:
        """Dominant-frequency map ``(Nx, Ny)`` in Hz (fibrillation analysis)."""
        from . import analysis
        return analysis.dominant_frequency_map(self.Vm, self.times)

    def cv_between(self, p1, p2, **kw) -> float:
        """Conduction velocity (cm/s) along the line between nodes ``p1=(ix,iy)`` and ``p2``."""
        from . import analysis
        return analysis.cv_between(self.Vm, self.times, p1, p2, self.dx, self.dy, **kw)

    def radial_cv(self, center, **kw) -> torch.Tensor:
        """Outward CV map ``(Nx, Ny)`` from a point source ``center=(ix,iy)``."""
        from . import analysis
        return analysis.radial_cv(self.Vm, self.times, center, self.dx, self.dy, **kw)

    def apd_per_beat(self, ix: int, iy: int, **kw) -> torch.Tensor:
        """Per-beat APD ``(n_beats,)`` at node ``(ix, iy)`` from a multi-beat run."""
        from . import analysis
        return analysis.apd_per_beat(self.Vm, self.times, ix, iy, **kw)

    def restitution_slope(self, ix: int, iy: int, **kw) -> dict:
        """Max restitution slope + DI* (alternans onset) at node ``(ix, iy)``."""
        from . import analysis
        DI, APD = analysis.restitution_curve(self.Vm, self.times, ix, iy, **kw)
        return analysis.restitution_slope(DI, APD)


def _collect(sim, t_end, save_every, output_device):
    """Run sim, collect snapshots, return tensors on output_device."""
    times = []
    V_list = []
    phi_e_list = []
    has_phi_e = False

    for snap in sim.snapshots(t_end, save_every):   # run() is now eager; snapshots() is the generator
        times.append(snap.t)
        V_list.append(snap.Vm)
        if snap.phi_e is not None:
            has_phi_e = True
            phi_e_list.append(snap.phi_e)

    if not V_list:
        # Zero save-points (e.g. t_end < save_every) — degrade like the eager run()
        # path instead of an IndexError on V_list[0] (Audit #5). Preserve the spatial
        # dims as a rank-3 (0, Nx, Ny) empty so the analysis hooks (.apd()/.lat()/.cv())
        # return NaN maps rather than crashing on a rank-1 (0,) tensor (F1, 2026-07-15).
        dev = torch.device(output_device) if output_device else torch.device('cpu')
        nx, ny = getattr(sim, '_Nx', None), getattr(sim, '_Ny', None)
        empty_t = torch.empty(0, dtype=torch.float64, device=dev)
        empty_v = (torch.empty(0, nx, ny, dtype=torch.float64, device=dev)
                   if nx is not None and ny is not None
                   else torch.empty(0, dtype=torch.float64, device=dev))
        return empty_t, empty_v, None

    dev = torch.device(output_device) if output_device else V_list[0].device

    times_t = torch.tensor(times, dtype=torch.float64, device=dev)
    V_t = torch.stack(V_list).to(dev)
    phi_e_t = torch.stack(phi_e_list).to(dev) if has_phi_e else None

    return times_t, V_t, phi_e_t


def run_monodomain(
    mesh: Union[str, CardiacMeshData],
    t_end: float,
    save_every: float = 1.0,
    *,
    ionic_model: Optional[str] = None,
    dt: Optional[float] = None,
    splitting: str = 'strang',
    diffusion_solver: str = 'crank_nicolson',
    linear_solver: str = 'pcg',
    device: str = 'cpu',
    output_device: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run monodomain simulation, return (times, V) as plain tensors.

    Parameters
    ----------
    mesh : str or CardiacMeshData
        Path to .npz or CardiacMeshData object.
    t_end : float
        Simulation end time (ms).
    save_every : float
        Save interval (ms).
    ionic_model, dt, splitting, diffusion_solver, linear_solver
        Forwarded to monodomain().
    device : str
        Compute device.
    output_device : str, optional
        Device for output tensors. None = same as device.

    Returns
    -------
    times : torch.Tensor
        (n_saves,) save times in ms.
    V : torch.Tensor
        (n_saves, Nx, Ny) membrane potential.
    """
    sim = monodomain(
        mesh, ionic_model=ionic_model, dt=dt, splitting=splitting,
        diffusion_solver=diffusion_solver, linear_solver=linear_solver,
        device=device,
    )
    times, V, _ = _collect(sim, t_end, save_every, output_device)
    return times, V


def run_bidomain(
    mesh: Union[str, CardiacMeshData],
    t_end: float,
    save_every: float = 1.0,
    *,
    ionic_model: Optional[str] = None,
    dt: Optional[float] = None,
    sigma_ratio: float = 3.59,
    boundary: Optional[str] = None,
    elliptic_solver: str = 'auto',
    theta: float = 0.5,
    device: str = 'cpu',
    output_device: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run bidomain simulation, return (times, V, phi_e) as plain tensors.

    Parameters
    ----------
    mesh : str or CardiacMeshData
        Path to .npz or CardiacMeshData object.
    t_end : float
        Simulation end time (ms).
    save_every : float
        Save interval (ms).
    ionic_model, dt, sigma_ratio, boundary, elliptic_solver, theta
        Forwarded to bidomain().
    device : str
        Compute device.
    output_device : str, optional
        Device for output tensors. None = same as device.

    Returns
    -------
    times : torch.Tensor
        (n_saves,) save times in ms.
    V : torch.Tensor
        (n_saves, Nx, Ny) membrane potential.
    phi_e : torch.Tensor
        (n_saves, Nx, Ny) extracellular potential.
    """
    sim = bidomain(
        mesh, ionic_model=ionic_model, dt=dt, sigma_ratio=sigma_ratio,
        boundary=boundary, elliptic_solver=elliptic_solver, theta=theta,
        device=device,
    )
    times, V, phi_e = _collect(sim, t_end, save_every, output_device)
    return times, V, phi_e


def run_lbm(
    mesh: Union[str, CardiacMeshData],
    t_end: float,
    save_every: float = 1.0,
    *,
    ionic_model: Optional[str] = None,
    dt: Optional[float] = None,
    lattice: str = 'd2q5',
    boundary: Optional[str] = None,
    alpha: float = 1.0,
    device: str = 'cpu',
    output_device: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run LBM simulation, return (times, V) as plain tensors.

    ``boundary`` defaults (``None``) lattice-aware: 'neumann' on d2q5, 'hbb' on d2q9. The
    D2Q9-only flat-wall modes ('hbb'/'ncs'/'scs'/'combined') require ``lattice='d2q9'``.

    Parameters
    ----------
    mesh : str or CardiacMeshData
        Path to .npz or CardiacMeshData object.
    t_end : float
        Simulation end time (ms).
    save_every : float
        Save interval (ms).
    ionic_model, dt, lattice, boundary, alpha
        Forwarded to lbm() (``boundary``/``alpha`` select the flat-wall mode).
    device : str
        Compute device.
    output_device : str, optional
        Device for output tensors. None = same as device.

    Returns
    -------
    times : torch.Tensor
        (n_saves,) save times in ms.
    V : torch.Tensor
        (n_saves, Nx, Ny) membrane potential.
    """
    sim = lbm(
        mesh, ionic_model=ionic_model, dt=dt, lattice=lattice,
        boundary=boundary, alpha=alpha, device=device,
    )
    times, V, _ = _collect(sim, t_end, save_every, output_device)
    return times, V


def simulate(
    mesh: Union[str, CardiacMeshData],
    t_end: float,
    save_every: float = 1.0,
    *,
    engine: str = 'monodomain',
    device: str = 'cpu',
    output_device: Optional[str] = None,
    **kwargs,
) -> SimulationResult:
    """Engine-agnostic one-shot simulation.

    Parameters
    ----------
    mesh : str or CardiacMeshData
        Path to .npz or CardiacMeshData object.
    t_end : float
        Simulation end time (ms).
    save_every : float
        Save interval (ms).
    engine : str
        'monodomain', 'bidomain', or 'lbm'.
    device : str
        Compute device.
    output_device : str, optional
        Device for output tensors. None = same as device.
    **kwargs
        Forwarded to the engine constructor (e.g. sigma_ratio, lattice, etc.).

    Returns
    -------
    SimulationResult
        .times (n_saves,), .V (n_saves, Nx, Ny), .phi_e (n_saves, Nx, Ny) or None.
    """
    constructors = {
        'monodomain': monodomain,
        'bidomain': bidomain,
        'lbm': lbm,
    }
    ctor = constructors.get(engine)
    if ctor is None:
        raise ValueError(f"Unknown engine: {engine}. Use 'monodomain', 'bidomain', or 'lbm'.")

    sim = ctor(mesh, device=device, **kwargs)
    times, V, phi_e = _collect(sim, t_end, save_every, output_device)
    from ._result_context import build_result_context
    ctx = build_result_context(sim, V.device)
    return SimulationResult(times=times, Vm=V, phi_e=phi_e, dx=sim.dx, dy=sim.dy, **ctx)
