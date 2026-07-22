"""``cardiac_core.fields`` — the analysis-fields layer.

* :mod:`cardiac_core.fields.derivatives` — the primitive operator toolkit
  (``grad``/``div``/``curl``/``laplacian`` + the ``diffusion_term`` source_sink core), torch,
  on-device, boundary/mask-aware.
* :class:`VectorField` — a light wrapper over a ``(..., 2)`` vector tensor.
* :class:`Fields` — the ``r.fields`` accessor: lazily-computed, cached NAMED physical fields on a
  ``SimulationResult`` (Phase 3: the Vm/φ_e fields ``voltage_gradient``/``voltage_flux``/
  ``source_sink``/``electric_field``/``current_flux``; the LAT-based velocity/curvature/vorticity
  family lands in Phase 4), plus a grid-bound ``r.fields.derivatives`` toolkit.
"""

import torch

from . import derivatives
from . import integrals as _integrals
from .derivatives import grad, div, curl, laplacian
from .lat_fields import lat_field_bundle

__all__ = ["grad", "div", "curl", "laplacian", "VectorField", "Fields"]


class VectorField:
    """Ergonomic wrapper over a ``(..., 2)`` vector tensor (x-, y-component on the trailing axis).

    ``.components`` is the raw tensor (npz-friendly, what's cached); ``.x``/``.y`` the components,
    ``.magnitude`` the L2 norm, ``.angle`` = ``atan2(y, x)``. So the caller never indexes a raw axis.
    """

    __slots__ = ("_v",)

    def __init__(self, v: torch.Tensor):
        self._v = v

    @property
    def components(self) -> torch.Tensor:
        return self._v

    @property
    def x(self) -> torch.Tensor:
        return self._v[..., 0]

    @property
    def y(self) -> torch.Tensor:
        return self._v[..., 1]

    @property
    def magnitude(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self._v, dim=-1)

    @property
    def angle(self) -> torch.Tensor:
        return torch.atan2(self._v[..., 1], self._v[..., 0])

    @property
    def shape(self):
        return self._v.shape

    @property
    def device(self):
        return self._v.device

    @property
    def dtype(self):
        return self._v.dtype

    def __repr__(self) -> str:
        return f"VectorField(shape={tuple(self._v.shape)})"


def _op_kw(r) -> dict:
    return dict(boundary_mode=getattr(r, 'boundary_mode', 'face_mirror'),
                mask=getattr(r, 'domain_mask', None))


class _BoundOperators:
    """``r.fields.derivatives`` — ``grad``/``div``/``curl``/``laplacian`` pre-bound to the result's
    ``dx``/``dy``/``boundary_mode``/``domain_mask`` (so the caller doesn't re-pass them). ``grad``
    returns a :class:`VectorField`; ``div``/``curl`` accept a ``VectorField`` or a raw ``(...,2)``."""

    def __init__(self, result):
        self._r = result

    @staticmethod
    def _raw(F):
        return F.components if isinstance(F, VectorField) else F

    def grad(self, f) -> VectorField:
        return VectorField(derivatives.grad(f, self._r.dx, self._r.dy, **_op_kw(self._r)))

    def div(self, F) -> torch.Tensor:
        return derivatives.div(self._raw(F), self._r.dx, self._r.dy, **_op_kw(self._r))

    def curl(self, F) -> torch.Tensor:
        return derivatives.curl(self._raw(F), self._r.dx, self._r.dy, **_op_kw(self._r))

    def laplacian(self, f) -> torch.Tensor:
        return derivatives.laplacian(f, self._r.dx, self._r.dy, **_op_kw(self._r))


class _BoundIntegrals:
    """``r.fields.integrals`` — global reductions pre-bound to the result's grid / mask / canonical
    LAT (Stokes / divergence-theorem partners of the ``derivatives`` operators)."""

    def __init__(self, result):
        self._r = result

    @staticmethod
    def _raw(f):
        return f.components if isinstance(f, VectorField) else f

    def _lat(self):
        from .. import analysis
        return analysis.activation_time(self._r.Vm, self._r.times)

    def region_integral(self, f, over=None):
        return _integrals.region_integral(self._raw(f), self._r.dx, self._r.dy, over=over)

    def net_flux(self, F, region=None):
        return _integrals.net_flux(F, self._r.dx, self._r.dy, region,
                                   mask=getattr(self._r, 'domain_mask', None))

    def circulation(self, v, region=None):
        return _integrals.circulation(v, self._r.dx, self._r.dy, region,
                                      boundary_mode=getattr(self._r, 'boundary_mode', 'face_mirror'),
                                      mask=getattr(self._r, 'domain_mask', None))

    def winding_number(self, phase, region=None):
        return _integrals.winding_number(phase, region)

    def conduction_time(self, a, b):
        return _integrals.conduction_time(self._lat(), a, b)

    def activated_area(self, threshold: float = -40.0, over=None):
        return _integrals.activated_area(self._r.Vm, self._r.dx, self._r.dy,
                                         threshold=threshold, over=over)

    def state_fraction(self, threshold: float = -40.0, over=None):
        return _integrals.state_fraction(self._r.Vm, threshold=threshold, over=over)

    def wavefront_length(self, at_time=None, level=None):
        lv = level if level is not None else at_time
        return _integrals.wavefront_length(self._lat(), lv, self._r.dx, self._r.dy)

    def global_curvature(self, at_time=None, level=None):
        lv = level if level is not None else at_time
        return _integrals.global_curvature(self._lat(), lv, self._r.dx, self._r.dy)


class Fields:
    """The ``r.fields`` accessor — lazily-computed, cached named physical fields.

    ``SimulationResult`` is an IMMUTABLE post-run snapshot (its ``Vm``/``phi_e`` can't change), so
    each field is computed once and cached for the result's lifetime — there is nothing to
    invalidate (``scale_conductance``/``reset`` are SIM methods that produce a NEW result → a fresh
    accessor). Vector fields are :class:`VectorField`; scalar fields are plain tensors.
    """

    def __init__(self, result):
        self._r = result
        self._cache: dict = {}

    @property
    def derivatives(self) -> _BoundOperators:
        """The grid-bound operator toolkit (``.grad``/``.div``/``.curl``/``.laplacian``)."""
        return _BoundOperators(self._r)

    @property
    def integrals(self) -> _BoundIntegrals:
        """The grid-bound reduction toolkit (``.region_integral``/``.net_flux``/``.circulation``/
        ``.winding_number``/``.conduction_time``/``.activated_area``/``.wavefront_length``/…)."""
        return _BoundIntegrals(self._r)

    def _cached(self, name, fn):
        if name not in self._cache:
            self._cache[name] = fn()
        return self._cache[name]

    # ---------------------------------------------------------------- Vm-based (monodomain) fields
    @property
    def voltage_gradient(self) -> VectorField:
        """∇Vm — steepest-ascent of V (large at the front). ``(T, Nx, Ny, 2)``."""
        return self._cached('voltage_gradient', lambda: VectorField(
            derivatives.grad(self._r.Vm, self._r.dx, self._r.dy, **_op_kw(self._r))))

    @property
    def voltage_flux(self) -> VectorField:
        """D_eff·∇Vm — the diffusion flux of voltage (``div`` of it ≈ ``source_sink``). ``(T,Nx,Ny,2)``."""
        def _f():
            r = self._r
            D = self._require_D_eff('voltage_flux')
            g = derivatives.grad(r.Vm, r.dx, r.dy, **_op_kw(r))
            return VectorField(D.unsqueeze(-1) * g)
        return self._cached('voltage_flux', _f)

    @property
    def source_sink(self) -> torch.Tensor:
        """∇·(D_eff∇Vm) — the electrotonic source–sink map (the SF numerator). ``(T, Nx, Ny)``.

        Conservative staggered operator with harmonic face-D — equals the solver's own diffusion
        term. Monodomain + isotropic only (raises otherwise)."""
        def _f():
            r = self._r
            D = self._require_D_eff('source_sink')
            return derivatives.diffusion_term(r.Vm, D, r.dx, r.dy, **_op_kw(r))
        return self._cached('source_sink', _f)

    # ---------------------------------------------------------------- LAT-based fields (Phase 4)
    def _lat_bundle(self) -> dict:
        """All LAT-derived fields computed once (they share ∇T) and cached."""
        return self._cached('_lat_bundle', lambda: lat_field_bundle(self._r))

    @property
    def velocity(self) -> VectorField:
        """∇T/|∇T|² — the conduction-velocity vector field (cm/s). ``(Nx, Ny, 2)``, NaN at gated nodes."""
        return VectorField(self._lat_bundle()['velocity'])

    @property
    def direction(self) -> VectorField:
        """n̂ = ∇T/|∇T| — unit propagation direction. ``(Nx, Ny, 2)``."""
        return VectorField(self._lat_bundle()['direction'])

    @property
    def speed(self) -> torch.Tensor:
        """1/|∇T| — front-normal conduction speed (cm/s). ``(Nx, Ny)``, NaN at gated nodes."""
        return self._lat_bundle()['speed']

    @property
    def curvature(self) -> torch.Tensor:
        """κ = ∇·n̂ — wavefront curvature (Osher–Sethian). ``(Nx, Ny)``."""
        return self._lat_bundle()['curvature']

    @property
    def divergence(self) -> torch.Tensor:
        """∇·n̂ — the collision/focus gating signal (== ``curvature``). ``(Nx, Ny)``."""
        return self._lat_bundle()['divergence']

    @property
    def vorticity(self) -> torch.Tensor:
        """curl(velocity) — rotation → rotor cores. ``(Nx, Ny)``."""
        return self._lat_bundle()['vorticity']

    @property
    def quality(self) -> torch.Tensor:
        """SG fit residual — confidence map (0 on a smooth front, high at collisions). ``(Nx, Ny)``."""
        return self._lat_bundle()['quality']

    @property
    def mask(self) -> torch.Tensor:
        """Boolean gate (True = low |∇T| / collision / window touches a block / high residual). ``(Nx, Ny)``."""
        return self._lat_bundle()['mask']

    # ---------------------------------------------------------------- φ_e-based (bidomain) fields
    @property
    def electric_field(self) -> VectorField:
        """−∇φ_e — the extracellular E-field (bidomain only). ``(T, Nx, Ny, 2)``."""
        def _f():
            r = self._r
            phi = self._require_phi_e('electric_field')
            return VectorField(-derivatives.grad(phi, r.dx, r.dy, **_op_kw(r)))
        return self._cached('electric_field', _f)

    @property
    def current_flux(self) -> VectorField:
        """−σ_e·∇φ_e — the extracellular current field (bidomain only). ``(T, Nx, Ny, 2)``."""
        def _f():
            r = self._r
            phi = self._require_phi_e('current_flux')
            sig = self._require_sigma_e('current_flux')          # (xx, yy, xy) fields
            g = derivatives.grad(phi, r.dx, r.dy, **_op_kw(r))
            flux = -torch.stack([sig[0] * g[..., 0], sig[1] * g[..., 1]], dim=-1)
            return VectorField(flux)
        return self._cached('current_flux', _f)

    # ---------------------------------------------------------------- guards
    def _require_D_eff(self, field):
        r = self._r
        if r.phi_e is not None:      # bidomain — keyed on phi_e (robust incl. legacy D-based bidomain)
            raise ValueError(
                f"{field} is a monodomain quantity ∇·(D∇V); this is a bidomain result — use "
                f"electric_field / current_flux for the φ_e current fields.")
        cond = r.conductivity
        if cond is None or getattr(cond, 'D_eff', None) is None:
            raise ValueError(
                f"{field} needs the effective diffusivity D_eff, which was not recorded on this "
                f"result.")
        if cond.is_anisotropic:
            raise ValueError(
                f"{field} = ∇·(D∇V) is scoped to ISOTROPIC D (scalar D_eff·∇²V); this result has an "
                f"anisotropic tensor (D_xy≠0 or D_xx≠D_yy) — the ∂_i(D_ij ∂_j V) contraction is not "
                f"yet implemented.")
        return cond.D_eff

    def _require_phi_e(self, field):
        r = self._r
        if r.phi_e is None:
            raise ValueError(
                f"{field} is a bidomain quantity (needs φ_e); this is a monodomain result. Use "
                f"voltage_gradient / source_sink instead.")
        return r.phi_e

    def _require_sigma_e(self, field):
        cond = self._r.conductivity
        if cond is None or getattr(cond, 'sigma_e', None) is None:
            raise ValueError(
                f"{field} needs σ_e (extracellular conductivity), not recorded on this result "
                f"(a legacy D-based bidomain carries no σ tuples).")
        return cond.sigma_e
