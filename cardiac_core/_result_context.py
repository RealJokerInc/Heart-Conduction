"""Shared builder for the analysis-context fields carried on ``SimulationResult``.

The result must carry the SAME edge treatment, mask, conductivity, and Cm the solver
used, plus the resolved ionic-model identity — otherwise the field ops
(``source_sink = div(D_eff ∇V)``, the safety-factor ``Cm·ΔV`` numerator, and every
mask/boundary-aware stencil) are silently wrong.

BOTH result builders call :func:`build_result_context` so the ``.run()`` path
(``api._result_from``) and the ``simulate()`` path (``run._collect``) populate the new
fields identically. This module imports only numpy/torch (no ``run``/``api``) so it is
free of the api↔run circular-import hazard.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class Conductivity:
    """Resolved conductivity a :class:`SimulationResult` carries for the field ops.

    Attributes
    ----------
    chi, Cm : float
        The scalars the solver used; ``chi*Cm`` is the solver's mass factor, so
        ``D_eff = D_raw/(chi*Cm)`` (see ``api.py`` ``chi_Cm``).
    D_eff : torch.Tensor | None
        Effective diffusivity ``D_raw/(chi*Cm)`` (cm²/ms) — the quantity the monodomain
        / LBM solver actually steps ``∇·(D_eff∇V)`` with. ``(Nx, Ny)``. ``None`` on a
        bidomain result (bidomain carries ``sigma_i``/``sigma_e`` instead).
    D_raw, D_xx, D_yy, D_xy : torch.Tensor | None
        The RAW stored conductivity-tensor components ``(Nx, Ny)`` (``D_raw`` aliases
        ``D_xx``). Kept so the field layer can (a) recompute ``D_eff`` and (b) detect
        anisotropy — ``∇·(D∇V)`` is a tensor contraction, not ``D·∇²V``, when ``D_xy≠0``
        or ``D_xx≠D_yy``.
    sigma_i, sigma_e : tuple[torch.Tensor, ...] | None
        ``(xx, yy, xy)`` intra/extra-cellular conductivity fields — bidomain only;
        ``None`` for monodomain/LBM.
    """

    chi: float
    Cm: float
    D_eff: Optional[torch.Tensor] = None
    D_raw: Optional[torch.Tensor] = None
    D_xx: Optional[torch.Tensor] = None
    D_yy: Optional[torch.Tensor] = None
    D_xy: Optional[torch.Tensor] = None
    sigma_i: Optional[tuple] = None
    sigma_e: Optional[tuple] = None

    @property
    def is_bidomain(self) -> bool:
        """True when this holds bidomain conductivity (``sigma`` fields, no ``D_eff``)."""
        return self.sigma_i is not None and self.sigma_e is not None

    @property
    def is_anisotropic(self) -> bool:
        """True when the RAW diffusivity tensor is non-scalar (``D_xy≠0`` or ``D_xx≠D_yy``).

        The field layer uses this to reject the scalar ``D_eff·∇V`` ``source_sink`` route
        on an anisotropic field (which needs the full ``∂_i(D_ij ∂_j V)`` contraction).
        """
        if self.D_xx is None:
            return False
        if self.D_xy is not None and bool(torch.any(self.D_xy != 0)):
            return True
        return bool(torch.any(self.D_xx != self.D_yy))


def _to_tensor(a, device) -> torch.Tensor:
    """NumPy (or scalar) conductivity component -> float64 torch tensor on ``device``."""
    return torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device)


def _conductivity_from(data, device) -> Conductivity:
    """Resolve a :class:`Conductivity` from a ``CardiacMeshData`` on ``device``.

    ``D_eff = D_raw/(data.chi*data.Cm)`` — NEVER a hard-coded 1400: the declarative
    monodomain path sets ``data.chi=1`` (Form-A, ``ConductivityConfig.for_monodomain``),
    so a 1400 fallback would be 1400× wrong there. The bidomain branch carries the
    ``sigma_i``/``sigma_e`` tuples (mirrors ``_rebuild_with_conductivity``'s dual
    representation) and leaves ``D_eff`` None.
    """
    chi = float(data.chi)
    Cm = float(data.Cm)
    chi_Cm = chi * Cm

    D_xx = _to_tensor(data.D_xx, device)
    D_yy = _to_tensor(data.D_yy, device)
    D_xy = _to_tensor(data.D_xy, device)

    if data.sigma_i is not None and data.sigma_e is not None:
        # Bidomain: keep sigma tuples; no scalar D_eff (source_sink is monodomain-only).
        sig_i = tuple(_to_tensor(c, device) for c in data.sigma_i)
        sig_e = tuple(_to_tensor(c, device) for c in data.sigma_e)
        return Conductivity(chi=chi, Cm=Cm, D_eff=None, D_raw=D_xx,
                            D_xx=D_xx, D_yy=D_yy, D_xy=D_xy,
                            sigma_i=sig_i, sigma_e=sig_e)

    D_eff = D_xx / chi_Cm
    return Conductivity(chi=chi, Cm=Cm, D_eff=D_eff, D_raw=D_xx,
                        D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)


def _resolved_ionic_name(sim) -> Optional[str]:
    """The ionic-model NAME actually run (resolved override), not the stale mesh value.

    The factories compute ``ionic = ionic_model or data.ionic_model`` and stash it in
    ``build_kwargs['ionic_model']`` — so with an explicit ``ionic_model=`` override on a
    legacy mesh, THAT is the resolved name (``data.ionic_model`` is stale). Falls back to
    ``data.ionic_model`` when a pre-built model INSTANCE was injected (not a str).
    """
    name = None
    bk = getattr(sim, '_build_kwargs', None)
    if isinstance(bk, dict):
        name = bk.get('ionic_model')
    if not isinstance(name, str):
        data = getattr(sim, '_data', None)
        name = getattr(data, 'ionic_model', None)
    return name if isinstance(name, str) else None


def _cell_type_of(data, engine_type: str) -> str:
    """The cell type the engine ACTUALLY ran (so a Phase-7 ``I_ion`` re-eval rebuilds the right model).

    ONLY the monodomain factory threads the mesh's ``group_cell_types[0]`` into the ionic model
    (``api.py`` monodomain construction). The bidomain and LBM factories pass NO ``cell_type`` to
    ``build_ionic_model`` → the registry's ENDO default always wins there. So recording
    ``group_cell_types[0]`` for bidomain/LBM would disagree with the model actually run and make a
    downstream re-eval build the wrong cell type on the right trajectory. Force ENDO for those.
    """
    if engine_type == 'monodomain':
        gct = getattr(data, 'group_cell_types', None)
        return gct[0] if gct else 'ENDO'
    return 'ENDO'


def build_result_context(sim, device) -> dict:
    """Build the dict of analysis-context fields to pass to ``SimulationResult(...)``.

    ``device`` is the result's Vm device (the field ops are torch/on-device). Returns
    ``domain_mask`` (``None`` when the domain is full-rectangle — no mask needed),
    ``boundary_mode``, ``Cm``, ``chi``, ``conductivity``, ``ionic_model``, ``cell_type``.
    ``sim`` is the live ``CardiacSimulation``; a ``None`` ``sim`` yields ``{}`` (defaults
    preserve back-compat).
    """
    if sim is None:
        return {}
    data = getattr(sim, '_data', None)
    if data is None:
        return {}

    dev = torch.device(device) if device is not None else torch.device('cpu')

    # Mask: carry only a genuine sub-domain; a full-rectangle mask -> None (no-op for ops).
    mask_np = np.asarray(data.mask, dtype=bool)
    domain_mask = (None if bool(mask_np.all())
                   else torch.as_tensor(mask_np, dtype=torch.bool, device=dev))

    # boundary_mode: explicitly set on the sim by every factory (mono -> its arg;
    # bidomain/LBM -> 'face_mirror', the no-flux tissue-Vm edge rule the analysis stencil
    # honors regardless of the bath/insulated BoundarySpec or the LBM wall mode).
    boundary_mode = getattr(sim, '_boundary_mode', 'face_mirror')

    return dict(
        domain_mask=domain_mask,
        boundary_mode=boundary_mode,
        Cm=float(data.Cm),
        chi=float(data.chi),
        conductivity=_conductivity_from(data, dev),
        ionic_model=_resolved_ionic_name(sim),
        cell_type=_cell_type_of(data, getattr(sim, '_engine_type', 'monodomain')),
    )
