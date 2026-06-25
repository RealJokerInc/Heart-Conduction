"""
ConductivityConfig — the chi/Cm + Formulation-A/B firewall.

The single owner of conductivity, ``chi``, and ``Cm`` in cardiac_core. Stores **physics**
(raw conductivities ``sigma_*`` in mS/cm, ``chi``, ``Cm``, ``fiber_angle``) and emits the
diffusion input each engine actually wants via ``for_monodomain()`` / ``for_bidomain()`` /
``for_lbm()``. This is the ONLY place the Formulation-A/B asymmetry lives — the user never sees it.

Design + verified arithmetic: ``Research/Active/engine_consolidation/API_DESIGN.md`` §4 and
``API_REFERENCE.md`` "Conductivity". The Cm-handling was gate-verified to machine precision and
against a live V5.5 cable CV (2026-06-24) — see ``Monodomain/Engine_V5.5/_probe_conductivity_firewall.py``,
whose arithmetic this module mirrors exactly.

Units
-----
``sigma_*`` are raw **conductivities in mS/cm**. The effective diffusivity
``D_eff = sigma_eff / (chi * Cm)`` (cm^2/ms) is **derived** — do NOT pass a pre-divided ``D`` as
``sigma``. ``0.00097`` is a *D_eff*, not a conductivity.

The Cm trap (do NOT regress)
----------------------------
The real ``Cm`` must reach EVERY engine (the reaction divides by it). Only the *diffusion input's*
Cm-scaling differs by formulation:

- ``for_monodomain`` (Form A): the engine scales diffusion by ``chi*Cm`` internally (mass term), so
  feed it the Cm-**un**scaled ``D = D_eff*Cm = sigma_eff/chi`` with engine ``chi=1`` (inert) and the
  **real** ``Cm``. Pinning the engine ``Cm=1`` here would silently break the reaction at Cm!=1.
- ``for_bidomain`` / ``for_lbm`` (Form B): feed fully-scaled ``D = sigma/(chi*Cm)`` + real ``Cm``.

At the pinned ``Cm=1`` all paths collapse to ``D = sigma_eff/chi``.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

Tensor3 = Tuple[float, float, float]


@dataclass(frozen=True)
class ConductivityConfig:
    """Physics in, per-engine diffusion inputs out. Build via the classmethods.

    Exactly one "mode" is populated:
    - isotropic   -> ``sigma_iso``
    - bidomain    -> ``sigma_i`` + ``sigma_e``
    - anisotropic -> ``sigma_l`` + ``sigma_t`` + ``fiber_angle``
    """

    sigma_i: Optional[float] = None      # bidomain intracellular conductivity (mS/cm)
    sigma_e: Optional[float] = None      # bidomain extracellular conductivity (mS/cm)
    sigma_l: Optional[float] = None      # anisotropic longitudinal (fiber) conductivity (mS/cm)
    sigma_t: Optional[float] = None      # anisotropic transverse conductivity (mS/cm)
    sigma_iso: Optional[float] = None    # isotropic single-domain effective conductivity (mS/cm)
    chi: float = 1400.0                  # surface-to-volume ratio (cm^-1)
    Cm: float = 1.0                      # membrane capacitance (uF/cm^2)
    fiber_angle: float = 0.0             # fiber angle (rad) — anisotropic only

    # ------------------------------------------------------------------ constructors
    @classmethod
    def isotropic(cls, sigma: float, chi: float = 1400.0, Cm: float = 1.0) -> "ConductivityConfig":
        """Single effective conductivity (mS/cm), isotropic."""
        return cls(sigma_iso=float(sigma), chi=float(chi), Cm=float(Cm))

    @classmethod
    def bidomain(cls, sigma_i: float, sigma_e: float,
                 chi: float = 1400.0, Cm: float = 1.0) -> "ConductivityConfig":
        """Paired intra/extra conductivities (mS/cm) — the most physical input."""
        return cls(sigma_i=float(sigma_i), sigma_e=float(sigma_e),
                   chi=float(chi), Cm=float(Cm))

    @classmethod
    def anisotropic(cls, sigma_l: float, sigma_t: float, fiber_angle: float,
                    chi: float = 1400.0, Cm: float = 1.0) -> "ConductivityConfig":
        """Fiber-aligned conductivity tensor: longitudinal/transverse + fiber angle (rad)."""
        return cls(sigma_l=float(sigma_l), sigma_t=float(sigma_t),
                   fiber_angle=float(fiber_angle), chi=float(chi), Cm=float(Cm))

    # ------------------------------------------------------------------ read-only properties
    @property
    def _is_anisotropic(self) -> bool:
        return self.sigma_l is not None and self.sigma_t is not None

    @property
    def sigma_eff(self) -> Union[float, Tensor3]:
        """Effective conductivity (mS/cm).

        - isotropic   -> the stored scalar
        - bidomain    -> harmonic i/e collapse ``sigma_i*sigma_e/(sigma_i+sigma_e)``
        - anisotropic -> rotated conductivity tensor ``(sigma_xx, sigma_yy, sigma_xy)`` (un-scaled)
        """
        if self.sigma_iso is not None:
            return self.sigma_iso
        if self.sigma_i is not None and self.sigma_e is not None:
            return self.sigma_i * self.sigma_e / (self.sigma_i + self.sigma_e)
        if self._is_anisotropic:
            cos_a = math.cos(self.fiber_angle)
            sin_a = math.sin(self.fiber_angle)
            sxx = self.sigma_t + (self.sigma_l - self.sigma_t) * cos_a ** 2
            syy = self.sigma_t + (self.sigma_l - self.sigma_t) * sin_a ** 2
            sxy = (self.sigma_l - self.sigma_t) * cos_a * sin_a
            return (sxx, syy, sxy)
        raise ValueError(
            "ConductivityConfig has no conductivity data — "
            "construct via .isotropic(...) / .bidomain(...) / .anisotropic(...)"
        )

    @property
    def D_eff(self) -> Union[float, Tensor3]:
        """True physical effective diffusivity (cm^2/ms): ``sigma_eff / (chi*Cm)``.

        Scalar for isotropic/bidomain; ``(Dxx, Dyy, Dxy)`` tuple for anisotropic
        (matches LBM ``sigma_to_D``).
        """
        s = self.sigma_eff
        scale = 1.0 / (self.chi * self.Cm)
        if isinstance(s, tuple):
            return tuple(v * scale for v in s)
        return s * scale

    # ------------------------------------------------------------------ per-engine emitters
    def for_monodomain(self) -> dict:
        """Form A: Cm-UNscaled ``D = D_eff*Cm = sigma_eff/chi``, engine ``chi=1`` (inert), real ``Cm``.

        The monodomain engine re-applies ``chi*Cm`` in its mass term, so the diffusion input must NOT
        be pre-divided by Cm; the real ``Cm`` still flows through to drive the reaction ``/Cm``.
        (Deleted when monodomain converts to Form B in the Phase-4 engine rewire.)
        """
        D = self.D_eff
        if isinstance(D, tuple):
            D = tuple(v * self.Cm for v in D)
        else:
            D = D * self.Cm
        return {'D': D, 'chi': 1.0, 'Cm': self.Cm}

    def for_bidomain(self) -> dict:
        """Form B: ``D_i, D_e = sigma_*/(chi*Cm)`` (fully scaled) + real ``Cm``."""
        if self.sigma_i is None or self.sigma_e is None:
            raise ValueError(
                "for_bidomain requires sigma_i and sigma_e — "
                "construct via ConductivityConfig.bidomain(sigma_i, sigma_e, ...)"
            )
        scale = 1.0 / (self.chi * self.Cm)
        return {'D_i': self.sigma_i * scale, 'D_e': self.sigma_e * scale, 'Cm': self.Cm}

    def for_lbm(self) -> dict:
        """Form B: ``D = D_eff`` (fully scaled) + real ``Cm``."""
        return {'D': self.D_eff, 'Cm': self.Cm}
