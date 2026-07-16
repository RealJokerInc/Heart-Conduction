"""D2Q9 flat-wall boundary modes (boundary_conduction_speedup research → productized).

These OVERLAY the top/bottom non-corner DIAGONAL slots AFTER a full Neumann (HBB) pass
(which handles corners + east/west walls). D2Q9 slot convention: 5=NE, 6=NW, 7=SW, 8=SE.
Kernels ported verbatim from Research/Active/boundary_conduction_speedup/diag_lbm_specular.py
(slot maps verified against the lattice in BC_IMPLEMENTATION_AUDIT.md §1).

Modes:
  neumann              — generic halfway bounce-back (default; ANY lattice; no overlay; forward crescent)
  hbb                  — the D2Q9 flat-wall HBB baseline; numerically == neumann (no overlay), but it is
                         the specular family's control and is therefore D2Q9-ONLY (see D2Q9_ONLY below)
  specular_nextcell    — next-cell specular: flip y, displace 1 cell east/west → ZERO bias (D2Q9-only)
  specular_samecell    — flip y, keep x, same cell → INVERSE crescent (== combined, alpha=0) (D2Q9-only)
  combined(alpha)      — HBB (alpha=1) ↔ same-cell specular (alpha=0) blend; the β-controlled
                         curvature knob (see KNOWLEDGE "Curvature control: the α-blend") (D2Q9-only)

All are diagonal→diagonal weight-matched → rest-neutral (no wall pre-charge) and mass-conserving.
Restricted to flat, axis-aligned top/bottom walls; corners + east/west stay HBB. NOTE (physics):
the same-cell-specular inverse branch is β = D·dt/dx² (τ) controlled — carry that at the API.
"""
from torch import Tensor

WALL_MODES = ('neumann', 'hbb', 'specular_nextcell', 'specular_samecell', 'combined')
# The D2Q9 flat-wall boundary family, restricted to lattice='d2q9'. The specular/combined
# modes act on diagonal populations (which d2q5 lacks); 'hbb' is the D2Q9 HBB baseline they
# are measured against — grouped here so it is not silently usable on d2q5 where only the
# generic 'neumann' bounce-back applies (user 2026-07-15).
D2Q9_ONLY = ('hbb', 'specular_nextcell', 'specular_samecell', 'combined')

# Standard abbreviations accepted as aliases: NCS = next-cell specular, SCS = same-cell specular.
_ALIASES = {'ncs': 'specular_nextcell', 'scs': 'specular_samecell'}


def normalize_mode(mode: str) -> str:
    """Map the standard abbreviations 'ncs'/'scs' to their canonical mode names."""
    return _ALIASES.get(mode, mode)


def apply_specular_nextcell_d2q9(f: Tensor, f_star: Tensor, NX: int, NY: int) -> Tensor:
    """Next-cell specular (zero bias): flip y, displace one cell. Non-corner cells only."""
    # TOP: NE(i)→SE(i+1), NW(i)→SW(i-1)
    f[8, 2:NX - 1, NY - 1] = f_star[5, 1:NX - 2, NY - 1]
    f[7, 1:NX - 2, NY - 1] = f_star[6, 2:NX - 1, NY - 1]
    # BOTTOM (y-mirror): SE(i)→NE(i+1), SW(i)→NW(i-1)
    f[5, 2:NX - 1, 0] = f_star[8, 1:NX - 2, 0]
    f[6, 1:NX - 2, 0] = f_star[7, 2:NX - 1, 0]
    return f


def apply_combined_d2q9(f: Tensor, f_star: Tensor, NX: int, NY: int, alpha: float) -> Tensor:
    """HBB (alpha=1) ↔ same-cell specular (alpha=0) blend. Non-corner cells only.

    Top:  f7(SW) = a·f*5(NE) + b·f*6(NW);  f8(SE) = a·f*6(NW) + b·f*5(NE)   (b = 1-a)
    At a=1 this reproduces the HBB fill exactly (bit-identical to the Neumann pass).
    """
    a = float(alpha)
    b = 1.0 - a
    f[7, 1:NX - 1, NY - 1] = a * f_star[5, 1:NX - 1, NY - 1] + b * f_star[6, 1:NX - 1, NY - 1]
    f[8, 1:NX - 1, NY - 1] = a * f_star[6, 1:NX - 1, NY - 1] + b * f_star[5, 1:NX - 1, NY - 1]
    f[5, 1:NX - 1, 0] = a * f_star[7, 1:NX - 1, 0] + b * f_star[8, 1:NX - 1, 0]
    f[6, 1:NX - 1, 0] = a * f_star[8, 1:NX - 1, 0] + b * f_star[7, 1:NX - 1, 0]
    return f


def apply_wall_overlay(f: Tensor, f_star: Tensor, mode: str, alpha: float,
                       NX: int, NY: int) -> Tensor:
    """Overlay a top/bottom wall mode. Call AFTER apply_neumann_d2q9. No-op for neumann/hbb."""
    mode = normalize_mode(mode)
    if mode in ('neumann', 'hbb'):
        return f
    if mode == 'specular_nextcell':
        return apply_specular_nextcell_d2q9(f, f_star, NX, NY)
    if mode == 'specular_samecell':
        return apply_combined_d2q9(f, f_star, NX, NY, 0.0)
    if mode == 'combined':
        return apply_combined_d2q9(f, f_star, NX, NY, alpha)
    raise ValueError(f"unknown boundary mode {mode!r}; valid: {WALL_MODES}")
