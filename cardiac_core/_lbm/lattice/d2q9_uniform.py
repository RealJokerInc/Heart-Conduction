"""D2Q9 lattice with UNIFORM weights — non-canonical variant for connectivity studies.

Provided as a controlled contrast against canonical D2Q9, to isolate the effect of the
weight distribution across lattice directions. NOT a standard LBM scheme. Deviations from
canonical D2Q9:

  - The rest particle (index 0) has weight 0, so f[0] is driven to zero
    each step. This makes the lattice effectively 8-velocity (D2Q8-like)
    but we keep the 9-velocity index space for code-path compatibility.

  - The fourth moment Σ w_i e_iα²·e_iβ² is NOT 2·cs²²·δ_αβ as required
    for Galilean-isotropic Navier-Stokes. For pure DIFFUSION (no advection),
    only the second moment matters, so this lattice still recovers the heat
    equation correctly. For coupled NS / advection-diffusion work it would
    be invalid.

  - Direction order is IDENTICAL to canonical D2Q9 (rest, E, W, N, S, NE,
    NW, SW, SE). Only `w` and `cs2` differ. `opposite` is identical so
    bounce-back BC works without modification.

  - cs2 = 0.75 derived from the second moment:
      Σ w_i · e_iα² = (1/8)·(2·1²) [cardinals] + (1/8)·(4·1²) [diagonals]
                    = 2/8 + 4/8 = 6/8 = 0.75   for both x and y axes
      Σ w_i · e_ix · e_iy = 0  (by symmetry; verified in test)
"""

from .base import Lattice


class D2Q9_uniform(Lattice):
    Q = 9
    cs2 = 0.75    # second-moment-derived; see module docstring

    e = (
        (0, 0),     # 0: rest
        (1, 0),     # 1: east
        (-1, 0),    # 2: west
        (0, 1),     # 3: north
        (0, -1),    # 4: south
        (1, 1),     # 5: NE
        (-1, 1),    # 6: NW
        (-1, -1),   # 7: SW
        (1, -1),    # 8: SE
    )

    # Uniform weights on 8 moving particles; rest particle is zero by design.
    w = (
        0.0,                                    # rest (deliberately dead)
        1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0,     # cardinals (E, W, N, S)
        1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0,     # diagonals (NE, NW, SW, SE)
    )

    # Identical to canonical D2Q9 — required for bounce-back to work correctly.
    opposite = (0, 2, 1, 4, 3, 7, 8, 5, 6)
