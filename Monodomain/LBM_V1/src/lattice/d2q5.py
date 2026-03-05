"""D2Q5 lattice definition — 5 velocities (rest + 4 cardinal)."""

from .base import Lattice


class D2Q5(Lattice):
    Q = 5
    cs2 = 1.0 / 3.0

    e = (
        (0, 0),    # 0: rest
        (1, 0),    # 1: east  (+x)
        (-1, 0),   # 2: west  (-x)
        (0, 1),    # 3: north (+y)
        (0, -1),   # 4: south (-y)
    )

    w = (
        1.0 / 3.0,   # rest
        1.0 / 6.0,   # east
        1.0 / 6.0,   # west
        1.0 / 6.0,   # north
        1.0 / 6.0,   # south
    )

    opposite = (0, 2, 1, 4, 3)
