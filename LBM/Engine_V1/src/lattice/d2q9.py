"""D2Q9 lattice definition — 9 velocities (rest + 4 cardinal + 4 diagonal)."""

from .base import Lattice


class D2Q9(Lattice):
    Q = 9
    cs2 = 1.0 / 3.0

    e = (
        (0, 0),     # 0: rest
        (1, 0),     # 1: east   (+x)
        (-1, 0),    # 2: west   (-x)
        (0, 1),     # 3: north  (+y)
        (0, -1),    # 4: south  (-y)
        (1, 1),     # 5: NE     (+x, +y)
        (-1, 1),    # 6: NW     (-x, +y)
        (-1, -1),   # 7: SW     (-x, -y)
        (1, -1),    # 8: SE     (+x, -y)
    )

    w = (
        4.0 / 9.0,    # rest
        1.0 / 9.0,    # east
        1.0 / 9.0,    # west
        1.0 / 9.0,    # north
        1.0 / 9.0,    # south
        1.0 / 36.0,   # NE
        1.0 / 36.0,   # NW
        1.0 / 36.0,   # SW
        1.0 / 36.0,   # SE
    )

    opposite = (0, 2, 1, 4, 3, 7, 8, 5, 6)
