"""D2Q5 streaming step — roll-based periodic streaming.

Layer 2: pure function, torch.compile compatible.
"""

import torch
from torch import Tensor


def stream_d2q5(f: Tensor) -> Tensor:
    """Stream D2Q5 distributions via torch.roll (periodic boundaries).

    Pull convention: f_post[a](x) = f_pre[a](x - e_a).
    roll(shifts=+s) gives output[i] = input[i-s], so shift = +e_component.

    Direction mapping:
        0: rest   (0,0)   -> no shift
        1: east   (+1,0)  -> roll(dim=0, shifts=+1)
        2: west   (-1,0)  -> roll(dim=0, shifts=-1)
        3: north  (0,+1)  -> roll(dim=1, shifts=+1)
        4: south  (0,-1)  -> roll(dim=1, shifts=-1)

    Args:
        f: (5, Nx, Ny) pre-streaming distributions

    Returns:
        f_streamed: (5, Nx, Ny) post-streaming distributions
    """
    f_out = torch.empty_like(f)
    f_out[0] = f[0]                                   # rest: no shift
    f_out[1] = torch.roll(f[1], shifts=+1, dims=0)    # east: pull from x-1
    f_out[2] = torch.roll(f[2], shifts=-1, dims=0)    # west: pull from x+1
    f_out[3] = torch.roll(f[3], shifts=+1, dims=1)    # north: pull from y-1
    f_out[4] = torch.roll(f[4], shifts=-1, dims=1)    # south: pull from y+1
    return f_out
