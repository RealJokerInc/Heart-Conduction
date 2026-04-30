"""D2Q9 streaming step — roll-based periodic streaming.

Layer 2: pure function, torch.compile compatible.
"""

import torch
from torch import Tensor


def stream_d2q9(f: Tensor) -> Tensor:
    """Stream D2Q9 distributions via torch.roll (periodic boundaries).

    Pull convention: f_post[a](x) = f_pre[a](x - e_a).
    roll(shifts=+s) gives output[i] = input[i-s], so shift = +e_component.

    Direction mapping:
        0: rest  (0,0)    -> no shift
        1: E     (+1,0)   -> roll(dim=0, +1)
        2: W     (-1,0)   -> roll(dim=0, -1)
        3: N     (0,+1)   -> roll(dim=1, +1)
        4: S     (0,-1)   -> roll(dim=1, -1)
        5: NE    (+1,+1)  -> roll(dim=0, +1) then roll(dim=1, +1)
        6: NW    (-1,+1)  -> roll(dim=0, -1) then roll(dim=1, +1)
        7: SW    (-1,-1)  -> roll(dim=0, -1) then roll(dim=1, -1)
        8: SE    (+1,-1)  -> roll(dim=0, +1) then roll(dim=1, -1)

    Args:
        f: (9, Nx, Ny) pre-streaming distributions

    Returns:
        f_streamed: (9, Nx, Ny) post-streaming distributions
    """
    f_out = torch.empty_like(f)
    f_out[0] = f[0]                                                         # rest
    f_out[1] = torch.roll(f[1], shifts=+1, dims=0)                          # E: pull from x-1
    f_out[2] = torch.roll(f[2], shifts=-1, dims=0)                          # W: pull from x+1
    f_out[3] = torch.roll(f[3], shifts=+1, dims=1)                          # N: pull from y-1
    f_out[4] = torch.roll(f[4], shifts=-1, dims=1)                          # S: pull from y+1
    f_out[5] = torch.roll(torch.roll(f[5], shifts=+1, dims=0), shifts=+1, dims=1)  # NE
    f_out[6] = torch.roll(torch.roll(f[6], shifts=-1, dims=0), shifts=+1, dims=1)  # NW
    f_out[7] = torch.roll(torch.roll(f[7], shifts=-1, dims=0), shifts=-1, dims=1)  # SW
    f_out[8] = torch.roll(torch.roll(f[8], shifts=+1, dims=0), shifts=-1, dims=1)  # SE
    return f_out
