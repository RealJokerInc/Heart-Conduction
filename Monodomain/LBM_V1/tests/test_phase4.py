"""Phase 4 validation: Boundary conditions."""

import sys
sys.path.insert(0, '.')

import torch
torch.set_default_dtype(torch.float64)

from src.lattice import D2Q5, D2Q9
from src.collision.bgk import bgk_collide
from src.streaming.d2q5 import stream_d2q5
from src.streaming.d2q9 import stream_d2q9
from src.state import recover_voltage
from src.boundary.masks import precompute_bounce_masks
from src.boundary.neumann import apply_neumann_d2q5, apply_neumann_d2q9
from src.boundary.dirichlet import apply_dirichlet_d2q5, apply_dirichlet_d2q9
from src.boundary.absorbing import apply_absorbing_d2q5
from src.diffusion import tau_from_D


def make_gaussian(Nx, Ny, cx, cy, sigma):
    """Create a 2D Gaussian voltage field."""
    x = torch.arange(Nx, dtype=torch.float64)
    y = torch.arange(Ny, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    return torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))


def make_rect_bounce_masks(Nx, Ny, lattice):
    """Create bounce masks for a rectangular domain filling the full grid.

    With periodic streaming (torch.roll), distributions wrap around. These
    masks mark edge nodes so bounce-back replaces the wrapped distributions.
    The wrapped values are exchanged symmetrically between opposite walls,
    preserving total mass exactly.
    """
    bounce_masks = {}
    for a in range(1, lattice.Q):
        m = torch.zeros(Nx, Ny, dtype=torch.bool)
        ex, ey = lattice.e[a]
        # Direction a is outgoing where moving in e_a exits the domain
        if ex == 1:   m[-1, :] = True   # east wall
        if ex == -1:  m[0, :] = True    # west wall
        if ey == 1:   m[:, -1] = True   # north wall
        if ey == -1:  m[:, 0] = True    # south wall
        bounce_masks[a] = m
    return bounce_masks


def run_diffusion_neumann(lattice, stream_fn, neumann_fn, Nx, Ny, n_steps,
                          D=0.1, dx=1.0, dt=1.0):
    """Run diffusion with Neumann BC and return total voltage over time.

    Uses full-grid domain with manual edge bounce masks. Periodic streaming
    wraps distributions; bounce-back replaces them, preserving mass.
    """
    w = torch.tensor(lattice.w)
    tau = tau_from_D(D, dx, dt)
    omega = 1.0 / tau

    bounce_masks = make_rect_bounce_masks(Nx, Ny, lattice)

    # Initial Gaussian
    V = make_gaussian(Nx, Ny, Nx // 2, Ny // 2, 3.0)
    f = w[:, None, None] * V[None, :, :]
    R = torch.zeros(Nx, Ny)

    totals = [V.sum().item()]
    for _ in range(n_steps):
        V = recover_voltage(f)
        f = bgk_collide(f, V, R, dt, omega, w)
        f_star = f.clone()
        f = stream_fn(f)
        f = neumann_fn(f, f_star, bounce_masks)
        totals.append(recover_voltage(f).sum().item())

    return totals


def test_4v1_neumann_conservation():
    """Neumann: Gaussian diffusion conserves total V."""
    d5 = D2Q5()
    totals = run_diffusion_neumann(d5, stream_d2q5, apply_neumann_d2q5,
                                   30, 20, 1000)
    initial = totals[0]
    max_drift = max(abs(t - initial) / abs(initial) for t in totals)
    assert max_drift < 1e-10, f"D2Q5 relative drift: {max_drift}"
    print(f"4-V1a PASS: D2Q5 Neumann conservation (drift={max_drift:.1e})")

    d9 = D2Q9()
    totals9 = run_diffusion_neumann(d9, stream_d2q9, apply_neumann_d2q9,
                                    30, 20, 1000)
    initial9 = totals9[0]
    max_drift9 = max(abs(t - initial9) / abs(initial9) for t in totals9)
    assert max_drift9 < 1e-10, f"D2Q9 relative drift: {max_drift9}"
    print(f"4-V1b PASS: D2Q9 Neumann conservation (drift={max_drift9:.1e})")

    # Note: Circular mask conservation requires full domain masking (collision +
    # streaming masking) which is a Phase 7 concern. Bounce-back alone handles
    # rectangular domains; irregular domains need the simulation orchestrator.


def test_4v2_dirichlet_steady_state():
    """Dirichlet: 1D linear profile between two fixed-V boundaries.

    Uses ghost nodes at x=0 and x=Nx-1, physical domain at x=1..Nx-2.
    Ghost nodes are set to equilibrium at V_bc each step (before collision),
    providing correct incoming distributions via periodic streaming.
    """
    N_phys = 50
    Nx = N_phys + 2  # physical + 2 ghost
    Ny = 1
    d5 = D2Q5()
    w5 = torch.tensor(d5.w)
    D = 0.1
    dx, dt = 1.0, 1.0
    tau = tau_from_D(D, dx, dt)
    omega = 1.0 / tau

    mask = torch.ones(Nx, Ny, dtype=torch.bool)
    mask[0, :] = False
    mask[Nx - 1, :] = False
    dir_masks = precompute_bounce_masks(mask, d5)

    V_left, V_right = 0.0, 1.0
    V_bc = torch.zeros(Nx, Ny)
    V_bc[1, :] = V_left
    V_bc[Nx - 2, :] = V_right

    V = torch.full((Nx, Ny), 0.5) * mask.float()
    f = w5[:, None, None] * V[None, :, :]
    R = torch.zeros(Nx, Ny)

    for _ in range(10000):
        # Fix ghost nodes to equilibrium at their V_bc value
        f[:, 0, :] = w5[:, None] * V_left
        f[:, Nx - 1, :] = w5[:, None] * V_right

        V = recover_voltage(f)
        f = bgk_collide(f, V, R, dt, omega, w5)
        f_star = f.clone()
        f = stream_d2q5(f)
        f = apply_dirichlet_d2q5(f, f_star, dir_masks, V_bc, w5)

    V_final = recover_voltage(f).squeeze()
    V_line = V_final[1:Nx - 1]  # physical nodes only

    # Validate linearity: fit a line and check max deviation
    x = torch.arange(N_phys, dtype=torch.float64)
    slope = (V_line[-1] - V_line[0]) / (N_phys - 1)
    V_fit = V_line[0] + slope * x
    linearity_err = (V_line - V_fit).abs().max().item()
    assert linearity_err < 5e-3, f"Profile not linear: {linearity_err}"

    # Validate endpoints approach V_left and V_right
    assert abs(V_line[0].item() - V_left) < 0.02, f"Left endpoint off: {V_line[0].item()}"
    assert abs(V_line[-1].item() - V_right) < 0.02, f"Right endpoint off: {V_line[-1].item()}"

    # Validate monotonicity
    diffs = V_line[1:] - V_line[:-1]
    assert (diffs > 0).all(), "Profile not monotonically increasing"

    print(f"4-V2 PASS: Dirichlet steady state (linearity={linearity_err:.2e}, "
          f"V[0]={V_line[0].item():.4f}, V[-1]={V_line[-1].item():.4f})")


def test_4v3_absorbing():
    """Absorbing: pulse exits without significant reflection.

    Uses full-grid domain with manual edge bounce masks.
    Gaussian near right wall with absorbing BC on all walls.
    """
    Nx, Ny = 100, 5
    d5 = D2Q5()
    w5 = torch.tensor(d5.w)
    D = 0.1
    dx, dt = 1.0, 1.0
    tau = tau_from_D(D, dx, dt)
    omega = 1.0 / tau

    bounce_masks = make_rect_bounce_masks(Nx, Ny, d5)

    # Gaussian near right boundary
    V = make_gaussian(Nx, Ny, 85, Ny // 2, 5.0)
    initial_energy = V.sum().item()
    f = w5[:, None, None] * V[None, :, :]
    R = torch.zeros(Nx, Ny)

    for _ in range(500):
        V = recover_voltage(f)
        f = bgk_collide(f, V, R, dt, omega, w5)
        f_star = f.clone()
        f = stream_d2q5(f)
        f = apply_absorbing_d2q5(f, bounce_masks, V, w5)

    V_final = recover_voltage(f)
    final_energy = V_final.sum().item()
    remaining = final_energy / initial_energy
    # Check: very little energy reflected back to the left half
    left_energy = V_final[:50].sum().item() / initial_energy
    print(f"    Remaining energy: {remaining:.4f}, Left-half reflected: {left_energy:.4f}")
    assert left_energy < 0.05, f"Too much reflected energy: {left_energy}"
    print(f"4-V3 PASS: Absorbing BC (remaining={remaining:.4f}, left={left_energy:.4f})")


def test_4v4_mixed_bc():
    """Mixed: Dirichlet top/bottom + Neumann left/right -> vertical gradient.

    Full-grid domain with manual edge bounce masks split by wall type:
    - North/south walls: Dirichlet (V=0 at south, V=1 at north)
    - East/west walls: Neumann (no flux)
    Expected: linear profile in y, uniform in x.
    """
    Nx, Ny = 20, 30
    d5 = D2Q5()
    w5 = torch.tensor(d5.w)
    D = 0.1
    dx, dt = 1.0, 1.0
    tau = tau_from_D(D, dx, dt)
    omega = 1.0 / tau

    all_masks = make_rect_bounce_masks(Nx, Ny, d5)

    # Split: Neumann on east/west (dirs 1,2), Dirichlet on north/south (dirs 3,4)
    neu_masks = {}
    dir_masks = {}
    for a in range(1, 5):
        if a in (1, 2):  # east/west -> Neumann
            neu_masks[a] = all_masks[a]
            dir_masks[a] = torch.zeros(Nx, Ny, dtype=torch.bool)
        else:  # north/south -> Dirichlet
            dir_masks[a] = all_masks[a]
            neu_masks[a] = torch.zeros(Nx, Ny, dtype=torch.bool)

    V_bottom, V_top = 0.0, 1.0
    V_bc = torch.zeros(Nx, Ny)
    V_bc[:, 0] = V_bottom      # south wall
    V_bc[:, Ny - 1] = V_top    # north wall

    V = torch.full((Nx, Ny), 0.5)
    f = w5[:, None, None] * V[None, :, :]
    R = torch.zeros(Nx, Ny)

    for _ in range(5000):
        V = recover_voltage(f)
        f = bgk_collide(f, V, R, dt, omega, w5)
        f_star = f.clone()
        f = stream_d2q5(f)
        f = apply_neumann_d2q5(f, f_star, neu_masks)
        f = apply_dirichlet_d2q5(f, f_star, dir_masks, V_bc, w5)

    V_final = recover_voltage(f)

    # Check x-uniformity
    col_spread = V_final.std(dim=0).max().item()
    assert col_spread < 1e-6, f"Not x-uniform: std={col_spread}"

    # Check y-linearity: fit a line
    V_avg = V_final.mean(dim=0)
    y = torch.arange(Ny, dtype=torch.float64)
    slope = (V_avg[-1] - V_avg[0]) / (Ny - 1)
    V_fit = V_avg[0] + slope * y
    linearity_err = (V_avg - V_fit).abs().max().item()
    assert linearity_err < 5e-3, f"Mixed BC not linear: {linearity_err}"

    # Check monotonicity
    diffs = V_avg[1:] - V_avg[:-1]
    assert (diffs > 0).all(), "Profile not monotonically increasing"

    print(f"4-V4 PASS: Mixed BC (x-spread={col_spread:.2e}, "
          f"linearity={linearity_err:.2e})")


if __name__ == "__main__":
    test_4v1_neumann_conservation()
    test_4v2_dirichlet_steady_state()
    test_4v3_absorbing()
    test_4v4_mixed_bc()
    print("\nPhase 4: ALL 4 TESTS PASS")
