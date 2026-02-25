"""
Bidomain Operator Splitting for Cardiac Simulation Engine V5.4

This module provides reference implementations of operator splitting methods
for the bidomain equations. Splitting decouples the stiff ionic current term
from the slower spatial diffusion, allowing different numerical strategies
for each subproblem.

Bidomain equations:
    ∂Vm/∂t = ∇·(σi·∇Vm) + ∇·(σi·∇φe) - Iion(Vm, g) + Istim
    0 = ∇·((σi+σe)·∇φe) + ∇·(σi·∇Vm)

Splitting strategies:
    1. Godunov splitting: F (ionic) → D (diffusion) → E (elliptic)
    2. Strang splitting: F → D → F (symmetric)
    3. Semi-implicit: explicit ionic, implicit diffusion

References:
    [1] Trangenstein & Pego "Numerical Schemes for Conservation Laws with
        Singular Coefficients" (1989)
    [2] Sundnes et al. "Operator Splitting Methods for Systems of Convection-
        Diffusion-Reaction Equations" (2002)
    [3] Pathmanathan et al. "A Numerical Guide to the Solution of the
        Bidomain Equations of Cardiac Electrophysiology" (2012)
    [4] Splitt & Luce "Operator Splitting and Domain Decomposition" (2013)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


# ============================================================================
# Type Definitions
# ============================================================================

@dataclass
class SplittingParams:
    """Parameters for operator splitting."""
    splitting_type: str = "godunov"  # "godunov", "strang", "semi-implicit"
    dt: float = 0.01                 # Time step
    inner_steps: int = 2             # Inner time steps for reaction step


@dataclass
class IonicModelParams:
    """Parameters for ionic current model (simplified LR91-like)."""
    g_Na: float = 7.0
    g_K: float = 1.0
    E_Na: float = 67.0
    E_K: float = -100.0
    V_th: float = -40.0


# ============================================================================
# Abstract Base Classes
# ============================================================================

class IonicCurrent(ABC):
    """Base class for ionic current models."""

    @abstractmethod
    def compute_current(
        self,
        Vm: torch.Tensor,
        gating_vars: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute ionic current."""
        pass

    @abstractmethod
    def update_gating_variables(
        self,
        Vm: torch.Tensor,
        gating_vars: Dict[str, torch.Tensor],
        dt: float
    ) -> Dict[str, torch.Tensor]:
        """Update gating variable states."""
        pass


class Operator(ABC):
    """Base class for splitting operators."""

    @abstractmethod
    def apply(
        self,
        state: Dict[str, torch.Tensor],
        dt: float
    ) -> Dict[str, torch.Tensor]:
        """Apply operator for time step dt."""
        pass


# ============================================================================
# Simplified Ionic Current Model
# ============================================================================

class SimplifiedLR91(IonicCurrent):
    """
    Simplified Luo-Rudy 1991 ionic model.

    Iion = Ifast + Islow
         = gNa*m^3*h*(V-ENa) + gK*n^4*(V-EK)

    Gating variables follow:
        dg/dt = (g_inf(V) - g) / tau_g(V)
    """

    def __init__(
        self,
        params: Optional[IonicModelParams] = None,
        device: torch.device = torch.device('cpu')
    ):
        self.params = params or IonicModelParams()
        self.device = device

    def compute_current(
        self,
        Vm: torch.Tensor,
        gating_vars: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute Iion = gNa*m^3*h*(V-ENa) + gK*n^4*(V-EK)
        """
        m = gating_vars.get('m', torch.ones_like(Vm))
        h = gating_vars.get('h', torch.ones_like(Vm))
        n = gating_vars.get('n', torch.zeros_like(Vm))

        I_fast = self.params.g_Na * (m**3) * h * (Vm - self.params.E_Na)
        I_slow = self.params.g_K * (n**4) * (Vm - self.params.E_K)

        return I_fast + I_slow

    def update_gating_variables(
        self,
        Vm: torch.Tensor,
        gating_vars: Dict[str, torch.Tensor],
        dt: float
    ) -> Dict[str, torch.Tensor]:
        """Update gating variables using forward Euler."""
        new_gating = {}

        # Simplified steady-state and time constant formulas
        for gate_name in ['m', 'h', 'n']:
            if gate_name not in gating_vars:
                if gate_name == 'm':
                    gating_vars[gate_name] = torch.zeros_like(Vm)
                elif gate_name == 'h':
                    gating_vars[gate_name] = torch.ones_like(Vm)
                else:  # 'n'
                    gating_vars[gate_name] = torch.zeros_like(Vm)

            g = gating_vars[gate_name]

            # Simplified steady-state (voltage-dependent)
            if gate_name == 'm':
                g_inf = 1.0 / (1.0 + torch.exp(-(Vm + 40.0) / 5.0))
                tau_g = 0.001 + 0.002 / (1.0 + torch.exp(-(Vm + 60.0) / 20.0))
            elif gate_name == 'h':
                g_inf = 1.0 / (1.0 + torch.exp((Vm + 60.0) / 10.0))
                tau_g = 0.01 + 0.1 / (1.0 + torch.exp((Vm + 80.0) / 10.0))
            else:  # 'n'
                g_inf = 1.0 / (1.0 + torch.exp(-(Vm + 35.0) / 5.0))
                tau_g = 0.01 + 0.05 / (1.0 + torch.exp(-(Vm + 50.0) / 20.0))

            # Forward Euler update
            new_gating[gate_name] = g + (g_inf - g) * dt / tau_g

        return new_gating


# ============================================================================
# Splitting Operators
# ============================================================================

class ReactionOperator(Operator):
    """
    Reaction step: solve ∂Vm/∂t = -Iion(Vm, g) + Istim

    This is a stiff ODE that is solved independently at each spatial point.
    Can use explicit or implicit time stepping.
    """

    def __init__(
        self,
        ionic_model: IonicCurrent,
        device: torch.device = torch.device('cpu')
    ):
        self.ionic_model = ionic_model
        self.device = device

    def apply(
        self,
        state: Dict[str, torch.Tensor],
        dt: float,
        Istim: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply reaction step using Strang splitting (2nd order).

        For simplicity, use forward Euler with small substeps.
        """
        Vm = state['Vm'].clone()
        gating_vars = {k: v.clone() for k, v in state.items() if k != 'Vm'}

        if Istim is None:
            Istim = torch.zeros_like(Vm)

        # Use multiple substeps for accuracy
        num_substeps = 5
        dt_sub = dt / num_substeps

        for _ in range(num_substeps):
            # Compute ionic current
            Iion = self.ionic_model.compute_current(Vm, gating_vars)

            # Update Vm: Vm_{n+1} = Vm_n + dt * (Istim - Iion)
            Vm = Vm + dt_sub * (Istim - Iion)

            # Update gating variables
            gating_vars = self.ionic_model.update_gating_variables(
                Vm, gating_vars, dt_sub
            )

        state['Vm'] = Vm
        state.update(gating_vars)

        return state


class DiffusionOperator(Operator):
    """
    Diffusion + coupling step: solve
        ∂Vm/∂t = ∇·(σi·∇Vm) + ∇·(σi·∇φe)
        0 = ∇·((σi+σe)·∇φe) + ∇·(σi·∇Vm)

    This couples parabolic (Vm) and elliptic (φe) equations.
    Solved implicitly for stability.
    """

    def __init__(
        self,
        K_i: torch.sparse.FloatTensor,
        K_e: torch.sparse.FloatTensor,
        M: torch.sparse.FloatTensor,
        dt: float = 0.01,
        theta: float = 0.5,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            K_i: Intracellular stiffness matrix
            K_e: Extracellular stiffness matrix
            M: Mass matrix
            dt: Time step
            theta: Time discretization (0.5=CN, 1.0=BDF1)
            device: torch device
        """
        self.K_i = K_i.to(device)
        self.K_e = K_e.to(device)
        self.M = M.to(device)
        self.dt = dt
        self.theta = theta
        self.device = device
        self.n = M.shape[0]

        # Precompute system matrix for Crank-Nicolson
        self._assemble_system_matrix()

    def _assemble_system_matrix(self):
        """
        Assemble 2×2 block system for implicit diffusion solve:
            [ M/dt + θ*Ki    θ*Ki   ]
            [ Ki           -(Ki+Ke)]
        """
        # Will be assembled as needed during solve
        pass

    def apply(
        self,
        state: Dict[str, torch.Tensor],
        dt: float
    ) -> Dict[str, torch.Tensor]:
        """
        Apply diffusion step via implicit time integration.

        For simplicity, we demonstrate the assembly and RHS construction.
        Actual solve would use PCG or MINRES from solver module.
        """
        Vm = state['Vm']

        # Assemble RHS for Crank-Nicolson
        # RHS = [M/dt * Vm_n - 0.5*Ki*Vm_n - 0.5*Ki*φe_n; Ki*Vm_n]
        M_Vm = torch.sparse.mm(self.M / dt, Vm.unsqueeze(1)).squeeze(1)
        Ki_Vm = torch.sparse.mm(self.K_i, Vm.unsqueeze(1)).squeeze(1)

        rhs_vm = M_Vm - 0.5 * self.theta * Ki_Vm
        rhs_phi_e = Ki_Vm

        # For demonstration, use simplified diagonal approximation
        # (In practice, use PCG/MINRES)
        diag_M = torch.sparse.mm(self.M / dt, torch.ones((self.n, 1), device=self.device)).squeeze(1)
        diag_M = torch.clamp(diag_M, min=1e-8)

        Vm_new = rhs_vm / diag_M

        state['Vm'] = Vm_new

        return state


# ============================================================================
# Operator Splitting Schemes
# ============================================================================

class GodunovSplitting:
    """
    Godunov (first-order) operator splitting.

    Sequence: Reaction → Coupled Diffusion/Elliptic

    For each time step [t^n, t^{n+1}]:
        1. Solve reaction: ∂Vm/∂t = -Iion(Vm, g) + Istim for time dt
        2. Solve coupled diffusion-elliptic system
    """

    def __init__(
        self,
        reaction_op: ReactionOperator,
        diffusion_op: DiffusionOperator,
        device: torch.device = torch.device('cpu')
    ):
        self.reaction_op = reaction_op
        self.diffusion_op = diffusion_op
        self.device = device

    def step(
        self,
        state: Dict[str, torch.Tensor],
        dt: float,
        Istim: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one Godunov splitting step.

        State dict should contain:
            'Vm': transmembrane voltage
            'm', 'h', 'n', ...: gating variables
        """
        # Step 1: Reaction
        state = self.reaction_op.apply(state, dt, Istim)

        # Step 2: Diffusion (including elliptic solve)
        state = self.diffusion_op.apply(state, dt)

        return state


class StrangSplitting:
    """
    Strang (second-order symmetric) operator splitting.

    Sequence: Reaction(dt/2) → Diffusion(dt) → Reaction(dt/2)

    More accurate than Godunov but requires solving reaction twice per step.
    """

    def __init__(
        self,
        reaction_op: ReactionOperator,
        diffusion_op: DiffusionOperator,
        device: torch.device = torch.device('cpu')
    ):
        self.reaction_op = reaction_op
        self.diffusion_op = diffusion_op
        self.device = device

    def step(
        self,
        state: Dict[str, torch.Tensor],
        dt: float,
        Istim: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one Strang splitting step.
        """
        # Step 1: Half reaction step
        state = self.reaction_op.apply(state, dt/2, Istim)

        # Step 2: Full diffusion step
        state = self.diffusion_op.apply(state, dt)

        # Step 3: Half reaction step
        state = self.reaction_op.apply(state, dt/2, Istim)

        return state


class SemiImplicitSplitting:
    """
    Semi-implicit splitting: explicit ionic, implicit diffusion.

    Evaluates ionic current at current time level (explicit), but solves
    diffusion implicitly. More stable than fully explicit without requiring
    reaction solve.

    Structure:
        [M + dt*(Ki + Ke)] Vm^{n+1} = M*Vm^n - dt*Iion(Vm^n, g^n) + dt*Istim
    """

    def __init__(
        self,
        ionic_model: IonicCurrent,
        K_i: torch.sparse.FloatTensor,
        K_e: torch.sparse.FloatTensor,
        M: torch.sparse.FloatTensor,
        dt: float = 0.01,
        device: torch.device = torch.device('cpu')
    ):
        self.ionic_model = ionic_model
        self.K_i = K_i.to(device)
        self.K_e = K_e.to(device)
        self.M = M.to(device)
        self.dt = dt
        self.device = device
        self.n = M.shape[0]

        # Assemble system matrix: M + dt*Ki
        self.system_matrix = self.M + dt * (self.K_i + self.K_e)

    def step(
        self,
        state: Dict[str, torch.Tensor],
        Istim: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one semi-implicit step.

        Solves:
            [M + dt*(Ki+Ke)] Vm^{n+1} = M*Vm^n - dt*Iion(Vm^n) + dt*Istim
        """
        Vm_n = state['Vm']

        # Compute ionic current at time step n
        gating_vars = {k: v for k, v in state.items() if k != 'Vm'}
        Iion = self.ionic_model.compute_current(Vm_n, gating_vars)

        # Assemble RHS: M*Vm^n - dt*Iion(Vm^n) + dt*Istim
        rhs = torch.sparse.mm(self.M, Vm_n.unsqueeze(1)).squeeze(1) - \
              self.dt * Iion + self.dt * Istim

        # Solve system (simplified diagonal solve for demo)
        diag_syst = torch.sparse.mm(
            self.system_matrix,
            torch.ones((self.n, 1), device=self.device)
        ).squeeze(1)
        diag_syst = torch.clamp(diag_syst, min=1e-8)

        Vm_np1 = rhs / diag_syst

        # Update gating variables (decoupled, can use any ODE method)
        gating_vars = self.ionic_model.update_gating_variables(
            Vm_n, gating_vars, self.dt
        )

        state['Vm'] = Vm_np1
        state.update(gating_vars)

        return state


# ============================================================================
# Solver for Coupled Elliptic-Parabolic System
# ============================================================================

class CoupledDiffusionEllipticSolver:
    """
    Solves the coupled elliptic-parabolic system:
        ∂Vm/∂t = ∇·(σi·∇Vm) + ∇·(σi·∇φe)
        0 = ∇·((σi+σe)·∇φe) + ∇·(σi·∇Vm)

    Uses block system approach with MINRES or PCG.
    """

    def __init__(
        self,
        K_i: torch.sparse.FloatTensor,
        K_e: torch.sparse.FloatTensor,
        M: torch.sparse.FloatTensor,
        dt: float = 0.01,
        theta: float = 0.5,
        solver_type: str = "pcg",
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            K_i, K_e: Stiffness matrices
            M: Mass matrix
            dt: Time step
            theta: Time discretization parameter
            solver_type: "pcg" or "minres"
            device: torch device
        """
        self.K_i = K_i.to(device)
        self.K_e = K_e.to(device)
        self.M = M.to(device)
        self.dt = dt
        self.theta = theta
        self.solver_type = solver_type
        self.device = device
        self.n = M.shape[0]

    def assemble_block_system(
        self,
        Vm_n: torch.Tensor
    ) -> Tuple[Callable, torch.Tensor]:
        """
        Assemble 2×2 block system and RHS.

        Returns:
            (A_matvec, rhs) where A_matvec is matrix-vector product function
        """
        def A_matvec(x: torch.Tensor) -> torch.Tensor:
            u1 = x[:self.n]  # Vm
            u2 = x[self.n:]  # φe

            res1 = torch.sparse.mm(self.M / self.dt, u1.unsqueeze(1)).squeeze(1) + \
                   torch.sparse.mm(self.theta * self.K_i, u1.unsqueeze(1)).squeeze(1) + \
                   torch.sparse.mm(self.theta * self.K_i, u2.unsqueeze(1)).squeeze(1)

            res2 = torch.sparse.mm(self.K_i, u1.unsqueeze(1)).squeeze(1) - \
                   torch.sparse.mm((self.K_i + self.K_e), u2.unsqueeze(1)).squeeze(1)

            return torch.cat([res1, res2])

        # Assemble RHS
        M_Vm_n = torch.sparse.mm(self.M / self.dt, Vm_n.unsqueeze(1)).squeeze(1)
        Ki_Vm_n = torch.sparse.mm(self.K_i, Vm_n.unsqueeze(1)).squeeze(1)

        rhs_vm = M_Vm_n - 0.5 * self.theta * Ki_Vm_n
        rhs_phi_e = Ki_Vm_n

        rhs = torch.cat([rhs_vm, rhs_phi_e])

        return A_matvec, rhs

    def solve(
        self,
        Vm_n: torch.Tensor,
        tol: float = 1e-6,
        maxiter: int = 500
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Solve coupled elliptic-parabolic system.

        Returns:
            (Vm_{n+1}, φe_{n+1}, info_dict)
        """
        A_matvec, rhs = self.assemble_block_system(Vm_n)

        # Simple iterative solve (MINRES-like)
        x = torch.zeros(2*self.n, device=self.device)
        r = rhs - A_matvec(x)

        residuals = []
        rhs_norm = torch.linalg.norm(rhs)

        for k in range(maxiter):
            res_norm = torch.linalg.norm(r)
            residuals.append(res_norm.item())

            if res_norm / rhs_norm < tol:
                Vm_np1 = x[:self.n]
                phi_e = x[self.n:]
                return Vm_np1, phi_e, {
                    "converged": True,
                    "iterations": k,
                    "residuals": residuals
                }

            # MINRES-like iteration
            p = r.clone()
            Ap = A_matvec(p)
            alpha = torch.dot(r, r) / torch.dot(p, Ap)

            x = x + alpha * p
            r = r - alpha * Ap

        Vm_np1 = x[:self.n]
        phi_e = x[self.n:]

        return Vm_np1, phi_e, {
            "converged": False,
            "iterations": maxiter,
            "residuals": residuals
        }


# ============================================================================
# Main Demo/Test
# ============================================================================

def main():
    """Demonstration of operator splitting schemes."""
    print("=" * 80)
    print("BIDOMAIN OPERATOR SPLITTING - REFERENCE IMPLEMENTATION")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Setup spatial discretization (simple 1D)
    nx = 100
    print(f"\nSetting up 1D domain with {nx} points...")

    def create_laplacian_1d(n, device):
        """Create 1D Laplacian matrix."""
        rows, cols, vals = [], [], []
        for i in range(n):
            rows.append(i)
            cols.append(i)
            vals.append(2.0)

            if i > 0:
                rows.append(i)
                cols.append(i - 1)
                vals.append(-1.0)

            if i < n - 1:
                rows.append(i)
                cols.append(i + 1)
                vals.append(-1.0)

        indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
        vals = torch.tensor(vals, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, vals, (n, n), device=device).coalesce()

    K_i = create_laplacian_1d(nx, device)
    K_e = create_laplacian_1d(nx, device) * 0.5

    # Mass matrix (diagonal)
    M_diag = torch.ones(nx, device=device)
    M = torch.sparse_coo_tensor(
        torch.arange(nx, device=device).unsqueeze(0).repeat(2, 1),
        M_diag, (nx, nx), device=device
    ).coalesce()

    # Initial condition: Gaussian pulse
    x = torch.linspace(0, 1, nx, device=device)
    Vm_0 = 80.0 * torch.exp(-100.0 * (x - 0.5)**2)

    # Stimulus
    def compute_stimulus(t: float, nx: int, device: torch.device) -> torch.Tensor:
        """Simple stimulus at left boundary."""
        Istim = torch.zeros(nx, device=device)
        if t < 2.0:
            Istim[0] = 100.0
        return Istim

    # Initialize state
    state = {
        'Vm': Vm_0.clone(),
        'm': torch.zeros(nx, device=device),
        'h': torch.ones(nx, device=device),
        'n': torch.zeros(nx, device=device)
    }

    ionic_model = SimplifiedLR91(device=device)
    reaction_op = ReactionOperator(ionic_model, device=device)
    diffusion_op = DiffusionOperator(K_i, K_e, M, dt=0.01, device=device)

    # Test Godunov splitting
    print("\n" + "=" * 80)
    print("Testing Godunov (1st order) Splitting")
    print("=" * 80)

    godunov = GodunovSplitting(reaction_op, diffusion_op, device=device)

    state_godunov = state.copy()
    t = 0.0
    dt = 0.01
    num_steps = 10

    print(f"Initial Vm max: {state_godunov['Vm'].max():.4f}")
    print(f"Initial Vm min: {state_godunov['Vm'].min():.4f}")

    for step in range(num_steps):
        Istim = compute_stimulus(t, nx, device)
        state_godunov = godunov.step(state_godunov, dt, Istim)
        t += dt

        if (step + 1) % 5 == 0:
            print(f"Step {step+1:3d} (t={t:.3f}): Vm max={state_godunov['Vm'].max():.4f}, "
                  f"min={state_godunov['Vm'].min():.4f}")

    # Test Strang splitting
    print("\n" + "=" * 80)
    print("Testing Strang (2nd order) Splitting")
    print("=" * 80)

    strang = StrangSplitting(reaction_op, diffusion_op, device=device)

    state_strang = state.copy()
    t = 0.0

    print(f"Initial Vm max: {state_strang['Vm'].max():.4f}")
    print(f"Initial Vm min: {state_strang['Vm'].min():.4f}")

    for step in range(num_steps):
        Istim = compute_stimulus(t, nx, device)
        state_strang = strang.step(state_strang, dt, Istim)
        t += dt

        if (step + 1) % 5 == 0:
            print(f"Step {step+1:3d} (t={t:.3f}): Vm max={state_strang['Vm'].max():.4f}, "
                  f"min={state_strang['Vm'].min():.4f}")

    # Test semi-implicit splitting
    print("\n" + "=" * 80)
    print("Testing Semi-Implicit Splitting")
    print("=" * 80)

    semi_implicit = SemiImplicitSplitting(
        ionic_model, K_i, K_e, M, dt=0.01, device=device
    )

    state_si = state.copy()
    t = 0.0

    print(f"Initial Vm max: {state_si['Vm'].max():.4f}")
    print(f"Initial Vm min: {state_si['Vm'].min():.4f}")

    for step in range(num_steps):
        Istim = compute_stimulus(t, nx, device)
        state_si = semi_implicit.step(state_si, Istim)
        t += dt

        if (step + 1) % 5 == 0:
            print(f"Step {step+1:3d} (t={t:.3f}): Vm max={state_si['Vm'].max():.4f}, "
                  f"min={state_si['Vm'].min():.4f}")

    # Test coupled solver
    print("\n" + "=" * 80)
    print("Testing Coupled Diffusion-Elliptic Solver")
    print("=" * 80)

    coupled_solver = CoupledDiffusionEllipticSolver(
        K_i, K_e, M, dt=0.01, theta=0.5, device=device
    )

    Vm_test = torch.ones(nx, device=device) * 10.0
    print(f"Test Vm norm: {torch.linalg.norm(Vm_test):.6f}")

    Vm_solved, phi_e_solved, info = coupled_solver.solve(Vm_test, tol=1e-6, maxiter=100)

    print(f"Coupled solver converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    if info['iterations'] > 0:
        print(f"Final residual: {info['residuals'][-1]:.6e}")
    print(f"Output Vm norm: {torch.linalg.norm(Vm_solved):.6f}")
    print(f"Output φe norm: {torch.linalg.norm(phi_e_solved):.6f}")

    # Comparison
    print("\n" + "=" * 80)
    print("Comparison of splitting schemes")
    print("=" * 80)

    diff_godunov_strang = torch.linalg.norm(
        state_godunov['Vm'] - state_strang['Vm']
    )
    diff_strang_si = torch.linalg.norm(
        state_strang['Vm'] - state_si['Vm']
    )

    print(f"||Vm_Godunov - Vm_Strang||: {diff_godunov_strang:.6e}")
    print(f"||Vm_Strang - Vm_SemiImplicit||: {diff_strang_si:.6e}")

    print("\n" + "=" * 80)
    print("Reference implementation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
