"""
Bidomain Model Solver
=====================

2D Bidomain equation solver for cardiac electrophysiology.

The bidomain model describes the electrical activity of cardiac tissue using
two coupled PDEs:

1. Parabolic (transmembrane voltage):
   χCm ∂Vm/∂t = ∇·(σi∇Vm) + ∇·(σi∇φe) - χIion

2. Elliptic (extracellular potential):
   ∇·((σi + σe)∇φe) + ∇·(σi∇Vm) = 0

where:
   Vm = φi - φe (transmembrane voltage)
   φe = extracellular potential
   σi, σe = intracellular, extracellular conductivity tensors
   χ = surface-to-volume ratio
   Cm = membrane capacitance
   Iion = ionic current

Units: cm, ms, mV, uA/cm², mS/cm, uF/cm²

Reference:
    Henriquez CS. "Simulating the electrical behavior of cardiac tissue using
    the bidomain model." Crit Rev Biomed Eng. 1993;21(1):1-77.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from scipy.sparse import diags, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, cg
import warnings


# =============================================================================
# Default Parameters
# =============================================================================

DEFAULT_PARAMS = {
    # Conductivities [mS/cm]
    'sigma_il': 3.0,      # Intracellular longitudinal
    'sigma_it': 0.31,     # Intracellular transverse
    'sigma_el': 2.0,      # Extracellular longitudinal
    'sigma_et': 1.65,     # Extracellular transverse

    # Tissue properties
    'chi': 2000.0,        # Surface-to-volume ratio [1/cm]
    'C_m': 1.0,           # Membrane capacitance [uF/cm²]

    # Domain (2D slab)
    'Lx': 2.0,            # Length in x [cm]
    'Ly': 2.0,            # Length in y [cm]
    'dx': 0.01,           # Grid spacing x [cm] (100 um)
    'dy': 0.01,           # Grid spacing y [cm] (100 um)

    # Fiber orientation (angle from x-axis)
    'fiber_angle': 0.0,   # degrees, 0 = fibers along x
}


class BidomainSolver:
    """
    2D Bidomain equation solver.

    Uses operator splitting:
    1. Solve ionic model (ODE step) to get Iion
    2. Solve parabolic PDE for Vm update
    3. Solve elliptic PDE for φe
    """

    def __init__(self, params: Dict = None, ionic_model = None):
        """
        Initialize bidomain solver.

        Parameters
        ----------
        params : dict
            Simulation parameters (uses defaults if not specified)
        ionic_model : object
            Ionic model instance with step() method
        """
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.ionic_model = ionic_model

        # Grid setup
        self.Lx = self.params['Lx']
        self.Ly = self.params['Ly']
        self.dx = self.params['dx']
        self.dy = self.params['dy']

        self.Nx = int(self.Lx / self.dx) + 1
        self.Ny = int(self.Ly / self.dy) + 1
        self.N = self.Nx * self.Ny  # Total nodes

        # Conductivities
        self.sigma_il = self.params['sigma_il']
        self.sigma_it = self.params['sigma_it']
        self.sigma_el = self.params['sigma_el']
        self.sigma_et = self.params['sigma_et']

        # Tissue properties
        self.chi = self.params['chi']
        self.C_m = self.params['C_m']

        # Fiber angle (convert to radians)
        theta = np.radians(self.params['fiber_angle'])
        self.cos_theta = np.cos(theta)
        self.sin_theta = np.sin(theta)

        # Build effective conductivities (rotated tensor components)
        self._compute_conductivities()

        # Build matrices
        self._build_matrices()

        # Storage for state
        self.Vm = None      # Transmembrane voltage [mV]
        self.phi_e = None   # Extracellular potential [mV]
        self.ionic_states = None  # Ionic model states

    def _compute_conductivities(self):
        """
        Compute effective conductivity tensor components.

        For rotated fibers at angle θ from x-axis:
        σ_xx = σ_l cos²θ + σ_t sin²θ
        σ_yy = σ_l sin²θ + σ_t cos²θ
        σ_xy = (σ_l - σ_t) sinθ cosθ
        """
        c2 = self.cos_theta**2
        s2 = self.sin_theta**2
        cs = self.cos_theta * self.sin_theta

        # Intracellular
        self.sigma_i_xx = self.sigma_il * c2 + self.sigma_it * s2
        self.sigma_i_yy = self.sigma_il * s2 + self.sigma_it * c2
        self.sigma_i_xy = (self.sigma_il - self.sigma_it) * cs

        # Extracellular
        self.sigma_e_xx = self.sigma_el * c2 + self.sigma_et * s2
        self.sigma_e_yy = self.sigma_el * s2 + self.sigma_et * c2
        self.sigma_e_xy = (self.sigma_el - self.sigma_et) * cs

        # Combined for elliptic equation
        self.sigma_sum_xx = self.sigma_i_xx + self.sigma_e_xx
        self.sigma_sum_yy = self.sigma_i_yy + self.sigma_e_yy
        self.sigma_sum_xy = self.sigma_i_xy + self.sigma_e_xy

    def _idx(self, i: int, j: int) -> int:
        """Convert 2D indices to 1D index."""
        return j * self.Nx + i

    def _build_matrices(self):
        """
        Build sparse matrices for the bidomain system.

        For the parabolic equation (Vm update):
        χCm ∂Vm/∂t = ∇·(σi∇Vm) + ∇·(σi∇φe) - χIion

        Using backward Euler: (I - dt*Ai)Vm^{n+1} = Vm^n + dt*(Ai*φe - χIion)/χCm

        For the elliptic equation (φe solve):
        ∇·((σi+σe)∇φe) = -∇·(σi∇Vm)
        """
        N = self.N
        dx, dy = self.dx, self.dy
        dx2, dy2 = dx**2, dy**2

        # Build Laplacian-like matrices using finite differences
        # 5-point stencil for diagonal conductivity (simplified - ignoring cross terms for now)

        # Matrix for ∇·(σi∇•) operator
        self.A_i = lil_matrix((N, N))

        # Matrix for ∇·((σi+σe)∇•) operator
        self.A_sum = lil_matrix((N, N))

        for j in range(self.Ny):
            for i in range(self.Nx):
                idx = self._idx(i, j)

                # Interior points use 5-point stencil
                # ∂/∂x(σ_xx ∂/∂x) + ∂/∂y(σ_yy ∂/∂y)

                # Coefficients for σi
                cx_i = self.sigma_i_xx / dx2
                cy_i = self.sigma_i_yy / dy2

                # Coefficients for σi + σe
                cx_sum = self.sigma_sum_xx / dx2
                cy_sum = self.sigma_sum_yy / dy2

                # Handle boundaries with no-flux (Neumann) BC
                # This is implemented by reflecting ghost nodes

                # West neighbor (i-1)
                if i > 0:
                    self.A_i[idx, self._idx(i-1, j)] = cx_i
                    self.A_sum[idx, self._idx(i-1, j)] = cx_sum
                    self.A_i[idx, idx] -= cx_i
                    self.A_sum[idx, idx] -= cx_sum

                # East neighbor (i+1)
                if i < self.Nx - 1:
                    self.A_i[idx, self._idx(i+1, j)] = cx_i
                    self.A_sum[idx, self._idx(i+1, j)] = cx_sum
                    self.A_i[idx, idx] -= cx_i
                    self.A_sum[idx, idx] -= cx_sum

                # South neighbor (j-1)
                if j > 0:
                    self.A_i[idx, self._idx(i, j-1)] = cy_i
                    self.A_sum[idx, self._idx(i, j-1)] = cy_sum
                    self.A_i[idx, idx] -= cy_i
                    self.A_sum[idx, idx] -= cy_sum

                # North neighbor (j+1)
                if j < self.Ny - 1:
                    self.A_i[idx, self._idx(i, j+1)] = cy_i
                    self.A_sum[idx, self._idx(i, j+1)] = cy_sum
                    self.A_i[idx, idx] -= cy_i
                    self.A_sum[idx, idx] -= cy_sum

        # Convert to CSR for efficient arithmetic
        self.A_i = csr_matrix(self.A_i)
        self.A_sum = csr_matrix(self.A_sum)

        # Fix singular elliptic system (φe is determined up to a constant)
        # Pin one node to zero
        self.A_sum_pinned = self.A_sum.tolil()
        self.A_sum_pinned[0, :] = 0
        self.A_sum_pinned[0, 0] = 1.0
        self.A_sum_pinned = csr_matrix(self.A_sum_pinned)

    def initialize(self, V_rest: float = -84.0) -> None:
        """
        Initialize state variables.

        Parameters
        ----------
        V_rest : float
            Resting membrane potential [mV]
        """
        # Transmembrane voltage
        self.Vm = np.ones((self.Ny, self.Nx)) * V_rest

        # Extracellular potential
        self.phi_e = np.zeros((self.Ny, self.Nx))

        # Initialize ionic model states at each node
        if self.ionic_model is not None:
            self.ionic_states = [[self.ionic_model.initialize_state()
                                  for _ in range(self.Nx)]
                                 for _ in range(self.Ny)]

        # Storage for ionic current
        self.I_ion = np.zeros((self.Ny, self.Nx))

    def apply_stimulus(self, region: Tuple[slice, slice], amplitude: float) -> None:
        """
        Apply stimulus current to a region.

        Parameters
        ----------
        region : tuple of slices
            (y_slice, x_slice) defining the stimulus region
        amplitude : float
            Stimulus current amplitude [uA/cm²]
        """
        self.I_stim = np.zeros((self.Ny, self.Nx))
        self.I_stim[region] = amplitude

    def _solve_ionic_step(self, dt: float) -> None:
        """
        Solve ionic model ODEs at each node.

        This updates I_ion and the ionic states.
        """
        if self.ionic_model is None:
            return

        for j in range(self.Ny):
            for i in range(self.Nx):
                state = self.ionic_states[j][i]
                state['V'] = self.Vm[j, i]

                # Get stimulus if present
                I_stim = self.I_stim[j, i] if hasattr(self, 'I_stim') else 0.0

                # Step the ionic model (it will compute currents internally)
                currents = self.ionic_model.compute_currents(state)
                self.I_ion[j, i] = currents['I_ion']

                # Update ionic state (gating variables, concentrations)
                # But NOT voltage - that's handled by the PDE solver
                self.ionic_states[j][i] = self.ionic_model.step(state, I_stim)

    def _solve_parabolic_step(self, dt: float) -> None:
        """
        Solve parabolic equation for Vm update using forward Euler.

        χCm ∂Vm/∂t = ∇·(σi∇Vm) + ∇·(σi∇φe) - χIion + Istim
        """
        chi = self.chi
        C_m = self.C_m

        # Flatten arrays
        Vm_flat = self.Vm.flatten()
        phi_e_flat = self.phi_e.flatten()
        I_ion_flat = self.I_ion.flatten()
        I_stim_flat = self.I_stim.flatten() if hasattr(self, 'I_stim') else np.zeros(self.N)

        # RHS = ∇·(σi∇Vm) + ∇·(σi∇φe) - χIion + Istim
        diff_Vm = self.A_i @ Vm_flat
        diff_phi_e = self.A_i @ phi_e_flat

        dVm_dt = (diff_Vm + diff_phi_e - chi * I_ion_flat + I_stim_flat) / (chi * C_m)

        # Forward Euler update
        Vm_new = Vm_flat + dt * dVm_dt

        # Reshape
        self.Vm = Vm_new.reshape((self.Ny, self.Nx))

    def _solve_elliptic_step(self) -> None:
        """
        Solve elliptic equation for φe.

        ∇·((σi+σe)∇φe) = -∇·(σi∇Vm)
        """
        Vm_flat = self.Vm.flatten()

        # RHS = -∇·(σi∇Vm)
        rhs = -self.A_i @ Vm_flat

        # Pin first node to zero to remove singularity
        rhs[0] = 0.0

        # Solve
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            phi_e_flat = spsolve(self.A_sum_pinned, rhs)

        # Reshape
        self.phi_e = phi_e_flat.reshape((self.Ny, self.Nx))

    def step(self, dt: float) -> None:
        """
        Advance the simulation by one time step using operator splitting.

        Parameters
        ----------
        dt : float
            Time step [ms]
        """
        # Step 1: Solve ionic model ODEs
        self._solve_ionic_step(dt)

        # Step 2: Solve elliptic equation for φe
        self._solve_elliptic_step()

        # Step 3: Solve parabolic equation for Vm update
        self._solve_parabolic_step(dt)

        # Update ionic model voltage states
        if self.ionic_model is not None:
            for j in range(self.Ny):
                for i in range(self.Nx):
                    self.ionic_states[j][i]['V'] = self.Vm[j, i]

    def run(self, t_end: float, dt: float,
            stim_region: Tuple[slice, slice] = None,
            stim_times: list = None,
            stim_duration: float = 1.0,
            stim_amplitude: float = -80.0,
            save_interval: int = 100,
            callback: Callable = None) -> Dict:
        """
        Run a simulation.

        Parameters
        ----------
        t_end : float
            End time [ms]
        dt : float
            Time step [ms]
        stim_region : tuple
            (y_slice, x_slice) for stimulus
        stim_times : list
            Times to apply stimulus [ms]
        stim_duration : float
            Stimulus duration [ms]
        stim_amplitude : float
            Stimulus amplitude [uA/cm²]
        save_interval : int
            Save state every N steps
        callback : callable
            Called each save_interval with (t, Vm, phi_e)

        Returns
        -------
        results : dict
            Time series of saved states
        """
        if stim_region is None:
            stim_region = (slice(0, 5), slice(0, 5))
        if stim_times is None:
            stim_times = [10.0]

        n_steps = int(t_end / dt)
        n_saves = n_steps // save_interval + 1

        # Storage
        results = {
            't': [],
            'Vm': [],
            'phi_e': [],
        }

        self.I_stim = np.zeros((self.Ny, self.Nx))

        for step in range(n_steps):
            t = step * dt

            # Check stimulus
            is_stimulating = False
            for t_stim in stim_times:
                if t_stim <= t < t_stim + stim_duration:
                    is_stimulating = True
                    break

            if is_stimulating:
                self.I_stim[stim_region] = stim_amplitude
            else:
                self.I_stim[:] = 0.0

            # Advance
            self.step(dt)

            # Save
            if step % save_interval == 0:
                results['t'].append(t)
                results['Vm'].append(self.Vm.copy())
                results['phi_e'].append(self.phi_e.copy())

                if callback:
                    callback(t, self.Vm, self.phi_e)

        # Final state
        results['t'].append(t_end)
        results['Vm'].append(self.Vm.copy())
        results['phi_e'].append(self.phi_e.copy())

        return results


# =============================================================================
# Monodomain Solver (simplified, faster)
# =============================================================================

class MonodomainSolver:
    """
    Simplified monodomain solver.

    The monodomain model assumes σi/σe = constant (equal anisotropy ratios):

    χCm ∂Vm/∂t = ∇·(σ∇Vm) - χIion

    where σ = σiσe/(σi+σe) is the effective conductivity.

    This is much faster than bidomain but doesn't capture φe.
    """

    def __init__(self, params: Dict = None, ionic_model = None):
        """Initialize monodomain solver."""
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.ionic_model = ionic_model

        # Grid setup
        self.Lx = self.params['Lx']
        self.Ly = self.params['Ly']
        self.dx = self.params['dx']
        self.dy = self.params['dy']

        self.Nx = int(self.Lx / self.dx) + 1
        self.Ny = int(self.Ly / self.dy) + 1
        self.N = self.Nx * self.Ny

        # Effective conductivity: σ = σi*σe/(σi+σe)
        sigma_i_l = self.params['sigma_il']
        sigma_i_t = self.params['sigma_it']
        sigma_e_l = self.params['sigma_el']
        sigma_e_t = self.params['sigma_et']

        self.sigma_l = sigma_i_l * sigma_e_l / (sigma_i_l + sigma_e_l)
        self.sigma_t = sigma_i_t * sigma_e_t / (sigma_i_t + sigma_e_t)

        # Tissue properties
        self.chi = self.params['chi']
        self.C_m = self.params['C_m']

        # Fiber angle
        theta = np.radians(self.params['fiber_angle'])
        c2 = np.cos(theta)**2
        s2 = np.sin(theta)**2

        self.sigma_xx = self.sigma_l * c2 + self.sigma_t * s2
        self.sigma_yy = self.sigma_l * s2 + self.sigma_t * c2

        # Build diffusion matrix
        self._build_matrix()

        # State
        self.Vm = None
        self.ionic_states = None

    def _idx(self, i: int, j: int) -> int:
        """Convert 2D indices to 1D index."""
        return j * self.Nx + i

    def _build_matrix(self):
        """Build sparse Laplacian matrix."""
        N = self.N
        dx2, dy2 = self.dx**2, self.dy**2

        cx = self.sigma_xx / dx2
        cy = self.sigma_yy / dy2

        self.A = lil_matrix((N, N))

        for j in range(self.Ny):
            for i in range(self.Nx):
                idx = self._idx(i, j)

                if i > 0:
                    self.A[idx, self._idx(i-1, j)] = cx
                    self.A[idx, idx] -= cx
                if i < self.Nx - 1:
                    self.A[idx, self._idx(i+1, j)] = cx
                    self.A[idx, idx] -= cx
                if j > 0:
                    self.A[idx, self._idx(i, j-1)] = cy
                    self.A[idx, idx] -= cy
                if j < self.Ny - 1:
                    self.A[idx, self._idx(i, j+1)] = cy
                    self.A[idx, idx] -= cy

        self.A = csr_matrix(self.A)

    def initialize(self, V_rest: float = -84.0) -> None:
        """Initialize state variables."""
        self.Vm = np.ones((self.Ny, self.Nx)) * V_rest

        if self.ionic_model is not None:
            self.ionic_states = [[self.ionic_model.initialize_state()
                                  for _ in range(self.Nx)]
                                 for _ in range(self.Ny)]

        self.I_ion = np.zeros((self.Ny, self.Nx))
        self.I_stim = np.zeros((self.Ny, self.Nx))

    def step(self, dt: float) -> None:
        """Advance by one time step using operator splitting.

        Step 1: Update ionic model (gating variables, concentrations) at each node
        Step 2: Solve diffusion PDE for Vm

        The monodomain equation is:
            Cm * dVm/dt = (1/chi) * div(sigma * grad(Vm)) - I_ion + I_stim/chi

        Or equivalently:
            dVm/dt = D * Laplacian(Vm) - I_ion/Cm + I_stim/(chi*Cm)

        where D = sigma/(chi*Cm) is the diffusion coefficient.
        """
        C_m = self.C_m
        chi = self.chi

        # Step 1: Update ionic model at each node
        # The ionic model computes I_ion and updates gating variables
        if self.ionic_model is not None:
            for j in range(self.Ny):
                for i in range(self.Nx):
                    state = self.ionic_states[j][i]
                    state['V'] = self.Vm[j, i]

                    # Compute ionic current for this node
                    currents = self.ionic_model.compute_currents(state)
                    self.I_ion[j, i] = currents['I_ion']

                    # Update gating variables and concentrations via ionic model step
                    # Pass stimulus current directly to ionic model
                    I_stim_local = self.I_stim[j, i]
                    self.ionic_states[j][i] = self.ionic_model.step(state, I_stim_local)

                    # Get the voltage from ionic model (includes stimulus effect)
                    self.Vm[j, i] = self.ionic_states[j][i]['V']

        # Step 2: Diffusion step
        # dVm/dt = D * Laplacian(Vm) where D = sigma/(chi*Cm)
        # The A matrix already has sigma/dx^2, so we just need to divide by chi*Cm
        Vm_flat = self.Vm.flatten()

        diff = self.A @ Vm_flat  # This gives sigma * Laplacian(Vm)
        dVm_dt_diff = diff / (chi * C_m)  # Convert to dV/dt

        Vm_new = Vm_flat + dt * dVm_dt_diff
        self.Vm = Vm_new.reshape((self.Ny, self.Nx))

        # Update ionic model voltage to match diffused value
        if self.ionic_model is not None:
            for j in range(self.Ny):
                for i in range(self.Nx):
                    self.ionic_states[j][i]['V'] = self.Vm[j, i]

    def run(self, t_end: float, dt: float,
            stim_region: Tuple[slice, slice] = None,
            stim_times: list = None,
            stim_duration: float = 1.0,
            stim_amplitude: float = -80.0,
            save_interval: int = 100,
            callback: Callable = None) -> Dict:
        """Run simulation."""
        if stim_region is None:
            stim_region = (slice(0, 5), slice(0, 5))
        if stim_times is None:
            stim_times = [10.0]

        n_steps = int(t_end / dt)

        results = {
            't': [],
            'Vm': [],
        }

        self.I_stim = np.zeros((self.Ny, self.Nx))

        for step in range(n_steps):
            t = step * dt

            # Check stimulus
            is_stimulating = False
            for t_stim in stim_times:
                if t_stim <= t < t_stim + stim_duration:
                    is_stimulating = True
                    break

            if is_stimulating:
                self.I_stim[stim_region] = stim_amplitude
            else:
                self.I_stim[:] = 0.0

            self.step(dt)

            if step % save_interval == 0:
                results['t'].append(t)
                results['Vm'].append(self.Vm.copy())

                if callback:
                    callback(t, self.Vm)

        results['t'].append(t_end)
        results['Vm'].append(self.Vm.copy())

        return results


if __name__ == '__main__':
    # Quick test without ionic model
    print("Testing bidomain solver structure...")

    params = {
        'Lx': 1.0,
        'Ly': 1.0,
        'dx': 0.05,
        'dy': 0.05,
    }

    solver = BidomainSolver(params)
    solver.initialize(V_rest=-84.0)

    print(f"Grid: {solver.Nx} x {solver.Ny} = {solver.N} nodes")
    print(f"Matrix A_i shape: {solver.A_i.shape}, nnz: {solver.A_i.nnz}")
    print(f"Matrix A_sum shape: {solver.A_sum.shape}, nnz: {solver.A_sum.nnz}")
    print("✓ Bidomain solver initialized successfully")
