"""
Mesh Builder for Monodomain Tissue Simulation

Provides a structured way to configure tissue geometry and diffusion parameters
with validation against numerical stability and propagation limits.

Includes:
- MeshBuilder: 2D tissue configuration (default 15x15 cm)
- CableMesh: 1D cable for CV tuning and validation

Tuned Diffusion Coefficients (calibrated via 1D cable simulations):
- D_L = 0.002161 cm²/ms for CV = 0.6 m/s (longitudinal)
- D_T = 0.000819 cm²/ms for CV = 0.3 m/s (transverse)

These values were empirically tuned for dx=0.02 cm, dt=0.02 ms using the
ORd ionic model. They account for numerical discretization effects.

Author: Created for Engine V5.1
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import torch
import numpy as np

from tissue.diffusion import (
    CV_EMPIRICAL_CONSTANT, compute_D_min, validate_D_for_mesh,
    CHI, CM, APD_REF, SAFETY_MARGIN_DEFAULT
)


# =============================================================================
# Tuned Diffusion Coefficients
# =============================================================================
# These values were calibrated via 1D cable simulations at dx=0.02 cm, dt=0.02 ms
# using the ORd ionic model. They provide accurate CV matching.

# For CV = 0.6 m/s (0.06 cm/ms) - longitudinal
D_TUNED_CV06 = 0.002161  # cm²/ms

# For CV = 0.3 m/s (0.03 cm/ms) - transverse
D_TUNED_CV03 = 0.000819  # cm²/ms

# Default tuned values
D_L_DEFAULT = D_TUNED_CV06  # Longitudinal: CV = 0.6 m/s
D_T_DEFAULT = D_TUNED_CV03  # Transverse: CV = 0.3 m/s


# =============================================================================
# 1D Cable Mesh for CV Tuning
# =============================================================================

@dataclass
class CableConfig:
    """
    Configuration for 1D cable simulation (CV tuning).

    Units:
    - Length: cm
    - Time: ms
    - Velocity: cm/ms (1 m/s = 0.1 cm/ms)
    - Diffusion: cm²/ms
    """
    length: float = 5.0         # Cable length (cm)
    dx: float = 0.02            # Grid spacing (cm) - 200 µm
    dt: float = 0.02            # Time step (ms)

    # Initial guess for diffusion coefficient
    D: float = 0.001            # Diffusion coefficient (cm²/ms)

    # Target CV for tuning
    target_cv: float = 0.06     # Target CV (cm/ms) = 0.6 m/s

    # Physical constants
    chi: float = CHI
    Cm: float = CM
    apd_ms: float = APD_REF

    # Computed values
    n_cells: int = field(init=False)

    def __post_init__(self):
        self.n_cells = int(round(self.length / self.dx))


class CableMesh:
    """
    1D cable mesh for conduction velocity tuning.

    Use this to empirically tune diffusion coefficients to achieve
    target CVs before applying to 2D tissue simulations.

    Example
    -------
    >>> cable = CableMesh.create_for_cv_tuning(target_cv=0.06)  # 0.6 m/s
    >>> measured_cv = cable.measure_cv()
    >>> D_tuned = cable.tune_D_to_cv()
    """

    def __init__(self, config: Optional[CableConfig] = None):
        """Initialize cable mesh with configuration."""
        self._config = config or CableConfig()
        self._states = None
        self._time = 0.0

    @classmethod
    def create_for_cv_tuning(
        cls,
        target_cv: float = 0.06,
        dx: float = 0.02,
        dt: float = 0.02,
        length: float = 5.0
    ) -> 'CableMesh':
        """
        Create cable mesh for CV tuning.

        Parameters
        ----------
        target_cv : float
            Target conduction velocity (cm/ms). Default 0.06 = 0.6 m/s.
        dx : float
            Grid spacing (cm). Default 0.02 = 200 µm.
        dt : float
            Time step (ms). Default 0.02.
        length : float
            Cable length (cm). Default 5.0.

        Returns
        -------
        CableMesh
            Configured cable mesh with initial D estimate.
        """
        # Initial D estimate from empirical formula
        D_initial = (target_cv / CV_EMPIRICAL_CONSTANT) ** 2

        config = CableConfig(
            length=length,
            dx=dx,
            dt=dt,
            D=D_initial,
            target_cv=target_cv
        )
        return cls(config)

    @property
    def config(self) -> CableConfig:
        """Get cable configuration."""
        return self._config

    def _compute_D_from_cv(self, cv: float) -> float:
        """Compute D from target CV using empirical relationship."""
        return (cv / CV_EMPIRICAL_CONSTANT) ** 2

    def _compute_cv_from_D(self, D: float) -> float:
        """Estimate CV from D using empirical relationship."""
        return CV_EMPIRICAL_CONSTANT * math.sqrt(D)

    def validate(self) -> Dict[str, Any]:
        """
        Validate cable configuration.

        Returns
        -------
        dict
            Validation results with status and messages.
        """
        cfg = self._config
        results = {
            'status': 'OK',
            'messages': []
        }

        # Stability check
        dt_max = 0.25 * cfg.dx ** 2 / cfg.D
        if cfg.dt > dt_max:
            results['status'] = 'CRITICAL'
            results['messages'].append(
                f"dt={cfg.dt} exceeds stability limit dt_max={dt_max:.5f} ms"
            )

        # D_min check
        D_min = compute_D_min(cfg.dx, cfg.apd_ms)
        if cfg.D < D_min:
            results['status'] = 'CRITICAL'
            results['messages'].append(
                f"D={cfg.D:.6f} is below D_min={D_min:.6f} for dx={cfg.dx*10:.1f}mm"
            )

        return results

    def run_propagation(
        self,
        D: Optional[float] = None,
        celltype=None,
        device: str = 'cuda',
        stim_duration: float = 1.0,
        total_time: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run 1D cable propagation simulation.

        Parameters
        ----------
        D : float, optional
            Diffusion coefficient to test. If None, uses config.D.
        celltype : CellType, optional
            Cell type. Default: ENDO.
        device : str
            Device for computation.
        stim_duration : float
            Stimulus duration (ms).
        total_time : float, optional
            Total simulation time (ms). If None, auto-calculated.

        Returns
        -------
        t : ndarray
            Time array
        V : ndarray (n_times, n_cells)
            Voltage history along cable
        """
        from ionic import CellType, ORdModel, StateIndex
        from tissue.diffusion import DiffusionOperator

        if celltype is None:
            celltype = CellType.ENDO
        if D is None:
            D = self._config.D

        cfg = self._config

        # Create as 1D (ny=1, nx=n_cells)
        ny, nx = 1, cfg.n_cells

        # Estimate time needed for propagation
        cv_est = self._compute_cv_from_D(D)
        if total_time is None:
            # Time to traverse cable + margin
            total_time = 1.5 * cfg.length / cv_est + 50.0

        # Initialize
        model = ORdModel(celltype=celltype, device=device)
        initial_state = model.get_initial_state()
        states = initial_state.unsqueeze(0).unsqueeze(0).expand(ny, nx, -1).clone()

        # Create diffusion (1D: D_L = D_T = D)
        diffusion = DiffusionOperator(
            ny=ny, nx=nx,
            dx=cfg.dx, dy=cfg.dx,
            D_L=D, D_T=D,
            fiber_angle=0.0,
            device=device
        )

        # Storage
        n_steps = int(total_time / cfg.dt)
        save_every = max(1, int(0.1 / cfg.dt))  # Save every ~0.1 ms
        t_history = []
        V_history = []

        t = 0.0
        stim_amplitude = 80.0  # µA/µF

        for step in range(n_steps):
            # Apply stimulus to first few cells
            if t < stim_duration:
                states[0, 0:3, StateIndex.V] = 20.0  # Strong depolarization

            # Ionic step
            Istim = torch.zeros((ny, nx), device=device, dtype=states.dtype)
            states = model.step(states, cfg.dt, Istim)

            # Diffusion step
            V = states[:, :, StateIndex.V]
            diff = diffusion.apply(V[0, :].unsqueeze(0))
            states[0, :, StateIndex.V] = states[0, :, StateIndex.V] + cfg.dt * diff[0, :]

            t += cfg.dt

            # Save
            if step % save_every == 0:
                t_history.append(t)
                V_history.append(states[0, :, StateIndex.V].cpu().numpy().copy())

        return np.array(t_history), np.array(V_history)

    def measure_cv(
        self,
        D: Optional[float] = None,
        threshold: float = -40.0,
        **kwargs
    ) -> float:
        """
        Measure conduction velocity for given D.

        Parameters
        ----------
        D : float, optional
            Diffusion coefficient to test.
        threshold : float
            Voltage threshold for activation detection (mV).

        Returns
        -------
        cv : float
            Measured conduction velocity (cm/ms).
        """
        t, V = self.run_propagation(D=D, **kwargs)
        cfg = self._config

        # Compute activation times
        n_cells = V.shape[1]
        act_times = np.full(n_cells, np.nan)

        for j in range(n_cells):
            trace = V[:, j]
            for i in range(1, len(trace)):
                if trace[i] > threshold and trace[i-1] <= threshold:
                    # Linear interpolation
                    frac = (threshold - trace[i-1]) / (trace[i] - trace[i-1])
                    act_times[j] = t[i-1] + frac * (t[i] - t[i-1])
                    break

        # Compute CV from activation times
        valid = ~np.isnan(act_times)
        if np.sum(valid) < 10:
            return np.nan

        # Use middle portion of cable (avoid boundary effects)
        n_valid = np.sum(valid)
        start_idx = n_valid // 4
        end_idx = 3 * n_valid // 4

        x = np.arange(n_cells)[valid][start_idx:end_idx] * cfg.dx
        t_act = act_times[valid][start_idx:end_idx]

        # Linear fit: x = CV * t + offset
        coeffs = np.polyfit(t_act, x, 1)
        cv = coeffs[0]  # slope = CV in cm/ms

        return cv

    def tune_D_to_cv(
        self,
        target_cv: Optional[float] = None,
        tolerance: float = 0.001,
        max_iterations: int = 10,
        verbose: bool = True,
        **kwargs
    ) -> float:
        """
        Iteratively tune D to achieve target CV.

        Uses Newton-Raphson-like iteration based on CV ∝ sqrt(D).

        Parameters
        ----------
        target_cv : float, optional
            Target CV (cm/ms). If None, uses config.target_cv.
        tolerance : float
            Relative tolerance for CV matching.
        max_iterations : int
            Maximum tuning iterations.
        verbose : bool
            Print progress.

        Returns
        -------
        D_tuned : float
            Tuned diffusion coefficient (cm²/ms).
        """
        if target_cv is None:
            target_cv = self._config.target_cv

        # Initial guess
        D = self._compute_D_from_cv(target_cv)

        if verbose:
            print(f"Tuning D for target CV = {target_cv*10:.3f} m/s ({target_cv:.5f} cm/ms)")
            print(f"Initial D estimate: {D:.6f} cm²/ms")
            print("-" * 50)

        for iteration in range(max_iterations):
            # Measure CV
            cv_measured = self.measure_cv(D=D, **kwargs)

            if np.isnan(cv_measured):
                print(f"  Iteration {iteration+1}: Propagation failed at D={D:.6f}")
                # D too low, increase
                D *= 1.5
                continue

            error = (cv_measured - target_cv) / target_cv

            if verbose:
                print(f"  Iteration {iteration+1}: D={D:.6f} -> CV={cv_measured*10:.4f} m/s "
                      f"(target: {target_cv*10:.4f}, error: {error*100:.2f}%)")

            if abs(error) < tolerance:
                if verbose:
                    print(f"\nConverged! D = {D:.6f} cm²/ms for CV = {target_cv*10:.3f} m/s")
                return D

            # Update D using CV ∝ sqrt(D) relationship
            # If CV is too high, reduce D; if too low, increase D
            # New D = D * (target_cv / cv_measured)^2
            D_new = D * (target_cv / cv_measured) ** 2

            # Damping for stability
            D = 0.5 * D + 0.5 * D_new

        if verbose:
            print(f"\nMax iterations reached. Final D = {D:.6f} cm²/ms")

        return D

    def print_summary(self):
        """Print cable configuration summary."""
        cfg = self._config
        cv_est = self._compute_cv_from_D(cfg.D)

        print("=" * 50)
        print("1D CABLE CONFIGURATION (CV Tuning)")
        print("=" * 50)
        print(f"Length: {cfg.length:.1f} cm")
        print(f"Cells: {cfg.n_cells}")
        print(f"dx: {cfg.dx*10:.1f} mm ({cfg.dx:.4f} cm)")
        print(f"dt: {cfg.dt:.4f} ms")
        print(f"D: {cfg.D:.6f} cm²/ms")
        print(f"Target CV: {cfg.target_cv*10:.2f} m/s")
        print(f"Estimated CV: {cv_est*10:.2f} m/s")
        print("=" * 50)


# =============================================================================
# 2D Tissue Mesh Configuration
# =============================================================================

@dataclass
class MeshConfig:
    """
    Configuration container for 2D tissue mesh parameters.

    Units:
    - Lengths: cm
    - Time: ms
    - Velocity: cm/ms (1 m/s = 0.1 cm/ms)
    - Diffusion: cm²/ms
    """
    # Domain size (cm)
    Lx: float = 15.0        # Domain width (x-direction)
    Ly: float = 15.0        # Domain height (y-direction)

    # Grid spacing (cm)
    dx: float = 0.02        # Grid spacing in x (200 µm)
    dy: float = 0.02        # Grid spacing in y (200 µm)

    # Time step (ms)
    dt: float = 0.02        # Time step

    # Anisotropy control
    anisotropic: bool = True  # If False, D_T = D_L (isotropic, CV = 0.6 m/s both)

    # Target conduction velocities (cm/ms)
    cv_longitudinal: float = 0.06   # 0.6 m/s along fibers
    cv_transverse: float = 0.03     # 0.3 m/s across fibers (ignored if anisotropic=False)

    # Fiber orientation
    fiber_angle: float = 0.0        # Fiber angle (radians), 0 = fibers along x

    # Physical constants
    chi: float = CHI
    Cm: float = CM

    # Expected APD for stability validation
    apd_ms: float = APD_REF

    # Computed values (set by __post_init__ or MeshBuilder)
    nx: int = field(init=False)
    ny: int = field(init=False)
    D_L: float = field(init=False)
    D_T: float = field(init=False)

    def __post_init__(self):
        """Compute derived values."""
        self.nx = int(round(self.Lx / self.dx))
        self.ny = int(round(self.Ly / self.dy))
        # D values: use tuned defaults
        self.D_L = D_L_DEFAULT
        if self.anisotropic:
            self.D_T = D_T_DEFAULT
        else:
            # Isotropic: D_T = D_L (both CV = 0.6 m/s)
            self.D_T = D_L_DEFAULT


class MeshBuilder:
    """
    Builder class for configuring 2D tissue simulation mesh.

    Features:
    - Fluent API for configuration
    - Pre-tuned diffusion coefficients for accurate CV
    - Anisotropic (default) or isotropic diffusion
    - Integration with CableMesh for custom CV tuning
    - Validation against stability and D_min constraints

    Default Configuration:
    - Domain: 15 x 15 cm
    - dx = dy = 0.02 cm (200 µm)
    - dt = 0.02 ms
    - Anisotropic: True (CV_long = 0.6 m/s, CV_trans = 0.3 m/s)
    - Pre-tuned D values: D_L = 0.002161, D_T = 0.000819 cm²/ms

    Example
    -------
    >>> # Default anisotropic (CV_long=0.6, CV_trans=0.3 m/s)
    >>> mesh = MeshBuilder.create_default()

    >>> # Isotropic (CV = 0.6 m/s in all directions)
    >>> mesh = MeshBuilder.create_default(anisotropic=False)

    >>> # Custom CV tuning via 1D cable simulations
    >>> mesh = MeshBuilder.create_with_tuned_cv(cv_long=0.07, cv_trans=0.025)

    >>> # Manual configuration
    >>> mesh = (MeshBuilder()
    ...         .set_domain(15.0, 15.0)
    ...         .set_resolution(0.02)
    ...         .set_anisotropic(False)
    ...         .build())
    """

    def __init__(self):
        """Initialize with default values."""
        self._config = MeshConfig()
        self._validated = False
        self._anisotropic = True  # Default: anisotropic

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def create_default(cls, anisotropic: bool = True) -> 'MeshBuilder':
        """
        Create mesh with default parameters and pre-tuned D values.

        Uses pre-calibrated diffusion coefficients for accurate CV:
        - D_L = 0.002161 cm²/ms -> CV = 0.6 m/s (longitudinal)
        - D_T = 0.000819 cm²/ms -> CV = 0.3 m/s (transverse, if anisotropic)

        Parameters
        ----------
        anisotropic : bool
            If True (default): D_L != D_T (CV_long=0.6, CV_trans=0.3 m/s)
            If False: D_T = D_L (isotropic, CV = 0.6 m/s in all directions)

        Returns
        -------
        MeshBuilder
            Configured mesh with tuned D values.
        """
        builder = cls()
        builder.set_domain(15.0, 15.0)
        builder.set_resolution(0.02, 0.02)
        builder.set_time_step(0.02)
        builder.set_anisotropic(anisotropic)
        builder.validate()
        return builder

    @classmethod
    def create_isotropic(cls) -> 'MeshBuilder':
        """
        Create mesh with isotropic diffusion (CV = 0.6 m/s in all directions).

        Shortcut for create_default(anisotropic=False).
        Useful for spiral wave simulations where isotropic propagation is desired.
        """
        return cls.create_default(anisotropic=False)

    @classmethod
    def create_with_tuned_cv(
        cls,
        cv_long: float = 0.06,
        cv_trans: Optional[float] = None,
        dx: float = 0.02,
        dt: float = 0.02,
        anisotropic: bool = True,
        tune_verbose: bool = True,
        device: str = 'cuda'
    ) -> 'MeshBuilder':
        """
        Create mesh with CV tuning via 1D cable simulations.

        Runs actual 1D cable simulations to calibrate D_L and D_T
        for accurate CV matching.

        Parameters
        ----------
        cv_long : float
            Target longitudinal CV (cm/ms). Default 0.06 = 0.6 m/s.
        cv_trans : float, optional
            Target transverse CV (cm/ms). Default 0.03 = 0.3 m/s.
            Ignored if anisotropic=False.
        dx : float
            Grid spacing (cm). Default 0.02 = 200 µm.
        dt : float
            Time step (ms). Default 0.02.
        anisotropic : bool
            If True: tune both D_L and D_T separately.
            If False: D_T = D_L (only tune longitudinal, use for both).
        tune_verbose : bool
            Print tuning progress.
        device : str
            Device for tuning simulations.

        Returns
        -------
        MeshBuilder
            Configured mesh with tuned D values.
        """
        if cv_trans is None:
            cv_trans = 0.03

        builder = cls()
        builder.set_domain(15.0, 15.0)
        builder.set_resolution(dx, dx)
        builder.set_time_step(dt)
        builder._anisotropic = anisotropic

        # Tune D_L for longitudinal CV
        if tune_verbose:
            print("\n" + "=" * 60)
            print("TUNING LONGITUDINAL DIFFUSION COEFFICIENT")
            print("=" * 60)

        cable_L = CableMesh.create_for_cv_tuning(
            target_cv=cv_long, dx=dx, dt=dt
        )
        D_L = cable_L.tune_D_to_cv(verbose=tune_verbose, device=device)

        if anisotropic:
            # Tune D_T for transverse CV
            if tune_verbose:
                print("\n" + "=" * 60)
                print("TUNING TRANSVERSE DIFFUSION COEFFICIENT")
                print("=" * 60)

            cable_T = CableMesh.create_for_cv_tuning(
                target_cv=cv_trans, dx=dx, dt=dt
            )
            D_T = cable_T.tune_D_to_cv(verbose=tune_verbose, device=device)
        else:
            # Isotropic: D_T = D_L
            D_T = D_L
            if tune_verbose:
                print(f"\nIsotropic mode: D_T = D_L = {D_L:.6f} cm²/ms")

        # Set tuned values
        builder._config.anisotropic = anisotropic
        builder._config.cv_longitudinal = cv_long
        builder._config.cv_transverse = cv_trans if anisotropic else cv_long
        builder._config.D_L = D_L
        builder._config.D_T = D_T
        builder._config.__post_init__()  # Recompute grid dimensions

        builder.validate()
        return builder

    # =========================================================================
    # Fluent Configuration API
    # =========================================================================

    def set_domain(self, Lx: float, Ly: Optional[float] = None) -> 'MeshBuilder':
        """Set domain size in cm."""
        self._config.Lx = Lx
        self._config.Ly = Ly if Ly is not None else Lx
        self._validated = False
        return self

    def set_resolution(self, dx: float, dy: Optional[float] = None) -> 'MeshBuilder':
        """Set spatial resolution in cm."""
        self._config.dx = dx
        self._config.dy = dy if dy is not None else dx
        self._validated = False
        return self

    def set_time_step(self, dt: float) -> 'MeshBuilder':
        """Set time step in ms."""
        self._config.dt = dt
        self._validated = False
        return self

    def set_anisotropic(self, anisotropic: bool) -> 'MeshBuilder':
        """
        Set anisotropic or isotropic diffusion mode.

        Parameters
        ----------
        anisotropic : bool
            If True: D_L != D_T (default CV_long=0.6, CV_trans=0.3 m/s)
            If False: D_T = D_L (isotropic, CV = 0.6 m/s in all directions)
        """
        self._anisotropic = anisotropic
        self._config.anisotropic = anisotropic
        self._validated = False
        return self

    def set_cv(
        self,
        cv_long: float,
        cv_trans: Optional[float] = None,
        anisotropy_ratio: float = 2.0
    ) -> 'MeshBuilder':
        """
        Set target conduction velocities.

        Note: This updates CV targets but uses pre-tuned D values.
        For custom CVs, use create_with_tuned_cv() instead.
        """
        self._config.cv_longitudinal = cv_long
        if cv_trans is not None:
            self._config.cv_transverse = cv_trans
        else:
            self._config.cv_transverse = cv_long / anisotropy_ratio
        self._validated = False
        return self

    def set_D(self, D_L: float, D_T: float) -> 'MeshBuilder':
        """
        Set diffusion coefficients directly.

        Use after tuning with CableMesh.
        """
        self._config.D_L = D_L
        self._config.D_T = D_T
        self._validated = False
        return self

    def set_fiber_angle(self, angle: float) -> 'MeshBuilder':
        """Set fiber orientation angle in radians."""
        self._config.fiber_angle = angle
        self._validated = False
        return self

    def set_apd(self, apd_ms: float) -> 'MeshBuilder':
        """Set expected APD for stability validation."""
        self._config.apd_ms = apd_ms
        self._validated = False
        return self

    # =========================================================================
    # Validation
    # =========================================================================

    def _compute_D_from_cv(self, cv: float) -> float:
        """Compute D from CV using empirical relationship."""
        return (cv / CV_EMPIRICAL_CONSTANT) ** 2

    def _compute_diffusion_coefficients(self):
        """Set D_L and D_T using tuned defaults based on anisotropic mode."""
        # Use pre-tuned defaults
        self._config.D_L = D_L_DEFAULT  # CV = 0.6 m/s

        if self._anisotropic:
            self._config.D_T = D_T_DEFAULT  # CV = 0.3 m/s
            self._config.cv_transverse = 0.03
        else:
            # Isotropic: D_T = D_L (CV = 0.6 m/s in all directions)
            self._config.D_T = D_L_DEFAULT
            self._config.cv_transverse = 0.06

    def validate(self, raise_error: bool = False) -> Dict[str, Any]:
        """Validate mesh configuration."""
        self._config.__post_init__()
        self._compute_diffusion_coefficients()

        results = {
            'status': 'OK',
            'grid': {},
            'stability': {},
            'diffusion': {},
            'messages': []
        }

        cfg = self._config

        # Grid validation
        results['grid'] = {
            'nx': cfg.nx,
            'ny': cfg.ny,
            'total_cells': cfg.nx * cfg.ny
        }

        # Stability validation
        D_max = max(cfg.D_L, cfg.D_T)
        dx_min = min(cfg.dx, cfg.dy)
        dt_max = 0.25 * dx_min ** 2 / D_max
        stability_ratio = cfg.dt / dt_max

        results['stability'] = {
            'dt': cfg.dt,
            'dt_max': dt_max,
            'ratio': stability_ratio,
            'safe': stability_ratio < 0.9
        }

        if stability_ratio > 1.0:
            results['status'] = 'CRITICAL'
            results['messages'].append(
                f"dt={cfg.dt} exceeds stability limit dt_max={dt_max:.5f} ms"
            )

        # Diffusion validation
        D_min = compute_D_min(cfg.dx, cfg.apd_ms)
        results['diffusion'] = {
            'D_L': cfg.D_L,
            'D_T': cfg.D_T,
            'D_min': D_min,
            'D_L_ratio': cfg.D_L / D_min,
            'D_T_ratio': cfg.D_T / D_min
        }

        # Check D values
        for name, D in [('longitudinal', cfg.D_L), ('transverse', cfg.D_T)]:
            status, msg = validate_D_for_mesh(D, cfg.dx, cfg.apd_ms, direction=name)
            if status == 'CRITICAL':
                results['status'] = 'CRITICAL'
                results['messages'].append(msg)
            elif status == 'WARNING' and results['status'] == 'OK':
                results['status'] = 'WARNING'
                results['messages'].append(msg)

        self._validated = True

        if raise_error and results['status'] == 'CRITICAL':
            raise ValueError('\n'.join(results['messages']))

        return results

    # =========================================================================
    # Build
    # =========================================================================

    def build(self) -> MeshConfig:
        """Build and return the mesh configuration."""
        if not self._validated:
            self.validate()
        return self._config

    def get_config(self) -> MeshConfig:
        """Alias for build()."""
        return self.build()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def print_summary(self):
        """Print configuration summary."""
        if not self._validated:
            self.validate()

        cfg = self._config
        D_min = compute_D_min(cfg.dx, cfg.apd_ms)

        print("=" * 60)
        print("2D TISSUE MESH CONFIGURATION")
        print("=" * 60)
        print(f"\nDomain:")
        print(f"  Size: {cfg.Lx:.1f} x {cfg.Ly:.1f} cm")
        print(f"  Grid: {cfg.nx} x {cfg.ny} = {cfg.nx * cfg.ny:,} cells")
        print(f"  Resolution: dx={cfg.dx*10:.1f} mm, dy={cfg.dy*10:.1f} mm")

        print(f"\nTime:")
        print(f"  dt = {cfg.dt:.4f} ms")
        D_max = max(cfg.D_L, cfg.D_T)
        dx_min = min(cfg.dx, cfg.dy)
        dt_max = 0.25 * dx_min ** 2 / D_max
        print(f"  dt_max (stability) = {dt_max:.4f} ms")
        print(f"  Safety margin: {100 * cfg.dt / dt_max:.0f}% of limit")

        print(f"\nDiffusion Mode:")
        if cfg.anisotropic:
            print(f"  Mode: ANISOTROPIC (CV_long != CV_trans)")
            print(f"  CV_long  = {cfg.cv_longitudinal*10:.2f} m/s ({cfg.cv_longitudinal:.4f} cm/ms)")
            print(f"  CV_trans = {cfg.cv_transverse*10:.2f} m/s ({cfg.cv_transverse:.4f} cm/ms)")
            print(f"  Anisotropy ratio: {cfg.cv_longitudinal/cfg.cv_transverse:.1f}:1")
        else:
            print(f"  Mode: ISOTROPIC (CV = 0.6 m/s in all directions)")
            print(f"  CV = {cfg.cv_longitudinal*10:.2f} m/s ({cfg.cv_longitudinal:.4f} cm/ms)")

        print(f"\nDiffusion Coefficients (pre-tuned):")
        print(f"  D_L = {cfg.D_L:.6f} cm²/ms")
        print(f"  D_T = {cfg.D_T:.6f} cm²/ms")
        print(f"  D_min = {D_min:.6f} cm²/ms (for dx={cfg.dx*10:.1f}mm)")
        print(f"  D_L/D_min = {cfg.D_L/D_min:.2f}x")
        print(f"  D_T/D_min = {cfg.D_T/D_min:.2f}x")

        print(f"\nFiber Orientation:")
        print(f"  Angle: {math.degrees(cfg.fiber_angle):.1f}° from x-axis")
        print("=" * 60)

    def create_simulation(
        self,
        celltype=None,
        device: str = 'cuda',
        dtype=torch.float64,
        params_override: Optional[Dict[str, float]] = None
    ):
        """
        Create MonodomainSimulation from this mesh configuration.

        Parameters
        ----------
        celltype : CellType, optional
            Cell type (ENDO, EPI, M_CELL). Default: ENDO.
        device : str
            Computation device.
        dtype : torch.dtype
            Data type for tensors.
        params_override : dict, optional
            ORd model parameter overrides.

        Returns
        -------
        MonodomainSimulation
            Configured simulation instance.
        """
        from ionic import CellType
        from tissue.simulation import MonodomainSimulation

        if celltype is None:
            celltype = CellType.ENDO

        if not self._validated:
            self.validate()

        cfg = self._config

        return MonodomainSimulation(
            ny=cfg.ny,
            nx=cfg.nx,
            dx=cfg.dx,
            dy=cfg.dy,
            D_L=cfg.D_L,
            D_T=cfg.D_T,
            fiber_angle=cfg.fiber_angle,
            celltype=celltype,
            chi=cfg.chi,
            Cm=cfg.Cm,
            device=device,
            dtype=dtype,
            params_override=params_override,
            apd_ms=cfg.apd_ms
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_default_mesh() -> MeshBuilder:
    """Create default 15x15 cm mesh with dx=0.02, dt=0.02."""
    return MeshBuilder.create_default()


def create_cv_tuning_cable(
    target_cv: float = 0.06,
    dx: float = 0.02,
    dt: float = 0.02
) -> CableMesh:
    """
    Create 1D cable for CV tuning.

    Parameters
    ----------
    target_cv : float
        Target CV in cm/ms (0.06 = 0.6 m/s).
    dx : float
        Grid spacing in cm.
    dt : float
        Time step in ms.

    Returns
    -------
    CableMesh
        Configured cable for CV tuning.
    """
    return CableMesh.create_for_cv_tuning(target_cv=target_cv, dx=dx, dt=dt)
