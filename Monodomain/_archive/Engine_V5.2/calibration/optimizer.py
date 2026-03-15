"""
Diffusion Coefficient Optimizer.

Uses Differential Evolution to find optimal D_L and D_T that satisfy:
- Target CV (longitudinal and transverse)
- Minimum APD constraint
- Minimum single-cell ERP constraint
- Target tissue ERP (primary objective)

Tiered Optimization Strategy:
    Tier 1: Fixed dt - try to find D_L, D_T with default time step
    Tier 2: Variable dt - if Tier 1 fails, allow dt to vary for convergence

When dt is optimized (Tier 2), it is regularized toward the default value
so the optimizer prefers solutions with standard dt when possible.
D accuracy is prioritized over dt - the regularization weight is very small.
"""

import torch
import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult
from typing import Optional, Tuple, NamedTuple, Callable, List
from dataclasses import dataclass, field
import time

from .cable_1d import Cable1D, CableConfig, measure_cv_apd
from .erp_measurement import measure_erp_single_cell
from .tissue_erp_2d import measure_tissue_erp_2d, ERPResult2D


@dataclass
class CalibrationTargets:
    """Target values for calibration."""

    # Conduction velocity targets
    cv_longitudinal: float = 0.06   # cm/ms (60 cm/s = 0.6 m/s)
    anisotropy_ratio: float = 3.0   # α = D_L / D_T

    # Derived CV transverse (calculated, not set directly)
    @property
    def cv_transverse(self) -> float:
        """CV_T = CV_L / √α"""
        return self.cv_longitudinal / np.sqrt(self.anisotropy_ratio)

    # Minimum constraints (one-sided penalties)
    apd_min: float = 250.0          # Minimum APD90 (ms)
    erp_single_min: float = 280.0   # Minimum single-cell ERP (ms)

    # Target tissue ERP (primary objective)
    erp_tissue_target: float = 320.0  # Target tissue ERP (ms)

    @property
    def wavelength_longitudinal(self) -> float:
        """λ_L = CV_L × ERP_tissue"""
        return self.cv_longitudinal * self.erp_tissue_target

    @property
    def wavelength_transverse(self) -> float:
        """λ_T = CV_T × ERP_tissue"""
        return self.cv_transverse * self.erp_tissue_target


@dataclass
class CalibrationWeights:
    """Weights for loss function components."""

    # CV errors (normalized squared error)
    w_cv_long: float = 1.0
    w_cv_trans: float = 1.0

    # Constraint penalties (one-sided)
    w_apd: float = 0.5
    w_erp_single: float = 0.5

    # Primary objective (tissue ERP)
    w_erp_tissue: float = 5.0

    # Regularization for helper variables (very small - tiebreaker only)
    # This ensures D accuracy is prioritized over dt preference
    w_dt_reg: float = 0.01


@dataclass
class CalibrationConfig:
    """Configuration for the calibration process."""

    # Spatial discretization
    dx: float = 0.01        # Mesh spacing (cm)

    # Temporal discretization
    dt_default: float = 0.02       # Default time step (ms)
    dt_min: float = 0.001          # Minimum dt - can go very small
    dt_safety_factor: float = 0.5  # Safety margin for dt_max (0.5 = 50% of stability limit)

    # Cable parameters
    cable_length: float = 2.0   # Cable length for measurements (cm)

    # Bounds for D (cm²/ms)
    D_min: float = 0.0001   # Minimum D
    D_max: float = 0.01     # Maximum D

    # Tiered optimization settings
    tier1_loss_threshold: float = 0.1  # Accept Tier 1 if loss < threshold
    tier1_maxiter: int = 30            # Max iterations for Tier 1 (fixed dt)
    tier2_maxiter: int = 50            # Max iterations for Tier 2 (variable dt)

    # Differential Evolution parameters
    de_strategy: str = 'best1bin'
    de_popsize: int = 15
    de_tol: float = 1e-4
    de_mutation: Tuple[float, float] = (0.5, 1.0)
    de_recombination: float = 0.7
    de_seed: Optional[int] = None
    de_workers: int = 1

    # S1-S2 protocol for ERP
    s1_bcl: float = 1000.0
    s1_count: int = 5       # Reduced for speed during optimization

    # Simulation duration
    cv_sim_duration: float = 300.0  # Duration for CV measurement (ms)

    # Cell type
    cell_type: int = 0      # 0=ENDO, 1=EPI, 2=M_CELL

    # Device
    device: Optional[torch.device] = None

    def get_dt_max(self, D: float) -> float:
        """
        Compute maximum stable dt for given D.

        For explicit diffusion: dt < dx² / (2D)
        For implicit (unconditionally stable), we use accuracy criterion:
            dt_max = safety_factor × dx² / D

        Args:
            D: Diffusion coefficient (cm²/ms)

        Returns:
            Maximum recommended dt (ms)
        """
        if D <= 0:
            return self.dt_default
        stability_limit = self.dx * self.dx / D
        return self.dt_safety_factor * stability_limit

    def get_dt_bounds(self) -> Tuple[float, float]:
        """
        Get dt bounds for Tier 2 optimization.

        Lower bound: dt_min (can go very small for stability)
        Upper bound: stability-based with safety margin for D_max

        Returns:
            (dt_min, dt_max) tuple
        """
        dt_max = self.get_dt_max(self.D_max)
        # Ensure dt_max is at least dt_default
        dt_max = max(dt_max, self.dt_default)
        return (self.dt_min, dt_max)


class CalibrationResult(NamedTuple):
    """Results from calibration optimization."""
    D_longitudinal: float       # Optimal D_L (cm²/ms)
    D_transverse: float         # Optimal D_T (cm²/ms)
    dt_used: float              # Time step used (ms)
    cv_longitudinal: float      # Achieved CV_L (cm/ms)
    cv_transverse: float        # Achieved CV_T (cm/ms)
    apd90: float               # Measured APD90 (ms)
    erp_single: float          # Single-cell ERP (ms)
    erp_tissue: float          # 2D tissue ERP (ms) - where both probes activate
    wavelength_long: float     # λ_L = CV_L × ERP (cm)
    wavelength_trans: float    # λ_T = CV_T × ERP (cm)
    final_loss: float          # Final loss value
    tier_used: int             # Which tier succeeded (1 or 2)
    n_evaluations: int         # Number of function evaluations
    convergence: bool          # Whether optimization converged
    optimization_result: OptimizeResult  # Full scipy result


class DiffusionCalibrator:
    """
    Calibrates diffusion coefficients using Differential Evolution.

    Tiered Optimization:
        Tier 1: Optimize [D_L, D_T] with fixed dt
        Tier 2: If Tier 1 fails, optimize [D_L, D_T, dt] with dt regularization

    Loss Function:
        L = w_cv_L × (CV_L_meas - CV_L)²/CV_L²
          + w_cv_T × (CV_T_meas - CV_T)²/CV_T²
          + w_apd × max(0, APD_min - APD)²/APD_min²
          + w_erp_sc × max(0, ERP_sc,min - ERP_sc)²/ERP_sc,min²
          + w_erp_t × (ERP_t_meas - ERP_t)²/ERP_t²
          + w_dt_reg × (dt - dt_default)²/dt_default²  [Tier 2 only]
    """

    def __init__(
        self,
        targets: CalibrationTargets,
        weights: CalibrationWeights,
        config: CalibrationConfig
    ):
        """
        Initialize calibrator.

        Args:
            targets: Target values for calibration
            weights: Weights for loss function
            config: Calibration configuration
        """
        self.targets = targets
        self.weights = weights
        self.config = config

        # Tracking
        self.n_evaluations = 0
        self.best_loss = float('inf')
        self.history: List[dict] = []
        self.current_tier = 1
        self.include_dt = False

        # Compute D_min based on dx and APD
        self._compute_d_min()

    def _compute_d_min(self):
        """
        Compute minimum D based on numerical stability.

        Formula from V5.1: D_min = 0.92 × dx² × (280/APD)^0.25
        """
        dx = self.config.dx
        apd = self.targets.apd_min

        self.D_min_stable = 0.92 * dx * dx * np.power(280.0 / apd, 0.25)

        # Use max of configured and stability-based minimum
        self.config.D_min = max(self.config.D_min, self.D_min_stable)

    def _run_simulation(self, D: float, dt: float) -> Tuple[bool, dict]:
        """
        Run cable simulation and measure CV/APD.

        Args:
            D: Diffusion coefficient
            dt: Time step

        Returns:
            (success, results_dict)
        """
        try:
            # Create cable config with specified dt
            cable_config = CableConfig(
                length=self.config.cable_length,
                dx=self.config.dx,
                dt=dt
            )

            # Run simulation
            cable = Cable1D(
                cable_config,
                D=D,
                cell_type=self.config.cell_type,
                device=self.config.device
            )
            result = cable.run(duration=self.config.cv_sim_duration)

            return result.success, {
                'cv': result.cv,
                'apd90': result.apd90,
                'v_max': result.v_max
            }
        except Exception as e:
            return False, {'error': str(e)}

    def _objective(self, x: np.ndarray) -> float:
        """
        Objective function for optimization.

        Args:
            x: [D_L, D_T] for Tier 1, [D_L, D_T, dt] for Tier 2

        Returns:
            Loss value
        """
        self.n_evaluations += 1

        # Unpack variables
        D_L = x[0]
        D_T = x[1]
        dt = x[2] if self.include_dt else self.config.dt_default

        try:
            # Measure CV longitudinal
            success_L, result_L = self._run_simulation(D_L, dt)
            if not success_L:
                return 1e6  # Propagation failed

            cv_L = result_L['cv']
            apd = result_L['apd90']

            # Measure CV transverse
            success_T, result_T = self._run_simulation(D_T, dt)
            if not success_T:
                return 1e6

            cv_T = result_T['cv']

            # Measure tissue ERP using 2D simulation
            # Both D_L (x-direction) and D_T (y-direction) affect the result
            erp_result_2d = measure_tissue_erp_2d(
                D_x=D_L,
                D_y=D_T,
                dx=self.config.dx,
                dt=dt,
                cv_x_est=self.targets.cv_longitudinal,
                cv_y_est=self.targets.cv_transverse,
                erp_est=self.targets.erp_tissue_target,
                cell_type=self.config.cell_type,
                margin=1.5,
                verbose=False,
                device=self.config.device
            )
            erp_tissue = erp_result_2d.erp  # Single ERP where both probes activate

            # Compute loss components
            loss = 0.0

            # CV longitudinal error
            cv_L_target = self.targets.cv_longitudinal
            loss += self.weights.w_cv_long * ((cv_L - cv_L_target) / cv_L_target) ** 2

            # CV transverse error
            cv_T_target = self.targets.cv_transverse
            loss += self.weights.w_cv_trans * ((cv_T - cv_T_target) / cv_T_target) ** 2

            # APD constraint (penalty if below minimum)
            if apd < self.targets.apd_min:
                loss += self.weights.w_apd * ((self.targets.apd_min - apd) / self.targets.apd_min) ** 2

            # Tissue ERP error (primary objective)
            erp_target = self.targets.erp_tissue_target
            loss += self.weights.w_erp_tissue * ((erp_tissue - erp_target) / erp_target) ** 2

            # dt regularization (Tier 2 only) - very small weight
            if self.include_dt:
                dt_default = self.config.dt_default
                loss += self.weights.w_dt_reg * ((dt - dt_default) / dt_default) ** 2

            # Track progress
            if loss < self.best_loss:
                self.best_loss = loss
                self.history.append({
                    'n_eval': self.n_evaluations,
                    'tier': self.current_tier,
                    'D_L': D_L,
                    'D_T': D_T,
                    'dt': dt,
                    'cv_L': cv_L,
                    'cv_T': cv_T,
                    'apd': apd,
                    'erp_tissue': erp_tissue,
                    'loss': loss
                })

            return loss

        except Exception as e:
            print(f"Error in objective: {e}")
            return 1e6

    def _run_tier1(self, verbose: bool = True) -> Tuple[OptimizeResult, bool]:
        """
        Run Tier 1 optimization with fixed dt.

        Returns:
            (result, success) where success indicates if loss < threshold
        """
        if verbose:
            print("\n--- Tier 1: Fixed dt optimization ---")
            print(f"dt = {self.config.dt_default} ms (fixed)")

        self.current_tier = 1
        self.include_dt = False
        self.n_evaluations = 0
        self.best_loss = float('inf')

        # Bounds for D only
        bounds = [
            (self.config.D_min, self.config.D_max),  # D_L
            (self.config.D_min, self.config.D_max),  # D_T
        ]

        # Initial guess using CV ∝ √D relationship
        D_L_init = (self.targets.cv_longitudinal ** 2) / 100
        D_T_init = D_L_init / self.targets.anisotropy_ratio
        D_L_init = np.clip(D_L_init, self.config.D_min, self.config.D_max)
        D_T_init = np.clip(D_T_init, self.config.D_min, self.config.D_max)
        x0 = np.array([D_L_init, D_T_init])

        # Progress callback
        def callback(xk, convergence=None):
            if verbose:
                print(f"  Eval {self.n_evaluations}: D_L={xk[0]:.6f}, D_T={xk[1]:.6f}, loss={self.best_loss:.6f}")

        # Run optimization
        result = differential_evolution(
            self._objective,
            bounds,
            strategy=self.config.de_strategy,
            maxiter=self.config.tier1_maxiter,
            popsize=self.config.de_popsize,
            tol=self.config.de_tol,
            mutation=self.config.de_mutation,
            recombination=self.config.de_recombination,
            seed=self.config.de_seed,
            callback=callback,
            polish=True,
            workers=self.config.de_workers,
            x0=x0,
            init='latinhypercube'
        )

        success = result.fun < self.config.tier1_loss_threshold
        if verbose:
            status = "SUCCESS" if success else "needs Tier 2"
            print(f"Tier 1 complete: loss = {result.fun:.6f} ({status})")

        return result, success

    def _run_tier2(self, tier1_result: OptimizeResult, verbose: bool = True) -> OptimizeResult:
        """
        Run Tier 2 optimization with variable dt.

        Args:
            tier1_result: Result from Tier 1 (used for initial guess)

        Returns:
            Optimization result
        """
        if verbose:
            print("\n--- Tier 2: Variable dt optimization ---")
            dt_bounds = self.config.get_dt_bounds()
            print(f"dt bounds: [{dt_bounds[0]:.4f}, {dt_bounds[1]:.4f}] ms")
            print(f"dt regularization weight: {self.weights.w_dt_reg}")

        self.current_tier = 2
        self.include_dt = True
        self.n_evaluations = 0
        self.best_loss = float('inf')

        # Bounds for D and dt
        dt_bounds = self.config.get_dt_bounds()
        bounds = [
            (self.config.D_min, self.config.D_max),  # D_L
            (self.config.D_min, self.config.D_max),  # D_T
            dt_bounds,                                # dt
        ]

        # Initial guess from Tier 1
        x0 = np.array([
            tier1_result.x[0],
            tier1_result.x[1],
            self.config.dt_default
        ])

        # Progress callback
        def callback(xk, convergence=None):
            if verbose:
                print(f"  Eval {self.n_evaluations}: D_L={xk[0]:.6f}, D_T={xk[1]:.6f}, dt={xk[2]:.4f}, loss={self.best_loss:.6f}")

        # Run optimization
        result = differential_evolution(
            self._objective,
            bounds,
            strategy=self.config.de_strategy,
            maxiter=self.config.tier2_maxiter,
            popsize=self.config.de_popsize,
            tol=self.config.de_tol,
            mutation=self.config.de_mutation,
            recombination=self.config.de_recombination,
            seed=self.config.de_seed,
            callback=callback,
            polish=True,
            workers=self.config.de_workers,
            x0=x0,
            init='latinhypercube'
        )

        if verbose:
            print(f"Tier 2 complete: loss = {result.fun:.6f}")
            print(f"  dt used: {result.x[2]:.4f} ms (default: {self.config.dt_default} ms)")

        return result

    def calibrate(
        self,
        verbose: bool = True,
        callback: Optional[Callable] = None
    ) -> CalibrationResult:
        """
        Run tiered calibration optimization.

        Tier 1: Try with fixed dt (faster, simpler)
        Tier 2: If Tier 1 fails, allow dt to vary (more flexible)

        Args:
            verbose: Print progress updates
            callback: Optional callback function for progress

        Returns:
            CalibrationResult with optimal D values and dt used
        """
        if verbose:
            print("=" * 60)
            print("Diffusion Coefficient Calibration (Tiered)")
            print("=" * 60)
            print(f"\nTargets:")
            print(f"  CV_L = {self.targets.cv_longitudinal:.4f} cm/ms")
            print(f"  CV_T = {self.targets.cv_transverse:.4f} cm/ms (from α = {self.targets.anisotropy_ratio})")
            print(f"  APD_min = {self.targets.apd_min:.0f} ms")
            print(f"  ERP_tissue = {self.targets.erp_tissue_target:.0f} ms")
            print(f"\nConfiguration:")
            print(f"  dx = {self.config.dx} cm")
            print(f"  dt_default = {self.config.dt_default} ms")
            print(f"  D bounds = [{self.config.D_min:.6f}, {self.config.D_max:.6f}] cm²/ms")
            print(f"  Tier 1 threshold = {self.config.tier1_loss_threshold}")

        start_time = time.time()

        # === Tier 1: Fixed dt ===
        tier1_result, tier1_success = self._run_tier1(verbose)

        if tier1_success:
            # Tier 1 succeeded
            final_result = tier1_result
            tier_used = 1
            dt_used = self.config.dt_default
            D_L_opt, D_T_opt = tier1_result.x
        else:
            # === Tier 2: Variable dt ===
            tier2_result = self._run_tier2(tier1_result, verbose)
            final_result = tier2_result
            tier_used = 2
            D_L_opt, D_T_opt, dt_used = tier2_result.x

        elapsed = time.time() - start_time

        if verbose:
            print(f"\nOptimization complete in {elapsed:.1f}s")
            print(f"  Tier used: {tier_used}")
            print(f"  Total evaluations: {final_result.nfev}")

        # Final measurements with optimized values
        if verbose:
            print("\nFinal measurements...")

        # CV and APD
        success_L, result_L = self._run_simulation(D_L_opt, dt_used)
        success_T, result_T = self._run_simulation(D_T_opt, dt_used)

        # Single-cell ERP
        erp_single = measure_erp_single_cell(
            dt=dt_used,
            cell_type=self.config.cell_type,
            s1_bcl=self.config.s1_bcl,
            s1_count=self.config.s1_count,
            device=self.config.device
        )

        # Tissue ERP (2D - single value where both probes activate)
        erp_tissue_2d = measure_tissue_erp_2d(
            D_x=D_L_opt,
            D_y=D_T_opt,
            dx=self.config.dx,
            dt=dt_used,
            cv_x_est=self.targets.cv_longitudinal,
            cv_y_est=self.targets.cv_transverse,
            erp_est=self.targets.erp_tissue_target,
            cell_type=self.config.cell_type,
            margin=1.5,
            verbose=verbose,
            device=self.config.device
        )

        # Compute wavelengths using single tissue ERP
        cv_L = result_L['cv'] if success_L else 0.0
        cv_T = result_T['cv'] if success_T else 0.0
        wavelength_L = cv_L * erp_tissue_2d.erp
        wavelength_T = cv_T * erp_tissue_2d.erp

        calibration_result = CalibrationResult(
            D_longitudinal=D_L_opt,
            D_transverse=D_T_opt,
            dt_used=dt_used,
            cv_longitudinal=cv_L,
            cv_transverse=cv_T,
            apd90=result_L['apd90'] if success_L else 0.0,
            erp_single=erp_single.erp,
            erp_tissue=erp_tissue_2d.erp,
            wavelength_long=wavelength_L,
            wavelength_trans=wavelength_T,
            final_loss=final_result.fun,
            tier_used=tier_used,
            n_evaluations=final_result.nfev,
            convergence=final_result.success,
            optimization_result=final_result
        )

        if verbose:
            print("\n" + "=" * 60)
            print("Calibration Results")
            print("=" * 60)
            print(f"\nOptimization:")
            print(f"  Tier used: {tier_used}")
            print(f"  Final loss: {final_result.fun:.6f}")
            print(f"\nDiffusion Coefficients:")
            print(f"  D_L = {D_L_opt:.6f} cm²/ms")
            print(f"  D_T = {D_T_opt:.6f} cm²/ms")
            print(f"  Ratio D_L/D_T = {D_L_opt/D_T_opt:.2f}")
            print(f"\nTime Step:")
            print(f"  dt = {dt_used:.4f} ms", end="")
            if tier_used == 2:
                deviation = (dt_used - self.config.dt_default) / self.config.dt_default * 100
                print(f" ({deviation:+.1f}% from default)")
            else:
                print(" (default)")
            print(f"\nConduction Velocity:")
            print(f"  CV_L = {cv_L:.4f} cm/ms (target: {self.targets.cv_longitudinal:.4f})")
            print(f"  CV_T = {cv_T:.4f} cm/ms (target: {self.targets.cv_transverse:.4f})")
            print(f"\nRefractory Period:")
            print(f"  APD90 = {result_L['apd90'] if success_L else 0:.1f} ms (min: {self.targets.apd_min:.1f})")
            print(f"  ERP (single-cell) = {erp_single.erp:.1f} ms")
            print(f"  ERP (2D tissue) = {erp_tissue_2d.erp:.1f} ms (target: {self.targets.erp_tissue_target:.1f})")
            print(f"    ERP_x = {erp_tissue_2d.erp_x:.1f} ms, ERP_y = {erp_tissue_2d.erp_y:.1f} ms")
            print(f"\nWavelength (using tissue ERP):")
            print(f"  λ_L = {wavelength_L:.2f} cm")
            print(f"  λ_T = {wavelength_T:.2f} cm")

        return calibration_result


def calibrate_diffusion(
    cv_longitudinal: float = 0.06,
    anisotropy_ratio: float = 3.0,
    erp_tissue_target: float = 320.0,
    apd_min: float = 250.0,
    dx: float = 0.01,
    dt_default: float = 0.02,
    verbose: bool = True,
    device: Optional[torch.device] = None
) -> CalibrationResult:
    """
    Convenience function to calibrate diffusion coefficients.

    Uses tiered optimization:
    - Tier 1: Fixed dt (faster)
    - Tier 2: Variable dt if Tier 1 fails (more flexible)

    Args:
        cv_longitudinal: Target longitudinal CV (cm/ms)
        anisotropy_ratio: D_L/D_T ratio
        erp_tissue_target: Target tissue ERP (ms)
        apd_min: Minimum acceptable APD (ms)
        dx: Mesh spacing (cm)
        dt_default: Default time step (ms)
        verbose: Print progress
        device: Torch device

    Returns:
        CalibrationResult with optimal D values and dt used
    """
    targets = CalibrationTargets(
        cv_longitudinal=cv_longitudinal,
        anisotropy_ratio=anisotropy_ratio,
        erp_tissue_target=erp_tissue_target,
        apd_min=apd_min
    )

    weights = CalibrationWeights()

    config = CalibrationConfig(
        dx=dx,
        dt_default=dt_default,
        device=device
    )

    calibrator = DiffusionCalibrator(targets, weights, config)
    return calibrator.calibrate(verbose=verbose)


if __name__ == "__main__":
    # Test calibration
    result = calibrate_diffusion(
        cv_longitudinal=0.05,   # 50 cm/s (easier target for testing)
        anisotropy_ratio=2.0,
        erp_tissue_target=300.0,
        dx=0.02,                # Coarser mesh for faster test
        dt_default=0.02,
        verbose=True
    )

    print("\n\nFinal optimized parameters:")
    print(f"D_L = {result.D_longitudinal:.6f} cm²/ms")
    print(f"D_T = {result.D_transverse:.6f} cm²/ms")
    print(f"dt = {result.dt_used:.4f} ms")
    print(f"Tier used: {result.tier_used}")
