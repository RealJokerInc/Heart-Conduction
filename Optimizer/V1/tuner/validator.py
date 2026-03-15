"""
Optimizer V1 — Validation Suite

Automated validation of tuned parameter sets:
- Novel CL pacing (untrained rates)
- CV verification
- Stimulus robustness
- Steady-state stability
- Spontaneous CL within tolerance
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .config import TuningConfig, TuningTargets, apply_scaling, theta_to_dict, dict_to_theta
from .cell_runner import run_single_cell, run_spontaneous, CellResult
from .tissue_runner import run_cv_measurement, CVResult
from .metrics import detect_aps, measure_apd


@dataclass
class ValidationCheck:
    """Single validation test result."""
    name: str
    passed: bool
    value: Optional[float] = None
    target: Optional[float] = None
    tolerance: Optional[float] = None
    message: str = ""


@dataclass
class ValidationResult:
    """Full validation suite result."""
    checks: List[ValidationCheck] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_total(self) -> int:
        return len(self.checks)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        lines = [f"Validation: {self.n_passed}/{self.n_total} passed"]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            val_str = f" = {c.value:.2f}" if c.value is not None else ""
            lines.append(f"  [{status}] {c.name}{val_str}: {c.message}")
        return "\n".join(lines)


def validate(theta_ionic: Dict[str, float],
             D_long: float,
             D_trans: float,
             config: TuningConfig,
             targets: TuningTargets,
             verbose: bool = True) -> ValidationResult:
    """
    Run full validation suite on a tuned parameter set.

    Parameters
    ----------
    theta_ionic : ionic scaling factors (name -> value)
    D_long : longitudinal diffusion coefficient
    D_trans : transverse diffusion coefficient
    config : TuningConfig
    targets : TuningTargets
    verbose : print progress

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()
    theta = dict_to_theta(theta_ionic, config.tier, config.dtype)

    if verbose:
        print("Validation Suite")
        print("=" * 50)

    # 1. Novel CL pacing
    novel_cls = [800.0, 600.0, 1500.0]  # ms
    for cl in novel_cls:
        novel_config = _with_cl(config, cl)
        cell_res = run_single_cell(theta, novel_config)
        check = ValidationCheck(
            name=f"novel_CL_{int(cl)}ms",
            passed=cell_res.converged and cell_res.apd90 is not None,
            value=cell_res.apd90,
            message=f"CL={cl}: APD90={cell_res.apd90:.0f} ms" if cell_res.apd90 else "Failed",
        )
        result.checks.append(check)

    # 2. CV verification
    cv_res = run_cv_measurement(theta, D_long, config)
    cv_check = ValidationCheck(
        name="cv_longitudinal",
        passed=(cv_res.cv is not None and
                abs(cv_res.cv - targets.cv_longitudinal) < targets.cv_longitudinal * 0.2),
        value=cv_res.cv,
        target=targets.cv_longitudinal,
        tolerance=targets.cv_longitudinal * 0.2,
        message=f"CV={cv_res.cv:.1f} cm/s (target {targets.cv_longitudinal})" if cv_res.cv else "Failed",
    )
    result.checks.append(cv_check)

    # 3. Stimulus robustness
    for stim_scale, label in [(2.0, "2x_stim"), (0.5, "0.5x_stim")]:
        robust_config = _with_stim_amplitude(config, config.stim_amplitude * stim_scale)
        cell_res = run_single_cell(theta, robust_config)
        check = ValidationCheck(
            name=f"stimulus_robustness_{label}",
            passed=cell_res.converged and cell_res.apd90 is not None,
            value=cell_res.apd90,
            message=f"APD90={cell_res.apd90:.0f} ms at {stim_scale}x stim" if cell_res.apd90 else "Failed",
        )
        result.checks.append(check)

    # 4. Steady-state stability (40 beats, no APD drift)
    stability_config = _with_beats(config, 40)
    cell_res = run_single_cell(theta, stability_config, return_trace=True)
    if cell_res.V_trace is not None and cell_res.converged:
        aps = detect_aps(cell_res.V_trace, cell_res.t_trace)
        if len(aps) >= 3:
            apds = []
            for ap in aps:
                apd = ap.end_time - ap.peak_time
                apds.append(apd)
            apd_std = np.std(apds[-5:]) if len(apds) >= 5 else np.std(apds)
            drift_ok = apd_std < 5.0  # Less than 5 ms variation
            check = ValidationCheck(
                name="steady_state_stability",
                passed=drift_ok,
                value=apd_std,
                tolerance=5.0,
                message=f"APD std = {apd_std:.1f} ms (last 5 beats)",
            )
        else:
            check = ValidationCheck(
                name="steady_state_stability",
                passed=False,
                message=f"Only {len(aps)} APs detected in 40 beats",
            )
    else:
        check = ValidationCheck(
            name="steady_state_stability",
            passed=False,
            message="Simulation failed or no trace",
        )
    result.checks.append(check)

    # 5. V_peak constraint
    cell_res_standard = run_single_cell(theta, config)
    vpeak_check = ValidationCheck(
        name="v_peak_constraint",
        passed=cell_res_standard.v_peak > targets.v_peak_min,
        value=cell_res_standard.v_peak,
        target=targets.v_peak_min,
        message=f"V_peak={cell_res_standard.v_peak:.1f} mV (min {targets.v_peak_min})",
    )
    result.checks.append(vpeak_check)

    # 6. Spontaneous CL (if target exists)
    if targets.spontaneous_cl is not None:
        spont = run_spontaneous(theta, config, duration_ms=8000.0)
        if spont.cl is not None:
            cl_ok = abs(spont.cl - targets.spontaneous_cl) < targets.spontaneous_cl * 0.1
            check = ValidationCheck(
                name="spontaneous_cl",
                passed=cl_ok,
                value=spont.cl,
                target=targets.spontaneous_cl,
                tolerance=targets.spontaneous_cl * 0.1,
                message=f"CL={spont.cl:.0f} ms (target {targets.spontaneous_cl})",
            )
        else:
            check = ValidationCheck(
                name="spontaneous_cl",
                passed=False,
                message="No spontaneous beating detected",
            )
        result.checks.append(check)

    if verbose:
        print(result.summary())

    return result


def _with_cl(config: TuningConfig, cl: float) -> TuningConfig:
    """Return a copy of config with different pacing CL."""
    from dataclasses import replace
    return replace(config, pacing_cl=cl)


def _with_stim_amplitude(config: TuningConfig, amp: float) -> TuningConfig:
    """Return a copy of config with different stimulus amplitude."""
    from dataclasses import replace
    return replace(config, stim_amplitude=amp)


def _with_beats(config: TuningConfig, n: int) -> TuningConfig:
    """Return a copy of config with different beat count."""
    from dataclasses import replace
    return replace(config, n_beats=n)
