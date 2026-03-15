#!/usr/bin/env python3
"""
Debug Script: Diffusion vs Ionic Current Balance Analysis

Verifies unit scaling consistency in the monodomain equation:
    dV/dt = D·∇²V - Iion/Cm

Key measurements:
1. Diffusion term magnitude at wavefront
2. Total dV/dt at wavefront
3. Ionic contribution (inferred as dV_total - dV_diff)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from tissue import MonodomainSimulation
from ionic import CellType


def analyze_wavefront_currents(
    grid_size: int = 200,
    domain_cm: float = 4.0,
    verbose: bool = True
):
    """
    Measure diffusion and ionic contributions at wavefront.
    """
    dx = domain_cm / grid_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("WAVEFRONT CURRENT BALANCE ANALYSIS")
    print("=" * 70)
    print()
    print(f"Grid: {grid_size}x{grid_size}, dx = {dx*10:.2f} mm")
    print()

    # Use safe D values to avoid propagation failure
    D_L = 0.00151  # For CV ~ 0.06 cm/ms
    D_T = 0.00151  # Isotropic to avoid transverse issues

    sim = MonodomainSimulation(
        ny=grid_size, nx=grid_size,
        dx=dx, dy=dx,
        D_L=D_L, D_T=D_T,  # Direct D specification
        celltype=CellType.ENDO,
        device=device,
        params_override=None
    )

    dt = sim.diffusion.get_stability_limit() * 0.8

    print(f"D_L = {D_L:.6f} cm²/ms")
    print(f"D_T = {D_T:.6f} cm²/ms")
    print(f"dt = {dt:.5f} ms")
    print()

    # Plane wave stimulus from left edge
    V = sim.get_voltage()
    V[:, :5] = 20.0
    sim.set_voltage(V)

    print("Measuring wavefront dynamics...")
    print("-" * 70)
    print(f"{'t(ms)':<8} | {'x_front':<10} | {'V_front':<10} | {'dV_diff':<12} | {'dV_total':<12} | {'dV_ionic':<12}")
    print("-" * 70)

    cy = grid_size // 2
    V_prev = sim.get_voltage().clone()

    for target_t in [2, 4, 6, 8, 10, 15, 20, 25, 30]:
        while sim.time < target_t:
            V_prev = sim.get_voltage().clone()
            sim.step(dt)

        V = sim.get_voltage()
        V_np = V.cpu().numpy()

        # Find wavefront position
        wavefront_x = 0
        for i in range(grid_size):
            if V_np[cy, i] > -40:
                wavefront_x = i
            else:
                break

        if wavefront_x < 3 or wavefront_x >= grid_size - 3:
            continue

        V_wf = V_np[cy, wavefront_x]
        x_cm = wavefront_x * dx

        # Compute diffusion term at wavefront
        diff_term = sim.diffusion.apply(V)
        dV_diff = diff_term[cy, wavefront_x].item()

        # Compute total voltage change
        dV_total = (V[cy, wavefront_x] - V_prev[cy, wavefront_x]).item() / dt

        # Ionic contribution is the difference
        dV_ionic = dV_total - dV_diff

        print(f"{sim.time:<8.1f} | {x_cm:<10.2f} | {V_wf:<+10.1f} | {dV_diff:<+12.2f} | {dV_total:<+12.2f} | {dV_ionic:<+12.2f}")

    print()


def analyze_unit_scaling():
    """
    Verify unit scaling consistency in the monodomain equation.
    """
    print("=" * 70)
    print("UNIT SCALING VERIFICATION")
    print("=" * 70)
    print()
    print("Monodomain equation (per-capacitance form):")
    print("    dV/dt = D·∇²V - Iion/Cm")
    print()
    print("Unit analysis:")
    print("  V: [mV]")
    print("  D: [cm²/ms]")
    print("  ∇²V = d²V/dx²: [mV/cm²]")
    print("  D·∇²V: [cm²/ms]·[mV/cm²] = [mV/ms] ✓")
    print()
    print("  Iion: [µA/µF] (per membrane capacitance)")
    print("  Cm: [µF/cm²] = 1.0 (normalized)")
    print("  Iion/Cm: [µA/µF] / 1.0 = [µA/µF] ≡ [mV/ms] ✓")
    print()
    print("CONCLUSION: Units are consistent")
    print()

    # Numerical example
    print("Numerical example at wavefront:")
    print("-" * 50)

    D_L = 0.00151  # cm²/ms
    dx = 0.02  # cm

    # Typical wavefront voltages
    V_left = 30  # mV (depolarized)
    V_center = -50  # mV (wavefront)
    V_right = -87  # mV (resting)

    d2V_dx2 = (V_left - 2*V_center + V_right) / (dx**2)
    dV_diff = D_L * d2V_dx2

    print(f"  V profile: {V_left} → {V_center} → {V_right} mV")
    print(f"  d²V/dx² = {d2V_dx2:.0f} mV/cm²")
    print(f"  D·d²V/dx² = {dV_diff:.1f} mV/ms")
    print()
    print("  This diffusion term provides the 'source' current")
    print("  that drives excitation of the resting tissue ahead.")
    print()


def compare_source_sink_geometry():
    """
    Compare source-sink balance for different stimulus geometries.
    """
    print("=" * 70)
    print("SOURCE-SINK GEOMETRY COMPARISON")
    print("=" * 70)
    print()

    grid_size = 150
    domain_cm = 3.0
    dx = domain_cm / grid_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    D_L = 0.00151
    D_T = 0.00151  # Isotropic

    for stim_type, stim_name in [('plane', 'Plane Wave'), ('point', 'Point Stimulus')]:
        print(f"\n{stim_name}:")
        print("-" * 40)

        sim = MonodomainSimulation(
            ny=grid_size, nx=grid_size,
            dx=dx, dy=dx,
            D_L=D_L, D_T=D_T,
            celltype=CellType.ENDO,
            device=device,
            params_override=None
        )

        dt = sim.diffusion.get_stability_limit() * 0.8

        V = sim.get_voltage()

        if stim_type == 'plane':
            V[:, :5] = 20.0
        else:
            cy, cx = grid_size // 2, grid_size // 2
            r = 5
            for dy in range(-r, r+1):
                for ddx in range(-r, r+1):
                    if dy*dy + ddx*ddx <= r*r:
                        V[cy+dy, cx+ddx] = 20.0

        sim.set_voltage(V)

        print(f"{'t(ms)':<8} | {'X front':<10} | {'V_max':<10}")
        print("-" * 35)

        cy = grid_size // 2

        for target_t in [5, 10, 20, 30, 50]:
            while sim.time < target_t:
                sim.step(dt)

            V_np = sim.get_voltage().cpu().numpy()
            V_max = V_np.max()

            if stim_type == 'plane':
                x_front = 0
                for i in range(grid_size):
                    if V_np[cy, i] > -40:
                        x_front = i
                front_cm = x_front * dx
            else:
                cx = grid_size // 2
                x_front = 0
                for i in range(cx, grid_size):
                    if V_np[cy, i] > -40:
                        x_front = i - cx
                front_cm = x_front * dx

            print(f"{sim.time:<8.0f} | {front_cm:<10.2f} | {V_max:<+10.1f}")

    print()
    print("Both should propagate with isotropic D.")
    print("Source-sink issues arise with anisotropic D (D_T << D_L).")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze diffusion vs ionic balance")
    parser.add_argument('--wavefront', action='store_true', help="Analyze wavefront currents")
    parser.add_argument('--units', action='store_true', help="Verify unit scaling")
    parser.add_argument('--source-sink', action='store_true', help="Compare stimulus geometries")
    parser.add_argument('--all', action='store_true', help="Run all analyses")

    args = parser.parse_args()

    if args.all or not any([args.wavefront, args.units, args.source_sink]):
        args.units = True
        args.wavefront = True
        args.source_sink = True

    if args.units:
        analyze_unit_scaling()

    if args.wavefront:
        analyze_wavefront_currents()

    if args.source_sink:
        compare_source_sink_geometry()
