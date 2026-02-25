#!/usr/bin/env python3
"""
Re-entry Circuit Simulation

Simulates anatomical re-entry in a dual-pathway circuit with:
- Fast pathway (upper): High conduction velocity
- Slow pathway (lower): Low conduction velocity

Geometry:
- 16x16 cm domain
- Ring: inner radius 4 cm, track width 1.5 cm
- Horizontal inlet (left) and outlet (right) at 3 and 9 o'clock

Uses voltage clamping method for stimulus (direct voltage setting).

Controls:
- Slider: Adjust pacing rate (0.5-4 Hz)
- S/SPACE: Manual premature stimulus
- R: Reset simulation
- P: Toggle automatic pacing
- +/-: Adjust simulation speed
- Q/ESC: Quit

Author: Generated with Claude Code
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
from time import perf_counter
from dataclasses import dataclass
from typing import Optional, Tuple

from ionic import TTP06Model, CellType
from ionic.ttp06.parameters import StateIndex
from ionic.ttp06.celltypes.custom.reentry_study import (
    ReentryCellTypes,
    get_reentry_parameters,
)
from utils import Backend


@dataclass
class ReentryConfig:
    """Configuration for re-entry circuit simulation."""
    # Domain
    Lx: float = 16.0          # Domain width (cm)
    Ly: float = 16.0          # Domain height (cm)
    dx: float = 0.02          # Grid spacing (cm) - 200 um

    # Ring geometry (centered)
    ring_inner_radius: float = 2.0    # Inner radius (cm)
    ring_track_width: float = 1.5     # Track width (cm)

    # Inlet/outlet
    inlet_outlet_width: float = 1.5   # Width of inlet/outlet (cm)

    # Diffusion coefficients
    D_fast: float = 0.00154   # Fast pathway (cm^2/ms) - CV ~60 cm/s
    D_slow: float = 0.000385  # Slow pathway (cm^2/ms) - CV ~30 cm/s (4:1 ratio for 2:1 CV)

    # Time stepping
    dt: float = 0.02          # Time step (ms)

    @property
    def nx(self) -> int:
        return int(self.Lx / self.dx)

    @property
    def ny(self) -> int:
        return int(self.Ly / self.dx)

    @property
    def ring_outer_radius(self) -> float:
        return self.ring_inner_radius + self.ring_track_width

    @property
    def center(self) -> Tuple[float, float]:
        return (self.Lx / 2, self.Ly / 2)


class ReentryCircuitGeometry:
    """
    Defines the re-entry circuit geometry with dual pathways.

    Creates masks for:
    - tissue_mask: Boolean mask of conductive tissue
    - fast_path_mask: Boolean mask of fast pathway (upper half of ring)
    - slow_path_mask: Boolean mask of slow pathway (lower half of ring)
    - D_field: Spatially varying diffusion coefficient
    """

    def __init__(self, config: ReentryConfig, device: torch.device):
        self.config = config
        self.device = device
        self.nx = config.nx
        self.ny = config.ny

        # Create coordinate grids (in cm)
        y_coords = torch.linspace(0, config.Ly, config.ny, device=device)
        x_coords = torch.linspace(0, config.Lx, config.nx, device=device)
        self.Y, self.X = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Center coordinates
        cx, cy = config.center
        self.dX = self.X - cx
        self.dY = self.Y - cy
        self.R = torch.sqrt(self.dX**2 + self.dY**2)

        # Build geometry
        self._build_masks()

    def _build_masks(self):
        """Build all geometry masks."""
        cfg = self.config
        cx, cy = cfg.center

        # Ring mask (annulus)
        ring_mask = (self.R >= cfg.ring_inner_radius) & (self.R <= cfg.ring_outer_radius)

        # Inlet/outlet y-range (centered, matching track width)
        inlet_y_min = cy - cfg.inlet_outlet_width / 2
        inlet_y_max = cy + cfg.inlet_outlet_width / 2

        # Carve out junction regions from ring to avoid corner blockage
        # These are rectangular regions at 9 o'clock and 3 o'clock
        inlet_junction = (
            (self.X >= cx - cfg.ring_outer_radius) &
            (self.X <= cx - cfg.ring_inner_radius) &
            (self.Y >= inlet_y_min) &
            (self.Y <= inlet_y_max)
        )
        outlet_junction = (
            (self.X >= cx + cfg.ring_inner_radius) &
            (self.X <= cx + cfg.ring_outer_radius) &
            (self.Y >= inlet_y_min) &
            (self.Y <= inlet_y_max)
        )

        # Remove junction regions from ring (will be replaced by inlet/outlet)
        ring_mask_carved = ring_mask & ~inlet_junction & ~outlet_junction

        # Inlet mask (horizontal strip from left edge through junction)
        inlet_mask = (
            (self.X <= cx - cfg.ring_inner_radius) &
            (self.Y >= inlet_y_min) &
            (self.Y <= inlet_y_max)
        )

        # Outlet mask (horizontal strip from junction to right edge)
        outlet_mask = (
            (self.X >= cx + cfg.ring_inner_radius) &
            (self.Y >= inlet_y_min) &
            (self.Y <= inlet_y_max)
        )

        # Complete tissue mask (carved ring + inlet + outlet)
        self.tissue_mask = ring_mask_carved | inlet_mask | outlet_mask

        # Fast pathway: upper half of ring (y > center), excluding junctions
        self.fast_path_mask = ring_mask_carved & (self.dY > 0)

        # Slow pathway: lower half of ring (y < center), excluding junctions
        self.slow_path_mask = ring_mask_carved & (self.dY < 0)

        # Inlet and outlet masks (include junction regions)
        self.inlet_mask = inlet_mask | inlet_junction
        self.outlet_mask = outlet_mask | outlet_junction

        # Create D field - fast everywhere except slow path
        self.D_field = torch.full((self.ny, self.nx), cfg.D_fast,
                                   device=self.device, dtype=torch.float64)
        self.D_field[self.slow_path_mask] = cfg.D_slow

        # Stimulus region (left end of inlet)
        stim_width = 0.3  # cm
        self.stim_mask = self.inlet_mask & (self.X <= stim_width)


class HeterogeneousFDMDiffusion:
    """
    Finite Difference diffusion with spatially varying D and tissue mask.

    Applies no-flux boundary conditions at tissue boundaries.
    """

    def __init__(self, geometry: ReentryCircuitGeometry, device: torch.device):
        self.geometry = geometry
        self.device = device
        self.ny = geometry.ny
        self.nx = geometry.nx
        self.dx = geometry.config.dx

        # Get tissue mask and D field
        self.tissue_mask = geometry.tissue_mask
        self.D_field = geometry.D_field

        # Precompute alpha field: D / dx^2
        self.alpha = self.D_field / (self.dx ** 2)

        # Stability limit (use max D for conservative estimate)
        D_max = self.D_field[self.tissue_mask].max().item()
        self.dt_max = self.dx ** 2 / (4 * D_max)

    def step(self, V: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Advance diffusion by one time step with heterogeneous D.

        Uses explicit Euler with 5-point stencil.
        No-flux BC at tissue boundaries.
        """
        # Pad with replicate (Neumann BC)
        V_pad = torch.nn.functional.pad(
            V.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1),
            mode='replicate'
        ).squeeze()

        # Also pad tissue mask for boundary detection
        mask_pad = torch.nn.functional.pad(
            self.tissue_mask.unsqueeze(0).unsqueeze(0).float(),
            (1, 1, 1, 1),
            mode='constant',
            value=0
        ).squeeze().bool()

        # Apply no-flux at tissue boundaries by copying V values
        # where neighbors are outside tissue
        V_right = torch.where(mask_pad[1:-1, 2:], V_pad[1:-1, 2:], V)
        V_left = torch.where(mask_pad[1:-1, :-2], V_pad[1:-1, :-2], V)
        V_up = torch.where(mask_pad[2:, 1:-1], V_pad[2:, 1:-1], V)
        V_down = torch.where(mask_pad[:-2, 1:-1], V_pad[:-2, 1:-1], V)

        # 5-point stencil Laplacian
        laplacian = V_right + V_left + V_up + V_down - 4 * V

        # Apply diffusion with spatially varying D
        dV = self.alpha * laplacian * dt

        # Only update within tissue
        V_new = V.clone()
        V_new[self.tissue_mask] = V[self.tissue_mask] + dV[self.tissue_mask]

        return V_new


class HeterogeneousTTP06:
    """
    TTP06 model with spatially varying cell types for re-entry study.
    Uses different ionic parameters for:
    - Fast pathway: Long APD/ERP (reduced GKs)
    - Slow pathway: Short APD/ERP (enhanced GKs)
    - Inlet/outlet: Standard EPI parameters
    """

    def __init__(self, geometry: ReentryCircuitGeometry, device: torch.device, use_lut: bool = True):
        self.geometry = geometry
        self.device = device
        self.ny = geometry.ny
        self.nx = geometry.nx
        self.n_states = StateIndex.N_STATES

        # Create models with different cell type configurations
        fast_config = get_reentry_parameters(ReentryCellTypes.FAST_PATH)
        slow_config = get_reentry_parameters(ReentryCellTypes.SLOW_PATH)
        io_config = get_reentry_parameters(ReentryCellTypes.INLET_OUTLET)

        self.model_fast = TTP06Model.from_config(fast_config, device=device, use_lut=use_lut)
        self.model_slow = TTP06Model.from_config(slow_config, device=device, use_lut=use_lut)
        self.model_io = TTP06Model.from_config(io_config, device=device, use_lut=use_lut)

        # Store masks
        self.tissue_mask = geometry.tissue_mask
        self.fast_mask = geometry.fast_path_mask
        self.slow_mask = geometry.slow_path_mask
        self.io_mask = geometry.inlet_mask | geometry.outlet_mask

        print(f"  Cell types:")
        print(f"    FAST_PATH: GKs={fast_config.GKs} (long APD)")
        print(f"    SLOW_PATH: GKs={slow_config.GKs} (short APD)")
        print(f"    INLET_OUTLET: GKs={io_config.GKs} (normal APD)")

    def get_initial_state(self) -> torch.Tensor:
        """Get initial state for all nodes."""
        initial = self.model_io.get_initial_state(1)
        states = initial.unsqueeze(0).unsqueeze(0).expand(self.ny, self.nx, -1).clone()
        return states

    def step(self, states: torch.Tensor, dt: float,
             I_stim: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Advance ionic model by one time step.

        Applies region-specific parameters from custom cell types.

        Parameters
        ----------
        states : torch.Tensor
            State tensor, shape (n_cells, n_states) - flattened tissue nodes
        dt : float
            Time step (ms)
        I_stim : torch.Tensor, optional
            Stimulus current (not used - stimulus via voltage clamping)

        Returns
        -------
        torch.Tensor
            Updated state tensor, same shape as input
        """
        # States come in as (n_tissue_nodes, n_states) from the simulation step
        # We need to map them back to the original grid positions

        n_cells, n_states = states.shape

        # Get flattened masks for tissue nodes only
        tissue_flat = self.tissue_mask.view(-1)
        fast_flat = self.fast_mask.view(-1)
        slow_flat = self.slow_mask.view(-1)
        io_flat = self.io_mask.view(-1)

        # Create index mapping: which tissue index corresponds to which region
        tissue_indices = torch.where(tissue_flat)[0]

        # For each region, find which tissue indices belong to it
        fast_in_tissue = fast_flat[tissue_flat]
        slow_in_tissue = slow_flat[tissue_flat]
        io_in_tissue = io_flat[tissue_flat]

        # Process fast pathway cells
        if fast_in_tissue.any():
            fast_states = states[fast_in_tissue]
            fast_new = self.model_fast.step(fast_states, dt, None)
            states[fast_in_tissue] = fast_new

        # Process slow pathway cells
        if slow_in_tissue.any():
            slow_states = states[slow_in_tissue]
            slow_new = self.model_slow.step(slow_states, dt, None)
            states[slow_in_tissue] = slow_new

        # Process inlet/outlet cells
        if io_in_tissue.any():
            io_states = states[io_in_tissue]
            io_new = self.model_io.step(io_states, dt, None)
            states[io_in_tissue] = io_new

        return states


class ReentryCircuitSimulation:
    """
    Complete re-entry circuit simulation with interactive controls.

    Uses voltage clamping method for stimulus and heterogeneous TTP06 model
    with different APD/ERP in fast vs slow pathways.
    """

    def __init__(self, config: Optional[ReentryConfig] = None):
        self.config = config or ReentryConfig()

        print("=" * 70)
        print("RE-ENTRY CIRCUIT SIMULATION")
        print("Dual Pathway with Heterogeneous CV and APD/ERP")
        print("=" * 70)

        # Setup device
        self.backend = Backend(device='auto', verbose=True)
        self.device = self.backend.device

        # Build geometry
        print("\nBuilding geometry...")
        self.geometry = ReentryCircuitGeometry(self.config, self.device)

        n_tissue = self.geometry.tissue_mask.sum().item()
        n_fast = self.geometry.fast_path_mask.sum().item()
        n_slow = self.geometry.slow_path_mask.sum().item()

        print(f"  Domain: {self.config.Lx} x {self.config.Ly} cm")
        print(f"  Grid: {self.config.nx} x {self.config.ny} ({self.config.nx * self.config.ny:,} total)")
        print(f"  Tissue nodes: {n_tissue:,}")
        print(f"  Fast pathway: {n_fast:,} nodes (D={self.config.D_fast:.5f}, long ERP)")
        print(f"  Slow pathway: {n_slow:,} nodes (D={self.config.D_slow:.5f}, short ERP)")

        # Create heterogeneous ionic model with custom cell types
        print("\nInitializing heterogeneous ionic model...")
        self.ionic = HeterogeneousTTP06(self.geometry, self.device, use_lut=True)
        self.n_states = self.ionic.n_states

        # Create diffusion operator
        print("Initializing diffusion operator...")
        self.diffusion = HeterogeneousFDMDiffusion(self.geometry, self.device)
        print(f"  dt_max: {self.diffusion.dt_max:.4f} ms")

        # Initialize states for tissue only
        self._init_states()
        self.time = 0.0

        # Time stepping
        self.dt = min(self.config.dt, self.diffusion.dt_max * 0.8)
        self.steps_per_frame = max(1, int(0.5 / self.dt))  # ~0.5 ms per frame
        self.speed_multiplier = 1.0

        # Pacing parameters
        self.pacing_rate_hz = 1.0  # Default 1 Hz
        self.pacing_enabled = False
        self.last_pace_time = -1000.0  # ms

        # Protocol state
        self.stimulus_applied = False

        # Display
        self._setup_display()

        # FPS tracking
        self.frame_times = []
        self.max_frame_times = 30

        print(f"\nTime stepping:")
        print(f"  dt = {self.dt:.4f} ms")
        print(f"  steps/frame = {self.steps_per_frame}")
        print()
        print("Controls:")
        print("  Slider    - Adjust pacing rate (0.5-4 Hz)")
        print("  P         - Toggle automatic pacing")
        print("  S/SPACE   - Manual premature stimulus")
        print("  R         - Reset simulation")
        print("  +/-       - Adjust speed")
        print("  Q/ESC     - Quit")
        print()

    def _init_states(self):
        """Initialize state array for all nodes."""
        # HeterogeneousTTP06.get_initial_state() returns full (ny, nx, n_states) array
        self.states = self.ionic.get_initial_state()

    def _setup_display(self):
        """Setup display parameters."""
        target_size = 600
        self.scale = max(1, target_size // max(self.config.ny, self.config.nx))
        self.img_w = self.config.nx * self.scale
        self.img_h = self.config.ny * self.scale

        # Margins
        self.margin_left = 60
        self.margin_bottom = 80  # Extra space for slider
        self.margin_right = 80
        self.margin_top = 50

        # Canvas
        self.canvas_h = self.img_h + self.margin_top + self.margin_bottom
        self.canvas_w = self.img_w + self.margin_left + self.margin_right

        # Static canvas
        self.static_canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        self.static_canvas[:] = (30, 30, 30)

        # Colorbar
        cbar_x = self.margin_left + self.img_w + 15
        cbar_w = 20
        cbar_h = self.img_h
        cbar_y = self.margin_top

        gradient = np.linspace(255, 0, cbar_h).astype(np.uint8).reshape(-1, 1)
        colorbar = cv2.applyColorMap(np.tile(gradient, (1, cbar_w)), cv2.COLORMAP_JET)
        self.static_canvas[cbar_y:cbar_y+cbar_h, cbar_x:cbar_x+cbar_w] = colorbar

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.static_canvas, '+40', (cbar_x + cbar_w + 5, cbar_y + 15),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '-90', (cbar_x + cbar_w + 5, cbar_y + cbar_h),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, 'mV', (cbar_x + 2, cbar_y - 8),
                   font, 0.4, (255, 255, 255), 1)

        # Create tissue boundary overlay
        self._create_boundary_overlay()

    def _create_boundary_overlay(self):
        """Create overlay showing tissue boundaries."""
        self.boundary_overlay = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        cfg = self.config
        cx_px = int(cfg.center[0] / cfg.dx * self.scale)
        cy_px = int((cfg.Ly - cfg.center[1]) / cfg.dx * self.scale)  # Flip Y

        # Draw ring boundaries
        r_inner_px = int(cfg.ring_inner_radius / cfg.dx * self.scale)
        r_outer_px = int(cfg.ring_outer_radius / cfg.dx * self.scale)

        cv2.circle(self.boundary_overlay, (cx_px, cy_px), r_inner_px, (100, 100, 100), 1)
        cv2.circle(self.boundary_overlay, (cx_px, cy_px), r_outer_px, (100, 100, 100), 1)

        # Draw pathway labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Fast path label (upper)
        fast_y = cy_px - int((cfg.ring_inner_radius + cfg.ring_track_width/2) / cfg.dx * self.scale)
        cv2.putText(self.boundary_overlay, "FAST", (cx_px - 25, fast_y),
                   font, 0.5, (0, 200, 255), 1)

        # Slow path label (lower)
        slow_y = cy_px + int((cfg.ring_inner_radius + cfg.ring_track_width/2) / cfg.dx * self.scale)
        cv2.putText(self.boundary_overlay, "SLOW", (cx_px - 25, slow_y + 15),
                   font, 0.5, (255, 100, 100), 1)

    def get_voltage(self) -> torch.Tensor:
        """Get voltage field."""
        return self.states[:, :, StateIndex.V]

    def set_voltage(self, V: torch.Tensor):
        """Set voltage field."""
        self.states[:, :, StateIndex.V] = V

    def apply_stimulus(self):
        """Apply stimulus at inlet using voltage clamping."""
        V = self.get_voltage()
        V[self.geometry.stim_mask] = 20.0  # Depolarize to +20 mV
        self.set_voltage(V)
        self.stimulus_applied = True
        self.last_pace_time = self.time
        print(f"  Stimulus applied at t = {self.time:.1f} ms")

    def step(self):
        """
        Advance simulation by one time step.

        Uses operator splitting:
        1. Ionic step (only for tissue nodes)
        2. Diffusion step (with heterogeneous D)
        """
        # Check if automatic pacing is due
        if self.pacing_enabled:
            cycle_length = 1000.0 / self.pacing_rate_hz  # ms
            if self.time - self.last_pace_time >= cycle_length:
                self.apply_stimulus()

        ny, nx = self.config.ny, self.config.nx
        tissue_mask = self.geometry.tissue_mask

        # === Ionic step (only for tissue nodes) ===
        # Flatten states for ionic model
        states_flat = self.states.view(-1, self.n_states)
        tissue_flat = tissue_mask.view(-1)

        # Only process tissue nodes
        tissue_states = states_flat[tissue_flat]
        tissue_states_new = self.ionic.step(tissue_states, self.dt, None)
        states_flat[tissue_flat] = tissue_states_new

        # Reshape back
        self.states = states_flat.view(ny, nx, self.n_states)

        # === Diffusion step ===
        V = self.get_voltage()
        V_new = self.diffusion.step(V, self.dt)
        self.set_voltage(V_new)

        self.time += self.dt

    def reset(self):
        """Reset simulation to initial conditions."""
        self._init_states()
        self.time = 0.0
        self.stimulus_applied = False
        self.last_pace_time = -1000.0
        print("Simulation reset")

    def voltage_to_image(self, V: np.ndarray) -> np.ndarray:
        """Convert voltage to display image."""
        # Normalize voltage
        V_norm = np.clip((V + 90) / 130 * 255, 0, 255).astype(np.uint8)

        # Apply colormap
        img = cv2.applyColorMap(V_norm, cv2.COLORMAP_JET)

        # Mask non-tissue regions (show as dark)
        tissue_np = self.geometry.tissue_mask.cpu().numpy()
        img[~tissue_np] = (30, 30, 30)

        # Flip Y axis (image coordinates)
        img = cv2.flip(img, 0)

        # Scale up
        if self.scale > 1:
            img = cv2.resize(img, (self.img_w, self.img_h),
                           interpolation=cv2.INTER_NEAREST)

        # Add boundary overlay
        mask = self.boundary_overlay > 0
        img[mask] = self.boundary_overlay[mask]

        # Place on canvas
        canvas = self.static_canvas.copy()
        canvas[self.margin_top:self.margin_top+self.img_h,
               self.margin_left:self.margin_left+self.img_w] = img

        return canvas

    def draw_slider(self, img: np.ndarray):
        """Draw pacing rate slider on image."""
        # Slider position
        slider_y = self.margin_top + self.img_h + 30
        slider_x_start = self.margin_left + 80
        slider_x_end = self.margin_left + self.img_w - 20
        slider_width = slider_x_end - slider_x_start

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Label
        cv2.putText(img, "Rate:", (self.margin_left, slider_y + 5),
                   font, 0.5, (255, 255, 255), 1)

        # Track
        cv2.line(img, (slider_x_start, slider_y), (slider_x_end, slider_y),
                (100, 100, 100), 2)

        # Slider position (0.5-4 Hz mapped to slider range)
        rate_norm = (self.pacing_rate_hz - 0.5) / 3.5
        slider_pos = int(slider_x_start + rate_norm * slider_width)

        # Knob
        knob_color = (0, 255, 0) if self.pacing_enabled else (100, 100, 100)
        cv2.circle(img, (slider_pos, slider_y), 8, knob_color, -1)
        cv2.circle(img, (slider_pos, slider_y), 8, (255, 255, 255), 1)

        # Value label
        cl_ms = 1000.0 / self.pacing_rate_hz
        rate_text = f"{self.pacing_rate_hz:.1f} Hz ({cl_ms:.0f} ms)"
        cv2.putText(img, rate_text, (slider_x_end + 10, slider_y + 5),
                   font, 0.4, (255, 255, 255), 1)

        # Min/max labels
        cv2.putText(img, "0.5", (slider_x_start - 15, slider_y + 20),
                   font, 0.3, (150, 150, 150), 1)
        cv2.putText(img, "4.0", (slider_x_end - 10, slider_y + 20),
                   font, 0.3, (150, 150, 150), 1)

        # Store slider bounds for mouse interaction
        self.slider_bounds = (slider_x_start, slider_x_end, slider_y)

    def handle_mouse(self, event, x, y, flags, param):
        """Handle mouse events for slider."""
        if not hasattr(self, 'slider_bounds'):
            return

        slider_x_start, slider_x_end, slider_y = self.slider_bounds
        slider_width = slider_x_end - slider_x_start

        # Check if click/drag is on slider
        if abs(y - slider_y) < 15 and slider_x_start <= x <= slider_x_end:
            if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
                # Update rate
                rate_norm = (x - slider_x_start) / slider_width
                rate_norm = max(0, min(1, rate_norm))
                self.pacing_rate_hz = 0.5 + rate_norm * 3.5

    def run(self):
        """Main simulation loop."""
        window_name = 'Re-entry Circuit'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 850)
        cv2.setMouseCallback(window_name, self.handle_mouse)

        print("Starting simulation...")
        print("Press 'P' to start automatic pacing, or 'S' for manual stimulus")

        while True:
            frame_start = perf_counter()

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s') or key == ord(' '):
                print(f"\n--- Manual stimulus ---")
                self.apply_stimulus()
            elif key == ord('p'):
                self.pacing_enabled = not self.pacing_enabled
                status = "ON" if self.pacing_enabled else "OFF"
                print(f"Automatic pacing: {status}")
                if self.pacing_enabled:
                    self.last_pace_time = self.time - 1000.0  # Trigger immediate pace
            elif key == ord('r'):
                self.reset()
            elif key == ord('+') or key == ord('='):
                self.speed_multiplier = min(4.0, self.speed_multiplier * 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")
            elif key == ord('-'):
                self.speed_multiplier = max(0.25, self.speed_multiplier / 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")

            # Simulation steps
            actual_steps = max(1, int(self.steps_per_frame * self.speed_multiplier))
            for _ in range(actual_steps):
                self.step()

            # Get voltage and create image
            V = self.get_voltage().cpu().numpy()
            img = self.voltage_to_image(V)

            # Draw slider
            self.draw_slider(img)

            # FPS
            frame_time = perf_counter() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_times:
                self.frame_times.pop(0)
            fps = len(self.frame_times) / sum(self.frame_times)

            # Title
            title = f"Re-entry Circuit ({self.config.Lx:.0f}x{self.config.Ly:.0f} cm)"
            cv2.putText(img, title, (60, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 1)

            # Status
            V_tissue = V[self.geometry.tissue_mask.cpu().numpy()]
            V_min, V_max = V_tissue.min(), V_tissue.max()
            status = f"t={self.time:.1f}ms | {fps:.1f}FPS | V:[{V_min:.0f},{V_max:.0f}]mV"
            cv2.putText(img, status, (60, img.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (200, 200, 200), 1)

            # Pacing status
            if self.pacing_enabled:
                pacing_status = f"PACING @ {self.pacing_rate_hz:.1f} Hz"
                cv2.putText(img, pacing_status, (img.shape[1] - 200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(img, "Press P to pace", (img.shape[1] - 180, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

            cv2.imshow(window_name, img)

        cv2.destroyAllWindows()
        print("Simulation ended.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Re-entry circuit simulation')
    parser.add_argument('--domain', type=float, default=16.0,
                       help='Domain size in cm (default: 16.0)')
    parser.add_argument('--dx', type=float, default=0.02,
                       help='Grid spacing in cm (default: 0.02)')
    parser.add_argument('--inner-radius', type=float, default=4.0,
                       help='Ring inner radius in cm (default: 4.0)')
    parser.add_argument('--track-width', type=float, default=1.5,
                       help='Track width in cm (default: 1.5)')
    args = parser.parse_args()

    config = ReentryConfig(
        Lx=args.domain,
        Ly=args.domain,
        dx=args.dx,
        ring_inner_radius=args.inner_radius,
        ring_track_width=args.track_width
    )

    sim = ReentryCircuitSimulation(config)
    sim.run()


if __name__ == '__main__':
    main()
