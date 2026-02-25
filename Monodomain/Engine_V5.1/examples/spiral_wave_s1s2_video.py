"""
S1-S2 Spiral Wave Video Generator

Generates a video of S1-S2 cross-field stimulation protocol.
- S1 applied at t=0 (plane wave from left edge)
- S2 applied at specified interval (default: 459ms)
- Video output at 720p 30fps
"""

import sys
import os
print("DEBUG: Script starting...", flush=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
print("DEBUG: numpy imported", flush=True)
import torch
print(f"DEBUG: torch imported, CUDA available: {torch.cuda.is_available()}", flush=True)
import cv2
print("DEBUG: cv2 imported", flush=True)
from time import perf_counter

from ionic import CellType
print("DEBUG: ionic imported", flush=True)
from tissue import MeshBuilder, compute_D_min
print("DEBUG: tissue imported", flush=True)


class S1S2VideoGenerator:
    """Generate video of S1-S2 spiral wave induction."""

    DEFAULT_APD_MS = 250.0

    def __init__(self, domain_cm=16.0, apd_ms=None):
        self.domain_cm = domain_cm
        self.apd_ms = apd_ms if apd_ms is not None else self.DEFAULT_APD_MS
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("=" * 70)
        print("S1-S2 VIDEO GENERATOR")
        print("=" * 70)

        # Create mesh using MeshBuilder with default anisotropic diffusion
        mesh = (MeshBuilder.create_default(anisotropic=True)
                .set_domain(domain_cm, domain_cm)
                .set_apd(self.apd_ms))

        mesh.print_summary()

        # Create simulation from mesh
        self.sim = mesh.create_simulation(
            celltype=CellType.ENDO,
            device=self.device
        )

        # Get actual grid dimensions from mesh
        cfg = mesh.get_config()
        self.ny, self.nx = cfg.ny, cfg.nx
        self.dx = cfg.dx

        # Time stepping
        dt_stability = self.sim.diffusion.get_stability_limit()
        self.dt = dt_stability * 0.8

        print(f"dt = {self.dt:.5f} ms")
        print(f"Device: {self.device}")
        print()

        # S1-S2 Protocol parameters
        self.s1_width_cm = 0.3  # Width of S1 stimulus region (3mm)
        self.s2_width_cm = 8.0  # 8 cm in x direction
        self.s2_height_cm = 8.0  # 8 cm in y direction

        # Protocol state
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None

        # Setup display for 720p output
        self._setup_display_720p()

    def _setup_display_720p(self):
        """Setup display for 720p video output."""
        # Target 720p: 1280x720
        # We want the image to fit nicely with margins
        target_img_h = 620  # Leave room for margins

        # Calculate scale to fit
        self.scale = max(1, target_img_h // self.ny)
        self.img_w = self.nx * self.scale
        self.img_h = self.ny * self.scale

        # If image is too large, we need to downsample
        if self.img_h > 620:
            self.scale = 1
            self.img_w = self.nx
            self.img_h = self.ny

        # Margins
        self.margin_left = 60
        self.margin_bottom = 50
        self.margin_right = 80
        self.margin_top = 50

        # Canvas dimensions (aim for 720p aspect ratio)
        self.canvas_h = self.img_h + self.margin_top + self.margin_bottom
        self.canvas_w = self.img_w + self.margin_left + self.margin_right

        # Pre-create static canvas
        self.static_canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        self.static_canvas[:] = (30, 30, 30)

        # Colorbar
        cbar_x = self.margin_left + self.img_w + 15
        cbar_w = 20
        cbar_h = self.img_h
        cbar_y = self.margin_top

        gradient = np.linspace(255, 0, cbar_h).astype(np.uint8).reshape(-1, 1)
        gradient_2d = np.tile(gradient, (1, cbar_w))
        colorbar = cv2.applyColorMap(gradient_2d, cv2.COLORMAP_JET)
        self.static_canvas[cbar_y:cbar_y+cbar_h, cbar_x:cbar_x+cbar_w] = colorbar

        cv2.rectangle(self.static_canvas, (cbar_x, cbar_y),
                     (cbar_x + cbar_w, cbar_y + cbar_h), (200, 200, 200), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.static_canvas, '+50', (cbar_x + cbar_w + 5, cbar_y + 15),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '0', (cbar_x + cbar_w + 5, cbar_y + cbar_h // 3 + 5),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '-40', (cbar_x + cbar_w + 5, cbar_y + 2 * cbar_h // 3 + 5),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '-90', (cbar_x + cbar_w + 5, cbar_y + cbar_h),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, 'mV', (cbar_x + 2, cbar_y - 8),
                   font, 0.4, (255, 255, 255), 1)

        # Axis labels - reduce tick density for clarity
        tick_step = max(1, int(self.domain_cm) // 8)  # Max ~8 ticks
        for i in range(0, int(self.domain_cm) + 1, tick_step):
            x_pos = self.margin_left + int(i / self.domain_cm * self.img_w)
            cv2.line(self.static_canvas, (x_pos, self.margin_top + self.img_h),
                    (x_pos, self.margin_top + self.img_h + 5), (200, 200, 200), 1)
            cv2.putText(self.static_canvas, f'{i}', (x_pos - 5, self.margin_top + self.img_h + 20),
                       font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, 'x (cm)',
                   (self.margin_left + self.img_w // 2 - 20, self.canvas_h - 5),
                   font, 0.45, (255, 255, 255), 1)

        for i in range(0, int(self.domain_cm) + 1, tick_step):
            y_pos = self.margin_top + self.img_h - int(i / self.domain_cm * self.img_h)
            cv2.line(self.static_canvas, (self.margin_left - 5, y_pos),
                    (self.margin_left, y_pos), (200, 200, 200), 1)
            cv2.putText(self.static_canvas, f'{i}', (self.margin_left - 25, y_pos + 5),
                       font, 0.4, (255, 255, 255), 1)

        cv2.putText(self.static_canvas, 'y', (5, self.margin_top + self.img_h // 2 - 10),
                   font, 0.45, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '(cm)', (5, self.margin_top + self.img_h // 2 + 10),
                   font, 0.35, (255, 255, 255), 1)

        print(f"Video frame size: {self.canvas_w}x{self.canvas_h}")

    def apply_s1_stimulus(self):
        """Apply S1 stimulus on left edge."""
        V = self.sim.get_voltage()
        s1_cells = max(3, int(self.s1_width_cm / self.dx))
        V[:, :s1_cells] = 20.0
        self.sim.set_voltage(V)
        self.s1_applied = True
        self.s1_time = self.sim.time
        print(f"  S1 applied at t = {self.sim.time:.1f} ms")

    def apply_s2_stimulus(self):
        """Apply S2 stimulus in lower-left quadrant."""
        V = self.sim.get_voltage()
        s2_x_cells = max(3, int(self.s2_width_cm / self.dx))
        s2_y_cells = max(3, int(self.s2_height_cm / self.dx))
        V[:s2_y_cells, :s2_x_cells] = 20.0
        self.sim.set_voltage(V)
        self.s2_applied = True
        elapsed = self.sim.time - self.s1_time if self.s1_time else 0
        print(f"  S2 applied at t = {self.sim.time:.1f} ms (S1-S2 interval: {elapsed:.0f} ms)")

    def voltage_to_image(self, V):
        """Convert voltage to display image."""
        V_norm = np.clip((V + 90) / 140 * 255, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(V_norm, cv2.COLORMAP_JET)
        img = cv2.flip(img, 0)

        if self.scale > 1:
            img = cv2.resize(img, (self.img_w, self.img_h),
                           interpolation=cv2.INTER_NEAREST)

        canvas = self.static_canvas.copy()
        canvas[self.margin_top:self.margin_top+self.img_h,
               self.margin_left:self.margin_left+self.img_w] = img

        return canvas

    def generate_video(self, s1s2_interval_ms=459.0, duration_ms=2000.0,
                       fps=30, output_path=None):
        """
        Generate video of S1-S2 protocol.

        Args:
            s1s2_interval_ms: Time between S1 and S2 in ms
            duration_ms: Total video duration in ms
            fps: Video framerate
            output_path: Output file path (default: same directory as script)
        """
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, f"s1s2_spiral_{int(s1s2_interval_ms)}ms.mp4")

        print(f"\nGenerating video:")
        print(f"  S1-S2 interval: {s1s2_interval_ms} ms")
        print(f"  Duration: {duration_ms} ms")
        print(f"  FPS: {fps}")
        print(f"  Output: {output_path}")
        print()

        # Calculate timing
        ms_per_frame = 1000.0 / fps  # ~33.3 ms per frame at 30fps
        total_frames = int(duration_ms / ms_per_frame)
        steps_per_frame = max(1, int(ms_per_frame / self.dt))

        print(f"  Total frames: {total_frames}")
        print(f"  Steps per frame: {steps_per_frame}")
        print(f"  Actual dt: {self.dt:.4f} ms")
        print()
        import sys
        sys.stdout.flush()

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                             (self.canvas_w, self.canvas_h))

        if not out.isOpened():
            print("ERROR: Could not open video writer")
            return None

        # Reset simulation
        self.sim.reset()
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None

        # Apply S1 at t=0
        self.apply_s1_stimulus()

        start_time = perf_counter()

        for frame_idx in range(total_frames):
            # Check if we need to apply S2
            if self.s1_applied and not self.s2_applied:
                elapsed = self.sim.time - self.s1_time
                if elapsed >= s1s2_interval_ms:
                    self.apply_s2_stimulus()

            # Run simulation steps for this frame
            for _ in range(steps_per_frame):
                self.sim.step(self.dt)

            # Get voltage and create frame
            V = self.sim.get_voltage().cpu().numpy()
            frame = self.voltage_to_image(V)

            # Add title
            title = f"S1-S2 Protocol ({self.domain_cm:.0f}x{self.domain_cm:.0f} cm)"
            cv2.putText(frame, title, (60, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 1)

            # Add time and voltage range (no FPS)
            V_min, V_max = V.min(), V.max()
            status = f"t = {self.sim.time:.1f} ms | V: [{V_min:.0f}, {V_max:.0f}] mV"
            cv2.putText(frame, status, (60, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Write frame
            out.write(frame)

            # Progress update - every frame for debugging
            elapsed_real = perf_counter() - start_time
            progress = (frame_idx + 1) / total_frames * 100
            eta = elapsed_real / (frame_idx + 1) * (total_frames - frame_idx - 1) if frame_idx > 0 else 0
            print(f"  Frame {frame_idx + 1}/{total_frames} ({progress:.1f}%) "
                  f"| t={self.sim.time:.1f}ms | ETA: {eta:.1f}s")
            sys.stdout.flush()

        out.release()

        total_time = perf_counter() - start_time
        print(f"\nVideo generation complete!")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Output: {output_path}")

        return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate S1-S2 spiral wave video')
    parser.add_argument('--domain', type=float, default=16.0,
                       help='Domain size in cm (default: 16.0)')
    parser.add_argument('--apd', type=float, default=250.0,
                       help='Action potential duration in ms (default: 250)')
    parser.add_argument('--s1s2', type=float, default=459.0,
                       help='S1-S2 interval in ms (default: 459)')
    parser.add_argument('--duration', type=float, default=2000.0,
                       help='Video duration in ms (default: 2000)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video framerate (default: 30)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path')
    args = parser.parse_args()

    generator = S1S2VideoGenerator(
        domain_cm=args.domain,
        apd_ms=args.apd
    )

    generator.generate_video(
        s1s2_interval_ms=args.s1s2,
        duration_ms=args.duration,
        fps=args.fps,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
