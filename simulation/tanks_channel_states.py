# -*- coding: utf-8 -*-
"""Storage-tank simulation with configurable per-channel pump laws.

Extension of Zimmerman's original `storagetanks.py`: channels are now first-class
objects with a `state` attribute that controls the flux law across that link.

Motivation: the original model's boundary speedup relies on a *source-limited*,
fixed-rate pump (flux per open channel ∝ f(u_s) independent of u_d). Under that
rule, an edge tank with fewer open channels keeps pumping its remaining neighbours
at the full rate for longer — the boundary effect is purely geometric.

This module lets us swap in alternate flux laws to test whether the effect
survives when the physically dubious "fixed rate" assumption is relaxed:

  'constant'               John's original rule. max_pump per open channel.
  'gradient'               Rate proportional to (u_s - u_d). Threshold-gated.
                           Self-limiting: as the downstream fills, drive falls.

Run:
    python tanks_channel_states.py
produces outputs/constant_mode.mp4, outputs/gradient_mode.mp4,
outputs/front_comparison.png, and outputs/summary.txt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core objects
# ---------------------------------------------------------------------------

@dataclass
class Tank:
    id: int
    max_volume: float
    threshold: float
    current_volume: float = 0.0
    virtual_volume: float = 0.0
    is_inlet: bool = False
    is_outlet: bool = False
    outgoing_channels: list = field(default_factory=list)


@dataclass
class Channel:
    """Directed transport link with a state-dependent flux law."""

    source: Tank
    destination: Tank
    state: str = "constant"
    max_pump: float = 5.0
    gradient_k: float = 0.05

    def pump_amount(self) -> float:
        u_s = self.source.current_volume
        u_d = self.destination.current_volume
        if u_s <= u_d:
            return 0.0
        if u_s <= self.source.threshold:
            return 0.0

        if self.state == "constant":
            return self._constant_rule(u_s, u_d)
        if self.state == "gradient":
            return self._gradient_rule(u_s, u_d)
        raise ValueError(f"unknown channel state: {self.state!r}")

    def _constant_rule(self, u_s: float, u_d: float) -> float:
        pumpfactor = np.sqrt(self.source.max_volume - self.source.threshold)
        amount = (np.sqrt(u_s - self.source.threshold) / pumpfactor) * self.max_pump
        if amount > self.max_pump:
            amount = self.max_pump
        if amount > abs(u_s - u_d):
            midpoint = 0.5 * (u_s + u_d)
            amount = (midpoint - u_d) / 2.0
        return max(amount, 0.0)

    def _gradient_rule(self, u_s: float, u_d: float) -> float:
        return self.gradient_k * (u_s - u_d)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_grid(
    Nx: int,
    Ny: int,
    channel_state: str,
    threshold: float = 45.0,
    max_volume: float = 100.0,
    max_pump: float = 5.0,
    gradient_k: float = 0.05,
    inlet_xs: Sequence[int] = (0,),
    outlet_xs: Sequence[int] | None = None,
    inlet_cells: Sequence[tuple[int, int]] | None = None,
    outlet_cells: Sequence[tuple[int, int]] | None = None,
) -> tuple[List[Tank], List[Channel]]:
    if inlet_cells is None:
        inlet_cells = [(x, y) for x in inlet_xs for y in range(Ny)]
    if outlet_cells is None:
        if outlet_xs is None:
            outlet_xs = (Nx - 1,)
        outlet_cells = [(x, y) for x in outlet_xs for y in range(Ny)]
    inlet_set = set(inlet_cells)
    outlet_set = set(outlet_cells)

    tanks: dict[tuple[int, int], Tank] = {}
    for x in range(Nx):
        for y in range(Ny):
            tid = x * Ny + y
            t = Tank(id=tid, max_volume=max_volume, threshold=threshold)
            if (x, y) in inlet_set:
                t.is_inlet = True
            if (x, y) in outlet_set:
                t.is_outlet = True
            tanks[(x, y)] = t

    channels: List[Channel] = []
    for (x, y), tank in tanks.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < Nx and 0 <= ny < Ny:
                    ch = Channel(
                        source=tank,
                        destination=tanks[(nx, ny)],
                        state=channel_state,
                        max_pump=max_pump,
                        gradient_k=gradient_k,
                    )
                    tank.outgoing_channels.append(ch)
                    channels.append(ch)

    tank_list = [tanks[(x, y)] for x in range(Nx) for y in range(Ny)]
    return tank_list, channels


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_sim(
    tanks: List[Tank],
    channels: List[Channel],
    Nx: int,
    Ny: int,
    steps: int = 600,
    record_every: int = 5,
) -> np.ndarray:
    """Run a simulation and return a history array of shape (T_rec, Ny, Nx)."""
    T_rec = steps // record_every
    history = np.zeros((T_rec, Ny, Nx), dtype=np.float32)

    rec_idx = 0
    for step in range(steps):
        if step % 100 == 0:
            print(f"  step {step}/{steps}")

        for t in tanks:
            t.virtual_volume = 0.0

        for ch in channels:
            amt = ch.pump_amount()
            if amt > 0.0:
                ch.source.virtual_volume -= amt
                ch.destination.virtual_volume += amt

        for t in tanks:
            if t.is_inlet:
                t.current_volume = t.max_volume
            elif t.is_outlet:
                t.current_volume = 0.0
            else:
                t.current_volume += t.virtual_volume
                if t.current_volume > t.max_volume:
                    t.current_volume = t.max_volume
                if t.current_volume < 0.0:
                    t.current_volume = 0.0
            t.virtual_volume = 0.0

        if step % record_every == 0 and rec_idx < T_rec:
            for t in tanks:
                x = t.id // Ny
                y = t.id % Ny
                history[rec_idx, y, x] = t.current_volume
            rec_idx += 1

    return history


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def history_to_isochrone(
    history: np.ndarray, threshold: float, record_every: int = 1
) -> np.ndarray:
    """Return iso[y, x] = step at which V[y, x] first crossed threshold, -1 if unreached."""
    T, Ny, Nx = history.shape
    iso = np.full((Ny, Nx), -1, dtype=np.int32)
    for t in range(T):
        newly = (history[t] > threshold) & (iso < 0)
        iso = np.where(newly, t * record_every, iso)
    return iso


def measure_front(history: np.ndarray, threshold: float) -> np.ndarray:
    """Largest x such that V[t, y, x] > threshold for each (t, y)."""
    T, Ny, Nx = history.shape
    front = np.full((T, Ny), -1, dtype=np.int32)
    for t in range(T):
        above = history[t] > threshold
        for y in range(Ny):
            idx = np.where(above[y])[0]
            if len(idx) > 0:
                front[t, y] = idx.max()
    return front


def render_video(history: np.ndarray, path: str, fps: int = 10, scale: int = 5) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    T, Ny, Nx = history.shape
    W, H = Nx * scale, Ny * scale
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for t in range(T):
        norm = np.clip(history[t] / 100.0 * 255, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            img, f"frame {t}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(img)
    writer.release()
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main() -> None:
    import matplotlib.pyplot as plt

    Nx, Ny = 80, 50
    steps = 600
    record_every = 5
    threshold = 45.0
    gradient_k = 0.05

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    histories: dict[str, np.ndarray] = {}
    for state in ("constant", "gradient"):
        print(f"=== {state} mode ===")
        tanks, channels = build_grid(
            Nx, Ny,
            channel_state=state,
            threshold=threshold,
            gradient_k=gradient_k,
            inlet_xs=(0,),
            outlet_xs=(Nx - 1,),
        )
        history = run_sim(tanks, channels, Nx, Ny, steps=steps, record_every=record_every)
        histories[state] = history
        render_video(history, str(out_dir / f"{state}_mode.mp4"))

    # Quantitative comparison: front position per row, edge vs mid
    summary_lines: list[str] = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    y_edge, y_mid, y_other = 0, Ny // 2, Ny - 1
    for ax, state in zip(axes, ("constant", "gradient")):
        front = measure_front(histories[state], threshold)
        t_axis = np.arange(front.shape[0]) * record_every
        ax.plot(t_axis, front[:, y_edge], label=f"y={y_edge} (edge)", lw=2)
        ax.plot(t_axis, front[:, y_mid], label=f"y={y_mid} (mid)", lw=2)
        ax.plot(t_axis, front[:, y_other], label=f"y={y_other} (edge)",
                lw=2, ls="--")
        ax.set_xlabel("step")
        ax.set_title(f"{state} mode")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        # Late-time lead metric: average of edges minus middle over last 5 frames
        late = front[-5:]
        edge_mean = 0.5 * (late[:, y_edge].astype(float).mean() + late[:, y_other].astype(float).mean())
        mid_mean = late[:, y_mid].astype(float).mean()
        delta = edge_mean - mid_mean
        summary_lines.append(
            f"{state}: edge_front={edge_mean:6.2f}  mid_front={mid_mean:6.2f}  "
            f"edge_lead={delta:+6.2f}  (ratio={edge_mean / max(mid_mean, 1e-6):6.3f})"
        )

    axes[0].set_ylabel("front x position (tank index)")
    fig.suptitle("Wavefront position per row — if edge leads, boundary speedup is present")
    fig.tight_layout()
    fig.savefig(out_dir / "front_comparison.png", dpi=120)
    print(f"  wrote {out_dir / 'front_comparison.png'}")

    summary = "\n".join(summary_lines) + "\n"
    (out_dir / "summary.txt").write_text(summary)
    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":
    main()
