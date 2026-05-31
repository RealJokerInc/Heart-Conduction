"""Qualitative schematic: why the boundary condition sets wavefront curvature.
Conceptual (not data). For the May progress report, boundary-speedup section.

Story: a wall cell has fewer neighbours than an interior cell -> two competing
effects (drains less = speedup; receives less = slowdown). The boundary
condition decides how the off-grid diagonal channels are handled, which tips
the balance:
  face_mirror / HBB        -> diagonal returns the cell's OWN value (dead source)
                              -> "receives less" wins -> boundary LAGS -> forward crescent
  specular / face_mirror_iso-> diagonal pulls a REAL upstream neighbour value
                              -> effects balance -> flat wavefront
  horizontal redirect (new)-> diagonal mass funnelled along the wall (extra source)
                              -> "drains less" wins -> boundary LEADS -> inverse crescent
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
import numpy as np

# PPT-readable fonts (obs #19: labels must survive scaling into a slide placeholder)
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "font.family": "DejaVu Sans",
})

GREEN = "#1b7a3d"   # live upstream source
GREY  = "#9aa0a6"   # dead / self channel
ORANGE = "#d9620a"  # redirected along-wall source
BLUE  = "#1f3c88"   # cell fill
WALL  = "#222222"

fig = plt.figure(figsize=(13.0, 6.6), dpi=200)

# ---- top header: the two competing effects --------------------------------
axh = fig.add_axes([0.04, 0.74, 0.92, 0.22])
axh.axis("off")
axh.set_xlim(0, 10); axh.set_ylim(0, 2.2)

# interior vs boundary cell pipe-count cartoon
def star_cell(ax, cx, cy, dirs, r=0.34, arr=0.42, color_map=None):
    ax.add_patch(Rectangle((cx-r, cy-r), 2*r, 2*r, fc=BLUE, ec="k", lw=1.4, zorder=3))
    for (dx, dy) in dirs:
        c = (color_map or {}).get((dx, dy), GREEN)
        ax.add_patch(FancyArrowPatch((cx+dx*(r+arr), cy+dy*(r+arr)), (cx+dx*r, cy+dy*r),
                     arrowstyle="-|>", mutation_scale=13, lw=2.0, color=c, zorder=2))

alld = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]
star_cell(axh, 1.0, 1.0, alld)
axh.text(1.0, 1.0-0.62, "interior cell\n8 neighbours", ha="center", va="top", fontsize=12.5)

# boundary cell: top row off-grid (NW,N,NE missing)
bdry = [(-1,0),(1,0),(0,-1),(-1,-1),(1,-1)]
star_cell(axh, 3.2, 1.0, bdry)
axh.plot([2.4, 4.0], [1.42, 1.42], color=WALL, lw=3.0)  # the wall
axh.text(4.05, 1.42, "wall", ha="left", va="center", fontsize=12, color=WALL)
axh.text(3.2, 1.0-0.62, "boundary cell\nonly 5 neighbours", ha="center", va="top", fontsize=12.5)

# the tug of war text
axh.text(5.2, 1.62, "Fewer neighbours → two competing effects:", fontsize=14.5, fontweight="bold", va="center")
axh.text(5.2, 1.06, "Effect 1  —  receives less source  (fewer inbound channels)  →  tends to SLOW DOWN",
         fontsize=13.5, color=GREEN, va="center")
axh.text(5.2, 0.52, "Effect 2  —  streams to less sink  (less downstream load)  →  tends to SPEED UP",
         fontsize=13.5, color=ORANGE, va="center")
axh.text(5.2, 0.02, "The boundary condition decides which effect wins.",
         fontsize=13.5, fontstyle="italic", va="center")

# ---- three BC-family columns ----------------------------------------------
titles = ["face_mirror  ≡  HBB", "specular  ≡  face_mirror_iso", "horizontal redirect  (new)"]
sub    = ["zero-flux wall", "transparent wall", "along-wall channel"]
results = ["FORWARD crescent\nboundary LAGS",
           "FLAT wavefront\nno bias",
           "INVERSE crescent\nboundary LEADS  =  SPEEDUP"]
takeaway = ["Off-grid diagonal returns the cell's OWN\nvalue → no upstream source through it.\nEffect 1 wins: boundary receives less.",
            "Off-grid diagonal pulls a REAL upstream\nneighbour's value → source restored.\nEffects balance: no curvature.",
            "Diagonal mass streamed along the wall to\nthe next cell → wall channel races ahead.\nEffect 2 wins: boundary leads."]
crescent_curv = [+1.0, 0.0, -1.0]
res_color = [GREEN, "#444444", ORANGE]

xs = [0.045, 0.365, 0.685]
W = 0.27
for i in range(3):
    # --- mechanism mini-schematic (upper) ---
    ax = fig.add_axes([xs[i], 0.40, W, 0.27])
    ax.axis("off"); ax.set_xlim(0,3); ax.set_ylim(0,3)
    ax.set_title(titles[i], fontsize=15.5, fontweight="bold", pad=2)
    ax.text(1.5, 2.62, sub[i], ha="center", fontsize=12, color="#555")
    # wall
    ax.plot([0.2, 2.8], [2.05, 2.05], color=WALL, lw=3.5)
    # boundary cell
    ax.add_patch(Rectangle((1.2, 1.25), 0.6, 0.6, fc=BLUE, ec="k", lw=1.4, zorder=3))
    # ghost diagonal slot above wall (dashed)
    ax.add_patch(Rectangle((0.55, 2.15), 0.55, 0.55, fc="none", ec=GREY, lw=1.3, ls="--", zorder=2))
    ax.text(0.825, 2.42, "ghost", ha="center", va="center", fontsize=9.5, color=GREY)
    # real upstream (west) cell
    ax.add_patch(Rectangle((0.25, 1.25), 0.55, 0.6, fc="#c9d6ef", ec="k", lw=1.0, zorder=2))
    ax.text(0.52, 1.05, "upstream", ha="center", va="top", fontsize=9.5)
    if i == 0:  # HBB: ghost = self (dead), only cardinal+lower-diag feed
        ax.add_patch(FancyArrowPatch((0.82,2.18),(1.35,1.86), arrowstyle="-|>",
                     mutation_scale=12, lw=2.2, color=GREY, ls="--", zorder=4))
        ax.text(1.55, 2.0, "dead", color=GREY, fontsize=10.5, va="center")
        ax.add_patch(FancyArrowPatch((0.8,1.55),(1.18,1.55), arrowstyle="-|>",
                     mutation_scale=12, lw=2.2, color=GREEN, zorder=4))
    elif i == 1:  # specular: ghost filled from real upstream (green)
        ax.add_patch(FancyArrowPatch((0.8,1.6),(0.82,2.12), arrowstyle="-|>",
                     mutation_scale=12, lw=2.2, color=GREEN, zorder=4))
        ax.add_patch(FancyArrowPatch((0.92,2.4),(1.4,1.86), arrowstyle="-|>",
                     mutation_scale=12, lw=2.2, color=GREEN, zorder=4))
        ax.add_patch(FancyArrowPatch((0.8,1.55),(1.18,1.55), arrowstyle="-|>",
                     mutation_scale=12, lw=2.2, color=GREEN, zorder=4))
    else:  # horizontal: diagonal mass redirected along wall into cardinal slot
        ax.add_patch(FancyArrowPatch((0.82,2.18),(1.18,1.7), arrowstyle="-|>",
                     mutation_scale=12, lw=2.2, color=ORANGE, zorder=4))
        ax.add_patch(FancyArrowPatch((0.8,1.7),(1.18,1.7), arrowstyle="-|>",
                     mutation_scale=12, lw=2.4, color=ORANGE, zorder=4))
        ax.text(1.55, 1.95, "extra", color=ORANGE, fontsize=10.5, va="center")
        ax.add_patch(FancyArrowPatch((0.8,1.5),(1.18,1.5), arrowstyle="-|>",
                     mutation_scale=12, lw=2.2, color=GREEN, zorder=4))

    # --- wavefront curvature cartoon (lower) ---
    axc = fig.add_axes([xs[i], 0.135, W, 0.215])
    axc.set_xlim(0,1); axc.set_ylim(0,1)
    axc.set_xticks([]); axc.set_yticks([])
    for sp in axc.spines.values():
        sp.set_edgecolor("#bbb")
    # wall at top
    axc.plot([0,1],[1.0,1.0], color=WALL, lw=3.0, clip_on=False)
    axc.text(0.5, 1.04, "wall", ha="center", va="bottom", fontsize=10, color=WALL, transform=axc.transAxes)
    y = np.linspace(0, 1, 100)
    bend = (y**2)  # strongest near wall (y=1)
    for x0, alpha in [(0.32,0.35),(0.52,0.6),(0.72,1.0)]:
        x = x0 - crescent_curv[i]*0.18*bend
        axc.plot(x, y, color=res_color[i], lw=2.2+1.4*alpha, alpha=0.35+0.5*alpha)
    axc.annotate("", xy=(0.93,0.5), xytext=(0.80,0.5),
                 arrowprops=dict(arrowstyle="-|>", color="#777", lw=1.6))
    axc.text(0.04, 0.06, "wave →", fontsize=10, color="#777")
    axc.text(0.5, -0.13, results[i], ha="center", va="top", fontsize=12.5,
             fontweight="bold", color=res_color[i], transform=axc.transAxes)

    # --- takeaway ---
    fig.text(xs[i]+W/2, 0.055, takeaway[i], ha="center", va="top", fontsize=10.8,
             color="#333", linespacing=1.25)

out = "/home/norepinephrine/Documents/Heart-Conduction/MonthlyReport/May/figures/bc_mechanism.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
