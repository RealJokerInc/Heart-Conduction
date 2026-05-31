"""Collage of the cardiac computational-modeling textbook for the May report.
One slide: 'what's happening' in the textbook — page gallery spanning Parts I-III."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

plt.rcParams.update({"font.family": "DejaVu Sans"})

D = "/home/norepinephrine/Documents/Heart-Conduction/MonthlyReport/May/figures/tb_hi"
pages = [
    ("p008-008.png", "Part I — Hodgkin–Huxley\n& gating kinetics"),
    ("p018-018.png", "Part I — Calcium handling\n(Intuition callout box)"),
    ("p030-030.png", "Part I — O'Hara–Rudy 2011\nionic model"),
    ("p045-045.png", "Part II — Monodomain eqn\n& spatial discretization"),
    ("p060-060.png", "Part II — Diffusion solvers\n(explicit / implicit)"),
    ("p095-095.png", "Part III — Bidomain:\nmatrices & face stencils"),
]

NAVY = "#1f3c88"
fig = plt.figure(figsize=(10.5, 6.4), dpi=200)
fig.patch.set_facecolor("white")

# header band (slide title comes from the deck's title bar; keep only the stats)
fig.text(0.5, 0.955,
         "20 chapters · 4 parts · 139 pp · Single-Cell → Monodomain → Bidomain → LBM",
         ha="center", va="top", fontsize=12.5, fontweight="bold", color=NAVY)
fig.text(0.5, 0.915,
         "Feynman-style; equations verified against Engine V5.4 code",
         ha="center", va="top", fontsize=10.5, fontstyle="italic", color="#666")

# 2x3 gallery
cols, rows = 3, 2
left, right, top, bot = 0.035, 0.965, 0.865, 0.115
gap_x, gap_y = 0.022, 0.105
cell_w = (right - left - (cols - 1) * gap_x) / cols
cell_h = (top - bot - (rows - 1) * gap_y) / rows

for idx, (fname, cap) in enumerate(pages):
    r, c = divmod(idx, cols)
    x0 = left + c * (cell_w + gap_x)
    y0 = top - (r + 1) * cell_h - r * gap_y
    ax = fig.add_axes([x0, y0, cell_w, cell_h])
    img = Image.open(f"{D}/{fname}")
    # fit portrait page into the cell, centered
    ax.imshow(img, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#c7c7c7"); s.set_linewidth(1.2)
    # caption strip beneath
    fig.text(x0 + cell_w / 2, y0 - 0.012, cap, ha="center", va="top",
             fontsize=10.0, color="#222", linespacing=1.15)

fig.text(0.5, 0.018,
         "Self-authored reference, ion-channel kinetics → tissue-scale solvers — the conceptual backbone for all project engines.",
         ha="center", va="bottom", fontsize=9.5, fontstyle="italic", color="#666")

out = "/home/norepinephrine/Documents/Heart-Conduction/MonthlyReport/May/figures/textbook_collage.png"
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
