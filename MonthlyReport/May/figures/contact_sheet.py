"""Contact sheet of sampled textbook pages -> pick collage candidates."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import glob, os, re

d = "/home/norepinephrine/Documents/Heart-Conduction/MonthlyReport/May/figures/tb_pages"
files = sorted(glob.glob(os.path.join(d, "pg_*.png")))
n = len(files)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2*rows), dpi=110)
axes = axes.ravel()
for ax in axes:
    ax.axis("off")
for ax, f in zip(axes, files):
    img = Image.open(f)
    ax.imshow(img)
    pg = re.search(r"pg_(\d+)", f).group(1)
    ax.set_title(f"page {int(pg)}", fontsize=13, fontweight="bold")
    ax.axis("off")
out = os.path.join(d, "..", "contact_sheet.png")
fig.tight_layout()
fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
print("wrote", os.path.abspath(out))
