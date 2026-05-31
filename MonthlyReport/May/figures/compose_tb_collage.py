"""14x4 textbook page grid (56 tiles), text-free, compact.
- Mostly-empty pages (cover, dividers, end pages) excluded by ink density.
- The LBM 'Complete LBM-EP Algorithm' page (p113) embedded at row 2, last column.
lbm_page.png (the standalone LBM page) is produced separately and left untouched.
"""
from PIL import Image
import glob, re, numpy as np

FIG    = "/home/norepinephrine/Documents/Heart-Conduction/MonthlyReport/May/figures"
THUMBS = FIG + "/_tb_all"
GRID_OUT = FIG + "/textbook_pages_collage.png"
LBM_PAGE = 113
INK_MIN = 0.04   # exclude pages with < 4% non-white pixels (near-blank)

paths = {int(re.search(r"t-(\d+)", f).group(1)): f for f in glob.glob(THUMBS + "/t-*.png")}
npages = max(paths)

# ink density per page
ink = {}
for pg, f in paths.items():
    g = np.asarray(Image.open(f).convert("L"))
    ink[pg] = float((g < 245).mean())
empty = sorted(p for p in ink if ink[p] < INK_MIN)
print("excluded (mostly empty):", empty)

COLS, ROWS = 14, 4
N = COLS * ROWS                    # 56
LBM_POS = 1 * COLS + (COLS - 1)    # row2, last col -> index 27

# content pool: not empty, not the LBM page (inserted separately)
pool = [p for p in range(1, npages + 1) if ink[p] >= INK_MIN and p != LBM_PAGE]
samp = sorted(set(int(round(v)) for v in np.linspace(0, len(pool) - 1, N - 1)))
i = 0
while len(samp) < N - 1:
    if i not in samp and i < len(pool):
        samp.append(i)
    i += 1
sel = [pool[j] for j in sorted(samp)[:N - 1]]
order = sel[:LBM_POS] + [LBM_PAGE] + sel[LBM_POS:]
assert order[LBM_POS] == LBM_PAGE and len(order) == N

TILE_W = 160
TILE_H = int(round(TILE_W / 0.707))
GAP, MARGIN = 6, 16
grid_w = COLS * TILE_W + (COLS - 1) * GAP
grid_h = ROWS * TILE_H + (ROWS - 1) * GAP
canvas = Image.new("RGB", (MARGIN * 2 + grid_w, MARGIN * 2 + grid_h), "white")
for k, pg in enumerate(order):
    r, c = divmod(k, COLS)
    t = Image.open(paths[pg]).convert("RGB").resize((TILE_W, TILE_H), Image.LANCZOS)
    canvas.paste(t, (MARGIN + c * (TILE_W + GAP), MARGIN + r * (TILE_H + GAP)))
canvas.save(GRID_OUT)
print("grid:", GRID_OUT, canvas.size, "| tiles:", N, "| pool:", len(pool),
      "| LBM at index", LBM_POS, "(row2/col14)")
print("pages shown:", order)
