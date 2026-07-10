# AP-Morphology / Ion-Current Playground — Design Brief

> ✅ **BUILT & WORKING 2026-07-10** (branch `textbook-website-refresh`). Generator `website/build/gen_ap_traces.py`
> (torch.compile-batched, CPU) produced the exact trace bank `website/data/ap_explorer/*.json` (8 configs, 924 KB);
> widget `website/figures/ap-explorer.js` + page `website/chapters/playground.html` + "Interactive Tools" nav.
> Validated: all 4 engines load (TTP06 APD90 236/236/293 ENDO/EPI/M · ORd 258/239/377 · PHAS13 CL 1634 ms APD90 568 ·
> MHAS13 537); GNa 1×→0× drops V_peak 57.7→22.8 live; PHAS13 shows spontaneous beats with I_f as the flagship knob;
> both themes, 0 console errors; NOT in the PDF (still 195 pp). Design brief below retained for provenance.


> A new interactive **simulation page** for the textbook website: pick an ionic engine, tune its
> conductance knobs (incl. unfamiliar ones like the funny current `I_f`), and watch the action-potential
> morphology change — physically exact, from the real cardiac_core engines.
> Requested 2026-07-10. Architecture decided by the user: **precompute + knob grids** (exact real-engine
> traces; combinable grids for headline currents). Builds on the Phase 1–2 website refresh (design tokens +
> figure-widget framework). This brief is the pre-blueprint plan; `/blueprint` turns it into PLAN.md steps.

## Goal
Expose, interactively, **how each ionic current shapes the AP** — the thing the engines currently "do too good a
job" of hiding. Choose among **all four production ionic engines** cardiac_core wires up, tune per-current
conductances, and see the AP waveform + derived metrics respond in real time. Double as a teaching tool that
demystifies unfamiliar currents (funny current, NCX, pumps, backgrounds).

## The reality that sets the architecture
The engines are **PyTorch** → they cannot run in the browser (no Pyodide/torch path). So AP traces are **precomputed
offline with the real cardiac_core engines** and shipped as compact data; the page plots/scrubs them instantly.
**Every displayed trace is exact real-engine output** (honors "ground truth is reality"). To allow a few currents to
be tuned *together*, we precompute a small **N-D grid** over the headline knobs; the rest are **1-D isolated sweeps**.
Sliders **snap to precomputed grid levels** (each detent = an exact trace) — no physically-invalid interpolation.

## The engines & knobs (verified against code)
Selected via `cardiac_core.ionic.registry.build_ionic_model(name, cell_type, device)`.
Conductances are mutable dataclass fields on `model.params`; tuning = scale × default (relative factor).

| Engine (name) | Model | States | Beating | Cell types | Primary grid knobs (combine) | Isolated sweep knobs |
|---|---|---|---|---|---|---|
| `ttp06` (default) | ten Tusscher–Panfilov 2006 | 18 | paced | ENDO/EPI/M | `GNa`, `PCa`(I_CaL), `GKr` | `GKs, GK1, Gto, GpCa, GpK, GbNa, GbCa` |
| `ord` | O'Hara–Rudy 2011 | 40 | paced | ENDO/EPI/M | `GNa`, `PCa`(I_CaL), `GKr` | `GNaL`(late Na), `GKs, GK1, Gto, …` ⚠ confirm exact names from `cardiac_core/ionic/ord/parameters.py` |
| `phas13` ("PHA-S") | Paci 2013 hiPSC-CM | 17 | **spontaneous** (I_f) | — | **`g_f`(funny)**, `g_CaL`, `g_Kr` | `g_Na, g_Ks, g_K1, g_to, kNaCa, PNaK, g_pCa, g_bNa, g_bCa` |
| `mhas13` | matured PHAS13 (g_f=0, TTP06 IK1) | 17 | paced | — | `g_Na`, `g_CaL`, `g_Kr` | `g_Ks, GK1_ttp06, g_to, kNaCa, PNaK, …` |

**Verified defaults** — TTP06: `GNa=14.838, GK1=5.405, Gto=0.294, GKr=0.153, GKs=0.392, PCa=3.98e-5, GpCa, GpK, GbNa, GbCa`.
PHAS13: `g_Na=3.6712302, g_CaL=8.635702e-5, g_Kr=0.0298667, g_Ks=0.002041, g_K1=0.0281492, g_to=0.0299038, g_f=0.03010312, kNaCa=4900, PNaK=1.841424, g_pCa=0.4125, g_bNa=0.0009, g_bCa=0.00069264`.
MHAS13: PHAS13 with `g_f=0.0`, `GK1_ttp06=3.170`.

**Per-engine nuance that MUST be honored:**
- **PHAS13 self-oscillates** (no stimulus) — capture the free-running AP; report **spontaneous cycle length / rate**.
  Cranking `g_f` speeds the rate; `g_f→0` → quiescent (≈ MHAS13). This is the flagship funny-current demo.
- **TTP06 / ORd / MHAS13** — pace to steady state (BCL 1000 ms, ~50 beats), capture the **last beat**.

## The "unfamiliar currents" (PHAS13's 12 — the education layer)
Each gets a one-line glossary card on the page; the flagged ones are the point of the tool:
`I_Na` fast Na⁺ (upstroke) · `I_CaL` L-type Ca²⁺ (plateau, GHK) · `I_Kr` rapid K⁺ (repol) · `I_Ks` slow K⁺ (Ca-dependent) ·
`I_K1` inward-rectifier K⁺ (resting) · `I_to` transient-outward K⁺ (notch) · **`I_f` funny/HCN pacemaker (automaticity,
E_f=−17 mV)** · **`I_NaCa` Na/Ca exchanger (electrogenic antiporter)** · **`I_NaK` Na/K ATPase pump** ·
**`I_pCa` sarcolemmal Ca pump** · **`I_bNa`/`I_bCa` background leaks**.

## Generation script (offline, Python) — NEW `website/build/gen_ap_traces.py`
- Uses **`cardiac_core.ionic`** models directly (avoid the legacy `cardiac_sim` dep). For each engine/cell-type:
  build model → set `model.params.<knob> = level × default` → run to the correct endpoint (paced last beat, or
  free-run for PHAS13) → capture `Vm(t)` (and `dV/dt`).
- **Steady state:** paced models = 50 beats @ BCL 1000 ms, dt=0.01 ms, stim −80 A/F × 1 ms; keep the last beat
  (~400 ms window). PHAS13 = free-run ~5–10 s, keep the last full spontaneous cycle.
- **Grid:** 3 primary knobs × 5 levels `{0, 0.5, 1.0, 1.5, 2.0}×` = 125 traces/config (combinable, snap-to-grid).
- **Sweeps:** each secondary knob swept alone at ~11 levels `{0…2.0}×`, others baseline.
- **Cost is trivial** (GPU ~77 M cell-steps/s; one steady-state pace ≈ 0.065 s) — batch all grid+sweep cells per
  engine with the batched `step()`; whole dataset generates in minutes. Run on the RTX 4500.
- **Downsample + quantize** each captured Vm(t) to ~400 samples, int8 over a fixed [−95, +55] mV window (≈400 B/trace).
- **Derived metrics per trace** (computed in Python via `cardiac_core.analysis.apd_at`): APD90, APD50, dV/dt_max,
  V_rest, V_peak, (spontaneous CL for PHAS13). Stored alongside so the page shows them without recompute.
- **Validation:** assert baseline APD/CL against known values (TTP06 EPI APD90 ≈ 300 ms; PHAS13 spontaneous CL in
  the physiological hiPSC range) — a generation self-check, since the traces ARE the real engine.

## Data format — `website/data/ap_explorer/<engine>[_<celltype>].json` (lazy-loaded per selection)
```
{ "engine":"ttp06", "cellType":"EPI", "beating":"paced", "bcl":1000,
  "t_ms":[…400 sample times…], "vmin":-95, "vmax":55,           // int8 dequant window
  "baseline": {"vm":[…int8…], "apd90":300, "apd50":220, "dvdt":250, "vrest":-85, "vpeak":40},
  "knobs":[{"id":"GNa","label":"I_Na — fast sodium","default":14.838,"tier":"grid"}, …],
  "grid": {"axes":["GNa","PCa","GKr"], "levels":[0,0.5,1,1.5,2],
           "traces":[{"idx":[i,j,k],"vm":[…],"apd90":…,…}, …]},   // 125 entries
  "sweeps": {"GKs":[{"level":0,"vm":[…],"apd90":…}, …11…], "GK1":[…], …} }
```

## The page (browser) — reuses Phase 1–2 design system + `figures/_canvas.js`
A dedicated **full-width tool** (not a book figure). Layout:
- **Engine bar** — segmented control: TTP06 · ORd · PHAS13 · MHAS13; a cell-type toggle (ENDO/EPI/M) shown only for
  TTP06/ORd; a "spontaneous" badge for PHAS13.
- **AP plot** (canvas, themeable `--fig-*`) — current Vm(t) in crimson over the **baseline in grey** for comparison;
  axis in mV/ms (mono labels). PHAS13 shows ~2 spontaneous beats.
- **Metrics strip** — APD90 · APD50 · dV/dt_max · V_rest · V_peak · (CL for PHAS13), tabular-nums, updating live.
- **Combine panel** — the 3 primary knob sliders (snap to the 5 grid levels) → indexes the grid → exact combined trace.
- **Isolate-a-current panel** — the secondary knobs; moving one shows its solo sweep (primaries reset to baseline,
  clearly labelled "isolated effect"). Each knob row carries its glossary one-liner; unfamiliar ones badged.
- **Reset** to defaults; a small "what am I looking at?" explainer + a link back to the relevant chapters.

## Website integration
- New sidebar section **"Interactive / Tools"** with an **"Ion-Current Playground"** entry; loaded by the SPA as a
  fragment `chapters/playground.html` carrying `<div data-widget="ap-explorer">` → the widget mounts via the existing
  `figures.js` loader. New widget module `website/figures/ap-explorer.js` (fetches the per-engine JSON on select).
- **Not in the PDF** — it's a live tool. Exclude `playground.html` from `toc.json`'s PDF assembly path (or give it a
  static screenshot page in print). `html_to_pdf.py` already only assembles part/chapter fragments, so simply not
  adding it to a `part` keeps it out of the PDF.
- Reuses `style.css` tokens + `.fw-*` control styles; add a `.tool`/`.ap-explorer` layout block.

## Phasing
1. **Generation** — `gen_ap_traces.py` + validated data for all 4 engines (TTP06 ×3 cell types, ORd ×3, PHAS13, MHAS13)
   → `website/data/ap_explorer/*.json`. Confirm ORd knob names from `ord/parameters.py`. Deliverable: exact trace bank.
2. **Page/widget** — `ap-explorer.js` (engine select → lazy-load JSON → plot + metrics + combine/isolate panels),
   `playground.html` fragment, Tools nav section, CSS. Themeable, reduced-motion, keyboard-operable.
3. **Education + polish** — current glossary cards, the funny-current spontaneous-rate demo, cross-links to chapters,
   a11y pass, screenshots. Optional later: live-JS TTP06 for off-grid combined tuning.

## Open items to confirm at build time
- Exact ORd conductance field names + which 3 are the primary grid (read `cardiac_core/ionic/ord/parameters.py`;
  ORd's **late Na `GNaL`** is a strong secondary-sweep candidate — big APD effect).
- PHAS13 free-run duration to reach a stable spontaneous CL; whether to also offer a "paced PHAS13" mode.
- Cheatsheet doc gap: `API_CHEATSHEET.md §4` lists only ttp06/ord — worth adding phas13/mhas13 (separate small fix).
- Grid at 5 levels vs 7 (storage vs smoothness); int8 quantization window.
