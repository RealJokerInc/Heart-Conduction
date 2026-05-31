# Surrogate Training Data v2 — Specification

Created: 2026-04-21
Status: Active — T1 defined, T2–T12 TBD tier-by-tier.
Root: `/media/shared/norepinephrine/surrogate_data_v2/`

## Motivation

The v1 data (12 tiers on HDD, `/media/HDD/norepinephrine/surrogate_data/raw/`) has three structural problems that block Session 29 training plans:

1. **Sparse BCL/DI grids** — T1 had only 9 BCLs, T2 only 8 DIs. Splitting into train/val left fewer than 5 held-out protocols per tier; adding a test set was effectively impossible.
2. **Segment extraction ignored rest-start contract** — `SegmentDataset` used uniform stride from index 0, landing mid-AP where `z0 = zeros + INIT_CONC` doesn't match the ground truth, violating the NODE training invariant.
3. **Implicit schema** — 47-column layout encoded only in `preprocessor.py` constants, splits hardcoded in `data_cache.py` dict, no provenance. Not self-describing.

v2 fixes all three, tier by tier.

## Directory Layout

```
/media/shared/norepinephrine/surrogate_data_v2/
├── MANIFEST.yaml                 ← top-level index
├── DATA_V2_SPEC.md               ← symlink/copy of this file
├── raw/
│   ├── tier01_epi.h5             ← one file per (tier, cell_type)
│   ├── tier01_endo.h5            ← (future) T12 equivalents fold in as T1_endo etc.
│   ├── tier02_epi.h5
│   └── …
├── splits/
│   ├── tier01_v2.json            ← across-BCL + within-BCL split definitions
│   └── …
└── provenance/
    └── tier01_epi_genlog.json    ← per-group timing, git SHA, errors
```

## File-Level Conventions

### HDF5 file attrs (required on every file)

| Attr | Type | Example |
|------|------|---------|
| `dataset_version` | str | `"v2"` |
| `tier_id` | int | `1` |
| `tier_description` | str | `"Steady-state pacing, 35 BCLs × 50 beats"` |
| `cell_type` | str | `"EPI"` / `"ENDO"` / `"M_CELL"` |
| `ionic_model` | str | `"TTP06"` |
| `dt_ms` | float | `0.01` |
| `simulator_engine` | str | `"Monodomain/Engine_V5.4"` |
| `simulator_commit` | str | `<git rev-parse HEAD at gen time>` |
| `generated_at_utc` | str | ISO-8601 |
| `column_names` | array[str] | 47 strings |
| `column_units` | array[str] | 47 strings |
| `column_groups` | str (JSON) | `{"Vm":[0], "stim":[1], …}` |

### Column schema (47 cols, same as v1 — promote to attrs)

```
idx   name         units   notes
0     Vm           mV
1     I_stim       pA/pF
2     dt           ms      per-step dt (constant at 0.01 for v2)
3     K_i          mM      TTP06 StateIndex.Ki
4     Na_i         mM
5     Ca_i         mM
6     Ca_SR        mM
7     Ca_ss        mM
8     m            —       gate (0-1)
9     h            —
10    j            —
11    r            —
12    s            —
13    d            —
14    f            —
15    f2           —
16    fCass        —
17    Xr1          —
18    Xr2          —
19    Xs           —
20    RR           —       ryanodine receptor state
21    I_ion        pA/pF   total ionic current
22    clamp_mask   —       1 if Vm clamped at this step, else 0
23-34 gate_inf     —       12 HH gates g∞(V) at this V
35-46 gate_tau     ms      12 HH gate τ(V)
```

The definitive constant lives in `Surrogate/surrogate/data/schema.py` (new file; see Implementation). Both the generator and the loader read from there.

## Group Conventions

**Naming**: short structured names — `bcl200`, `bcl1000`, `di50`, `ramp_1000to300`, `alternans_bcl330`. No `steady_` / `_dt0.01` suffixes (redundant with file attrs).

**Required group attrs**:
- `protocol_type` — `"steady_pacing" | "s1s2" | "ramp" | "burst" | "alternans" | "random_pacing" | "voltage_clamp" | "current_injection" | "stitched"`
- `bcl_ms` — for steady pacing / alternans; `null` otherwise
- `n_beats` — integer
- `duration_ms` — total trajectory length
- `n_timesteps` — `data.shape[0]`
- `stim_amplitude_pA_pF` — float
- `stim_duration_ms` — float
- `stim_onsets_ms` — array[float], length `n_beats`, absolute ms within trajectory
- `beat_boundaries_idx` — array[int], length `n_beats+1`, timestep index of each stim onset (start-of-beat) + final index
- `capture_flag` — bool, 1:1 capture confirmed (see §Quality Flags)
- `alternans_flag` — bool, APD alternans detected (short-long beat pattern)

**Dataset**: single `data` dataset per group, shape `(n_timesteps, 47)` float64, gzip level 4, chunks `(65536, 47)`.

## Quality Flags (computed at generation time)

- `capture_flag = true` if every stim onset produces an upstroke with peak Vm > 0 mV within 5 ms.
- `alternans_flag = true` if consecutive APD values differ by > 10 % for the last 10 beats (after warm-up).
- Both written to group attrs; group is still saved even if flags fail — downstream can filter.

## Split File Format

One JSON sidecar per tier, in `splits/tier{NN}_v2.json`:

```json
{
  "dataset_version": "v2",
  "tier": 1,
  "cell_type": "EPI",
  "strategy": "two_axis",
  "rationale": "Stratified across low/mid/high BCL regimes. Within-BCL split tests temporal generalization on held-in BCLs.",
  "across_bcl": {
    "train": [200, 230, 250, 270, 300, 350, 400, 500, 600, 650, 700, 750, 800, 950, 1000, 1100, 1300, 1400, 1600, 1700, 2000, 210, 290],
    "val":   [220, 280, 450, 900, 1200, 1800],
    "test":  [240, 260, 550, 850, 1500, 1900]
  },
  "within_bcl": {
    "applies_to": "across_bcl.train",
    "warmup_beats": [0, 14],
    "train_beats":  [15, 39],
    "val_beats":    [40, 44],
    "test_beats":   [45, 49]
  }
}
```

Contract: loader reads the sidecar, never the h5 attrs, for split decisions.

## MANIFEST.yaml (top-level index)

```yaml
dataset_version: v2
root: /media/shared/norepinephrine/surrogate_data_v2
simulator:
  engine: Monodomain/Engine_V5.4
  ionic_model: TTP06
  git_sha: <...>
  dt_ms: 0.01
tiers:
  1:
    description: Steady-state pacing
    files: [raw/tier01_epi.h5]
    split: splits/tier01_v2.json
    n_bcls: 35
    beats_per_bcl: 50
    status: generating | done | todo
```

Regenerated by the generator on each tier completion (idempotent merge).

---

## T1 — Steady-State Pacing

### Grid

| Range | Step | BCLs | Count |
|-------|------|------|------:|
| 200–300 ms | 10 ms | 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300 | 11 |
| 350–1000 ms | 50 ms | 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000 | 14 |
| 1100–2000 ms | 100 ms | 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000 | 10 |
| **Total** | | | **35** |

### Beats per BCL: 50

Rationale — beats 0–14 are warmup (not fully settled); beats 15–49 are settled and usable. The last 20 beats capture slow concentration relaxation (CaSR, Na_i equilibrate over 60–120 s). Cost: ~2.5× the prior 20-beat budget, but adds strictly new information unlike extra BCLs.

### Split (locked)

Regime-stratified so each split sees low, mid, and high BCLs:

| Regime | BCLs in regime | Train | Val | Test |
|--------|---------------:|------:|----:|-----:|
| Low (200–300) | 11 | 7 | 2 | 2 |
| Mid (350–1000) | 14 | 10 | 2 | 2 |
| High (1100–2000) | 10 | 6 | 2 | 2 |
| **Total** | 35 | **23** | **6** | **6** |

**Across-BCL splits:**
- **Val BCLs (6)**: 220, 280, 450, 900, 1200, 1800
- **Test BCLs (6)**: 240, 260, 550, 850, 1500, 1900
- **Train BCLs (23)**: everything else

**Within-BCL splits (on the 23 train BCLs):**
- Warmup (unused): beats 0–14
- Train: beats 15–39 (25 beats)
- Val: beats 40–44 (5 beats)
- Test: beats 45–49 (5 beats)

### Effective segment counts

- Train: 23 train BCLs × 25 beats = **575 segments** (+ within-BCL train)
- Val total: 6 val BCLs × 35 settled beats + 23 train BCLs × 5 within = 210 + 115 = **325 segments**
- Test total: 6 test BCLs × 35 settled beats + 23 train BCLs × 5 within = 210 + 115 = **325 segments**

vs v1 oracle's 25 total segments — **~20×** more effective training data with genuine held-out splits on two axes.

### Wall-clock + disk estimate

- Per BCL: `bcl_ms × 50 beats / 0.01 ms = (bcl_ms × 5000)` timesteps
- Total timesteps: `sum(bcl_ms × 5000 for bcl_ms in all 35)` ≈ 5 × (11×250 + 14×675 + 10×1550) / 1000 = **~130 M timesteps**
- With 47 cols float64 uncompressed: 130M × 47 × 8 = ~48 GB
- With gzip-4: expected ~20 GB
- Sim wall-clock at prior rate: T1 v1 was 9 BCLs × 20 beats = 16.6 M timesteps in 2657 s (6.2 kstep/s effective, batched). T1 v2 scales to ~130 M / 16.6 M × 2657 s = ~350 min = **~5–6 hours**. May reduce with larger GPU batch (35 parallel trajectories).

### T1 generation command (spec)

```bash
python Surrogate/datagen/generate_t1_v2.py \
  --celltype EPI \
  --output /media/shared/norepinephrine/surrogate_data_v2/raw/tier01_epi.h5 \
  --split-out /media/shared/norepinephrine/surrogate_data_v2/splits/tier01_v2.json
```

---

## T2 — Restitution (S1S2) — TBD

Placeholder. Same design approach: dense DI grid, stratified train/val/test split. Specified after T1 completes.

## T3 — Dynamic Protocols — TBD

## T4 — Random Pacing — TBD

## T5–T12 — TBD

Specified tier by tier as we work through the audit.

---

## Implementation Files

| File | Role |
|------|------|
| `Surrogate/surrogate/data/schema.py` | Single source of truth for column names, units, groups. Read by generator + loader. |
| `Surrogate/datagen/generate_t1_v2.py` | T1 v2 generator. Bypasses v1 `TraceStorage` mount-check; writes directly via `h5py` with full v2 attrs. |
| `Surrogate/datagen/write_manifest.py` | Rebuild `MANIFEST.yaml` from current state of `raw/` + `splits/`. Idempotent. |
| `Surrogate/surrogate/data/loader_v2.py` | New loader that reads v2 h5 + split JSON, replaces `data_cache.py`. Written after T1 generated. |

## Contract Checklist (for every tier generator)

Before a tier's h5 is considered complete:

- [ ] File attrs include all required fields (§File-Level Conventions)
- [ ] Every group has all required attrs (§Group Conventions)
- [ ] `data` shape is `(n_timesteps, 47)`, no NaN, no Inf
- [ ] All gate columns in `[-1e-6, 1+1e-6]`; all concentrations > 0
- [ ] `capture_flag` and `alternans_flag` computed and stored
- [ ] Split sidecar JSON written, validated against h5 group names
- [ ] `provenance/{tier}_genlog.json` written with per-group timing + any warnings
- [ ] MANIFEST.yaml updated

Violations abort tier generation with an error log in `provenance/`.
