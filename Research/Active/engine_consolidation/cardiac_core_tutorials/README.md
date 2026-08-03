# cardiac_core tutorials

Hands-on notebooks that teach you to run cardiac simulations with `cardiac_core`. They assume you can
run a notebook cell and edit a number — not that you know numpy, torch, or object-oriented Python.

## Start here

| # | Notebook | You'll learn | Runtime |
|---|----------|--------------|---------|
| 1 | [`01_intro.ipynb`](./01_intro.ipynb) | A gentle tour of `cardiac_core`: install & import, the **object landscape** (what a simulation is made of), a single-cell **action potential**, and a monodomain tissue wave rendered as an inline **video** and a **snapshot**. | ~45 s |
| 2.1 | [`02_1_conductances.ipynb`](./02_1_conductances.ipynb) | **Conductances & AP morphology.** A channel-by-channel tour of the six currents that shape the ventricular action potential — `GNa` (upstroke), `Gto` (notch), `PCa` (plateau), `GKr`/`GKs` (repolarisation) and `GK1` (resting potential) — each scaled with the one-keyword `conductances={NAME: factor}` knob and overlaid on the baseline. | ~90 s |
| 2.2 | [`02_2_self_pacing.ipynb`](./02_2_self_pacing.ipynb) | **Self-pacing & the funny current** (advanced). A hiPSC `paci` cell that fires with no electrode (`stim_amplitude=0`): automaticity, diastolic depolarisation, slowing the spontaneous rate by blocking the funny current `g_f`, and silencing the pacemaker entirely in the matured `mhas13` cell. | ~45 s |
| 3.1 | [`03_1_monodomain.ipynb`](./03_1_monodomain.ipynb) | Build a tissue simulation in depth: the **grid** (`cc.Grid`), the **conductivity** (`ConductivityConfig`), the **stimulus** (`cc.Stim` — where you put the electrode), and a **scar** (`set_conductivity(D=0)`); run it on the **monodomain** engine and measure CV (**≈ 58 cm/s**). | ~20 s |
| 3.2 | [`03_2_bidomain.ipynb`](./03_2_bidomain.ipynb) | The **bidomain** engine: the same wave plus the extracellular potential **`phi_e`** (the seed of the ECG). Same `Vm`, nearly the same CV (**≈ 60 cm/s**) — bidomain earns its cost only when the world outside the cells matters. | ~10 s |
| 3.3 | [`03_3_lbm.ipynb`](./03_3_lbm.ipynb) | The **LBM** engine — the same physics by a different numerical route. Same flat wave, a higher CV (**≈ 64 cm/s**, a numerical offset, not an error), the **compare like-to-like** rule, and — the headline — how LBM's **wall rules** model discrete boundary effects the continuum engines can't. | ~5 s |
| 4.1 | [`04_1_pacing_restitution.ipynb`](./04_1_pacing_restitution.ipynb) | Pacing a single cell: **basic cycle length (BCL)**, how **APD shortens as you pace faster**, and the **restitution curve** (APD vs diastolic interval) whose steepness predicts arrhythmia. | ~80 s |
| 4.2 | [`04_2_reentry.ipynb`](./04_2_reentry.ipynb) | Pacing in tissue: the **voltage clamp** (`Stim(mask, clamp=…)`), then inducing a **reentrant rotor** with an **S1–S2 cross-field** protocol on a small sheet — confirmed with a **phase singularity** (\|charge\|≈1). | ~105 s |
| 5.1 | [`05_1_coupling.ipynb`](./05_1_coupling.ipynb) | **Tuning by hand — speed.** How the coupling knobs set conduction velocity: conductivity σ (CV ∝ √σ), the effective diffusivity `D_eff = σ/(χ·Cm)`, the χ/Cm denominators, and the (σ,χ) degeneracy. | ~25 s |
| 5.2 | [`05_2_geometry.ipynb`](./05_2_geometry.ipynb) | **Tuning by hand — shape & launch.** Isotropic circle vs anisotropic **ellipse** (axis ratio √(σ_l/σ_t)); a scar that curves and **blocks** the front (source–sink); stimulus strength/size that decide whether a wave launches at all. | ~45 s |
| 5.3 | [`05_3_cellular.ipynb`](./05_3_cellular.ipynb) | **Tuning by hand — cellular knobs at tissue scale.** Chapter 2's channels on the wave: `GNa` sets speed and, pushed far, **blocks**; `GKr` lengthens APD → longer **wavelength** (`cc.wavelength`). | ~45 s |
| 5.4 | [`05_4_numerical.ipynb`](./05_4_numerical.ipynb) | **Numerical convergence & trust.** Too-coarse **`dx`** gives a wrong, block-looking CV (refine to convergence); **`dt`** is an accuracy knob; compare engines **like-to-like**. | ~70 s |

Chapter 1 is self-contained and is the right place to begin regardless of what you eventually want to
do. It introduces the whole cast — geometry, conductivity, ionic model, stimulus, engine, result — and
the later chapters go deep on each one.

## Running them

Activate the Python environment you installed `cardiac_core` into, then:

```bash
jupyter lab            # or: jupyter notebook
```

Open the notebook and run the cells top to bottom. Each notebook runs in a fresh kernel and defines
everything it needs — you never have to run one notebook before another.

If `jupyter` isn't installed in that environment yet:

```bash
pip install jupyterlab nbformat nbconvert
```

## Notes

- **Figures are committed.** The notebooks ship with their figures rendered inline, so they're
  readable on GitHub without running anything. Regenerating a notebook from its `_build/` source
  empties the outputs; re-running it (or the regression gate below) fills them back in.
- **Regenerating a notebook.** Each notebook is generated from a reviewable Python source under
  [`_build/`](./_build/), which is the file to edit for anything more than a typo — a plain `.py` diff
  is far easier to review than an `.ipynb` diff. Re-run it to rewrite the notebook in place:
  ```bash
  python _build/build_01_intro.py          # run from inside the tutorials/ folder
  ```
  If you edit the `.ipynb` directly in Jupyter instead, fold the change back into the builder or the
  next regeneration will overwrite it.

## The arc

The series runs in five chapters, each built around an experiment rather than an API topic:

1. **Intro** — install, the object landscape, a single cell, and the image/video pipeline.
2. **Conductances & AP morphology** — which ion channel changes what about the action potential (+ a
   self-pacing hiPSC cell and its funny current).
3. **Tissue simulation, in depth** — grid, conductivity, stimulus, scar, and one example per engine
   (monodomain / bidomain / LBM).
4. **Pacing** — single-cell restitution, then a reentrant rotor in tissue.
5. **Tuning by hand** — what each raw knob (σ, D, χ/Cm, anisotropy, scar, conductances, `dx`) does to
   the wave: the capstone, where the earlier chapters converge into tuning intuition.

Chapters 1–2 are the afternoon quick-start; 3–4 open up the tissue; 5 is the payoff.

Because these notebooks execute the real library, an **execute-all regression gate** keeps them from
going stale as `cardiac_core` changes: [`run_all_tutorials.py`](./run_all_tutorials.py) runs every
notebook headless via `nbconvert` and fails on any cell error (all 12 currently pass). Run it directly:

```bash
python run_all_tutorials.py          # all 12 (~9 min of real simulation)
python run_all_tutorials.py 05_       # just one chapter
```

Each notebook is executed on an in-memory copy, so the committed notebooks (figures and all) are left
untouched. The reviewable source for every notebook is its `_build/build_*.py` — **edit the builder,
not the `.ipynb`.**
