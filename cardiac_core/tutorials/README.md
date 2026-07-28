# cardiac_core tutorials

Hands-on notebooks that teach you to run cardiac simulations with `cardiac_core`. They assume you can
run a notebook cell and edit a number — not that you know numpy, torch, or object-oriented Python.

## Start here

| # | Notebook | You'll learn | Runtime |
|---|----------|--------------|---------|
| 1 | [`01_intro.ipynb`](./01_intro.ipynb) | A gentle tour of `cardiac_core`: install & import, the **object landscape** (what a simulation is made of), a single-cell **action potential**, and a monodomain tissue wave rendered as an inline **video** and a **snapshot**. | ~45 s |

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

- **Outputs are not committed.** The notebook ships with empty output cells, so you generate every
  number and figure yourself when you run it.
- **Regenerating a notebook.** Each notebook is generated from a reviewable Python source under
  [`_build/`](./_build/), which is the file to edit for anything more than a typo — a plain `.py` diff
  is far easier to review than an `.ipynb` diff. Re-run it to rewrite the notebook in place:
  ```bash
  python cardiac_core/tutorials/_build/build_01_intro.py
  ```
  If you edit the `.ipynb` directly in Jupyter instead, fold the change back into the builder or the
  next regeneration will overwrite it.

## More lessons

Further notebooks are planned, each built around an experiment rather than an API topic: applying a
drug, pacing to measure restitution, carving a scar until conduction blocks, fibers and anisotropy,
voltage clamp, and the field maps. Because these notebooks execute the real library, a regression
gate that runs every one of them headless (`nbconvert --execute`) is planned alongside them, so they
cannot go stale as the library changes.
