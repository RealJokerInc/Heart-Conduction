# Vendored engines — source & re-vendor recipe

`cardiac_core` is the **centralized home** for the three simulation engines (decision 2026-06-25).
The engine code under `cardiac_core/_monodomain/`, `_bidomain/`, `_lbm/` was **copied** from the
original engine folders, which remain on disk **frozen** (legacy; future development happens here).

## Provenance (vendored 2026-06-25, tag `pre-consolidation-vendoring`)

| cardiac_core package | Copied from | Notes |
|---|---|---|
| `_monodomain/` | `Monodomain/Engine_V5.5/cardiac_sim/{simulation,utils,tissue}` | Cm-correct V5.5. 8 cross-imports rewritten → `cardiac_core.{ionic,mesh,stimulus}`. |
| `_bidomain/` | `Bidomain/Engine_V1/cardiac_sim/{simulation,utils,tissue}` | dead `simulation/lbm/` dropped. 9 cross-imports rewritten → `cardiac_core.{ionic,mesh}` (BidomainConductivity stays local under `_bidomain/tissue`). |
| `_lbm/` | `LBM/Engine_V1/src` | verbatim; 0 rewrites (fully relative; receives ionic model as an object). |
| `ionic/` | (already vendored, Phase-1 canonical copy) | shared. |
| `mesh/` | mono `tissue_builder/mesh` + bidomain `boundary.py`; `structured.py` = **bidomain superset** | union `__init__`. |
| `stimulus/` | mono `tissue_builder/stimulus` + **bidomain `protocol.py`** (canonical `+=` accumulate) | `=`/`+=` differ only for OVERLAPPING stimuli. |

**Naming:** solver packages are underscore-prefixed (`_monodomain` etc.) so they do not shadow the
public `monodomain()`/`bidomain()`/`lbm()` factory functions. Users never import `_*` directly —
build via the factories or `simulate(engine=...)`.

**The one intentional cross-reference:** `tests/_live_cv_gate_driver.py` subprocess-drives the
ORIGINAL V5.5 cable harness (`Monodomain/Engine_V5.5/test_phase10_cm_scaling.py`) to validate the
`ConductivityConfig` firewall against the reference CV. It is excluded from the self-containment
guard. Repoint it at `cardiac_core._monodomain` if the original is ever removed.

## Integrity

`tests/test_integrity.py` pins each engine's pre-vendor output as a bit-identical golden (atol=0) and
hashes the original source trees (`_integrity/engine_src_sha.json`) — proving the copy is
behavior-preserving and the originals stay untouched. `tests/test_self_contained.py` is the durable
guard (no cross-folder imports, no `_prepare_engine`).

## Re-vendor recipe (if an original is ever updated and you want to pull it forward)

```bash
# 1. copy the solver subtree (example: monodomain)
rm -rf cardiac_core/_monodomain/simulation
cp -r Monodomain/Engine_V5.5/cardiac_sim/simulation cardiac_core/_monodomain/simulation
# 2. rewrite the solver→shared cross-imports (\b-anchored so ionic_time_stepping stays relative):
#    from <dots>ionic            -> from cardiac_core.ionic
#    from <dots>tissue_builder.mesh     -> from cardiac_core.mesh
#    from <dots>tissue_builder.stimulus -> from cardiac_core.stimulus
#    from <dots>tissue_builder.tissue   -> from cardiac_core._<engine>.tissue
# 3. re-capture goldens + re-run the suite:
python cardiac_core/tests/_integrity/make_goldens.py
pytest cardiac_core/tests/
```
