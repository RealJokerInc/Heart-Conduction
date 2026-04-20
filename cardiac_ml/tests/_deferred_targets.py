"""Target-import-check exclusion list.

Step 2.5's `test_all_targets_importable` walks `_target_` strings in
composed configs and tries to import each module. Targets listed here
are expected to fail at this phase boundary and are skipped.

At Step 4.1 exit, `surrogate.training.node_step` is removed from this
set to activate the Phase-4 import check (Round-3 MED-6 mechanism).
"""
DEFERRED = {
    # Phase 4 — node_step adapter lands in Step 4.1. Remove at Step 4.1 exit.
    "surrogate.training.node_step",
    # Phase 3 — default_steps lands in Step 3.1. Remove at Step 3.1 exit.
    "cardiac_ml.training.default_steps",
    # Phase 3 — callbacks lands in Step 3.2. Remove at Step 3.2 exit.
    "cardiac_ml.training.callbacks",
    # Phase 3 — trainer lands in Step 3.4. Remove at Step 3.4 exit.
    "cardiac_ml.training.trainer",
}
