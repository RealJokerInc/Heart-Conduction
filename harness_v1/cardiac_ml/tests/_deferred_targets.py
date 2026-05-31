"""Target-import-check exclusion list.

Step 2.5's `test_all_targets_importable` walks `_target_` strings in
composed configs and tries to import each module. Targets listed here
are expected to fail at this phase boundary and are skipped.

At Step 4.1 exit, `surrogate.training.node_step` is removed from this
set to activate the Phase-4 import check (Round-3 MED-6 mechanism).
"""
DEFERRED = {
    # Phase 3 — default_steps landed in Step 3.1 (removed from DEFERRED).
    # Phase 3 — callbacks landed in Step 3.2 (removed from DEFERRED).
    # Phase 3 — trainer landed in Step 3.4 (removed from DEFERRED).
    # Phase 4 — node_step landed in Step 4.1 (removed from DEFERRED).
}
