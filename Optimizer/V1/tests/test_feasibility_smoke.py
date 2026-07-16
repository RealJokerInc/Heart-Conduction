"""
Smoke test for feasibility_map (PLAN Step 2.1 / P1a).

Runs a 1×1 grid (no media I/O) and checks the map returns the gate structure:
per-dx feasibility + a conductance-only boolean + per-point rows.
"""

import pytest


@pytest.mark.slow
def test_map_runs():
    from feasibility_map import feasibility_map

    res = feasibility_map("hipsc", gNa_grid=(0.5,), dx_mm_grid=(0.1,),
                          n_beats_cell=2, save_media=False, verbose=False)

    assert 'rows' in res and len(res['rows']) == 1
    assert 'any_feasible_by_dx' in res and set(res['any_feasible_by_dx']) == {0.1}
    assert isinstance(res['conductance_only_feasible'], bool)

    r = res['rows'][0]
    for key in ('dx_mm', 'gNa', 'D', 'cv', 'rstar_over_dx', 'dvdt',
                'cv_ok', 'resolved', 'feasible'):
        assert key in r
    assert isinstance(r['feasible'], bool)
