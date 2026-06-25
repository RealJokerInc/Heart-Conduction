"""Tests for cardiac_core.viz — standardized result figures/videos."""

import os

import pytest

import cardiac_core as cc
from cardiac_core import propagation_video, apd_map_figure, activation_isochrones


@pytest.fixture(scope="module")
def small_result():
    """A tiny propagating monodomain run (1.0 × 0.25 cm, 20 ms) reused by all viz tests."""
    g = cc.Grid(40, 10, 0.025)
    cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    stim = {"region": lambda x, y: x < 0.05, "start_time": 1.0, "duration": 2.0, "amplitude": -80.0}
    sim = cc.monodomain(g, "ttp06", cond, stim)
    return sim.run(t_end=20.0, save_every=1.0)


def _ok(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def test_propagation_video(small_result):
    p = propagation_video(small_result, "test-viz", bulk=True, fps=10)
    assert _ok(p), p
    assert p.endswith(".mp4") or p.endswith(".gif")
    assert "_sim_outputs" in p  # bulk → gitignored

def test_apd_map_figure(small_result):
    p = apd_map_figure(small_result, "test-viz", bulk=True)
    assert _ok(p) and p.endswith(".png"), p


def test_activation_isochrones(small_result):
    p = activation_isochrones(small_result, "test-viz", bulk=True)
    assert _ok(p) and p.endswith(".png"), p
