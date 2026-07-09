"""
Reproduction tests for the canonical aws-align numbers.

    S_cross = 0.617      cross-cluster citation coupling
    D_bar   = 0.976      mean pairwise vocabulary divergence
    CSC     = 0.402      = 1 - (S_cross * D_bar)

Run with:  pytest -v
"""
from pathlib import Path

import pytest

from aws_align import compute_csc, load_divergence, load_insularity
from aws_align.csc import compute_d_bar, compute_s_cross

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"


@pytest.fixture(scope="module")
def demo_inputs():
    ins = load_insularity(DEMO / "citation_cluster_insularity.csv")
    div = load_divergence(DEMO / "rbo_fragmentation.csv")
    return ins, div


# --------------------------------------------------------------------------- 
# Canonical plastics/recycling numbers
# ---------------------------------------------------------------------------
def test_s_cross_canonical(demo_inputs):
    ins, _ = demo_inputs
    out = compute_s_cross(ins)
    assert out["S_cross"] == pytest.approx(0.617, abs=0.005), f"S_cross={out['S_cross']}"


def test_d_bar_canonical(demo_inputs):
    _, div = demo_inputs
    d_bar = compute_d_bar(div)
    assert d_bar == pytest.approx(0.976, abs=0.005), f"D_bar={d_bar}"


def test_csc_canonical(demo_inputs):
    ins, div = demo_inputs
    res = compute_csc(ins, div, corpus="plastics", n_nulls=0)
    assert res.csc == pytest.approx(0.402, abs=0.005), f"CSC={res.csc}"
    assert res.n_clusters == 6


def test_csc_identity(demo_inputs):
    """CSC must equal 1 - (S_cross * D_bar) exactly."""
    ins, div = demo_inputs
    res = compute_csc(ins, div, n_nulls=0)
    assert res.csc == pytest.approx(1.0 - res.S_cross * res.D_bar, abs=1e-9)
    assert res.aws_score == pytest.approx(res.S_cross * res.D_bar, abs=1e-9)


# --------------------------------------------------------------------------- 
# Null model
# ---------------------------------------------------------------------------
def test_null_model_runs(demo_inputs):
    ins, div = demo_inputs
    res = compute_csc(ins, div, n_nulls=200, seed=42)
    # size-proportional null yields non-zero variance and a finite z-score
    assert res.sigma_null is not None and res.sigma_null > 0
    assert res.z_scross is not None
    # observed coupling is far from the size-proportional null
    assert abs(res.z_scross) > 5


def test_null_reproducible(demo_inputs):
    ins, div = demo_inputs
    r1 = compute_csc(ins, div, n_nulls=200, seed=42)
    r2 = compute_csc(ins, div, n_nulls=200, seed=42)
    assert r1.mu_null == pytest.approx(r2.mu_null, abs=1e-12)


# --------------------------------------------------------------------------- 
# Value ranges / sanity
# ---------------------------------------------------------------------------
def test_csc_in_unit_interval(demo_inputs):
    ins, div = demo_inputs
    res = compute_csc(ins, div, n_nulls=0)
    assert 0.0 <= res.csc <= 1.0
    assert 0.0 <= res.S_cross <= 1.0
    assert 0.0 <= res.D_bar <= 1.0
