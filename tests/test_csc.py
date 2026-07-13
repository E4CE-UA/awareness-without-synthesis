"""
Regression and invariant tests for the canonical CSC implementation.

Canonical plastic-recycling demo values:

    S_cross       ~= 0.617
    D_bar         ~= 0.976
    D_bar_w       ~= 0.969
    fragmentation ~= 0.598
    CSC           ~= 0.402

The canonical identity is:

    fragmentation = S_cross * D_bar_w
    CSC = 1 - fragmentation

D_bar is retained as an unweighted descriptive statistic, but it does not
enter the CSC formula.

Run with:

    pytest tests/test_csc.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aws_align import (
    compute_csc,
    load_divergence,
    load_insularity,
)
from aws_align.csc import (
    compute_s_cross,
    compute_weighted_divergence,
    empirical_p_value,
)


DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"
INSULARITY = DEMO / "citation_cluster_insularity.csv"
DIVERGENCE = DEMO / "rbo_fragmentation.csv"


@pytest.fixture(scope="module")
def demo_inputs():
    assert INSULARITY.is_file(), f"missing demo file: {INSULARITY}"
    assert DIVERGENCE.is_file(), f"missing demo file: {DIVERGENCE}"

    insularity = load_insularity(INSULARITY)
    divergence = load_divergence(DIVERGENCE)
    return insularity, divergence


@pytest.fixture(scope="module")
def demo_result(demo_inputs):
    insularity, divergence = demo_inputs
    return compute_csc(
        insularity,
        divergence,
        corpus="Plastic recycling",
        n_nulls=0,
    )


@pytest.fixture(scope="module")
def synthetic_cluster_sizes(demo_inputs):
    """
    Positive deterministic sizes aligned with the demo cluster identifiers.

    These sizes test the auxiliary null machinery itself. They are not used to
    reproduce a paper-level null result.
    """
    insularity, _ = demo_inputs
    values = np.arange(
        100,
        100 + 10 * len(insularity),
        10,
        dtype=int,
    )
    return pd.Series(
        values,
        index=insularity["cluster"].astype(str),
        name="n_documents",
    )


# -----------------------------------------------------------------------------
# Canonical plastic-recycling regression values
# -----------------------------------------------------------------------------
def test_s_cross_canonical(demo_inputs):
    insularity, _ = demo_inputs
    result = compute_s_cross(insularity)

    assert result["E_within_corpus"] == 1951
    assert result["E_intra_cluster"] == 747
    assert result["E_cross_cluster"] == 1204
    assert result["S_cross"] == pytest.approx(
        0.617,
        abs=0.005,
    )


def test_divergence_canonical(demo_inputs):
    insularity, divergence = demo_inputs

    D_bar, D_bar_w, pairwise = compute_weighted_divergence(
        insularity,
        divergence,
    )

    assert D_bar == pytest.approx(0.976, abs=0.005)
    assert D_bar_w == pytest.approx(0.969, abs=0.005)
    assert len(pairwise) == 15


def test_csc_canonical(demo_result):
    assert demo_result.corpus == "Plastic recycling"
    assert demo_result.n_clusters == 6
    assert demo_result.n_pairs == 15

    assert demo_result.S_cross == pytest.approx(
        0.617,
        abs=0.005,
    )
    assert demo_result.D_bar == pytest.approx(
        0.976,
        abs=0.005,
    )
    assert demo_result.D_bar_w == pytest.approx(
        0.969,
        abs=0.005,
    )
    assert demo_result.fragmentation == pytest.approx(
        0.598,
        abs=0.005,
    )
    assert demo_result.csc == pytest.approx(
        0.402,
        abs=0.005,
    )


# -----------------------------------------------------------------------------
# Canonical identities and pairwise audit table
# -----------------------------------------------------------------------------
def test_csc_uses_weighted_divergence(demo_result):
    expected_fragmentation = (
        demo_result.S_cross
        * demo_result.D_bar_w
    )

    assert demo_result.fragmentation == pytest.approx(
        expected_fragmentation,
        abs=1e-12,
    )
    assert demo_result.csc == pytest.approx(
        1.0 - expected_fragmentation,
        abs=1e-12,
    )


def test_unweighted_d_bar_is_not_substituted_into_csc(demo_result):
    """
    Protect against regression to the obsolete formula using D_bar.
    """
    obsolete_value = 1.0 - (
        demo_result.S_cross
        * demo_result.D_bar
    )

    assert demo_result.csc != pytest.approx(
        obsolete_value,
        abs=1e-6,
    )


def test_pair_weights_sum_to_one(demo_result):
    pairwise = demo_result.pairwise

    assert pairwise is not None
    assert pairwise["w_ij"].sum() == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_weighted_pair_contributions_sum_to_d_bar_w(demo_result):
    pairwise = demo_result.pairwise

    assert pairwise is not None
    assert pairwise["weighted_D_ij"].sum() == pytest.approx(
        demo_result.D_bar_w,
        abs=1e-12,
    )


def test_fragmentation_contributions_sum_to_fragmentation(demo_result):
    pairwise = demo_result.pairwise

    assert pairwise is not None
    assert pairwise[
        "fragmentation_contribution"
    ].sum() == pytest.approx(
        demo_result.fragmentation,
        abs=1e-12,
    )


def test_allocated_cross_flow_sums_to_s_cross(demo_result):
    pairwise = demo_result.pairwise

    assert pairwise is not None
    assert pairwise["S_ij_allocated"].sum() == pytest.approx(
        demo_result.S_cross,
        abs=1e-12,
    )


def test_pairwise_table_has_complete_unique_pairs(demo_result):
    pairwise = demo_result.pairwise

    assert pairwise is not None
    assert len(pairwise) == (
        demo_result.n_clusters
        * (demo_result.n_clusters - 1)
        // 2
    )
    assert not pairwise[
        ["cluster_i", "cluster_j"]
    ].duplicated().any()
    assert (
        pairwise["cluster_i"]
        != pairwise["cluster_j"]
    ).all()


# -----------------------------------------------------------------------------
# Value ranges
# -----------------------------------------------------------------------------
def test_scores_are_in_unit_interval(demo_result):
    assert 0.0 <= demo_result.S_cross <= 1.0
    assert 0.0 <= demo_result.D_bar <= 1.0
    assert 0.0 <= demo_result.D_bar_w <= 1.0
    assert 0.0 <= demo_result.fragmentation <= 1.0
    assert 0.0 <= demo_result.csc <= 1.0


def test_pairwise_values_are_valid(demo_result):
    pairwise = demo_result.pairwise

    assert pairwise is not None
    assert pairwise["D_ij"].between(0.0, 1.0).all()
    assert pairwise["w_ij"].between(0.0, 1.0).all()
    assert pairwise[
        "fragmentation_contribution"
    ].between(0.0, 1.0).all()


# -----------------------------------------------------------------------------
# Auxiliary size-mixing null for S_cross only
# -----------------------------------------------------------------------------
def test_null_requires_cluster_sizes(demo_inputs):
    insularity, divergence = demo_inputs

    with pytest.raises(
        ValueError,
        match="cluster_sizes is required",
    ):
        compute_csc(
            insularity,
            divergence,
            n_nulls=20,
            seed=42,
        )


def test_auxiliary_scross_null_runs(
    demo_inputs,
    synthetic_cluster_sizes,
):
    insularity, divergence = demo_inputs

    result = compute_csc(
        insularity,
        divergence,
        cluster_sizes=synthetic_cluster_sizes,
        n_nulls=200,
        seed=42,
    )

    assert result.n_nulls == 200
    assert result.null_scross is not None
    assert len(result.null_scross) == 200

    assert result.null_type == (
        "size-mixing null for S_cross only"
    )
    assert result.null_alternative == "two-sided"

    assert np.isfinite(result.mu_null_scross)
    assert result.sigma_null_scross > 0
    assert np.isfinite(result.z_scross)

    assert (
        1.0 / (result.n_nulls + 1)
        <= result.p_empirical_scross
        <= 1.0
    )


def test_auxiliary_null_is_reproducible(
    demo_inputs,
    synthetic_cluster_sizes,
):
    insularity, divergence = demo_inputs

    first = compute_csc(
        insularity,
        divergence,
        cluster_sizes=synthetic_cluster_sizes,
        n_nulls=200,
        seed=42,
    )
    second = compute_csc(
        insularity,
        divergence,
        cluster_sizes=synthetic_cluster_sizes,
        n_nulls=200,
        seed=42,
    )

    np.testing.assert_array_equal(
        first.null_scross,
        second.null_scross,
    )
    assert first.mu_null_scross == pytest.approx(
        second.mu_null_scross,
        abs=1e-12,
    )
    assert first.sigma_null_scross == pytest.approx(
        second.sigma_null_scross,
        abs=1e-12,
    )
    assert first.p_empirical_scross == pytest.approx(
        second.p_empirical_scross,
        abs=1e-12,
    )


def test_empirical_p_value_uses_plus_one_correction():
    null = np.array([0.10, 0.20, 0.30])

    p_value = empirical_p_value(
        null,
        observed=0.40,
        alternative="greater",
    )

    # No null value is >= 0.40:
    # p = (0 + 1) / (3 + 1) = 0.25
    assert p_value == pytest.approx(0.25)


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------
def test_incomplete_divergence_table_rejected(demo_inputs):
    insularity, divergence = demo_inputs
    incomplete = divergence.iloc[:-1].copy()

    with pytest.raises(ValueError, match="Expected"):
        compute_csc(
            insularity,
            incomplete,
            n_nulls=0,
        )


def test_duplicate_unordered_pair_rejected(demo_inputs):
    insularity, divergence = demo_inputs

    duplicate = divergence.iloc[[0]].copy()
    duplicate = duplicate.rename(
        columns={
            "cluster_a": "cluster_b",
            "cluster_b": "cluster_a",
        }
    )
    duplicated = pd.concat(
        [divergence, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate unordered cluster pairs",
    ):
        compute_csc(
            insularity,
            duplicated,
            n_nulls=0,
        )


def test_divergence_outside_unit_interval_rejected(demo_inputs):
    insularity, divergence = demo_inputs
    invalid = divergence.copy()
    invalid.loc[invalid.index[0], "D_ij"] = 1.5

    with pytest.raises(ValueError, match=r"D_ij must lie in \[0, 1\]"):
        compute_csc(
            insularity,
            invalid,
            n_nulls=0,
        )
