"""
Regression and unit tests for the LLM vocabulary-alignment layer.

The archived demo path must remain completely offline: consensus_from_raw()
rebuilds the summary from saved model responses and makes no API calls.

The synthetic tests cover the safeguards introduced in align.py:

- Unicode/case/whitespace normalisation;
- one vote per distinct model;
- duplicate removal within one model;
- backward-compatible relation inference for historical CSVs;
- conservative query-expansion safety;
- preservation of non-numeric cluster identifiers.
"""

from pathlib import Path

import pandas as pd
import pytest

from aws_align.align import (
    alignment_report,
    consensus_details_from_raw,
    consensus_from_raw,
    load_cluster_terms,
)


DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"
RAW = DEMO / "alignment_raw.csv"
TERMS = DEMO / "semantic_topics.csv"


@pytest.fixture(scope="module")
def summary():
    """Rebuild the archived summary without any live model calls."""
    return consensus_from_raw(
        RAW,
        min_confidence=3,
        min_models=2,
    )


def _row(
    *,
    model_key,
    model_label,
    term_a,
    term_b,
    confidence,
    reasoning="Supported relation",
    relation_type="",
    cluster_a=2,
    cluster_b=3,
):
    """Construct one historical-compatible raw alignment row."""
    return {
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "label_a": "Cluster A",
        "label_b": "Cluster B",
        "model_key": model_key,
        "model_label": model_label,
        "term_a": term_a,
        "term_b": term_b,
        "relation_type": relation_type,
        "confidence": confidence,
        "reasoning": reasoning,
        "n_alignable": 1,
        "barrier_assessment": "Moderate vocabulary barrier",
    }


# -----------------------------------------------------------------------------
# Archived demo regression
# -----------------------------------------------------------------------------
def test_demo_files_exist():
    assert RAW.is_file(), f"missing demo file: {RAW}"
    assert TERMS.is_file(), f"missing demo file: {TERMS}"


def test_fifteen_cluster_pairs(summary):
    assert len(summary) == 15
    assert not summary[
        ["cluster_a", "cluster_b"]
    ].duplicated().any()


def test_archived_consensus_split(summary):
    """
    Preserve the historical offline result: 8 pairs with consensus and 7
    without consensus at the default threshold.
    """
    with_consensus = int(
        summary["consensus_alignments"].gt(0).sum()
    )
    without_consensus = int(
        summary["consensus_alignments"].eq(0).sum()
    )

    assert with_consensus == 8
    assert without_consensus == 7


def test_archived_mean_confidence(summary):
    mean_confidence = summary["mean_confidence"].mean()
    assert mean_confidence == pytest.approx(1.96, abs=0.10)


def test_summary_has_new_audit_columns(summary):
    required = {
        "models_attempted",
        "failed_models",
        "total_alignments",
        "strong_alignments",
        "consensus_alignments",
        "query_expansion_safe_alignments",
        "mean_confidence",
        "mean_n_alignable",
        "top_consensus",
        "barrier_assessments",
    }
    assert required <= set(summary.columns)


def test_higher_model_threshold_cannot_increase_consensus():
    summary_two = consensus_from_raw(
        RAW,
        min_confidence=3,
        min_models=2,
    )
    summary_three = consensus_from_raw(
        RAW,
        min_confidence=3,
        min_models=3,
    )

    left = summary_two.set_index(
        ["cluster_a", "cluster_b"]
    )["consensus_alignments"]
    right = summary_three.set_index(
        ["cluster_a", "cluster_b"]
    )["consensus_alignments"]

    right = right.reindex(left.index)
    assert right.le(left).all()


def test_higher_confidence_threshold_cannot_increase_consensus():
    summary_three = consensus_from_raw(
        RAW,
        min_confidence=3,
        min_models=2,
    )
    summary_four = consensus_from_raw(
        RAW,
        min_confidence=4,
        min_models=2,
    )

    left = summary_three.set_index(
        ["cluster_a", "cluster_b"]
    )["consensus_alignments"]
    right = summary_four.set_index(
        ["cluster_a", "cluster_b"]
    )["consensus_alignments"]

    right = right.reindex(left.index)
    assert right.le(left).all()


def test_report_renders_thresholds_and_pair_count(summary):
    report = alignment_report(
        summary,
        min_confidence=3,
        min_models=2,
        top_k=20,
    )

    assert "VOCABULARY ALIGNMENT" in report
    assert "Cluster pairs analysed: 15" in report
    assert "confidence >= 3" in report
    assert ">= 2 models" in report
    assert "query-expansion-safe" in report


def test_load_demo_cluster_terms():
    terms = load_cluster_terms(TERMS, top_k=10)

    assert len(terms) >= 6

    for cluster_id, ranked_terms in terms.items():
        assert isinstance(cluster_id, (int, str))
        assert 1 <= len(ranked_terms) <= 10
        assert all(
            isinstance(term, str) and term.strip()
            for term in ranked_terms
        )


# -----------------------------------------------------------------------------
# Consensus normalisation and deduplication
# -----------------------------------------------------------------------------
def test_case_and_whitespace_variants_reach_consensus():
    raw = pd.DataFrame(
        [
            _row(
                model_key="m1",
                model_label="Model 1",
                term_a="Life cycle assessment",
                term_b="Environmental assessment",
                confidence=4,
            ),
            _row(
                model_key="m2",
                model_label="Model 2",
                term_a="  life   cycle assessment ",
                term_b="environmental assessment",
                confidence=4,
            ),
        ]
    )

    details = consensus_details_from_raw(
        raw,
        min_confidence=3,
        min_models=2,
    )

    assert len(details) == 1
    assert details.iloc[0]["n_models"] == 2
    assert details.iloc[0]["mean_confidence"] == pytest.approx(4.0)


def test_duplicate_rows_from_one_model_count_once():
    raw = pd.DataFrame(
        [
            _row(
                model_key="m1",
                model_label="Model 1",
                term_a="Term A",
                term_b="Term B",
                confidence=3,
            ),
            _row(
                model_key="m1",
                model_label="Model 1",
                term_a="term a",
                term_b="term b",
                confidence=5,
            ),
            _row(
                model_key="m2",
                model_label="Model 2",
                term_a="TERM A",
                term_b="TERM B",
                confidence=4,
            ),
        ]
    )

    details = consensus_details_from_raw(
        raw,
        min_confidence=3,
        min_models=2,
    )

    assert len(details) == 1
    assert details.iloc[0]["n_models"] == 2
    assert details.iloc[0]["mean_confidence"] == pytest.approx(4.5)


def test_repetition_by_one_model_is_not_consensus():
    raw = pd.DataFrame(
        [
            _row(
                model_key="m1",
                model_label="Model 1",
                term_a="Term A",
                term_b="Term B",
                confidence=5,
            ),
            _row(
                model_key="m1",
                model_label="Model 1",
                term_a="term a",
                term_b="term b",
                confidence=5,
            ),
        ]
    )

    details = consensus_details_from_raw(
        raw,
        min_confidence=3,
        min_models=2,
    )

    assert details.empty


# -----------------------------------------------------------------------------
# Relation types and query-expansion safety
# -----------------------------------------------------------------------------
def test_historical_confidence_four_infers_near_synonym():
    raw = pd.DataFrame(
        [
            _row(
                model_key="m1",
                model_label="Model 1",
                term_a="Term A",
                term_b="Term B",
                confidence=4,
            ),
            _row(
                model_key="m2",
                model_label="Model 2",
                term_a="Term A",
                term_b="Term B",
                confidence=4,
            ),
        ]
    )

    details = consensus_details_from_raw(
        raw,
        min_confidence=3,
        min_models=2,
    )

    assert len(details) == 1
    assert details.iloc[0]["relation_type"] == "near_synonym"
    assert bool(details.iloc[0]["query_expansion_safe"])


def test_functional_equivalent_is_not_query_expansion_safe():
    raw = pd.DataFrame(
        [
            _row(
                model_key="m1",
                model_label="Model 1",
                term_a="Term A",
                term_b="Term B",
                relation_type="functional_equivalent",
                confidence=5,
            ),
            _row(
                model_key="m2",
                model_label="Model 2",
                term_a="Term A",
                term_b="Term B",
                relation_type="functional_equivalent",
                confidence=5,
            ),
        ]
    )

    details = consensus_details_from_raw(
        raw,
        min_confidence=3,
        min_models=2,
    )

    assert len(details) == 1
    assert (
        details.iloc[0]["relation_type"]
        == "functional_equivalent"
    )
    assert not bool(details.iloc[0]["query_expansion_safe"])


def test_one_safe_model_is_not_enough_for_safe_consensus():
    raw = pd.DataFrame(
        [
            _row(
                model_key="m1",
                model_label="Model 1",
                term_a="Term A",
                term_b="Term B",
                relation_type="near_synonym",
                confidence=4,
            ),
            _row(
                model_key="m2",
                model_label="Model 2",
                term_a="Term A",
                term_b="Term B",
                relation_type="functional_equivalent",
                confidence=4,
            ),
        ]
    )

    details = consensus_details_from_raw(
        raw,
        min_confidence=3,
        min_models=2,
    )

    assert len(details) == 1
    assert details.iloc[0]["n_models"] == 2
    assert details.iloc[0]["safe_models"] == 1
    assert not bool(details.iloc[0]["query_expansion_safe"])


# -----------------------------------------------------------------------------
# Historical failures and input validation
# -----------------------------------------------------------------------------
def test_failed_historical_row_does_not_count_as_alignment():
    raw = pd.DataFrame(
        [
            _row(
                model_key="m1",
                model_label="Model 1",
                term_a="Term A",
                term_b="Term B",
                confidence=4,
            ),
            _row(
                model_key="m2",
                model_label="Model 2",
                term_a="",
                term_b="",
                confidence="",
                reasoning="FAILED",
            ),
        ]
    )

    result = consensus_from_raw(
        raw,
        min_confidence=3,
        min_models=2,
    )

    assert len(result) == 1
    assert result.iloc[0]["total_alignments"] == 1
    assert result.iloc[0]["consensus_alignments"] == 0


@pytest.mark.parametrize("minimum", [0, 6])
def test_invalid_min_confidence_rejected(minimum):
    raw = pd.DataFrame(
        [
            _row(
                model_key="m1",
                model_label="Model 1",
                term_a="Term A",
                term_b="Term B",
                confidence=4,
            )
        ]
    )

    with pytest.raises(ValueError, match="min_confidence"):
        consensus_from_raw(
            raw,
            min_confidence=minimum,
            min_models=1,
        )


def test_invalid_min_models_rejected():
    raw = pd.DataFrame(
        [
            _row(
                model_key="m1",
                model_label="Model 1",
                term_a="Term A",
                term_b="Term B",
                confidence=4,
            )
        ]
    )

    with pytest.raises(ValueError, match="min_models"):
        consensus_from_raw(
            raw,
            min_confidence=3,
            min_models=0,
        )


def test_missing_required_raw_column_rejected():
    raw = pd.DataFrame(
        {
            "cluster_a": [2],
            "cluster_b": [3],
            "model_key": ["m1"],
            "term_a": ["Term A"],
            # term_b intentionally missing
            "confidence": [4],
        }
    )

    with pytest.raises(ValueError, match="missing columns"):
        consensus_from_raw(raw)


# -----------------------------------------------------------------------------
# Term loading
# -----------------------------------------------------------------------------
def test_cluster_term_loader_preserves_string_ids_and_rank_order(tmp_path):
    topics = tmp_path / "semantic_topics.csv"

    pd.DataFrame(
        {
            "cluster": ["C2", "3"],
            "top_terms": [
                "Alpha; alpha; Beta; Gamma",
                "First, Second, Third",
            ],
        }
    ).to_csv(topics, index=False)

    terms = load_cluster_terms(topics, top_k=2)

    assert set(terms) == {"C2", 3}
    assert terms["C2"] == ["Alpha", "Beta"]
    assert terms[3] == ["First", "Second"]
