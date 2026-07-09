"""
Reproduction tests for the LLM vocabulary-alignment consensus table.

The offline path (consensus_from_raw) must reproduce the alignment summary from
the bundled raw responses without any API calls. The demo raw file holds 15
cluster pairs x 3 models.
"""
from pathlib import Path

import pytest

from aws_align.align import (alignment_report, consensus_from_raw,
                             load_cluster_terms)

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"
RAW = DEMO / "alignment_raw.csv"


@pytest.fixture(scope="module")
def summary():
    return consensus_from_raw(RAW)


def test_fifteen_pairs(summary):
    assert len(summary) == 15


def test_consensus_split(summary):
    """8 pairs reach >=2-model consensus, 7 do not (canonical)."""
    with_cons = int((summary["consensus_alignments"] > 0).sum())
    no_cons = int((summary["consensus_alignments"] == 0).sum())
    assert with_cons == 8
    assert no_cons == 7


def test_low_mean_confidence(summary):
    """Mean confidence is low (~1.96/5): few genuine bridges — the point."""
    assert summary["mean_confidence"].mean() == pytest.approx(1.96, abs=0.1)


def test_min_models_threshold():
    """Raising min_models to 3 can only reduce consensus counts."""
    s2 = consensus_from_raw(RAW, min_models=2)
    s3 = consensus_from_raw(RAW, min_models=3)
    assert (s3["consensus_alignments"] <= s2["consensus_alignments"]).all()


def test_report_renders(summary):
    txt = alignment_report(summary)
    assert "VOCABULARY ALIGNMENT" in txt
    assert "15" in txt


def test_load_cluster_terms():
    terms = load_cluster_terms(DEMO / "semantic_topics.csv", top_k=10)
    assert len(terms) >= 6
    for cid, tlist in terms.items():
        assert isinstance(cid, int)
        assert len(tlist) <= 10
