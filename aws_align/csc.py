"""

Canonical implementation of the Cross-Subdomain Coherence score (CSC)
used in the Awareness Without Synthesis (AWS) framework.

The diagnostic combines two complementary quantities defined for a
specific corpus partition:

1. S_cross
   Fraction of within-corpus citation edges that cross cluster boundaries.

2. D_bar_w
   Vocabulary divergence averaged across cluster pairs using exposure
   weights derived from each cluster's participation in cross-cluster
   citation flow.

Definitions
-----------
Let e_i be the number of cross-cluster citation edges originating from
cluster i. For each unordered cluster pair (i, j):

    raw_weight_ij = e_i * e_j
    w_ij          = raw_weight_ij / sum_{p<q}(raw_weight_pq)
    D_bar_w       = sum_{i<j}(w_ij * D_ij)

where D_ij is a pairwise vocabulary divergence in [0, 1], such as
1 - Rank-Biased Overlap (RBO).

The corpus-level scores are:

    fragmentation = S_cross * D_bar_w
    CSC           = 1 - fragmentation

Higher CSC indicates stronger coherence between citation structure and
vocabulary. Lower CSC indicates the Awareness Without Synthesis regime.

Important
---------
- S_cross is partition-dependent. It is not invariant to k.
- D_bar is retained as an unweighted descriptive statistic but does not
  enter the canonical CSC formula.
- The optional size-mixing null implemented here applies only to S_cross.
  It is not a null distribution for the full CSC. A full label-permutation
  null must recompute cluster vocabularies, D_ij, D_bar_w, and CSC from
  document-level data after every label permutation.

Expected input schemas
----------------------
Insularity CSV:
    cluster,total_edges,intra_edges

Divergence CSV:
    cluster_a,cluster_b,D_ij

Optional cluster-size CSV:
    cluster,n_documents

Example
-------
python aws_csc.py \
    --insularity citation_insularity.csv \
    --divergence rbo_divergence_long.csv \
    --corpus "Plastic recycling" \
    --output-dir results/csc

Optional auxiliary structural null:
python aws_csc.py \
    --insularity citation_insularity.csv \
    --divergence rbo_divergence_long.csv \
    --cluster-sizes cluster_sizes.csv \
    --n-nulls 1000 \
    --null-alternative two-sided \
    --output-dir results/csc
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd


Alternative = Literal["greater", "less", "two-sided"]


@dataclass
class CSCResult:
    """Container for one canonical CSC diagnostic run."""

    corpus: str
    n_clusters: int
    n_pairs: int
    E_within_corpus: int
    E_intra_cluster: int
    E_cross_cluster: int
    S_cross: float
    D_bar: float
    D_bar_w: float
    fragmentation: float
    csc: float

    # Optional auxiliary S_cross null.
    null_type: Optional[str] = None
    null_alternative: Optional[str] = None
    mu_null_scross: float = float("nan")
    sigma_null_scross: float = float("nan")
    z_scross: float = float("nan")
    p_empirical_scross: float = float("nan")
    n_nulls: int = 0

    pairwise: Optional[pd.DataFrame] = field(default=None, repr=False)
    null_scross: Optional[np.ndarray] = field(default=None, repr=False)

    def summary_row(self) -> dict:
        """Return a flat, serialization-friendly summary."""
        return {
            "corpus": self.corpus,
            "n_clusters": self.n_clusters,
            "n_pairs": self.n_pairs,
            "E_within_corpus": self.E_within_corpus,
            "E_intra_cluster": self.E_intra_cluster,
            "E_cross_cluster": self.E_cross_cluster,
            "S_cross": self.S_cross,
            "D_bar": self.D_bar,
            "D_bar_w": self.D_bar_w,
            "fragmentation_1_minus_CSC": self.fragmentation,
            "CSC": self.csc,
            "null_type": self.null_type,
            "null_alternative": self.null_alternative,
            "mu_null_scross": self.mu_null_scross,
            "sigma_null_scross": self.sigma_null_scross,
            "z_scross": self.z_scross,
            "p_empirical_scross": self.p_empirical_scross,
            "n_nulls": self.n_nulls,
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.summary_row()])

    def __str__(self) -> str:
        lines = [
            f"CSC diagnostic — {self.corpus}",
            f"  clusters             = {self.n_clusters}",
            f"  unordered pairs      = {self.n_pairs}",
            f"  within-corpus edges  = {self.E_within_corpus:,}",
            f"  intra-cluster edges  = {self.E_intra_cluster:,}",
            f"  cross-cluster edges  = {self.E_cross_cluster:,}",
            f"  S_cross              = {self.S_cross:.6f}",
            f"  D_bar                = {self.D_bar:.6f}",
            f"  D_bar_w              = {self.D_bar_w:.6f}",
            f"  fragmentation        = {self.fragmentation:.6f}",
            f"  CSC                   = {self.csc:.6f}",
        ]
        if self.n_nulls > 0:
            z_text = (
                "inf"
                if math.isinf(self.z_scross) and self.z_scross > 0
                else "-inf"
                if math.isinf(self.z_scross)
                else f"{self.z_scross:.3f}"
            )
            lines.extend(
                [
                    "  auxiliary S_cross null:",
                    f"    type                = {self.null_type}",
                    f"    alternative         = {self.null_alternative}",
                    f"    null mean           = {self.mu_null_scross:.6f}",
                    f"    null SD             = {self.sigma_null_scross:.6f}",
                    f"    z                   = {z_text}",
                    f"    empirical p         = {self.p_empirical_scross:.6f}",
                    f"    B                   = {self.n_nulls}",
                ]
            )
        return "\n".join(lines)


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    frame_name: str,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: {sorted(missing)}"
        )


def _normalise_cluster_ids(series: pd.Series) -> pd.Series:
    """Normalise cluster identifiers without changing their semantic labels."""
    return series.astype(str).str.strip()


def validate_insularity(insularity: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and standardise a cluster-level citation table.

    Required columns:
        cluster, total_edges, intra_edges

    Interpretation:
        total_edges = all within-corpus citation edges originating from cluster i
        intra_edges = within-corpus citation edges from i to the same cluster
        cross_edges = total_edges - intra_edges
    """
    required = {"cluster", "total_edges", "intra_edges"}
    _require_columns(insularity, required, "insularity")

    ins = insularity.copy()
    ins["cluster"] = _normalise_cluster_ids(ins["cluster"])

    if ins["cluster"].duplicated().any():
        duplicates = sorted(ins.loc[ins["cluster"].duplicated(), "cluster"].unique())
        raise ValueError(f"Duplicate cluster rows in insularity: {duplicates}")

    for column in ("total_edges", "intra_edges"):
        ins[column] = pd.to_numeric(ins[column], errors="raise")
        if not np.all(np.isfinite(ins[column])):
            raise ValueError(f"{column} contains non-finite values")
        if (ins[column] < 0).any():
            raise ValueError(f"{column} cannot contain negative values")
        if not np.allclose(ins[column], np.round(ins[column])):
            raise ValueError(f"{column} must contain integer edge counts")
        ins[column] = ins[column].astype(np.int64)

    if (ins["intra_edges"] > ins["total_edges"]).any():
        bad = ins.loc[
            ins["intra_edges"] > ins["total_edges"],
            ["cluster", "total_edges", "intra_edges"],
        ]
        raise ValueError(
            "intra_edges cannot exceed total_edges:\n"
            + bad.to_string(index=False)
        )

    ins["cross_edges"] = ins["total_edges"] - ins["intra_edges"]

    if len(ins) < 2:
        raise ValueError("At least two clusters are required")

    if int(ins["total_edges"].sum()) <= 0:
        raise ValueError("The within-corpus citation graph has no edges")

    return ins.sort_values("cluster").reset_index(drop=True)


def validate_divergence(
    divergence: pd.DataFrame,
    cluster_ids: set[str],
    require_complete_pairs: bool = True,
) -> pd.DataFrame:
    """
    Validate a long-form pairwise divergence table.

    Required columns:
        cluster_a, cluster_b, D_ij

    Each unordered pair must occur once. Self-pairs are forbidden.
    """
    required = {"cluster_a", "cluster_b", "D_ij"}
    _require_columns(divergence, required, "divergence")

    div = divergence.copy()
    div["cluster_a"] = _normalise_cluster_ids(div["cluster_a"])
    div["cluster_b"] = _normalise_cluster_ids(div["cluster_b"])
    div["D_ij"] = pd.to_numeric(div["D_ij"], errors="raise")

    if not np.all(np.isfinite(div["D_ij"])):
        raise ValueError("D_ij contains non-finite values")

    if not div["D_ij"].between(0.0, 1.0).all():
        bad = div.loc[~div["D_ij"].between(0.0, 1.0)]
        raise ValueError(
            "D_ij must lie in [0, 1]:\n" + bad.to_string(index=False)
        )

    if (div["cluster_a"] == div["cluster_b"]).any():
        bad = div.loc[div["cluster_a"] == div["cluster_b"]]
        raise ValueError(
            "Self-pairs are not allowed:\n" + bad.to_string(index=False)
        )

    present_clusters = set(div["cluster_a"]) | set(div["cluster_b"])
    unknown = present_clusters - cluster_ids
    if unknown:
        raise ValueError(
            f"Divergence table contains clusters absent from insularity: "
            f"{sorted(unknown)}"
        )

    # Canonicalise unordered pairs.
    ordered = np.sort(div[["cluster_a", "cluster_b"]].to_numpy(dtype=str), axis=1)
    div["cluster_i"] = ordered[:, 0]
    div["cluster_j"] = ordered[:, 1]

    if div[["cluster_i", "cluster_j"]].duplicated().any():
        duplicates = div.loc[
            div[["cluster_i", "cluster_j"]].duplicated(keep=False),
            ["cluster_i", "cluster_j", "D_ij"],
        ]
        raise ValueError(
            "Duplicate unordered cluster pairs:\n"
            + duplicates.to_string(index=False)
        )

    div = div[["cluster_i", "cluster_j", "D_ij"]].sort_values(
        ["cluster_i", "cluster_j"]
    )

    if require_complete_pairs:
        k = len(cluster_ids)
        expected = k * (k - 1) // 2
        if len(div) != expected:
            expected_pairs = {
                tuple(sorted((a, b)))
                for idx, a in enumerate(sorted(cluster_ids))
                for b in sorted(cluster_ids)[idx + 1 :]
            }
            observed_pairs = set(
                map(tuple, div[["cluster_i", "cluster_j"]].to_numpy())
            )
            missing_pairs = sorted(expected_pairs - observed_pairs)
            raise ValueError(
                f"Expected {expected} unordered pairs for k={k}, "
                f"but found {len(div)}. Missing pairs: {missing_pairs}"
            )

    return div.reset_index(drop=True)


def compute_s_cross(insularity: pd.DataFrame) -> dict[str, float | int]:
    """
    Compute the corpus-level cross-cluster citation fraction.

        S_cross = E_cross_cluster / E_within_corpus

    S_cross is defined for a specific partition and generally depends on
    the number, sizes, and composition of its clusters.
    """
    ins = validate_insularity(insularity)

    total_within = int(ins["total_edges"].sum())
    total_intra = int(ins["intra_edges"].sum())
    total_cross = int(ins["cross_edges"].sum())

    # Internal consistency check.
    if total_cross != total_within - total_intra:
        raise RuntimeError("Internal edge-count inconsistency")

    return {
        "S_cross": total_cross / total_within,
        "E_within_corpus": total_within,
        "E_intra_cluster": total_intra,
        "E_cross_cluster": total_cross,
    }


def compute_weighted_divergence(
    insularity: pd.DataFrame,
    divergence: pd.DataFrame,
    require_complete_pairs: bool = True,
) -> tuple[float, float, pd.DataFrame]:
    """
    Compute unweighted and canonical exposure-weighted divergence.

    Let e_i be the number of cross-cluster citation edges originating
    from cluster i. For each unordered pair (i, j):

        raw_weight_ij = e_i * e_j
        w_ij          = raw_weight_ij / sum_{p<q}(raw_weight_pq)
        D_bar_w       = sum_{i<j}(w_ij * D_ij)

    D_bar is also returned as an unweighted descriptive statistic.
    """
    ins = validate_insularity(insularity)
    cluster_ids = set(ins["cluster"])
    div = validate_divergence(
        divergence,
        cluster_ids=cluster_ids,
        require_complete_pairs=require_complete_pairs,
    )

    exposure = ins.set_index("cluster")["cross_edges"].astype(float)

    pairs = div.copy()
    pairs["e_i"] = pairs["cluster_i"].map(exposure)
    pairs["e_j"] = pairs["cluster_j"].map(exposure)

    if pairs[["e_i", "e_j"]].isna().any().any():
        raise RuntimeError("Could not map all pair clusters to cross-edge counts")

    pairs["raw_weight"] = pairs["e_i"] * pairs["e_j"]
    total_raw_weight = float(pairs["raw_weight"].sum())

    if total_raw_weight <= 0:
        raise ValueError(
            "Pair exposure weights sum to zero. At least two clusters must "
            "participate in cross-cluster citation flow."
        )

    pairs["w_ij"] = pairs["raw_weight"] / total_raw_weight
    pairs["weighted_D_ij"] = pairs["w_ij"] * pairs["D_ij"]

    D_bar = float(pairs["D_ij"].mean())
    D_bar_w = float(pairs["weighted_D_ij"].sum())

    # Helpful diagnostics.
    if not np.isclose(float(pairs["w_ij"].sum()), 1.0, atol=1e-12):
        raise RuntimeError("Normalised pair weights do not sum to one")

    return D_bar, D_bar_w, pairs


def _validate_cluster_sizes(
    cluster_sizes: pd.Series,
    expected_clusters: list[str],
) -> np.ndarray:
    """Return positive cluster sizes aligned to expected_clusters."""
    sizes = cluster_sizes.copy()
    sizes.index = sizes.index.astype(str).str.strip()
    sizes = pd.to_numeric(sizes, errors="raise")

    missing = set(expected_clusters) - set(sizes.index)
    if missing:
        raise ValueError(
            f"cluster_sizes is missing clusters: {sorted(missing)}"
        )

    aligned = sizes.loc[expected_clusters].to_numpy(dtype=float)
    if not np.all(np.isfinite(aligned)):
        raise ValueError("cluster_sizes contains non-finite values")
    if (aligned <= 0).any():
        raise ValueError("All cluster sizes must be positive")
    if not np.allclose(aligned, np.round(aligned)):
        raise ValueError("Cluster sizes must be integer counts")

    return aligned.astype(np.int64)


def size_mixing_null_scross(
    insularity: pd.DataFrame,
    cluster_sizes: pd.Series,
    n_nulls: int,
    seed: int,
) -> np.ndarray:
    """
    Auxiliary size-mixing null distribution for S_cross only.

    The observed number of within-corpus citation edges is distributed
    over all ordered document pairs under random mixing constrained only
    by cluster sizes:

        within-cluster weight i = n_i * (n_i - 1)
        cross-cluster weight ij = 2 * n_i * n_j, for i < j

    This null does not recompute vocabulary divergence and is therefore
    not a null model for the full CSC.
    """
    if n_nulls <= 0:
        raise ValueError("n_nulls must be positive")

    ins = validate_insularity(insularity)
    clusters = ins["cluster"].tolist()
    sizes = _validate_cluster_sizes(cluster_sizes, clusters)

    E_total = int(ins["total_edges"].sum())
    rng = np.random.default_rng(seed)

    weights: list[float] = []
    is_cross: list[bool] = []

    for i in range(len(clusters)):
        weights.append(float(sizes[i] * (sizes[i] - 1)))
        is_cross.append(False)

        for j in range(i + 1, len(clusters)):
            weights.append(float(2 * sizes[i] * sizes[j]))
            is_cross.append(True)

    weights_array = np.asarray(weights, dtype=float)
    cross_mask = np.asarray(is_cross, dtype=bool)

    if weights_array.sum() <= 0:
        raise ValueError("Null-model allocation weights sum to zero")

    probabilities = weights_array / weights_array.sum()
    draws = rng.multinomial(E_total, probabilities, size=n_nulls)
    cross_counts = draws[:, cross_mask].sum(axis=1)

    return cross_counts / E_total


def empirical_p_value(
    null: np.ndarray,
    observed: float,
    alternative: Alternative = "two-sided",
) -> float:
    """Empirical permutation p-value with the standard plus-one correction."""
    values = np.asarray(null, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("null must be a non-empty one-dimensional array")

    if alternative == "greater":
        extreme = int(np.count_nonzero(values >= observed))
    elif alternative == "less":
        extreme = int(np.count_nonzero(values <= observed))
    elif alternative == "two-sided":
        center = float(values.mean())
        extreme = int(
            np.count_nonzero(
                np.abs(values - center) >= abs(observed - center)
            )
        )
    else:
        raise ValueError(
            "alternative must be 'greater', 'less', or 'two-sided'"
        )

    return float((extreme + 1) / (len(values) + 1))


def _z_score(observed: float, null: np.ndarray) -> tuple[float, float, float]:
    """Return null mean, sample SD, and descriptive z-score."""
    values = np.asarray(null, dtype=float)
    mu = float(values.mean())
    sigma = float(values.std(ddof=1))

    if sigma < 1e-12:
        if observed > mu:
            z = float("inf")
        elif observed < mu:
            z = float("-inf")
        else:
            z = 0.0
    else:
        z = (observed - mu) / sigma

    return mu, sigma, float(z)


def compute_csc(
    insularity: pd.DataFrame,
    divergence: pd.DataFrame,
    corpus: str = "corpus",
    *,
    require_complete_pairs: bool = True,
    cluster_sizes: Optional[pd.Series] = None,
    n_nulls: int = 0,
    seed: int = 42,
    null_alternative: Alternative = "two-sided",
) -> CSCResult:
    """
    Compute the canonical Cross-Subdomain Coherence score.

    The optional null is an auxiliary size-mixing null for S_cross only.
    It is disabled by default.
    """
    ins = validate_insularity(insularity)
    sc = compute_s_cross(ins)

    D_bar, D_bar_w, pairs = compute_weighted_divergence(
        ins,
        divergence,
        require_complete_pairs=require_complete_pairs,
    )

    S_cross = float(sc["S_cross"])
    fragmentation = S_cross * D_bar_w
    csc = 1.0 - fragmentation

    # Bound checks protect against malformed inputs and floating error.
    if not (-1e-12 <= fragmentation <= 1.0 + 1e-12):
        raise RuntimeError(
            f"Fragmentation fell outside [0, 1]: {fragmentation}"
        )
    if not (-1e-12 <= csc <= 1.0 + 1e-12):
        raise RuntimeError(f"CSC fell outside [0, 1]: {csc}")

    fragmentation = float(np.clip(fragmentation, 0.0, 1.0))
    csc = float(np.clip(csc, 0.0, 1.0))

    # Allocate the observed corpus-level cross-flow across pairs using
    # the canonical exposure weights. This is useful for audit/output,
    # but it is not a raw observed pairwise citation fraction.
    pairs["S_ij_allocated"] = S_cross * pairs["w_ij"]
    pairs["fragmentation_contribution"] = (
        S_cross * pairs["weighted_D_ij"]
    )

    result = CSCResult(
        corpus=corpus,
        n_clusters=len(ins),
        n_pairs=len(pairs),
        E_within_corpus=int(sc["E_within_corpus"]),
        E_intra_cluster=int(sc["E_intra_cluster"]),
        E_cross_cluster=int(sc["E_cross_cluster"]),
        S_cross=S_cross,
        D_bar=D_bar,
        D_bar_w=D_bar_w,
        fragmentation=fragmentation,
        csc=csc,
        pairwise=pairs,
    )

    if n_nulls > 0:
        if cluster_sizes is None:
            raise ValueError(
                "cluster_sizes is required when n_nulls > 0. "
                "Citation degree is not used as a size proxy."
            )

        null = size_mixing_null_scross(
            insularity=ins,
            cluster_sizes=cluster_sizes,
            n_nulls=n_nulls,
            seed=seed,
        )
        mu, sigma, z = _z_score(S_cross, null)

        result.null_type = "size-mixing null for S_cross only"
        result.null_alternative = null_alternative
        result.mu_null_scross = mu
        result.sigma_null_scross = sigma
        result.z_scross = z
        result.p_empirical_scross = empirical_p_value(
            null,
            observed=S_cross,
            alternative=null_alternative,
        )
        result.n_nulls = int(n_nulls)
        result.null_scross = null

    return result


def load_cluster_sizes(path: Path) -> pd.Series:
    """Load cluster sizes from columns: cluster,n_documents."""
    frame = pd.read_csv(path)
    _require_columns(frame, {"cluster", "n_documents"}, "cluster sizes")
    frame["cluster"] = _normalise_cluster_ids(frame["cluster"])

    if frame["cluster"].duplicated().any():
        raise ValueError("cluster-size table contains duplicate clusters")

    return frame.set_index("cluster")["n_documents"]


def save_result(result: CSCResult, output_dir: Path) -> None:
    """Save summary, pairwise audit table, and optional null draws."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "csc_summary.csv"
    summary_json = output_dir / "csc_summary.json"
    pairwise_csv = output_dir / "csc_pairwise.csv"

    result.to_frame().to_csv(summary_csv, index=False)

    summary = result.summary_row()
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    if result.pairwise is None:
        raise RuntimeError("Pairwise audit table is unexpectedly absent")
    result.pairwise.to_csv(pairwise_csv, index=False)

    if result.null_scross is not None:
        pd.DataFrame(
            {"S_cross_null": result.null_scross}
        ).to_csv(output_dir / "scross_size_mixing_null.csv", index=False)

    with (output_dir / "csc_report.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(str(result))
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the canonical Cross-Subdomain Coherence score "
            "from cluster-level citation counts and pairwise divergence."
        )
    )
    parser.add_argument(
        "--insularity",
        required=True,
        type=Path,
        help="CSV with cluster,total_edges,intra_edges",
    )
    parser.add_argument(
        "--divergence",
        required=True,
        type=Path,
        help="CSV with cluster_a,cluster_b,D_ij",
    )
    parser.add_argument(
        "--corpus",
        default="corpus",
        help="Human-readable corpus label",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("csc_results"),
        help="Directory for summary and audit outputs",
    )
    parser.add_argument(
        "--allow-incomplete-pairs",
        action="store_true",
        help=(
            "Allow a divergence table that does not contain every "
            "unordered cluster pair"
        ),
    )
    parser.add_argument(
        "--cluster-sizes",
        type=Path,
        default=None,
        help="Optional CSV with cluster,n_documents",
    )
    parser.add_argument(
        "--n-nulls",
        type=int,
        default=0,
        help=(
            "Number of auxiliary size-mixing S_cross null draws; "
            "0 disables the null"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the auxiliary null",
    )
    parser.add_argument(
        "--null-alternative",
        choices=("greater", "less", "two-sided"),
        default="two-sided",
        help="Alternative hypothesis for the auxiliary S_cross null",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.n_nulls < 0:
        parser.error("--n-nulls cannot be negative")

    insularity = pd.read_csv(args.insularity)
    divergence = pd.read_csv(args.divergence)

    cluster_sizes = (
        load_cluster_sizes(args.cluster_sizes)
        if args.cluster_sizes is not None
        else None
    )

    result = compute_csc(
        insularity=insularity,
        divergence=divergence,
        corpus=args.corpus,
        require_complete_pairs=not args.allow_incomplete_pairs,
        cluster_sizes=cluster_sizes,
        n_nulls=args.n_nulls,
        seed=args.seed,
        null_alternative=args.null_alternative,
    )

    save_result(result, args.output_dir)
    print(result)
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
