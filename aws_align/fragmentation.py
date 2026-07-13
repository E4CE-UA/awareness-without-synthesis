"""
aws_align.fragmentation — pairwise fragmentation table and diagnostic map.

The canonical corpus-level diagnostic is:

    D_bar_w = sum(w_ij * D_ij)

    fragmentation = S_cross * D_bar_w

    CSC = 1 - fragmentation

where:

    S_cross
        Fraction of within-corpus citation edges that cross cluster boundaries.

    D_ij
        Ranked-vocabulary divergence between clusters i and j.

    e_i
        Number of cross-cluster citation edges originating from cluster i.

    w_ij
        Exposure weight assigned to cluster pair (i, j):

            raw_weight_ij = e_i * e_j

            w_ij = raw_weight_ij / sum(raw_weight_pq)

The pair weight w_ij is an exposure allocation derived from cluster-level
cross-citation activity. It is not the literal observed proportion of citations
between clusters i and j unless pair-level citation counts are supplied by a
different analysis.

For visualisation, this module defines the local contribution:

    local_fragmentation_ij = S_cross * w_ij * D_ij

The local contributions sum exactly to the corpus-level fragmentation score:

    sum(local_fragmentation_ij) = fragmentation
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


REQUIRED_INSULARITY_COLUMNS = {
    "cluster",
    "intra_edges",
    "total_edges",
}

REQUIRED_DIVERGENCE_COLUMNS = {
    "cluster_a",
    "cluster_b",
    "D_ij",
}


# -----------------------------------------------------------------------------
# Validation and identifier helpers
# -----------------------------------------------------------------------------
def _cluster_key(value: Any) -> str:
    """
    Return a conservative key for matching cluster identifiers.

    The following common representations are treated as equivalent:

        2, "2", "2.0", "C2", "cluster_2", "cluster 2"

    Other identifiers are preserved after Unicode, case, and whitespace
    normalisation.
    """
    if pd.isna(value):
        raise ValueError("cluster identifier cannot be missing")

    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    if not text:
        raise ValueError("cluster identifier cannot be empty")

    compact = re.sub(r"[\s_-]+", "", text)

    numeric_match = re.fullmatch(r"(?:cluster|c)?([+-]?\d+(?:\.0+)?)", compact)
    if numeric_match:
        number = float(numeric_match.group(1))
        if number.is_integer():
            return f"n:{int(number)}"

    return f"s:{re.sub(r'\s+', ' ', text)}"


def _display_label(
    cluster: Any,
    cluster_labels: Optional[Dict[Any, str]],
) -> str:
    """Resolve a user label without producing values such as CC2."""
    if not cluster_labels:
        return str(cluster)

    if cluster in cluster_labels:
        return str(cluster_labels[cluster])

    target_key = _cluster_key(cluster)
    for key, label in cluster_labels.items():
        if _cluster_key(key) == target_key:
            return str(label)

    return str(cluster)


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    table_name: str,
) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {sorted(missing)}"
        )


def _validated_insularity(insularity: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        insularity,
        REQUIRED_INSULARITY_COLUMNS,
        "insularity table",
    )

    frame = insularity[
        ["cluster", "intra_edges", "total_edges"]
    ].copy()

    if frame["cluster"].isna().any():
        raise ValueError("insularity table contains missing cluster identifiers")

    if frame["cluster"].map(_cluster_key).duplicated().any():
        duplicates = frame.loc[
            frame["cluster"].map(_cluster_key).duplicated(keep=False),
            "cluster",
        ].tolist()
        raise ValueError(
            "insularity table contains duplicate cluster identifiers after "
            f"normalisation: {duplicates}"
        )

    for column in ("intra_edges", "total_edges"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(
                f"insularity column {column!r} contains non-numeric values"
            )
        if (frame[column] < 0).any():
            raise ValueError(
                f"insularity column {column!r} contains negative values"
            )

    if (frame["intra_edges"] > frame["total_edges"]).any():
        raise ValueError(
            "intra_edges cannot exceed total_edges for any cluster"
        )

    frame["e_i"] = frame["total_edges"] - frame["intra_edges"]
    frame["_cluster_key"] = frame["cluster"].map(_cluster_key)

    return frame


def _validated_divergence(divergence: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        divergence,
        REQUIRED_DIVERGENCE_COLUMNS,
        "divergence table",
    )

    frame = divergence[
        ["cluster_a", "cluster_b", "D_ij"]
    ].copy()

    if frame[["cluster_a", "cluster_b"]].isna().any().any():
        raise ValueError(
            "divergence table contains missing cluster identifiers"
        )

    frame["D_ij"] = pd.to_numeric(frame["D_ij"], errors="coerce")
    if frame["D_ij"].isna().any():
        raise ValueError("D_ij contains non-numeric or missing values")

    if ((frame["D_ij"] < 0) | (frame["D_ij"] > 1)).any():
        raise ValueError("all D_ij values must lie in the interval [0, 1]")

    frame["_key_a"] = frame["cluster_a"].map(_cluster_key)
    frame["_key_b"] = frame["cluster_b"].map(_cluster_key)

    if (frame["_key_a"] == frame["_key_b"]).any():
        raise ValueError("divergence table contains self-pairs")

    frame["_pair_key"] = [
        tuple(sorted((a, b)))
        for a, b in zip(frame["_key_a"], frame["_key_b"])
    ]

    if frame["_pair_key"].duplicated().any():
        duplicates = frame.loc[
            frame["_pair_key"].duplicated(keep=False),
            ["cluster_a", "cluster_b"],
        ].to_dict("records")
        raise ValueError(
            "divergence table contains duplicate unordered pairs: "
            f"{duplicates}"
        )

    return frame


# -----------------------------------------------------------------------------
# Pairwise table
# -----------------------------------------------------------------------------
def fragmentation_matrix(
    insularity: pd.DataFrame,
    divergence: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the canonical long pairwise fragmentation audit table.

    Returns
    -------
    pandas.DataFrame
        One row per unordered cluster pair with columns:

        cluster_a
        cluster_b
        e_i
        e_j
        raw_weight_ij
        w_ij
        D_ij
        weighted_D_ij
        S_cross
        local_fragmentation_ij

    Notes
    -----
    ``w_ij`` is the same exposure weight used in the canonical weighted
    divergence. It should not be described as an observed pairwise citation
    share.

    The following identities hold, subject only to floating-point precision:

        sum(w_ij) = 1

        sum(weighted_D_ij) = D_bar_w

        sum(local_fragmentation_ij) = S_cross * D_bar_w
    """
    ins = _validated_insularity(insularity)
    div = _validated_divergence(divergence)

    exposure_by_key = dict(
        zip(ins["_cluster_key"], ins["e_i"].astype(float))
    )

    missing_clusters = sorted(
        (
            set(div["_key_a"])
            | set(div["_key_b"])
        )
        - set(exposure_by_key)
    )
    if missing_clusters:
        raise ValueError(
            "divergence table contains clusters absent from the insularity "
            f"table: {missing_clusters}"
        )

    within_edges = float(ins["total_edges"].sum())
    cross_edges = float(ins["e_i"].sum())

    if within_edges <= 0:
        raise ValueError(
            "the total number of within-corpus citation edges must be positive"
        )

    S_cross = cross_edges / within_edges
    if not 0 <= S_cross <= 1:
        raise ValueError(
            "computed S_cross lies outside [0, 1]; check citation counts"
        )

    result = div[
        ["cluster_a", "cluster_b", "D_ij", "_key_a", "_key_b"]
    ].copy()

    result["e_i"] = result["_key_a"].map(exposure_by_key)
    result["e_j"] = result["_key_b"].map(exposure_by_key)
    result["raw_weight_ij"] = result["e_i"] * result["e_j"]

    denominator = float(result["raw_weight_ij"].sum())
    if denominator <= 0:
        raise ValueError(
            "pair exposure weights cannot be normalised because "
            "sum(e_i * e_j) is zero"
        )

    result["w_ij"] = result["raw_weight_ij"] / denominator
    result["weighted_D_ij"] = result["w_ij"] * result["D_ij"]
    result["S_cross"] = S_cross
    result["local_fragmentation_ij"] = (
        result["S_cross"]
        * result["weighted_D_ij"]
    )

    columns = [
        "cluster_a",
        "cluster_b",
        "e_i",
        "e_j",
        "raw_weight_ij",
        "w_ij",
        "D_ij",
        "weighted_D_ij",
        "S_cross",
        "local_fragmentation_ij",
    ]

    output = result[columns].reset_index(drop=True)

    # Internal numerical invariants. These should only fail if the table was
    # corrupted or a future edit changes the formula.
    if not np.isclose(output["w_ij"].sum(), 1.0, atol=1e-12):
        raise RuntimeError("internal error: pair weights do not sum to one")

    expected_fragmentation = (
        float(output["S_cross"].iloc[0])
        * float(output["weighted_D_ij"].sum())
    )
    observed_fragmentation = float(
        output["local_fragmentation_ij"].sum()
    )
    if not np.isclose(
        observed_fragmentation,
        expected_fragmentation,
        atol=1e-12,
    ):
        raise RuntimeError(
            "internal error: local contributions do not sum to fragmentation"
        )

    return output


def to_square(
    long_df: pd.DataFrame,
    value: str,
    diagonal: float = np.nan,
) -> pd.DataFrame:
    """
    Pivot a long unordered-pair table into a symmetric square matrix.
    """
    required = {"cluster_a", "cluster_b", value}
    _require_columns(long_df, required, "long pairwise table")

    clusters = sorted(
        set(long_df["cluster_a"]).union(long_df["cluster_b"]),
        key=lambda item: str(item),
    )

    matrix = pd.DataFrame(
        np.nan,
        index=clusters,
        columns=clusters,
        dtype=float,
    )

    for row in long_df.itertuples(index=False):
        cluster_a = getattr(row, "cluster_a")
        cluster_b = getattr(row, "cluster_b")
        cell_value = getattr(row, value)

        matrix.loc[cluster_a, cluster_b] = cell_value
        matrix.loc[cluster_b, cluster_a] = cell_value

    np.fill_diagonal(matrix.values, diagonal)
    return matrix


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def plot_fragmentation_map(
    insularity: pd.DataFrame,
    divergence: pd.DataFrame,
    cluster_labels: Optional[Dict[Any, str]] = None,
    corpus: str = "corpus",
    csc: Optional[float] = None,
    outfile: Union[str, Path] = "fragmentation_map.png",
    dpi: int = 300,
):
    """
    Render and save the canonical pairwise fragmentation map.

    Left panel
        Symmetric heatmap of vocabulary divergence D_ij.

    Right panel
        Exposure weight w_ij against divergence D_ij. Point colour represents
        the exact local contribution S_cross * w_ij * D_ij.

    The x-axis is explicitly labelled as an exposure allocation, not an
    observed pairwise citation share.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    import matplotlib.pyplot as plt

    pairwise = fragmentation_matrix(insularity, divergence)
    divergence_matrix = to_square(pairwise, "D_ij")

    tick_labels = [
        _display_label(cluster, cluster_labels)
        for cluster in divergence_matrix.index
    ]

    figure, (heatmap_axis, plane_axis) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
    )

    # ------------------------------------------------------------------
    # Left: vocabulary-divergence matrix
    # ------------------------------------------------------------------
    image = heatmap_axis.imshow(
        divergence_matrix.values,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )

    heatmap_axis.set_xticks(range(len(tick_labels)))
    heatmap_axis.set_yticks(range(len(tick_labels)))
    heatmap_axis.set_xticklabels(
        tick_labels,
        rotation=45,
        ha="right",
    )
    heatmap_axis.set_yticklabels(tick_labels)

    for row_index in range(len(tick_labels)):
        for column_index in range(len(tick_labels)):
            value = divergence_matrix.values[row_index, column_index]
            if np.isnan(value):
                continue

            heatmap_axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
            )

    heatmap_axis.set_title("Vocabulary divergence D_ij = 1 - RBO")
    heatmap_colorbar = figure.colorbar(
        image,
        ax=heatmap_axis,
        fraction=0.046,
        pad=0.04,
    )
    heatmap_colorbar.set_label(
        "D_ij (1 = disjoint ranked vocabularies)"
    )

    # ------------------------------------------------------------------
    # Right: exposure × divergence plane
    # ------------------------------------------------------------------
    weights = pairwise["w_ij"].to_numpy(dtype=float)
    divergences = pairwise["D_ij"].to_numpy(dtype=float)
    local_contributions = pairwise[
        "local_fragmentation_ij"
    ].to_numpy(dtype=float)

    scatter = plane_axis.scatter(
        weights,
        divergences,
        c=local_contributions,
        s=90,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )

    saturation_threshold = 0.999
    saturated = pairwise[
        pairwise["D_ij"] >= saturation_threshold
    ]
    non_saturated = pairwise[
        pairwise["D_ij"] < saturation_threshold
    ]

    if len(pairwise):
        weight_midpoint = 0.5 * (
            float(np.nanmin(weights))
            + float(np.nanmax(weights))
        )
    else:
        weight_midpoint = 0.0

    for row in non_saturated.itertuples(index=False):
        label_a = _display_label(row.cluster_a, cluster_labels)
        label_b = _display_label(row.cluster_b, cluster_labels)

        place_left = row.w_ij > weight_midpoint
        plane_axis.annotate(
            f"{label_a}–{label_b}",
            (row.w_ij, row.D_ij),
            fontsize=6.5,
            xytext=(-4 if place_left else 4, -1),
            textcoords="offset points",
            ha="right" if place_left else "left",
            va="top",
            zorder=4,
        )

    if len(saturated):
        plane_axis.annotate(
            (
                f"{len(saturated)} of {len(pairwise)} pairs have "
                "D_ij >= 0.999"
            ),
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=7,
        )

    plane_axis.set_xlabel(
        "Exposure weight w_ij (not observed pair citation share)"
    )
    plane_axis.set_ylabel("Vocabulary divergence D_ij")
    plane_axis.set_title("Exposure-weighted fragmentation plane")
    plane_axis.margins(0.14)

    contribution_colorbar = figure.colorbar(
        scatter,
        ax=plane_axis,
        fraction=0.046,
        pad=0.04,
    )
    contribution_colorbar.set_label(
        "Local contribution S_cross × w_ij × D_ij"
    )

    S_cross = float(pairwise["S_cross"].iloc[0])
    D_bar_w = float(pairwise["weighted_D_ij"].sum())
    fragmentation = float(
        pairwise["local_fragmentation_ij"].sum()
    )
    derived_csc = 1.0 - fragmentation

    title = (
        f"Vocabulary fragmentation map — {corpus}\n"
        f"S_cross = {S_cross:.3f}; "
        f"D_bar_w = {D_bar_w:.3f}; "
        f"fragmentation = {fragmentation:.3f}; "
        f"CSC = {(derived_csc if csc is None else csc):.3f}"
    )

    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.91))

    output_path = Path(outfile).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    return figure


__all__ = [
    "fragmentation_matrix",
    "plot_fragmentation_map",
    "to_square",
]
