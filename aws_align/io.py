"""
Input loaders for the Cross-Subdomain Coherence (CSC) diagnostic.

Normalised outputs
------------------
Citation insularity:
    cluster, intra_edges, total_edges

Vocabulary divergence:
    cluster_a, cluster_b, D_ij

Cluster sizes:
    pandas Series indexed by cluster identifier

Cluster identifiers such as ``C2`` are preserved. Integer-like identifiers
such as ``2`` and ``2.0`` are normalised consistently to ``"2"``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]
InsularityFormat = Literal["auto", "cluster", "by_cluster"]

_DIVERGENCE_ALIASES = (
    "d_ij",
    "1_rbo",
    "one_minus_rbo",
    "jsd",
    "divergence",
    "value",
)

_SIZE_ALIASES = (
    "n_documents",
    "n_papers",
    "cluster_size",
    "size",
    "count",
)


def _read_csv(path: PathLike) -> pd.DataFrame:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {file_path}")

    frame = pd.read_csv(file_path, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]

    if frame.empty:
        raise ValueError(f"CSV contains no rows: {file_path}")

    return frame


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lookup = {
        str(column).strip().casefold(): str(column)
        for column in frame.columns
    }
    for alias in aliases:
        if alias.casefold() in lookup:
            return lookup[alias.casefold()]
    return None


def _normalise_cluster_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("cluster identifier cannot be missing")

    text = str(value).strip()
    if not text:
        raise ValueError("cluster identifier cannot be empty")

    # Convert 2 and 2.0 to the same identifier, but preserve C2 and 02.
    if re.fullmatch(r"[+-]?(?:0|[1-9]\d*)(?:\.0+)?", text):
        return str(int(float(text)))

    return text


def _normalise_clusters(series: pd.Series, name: str) -> pd.Series:
    try:
        return series.map(_normalise_cluster_id)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc


def _integer_counts(series: pd.Series, name: str) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains non-numeric values") from exc

    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    if (values < 0).any():
        raise ValueError(f"{name} cannot contain negative values")
    if not np.allclose(values, np.round(values)):
        raise ValueError(f"{name} must contain integer counts")

    return values.astype(np.int64)


def _divergence_values(series: pd.Series) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("D_ij contains non-numeric values") from exc

    if not np.isfinite(values).all():
        raise ValueError("D_ij contains non-finite values")
    if not values.between(0.0, 1.0).all():
        raise ValueError("D_ij values must lie in [0, 1]")

    return values


def load_insularity(
    path: PathLike,
    fmt: InsularityFormat = "auto",
) -> pd.DataFrame:
    """
    Load per-cluster citation counts.

    Accepted formats
    ----------------
    cluster:
        cluster, intra_edges, total_edges

    by_cluster:
        cluster, internal_citations, total_citations
    """
    if fmt not in {"auto", "cluster", "by_cluster"}:
        raise ValueError(
            "fmt must be 'auto', 'cluster', or 'by_cluster'"
        )

    frame = _read_csv(path)
    columns = set(frame.columns)

    if fmt == "auto":
        if {"cluster", "intra_edges", "total_edges"} <= columns:
            fmt = "cluster"
        elif {
            "cluster",
            "internal_citations",
            "total_citations",
        } <= columns:
            fmt = "by_cluster"
        else:
            raise ValueError(
                "could not detect insularity format; expected either "
                "(cluster, intra_edges, total_edges) or "
                "(cluster, internal_citations, total_citations)"
            )

    if fmt == "cluster":
        required = {"cluster", "intra_edges", "total_edges"}
        missing = required - columns
        if missing:
            raise ValueError(
                f"cluster format missing columns: {sorted(missing)}"
            )
        output = frame.copy()
    else:
        required = {
            "cluster",
            "internal_citations",
            "total_citations",
        }
        missing = required - columns
        if missing:
            raise ValueError(
                f"by_cluster format missing columns: {sorted(missing)}"
            )
        output = frame.rename(
            columns={
                "internal_citations": "intra_edges",
                "total_citations": "total_edges",
            }
        ).copy()

    output["cluster"] = _normalise_clusters(
        output["cluster"],
        "cluster",
    )
    output["intra_edges"] = _integer_counts(
        output["intra_edges"],
        "intra_edges",
    )
    output["total_edges"] = _integer_counts(
        output["total_edges"],
        "total_edges",
    )

    if output["cluster"].duplicated().any():
        duplicates = sorted(
            output.loc[
                output["cluster"].duplicated(keep=False),
                "cluster",
            ].unique()
        )
        raise ValueError(f"duplicate cluster rows: {duplicates}")

    if (output["intra_edges"] > output["total_edges"]).any():
        raise ValueError("intra_edges cannot exceed total_edges")

    if len(output) < 2:
        raise ValueError("at least two clusters are required")

    if int(output["total_edges"].sum()) <= 0:
        raise ValueError("the within-corpus citation graph has no edges")

    canonical = ["cluster", "intra_edges", "total_edges"]
    extra = [
        column for column in output.columns
        if column not in canonical
    ]
    return output[canonical + extra].reset_index(drop=True)


def _canonicalise_long_divergence(frame: pd.DataFrame) -> pd.DataFrame:
    cluster_a = _find_column(
        frame,
        ("cluster_a", "cluster_i", "source_cluster"),
    )
    cluster_b = _find_column(
        frame,
        ("cluster_b", "cluster_j", "target_cluster"),
    )

    if cluster_a is None or cluster_b is None:
        raise ValueError(
            "long divergence input needs cluster_a and cluster_b columns"
        )

    value_column = _find_column(frame, _DIVERGENCE_ALIASES)
    if value_column is None:
        remaining = [
            column for column in frame.columns
            if column not in {cluster_a, cluster_b}
        ]
        numeric = [
            column
            for column in remaining
            if pd.to_numeric(
                frame[column],
                errors="coerce",
            ).notna().all()
        ]
        if len(numeric) != 1:
            raise ValueError(
                "could not identify one divergence column; rename it "
                f"to one of {_DIVERGENCE_ALIASES}"
            )
        value_column = numeric[0]

    output = frame[
        [cluster_a, cluster_b, value_column]
    ].copy()
    output.columns = ["cluster_a", "cluster_b", "D_ij"]

    output["cluster_a"] = _normalise_clusters(
        output["cluster_a"],
        "cluster_a",
    )
    output["cluster_b"] = _normalise_clusters(
        output["cluster_b"],
        "cluster_b",
    )
    output["D_ij"] = _divergence_values(output["D_ij"])

    if (output["cluster_a"] == output["cluster_b"]).any():
        raise ValueError("self-pairs are not allowed")

    ordered = [
        tuple(sorted((a, b)))
        for a, b in zip(
            output["cluster_a"],
            output["cluster_b"],
        )
    ]
    output["cluster_a"] = [pair[0] for pair in ordered]
    output["cluster_b"] = [pair[1] for pair in ordered]

    rows: list[dict[str, object]] = []
    for (a, b), group in output.groupby(
        ["cluster_a", "cluster_b"],
        sort=True,
    ):
        values = group["D_ij"].to_numpy(dtype=float)
        if not np.allclose(values, values[0], atol=1e-12, rtol=0):
            raise ValueError(
                f"conflicting D_ij values for pair ({a}, {b})"
            )
        rows.append(
            {
                "cluster_a": a,
                "cluster_b": b,
                "D_ij": float(values[0]),
            }
        )

    return pd.DataFrame(rows)


def _matrix_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame.columns) < 3:
        raise ValueError(
            "divergence matrix needs an index column and at least "
            "two cluster columns"
        )

    index_column = frame.columns[0]
    row_labels = _normalise_clusters(
        frame[index_column],
        str(index_column),
    )
    column_labels = [
        _normalise_cluster_id(column)
        for column in frame.columns[1:]
    ]

    if row_labels.duplicated().any():
        raise ValueError("duplicate row labels in divergence matrix")
    if len(set(column_labels)) != len(column_labels):
        raise ValueError("duplicate column labels in divergence matrix")

    matrix = frame.drop(columns=[index_column]).copy()
    matrix.index = row_labels
    matrix.columns = column_labels

    if set(matrix.index) != set(matrix.columns):
        raise ValueError(
            "divergence matrix row and column labels do not match"
        )

    labels = sorted(matrix.index)
    matrix = matrix.loc[labels, labels].apply(
        pd.to_numeric,
        errors="coerce",
    )

    rows: list[dict[str, object]] = []

    for i, cluster_a in enumerate(labels):
        diagonal = matrix.loc[cluster_a, cluster_a]
        if pd.notna(diagonal) and not np.isclose(diagonal, 0.0):
            raise ValueError(
                "divergence matrix diagonal must be zero or empty"
            )

        for cluster_b in labels[i + 1:]:
            forward = matrix.loc[cluster_a, cluster_b]
            reverse = matrix.loc[cluster_b, cluster_a]
            values = [
                float(value)
                for value in (forward, reverse)
                if pd.notna(value)
            ]

            if not values:
                raise ValueError(
                    f"missing D_ij for pair ({cluster_a}, {cluster_b})"
                )
            if len(values) == 2 and not np.isclose(
                values[0],
                values[1],
                atol=1e-12,
                rtol=0,
            ):
                raise ValueError(
                    f"asymmetric matrix for pair ({cluster_a}, {cluster_b})"
                )

            rows.append(
                {
                    "cluster_a": cluster_a,
                    "cluster_b": cluster_b,
                    "D_ij": values[0],
                }
            )

    output = pd.DataFrame(rows)
    output["D_ij"] = _divergence_values(output["D_ij"])
    return output


def load_divergence(path: PathLike) -> pd.DataFrame:
    """
    Load pairwise divergence from a long table or square matrix.

    Returns exactly:
        cluster_a, cluster_b, D_ij
    """
    frame = _read_csv(path)

    has_a = _find_column(
        frame,
        ("cluster_a", "cluster_i", "source_cluster"),
    )
    has_b = _find_column(
        frame,
        ("cluster_b", "cluster_j", "target_cluster"),
    )

    if has_a is not None or has_b is not None:
        if has_a is None or has_b is None:
            raise ValueError(
                "long divergence input needs both cluster endpoints"
            )
        output = _canonicalise_long_divergence(frame)
    else:
        output = _matrix_to_long(frame)

    return output[
        ["cluster_a", "cluster_b", "D_ij"]
    ].sort_values(
        ["cluster_a", "cluster_b"]
    ).reset_index(drop=True)


def load_cluster_sizes(path: PathLike) -> pd.Series:
    """
    Load cluster sizes for the auxiliary S_cross null.

    Accepted inputs:
    - one row per document with a cluster column;
    - one row per cluster with cluster and n_documents/n_papers/size/count.
    """
    frame = _read_csv(path)

    cluster_column = _find_column(
        frame,
        ("cluster", "cluster_id"),
    )
    if cluster_column is None:
        raise ValueError("cluster-size input needs a cluster column")

    clusters = _normalise_clusters(
        frame[cluster_column],
        cluster_column,
    )
    size_column = _find_column(frame, _SIZE_ALIASES)

    if size_column is None:
        sizes = clusters.value_counts(sort=False)
    else:
        if clusters.duplicated().any():
            raise ValueError(
                "cluster-level size table contains duplicate clusters"
            )

        counts = _integer_counts(
            frame[size_column],
            size_column,
        )
        if (counts <= 0).any():
            raise ValueError("cluster sizes must be positive")

        sizes = pd.Series(
            counts.to_numpy(),
            index=clusters.to_numpy(),
        )

    sizes.index.name = "cluster"
    sizes.name = "n_documents"

    if sizes.empty:
        raise ValueError("cluster-size input contains no rows")
    if (sizes <= 0).any():
        raise ValueError("cluster sizes must be positive")

    return sizes.sort_index()


__all__ = [
    "InsularityFormat",
    "PathLike",
    "load_cluster_sizes",
    "load_divergence",
    "load_insularity",
]
