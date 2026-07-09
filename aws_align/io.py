"""
aws_align.io — input loaders for the CSC / AWS diagnostic.

Two input tables describe a clustered scholarly corpus:

1. Citation insularity  (one row per cluster)
   Two accepted formats:
     - "cluster"    : columns cluster, intra_edges, total_edges[, insularity]
     - "by_cluster" : columns cluster, n_papers, total_citations,
                      internal_citations, external_citations[, insularity_pct]
   Both are normalised to: cluster, intra_edges, total_edges.

2. Vocabulary divergence  (pairwise, D_ij = 1 - RBO in [0, 1])
   Two accepted formats:
     - long   : columns cluster_a, cluster_b, 1_rbo   (or D_ij / jsd / value)
     - matrix : first column is the cluster label index, remaining columns are
                cluster labels; the diagonal is 0.
   Both are normalised to a long DataFrame: cluster_a, cluster_b, D_ij.

The loaders are deliberately permissive about column names so the same tool
runs on outputs from different graph-construction pipelines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

# Candidate column names for the divergence value in long format.
_DIVERGENCE_ALIASES = ("1_rbo", "d_ij", "D_ij", "jsd", "divergence", "value", "one_minus_rbo")


def load_insularity(path: PathLike, fmt: str = "auto") -> pd.DataFrame:
    """
    Load per-cluster citation insularity and normalise to
    columns: cluster, intra_edges, total_edges.

    Parameters
    ----------
    path : str or Path
    fmt  : {"auto", "cluster", "by_cluster"}
        "auto" (default) detects the format from the columns present.

    Returns
    -------
    DataFrame with columns [cluster, intra_edges, total_edges]
        (plus any original columns, preserved).
    """
    df = pd.read_csv(path)
    cols = set(df.columns)

    if fmt == "auto":
        if {"intra_edges", "total_edges"} <= cols:
            fmt = "cluster"
        elif {"internal_citations", "total_citations"} <= cols:
            fmt = "by_cluster"
        else:
            raise ValueError(
                "Could not auto-detect insularity format. Expected either "
                "(intra_edges, total_edges) or "
                "(internal_citations, total_citations). Got: "
                f"{sorted(cols)}"
            )

    if fmt == "cluster":
        missing = {"cluster", "intra_edges", "total_edges"} - cols
        if missing:
            raise ValueError(f"'cluster' format missing columns: {sorted(missing)}")
        out = df.copy()

    elif fmt == "by_cluster":
        missing = {"cluster", "internal_citations", "total_citations"} - cols
        if missing:
            raise ValueError(f"'by_cluster' format missing columns: {sorted(missing)}")
        out = df.rename(
            columns={
                "internal_citations": "intra_edges",
                "total_citations": "total_edges",
            }
        )
    else:
        raise ValueError(f"Unknown fmt={fmt!r}; use auto/cluster/by_cluster")

    out["intra_edges"] = pd.to_numeric(out["intra_edges"], errors="coerce")
    out["total_edges"] = pd.to_numeric(out["total_edges"], errors="coerce")
    out = out.dropna(subset=["intra_edges", "total_edges"])
    return out.reset_index(drop=True)


def load_divergence(path: PathLike) -> pd.DataFrame:
    """
    Load pairwise vocabulary divergence and normalise to a long DataFrame
    with columns: cluster_a, cluster_b, D_ij  (upper triangle, i < j).

    Accepts both a long table and a square matrix. Divergence values are
    expected in [0, 1] (1 - RBO, or Jensen-Shannon distance).
    """
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    # Long format: has cluster_a & cluster_b
    if "cluster_a" in cols_lower and "cluster_b" in cols_lower:
        ca, cb = cols_lower["cluster_a"], cols_lower["cluster_b"]
        dcol = None
        for alias in _DIVERGENCE_ALIASES:
            if alias.lower() in cols_lower:
                dcol = cols_lower[alias.lower()]
                break
        if dcol is None:
            # last resort: the only remaining numeric column
            remaining = [c for c in df.columns if c not in (ca, cb)]
            numeric = [c for c in remaining if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric) != 1:
                raise ValueError(
                    "Long divergence file: could not identify the divergence "
                    f"column among {remaining}. Rename it to one of "
                    f"{_DIVERGENCE_ALIASES}."
                )
            dcol = numeric[0]
        out = df[[ca, cb, dcol]].copy()
        out.columns = ["cluster_a", "cluster_b", "D_ij"]
        out = out[out["cluster_a"] != out["cluster_b"]]
        out["D_ij"] = pd.to_numeric(out["D_ij"], errors="coerce")
        return out.dropna(subset=["D_ij"]).reset_index(drop=True)

    # Matrix format: first column is the index label
    idx_col = df.columns[0]
    mat = df.set_index(idx_col)
    labels = list(mat.index)
    rows = []
    for i, ci in enumerate(labels):
        for j, cj in enumerate(labels):
            if i < j:
                # column label may be str even if index is int-like
                col = cj if cj in mat.columns else str(cj)
                rows.append(
                    {"cluster_a": ci, "cluster_b": cj, "D_ij": float(mat.loc[ci, col])}
                )
    return pd.DataFrame(rows)


def load_cluster_sizes(path: PathLike) -> "pd.Series":
    """
    Load cluster sizes from a paper_topics.csv-style file (one row per paper,
    with a 'cluster' column). Returns a Series indexed by cluster id.
    Used only to weight the size-proportional null model.
    """
    df = pd.read_csv(path, on_bad_lines="skip")
    if "cluster" not in df.columns:
        raise ValueError("topics file needs a 'cluster' column")
    s = pd.to_numeric(df["cluster"], errors="coerce").dropna().astype(int)
    return s.groupby(s).size().rename("n_papers")
