"""
aws_align.csc — the Cross-cluster Synthesis Coefficient (CSC) diagnostic.

The instrument
--------------
A clustered scholarly corpus is characterised by two independent quantities:

  S_cross : structural coupling
      the fraction of within-corpus citations that cross cluster boundaries.
      High S_cross means the sub-communities *cite each other a lot* — they are
      structurally aware of one another.

  D_bar   : semantic divergence
      the mean pairwise ranked-vocabulary divergence (1 - RBO, or a
      Jensen-Shannon distance) across all cluster pairs. High D_bar means the
      sub-communities *describe their work in different words*.

  AWS = S_cross * D_bar        (Awareness Without Synthesis score)
  CSC = 1 - AWS                (Cross-cluster Synthesis Coefficient)

Reading the scale
-----------------
  CSC -> 1.0   aligned: clusters that cite each other also share vocabulary.
  CSC low      fragmented: high structural coupling + high vocabulary
               divergence. The literature is *aware* of the connection
               (citations cross) but has *not synthesised* it into shared
               language. This is the "awareness without synthesis" regime.

Reference points (from the accompanying paper, reproduced by the demo data):
  Plastic-recycling corpus : CSC = 0.402  (S_cross = 0.617, D_bar = 0.976)
  Recipe corpus            : CSC = 0.766  (S_cross = 0.395, D_bar = 0.592)

Null model
----------
S_cross is compared against a size-proportional null in which the observed
cross-cluster citations are redistributed over cluster pairs with probability
proportional to n_i * n_j. Z(S_cross) measures how far the observed coupling
sits above what cluster sizes alone would produce.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CSCResult:
    """Container for a CSC diagnostic run."""

    corpus: str
    n_clusters: int
    E_within: int
    E_cross: int
    S_cross: float
    D_bar: float
    aws_score: float
    csc: float
    # null model
    mu_null: float = float("nan")
    sigma_null: float = float("nan")
    z_scross: float = float("nan")
    p_value: float = float("nan")
    n_nulls: int = 0
    pairwise: Optional[pd.DataFrame] = field(default=None, repr=False)

    def summary_row(self) -> dict:
        return {
            "corpus": self.corpus,
            "n_clusters": self.n_clusters,
            "E_within": self.E_within,
            "E_cross": self.E_cross,
            "S_cross": round(self.S_cross, 4),
            "D_bar": round(self.D_bar, 4),
            "AWS_score": round(self.aws_score, 4),
            "CSC": round(self.csc, 4),
            "mu_null": round(self.mu_null, 6),
            "sigma_null": round(self.sigma_null, 6),
            "Z_Scross": round(self.z_scross, 2),
            "p_value": round(self.p_value, 4),
            "n_nulls": self.n_nulls,
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.summary_row()])

    def __str__(self) -> str:
        z = ">1000" if np.isinf(self.z_scross) else f"{self.z_scross:.1f}"
        return (
            f"CSC diagnostic — {self.corpus}\n"
            f"  clusters      = {self.n_clusters}\n"
            f"  S_cross       = {self.S_cross:.4f}   "
            f"(cross={self.E_cross:,} / within={self.E_within:,})\n"
            f"  D_bar         = {self.D_bar:.4f}\n"
            f"  AWS = SxD     = {self.aws_score:.4f}\n"
            f"  CSC = 1-AWS   = {self.csc:.4f}\n"
            f"  null model    : mu={self.mu_null:.4f} sigma={self.sigma_null:.5f} "
            f"Z(S_cross)={z} p={self.p_value:.4f}"
        )


def compute_s_cross(insularity: pd.DataFrame) -> dict:
    """
    S_cross = (sum total_edges - sum intra_edges) / sum total_edges.

    k-invariant: independent of the number of clusters.
    """
    total_within = int(insularity["total_edges"].sum())
    total_intra = int(insularity["intra_edges"].sum())
    total_cross = total_within - total_intra
    if total_within == 0:
        raise ValueError("total_edges sums to zero — no citations to analyse")
    return {
        "S_cross": total_cross / total_within,
        "total_within": total_within,
        "total_intra": total_intra,
        "total_cross": total_cross,
    }


def compute_d_bar(divergence: pd.DataFrame) -> float:
    """Mean pairwise divergence over the (already upper-triangular) long table."""
    if len(divergence) == 0:
        raise ValueError("divergence table is empty")
    return float(divergence["D_ij"].mean())


def _null_scross(
    insularity: pd.DataFrame,
    cluster_sizes: Optional[pd.Series],
    n_nulls: int,
    seed: int,
) -> np.ndarray:
    """
    Size-proportional null distribution of S_cross.

    All E_total observed citations are redistributed over *every* cluster pair
    — within-cluster pairs (i, i) as well as cross-cluster pairs (i, j), i<j —
    with probability proportional to the size product:

        within pair (i, i):  n_i * (n_i - 1)      (ordered, no self-loops)
        cross  pair (i, j):  2 * n_i * n_j        (both directions)

    Each multinomial draw yields a null count of cross-cluster edges; dividing
    by E_total gives one null S_cross. Because edges may fall on within- or
    cross-cluster pairs, the null total of cross edges varies from draw to draw
    (unlike a redistribution that conserves the cross total), so the null has
    non-zero variance and Z(S_cross) is meaningful.

    If cluster sizes are unavailable, cluster edge-degree (total_edges) is used
    as a size proxy.
    """
    rng = np.random.default_rng(seed)
    ins = insularity.copy()
    E_total = int(ins["total_edges"].sum())
    clusters = ins["cluster"].tolist()

    if cluster_sizes is not None:
        sizes = np.array([float(cluster_sizes.get(c, 1)) for c in clusters])
    else:
        # proxy: a cluster's citation degree scales with its size
        sizes = ins["total_edges"].to_numpy(dtype=float)
    sizes = np.where(sizes <= 0, 1.0, sizes)

    n = len(clusters)
    weights = []
    is_cross = []
    for ii in range(n):
        for jj in range(n):
            if ii == jj:
                weights.append(sizes[ii] * max(sizes[ii] - 1.0, 0.0))
                is_cross.append(False)
            elif ii < jj:
                weights.append(2.0 * sizes[ii] * sizes[jj])
                is_cross.append(True)
    weights = np.asarray(weights, dtype=float)
    is_cross = np.asarray(is_cross, dtype=bool)
    if weights.sum() == 0:
        weights[:] = 1.0
    probs = weights / weights.sum()

    draws = rng.multinomial(E_total, probs, size=n_nulls)  # (n_nulls, n_pairs)
    cross_counts = draws[:, is_cross].sum(axis=1)
    return cross_counts / E_total


def _z(obs: float, mu: float, sigma: float) -> float:
    if sigma < 1e-12:
        return float("inf") if obs > mu else float("-inf")
    return (obs - mu) / sigma


def compute_csc(
    insularity: pd.DataFrame,
    divergence: pd.DataFrame,
    corpus: str = "corpus",
    cluster_sizes: Optional[pd.Series] = None,
    n_nulls: int = 1000,
    seed: int = 42,
) -> CSCResult:
    """
    Full CSC diagnostic.

    Parameters
    ----------
    insularity : DataFrame  (cluster, intra_edges, total_edges) — see io.load_insularity
    divergence : DataFrame  (cluster_a, cluster_b, D_ij)        — see io.load_divergence
    corpus     : label for this run
    cluster_sizes : optional Series indexed by cluster id (weights the null model)
    n_nulls    : number of null permutations (0 disables the null model)
    seed       : RNG seed for reproducibility

    Returns
    -------
    CSCResult
    """
    sc = compute_s_cross(insularity)
    S_cross = sc["S_cross"]
    D_bar = compute_d_bar(divergence)
    aws = S_cross * D_bar
    csc = 1.0 - aws

    result = CSCResult(
        corpus=corpus,
        n_clusters=len(insularity),
        E_within=sc["total_within"],
        E_cross=sc["total_cross"],
        S_cross=S_cross,
        D_bar=D_bar,
        aws_score=aws,
        csc=csc,
    )

    if n_nulls and n_nulls > 0:
        null = _null_scross(insularity, cluster_sizes, n_nulls, seed)
        result.mu_null = float(null.mean())
        result.sigma_null = float(null.std())
        result.z_scross = _z(S_cross, result.mu_null, result.sigma_null)
        # one-sided p: fraction of null >= observed
        result.p_value = float((null >= S_cross).mean())
        result.n_nulls = n_nulls

    return result
