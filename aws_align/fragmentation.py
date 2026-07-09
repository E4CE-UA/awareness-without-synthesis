"""
aws_align.fragmentation — pairwise fragmentation matrix and the fragmentation map.

For each cluster pair (i, j) the local Awareness-Without-Synthesis contribution is

    AWS_ij = S_ij * D_ij

where S_ij is the share of cross-cluster citations carried by the pair and D_ij
is the pair's vocabulary divergence (1 - RBO). A pair is *fragmented* when it is
simultaneously high-coupling (cited across) and high-divergence (described in
different words) — the upper-right region of the map.

The fragmentation map is the paper's instrument figure: a heatmap of the
divergence matrix with the citation coupling of each pair overlaid, so a reader
sees at a glance which cluster pairs are aware of each other yet unsynthesised.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def fragmentation_matrix(
    insularity: pd.DataFrame,
    divergence: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the long pairwise fragmentation table.

    Returns a DataFrame with columns:
        cluster_a, cluster_b, S_ij, D_ij, AWS_ij
    S_ij is the fraction of *cross-cluster* citations on this pair. Because the
    per-pair cross-citation counts are not always available, S_ij is
    approximated from the cluster external-edge counts as a coupling proxy:
        S_ij ∝ external_i * external_j   (normalised over pairs)
    when only cluster-level insularity is provided. D_ij is taken directly from
    the divergence table.
    """
    div = divergence.copy()
    # coupling proxy from cluster external degree
    ins = insularity.copy()
    ins["external"] = ins["total_edges"] - ins["intra_edges"]

    def _norm(x):
        """Normalise a cluster id so 'C2', 'cluster_2', 2, '2' all match."""
        s = "".join(ch for ch in str(x) if ch.isdigit())
        return s if s else str(x)

    ext = {
        _norm(c): float(e)
        for c, e in zip(ins["cluster"], ins["external"])
    }

    def _coupling(a, b):
        ea, eb = ext.get(_norm(a)), ext.get(_norm(b))
        if ea is None or eb is None:
            return np.nan
        return ea * eb

    div["S_raw"] = [
        _coupling(a, b) for a, b in zip(div["cluster_a"], div["cluster_b"])
    ]
    total = div["S_raw"].sum(skipna=True)
    div["S_ij"] = div["S_raw"] / total if total and total > 0 else np.nan
    div["AWS_ij"] = div["S_ij"] * div["D_ij"]
    return div[["cluster_a", "cluster_b", "S_ij", "D_ij", "AWS_ij"]].reset_index(drop=True)


def to_square(long_df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot a long pairwise table to a symmetric square matrix on `value`."""
    clusters = sorted(
        set(long_df["cluster_a"]).union(long_df["cluster_b"]),
        key=lambda x: (str(type(x)), x),
    )
    mat = pd.DataFrame(np.nan, index=clusters, columns=clusters, dtype=float)
    for _, r in long_df.iterrows():
        a, b = r["cluster_a"], r["cluster_b"]
        mat.loc[a, b] = r[value]
        mat.loc[b, a] = r[value]
    np.fill_diagonal(mat.values, np.nan)
    return mat


def plot_fragmentation_map(
    insularity: pd.DataFrame,
    divergence: pd.DataFrame,
    cluster_labels: Optional[dict] = None,
    corpus: str = "corpus",
    csc: Optional[float] = None,
    outfile: str = "fragmentation_map.png",
    dpi: int = 300,
):
    """
    Render the fragmentation map and save to `outfile`.

    Left panel  : vocabulary-divergence heatmap D_ij (the semantic axis).
    Right panel : scatter of every cluster pair in coupling×divergence space,
                  the diagnostic plane. Pairs in the upper band are the
                  fragmented ones — cited across yet lexically disjoint.

    Requires matplotlib. Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    frag = fragmentation_matrix(insularity, divergence)
    Dmat = to_square(frag, "D_ij")
    labels = [
        (cluster_labels.get(c, f"C{c}") if cluster_labels else f"C{c}")
        for c in Dmat.index
    ]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # --- left: divergence heatmap ---
    im = axL.imshow(Dmat.values, cmap="magma", vmin=0.0, vmax=1.0, aspect="equal")
    axL.set_xticks(range(len(labels)))
    axL.set_yticks(range(len(labels)))
    axL.set_xticklabels(labels, rotation=45, ha="right")
    axL.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = Dmat.values[i, j]
            if not np.isnan(v):
                axL.text(
                    j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.6 else "black", fontsize=6.5,
                )
    axL.set_title("Vocabulary divergence  $D_{ij}=1-\\mathrm{RBO}$")
    cb = fig.colorbar(im, ax=axL, fraction=0.046, pad=0.04)
    cb.set_label("divergence (1 = disjoint vocabulary)")

    # --- right: coupling × divergence diagnostic plane ---
    # Divergence often saturates at 1.0 (fully disjoint vocabulary) for most
    # pairs. Labelling every point then piles text at the ceiling, so we label
    # only the pairs that DO share vocabulary (D_ij < ~1) — the exceptions —
    # and annotate the saturated mass once. The saturation itself is the
    # finding: pairs cite each other yet share almost no ranked vocabulary.
    S = frag["S_ij"].to_numpy()
    D = frag["D_ij"].to_numpy()
    aws = frag["AWS_ij"].to_numpy()
    sc = axR.scatter(S, D, c=aws, cmap="viridis", s=90, edgecolor="k",
                     linewidth=0.5, zorder=3)

    SAT = 0.999
    n_sat = int((frag["D_ij"] >= SAT).sum())
    shared = frag[frag["D_ij"] < SAT]
    s_mid = 0.5 * (np.nanmin(S) + np.nanmax(S))
    for _, r in shared.iterrows():
        la = cluster_labels.get(r["cluster_a"], f"C{r['cluster_a']}") if cluster_labels else f"C{r['cluster_a']}"
        lb = cluster_labels.get(r["cluster_b"], f"C{r['cluster_b']}") if cluster_labels else f"C{r['cluster_b']}"
        # points on the right half get left-anchored labels so they don't clip
        right = r["S_ij"] > s_mid
        axR.annotate(
            f"{la}–{lb}", (r["S_ij"], r["D_ij"]),
            fontsize=6.5,
            xytext=(-4 if right else 4, -1), textcoords="offset points",
            ha="right" if right else "left", va="top", zorder=4,
        )
    if n_sat:
        axR.annotate(
            f"{n_sat} of {len(frag)} pairs at $D\\approx1$\n(disjoint vocabulary)",
            xy=(0.97, 0.985), xycoords=("axes fraction", "data"),
            ha="right", va="top", fontsize=7, color="0.25",
        )
    axR.set_xlabel("citation coupling  $S_{ij}$  (share of cross-cluster edges)")
    axR.set_ylabel("vocabulary divergence  $D_{ij}$")
    axR.set_title("Fragmentation plane")
    axR.margins(0.14)
    cb2 = fig.colorbar(sc, ax=axR, fraction=0.046, pad=0.04)
    cb2.set_label("local AWS = $S_{ij}\\times D_{ij}$")

    suptitle = f"Vocabulary fragmentation map — {corpus}"
    if csc is not None:
        suptitle += f"   (CSC = {csc:.3f})"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    return fig
