# Awareness Without Synthesis

**Awareness Without Synthesis (AWS)** is a diagnostic framework for detecting
hidden vocabulary fragmentation in scholarly knowledge graphs.

Citation-based interdisciplinarity measures often treat cross-domain citation
flow as evidence of knowledge integration. That inference can fail when
subdomains cite one another extensively while describing related phenomena in
largely non-overlapping vocabularies. The literature is structurally aware of
the connection, but it has not synthesised that connection into a shared
searchable language.

This repository provides the reference implementation of the
**Cross-Subdomain Coherence score (CSC-score)**, an upstream diagnostic for
measuring that structural–lexical mismatch.

## What the diagnostic measures

The CSC-score combines two complementary quantities defined for a specific partition of a scholarly corpus.

### 1. Cross-cluster citation fraction$$S^{\mathrm{cross}}
=
\frac{E_{\mathrm{cross}}}{E_{\mathrm{within\ corpus}}}$$where:

-$E_{\mathrm{within\ corpus}}$is the number of citation edges whose source
  and target both belong to the corpus;
-$E_{\mathrm{cross}}$is the number of those edges that connect different
  clusters.

A high$S^{\mathrm{cross}}$means that the subdomains cite one another
frequently.

### 2. Vocabulary divergence

For each unordered pair of clusters$(i,j)$, the ranked-vocabulary divergence
is:$$D_{ij}=1-\mathrm{RBO}(L_i,L_j)$$where$L_i$and$L_j$are ranked characteristic-term lists, usually
constructed with class-based TF–IDF.

The package reports two summaries:$$\bar D=\frac{1}{\binom{k}{2}}\sum_{i<j}D_{ij}$$and the canonical exposure-weighted divergence:$$\bar D_w=\sum_{i<j}w_{ij}D_{ij}.$$Let$e_i$be the number of cross-cluster citation edges originating from
cluster$i$. Pair weights are:$$w_{ij}
=
\frac{e_i e_j}
{\sum_{p<q}e_p e_q}.$$These weights give greater importance to vocabulary interfaces involving
clusters that participate more heavily in cross-cluster citation flow. They
are an exposure allocation and should not be interpreted as the literal
observed proportion of citations between a specific pair of clusters.

### 3. Fragmentation and coherence$$\mathrm{fragmentation}
=
S^{\mathrm{cross}}\bar D_w$$$$\mathrm{CSC}
=
1-\mathrm{fragmentation}.$$Interpretation:

- **CSC near 1:** citation-connected clusters also use relatively aligned
  vocabularies;
- **low CSC:** extensive cross-cluster citation flow coexists with strong
  vocabulary divergence—the awareness-without-synthesis regime.

The unweighted$\bar D$is retained as a descriptive statistic, but it does
**not** enter the canonical CSC formula.

## Important methodological properties

- The core CSC computation is encoder-free and requires only citation counts
  and pairwise ranked-vocabulary divergences.
-$S^{\mathrm{cross}}$and the resulting CSC are
  **partition-dependent**. They can change with the number, sizes, and
  composition of clusters.
- Cross-corpus comparisons should therefore report the clustering procedure
  and, where relevant, sensitivity to$k$.
- The optional size-mixing null included in the package applies only to$S^{\mathrm{cross}}$. It is not a null distribution for the full CSC.
- A full label-permutation null must recompute cluster vocabularies,$D_{ij}$,$\bar D_w$, and CSC from document-level data after each
  permutation.

## Canonical demo

The bundled demo tables in `data/demo/` describe a plastic-recycling corpus of
3,138 publications partitioned into six subdomains.

The canonical result is:

| Quantity | Value |
|---|---:|
| Clusters | 6 |
| Within-corpus citation edges | 1,951 |
| Cross-cluster citation edges | 1,204 |
|$S^{\mathrm{cross}}$| 0.617 |
| Unweighted$\bar D$| 0.976 |
| Exposure-weighted$\bar D_w$| 0.969 |
| Fragmentation$=S^{\mathrm{cross}}\bar D_w$| 0.598 |
| CSC$=1-\mathrm{fragmentation}$| 0.402 |

The high cross-cluster citation fraction shows substantial structural
awareness, while the near-total vocabulary divergence indicates weak lexical
integration.

## Repository structure

```text
awareness-without-synthesis/
│
├── aws_align/
│   ├── csc.py              # Canonical CSC computation and validation
│   ├── io.py               # Input loaders
│   ├── fragmentation.py    # Pairwise fragmentation visualisation
│   ├── align.py            # Optional LLM vocabulary-alignment layer
│   ├── cli.py              # Command-line interface
│   └── __init__.py         # Public package exports
│
├── data/demo/              # Bundled plastic-recycling demo inputs
├── examples/               # Example outputs and notebooks
├── tests/                  # Regression and invariant tests
├── README.md
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/E4CE-UA/awareness-without-synthesis.git
cd awareness-without-synthesis
pip install -e .
```

The package supports Python 3.9–3.12.

For development and tests:

```bash
pip install -e ".[dev]"
```

The core diagnostic does not require an API key or an LLM dependency.

## Quick start: Python

```python
from aws_align import compute_csc, load_divergence, load_insularity

insularity = load_insularity(
    "data/demo/citation_cluster_insularity.csv"
)
divergence = load_divergence(
    "data/demo/rbo_fragmentation.csv"
)

result = compute_csc(
    insularity,
    divergence,
    corpus="Plastic recycling",
    n_nulls=0,
)

print(result)
```

Expected output:

```text
CSC diagnostic — Plastic recycling
  clusters             = 6
  unordered pairs      = 15
  within-corpus edges  = 1,951
  intra-cluster edges  = 747
  cross-cluster edges  = 1,204
  S_cross              ≈ 0.617
  D_bar                ≈ 0.976
  D_bar_w              ≈ 0.969
  fragmentation        ≈ 0.598
  CSC                  ≈ 0.402
```

The pairwise audit table is available as:

```python
pairwise = result.pairwise
print(
    pairwise[
        [
            "cluster_i",
            "cluster_j",
            "D_ij",
            "e_i",
            "e_j",
            "w_ij",
            "weighted_D_ij",
        ]
    ]
)
```

## Command line

Every subcommand can use explicit input paths. The bundled demo files are used
when the corresponding input is omitted.

### Compute the CSC-score

```bash
aws-align diagnose
```

With explicit inputs:

```bash
aws-align diagnose \
  --insularity data/demo/citation_cluster_insularity.csv \
  --divergence data/demo/rbo_fragmentation.csv \
  --corpus "Plastic recycling" \
  --n-nulls 0 \
  --out csc_pairwise.csv
```

### Render the pairwise fragmentation map

```bash
aws-align map \
  --insularity data/demo/citation_cluster_insularity.csv \
  --divergence data/demo/rbo_fragmentation.csv \
  --corpus "Plastic recycling" \
  --out fragmentation_map.png
```

### Rebuild the archived alignment summary offline

```bash
aws-align align \
  --dry-run \
  --raw data/demo/alignment_raw.csv \
  --out alignment_summary.csv
```

This path makes no API calls.

For all options:

```bash
aws-align <subcommand> --help
```

## Input formats

### Citation insularity

The loader accepts either of the following schemas.

Canonical format:

```csv
cluster,intra_edges,total_edges
C2,15,53
C3,22,234
C4,96,425
C5,403,703
C6,75,153
C7,136,383
```

Alternative format:

```csv
cluster,internal_citations,total_citations
C2,15,53
C3,22,234
```

Both are normalised internally to:

```text
cluster, intra_edges, total_edges
```

The citation counts must refer to **within-corpus** edges. Citations from corpus
papers to works outside the corpus should not be included in this table.

### Pairwise vocabulary divergence

Long format:

```csv
cluster_a,cluster_b,D_ij
C2,C3,1.000
C2,C4,1.000
C2,C5,1.000
```

The loader also accepts common value-column aliases such as `1_rbo`, `jsd`,
`divergence`, and `value`.

A square divergence matrix is also accepted. The first column must contain the
cluster identifiers, the remaining columns must use the same identifiers, and
the diagonal must represent zero divergence.

All$D_{ij}$values must lie in$[0,1]$.

### Cluster sizes for the auxiliary structural null

The current loader accepts a paper-level topic file containing a `cluster`
column:

```csv
paper_id,cluster
W123,C2
W456,C3
```

Cluster sizes are computed by counting papers per cluster.

## Auxiliary size-mixing null

The optional null asks whether the observed$S^{\mathrm{cross}}$differs from random mixing expected from cluster sizes.

Under this null, the observed number of within-corpus citation edges is
distributed over possible within- and cross-cluster document pairs with weights
proportional to cluster sizes.

Example:

```bash
aws-align diagnose \
  --sizes data/demo/paper_topics.csv \
  --n-nulls 1000 \
  --seed 42
```

The output reports:

- null mean of$S^{\mathrm{cross}}$;
- null standard deviation;
- descriptive$z$-score;
- empirical$p$-value with a plus-one correction.

This analysis concerns the structural component only. It does not replace the
full size-preserving label-permutation analysis required to test the composite
CSC.

## Optional LLM vocabulary alignment

The CSC diagnostic itself does not use an LLM. The optional alignment layer
asks multiple models to propose cross-cluster vocabulary bridges and then
retains only cross-model consensus.

### Offline reproduction

Archived raw responses can be processed without network access:

```bash
aws-align align \
  --dry-run \
  --raw data/demo/alignment_raw.csv \
  --min-confidence 3 \
  --min-models 2 \
  --out alignment_summary.csv
```

### Live execution

Install the optional runtime dependencies:

```bash
pip install requests python-dotenv
```

Set the OpenRouter key without committing it to the repository:

```bash
export OPENROUTER_API_KEY="your-key"
```

Then run:

```bash
aws-align align \
  --terms data/demo/semantic_topics.csv \
  --top-k 20 \
  --min-confidence 3 \
  --min-models 2 \
  --raw-out alignment_raw_new.csv \
  --out alignment_summary_new.csv
```

Hosted LLM outputs are best-effort reproducible rather than bit-exact, even at
temperature 0. Raw responses should therefore be preserved together with model
identifiers, prompt versions, and execution dates.

Consensus alignment should be interpreted as cross-model agreement, not
automatic proof that two terms are interchangeable. Query expansion should be
restricted to relations explicitly validated as exact equivalents or
near-synonyms.

## Tests

Run the regression suite from the repository root:

```bash
pytest -v
```

The canonical tests should verify approximately:

```text
S_cross       = 0.617
D_bar         = 0.976
D_bar_w       = 0.969
fragmentation = 0.598
CSC           = 0.402
n_clusters    = 6
```

The suite should also test:

- the identity
  `CSC == 1 - (S_cross * D_bar_w)`;
- complete and unique unordered cluster pairs;
- valid divergence values in$[0,1]$;
- pair weights summing to one;
- CSC remaining in$[0,1]$;
- reproducibility of seeded auxiliary null draws;
- empirical$p$-values using the plus-one correction.

## Scope

The CSC-score is a corpus-level diagnostic, not a general retrieval benchmark
and not a causal measure of why two communities use different vocabularies.

A low CSC indicates that:

1. citation links cross the selected subdomain boundaries;
2. the corresponding ranked vocabularies remain strongly divergent;
3. evidence retrieval across those interfaces may require explicit vocabulary
   bridging.

The magnitude of the score depends on the corpus, citation graph, text fields,
vocabulary construction, clustering procedure, and selected partition.
Applications should report those choices transparently.

## License

This project is distributed under the MIT License.

## Authors

- Ana Bossler — University of Alicante
- Enric Bas — University of Alicante
- Andrés Fullana — University of Alicante
