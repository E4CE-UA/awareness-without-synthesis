# Awareness Without Synthesis

**Awareness Without Synthesis (AWS)** is a diagnostic framework for detecting
hidden vocabulary fragmentation in scholarly knowledge graphs.

Citation-based interdisciplinarity metrics often treat cross-domain citation
flow as evidence of knowledge integration. This inference fails when
subdomains cite one another while describing closely related concepts in
largely non-overlapping vocabularies — the *awareness without synthesis*
regime. This repository provides the reference implementation of the
**Cross-Subdomain Coherence score (CSC-score)**, an upstream diagnostic that
quantifies whether the citation links between research communities are
accompanied by genuine vocabulary integration.

The diagnostic combines two independent corpus-level quantities:

```
S_cross : structural coupling   — fraction of citations crossing cluster boundaries
D_bar   : semantic divergence   — mean pairwise ranked-vocabulary divergence (1 − RBO)

AWS = S_cross · D_bar           (awareness-without-synthesis score)
CSC = 1 − AWS                   (Cross-Subdomain Coherence score)
```

**CSC → 1** means clusters that cite each other also share vocabulary
(integrated). **Low CSC** means high citation coupling *and* high vocabulary
divergence: the literature is aware of the connection but has not synthesised
it into shared language.

> **The core computation is encoder-free.** Computing the CSC-score requires
> only citation counts and ranked term lists — no LLM, no embedding model, no
> API key. The dependencies below (NumPy, Pandas, SciPy, Matplotlib) are the
> complete list. LLM components enter only in the optional vocabulary-alignment
> layer (`aws-align align`), and that layer ships with a fully offline
> `--dry-run` mode that rebuilds the consensus alignment table from archived
> raw model responses.

## Demo dataset

The bundled demo tables (`data/demo/`) describe a 3,138-publication plastic
recycling corpus clustered into six subdomains. Running the diagnostic on them
reproduces a canonical AWS-positive result, regression-tested to ±0.005:
**CSC = 0.402** (S_cross = 0.617, D_bar = 0.976).

## Repository structure

```
awareness-without-synthesis/
│
├── aws_align/          # Core implementation (csc, fragmentation, align, io, cli)
├── data/demo/          # Plastic recycling demo tables (reproduce CSC = 0.402)
├── tests/              # Regression suite asserting the canonical demo values
├── examples/           # Example figures and notebooks
├── README.md
├── pyproject.toml
└── LICENSE
```

## Installation

```bash
git clone https://github.com/E4CE-UA/awareness-without-synthesis.git
cd awareness-without-synthesis
pip install -e .
```

Requires Python 3.9–3.12. Editable install (`-e`) is recommended so the
command-line demo below finds the bundled `data/demo/` tables.

## Quick start (Python)

Run the diagnostic on the bundled demo data:

```python
from aws_align import compute_csc, load_insularity, load_divergence

ins = load_insularity("data/demo/citation_cluster_insularity.csv")
div = load_divergence("data/demo/rbo_fragmentation.csv")

result = compute_csc(ins, div)
print(result)
# CSC diagnostic — corpus
#   clusters      = 6
#   S_cross       = 0.617    (61.7% of citations cross cluster boundaries)
#   D_bar         = 0.976    (near-total vocabulary divergence)
#   CSC = 1-AWS   ≈ 0.402    → fragmented: awareness without synthesis
```

High structural coupling (S_cross = 0.617) combined with near-total vocabulary
divergence (D_bar = 0.976): the subdomains cite each other extensively yet
write in almost entirely non-overlapping vocabularies.

## Command line

Run from the repository root; every subcommand defaults to the bundled demo
data when no `--input` is given.

```bash
aws-align diagnose             # CSC-score + size-proportional null model
aws-align map                  # render the pairwise fragmentation map
aws-align align --dry-run      # rebuild the vocabulary-alignment table offline
aws-align <subcommand> --help  # full options
```

## Tests

The test suite is a regression suite: it asserts the canonical demo values
(S_cross = 0.617, D_bar = 0.976, CSC = 0.402, 6 clusters),
plus invariants of the null model and the input loaders.

```bash
pip install -e ".[dev]"
pytest
```

## Data and archival

- **Software (this package):** archived at Zenodo, DOI
  [10.5281/zenodo.XXXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXXX). <!-- TODO: DOI del depósito del paquete -->
- **Datasets, pre-computed embeddings, and verified citation pairs:** archived
  in a separate Zenodo deposit, DOI
  [10.5281/zenodo.YYYYYYYY](https://doi.org/10.5281/zenodo.YYYYYYYY). <!-- TODO: DOI del depósito de datos (bibliometric_api v2) -->

## Citation

If you use this software in academic work, please cite it using the metadata
in [`CITATION.cff`](CITATION.cff) or the software DOI above. Citation
information for the accompanying publication will be added here upon
publication.

## License

This project is distributed under the [MIT License](LICENSE).

## Authors

- Ana Bossler — University of Alicante
- Enric Bas — University of Alicante
- Andrés Fullana — University of Alicante
