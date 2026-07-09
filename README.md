# Awareness Without Synthesis

**Awareness Without Synthesis (AWS)** is a diagnostic framework for detecting hidden vocabulary fragmentation in scholarly knowledge graphs.

The repository provides a reference implementation of **Cross-Cluster Semantic Coverage (CSC)**, a metric designed to quantify whether citation links between research communities are accompanied by meaningful semantic integration.

Instead, it measures the semantic coherence of citation structures, revealing situations in which communities acknowledge one another through citations while continuing to use fragmented or incompatible vocabularies.

---

## Features

- Implementation of the Cross-Cluster Semantic Coverage (CSC) metric.
- Detection of hidden vocabulary fragmentation.
- Analysis of semantic alignment across citation communities.
- Command-line interface.
- Python API.
- Reproducible examples.

---

## Repository structure

```
awareness-without-synthesis/
│
├── aws_align/          # Core implementation
├── tests/              # Unit tests
├── examples/           # Example datasets and notebooks
├── README.md
├── pyproject.toml
└── LICENSE
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/E4CE-UA/awareness-without-synthesis.git
cd awareness-without-synthesis
```

Install the package

```bash
pip install .
```

or install the development version

```bash
pip install -e .
```

---

## Python example

```python
from aws_align import CSC

# Example usage
score = CSC(...)
print(score)
```

---

## Command line

```bash
aws-align --help
```

---

## Requirements

- Python 3.9+
- NumPy
- Pandas
- SciPy
- Matplotlib

---

## Citation

If you use this software in academic work, please cite the accompanying paper.

Citation information will be added upon publication.

---

## License

This project is distributed under the MIT License.

---

## Authors

- Ana Bossler
- Enric Bas
- Andrés Fullana
