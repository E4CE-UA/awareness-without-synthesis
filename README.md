
# Awareness Without Synthesis (AWS)

**Awareness Without Synthesis (AWS)** is a diagnostic framework designed to detect hidden vocabulary fragmentation within scholarly knowledge graphs.

While high citation flow between research domains is often taken as evidence of knowledge integration, this inference can fail when subdomains cite one another extensively while describing related phenomena using entirely different vocabularies. In this scenario, the literature is *structurally aware* of the connection, but it has not *synthesized* that connection into a shared, searchable language.

This repository provides the reference implementation for the **Cross-Subdomain Coherence score (CSC-score)**, a diagnostic for measuring this structural–lexical mismatch.

---

## 1. What the diagnostic measures

The CSC-score is derived from two complementary metrics based on a partition of a scholarly corpus.

### A. Cross-cluster citation fraction ($S_{cross}$)

This measures how frequently subdomains cite one another:


$$S_{cross} = \frac{E_{cross}}{E_{within\_corpus}}$$

* **$E_{within\_corpus}$:** Total citation edges where both source and target belong to the corpus.
* **$E_{cross}$:** The subset of those edges that connect different clusters.

### B. Vocabulary divergence ($D_{ij}$)

For every unordered cluster pair $(i, j)$, divergence is calculated using the Rank-Biased Overlap (RBO) of their characteristic-term lists:


$$D_{ij} = 1 - RBO(L_i, L_j)$$


The framework utilizes two summaries:

* **Unweighted Mean Divergence ($\bar{D}$):** The simple average of all $D_{ij}$.
* **Exposure-Weighted Divergence ($\bar{D}_w$):** Weighs divergence by the intensity of cross-cluster citation flow:
* $w_{ij} = \frac{e_i \cdot e_j}{\sum e_p \cdot e_q}$
* $\bar{D}_w = \sum (w_{ij} \cdot D_{ij})$
*(Where $e_i$ is the number of cross-cluster citation edges originating from cluster $i$.)*



### C. Fragmentation and Coherence

The final metrics are:


$$\text{fragmentation} = S_{cross} \cdot \bar{D}_w$$

$$\text{CSC} = 1 - \text{fragmentation}$$

* **CSC near 1:** Clusters are citation-connected and use aligned vocabularies.
* **Low CSC:** High structural awareness coexists with strong vocabulary divergence.

---

## 2. Canonical Demo (Plastic Recycling Corpus)

| Quantity | Value |
| --- | --- |
| Clusters | 6 |
| Within-corpus citation edges | 1,951 |
| Cross-cluster citation edges | 1,204 |
| **$S_{cross}$** | **0.617** |
| Unweighted $\bar{D}$ | 0.976 |
| Exposure-weighted $\bar{D}_w$ | 0.969 |
| **fragmentation** | **0.598** |
| **CSC** | **0.402** |

---

## 3. Implementation and Usage

### Quick Start: Python

```python
from aws_align import compute_csc, load_divergence, load_insularity

insularity = load_insularity("data/demo/citation_cluster_insularity.csv")
divergence = load_divergence("data/demo/rbo_fragmentation.csv")

result = compute_csc(insularity, divergence, corpus="Plastic recycling")
print(result)

```

### Command Line Interface

The package provides a built-in CLI:

* **Diagnose:** `aws-align diagnose`
* **Map:** `aws-align map --insularity [file] --divergence [file]`
* **Alignment:** `aws-align align` (Offline mode requires no API keys)

---

## 4. Key Methodological Properties

* **Encoder-Free:** The core CSC computation relies on citation counts and ranked-vocabulary divergence, not expensive embedding models.
* **Partition-Dependent:** Results change based on how the corpus is clustered (e.g., number of clusters $k$). Always report clustering parameters.
* **Validation:** The optional LLM vocabulary alignment layer provides cross-model consensus, but it is an *auxiliary* step; the CSC diagnostic itself is deterministic.

---

## 5. License & Authors

* **License:** MIT
* **Authors:** Ana Bossler, Enric Bas, Andrés Fullana (University of Alicante)

---

*Would you like to understand the theoretical implications of the "Awareness Without Synthesis" framework in the context of information retrieval, or are you looking for assistance with the repository's setup?*
