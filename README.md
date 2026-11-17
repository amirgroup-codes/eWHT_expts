# Protein Fitness Landscapes Are Simpler Under Evolutionary Distributions

Code to reproduce the results for the paper "Protein Fitness Landscapes Are
Simpler Under Evolutionary Distributions".

## Software package

We provide the Python package `ewht` available at [https://github.com/amirgroup-codes/ewht](https://github.com/amirgroup-codes/ewht) to facilitate analyzing combinatorial datasets with the evolutionary Walsh–Hadamard Transform (eWHT). The package includes our computationally efficient algorithms computing the eWHT in $O(N \log N)$ time, sparse recovery algorithms,
and utilities for constructing evolutionary measures from multiple sequence alignments and protein language models.

## Setup

```bash
conda env create -f ewht.yml && conda activate ewht
```

Download the processed data (CSVs) from Figshare into `data/`:
https://doi.org/10.6084/m9.figshare.32869433. ESM2-650M weights are read from
`models/esm2_t33_650M_UR50D.pt` (only needed to recompute the ESM scores / ps or new embeddings).

## Contents

| folder | what it does |
|---|---|
| `pwht/` | eWHT (p-biased Walsh–Hadamard transform) and its inverse (`pbias.py`), design matrices for compressed sensing (`cs.py`), landscape helpers (`gen_utils.py`) |
| `data/` | `raw_data/` + `fasta/` → `processed_data/process_data.py` → `processed_data/*.csv`; `ps/esm_wildtype_marginal/process_ps.py` scores every variant with ESM2 (wildtype-marginal LLR), splits each library at the median ESM score (ESM bin) and median fitness (experimental bin), and sets `p_i` = mutation frequency in the ESM bin; also logs the Spearman correlation of ESM score vs fitness |
| `fitness_landscapes/` | truncation analyses: R² on the ESM bin, MSE on the experimental bin, the ESM-bin UMAP figure, and synthetic-landscape controls |
| `compressed_sensing/` | Lasso recovery from n measured variants in the eWHT vs WHT basis (ESM-bin train/test pools) |
| `visualization/` | three-panel per-landscape summary figures (pairwise interactions, R² by epistasis order, ΔR² per interaction) |

## Reproducing the results

Run from the repository root.

```bash
# data (only if regenerating from raw data / rescoring with ESM2)
python data/processed_data/process_data.py
python data/ps/esm_wildtype_marginal/process_ps.py            # --spearman-only: from existing esm_data, no model

# fitness landscapes
python fitness_landscapes/r2_truncation.py      # R^2 on the ESM bin  -> fitness_landscapes/plots/esm_bin/
python fitness_landscapes/mse_truncation.py     # MSE on the experimental bin -> fitness_landscapes/plots/exp_bin/
python fitness_landscapes/plot_distribution.py  # ESM-bin UMAP figure -> fitness_landscapes/distribution/umap/
python fitness_landscapes/synthetic_test.py     # synthetic-landscape controls -> fitness_landscapes/synthetic/

# compressed sensing
python compressed_sensing/compressed_sensing.py       # -> compressed_sensing/cs/esm_bin/
python compressed_sensing/plot_compressed_sensing.py  # -> compressed_sensing/cs/esm_bin/plots/

# summary figures (needs fitness_landscapes/plots/esm_bin/)
python visualization/visualization.py           # -> visualization/plots/
```

`r2_truncation.py` and `mse_truncation.py` also print the pooled paper numbers (% fewer coefficients and
third-order-and-higher interactions for eWHT vs WHT over the 8 landscapes, excluding `cr9114_h3_mean_dev`).
Saved results are reused where possible: `r2_truncation.py --overwrite`, `mse_truncation.py` (recomputes; `--replot`
redraws), `compressed_sensing.py --force`, `synthetic_test.py --overwrite`.
