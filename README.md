# Protein Fitness Landscapes Are Simpler Under Evolutionary Distributions

Code to reproduce the results for the paper "Protein Fitness Landscapes Are
Simpler Under Evolutionary Distributions".

## Software package

We provide the Python package `ewht` available at [https://github.com/amirgroup-codes/ewht](https://github.com/amirgroup-codes/ewht) to facilitate analyzing combinatorial dataests with the evolutionary Walsh–Hadamard Transform (eWHT). The package includes our computationally efficient algorithms computing the eWHT in $O(N log N )$ time, sparse recovery algorithms,
and utilities for constructing evolutionary measures from multiple sequence alignments and protein language models.

## Contents

- `pwht/`: p-biased Walsh-Hadamard transform utilities.
- `fitness_landscapes/`: transform analyses and plots.
- `compressed_sensing/`: sparse recovery experiments.
- `visualization/`: structural figure scripts.

For the `data/` folder, find the MSAs and CSVs available on Figshare:
https://doi.org/10.6084/m9.figshare.32869433.

To process MSAs, install MMseqs2 separately. The script uses `mmseqs` on `PATH`,
or you can pass the binary explicitly:

```bash
python data/processed_msa/process_msa.py --mmseqs-bin /path/to/mmseqs
```

To regenerate MSA probability vectors, run:

```bash
python data/ps/msa/process_ps.py
```

To regenerate ESM2-650M probability vectors, run:

```bash
python data/ps/esm/process_ps.py
```

This downloads the ESM2-650M model through Hugging Face. By default it uses the standard Hugging Face cache, usually `~/.cache/huggingface`. On a cluster or
machine with limited home-directory space, set `HF_HOME` first:

```bash
export HF_HOME=/path/to/cache
```