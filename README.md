# Protein Fitness Landscapes Are Simpler Under Evolutionary Distributions

Code to reproduce the results for the paper "Protein Fitness Landscapes Are
Simpler Under Evolutionary Distributions".

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

To regenerate ESM probability vectors, run:

```bash
python data/ps/esm/process_ps.py
```

This downloads the ESM2 model through Hugging Face. By default it uses the
standard Hugging Face cache, usually `~/.cache/huggingface`. On a cluster or
machine with limited home-directory space, set `HF_HOME` first:

```bash
export HF_HOME=/path/to/cache
```

## Quick Check

```bash
python fitness_landscapes/get_stats.py --prior both
python compressed_sensing/get_stats.py --prior both
```
