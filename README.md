# Protein Fitness Landscapes Are Simpler Under Evolutionary Distributions

Code to reproduce the results for the paper "Protein Fitness Landscapes Are
Simpler Under Evolutionary Distributions".

## Contents

- `pwht/`: p-biased Walsh-Hadamard transform utilities.
- `fitness_landscapes/`: transform analyses and plots.
- `compressed_sensing/`: sparse recovery experiments.
- `visualization/`: structural figure scripts.

For the `data/` folder, find the MSAs and CSVs available on Figshare: https://doi.org/10.6084/m9.figshare.32869433.

## Quick Check

```bash
python fitness_landscapes/get_stats.py --prior both
python compressed_sensing/get_stats.py --prior both
```
