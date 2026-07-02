from __future__ import annotations
from itertools import product
import numpy as np
import pandas as pd

"""
Sampling utils
Functions to grab genotypes and fitness values from DataFrames, and to compute p-biased Walsh-Hadamard transforms.
"""
def get_genotypes(L: int) -> np.ndarray:
	"""Return all binary genotypes in Walsh-Hadamard order (binary count order)."""
	if L <= 0:
		raise ValueError("L must be positive.")
	return np.array(list(product((0, 1), repeat=L)), dtype=np.int8)


def _normalize_genotype_string(genotype: object, L: int) -> str:
	bits = "".join(ch for ch in str(genotype) if ch in {"0", "1"})
	if len(bits) != L:
		raise ValueError(f"Invalid genotype '{genotype}': expected {L} bits, got {len(bits)}")
	return bits


def sampling_function_df(df: pd.DataFrame, n: int) -> np.ndarray:
	"""Map a dataset DataFrame onto ordered genotype measurements.

	Expected columns:
	- genotype: bitstring-like field
	- fitness: scalar numeric value
	"""
	if "genotype" not in df.columns:
		raise KeyError("DataFrame must contain a 'genotype' column.")
	if "fitness" not in df.columns:
		raise KeyError("DataFrame must contain a 'fitness' column.")

	samples = get_genotypes(n)
	ordered_genotypes = ["".join(map(str, row.tolist())) for row in samples]

	work = df[["genotype", "fitness"]].copy()
	work["genotype"] = work["genotype"].apply(lambda g: _normalize_genotype_string(g, n))
	work["fitness"] = pd.to_numeric(work["fitness"], errors="coerce")

	# If there are repeated genotype entries, average their fitness values.
	grouped = work.groupby("genotype", as_index=True)["fitness"].mean()

	y = np.array([grouped.get(g, np.nan) for g in ordered_genotypes], dtype=float)
	return y



"""
Validation utils
"""
def _validate_transform_inputs(y: np.ndarray, ps: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, int, int]:
	a = np.asarray(y, dtype=float)
	if a.ndim != 1:
		raise ValueError("y must be a 1D array.")

	N = len(a)
	if N == 0 or (N & (N - 1)) != 0:
		raise ValueError("Input length must be a power of 2.")

	L = int(np.log2(N))
	p = _validate_ps(ps, L)
	return a.copy(), p, N, L


def _validate_ps(ps: np.ndarray | None, L: int) -> np.ndarray:
	if ps is None:
		return np.full(L, 0.5, dtype=float)
	p = np.asarray(ps, dtype=float)
	if p.shape != (L,):
		raise ValueError(f"ps must have shape ({L},).")
	if np.any((p <= 0.0) | (p >= 1.0)):
		raise ValueError("All ps entries must lie strictly between 0 and 1.")
	return p