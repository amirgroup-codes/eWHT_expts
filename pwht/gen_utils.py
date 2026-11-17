"""Landscape helpers: map a (genotype, fitness) table onto the full 2^L genotype vector, and input checks."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


def get_genotypes(L: int) -> np.ndarray:
    """All 2^L binary genotypes in counting order (site 0 = most significant bit)."""
    if L <= 0:
        raise ValueError("L must be positive.")
    return np.array(list(product((0, 1), repeat=L)), dtype=np.int8)


def _normalize_genotype_string(genotype: object, L: int) -> str:
    bits = "".join(ch for ch in str(genotype) if ch in {"0", "1"})
    if len(bits) != L:
        raise ValueError(f"Invalid genotype '{genotype}': expected {L} bits, got {len(bits)}")
    return bits


def sampling_function_df(df: pd.DataFrame, n: int) -> np.ndarray:
    """Fitness for every genotype in counting order (NaN where unmeasured; repeated genotypes averaged).

    df needs a 'genotype' bitstring column and a numeric 'fitness' column.
    """
    for col in ("genotype", "fitness"):
        if col not in df.columns:
            raise KeyError(f"DataFrame must contain a '{col}' column.")
    work = df[["genotype", "fitness"]].copy()
    work["genotype"] = work["genotype"].apply(lambda g: _normalize_genotype_string(g, n))
    work["fitness"] = pd.to_numeric(work["fitness"], errors="coerce")
    grouped = work.groupby("genotype")["fitness"].mean()
    ordered = ("".join(map(str, row)) for row in get_genotypes(n))
    return np.array([grouped.get(g, np.nan) for g in ordered], dtype=float)


def _validate_transform_inputs(y, ps) -> tuple[np.ndarray, np.ndarray, int, int]:
    a = np.asarray(y, dtype=float)
    if a.ndim != 1:
        raise ValueError("y must be a 1D array.")
    N = len(a)
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError("Input length must be a power of 2.")
    L = int(np.log2(N))
    return a.copy(), _validate_ps(ps, L), N, L


def _validate_ps(ps, L: int) -> np.ndarray:
    if ps is None:
        return np.full(L, 0.5, dtype=float)
    p = np.asarray(ps, dtype=float)
    if p.shape != (L,):
        raise ValueError(f"ps must have shape ({L},).")
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("All ps entries must lie strictly between 0 and 1.")
    return p
