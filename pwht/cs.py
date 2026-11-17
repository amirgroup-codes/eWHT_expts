"""Design matrices (WHT and eWHT basis functions evaluated at sequences) for compressed sensing.

Columns are ordered by interaction degree: the constant, then all single sites, then all pairs, ...
(the order of itertools.combinations over r = 0..L).
"""
from itertools import combinations

import numpy as np


def _degree_order(L: int) -> list[int]:
    """Column index (bitmask, bit i = site i) of each interaction, in degree order."""
    return [sum(1 << i for i in combo) for r in range(L + 1) for combo in combinations(range(L), r)]


def p_walsh_hadamard_from_seqs(bin_seqs, ps, norm: str = "expectation") -> np.ndarray:
    """eWHT design matrix.

    bin_seqs: (N, L) array of 0/1 (0 = wildtype, 1 = mutation); ps: (L,) mutation probabilities.
    Each site contributes sqrt(p/q) if wildtype and -sqrt(q/p) if mutated (q = 1 - p).
    norm: "expectation" (default) or "sqrt" (columns divided by sqrt(2^L)).
    """
    bin_seqs = np.asarray(bin_seqs, dtype=np.int8)
    ps = np.asarray(ps, dtype=np.float64)
    N, L = bin_seqs.shape
    coeff0 = np.sqrt(ps / (1.0 - ps))
    coeff1 = -np.sqrt((1.0 - ps) / ps)

    # Columns in bitmask order: column (s | 1<<j) = column s * (site j factor).
    X_lex = np.empty((N, 1 << L), dtype=np.float64)
    X_lex[:, 0] = 1.0
    span = 1
    for j in range(L):
        c = np.where(bin_seqs[:, [j]] == 1, coeff1[j], coeff0[j])
        X_lex[:, span:2 * span] = X_lex[:, :span] * c
        span <<= 1
    X = X_lex[:, _degree_order(L)]

    if norm == "expectation":
        return X
    if norm == "sqrt":
        return X / np.sqrt(1 << L)
    raise ValueError(f"norm must be 'expectation' or 'sqrt', got {norm!r}")


def walsh_hadamard_from_seqs(bin_seqs, normalize: bool = False) -> np.ndarray:
    """WHT design matrix (+-1 parities), same column order as p_walsh_hadamard_from_seqs."""
    S = np.asarray(bin_seqs, dtype=np.int8)
    L = S.shape[1]
    masks = np.array([[1 if i in combo else 0 for i in range(L)]
                      for r in range(L + 1) for combo in combinations(range(L), r)], dtype=np.int8)
    X = (1 - 2 * ((S @ masks.T) % 2)).astype(float)
    return X / np.sqrt(1 << L) if normalize else X
