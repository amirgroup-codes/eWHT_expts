"""Shared helpers for the fitness-landscape analyses (R^2 on the ESM bin, MSE on the experimental bin,
synthetic landscapes): paths, landscape loading, evaluation bins, centring, and the incremental
truncation curves.

Conventions
- y is the full landscape over all 2^L genotypes (site 0 = most significant bit); unmeasured genotypes are
  filled with the mean measured fitness (label_mask marks the measured ones).
- WHT = eWHT with every p_i = 0.5. Each transform subtracts its own mean before transforming:
  the uniform mean for the WHT, the mu_p-weighted mean for the eWHT.
- Coefficients are ranked by c^2 over the whole landscape; truncation curves are scored on an eval bin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

FITNESS_DIR = Path(__file__).resolve().parent
ROOT = FITNESS_DIR.parent
PROCESSED_DIR = ROOT / "data" / "processed_data"
PS_DIR = ROOT / "data" / "ps" / "esm_wildtype_marginal"
ESM_DATA_DIR = PS_DIR / "esm_data"
PLOTS_DIR = FITNESS_DIR / "plots"

sys.path.append(str(ROOT / "pwht"))
from gen_utils import sampling_function_df  # noqa: E402
from pbias import ipfwht, pfwht  # noqa: E402

# Eval bins (columns of data/ps/esm_wildtype_marginal/esm_data/{dataset}.csv).
BIN_COLUMNS = {"esm": "DMS_score_bin_esm", "experimental": "DMS_score_bin_experimental"}
DEFAULT_SKIP = ("cr9114_h3_mean_dev",)


def interaction_order_array(L: int) -> np.ndarray:
    """Interaction order (number of sites) of each coefficient index 0..2^L-1."""
    return np.array([bin(i).count("1") for i in range(1 << L)], dtype=int)


def dataset_names(skip=DEFAULT_SKIP) -> list[str]:
    """Datasets with a processed CSV and an ESM wildtype-marginal ps vector."""
    return [c.stem for c in sorted(PROCESSED_DIR.glob("*.csv"))
            if c.stem not in set(skip) and (PS_DIR / f"{c.stem}.npy").exists()]


def normalize_genotype(genotype: object, L: int) -> str | None:
    bits = "".join(ch for ch in str(genotype) if ch in "01").zfill(L)
    return bits if len(bits) == L else None


def load_landscape(name: str) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """(y, L, ps, label_mask): full fitness vector (NaNs filled with the mean), ps as float64."""
    ps = np.load(PS_DIR / f"{name}.npy").astype(np.float64)
    L = len(ps)
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv", dtype={"genotype": str, "genotype_sequence": str})
    df["genotype"] = df["genotype"].astype(str).str.strip().str.zfill(L)
    y = sampling_function_df(df, L)
    label_mask = ~np.isnan(y)
    if not label_mask.all():
        y = np.where(label_mask, y, float(np.nanmean(y)))
    return y, L, ps, label_mask


def bin_indices(name: str, L: int, label_mask: np.ndarray, which: str = "esm") -> np.ndarray:
    """Sorted genotype indices in the eval bin ('esm' or 'experimental') that have a measured fitness."""
    d = pd.read_csv(ESM_DATA_DIR / f"{name}.csv", dtype={"genotype": str})
    keep = d[BIN_COLUMNS[which]].fillna(0).astype(int) == 1
    bits = (normalize_genotype(g, L) for g in d.loc[keep, "genotype"])
    idx = np.unique([int(b, 2) for b in bits if b is not None])
    if idx.size == 0:
        raise ValueError(f"{name}: empty {which} bin")
    return idx[label_mask[idx]]


def mu_p(ps: np.ndarray) -> np.ndarray:
    """Product-measure probability of every genotype."""
    w = np.array([1.0])
    for p in ps:
        w = np.kron(w, [1 - p, p])
    return w


def center(y: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """y minus its mean under mu_p (the plain mean when every p_i = 0.5)."""
    return y - (y.mean() if np.allclose(ps, 0.5) else float(y @ mu_p(ps)))


def plotted_k_span(k_ewht: int | None, k_wht: int | None) -> int | None:
    """x-range of the truncation plots: 4/3 x the larger k95 (at least 20); if exactly one k95 is >= 1000,
    use the smaller one so the other curve stays readable."""
    if k_ewht is None and k_wht is None:
        return None
    if k_ewht is not None and k_wht is not None:
        k = min(k_ewht, k_wht) if (k_ewht < 1000) != (k_wht < 1000) else max(k_ewht, k_wht)
    else:
        k = k_ewht if k_ewht is not None else k_wht
    return max(int(np.ceil(4.0 / 3.0 * k)), 20)


def basis_block(s_idx: np.ndarray, x_idx: np.ndarray, ps: np.ndarray, L: int) -> np.ndarray:
    """Columns of the inverse eWHT (as ipfwht) for coefficient indices s, evaluated at genotypes x."""
    bits = np.arange(L)
    p = np.asarray(ps, dtype=np.float64)[L - 1 - bits]
    fac = np.where((x_idx[:, None] >> bits) & 1 == 0, np.sqrt(p / (1 - p)), -np.sqrt((1 - p) / p))
    sb = ((s_idx[:, None] >> bits) & 1).astype(bool)
    return np.stack([np.prod(np.where(m, fac, 1.0), axis=1) for m in sb], axis=1)


def coefficients(y: np.ndarray, ps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centred y, coefficients, indices ranked by c^2 descending)."""
    yc = center(y, ps)
    c = pfwht(yc.copy(), ps=ps, norm="expectation")
    return yc, c, np.argsort(c ** 2)[::-1]


def r2_curve(y: np.ndarray, ps: np.ndarray, L: int, ev: np.ndarray, kmin: int = 20, pad: float = 4.0 / 3.0):
    """R^2 on ev (clipped at 0) as the top-k coefficients are added one at a time.

    Same values as reconstructing with ipfwht(top-k coefficients) and r2_score on ev for every k, computed
    incrementally on ev only. Runs until R^2 >= 0.95 and the curve has max(ceil(pad*k95), kmin) points.
    Returns (k95, #order>=3 terms among the top k95, curve, c); k95 = 2^L if 0.95 is never reached.
    """
    yc, c, order = coefficients(y, ps)
    t = yc[ev]
    sst = ((t - t.mean()) ** 2).sum()
    pred = np.zeros(len(ev))
    curve, k95, N = [], None, len(y)
    for start in range(0, N, 512):
        s = order[start:start + 512]
        P = pred[:, None] + np.cumsum(basis_block(s, ev, ps, L) * c[s], axis=1)
        r2 = 1 - ((t[:, None] - P) ** 2).sum(0) / sst
        curve.extend(np.maximum(r2, 0).tolist())
        pred = P[:, -1]
        if k95 is None and (r2 >= 0.95).any():
            k95 = start + int(np.argmax(r2 >= 0.95)) + 1
        if k95 is not None and len(curve) >= min(N, max(int(np.ceil(pad * k95)), kmin)):
            break
    k95 = k95 or N
    n_ge3 = int((interaction_order_array(L)[order[:k95]] >= 3).sum())
    return k95, n_ge3, np.asarray(curve), c


def r2_curves(y: np.ndarray, ps: np.ndarray, L: int, ev: np.ndarray) -> dict:
    """eWHT and WHT R^2 curves on ev, both cut to the shared plotted span (plotted_k_span)."""
    wht_ps = np.full(L, 0.5)
    ke, ne, cv_e, c_e = r2_curve(y, ps, L, ev)
    kw, nw, cv_w, c_w = r2_curve(y, wht_ps, L, ev)
    x_max = min(len(y), plotted_k_span(ke, kw))
    if len(cv_e) < x_max:
        cv_e = r2_curve(y, ps, L, ev, kmin=x_max)[2]
    if len(cv_w) < x_max:
        cv_w = r2_curve(y, wht_ps, L, ev, kmin=x_max)[2]
    return dict(k95_ewht=ke, k95_wht=kw, ge3_ewht=ne, ge3_wht=nw, x_max=x_max,
                r2_ewht=cv_e[:x_max], r2_wht=cv_w[:x_max], c_ewht=c_e, c_wht=c_w)


def pooled_pct_fewer(k_wht, k_ewht) -> float:
    """100 * (sum k_WHT - sum k_eWHT) / sum k_WHT: % fewer coefficients eWHT needs, pooled over landscapes."""
    k_wht, k_ewht = np.asarray(k_wht, float), np.asarray(k_ewht, float)
    return float(100.0 * (k_wht.sum() - k_ewht.sum()) / k_wht.sum())
