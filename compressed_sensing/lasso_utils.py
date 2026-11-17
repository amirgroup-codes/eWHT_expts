"""Lasso fits for compressed sensing: per-replicate alpha selection (LassoCV) and the final fits / scores."""
import os
import sys
import warnings

import numpy as np
import sklearn
from joblib import Parallel, delayed
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pwht.cs import p_walsh_hadamard_from_seqs, walsh_hadamard_from_seqs  # noqa: E402

# sklearn >= 1.5 takes an int for `alphas`; older versions spell it `n_alphas`. Same behaviour.
_SK_ALPHAS_TAKES_INT = tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 5)


def get_design_matrix(bin_seqs: np.ndarray, ps: np.ndarray, basis_type: str = "p-WHT") -> np.ndarray:
    """eWHT ('p-WHT', expectation norm) or WHT ('WHT') design matrix for the given 0/1 sequences."""
    if basis_type == "p-WHT":
        return p_walsh_hadamard_from_seqs(bin_seqs, ps, norm="expectation")
    if basis_type == "WHT":
        return walsh_hadamard_from_seqs(bin_seqs, normalize=False)
    raise ValueError(f"basis_type must be 'p-WHT' or 'WHT', got {basis_type!r}")


def determine_alpha_lassocv(X, y, active_splits, cv: int = 5, random_state: int = 42, n_jobs: int = 10) -> np.ndarray:
    """One alpha per training split, chosen by LassoCV on that split's rows only.

    Each LassoCV uses its own 30-alpha path (down to 1e-4 of the all-zero alpha), an unpenalised intercept,
    and KFold(min(cv, n), shuffle, random_state + replicate index).
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    path = dict(eps=1e-4)
    path["alphas" if _SK_ALPHAS_TAKES_INT else "n_alphas"] = 30
    alphas = []
    for rep_idx, train_idx in enumerate(active_splits):
        train_idx = np.asarray(train_idx, dtype=int)
        if len(train_idx) < 2 or len(np.unique(train_idx)) != len(train_idx):
            raise ValueError("Each training split needs at least two unique samples.")
        inner_cv = KFold(n_splits=min(cv, len(train_idx)), shuffle=True, random_state=random_state + rep_idx)
        model = LassoCV(**path, cv=inner_cv, fit_intercept=True, n_jobs=n_jobs)
        model.fit(X[train_idx].astype(float, copy=False), y[train_idx])
        alphas.append(float(model.alpha_))
    return np.asarray(alphas)


def _fit_and_score(X, y, alpha: float, j: int, train_idx: np.ndarray, val_idx: np.ndarray):
    """Lasso on the training rows; returns (j, R^2, Spearman, coefficients, predictions on val)."""
    model = Lasso(alpha=alpha, fit_intercept=True, random_state=j).fit(
        X[train_idx].astype(float, copy=False), y[train_idx].astype(float))
    coef = model.coef_.copy()
    # Predict from the nonzero columns only (the full X[val_idx] is ~17 GB at L = 16).
    nz = np.flatnonzero(coef)
    y_pred = X[np.ix_(val_idx, nz)] @ coef[nz] + model.intercept_
    coef[0] += model.intercept_  # column 0 is the constant basis function
    y_val = y[val_idx].astype(float)
    if len(y_val) <= 1 or np.all(y_val == y_val[0]):
        return j, 0.0, 0.0, coef, y_pred
    return j, r2_score(y_val, y_pred), spearmanr(y_val, y_pred)[0], coef, y_pred


def run_lasso_expt(X, y, alphas: np.ndarray, train_splits: list, val_splits: list, n_jobs: int = 10) -> dict:
    """Fit replicate j on train_splits[j] with alphas[j] and score it on val_splits[j].

    Returns r2 and spearman (n_replicates,), coeffs (n_replicates, n_columns) and preds {j: predictions}.
    """
    reps = len(train_splits)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_score)(X, y, alphas[j], j, train_splits[j], val_splits[j]) for j in range(reps))
    r2, rho = np.zeros(reps), np.zeros(reps)
    coeffs, preds = np.zeros((reps, X.shape[1])), {}
    for j, r2_j, rho_j, coef, y_pred in results:
        r2[j], rho[j], coeffs[j], preds[j] = r2_j, rho_j, coef, y_pred
    return {"r2": r2, "spearman": rho, "coeffs": coeffs, "preds": preds}
