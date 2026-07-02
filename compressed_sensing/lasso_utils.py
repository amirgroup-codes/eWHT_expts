import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from collections import Counter
from itertools import combinations
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from scipy.stats import ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pwht.cs import walsh_hadamard_from_seqs, p_walsh_hadamard_from_seqs

def get_design_matrix(bin_seqs, ps, basis_type='p-WHT', norm="expectation"):
    if basis_type == 'p-WHT':
        if norm in ["expectation", "sqrt"]:
            X = p_walsh_hadamard_from_seqs(bin_seqs, ps, norm=norm)
            return X, None
        elif norm == "stable":
            X, p_norms = p_walsh_hadamard_from_seqs(bin_seqs, ps, norm=norm)
            return X, p_norms
        else:
            raise ValueError("Invalid norm option for p-WHT")
    else:
        X = walsh_hadamard_from_seqs(bin_seqs, normalize=False)

    return X


def _calculate_train_mean(y_train: np.ndarray, weights_train: np.ndarray | None = None) -> float:
    y_train = np.asarray(y_train, dtype=float)
    if weights_train is None:
        return np.mean(y_train)
    weights_train = np.asarray(weights_train, dtype=float)
    weights_train = np.clip(weights_train, 0.0, None)
    if np.sum(weights_train) <= 0.0:
        return np.mean(y_train)
    return np.sum(y_train * weights_train) / np.sum(weights_train)


def _center_targets_for_split(
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    base_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y_train = y[train_idx].astype(float)
    y_val = y[val_idx].astype(float)
    weights_train = base_weights[train_idx].astype(float) if base_weights is not None else None
    train_mean = _calculate_train_mean(y_train, weights_train)
    y_train -= train_mean
    y_val -= train_mean
    return y_train, y_val


def return_scaling_factors_for_original_p_wht_coeffs(p_norms):
    """
    Returns the scaling factors needed to convert coefficients from the numerically stable "stable" norm back to the original p-WHT coefficients.
    p_norms: (L,) array of Frobenius norms per site from p_walsh_hadamard_from_seqs.
    """
    L = len(p_norms)
    num_coeffs = 1 << L
    
    # 1. Generate all scaling factors in lexicographical order (Kronecker)
    scaling_lex = np.ones(num_coeffs, dtype=np.float64)
    span = 1
    for j in range(L):
        # Every interaction in the new 'span' includes site j
        scaling_lex[span : 2 * span] = scaling_lex[:span] * p_norms[j]
        span <<= 1
    
    # 2. Reorder to degree order (same as the stable_coeffs order)
    degree_idx = [
        sum(1 << i for i in combo) 
        for r in range(L + 1) 
        for combo in combinations(range(L), r)
    ]
    scaling_factors = scaling_lex[degree_idx]
    return scaling_factors


def determine_alpha(
    X,
    y,
    n,
    alphas=None,
    replicates=10,
    random_state=42,
    n_jobs=10,
    active_splits=None,
    y_weights=None,
):
    """
    Determines the optimal regularization parameter for n training points using repeated
    train/validation splits and split-specific target centering.
    """
    if alphas is None:
        alphas = [m * 10**i for i in range(-5, 1) for m in [1, 5]]

    if active_splits is None:
        n_samples = len(y)
        active_splits = [
            np.random.RandomState(random_state + j).choice(n_samples, n, replace=False)
            for j in range(replicates)
        ]

    opt_vals = Parallel(n_jobs=n_jobs)(
        delayed(_determine_alpha_single_rep)(
            X,
            y,
            alphas,
            active_splits[j],
            y_weights=y_weights,
            seed=random_state + j,
        )
        for j in range(replicates)
    )

    cts = Counter(opt_vals)
    opt_alpha = cts.most_common(1)[0][0]
    return opt_alpha

def lasso_grid_search(X, y, n_train, alphas, n_reps=3, random_state=0):
    """
    Takes design matrix and finds best alpha for given n_train using a grid search with multiple replicates.
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(y)
    
    # Ensure we have at least 1 sample left for validation
    if n_train >= n_samples:
        print(f"Skipping n={n_train}: total samples {n_samples} is too small for validation.")
        return None

    all_results = []
    for alpha in tqdm(alphas, desc=f"Grid Search n={n_train}", leave=False):
        metrics = {"r2": [], "spearman": []}
        for rep in range(n_reps):
            indices = rng.permutation(n_samples)
            Xtr, ytr = X[indices[:n_train]], y[indices[:n_train]]
            Xval, yval = X[indices[n_train:]], y[indices[n_train:]]
            
            model = Lasso(alpha=alpha).fit(Xtr, ytr)
            ypred = model.predict(Xval)
            
            metrics["r2"].append(r2_score(yval, ypred))
            metrics["spearman"].append(spearmanr(yval, ypred)[0])
        
        all_results.append({
            'alpha': alpha,
            'mean_r2': np.mean(metrics["r2"]),
            'std_r2': np.std(metrics["r2"]),
            'mean_spearman': np.mean(metrics["spearman"]),
            'std_spearman': np.std(metrics["spearman"])
        })
    return all_results

def run_lasso_expt(
    X,
    y,
    alpha_list,
    ns,
    p_norms=None,
    replicates=50,
    n_jobs=10,
    precalculated_splits=None,
    val_splits=None,
    y_weights=None,
):
    """
    Runs experiment over all ns using a list of optimized alphas.
    Returns calculated metrics, un-normalized coefficients, and raw uncentered predictions.
    """
    n_samples = len(y)
    M_val = X.shape[1]

    scaling_factors = None
    if p_norms is not None:
        scaling_factors = return_scaling_factors_for_original_p_wht_coeffs(p_norms)

    valid_mask = [n < n_samples for n in ns]
    valid_ns = np.array(ns)[valid_mask]
    valid_alphas = np.array(alpha_list)[valid_mask]

    r2_arr = np.zeros((len(valid_ns), replicates))
    spearman_arr = np.zeros((len(valid_ns), replicates))
    coeffs_out = np.zeros((len(valid_ns), replicates, M_val))
    
    # Store validation predictions per 'n' hyperparameter block
    preds_out = []
    
    for i, n in enumerate(valid_ns):
        current_alpha = valid_alphas[i]

        idx_matrix = None
        if precalculated_splits is not None and n in precalculated_splits:
            idx_matrix = precalculated_splits[n]

        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_lasso_rep)(
                X,
                y,
                n,
                current_alpha,
                j,
                n_samples,
                train_idx=idx_matrix[j] if idx_matrix is not None else None,
                val_idx=val_splits[j] if val_splits is not None else None,
                y_weights=y_weights,
            )
            for j in range(replicates)
        )

        n_preds = {}
        # FIXED: Unpack all 5 variables returned by the worker function
        for true_j, r2, rho, raw_coeffs, y_pred_raw in results:
            r2_arr[i, true_j] = r2
            spearman_arr[i, true_j] = rho
            n_preds[true_j] = y_pred_raw  # Map using the true split index
            
            if scaling_factors is not None:
                coeffs_out[i, true_j, :] = raw_coeffs / scaling_factors
            else:
                coeffs_out[i, true_j, :] = raw_coeffs
                
        preds_out.append(n_preds)

    return {
        'n': valid_ns, 
        'r2': r2_arr, 
        'spearman': spearman_arr, 
        'coeffs': coeffs_out,
        'preds': preds_out
    }


def _run_single_lasso_rep(
    X,
    y,
    n,
    current_alpha,
    j,
    n_samples,
    train_idx=None,
    val_idx=None,
    y_weights=None,
):
    """Worker function for run_lasso_expt returning metric values and raw predictions."""
    if train_idx is None:
        train_idx = np.random.RandomState(j).choice(n_samples, n, replace=False)

    if val_idx is None:
        val_mask = np.ones(n_samples, dtype=bool)
        val_mask[train_idx] = False
        val_idx = np.flatnonzero(val_mask)

    X_train, X_val = X[train_idx], X[val_idx]
    
    # Capture true ground-truth targets in raw format
    y_val_raw = y[val_idx].astype(float)
    
    # Isolate training split mean values natively based on target weights
    weights_train = y_weights[train_idx].astype(float) if y_weights is not None else None
    train_mean = _calculate_train_mean(y[train_idx], weights_train)
    
    # Center vectors specifically for penalization alignment
    y_train = y[train_idx].astype(float) - train_mean
    y_val_centered = y_val_raw - train_mean

    model = Lasso(alpha=current_alpha, fit_intercept=False, random_state=j).fit(X_train, y_train)
    ypred_centered = model.predict(X_val)
    
    # Shift predictions right back to uncentered scale units
    y_pred_raw = ypred_centered + train_mean
    
    if len(y_val_centered) <= 1 or np.all(y_val_centered == y_val_centered[0]):
        return j, 0.0, 0.0, model.coef_, y_pred_raw
        
    return j, r2_score(y_val_centered, ypred_centered), spearmanr(y_val_centered, ypred_centered)[0], model.coef_, y_pred_raw

def _determine_alpha_single_rep(X, y, alphas, train_idx, y_weights=None, seed=0):
    """Worker function to process a single alpha-determination replicate."""
    val_mask = np.ones(len(y), dtype=bool)
    val_mask[train_idx] = False
    val_idx = np.flatnonzero(val_mask)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = _center_targets_for_split(
        y, train_idx, val_idx, base_weights=y_weights
    )

    best_alpha = None
    best_score = -np.inf
    for alpha in alphas:
        model = Lasso(alpha=alpha, fit_intercept=False, random_state=seed).fit(X_train, y_train)
        ypred = model.predict(X_val)
        if len(y_val) == 0:
            continue
        score = r2_score(y_val, ypred)
        if score > best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha if best_alpha is not None else alphas[0]