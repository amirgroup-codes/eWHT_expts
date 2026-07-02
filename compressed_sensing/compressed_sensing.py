from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Define paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "fitness_landscapes" / "data" / "esm" # Switch between "msa" and "esm"
PS_BASE_DIR = ROOT / "data" / "ps"
LASSO_BASE_DIR = Path(__file__).resolve().parent / "cs"
sys.path.append(str(ROOT / "pwht"))
sys.path.append(str(ROOT / "fitness_landscapes"))
import lasso_utils

# Helpers
from analyze_transforms import dataset_key_from_csv

def resolve_ps_path(dataset_key: str, method: str) -> Path | None:
    ps_path = PS_BASE_DIR / method / f"{dataset_key}.npy"    
    if ps_path.exists():
        return ps_path
    return None


def load_msa_seed_mapping(dataset_name: str, method: str):
    msa_csv = ROOT / "fitness_landscapes" / "data" / method / f"{dataset_name}.csv"
    if not msa_csv.exists():
        return None
    msa_df = pd.read_csv(msa_csv, dtype={"genotype": str, "seed": str, "estimated_fitness": float})
    if "estimated_fitness" in msa_df.columns:
        msa_df = msa_df[msa_df["estimated_fitness"] == 0].copy()
    msa_df["seed_list"] = msa_df["seed"].fillna("").astype(str).apply(
        lambda s: [item.strip() for item in s.split(",") if item.strip()]
    )
    unique_seeds = sorted({int(item) for row in msa_df["seed_list"] for item in row})
    return msa_df, unique_seeds

def load_ground_truth_landscape(csv_path: Path):
    df = pd.read_csv(csv_path, dtype={"genotype": str, "genotype_sequence": str})
    if "genotype" not in df.columns or "fitness" not in df.columns:
        return None
    if "estimated_fitness" in df.columns:
        df = df[df["estimated_fitness"] == 0].copy()
    if df.empty:
        return None
    seqs = np.stack(df["genotype"].apply(lambda s: np.array([int(x) for x in str(s)])).to_list())
    y = df["fitness"].values.astype(float)
    return seqs, y, df

def biased_sample_indices(seqs: np.ndarray, ps: np.ndarray, sample_size: int, seed: int = 42) -> np.ndarray:
    weights = np.prod(np.where(seqs == 1, ps, 1 - ps), axis=1).astype(float)
    weights = np.clip(weights, 0.0, None)
    if np.all(weights == 0.0):
        weights = np.ones_like(weights)
    weights = weights / np.sum(weights)
    rng = np.random.default_rng(seed)
    bootstrap_draws = rng.choice(len(seqs), size=sample_size, replace=True, p=weights)
    return np.unique(bootstrap_draws)


def build_train_pools(n_samples: int, max_n: int, replicates: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.choice(n_samples, size=max_n, replace=False) for _ in range(replicates)]


def build_train_splits(train_pools: list[np.ndarray], n: int) -> list[np.ndarray]:
    return [pool[:n] for pool in train_pools]


def build_val_splits(n_samples: int, train_pools: list[np.ndarray]) -> list[np.ndarray]:
    return [np.setdiff1d(np.arange(n_samples), pool, assume_unique=True) for pool in train_pools]


def run_landscape_batch(csv_path: Path, method: str, method_dir: Path, args):
    """Processes a single landscape for a specific probability method."""
    dataset_name = csv_path.stem
    dataset_key = dataset_key_from_csv(csv_path)
    ps_path = resolve_ps_path(dataset_key, method)
    
    if ps_path is None:
        print(f"Skipping {dataset_name}: No ps file for method '{method}")
        return None

    data = load_ground_truth_landscape(csv_path)
    if data is None: return None
    
    seqs, y, data_df = data
    ps = np.load(ps_path)
    L = seqs.shape[1]
    n_samples = len(y)
    
    print(f"\n>>> Processing {dataset_name} | L={L}")

    # Setup directories
    p_norms_dir = method_dir / "p_norms"
    coeffs_base_dir = method_dir / "coefficients" / dataset_name
    ys_dir = method_dir / "ys" / dataset_name 
    coeffs_dirs = {
        'WHT': coeffs_base_dir / 'WHT',
        'p-WHT': coeffs_base_dir / 'p-WHT',
    }
    p_norms_dir.mkdir(parents=True, exist_ok=True)
    ys_dir.mkdir(parents=True, exist_ok=True)
    for d in coeffs_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Define training / test split parameters and ns range based on dataset
    if csv_path.name == "beta.csv":
        ns_full = [2, 4, 6]
    elif csv_path.name == "cov2.csv":
        ns_full = list(range(10, 81, 5))
    elif csv_path.name == "mtagbfp.csv":
        ns_full = list(range(10, 101, 10))
    elif csv_path.name in ["cr6261_h1_mean.csv", "cr6261_h9_mean.csv", "his3.csv"]:
        ns_full = list(range(10, 201, 10))
    elif csv_path.name == "cr9114_h3_mean.csv":
        ns_full = [3000, 3500]
    elif csv_path.name == "cov_rbd.csv":
        ns_full = list(range(10, 201, 10))
    else:
        ns_full = list(range(20, 401, 20))

    max_n = max(ns_full)
    msa_seed_data = load_msa_seed_mapping(dataset_name, method)
    if msa_seed_data is None:
        print(f"Skipping {dataset_name}: No MSA seed mapping file found.")
        return None
    msa_df, bootstrap_seeds = msa_seed_data
    if not bootstrap_seeds:
        print(f"Skipping {dataset_name}: No bootstrap seeds found in MSA mapping.")
        return None

    alphas_grid = [m * 10**i for i in range(-5, 0) for m in [1, 5]]
    
    # Accumulate results across bootstrap draws and train/test splits
    results = {
        'WHT': {'r2': [], 'spearman': [], 'best_alphas': [], 'n': [], 'bootstrap_seed': []},
        'p-WHT': {'r2': [], 'spearman': [], 'best_alphas': [], 'n': [], 'bootstrap_seed': []}
    }

    for bootstrap_seed in bootstrap_seeds:
        seed_str = str(bootstrap_seed)
        selected_genotypes = msa_df.loc[msa_df["seed_list"].apply(lambda sl: seed_str in sl), "genotype"].unique()
        if len(selected_genotypes) == 0:
            print(f"Skipping seed {bootstrap_seed}: no genotypes found for this seed.")
            continue
        available_idx = np.where(data_df["genotype"].isin(selected_genotypes))[0]
        if available_idx.size == 0:
            print(f"Skipping seed {bootstrap_seed}: no matching processed sequences for selected genotypes.")
            continue
        sampled_seqs = seqs[available_idx]
        sampled_y = y[available_idx]
        n_samples_bs = len(sampled_y)
        print(f"Seed {bootstrap_seed}: Kept {n_samples_bs} variants ({n_samples_bs / n_samples * 100:.2f}%).")
        ns_bs = ns_full
        max_n_bs = max_n
        if n_samples_bs <= max_n_bs:
            ns_bs = [n for n in ns_full if n < n_samples_bs]
            if not ns_bs:
                raise ValueError(f"Unique sampled points ({n_samples_bs}) insufficient for max n={max_n} in {csv_path.name}")
            max_n_bs = max(ns_bs)
            print(f"    [Bootstrap seed {bootstrap_seed}] Truncating active ns full range down to: {ns_bs}")

        if n_samples_bs <= max_n_bs:
            ns_bs = [n for n in ns_bs if n < n_samples_bs]
            if not ns_bs:
                raise ValueError(f"Unique sampled points ({n_samples_bs}) insufficient for max n={max_n_bs} in {csv_path.name}")
            max_n_bs = max(ns_bs)

        X_wht = lasso_utils.get_design_matrix(sampled_seqs, ps, basis_type='WHT')
        X_pwht, p_norms = lasso_utils.get_design_matrix(sampled_seqs, ps, basis_type='p-WHT', norm='stable')
        p_norms_path = p_norms_dir / f"{dataset_name}.npy"
        if not p_norms_path.exists():
            np.save(p_norms_path, p_norms)

        p_x = np.prod(np.where(sampled_seqs == 1, ps, 1 - ps), axis=1)
        y_raw = sampled_y

        # Build per-replicate train/test pools once for this bootstrap sample
        train_pools = build_train_pools(n_samples_bs, max_n_bs, args.reps, seed=bootstrap_seed)

        for n in ns_bs:
            active_splits = build_train_splits(train_pools, n)
            val_splits = build_val_splits(n_samples_bs, active_splits)
            
            # Setup a temporary dictionary to pair WHT and p-WHT predictions per replicate
            ys_preds = {rep_idx: {} for rep_idx in range(args.reps)}

            for b_type, X, y_weights in [
                ("WHT", X_wht, None),
                ("p-WHT", X_pwht, p_x)
            ]:

                best_alpha = lasso_utils.determine_alpha(
                    X, y_raw, n, alphas=alphas_grid, replicates=5,
                    random_state=42, active_splits=active_splits,
                    y_weights=y_weights
                )
                if best_alpha is None:
                    continue

                final_res = lasso_utils.run_lasso_expt(
                    X, y_raw, [best_alpha], [n], p_norms=p_norms,
                    replicates=args.reps, precalculated_splits={n: active_splits}, val_splits=val_splits,
                    y_weights=y_weights
                )
                results[b_type]['r2'].append(final_res['r2'])
                results[b_type]['spearman'].append(final_res['spearman'])
                results[b_type]['best_alphas'].append(best_alpha)
                results[b_type]['n'].append(n)
                results[b_type]['bootstrap_seed'].append(bootstrap_seed)

                # Extract model coefficients and uncentered predictions
                coeffs = final_res.get('coeffs', None)
                preds_dict_for_n = final_res.get('preds', [{}])[0]  # Get dict for this n

                if coeffs is not None:
                    coeffs_for_n = coeffs[0]  
                    for rep_idx, coef in enumerate(coeffs_for_n):
                        # Save coefficients safely
                        coeff_path = coeffs_dirs[b_type] / f"n{n}_bootstrap{bootstrap_seed}_rep{rep_idx}.npy"
                        np.save(coeff_path, coef)

                        # FIXED: Map raw predictions and ground-truth using the verified split index
                        ys_preds[rep_idx][b_type] = preds_dict_for_n[rep_idx]
                        ys_preds[rep_idx]['y_val'] = y_raw[val_splits[rep_idx]]

            # Global summary metrics check
            print(f'n={n} | p-WHT R^2: {np.mean(results["p-WHT"]["r2"][-1]):.2f} | WHT R^2: {np.mean(results["WHT"]["r2"][-1]):.2f}')

            # Combined Save block
            for rep_idx, pred_dict in ys_preds.items():
                if 'p-WHT' in pred_dict and 'WHT' in pred_dict:
                    y_val_raw = pred_dict['y_val']
                    p_wht_raw = pred_dict['p-WHT']
                    wht_raw = pred_dict['WHT']

                    # Save as a single 3 x n_samples array inside the unified shared folder
                    ys_path = ys_dir / f"n{n}_bootstrap{bootstrap_seed}_rep{rep_idx}.npy"
                    np.save(ys_path, np.vstack([p_wht_raw, wht_raw, y_val_raw]))
                    
    if results:
        save_outputs(dataset_name, method, method_dir, results)

def save_outputs(dataset_name, method, method_dir, results):
    """Saves raw data to subdirectory."""

    all_rows = []
    for b_type, res in results.items():
        ns = res['n']
        for i, n in enumerate(ns):
            r2_values = np.array(res['r2'][i]).flatten()
            spearman_values = np.array(res['spearman'][i]).flatten()
            alpha = res['best_alphas'][i]
            bootstrap_seed = res['bootstrap_seed'][i]
            for rep_idx, (r2_val, spear_val) in enumerate(zip(r2_values, spearman_values)):
                all_rows.append({
                    'n': n,
                    'basis_type': b_type,
                    'bootstrap_seed': bootstrap_seed,
                    'replicate': rep_idx,
                    'r2': r2_val,
                    'spearman': spear_val,
                    'best_alpha': alpha,
                })
    
    df = pd.DataFrame(all_rows)
    csv_path = method_dir / f"{dataset_name}.csv"
    df.to_csv(csv_path, index=False)

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'WHT': 'tab:blue', 'p-WHT': 'tab:orange'}
    for b_type in results.keys():
        subset = df[df['basis_type'] == b_type]
        for i, metric in enumerate(['r2', 'spearman']):
            summary = subset.groupby('n')[metric].agg(['mean', 'std']).reset_index()
            axes[i].errorbar(summary['n'], summary['mean'], yerr=summary['std'], 
                             label=b_type, color=colors[b_type], alpha=0.7, fmt='-o', capsize=3)
            axes[i].set_title(f"{metric.upper()} vs n")
            axes[i].set_xlabel("n")
            axes[i].legend()
    plt.suptitle(f"Method: {method} | Landscape: {dataset_name}")
    plt.tight_layout()
    plt.savefig(method_dir / f"{dataset_name}.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=10, help='Number of reps for the final score at each n')
    parser.add_argument('--methods', type=str, nargs='+', default=["esm"])
    args = parser.parse_args()

    METHODS = args.methods

    csv_paths = sorted(p for p in PROCESSED_DIR.glob("*.csv"))
    
    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"RUNNING METHOD: {method}")
        print('='*60)

        method_dir = LASSO_BASE_DIR / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for path in csv_paths:

            dataset_name = path.stem
            dataset_csv = method_dir / f"{dataset_name}.csv"
            if dataset_csv.exists():
                print(f" - Skipping {dataset_name}: Results already exist at {dataset_csv}")
                continue

            run_landscape_batch(path, method, method_dir, args)

if __name__ == "__main__":
    main()
