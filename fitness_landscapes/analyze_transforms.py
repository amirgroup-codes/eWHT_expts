from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
FITNESS_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed_data"
PROCESSED_MSA_DIR = ROOT / "data" / "processed_msa"
DEFAULT_PRIOR = "msa"
PS_DIR = ROOT / "data" / "ps" / DEFAULT_PRIOR
PLOT_DIR = FITNESS_DIR / "plots" / DEFAULT_PRIOR
DATA_DIR = FITNESS_DIR / "data" / DEFAULT_PRIOR
USE_ESM_PRIOR = DEFAULT_PRIOR == "esm"

sys.path.append(str(ROOT / "pwht"))
from gen_utils import sampling_function_df
from pbias import pfwht, ipfwht
from plot_utils import interaction_order_array, generate_and_render_r2_plots
from plot_distribution import PROTEIN_METADATA, _read_a2m, _infer_wt_mut_from_processed_data, dataset_key_from_csv

SEEDS = 5


def configure_prior(prior: str) -> None:
    global PS_DIR, PLOT_DIR, DATA_DIR, USE_ESM_PRIOR
    if prior not in {"msa", "esm"}:
        raise ValueError(f"prior must be one of 'msa' or 'esm', got {prior!r}")
    PS_DIR = ROOT / "data" / "ps" / prior
    PLOT_DIR = FITNESS_DIR / "plots" / prior
    DATA_DIR = FITNESS_DIR / "data" / prior
    USE_ESM_PRIOR = prior == "esm"

def get_r2_cache_paths(csv_path: Path) -> tuple[Path, Path]:
    return (
        PLOT_DIR / f"{csv_path.stem}_pwht.npy",
        PLOT_DIR / f"{csv_path.stem}_wht.npy",
    )


def load_cached_r2_arrays(csv_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    pwht_path, wht_path = get_r2_cache_paths(csv_path)
    if pwht_path.exists() and wht_path.exists():
        try:
            return np.load(pwht_path), np.load(wht_path)
        except Exception as exc:
            print(f"Warning: failed to load cached R^2 arrays for {csv_path.name}: {exc}")
    return None


def save_r2_cache_arrays(csv_path: Path, r2_pwht: np.ndarray, r2_wht: np.ndarray) -> None:
    pwht_path, wht_path = get_r2_cache_paths(csv_path)
    np.save(pwht_path, r2_pwht, allow_pickle=False)
    np.save(wht_path, r2_wht, allow_pickle=False)


def load_msa_sequences(dataset_key: str) -> tuple[list[str], list[int | str]] | None:
    if dataset_key not in PROTEIN_METADATA:
        return None
    positions, msa_name = PROTEIN_METADATA[dataset_key]
    msa_path = PROCESSED_MSA_DIR / f"{msa_name}.a2m"
    if not msa_path.exists():
        print(f"Skipping MSA analysis for {dataset_key}: missing MSA file {msa_path}")
        return None
    sequences = _read_a2m(msa_path)
    return sequences, positions


def _mutant_matches(aa: str, mut: str) -> bool:
    aa = aa.upper()
    mut = mut.upper()
    if mut == 'X':
        return aa in {'V', 'L', 'I'}
    return aa == mut


def _compute_plotted_k_span(k_cross_pwht: int | None, k_cross_wht: int | None) -> int | None:
    if k_cross_pwht is None and k_cross_wht is None:
        return None
    if k_cross_pwht is not None and k_cross_wht is not None:
        if (k_cross_pwht < 1000) != (k_cross_wht < 1000):
            return max(int(np.ceil((4.0 / 3.0) * min(k_cross_pwht, k_cross_wht))), 20)
        return max(int(np.ceil((4.0 / 3.0) * max(k_cross_pwht, k_cross_wht))), 20)
    k_known = k_cross_pwht if k_cross_pwht is not None else k_cross_wht
    return max(int(np.ceil((4.0 / 3.0) * k_known)), 20)


def _report_k95(k_cross: int | None, x_max_var: int) -> int:
    """Report k at R^2=0.95, or the 4/3 evaluation cap when 0.95 was not reached."""
    return k_cross if k_cross is not None else x_max_var


def _normalize_genotype_string(genotype: object, L: int) -> str | None:
    bits = "".join(ch for ch in str(genotype) if ch in {"0", "1"})
    if len(bits) != L:
        return None
    return bits


def save_msa_selected_rows(
    csv_path: Path, 
    df: pd.DataFrame, 
    index_to_seeds: dict[int, list[int]], 
    L: int
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    genotype_to_seeds_str = {
        format(int(idx), f"0{L}b"): ",".join(map(str, sorted(seeds)))
        for idx, seeds in index_to_seeds.items()
    }
    df_copy = df.copy()
    df_copy["_normalized_g"] = df_copy["genotype"].apply(lambda g: _normalize_genotype_string(g, L))
    selected_rows = df_copy[df_copy["_normalized_g"].isin(genotype_to_seeds_str.keys())].copy()
    selected_rows["seed"] = selected_rows["_normalized_g"].map(genotype_to_seeds_str)
    selected_rows = selected_rows.drop(columns=["_normalized_g"])
    out_path = DATA_DIR / f"{csv_path.stem}.csv"
    selected_rows.to_csv(out_path, index=False)


def build_y_and_label_mask(df: pd.DataFrame, L: int, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    y = sampling_function_df(df, L)
    mask = ~np.isnan(y)
    if np.isnan(y).any():
        missing = int(np.isnan(y).sum())
        fill_value = float(np.nanmean(y)) if not np.all(np.isnan(y)) else 0.0
        print(f"[{dataset_name}] Filling {missing} NaNs with {fill_value:.6g}")
        y = np.where(np.isnan(y), fill_value, y)
    return y, mask


def prepare_y(df: pd.DataFrame, L: int, dataset_name: str, mode: str = 'WHT', ps: np.ndarray = None) -> np.ndarray:
    y = sampling_function_df(df, L).copy()
    if np.isnan(y).any():
        missing = int(np.isnan(y).sum())
        fill_value = float(np.nanmean(y)) if not np.all(np.isnan(y)) else 0.0
        print(f"[{dataset_name}] Filling {missing} NaNs with {fill_value:.6g}")
        y = np.where(np.isnan(y), fill_value, y)
    if mode == 'WHT':
        return y - np.mean(y)
    if mode == 'p-WHT':
        if ps is None:
            raise ValueError("ps vector must be provided for p-WHT weighted mean calculation.")
        probs = np.array([1.0])
        for p in ps:
            probs = np.kron(probs, [1-p, p])
        weighted_mean = np.sum(y * probs)
        return y - weighted_mean
    return y


def top_k_energy_fraction(coeffs: np.ndarray, k: int = 20) -> float:
    """Percent of total coefficient energy contained in the top-k |coeff|^2 terms."""
    energy = np.square(np.asarray(coeffs, dtype=float))
    total_energy = float(energy.sum())
    if total_energy <= 0.0:
        return float("nan")
    k_eff = min(k, energy.size)
    top_indices = np.argsort(energy)[::-1][:k_eff]
    top_energy = float(energy[top_indices].sum())
    return 100.0 * top_energy / total_energy


def print_top_k_energy_fractions(
    dataset_name: str,
    coeffs_wht: np.ndarray,
    coeffs_pwht: np.ndarray,
    k: int = 20,
) -> tuple[float, float]:
    frac_wht = top_k_energy_fraction(coeffs_wht, k=k)
    frac_pwht = top_k_energy_fraction(coeffs_pwht, k=k)
    print(
        f"[{dataset_name}] Top-{k} coefficient energy fraction "
        f"(sum of top-{k} |coeff|^2 / sum of all |coeff|^2):"
    )
    print(f"  WHT:   {frac_wht:.2f}%")
    print(f"  p-WHT: {frac_pwht:.2f}%")
    return frac_wht, frac_pwht


def analyze_dataset(csv_path: Path) -> dict[str, object] | None:
    df = pd.read_csv(csv_path, dtype={"genotype": str, "genotype_sequence": str})
    if "genotype" not in df.columns or "fitness" not in df.columns:
        print(f"Skipping {csv_path.name}: missing genotype/fitness columns")
        return None

    dataset_key = dataset_key_from_csv(csv_path)
    if not USE_ESM_PRIOR:
        msa_data = load_msa_sequences(dataset_key)
        if msa_data is None:
            print(f"Skipping {csv_path.name}: no MSA metadata available.")
            return None

    ps_path = PS_DIR / f"{dataset_key}.npy"
    if not ps_path.exists():
        raise FileNotFoundError(f"Could not find ps file for dataset {dataset_key}")
    ps = np.load(ps_path)
    L = len(ps)
    N = 2 ** L

    y_full, label_mask = build_y_and_label_mask(df, L, csv_path.stem)
    
    y_wht = prepare_y(df, L, csv_path.stem, mode='WHT')
    if len(y_wht) != N:
        raise ValueError(f"Expected full landscape size 2^L for dataset {dataset_key}")

    y_pwht = prepare_y(df, L, csv_path.stem, mode='p-WHT', ps=ps)
    global_coeffs_wht = pfwht(y_wht.copy(), ps=np.full(L, 0.5))
    global_coeffs_pwht = pfwht(y_pwht.copy(), ps=ps)

    # Clean un-shifted energy sorting
    pwht_energy = np.square(global_coeffs_pwht)
    wht_energy = np.square(global_coeffs_wht)    
    idx_sorted_pwht = np.argsort(pwht_energy)[::-1]
    idx_sorted_wht = np.argsort(wht_energy)[::-1]
    orders_pwht = interaction_order_array(L)[idx_sorted_pwht]
    orders_wht = interaction_order_array(L)[idx_sorted_wht]

    top20_energy_frac_wht, top20_energy_frac_pwht = print_top_k_energy_fractions(
        csv_path.stem,
        global_coeffs_wht,
        global_coeffs_pwht,
        k=20,
    )

    shared_seeds_unique_indices = []
    index_to_seeds_map = {}

    cached_r2 = load_cached_r2_arrays(csv_path)
    if cached_r2 is not None:
        r2_pwht_final, r2_wht_final = cached_r2
        print(f"Loaded cached R^2 arrays for {csv_path.name}")
        mean_pwht = np.mean(r2_pwht_final, axis=0)
        mean_wht = np.mean(r2_wht_final, axis=0)
        k_cross_pwht = next((i + 1 for i, v in enumerate(mean_pwht) if v >= 0.95), None)
        k_cross_wht = next((i + 1 for i, v in enumerate(mean_wht) if v >= 0.95), None)
        x_max_var = _compute_plotted_k_span(k_cross_pwht, k_cross_wht) or r2_pwht_final.shape[1]
    else:
        ps_weights = np.array([1.0])
        for p in ps:
            ps_weights = np.kron(ps_weights, [1.0 - p, p])

        if not USE_ESM_PRIOR:
            sequences, positions = msa_data
            _, wt_aas, mut_aas, _ = _infer_wt_mut_from_processed_data(csv_path, sequences, positions)
            int_positions = [(idx, pos) for idx, pos in enumerate(positions) if pos != "NC"]
            msa_skeletons: set[tuple[int, ...]] = set()
            for seq in sequences:
                if any(not isinstance(pos, int) or len(seq) < pos for _, pos in int_positions):
                    continue
                genotype = [0] * L
                for idx, pos in int_positions:
                    aa = seq[pos - 1].upper()
                    wt = wt_aas[idx].upper()
                    mut = mut_aas[idx].upper()
                    genotype[idx] = 1 if (_mutant_matches(aa, mut) and aa != wt) else 0
                msa_skeletons.add(tuple(genotype))

            print(f"Unique structural skeletons extracted from MSA: {len(msa_skeletons)}")

            msa_neighborhood_mask = np.zeros(N, dtype=bool)
            for global_idx in range(N):
                bit_string = format(global_idx, f"0{L}b")
                genotype_tuple = tuple(int(b) for b in bit_string)

                if genotype_tuple in msa_skeletons:
                    msa_neighborhood_mask[global_idx] = True
                    continue

                for skel in msa_skeletons:
                    dist = sum(g1 != g2 for g1, g2 in zip(genotype_tuple, skel))
                    if dist <= 1:
                        msa_neighborhood_mask[global_idx] = True
                        break

            ps_weights = ps_weights * msa_neighborhood_mask

            if np.sum(ps_weights) > 0:
                ps_weights = ps_weights / np.sum(ps_weights)
            else:
                print(f"Warning: MSA neighborhood mask empty for {csv_path.name}, falling back to unmasked weights.")
                ps_weights = ps_weights / np.sum(ps_weights)
        else:
            ps_weights = ps_weights / np.sum(ps_weights)
            print(f"ESM prior mode: sampling from full p_i product distribution (no MSA neighborhood mask)")

        num_draws = int(0.75 * N)

        for seed in range(SEEDS):
            seed_rng = np.random.default_rng(seed)
            sampled_raw_indices = seed_rng.choice(N, size=num_draws, replace=True, p=ps_weights)
            unique_indices = np.unique(sampled_raw_indices)
            
            unique_indices = unique_indices[label_mask[unique_indices]]
            shared_seeds_unique_indices.append(unique_indices)
            
            for global_idx in unique_indices:
                if global_idx not in index_to_seeds_map:
                    index_to_seeds_map[global_idx] = []
                index_to_seeds_map[global_idx].append(seed)

        print(f"Unique indices across seeds: {len(index_to_seeds_map)}")
        save_msa_selected_rows(csv_path, df, index_to_seeds_map, L)

        r2_pwht_runs = [[] for _ in range(SEEDS)]
        r2_wht_runs = [[] for _ in range(SEEDS)]
        k_cross_pwht = None
        k_cross_wht = None
        x_max_var = N

        print(f"Generating Topology-Constrained R^2 plot for {csv_path.name} (L={L})")
        ps_wht = np.full(L, 0.5)

        with tqdm(range(N), desc=f"MSA Topology R^2 {csv_path.name}", unit="k") as pbar:
            for k_idx in range(N):
                k_coefs = k_idx + 1

                for s_idx, indices in enumerate(shared_seeds_unique_indices):
                    if indices.size == 0:
                        r2_pwht_runs[s_idx].append(0.0)
                        r2_wht_runs[s_idx].append(0.0)
                        continue

                    targets_pwht = y_pwht[indices]
                    targets_wht = y_wht[indices]

                    trunc_pwht = np.zeros_like(global_coeffs_pwht)
                    trunc_wht = np.zeros_like(global_coeffs_wht)
                    
                    lim_p = min(k_coefs, len(idx_sorted_pwht))
                    lim_w = min(k_coefs, len(idx_sorted_wht))
                    
                    if lim_p > 0: 
                        trunc_pwht[idx_sorted_pwht[:lim_p]] = global_coeffs_pwht[idx_sorted_pwht[:lim_p]]
                    if lim_w > 0: 
                        trunc_wht[idx_sorted_wht[:lim_w]] = global_coeffs_wht[idx_sorted_wht[:lim_w]]

                    pred_pwht = ipfwht(trunc_pwht, ps)[indices]
                    pred_wht = ipfwht(trunc_wht, ps_wht)[indices]

                    r2_pwht_runs[s_idx].append(max(0.0, r2_score(targets_pwht, pred_pwht)))
                    r2_wht_runs[s_idx].append(max(0.0, r2_score(targets_wht, pred_wht)))

                current_mean_pwht = np.mean([run[-1] for run in r2_pwht_runs])
                current_mean_wht = np.mean([run[-1] for run in r2_wht_runs])
                pbar.update(1)
                pbar.set_postfix_str(
                    f"k={k_coefs}. R^2 p-WHT={current_mean_pwht:.4f}. R^2 WHT={current_mean_wht:.4f}"
                )
                
                if k_cross_pwht is None and current_mean_pwht >= 0.95:
                    k_cross_pwht = k_coefs
                if k_cross_wht is None and current_mean_wht >= 0.95:
                    k_cross_wht = k_coefs

                if k_cross_pwht is not None and k_cross_wht is not None:
                    if (k_cross_pwht < 1000) != (k_cross_wht < 1000):
                        x_max_var = max(
                            int(np.ceil((4.0 / 3.0) * min(k_cross_pwht, k_cross_wht))),
                            20,
                        )
                    else:
                        x_max_var = max(
                            int(np.ceil((4.0 / 3.0) * max(k_cross_pwht, k_cross_wht))),
                            20,
                        )
                    if k_coefs >= x_max_var:
                        break

        r2_pwht_final = np.array(r2_pwht_runs)[:, :x_max_var]
        r2_wht_final = np.array(r2_wht_runs)[:, :x_max_var]
        save_r2_cache_arrays(csv_path, r2_pwht_final, r2_wht_final)

    out_pdf = PLOT_DIR / f"{csv_path.stem}.pdf"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    if len(shared_seeds_unique_indices) > 0:
        avg_eval_size = int(np.mean([len(indices) for indices in shared_seeds_unique_indices]))
    else:
        avg_eval_size = int(r2_pwht_final.shape[1])

    generate_and_render_r2_plots(
        csv_path.stem,
        L,
        r2_pwht_final,
        r2_wht_final,
        global_coeffs_pwht,
        global_coeffs_wht,
        orders_pwht,
        orders_wht,
        k_cross_pwht if k_cross_pwht else x_max_var,
        k_cross_wht if k_cross_wht else x_max_var,
        x_max_var,
        out_pdf,
    )

    return {
        "dataset": csv_path.stem,
        "L": L,
        "k95_wht_r2": _report_k95(k_cross_wht, x_max_var),
        "k95_pwht_r2": _report_k95(k_cross_pwht, x_max_var),
        "plotted_k_span": x_max_var,
        "avg_msa_eval_size": avg_eval_size,
        "top20_energy_frac_wht": top20_energy_frac_wht,
        "top20_energy_frac_pwht": top20_energy_frac_pwht,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WHT and p-WHT landscape truncation.")
    parser.add_argument(
        "--prior",
        choices=["msa", "esm"],
        default=DEFAULT_PRIOR,
        help=f"Evolutionary prior to use (default: {DEFAULT_PRIOR}).",
    )
    args = parser.parse_args()
    configure_prior(args.prior)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(PROCESSED_DIR.glob("*.csv"))

    summaries = []
    for csv_path in csv_paths:
        result = analyze_dataset(csv_path)
        if result is not None:
            summaries.append(result)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(PLOT_DIR / "k95_centered_r2_summary.csv", index=False)
        print(summary_df)


if __name__ == "__main__":
    main()