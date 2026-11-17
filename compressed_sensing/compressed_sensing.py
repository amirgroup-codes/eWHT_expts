"""Compressed sensing on the ESM bin: Lasso in the eWHT vs WHT basis from n measured variants.

Both the training pool and the test pool are the ESM-bin variants (DMS_score_bin_esm == 1, measured fitness only;
imputed rows are dropped). For each bootstrap seed, every basis gets the SAME uniform draw of training points
(replicate j uses the first n of its pool) and is scored on the same held-out set (the ESM bin minus that
replicate's whole pool), so eWHT and WHT see identical data. Alpha is chosen per replicate by LassoCV on the
training rows only. ps are the ESM wildtype-marginal ps (data/ps/esm_wildtype_marginal/).

Outputs (compressed_sensing/cs/esm_bin/):
  {dataset}.csv                                   n, basis_type, bootstrap_seed, replicate, r2, spearman, best_alpha
  ys/{dataset}/n{n}_bootstrap{s}_rep{j}.npy       rows: eWHT predictions, WHT predictions, measured fitness (held out)
  coefficients/{dataset}/{basis}/n{n}_bootstrap{s}_rep{j}.npy
Figures: plot_compressed_sensing.py.

  python compressed_sensing/compressed_sensing.py                      # all datasets without results yet
  python compressed_sensing/compressed_sensing.py --dataset cov2 --force
  python compressed_sensing/compressed_sensing.py --smoke              # cov2, few n, 2 replicates
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import lasso_utils

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed_data"
PS_DIR = ROOT / "data" / "ps" / "esm_wildtype_marginal"
ESM_DATA_DIR = PS_DIR / "esm_data"
OUT_DIR = Path(__file__).resolve().parent / "cs" / "esm_bin"
BOOTSTRAP_SEEDS = list(range(5))
BASES = ["p-WHT", "WHT"]  # row order of the ys files (then the measured fitness)
BIN_COLUMN = "DMS_score_bin_esm"


def load_landscape(name: str):
    """(0/1 sequences, fitness, ESM-bin mask) for the measured variants of one dataset."""
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv", dtype={"genotype": str, "genotype_sequence": str})
    if "estimated_fitness" in df.columns:
        df = df[df["estimated_fitness"] == 0].copy()
    esm = pd.read_csv(ESM_DATA_DIR / f"{name}.csv", dtype={"genotype": str})
    L = int(max(df["genotype"].str.len().max(), esm["genotype"].str.len().max()))
    df["genotype"] = df["genotype"].str.zfill(L)
    esm["genotype"] = esm["genotype"].str.zfill(L)
    df = df.merge(esm[["genotype", BIN_COLUMN]], on="genotype", how="left")
    seqs = np.stack(df["genotype"].apply(lambda s: np.array([int(x) for x in s])).to_list())
    in_bin = (df[BIN_COLUMN].fillna(0).astype(int) == 1).to_numpy()
    return seqs, df["fitness"].to_numpy(dtype=float), in_bin


def n_grid(name: str, smoke: bool) -> list[int]:
    """Training-set sizes per dataset."""
    if smoke:
        return [10, 15, 20] if name == "cov2" else [10, 20]
    return {
        "beta": [2, 4, 6],
        "cov2": list(range(10, 81, 5)),
        "mtagbfp": list(range(10, 101, 10)),
        "his3": list(range(10, 201, 10)),
        "cov_rbd": list(range(10, 201, 10)),
    }.get(name, list(range(20, 401, 20)))  # cr6261_*, cr9114_*


def run_dataset(name: str, out_dir: Path, reps: int, smoke: bool) -> None:
    seqs_all, y_all, in_bin = load_landscape(name)
    ps = np.load(PS_DIR / f"{name}.npy")
    seqs, y = seqs_all[in_bin], y_all[in_bin]
    L, pool = seqs.shape[1], np.arange(len(y))
    ns = [n for n in n_grid(name, smoke) if n < len(pool)]
    if not ns:
        raise ValueError(f"{name}: ESM bin of {len(pool)} variants is too small for {n_grid(name, smoke)}")
    print(f"\n>>> {name} | L={L} | ESM bin: {len(pool)} of {len(y_all)} measured variants | n = {ns}")

    X = {"WHT": lasso_utils.get_design_matrix(seqs, ps, "WHT"),
         "p-WHT": lasso_utils.get_design_matrix(seqs, ps, "p-WHT")}
    ys_dir, coef_dir = out_dir / "ys" / name, out_dir / "coefficients" / name
    for d in [ys_dir] + [coef_dir / b for b in BASES]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in BOOTSTRAP_SEEDS:
        rng = np.random.default_rng(seed)
        pools = [rng.choice(pool, size=max(ns), replace=False) for _ in range(reps)]
        val_splits = [np.setdiff1d(pool, p, assume_unique=True) for p in pools]
        for n in ns:
            train_splits = [p[:n] for p in pools]
            preds = {}
            for basis in ("WHT", "p-WHT"):
                alphas = lasso_utils.determine_alpha_lassocv(X[basis], y, train_splits, random_state=42)
                res = lasso_utils.run_lasso_expt(X[basis], y, alphas, train_splits, val_splits)
                for j in range(reps):
                    np.save(coef_dir / basis / f"n{n}_bootstrap{seed}_rep{j}.npy", res["coeffs"][j])
                    rows.append(dict(n=n, basis_type=basis, bootstrap_seed=seed, replicate=j, r2=res["r2"][j],
                                     spearman=res["spearman"][j], best_alpha=float(alphas[j]), eval_track="esm"))
                preds[basis] = res["preds"]
            for j in range(reps):
                np.save(ys_dir / f"n{n}_bootstrap{seed}_rep{j}.npy",
                        np.vstack([preds[b][j] for b in BASES] + [y[val_splits[j]]]))
            last = pd.DataFrame(rows[-2 * reps:]).groupby("basis_type")["r2"].mean()
            print(f"  seed {seed} n={n} | eWHT R^2 {last['p-WHT']:.2f} | WHT R^2 {last['WHT']:.2f}")
    pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, help="run one dataset (default: all with ps)")
    ap.add_argument("--reps", type=int, default=10, help="replicates per n and bootstrap seed")
    ap.add_argument("--force", action="store_true", help="re-run datasets that already have results")
    ap.add_argument("--smoke", action="store_true", help="quick check: cov2 (unless --dataset), few n, 2 replicates")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    if args.smoke:
        args.dataset = args.dataset or "cov2"
        args.reps = min(args.reps, 2)

    names = [args.dataset] if args.dataset else sorted(
        p.stem for p in PROCESSED_DIR.glob("*.csv") if (PS_DIR / f"{p.stem}.npy").exists())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (args.out_dir / f"{name}.csv").exists() and not args.force:
            print(f"Skipping {name}: results exist at {args.out_dir / (name + '.csv')} (use --force)")
            continue
        run_dataset(name, args.out_dir, args.reps, args.smoke)


if __name__ == "__main__":
    main()
