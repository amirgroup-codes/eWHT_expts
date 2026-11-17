#!/usr/bin/env python3
"""ESM wildtype-marginal scores, the ESM / experimental median bins, and the eWHT ps.

For each dataset in data/processed_data/:
  1. score every variant with ESM2-650M (sum of wildtype-marginal LLRs of its substitutions; sign-flipped for
     the stems in NEGATIVE_LLR_STEMS so that higher = more fit; WT = 0);
  2. DMS_score_bin_esm = score >= median score; DMS_score_bin_experimental = fitness >= median fitness;
  3. p_i = fraction of ESM-bin variants mutated at site i (clipped to [0.05, 0.95]).

Writes (next to this file):
  esm_data/{stem}.csv                processed data + esm_score + both bin columns
  {stem}.npy, {stem}.png             ps vector and a bar plot of it
  spearman/spearman.csv, {stem}.png  Spearman rho of ESM score vs experimental fitness, with scatter

  python data/ps/esm_wildtype_marginal/process_ps.py [--datasets beta cov2] [--gpu 0]
  python data/ps/esm_wildtype_marginal/process_ps.py --spearman-only   # from existing esm_data, no model run
  python data/ps/esm_wildtype_marginal/process_ps.py --bins-only       # only refresh the experimental bin
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from esm_scoring import (
    DATASETS_METADATA,
    LOCAL_ESM_WEIGHTS,
    compute_wt_llr_matrix,
    llr_column_for_position,
    load_esm_bundle,
    metadata_key,
    mutable_positions,
    score_variants_from_wt_llr,
    site_llr,
)

CURR_DIR = Path(__file__).resolve().parent
ROOT = CURR_DIR.parents[2]
PROCESSED_DIR = ROOT / "data" / "processed_data"
ESM_DATA_DIR = CURR_DIR / "esm_data"
SPEARMAN_DIR = CURR_DIR / "spearman"
SPEARMAN_CSV = SPEARMAN_DIR / "spearman.csv"
SPEARMAN_COLUMNS = ["dataset", "n_sequences", "spearman", "sign_flip"]

P_CLIP = (0.05, 0.95)
# Datasets whose fitness is anti-correlated with the raw LLR: their score is -LLR.
NEGATIVE_LLR_STEMS = {"beta", "cov2", "cov_rbd", "cr6261_h1_mean", "cr6261_h9_mean", "cr9114_h1_mean",
                      "cr9114_h3_mean_dev"}
BETA_SCORED_POSITIONS = [42, 104, 182, 238]


# ---------------------------------------------------------------- scoring
def _score_beta(llr_df: pd.DataFrame, genotypes: list[str], targets: dict) -> np.ndarray:
    """Bit 0 of beta is the non-coding (NC) site, which ESM cannot score: every genotype is scored on the
    4 amino-acid sites only (an NC genotype scores the same as its non-NC counterpart; NC alone = 0)."""
    site = {pos: site_llr(llr_df, llr_column_for_position(llr_df, pos), targets[pos])
            for pos in BETA_SCORED_POSITIONS}
    scores = np.zeros(len(genotypes), dtype=np.float64)
    for i, genotype in enumerate(genotypes):
        g = str(genotype).strip().zfill(5)
        total = 0.0
        for bit, pos in zip(g[1:], BETA_SCORED_POSITIONS):
            if bit == "1":
                total += site[pos]
        scores[i] = total
    return scores


def score_wildtype_marginal(stem: str, genotypes: list[str], device_id: int | None) -> np.ndarray:
    meta = DATASETS_METADATA[metadata_key(stem)]
    model, alphabet, batch_converter, device = load_esm_bundle(device_id)
    llr_df = compute_wt_llr_matrix(meta["wt"], model, alphabet, batch_converter, device)
    if stem == "beta":
        raw = _score_beta(llr_df, genotypes, meta["targets"])
    else:
        raw = score_variants_from_wt_llr(llr_df, mutable_positions(meta), meta["targets"], genotypes)
    scores = -raw if stem in NEGATIVE_LLR_STEMS else raw.astype(np.float64)
    L = len(str(genotypes[0]).strip())
    scores[[str(g).strip().zfill(L) == "0" * L for g in genotypes]] = 0.0  # wildtype
    return scores.astype(np.float64)


# ---------------------------------------------------------------- bins and ps
def assign_experimental_fitness_bin(stem: str, fitness: pd.Series) -> pd.Series:
    """1 for high experimental fitness (>= median). cr9114_h3_mean_dev: WT fitness is 6.0 and a median split
    would label the whole library high, so it uses fitness > 6."""
    fit = pd.to_numeric(fitness, errors="coerce")
    if stem == "cr9114_h3_mean_dev":
        return (fit > 6.0).astype(int)
    return (fit >= float(fit.median())).astype(int)


def compute_pi_from_bin(df: pd.DataFrame, L: int) -> np.ndarray:
    """Per-site mutation frequency among ESM-bin variants."""
    sub = df[df["DMS_score_bin_esm"] == 1]
    if sub.empty:
        raise ValueError("No ESM high-score variants to estimate p_i")
    seqs = np.stack([np.fromiter(str(g).strip().zfill(L), dtype=np.int8) for g in sub["genotype"].astype(str)])
    return np.clip(seqs.mean(axis=0), *P_CLIP)


def plot_pi_vector(stem: str, ps: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(np.arange(1, len(ps) + 1), ps, color="#2ca02c", edgecolor="0.2")
    ax.axhline(0.5, color="0.45", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Position $i$")
    ax.set_ylabel("$p_i$")
    ax.set_title(f"{stem}: ESM wildtype-marginal $p_i$ (high-ESM median bin)")
    ax.set_ylim(P_CLIP[0] - 0.02, P_CLIP[1] + 0.02)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Spearman
def spearman_fitness_vs_esm(fitness: np.ndarray, esm_score: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(fitness) & np.isfinite(esm_score)
    n = int(mask.sum())
    if n < 2:
        return float("nan"), n
    rho = spearmanr(fitness[mask], esm_score[mask])[0]
    return (float("nan") if np.isnan(rho) else float(rho)), n


def plot_spearman(stem: str, fitness: np.ndarray, esm_score: np.ndarray, rho: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.scatter(fitness, esm_score, s=12, alpha=0.45, edgecolors="none")
    ax.set_xlabel("Experimental fitness")
    ax.set_ylabel("ESM score (wildtype marginal)")
    ax.set_title(f"{stem}: Spearman ρ = {rho:.3f}" if np.isfinite(rho) else f"{stem}: Spearman ρ = nan")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def spearman_row(stem: str, df: pd.DataFrame) -> dict:
    fitness = pd.to_numeric(df["fitness"], errors="coerce").to_numpy(dtype=float)
    scores = pd.to_numeric(df["esm_score"], errors="coerce").to_numpy(dtype=float)
    rho, n = spearman_fitness_vs_esm(fitness, scores)
    SPEARMAN_DIR.mkdir(parents=True, exist_ok=True)
    plot_spearman(stem, fitness, scores, rho, SPEARMAN_DIR / f"{stem}.png")
    print(f"{stem}: spearman rho={rho:.6g} n={n}")
    return {"dataset": stem, "n_sequences": n, "spearman": rho, "sign_flip": stem in NEGATIVE_LLR_STEMS}


def write_spearman_csv(rows: list[dict], merge: bool) -> None:
    """merge=True keeps the other datasets' rows already in spearman.csv."""
    new = pd.DataFrame(rows)[SPEARMAN_COLUMNS]
    if merge and SPEARMAN_CSV.exists():
        old = pd.read_csv(SPEARMAN_CSV)
        new = pd.concat([old[~old["dataset"].astype(str).isin(new["dataset"])], new], ignore_index=True)
    SPEARMAN_DIR.mkdir(parents=True, exist_ok=True)
    new.sort_values("dataset")[SPEARMAN_COLUMNS].to_csv(SPEARMAN_CSV, index=False)
    print(f"Wrote {SPEARMAN_CSV} ({len(rows)} datasets updated)")


# ---------------------------------------------------------------- pipeline
def process_dataset(stem: str, device_id: int | None) -> dict:
    df = pd.read_csv(PROCESSED_DIR / f"{stem}.csv", dtype={"genotype": str})
    L = max(len(str(g).strip()) for g in df["genotype"].astype(str))
    genotypes = [str(g).strip().zfill(L) for g in df["genotype"].astype(str)]

    scores = score_wildtype_marginal(stem, genotypes, device_id)
    if stem == "beta" and "mutant" in df.columns:  # the NC-only mutant scores like WT
        nc_only = df["mutant"].astype(str).str.strip().str.upper().str.replace(r"[;\s]", "", regex=True) == "NC"
        scores[nc_only.to_numpy()] = 0.0

    df = df.copy()
    df["esm_score"] = scores
    esm_median = float(df["esm_score"].median())
    df["DMS_score_bin_esm"] = (df["esm_score"] >= esm_median).astype(int)
    df["DMS_score_bin_experimental"] = assign_experimental_fitness_bin(stem, df["fitness"])
    ESM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ESM_DATA_DIR / f"{stem}.csv", index=False)

    ps = compute_pi_from_bin(df, L)
    np.save(CURR_DIR / f"{stem}.npy", ps)
    plot_pi_vector(stem, ps, CURR_DIR / f"{stem}.png")
    print(f"{stem}: esm_median={esm_median:.4g}, n_high_esm={int(df['DMS_score_bin_esm'].sum())}, p_i={ps}")
    return spearman_row(stem, df)


def refresh_experimental_bin(stem: str) -> None:
    path = ESM_DATA_DIR / f"{stem}.csv"
    df = pd.read_csv(path, dtype={"genotype": str})
    df["DMS_score_bin_experimental"] = assign_experimental_fitness_bin(stem, df["fitness"])
    df.to_csv(path, index=False)
    print(f"{stem}: experimental bin n_high={int(df['DMS_score_bin_experimental'].sum())}/{len(df)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--bins-only", action="store_true", help="only refresh DMS_score_bin_experimental in esm_data/")
    ap.add_argument("--spearman-only", action="store_true", help="spearman.csv + scatters from existing esm_data/")
    args = ap.parse_args()
    stems = args.datasets or sorted(p.stem for p in PROCESSED_DIR.glob("*.csv"))

    if args.bins_only:
        for stem in stems:
            refresh_experimental_bin(stem)
        return
    if args.spearman_only:
        rows = [spearman_row(s, pd.read_csv(ESM_DATA_DIR / f"{s}.csv", dtype={"genotype": str})) for s in stems]
        write_spearman_csv(rows, merge=args.datasets is not None)
        return

    if not LOCAL_ESM_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing ESM weights: {LOCAL_ESM_WEIGHTS}")
    rows = [process_dataset(stem, args.gpu) for stem in stems]
    write_spearman_csv(rows, merge=args.datasets is not None)


if __name__ == "__main__":
    main()
