"""Truncation R^2 on the ESM bin: how many of the largest eWHT vs WHT coefficients reach R^2 = 0.95.

For each landscape, coefficients of the full (centred) landscape are ranked by c^2; the top k are kept and
R^2 is scored on the ESM-bin genotypes (DMS_score_bin_esm == 1), with the experimental fitness as target.

Outputs (plots/esm_bin/):
  {dataset}.pdf                         coefficient spectrum + eWHT / WHT R^2 curves
  {dataset}_pwht.npy, {dataset}_wht.npy R^2 curves, shape (1, plotted k span)   [eWHT, WHT]
  k95_centered_r2_summary.csv           k95, #order>=3 terms among the top k95, eval size, top-20 energy
and prints the pooled paper numbers (% fewer coefficients / order>=3 terms for eWHT vs WHT).

  python fitness_landscapes/r2_truncation.py                  # all datasets (cached curves are reused)
  python fitness_landscapes/r2_truncation.py --datasets beta --overwrite
  python fitness_landscapes/r2_truncation.py --replot         # redraw PDFs from the saved curves only
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # numpy transform: identical results on every machine

import numpy as np
import pandas as pd

import landscapes as ls
from plot_utils import generate_and_render_r2_plots

OUT_DIR = ls.PLOTS_DIR / "esm_bin"
SUMMARY_CSV = OUT_DIR / "k95_centered_r2_summary.csv"


def top_k_energy_pct(c: np.ndarray, k: int = 20) -> float:
    e = np.sort(c ** 2)[::-1]
    return float(100.0 * e[:k].sum() / e.sum())


def k95_from_curve(curve: np.ndarray) -> int | None:
    hit = np.flatnonzero(curve >= 0.95)
    return int(hit[0]) + 1 if hit.size else None


def analyze(name: str, overwrite: bool = False, replot: bool = False) -> dict | None:
    y, L, ps, label_mask = ls.load_landscape(name)
    ev = ls.bin_indices(name, L, label_mask, "esm")
    wht_ps = np.full(L, 0.5)
    yc_e, c_e, rank_e = ls.coefficients(y, ps)
    yc_w, c_w, rank_w = ls.coefficients(y, wht_ps)
    paths = {"ewht": OUT_DIR / f"{name}_pwht.npy", "wht": OUT_DIR / f"{name}_wht.npy"}

    if paths["ewht"].exists() and paths["wht"].exists() and not overwrite:
        r2_e, r2_w = np.load(paths["ewht"])[0], np.load(paths["wht"])[0]
        k_e, k_w = k95_from_curve(r2_e), k95_from_curve(r2_w)
        x_max = r2_e.shape[0]
        k_e, k_w = k_e or x_max, k_w or x_max
    elif replot:
        print(f"Skipping {name}: no saved curves")
        return None
    else:
        print(f"{name}: L={L}, ESM-bin eval set n={len(ev)}", flush=True)
        r = ls.r2_curves(y, ps, L, ev)
        r2_e, r2_w, x_max, k_e, k_w = r["r2_ewht"], r["r2_wht"], r["x_max"], r["k95_ewht"], r["k95_wht"]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(paths["ewht"], r2_e[None, :], allow_pickle=False)
        np.save(paths["wht"], r2_w[None, :], allow_pickle=False)

    order = ls.interaction_order_array(L)
    generate_and_render_r2_plots(
        name, L, r2_e[None, :], r2_w[None, :], c_e, c_w, order[rank_e], order[rank_w],
        k_e, k_w, x_max, OUT_DIR / f"{name}.pdf", y_wht=yc_w, y_pwht=yc_e, ps=ps,
    )
    return dict(
        dataset=name, L=L, k95_wht_r2=k_w, k95_pwht_r2=k_e,
        ge3_wht=int((order[rank_w[:k_w]] >= 3).sum()), ge3_pwht=int((order[rank_e[:k_e]] >= 3).sum()),
        plotted_k_span=x_max, eval_n=len(ev),
        top20_energy_frac_wht=top_k_energy_pct(c_w), top20_energy_frac_pwht=top_k_energy_pct(c_e),
    )


def print_paper_numbers(summary: pd.DataFrame, skip=ls.DEFAULT_SKIP) -> None:
    s = summary[~summary.dataset.isin(skip)]
    all_pct = ls.pooled_pct_fewer(s.k95_wht_r2, s.k95_pwht_r2)
    ge3_pct = ls.pooled_pct_fewer(s.ge3_wht, s.ge3_pwht)
    print(f"\nPooled over {len(s)} landscapes ({', '.join(s.dataset)}):")
    print(f"  eWHT needs {all_pct:.1f}% fewer coefficients than WHT to reach R^2 = 0.95 on the ESM bin")
    print(f"  and {ge3_pct:.1f}% fewer third-order and higher interactions among them.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=None, help="default: every dataset with ps")
    ap.add_argument("--overwrite", action="store_true", help="recompute curves even if saved")
    ap.add_argument("--replot", action="store_true", help="only redraw PDFs from saved curves")
    args = ap.parse_args()

    names = args.datasets or ls.dataset_names(skip=())
    rows = [r for n in names if (r := analyze(n, args.overwrite, args.replot)) is not None]
    if not rows:
        return
    new = pd.DataFrame(rows)
    if SUMMARY_CSV.exists() and args.datasets:  # partial run: keep the other datasets' rows
        old = pd.read_csv(SUMMARY_CSV)
        new = pd.concat([old[~old.dataset.isin(new.dataset)], new])
    summary = new.sort_values("dataset").reset_index(drop=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    if {"ge3_wht", "ge3_pwht"} <= set(summary.columns):
        print_paper_numbers(summary)


if __name__ == "__main__":
    main()
