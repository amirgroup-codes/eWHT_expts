"""Truncation MSE on the experimental bin: how many of the largest eWHT vs WHT coefficients reach MSE <= 0.05.

Same coefficients and ranking as r2_truncation.py (full centred landscape, top-k by c^2), scored by unweighted
MSE on the experimental high-fitness genotypes (DMS_score_bin_experimental == 1). Curves are extended until both
bases reach the threshold, then cut to the same plotted span as the R^2 figures (landscapes.plotted_k_span).

Outputs (plots/exp_bin/):
  {dataset}_mse.pdf, {dataset}_mse_curves.npz, mse_threshold_summary.csv
and prints the pooled paper numbers (% fewer coefficients / order>=3 terms for eWHT vs WHT).

  python fitness_landscapes/mse_truncation.py                    # all datasets except cr9114 (slow)
  python fitness_landscapes/mse_truncation.py --include-cr9114 --jobs 8
  python fitness_landscapes/mse_truncation.py --replot
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import landscapes as ls
from plot_utils import LINE_WIDTH, ORDER_COLORS, font_size

OUT_DIR = ls.PLOTS_DIR / "exp_bin"
SUMMARY_CSV = OUT_DIR / "mse_threshold_summary.csv"
MSE_THR = 0.05
BLOCK = 256
FIG_W_PTS, FIG_H_PTS = 240, 80  # single column, as the compressed-sensing figures


def mse_curve_until(yc: np.ndarray, c: np.ndarray, rank: np.ndarray, ps, L: int, ev: np.ndarray):
    """Generator over blocks: yields the running list of MSE values on ev as top coefficients are added."""
    resid, out = yc[ev].copy(), []
    for start in range(0, len(yc), BLOCK):
        s = rank[start:start + BLOCK]
        R = resid[:, None] - np.cumsum(ls.basis_block(s, ev, ps, L) * c[s], axis=1)
        out.extend((R ** 2).mean(axis=0).tolist())
        resid = R[:, -1]
        yield out


def compute_mse_pair(y: np.ndarray, ps: np.ndarray, L: int, ev: np.ndarray):
    """MSE curves for eWHT and WHT, the shared plotted span, and k at MSE <= MSE_THR for each."""
    gens, curves, k_thr = {}, {}, {}
    for basis, p in (("eWHT", ps), ("WHT", np.full(L, 0.5))):
        yc, c, rank = ls.coefficients(y, p)
        gens[basis] = mse_curve_until(yc, c, rank, p, L, ev)
        curves[basis], k_thr[basis] = [], None
    x_max = len(y)
    while True:
        for b in gens:
            if len(curves[b]) < x_max:
                curves[b] = next(gens[b])
                if k_thr[b] is None:
                    hit = np.flatnonzero(np.asarray(curves[b]) <= MSE_THR)
                    k_thr[b] = int(hit[0]) + 1 if hit.size else None
        if k_thr["eWHT"] is not None and k_thr["WHT"] is not None:
            x_max = min(len(y), ls.plotted_k_span(k_thr["eWHT"], k_thr["WHT"]))
        if all(len(curves[b]) >= x_max for b in gens):
            break
    return {b: np.asarray(curves[b][:x_max]) for b in gens}, x_max, k_thr


def order_label(order: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd", 6: "6th+"}.get(order, f"{order}th")


def plot_curves(name: str, curves: dict, orders: dict, x_max: int, k_thr: dict) -> None:
    y_top = max(1.05 * MSE_THR, 1.05 * max(float(np.max(curves[b])) for b in curves))
    fig = plt.figure(figsize=(FIG_W_PTS / 72, FIG_H_PTS / 72), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    for col, basis in enumerate(("eWHT", "WHT")):
        ax = fig.add_subplot(gs[0, col])
        m = curves[basis]
        ax.plot(np.arange(1, len(m) + 1), m, color="black", alpha=0.7, linewidth=LINE_WIDTH, zorder=3)
        k = k_thr[basis]
        if k and k <= x_max:
            guide = dict(linestyle="--", linewidth=LINE_WIDTH, color="black", alpha=0.7, zorder=99)
            ax.plot([k, k], [0.0, m[k - 1]], **guide)
            ax.plot([0, k], [m[k - 1], m[k - 1]], **guide)
            ax.scatter([k], [m[k - 1]], s=3, color="black", zorder=100, marker="o")
        ords = orders[basis][:len(m)]
        for order, color in ORDER_COLORS.items():
            x_pts = np.flatnonzero((ords >= order) if order == 6 else (ords == order)) + 1
            ax.scatter(x_pts, m[x_pts - 1], color=color, s=3, zorder=10 + order, marker="o")
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ORDER_COLORS[o], markersize=4,
                          linestyle="None", label=order_label(o)) for o in sorted(ORDER_COLORS)]
        leg = ax.legend(handles=handles, frameon=True, edgecolor="#d6d6d6", handlelength=0.4, handletextpad=0.5,
                        loc="upper right", ncol=2, fontsize=font_size, prop={"size": font_size})
        leg.set_zorder(200)
        leg.get_frame().set_linewidth(LINE_WIDTH)
        ax.set_xlabel(f"Number of largest {basis}\nepistatic interactions")
        ax.set_ylabel("MSE")
        ax.set_xlim(0, x_max + 1)
        ax.set_ylim(0, y_top)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}_mse.pdf", dpi=300)
    plt.close(fig)


def sorted_orders(y: np.ndarray, ps: np.ndarray, L: int) -> dict:
    order = ls.interaction_order_array(L)
    return {b: order[ls.coefficients(y, p)[2]] for b, p in (("eWHT", ps), ("WHT", np.full(L, 0.5)))}


def run_job(job: tuple[str, bool]) -> dict:
    name, replot = job
    y, L, ps, label_mask = ls.load_landscape(name)
    orders = sorted_orders(y, ps, L)
    npz = OUT_DIR / f"{name}_mse_curves.npz"
    if replot:
        z = np.load(npz)
        curves = {"eWHT": z["ewht_mse"], "WHT": z["wht_mse"]}
        x_max, eval_n, var = int(z["plotted_k"]), int(z["eval_n"]), float(z["var_u"])
        k_thr = {"eWHT": int(z["k_mse005_ewht"]), "WHT": int(z["k_mse005_wht"])}
        k_thr = {b: (k if k >= 0 else None) for b, k in k_thr.items()}
    else:
        ev = ls.bin_indices(name, L, label_mask, "experimental")
        curves, x_max, k_thr = compute_mse_pair(y, ps, L, ev)
        eval_n, var = len(ev), float(ls.center(y, np.full(L, 0.5))[ev].var())
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(npz, ewht_mse=curves["eWHT"], wht_mse=curves["WHT"], var_u=var, eval_n=eval_n, plotted_k=x_max,
                 L=L, k_mse005_ewht=k_thr["eWHT"] if k_thr["eWHT"] is not None else -1,
                 k_mse005_wht=k_thr["WHT"] if k_thr["WHT"] is not None else -1)
        print(f"done {name} (plotted_k={x_max}, eval_n={eval_n})", flush=True)
    plot_curves(name, curves, orders, x_max, k_thr)
    count_ge3 = {b: int((orders[b][:k_thr[b]] >= 3).sum()) if k_thr[b] else None for b in orders}
    return dict(dataset=name, L=L, eval_n=eval_n, threshold=MSE_THR, var=var, plotted_k=x_max,
                k_mse005_eWHT=k_thr["eWHT"], k_mse005_WHT=k_thr["WHT"],
                ge3_eWHT=count_ge3["eWHT"], ge3_WHT=count_ge3["WHT"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--include-cr9114", action="store_true", help="also run the (slow) cr9114 landscapes")
    ap.add_argument("--replot", action="store_true", help="redraw PDFs from the saved curves")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    names = args.datasets or [n for n in ls.dataset_names(skip=())
                              if args.include_cr9114 or args.replot or "cr9114" not in n]
    with Pool(min(len(names), args.jobs)) as pool:
        rows = pool.map(run_job, [(n, args.replot) for n in names])
    new = pd.DataFrame(rows)
    if SUMMARY_CSV.exists() and args.datasets:
        old = pd.read_csv(SUMMARY_CSV)
        new = pd.concat([old[~old.dataset.isin(new.dataset)], new])
    summary = new.sort_values("dataset").reset_index(drop=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))

    s = summary[~summary.dataset.isin(ls.DEFAULT_SKIP)].dropna(subset=["k_mse005_eWHT", "k_mse005_WHT"])
    print(f"\nPooled over {len(s)} landscapes: eWHT needs "
          f"{ls.pooled_pct_fewer(s.k_mse005_WHT, s.k_mse005_eWHT):.1f}% fewer coefficients than WHT to reach "
          f"MSE <= {MSE_THR} on the experimental bin, and "
          f"{ls.pooled_pct_fewer(s.ge3_WHT, s.ge3_eWHT):.1f}% fewer third-order and higher interactions.")


if __name__ == "__main__":
    main()
