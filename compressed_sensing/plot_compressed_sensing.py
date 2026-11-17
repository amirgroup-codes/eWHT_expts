"""Compressed-sensing figures: R^2 vs number of training samples (eWHT vs WHT), plus predicted-vs-measured
scatters for one representative replicate at the n where the better basis reaches 95% of its final R^2.

Reads compressed_sensing/cs/esm_bin/{dataset}.csv and ys/; writes cs/esm_bin/plots/{dataset}_prediction.pdf.

  python compressed_sensing/plot_compressed_sensing.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

CS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CS_DIR / "cs" / "esm_bin"
FIGURE_WIDTH_PTS, FIGURE_HEIGHT_PTS = 240, 80
LINE_WIDTH = 0.5
COLOR_PWHT, COLOR_WHT = "#1f77b4", "#d62728"
MAX_SCATTER_POINTS = 1000
# R^2 axis: 0-1.1 by default; these two landscapes top out lower, so they get 0-0.9.
R2_AXIS_DEFAULT = ((0.0, 1.1), (0.0, 0.5, 1.0))
R2_AXIS_LOW = ((0.0, 0.9), (0.0, 0.4, 0.8))
LOW_R2_DATASETS = {"cr6261_h1_mean", "his3"}

# NOTE: this looks for Helvetica.ttf in compressed_sensing/ (not the repo root), so the figures fall back
# to the default sans-serif font; kept as-is so the published figures do not change.
try:
    fm.fontManager.addfont(str(CS_DIR / "Helvetica.ttf"))
    font_name = fm.FontProperties(fname=str(CS_DIR / "Helvetica.ttf")).get_name()
except Exception:
    font_name = "sans-serif"
font_size = 6
plt.rcParams.update({"font.family": font_name, "font.size": font_size, "axes.labelsize": font_size,
                     "xtick.labelsize": font_size, "ytick.labelsize": font_size, "legend.fontsize": font_size,
                     "lines.linewidth": LINE_WIDTH, "axes.linewidth": LINE_WIDTH,
                     "xtick.major.width": LINE_WIDTH, "ytick.major.width": LINE_WIDTH,
                     "grid.linewidth": LINE_WIDTH, "pdf.compression": False})


def load_ys(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(measured, eWHT predictions, WHT predictions). Rows end with eWHT, WHT, measured; older files have
    extra (unused) rows in front."""
    arr = np.load(path)
    return arr[-1].astype(float), arr[-3].astype(float), arr[-2].astype(float)


def centered_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    center = float(np.mean(actual))
    return float(r2_score(actual - center, predicted - center))


def mean_r2(df: pd.DataFrame, n: int, basis: str) -> float:
    return float(df[(df["n"] == n) & (df["basis_type"] == basis)]["r2"].mean())


def threshold_n(df: pd.DataFrame) -> tuple[int, str]:
    """n whose mean R^2 is closest to 95% of the final mean R^2 of the basis that is best at the largest n."""
    df = df.dropna(subset=["r2"])
    final = df[df["n"] == df["n"].max()].groupby("basis_type")["r2"].mean()
    best = str(final.idxmax())
    curve = df[df["basis_type"] == best].groupby("n")["r2"].mean().reset_index()
    return int(curve.iloc[(curve["r2"] - 0.95 * float(final[best])).abs().argmin()]["n"]), best


def representative_split(df: pd.DataFrame, name: str, n: int) -> Path:
    """The bootstrap/replicate whose eWHT and WHT R^2 are jointly closest to the means at n."""
    paths = sorted((RESULTS_DIR / "ys" / name).glob(f"n{n}_bootstrap*_rep*.npy"))
    if not paths:
        raise FileNotFoundError(f"No ys files for {name} at n={n}")
    m_e, m_w = mean_r2(df, n, "p-WHT"), mean_r2(df, n, "WHT")

    def distance(path: Path) -> float:
        actual, pred_e, pred_w = load_ys(path)
        return abs(centered_r2(actual, pred_e) - m_e) + abs(centered_r2(actual, pred_w) - m_w)

    return min(paths, key=distance)


def scatter_panel(ax, empirical, predicted, label: str, r2: float, color: str, n: int, limits) -> None:
    if len(empirical) > MAX_SCATTER_POINTS:
        idx = np.random.default_rng(0).choice(len(empirical), size=MAX_SCATTER_POINTS, replace=False)
        empirical, predicted = empirical[idx], predicted[idx]
    ax.scatter(empirical, predicted, s=3, alpha=0.8, color=color, marker="o", edgecolors="none")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_box_aspect(1)
    locator = ticker.MaxNLocator(integer=True, nbins=5, steps=[1, 2, 3, 5, 10])
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    if len(empirical) > 1:
        slope, intercept = np.polyfit(empirical, predicted, 1)
        xs = np.array(limits)
        ax.plot(xs, slope * xs + intercept, color="black", linewidth=LINE_WIDTH, linestyle="--")
    ax.text(0.05, 0.95, f"{n} samples", transform=ax.transAxes, ha="left", va="top", fontsize=font_size)
    ax.text(0.95, 0.05, f"$R^2 = {r2:.2f}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=font_size)
    ax.set_xlabel("Empirical fitness", fontsize=font_size, fontfamily=font_name, labelpad=2)
    ax.set_ylabel(f"{label} fitness", fontsize=font_size, fontfamily=font_name, labelpad=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=LINE_WIDTH, labelsize=font_size)


def plot_dataset(csv_path: Path) -> Path:
    name = csv_path.stem
    df = pd.read_csv(csv_path)
    fig = plt.figure(figsize=(FIGURE_WIDTH_PTS / 72, FIGURE_HEIGHT_PTS / 72))
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1, 1, 1], wspace=0.45, left=0.12, right=0.98,
                          bottom=0.25, top=0.9)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in axes:
        ax.tick_params(axis="both", pad=1, labelsize=font_size, width=LINE_WIDTH)
        ax.yaxis.set_label_coords(-0.28, 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # R^2 vs n (mean +- sd over bootstrap seeds and replicates)
    for basis, color, label in (("p-WHT", COLOR_PWHT, "eWHT"), ("WHT", COLOR_WHT, "WHT")):
        g = df[df["basis_type"] == basis].dropna(subset=["r2"]).groupby("n")["r2"].agg(["mean", "std"]).reset_index()
        axes[0].plot(g["n"], g["mean"], color=color, marker="o", markersize=np.sqrt(3), label=label)
        axes[0].fill_between(g["n"], g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.2)
    ylim, yticks = R2_AXIS_LOW if name in LOW_R2_DATASETS else R2_AXIS_DEFAULT
    axes[0].set_xlabel("Number of samples", fontsize=font_size, labelpad=2)
    axes[0].set_ylabel("$R^2$", fontsize=font_size, labelpad=2)
    axes[0].set_ylim(*ylim)
    axes[0].set_xlim(left=0)
    axes[0].set_yticks(list(yticks))
    leg = axes[0].legend(fontsize=font_size, prop={"size": font_size}, frameon=True, edgecolor="#d6d6d6",
                         handlelength=0.8, handletextpad=0.5, loc="lower right")
    leg.set_zorder(200)
    leg.get_frame().set_linewidth(LINE_WIDTH)

    # guide lines at the 95%-of-final n
    n_sel, best = threshold_n(df)
    y_sel = float(df[(df["basis_type"] == best) & (df["n"] == n_sel)].dropna(subset=["r2"])["r2"].mean())
    axes[0].plot([n_sel, n_sel], [0.0, y_sel], linestyle="--", linewidth=LINE_WIDTH, color="black", alpha=0.7,
                 zorder=1)
    axes[0].hlines(y=y_sel, xmin=0, xmax=n_sel, linestyle="--", linewidth=LINE_WIDTH, color="black", alpha=0.7,
                   zorder=1)
    axes[0].scatter([n_sel], [y_sel], s=3, color="black", zorder=100, marker="o")

    # predicted vs measured for one representative split at n_sel
    actual, pred_e, pred_w = load_ys(representative_split(df, name, n_sel))
    vals = np.concatenate([actual, pred_e, pred_w])
    pad = (vals.max() - vals.min()) * 0.05
    limits = [vals.min() - pad, vals.max() + pad]
    scatter_panel(axes[1], actual, pred_e, "eWHT", mean_r2(df, n_sel, "p-WHT"), COLOR_PWHT, n_sel, limits)
    scatter_panel(axes[2], actual, pred_w, "WHT", mean_r2(df, n_sel, "WHT"), COLOR_WHT, n_sel, limits)

    fig.align_labels()
    out = RESULTS_DIR / "plots" / f"{name}_prediction.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        print(f"  {plot_dataset(csv_path).name}")


if __name__ == "__main__":
    main()
