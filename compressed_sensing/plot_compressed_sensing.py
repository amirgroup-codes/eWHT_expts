from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]

# Constants
FIGURE_WIDTH_PTS = 240
FIGURE_HEIGHT_PTS = 80
LINE_WIDTH = 0.5
COLOR_PWHT = "#1f77b4"
COLOR_WHT = "#d62728"

ROOT = REPO_ROOT / "compressed_sensing"
CS_MSA_DIR = ROOT / "cs" / "msa"
YS_DIR = CS_MSA_DIR / "ys"

font_path = ROOT / 'Helvetica.ttf'
try:
    font_prop = fm.FontProperties(fname=str(font_path))
    fm.fontManager.addfont(str(font_path))
    font_name = font_prop.get_name()
except Exception:
    font_name = 'sans-serif'

font_size = 6
plt.rcParams['font.family'] = font_name
plt.rcParams['font.size'] = font_size
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['lines.linewidth'] = LINE_WIDTH
plt.rcParams['axes.linewidth'] = LINE_WIDTH
plt.rcParams['xtick.major.width'] = LINE_WIDTH
plt.rcParams['ytick.major.width'] = LINE_WIDTH
plt.rcParams['grid.linewidth'] = LINE_WIDTH
plt.rcParams['pdf.compression'] = False


def load_saved_y_values(dataset_name: str, n: int, bootstrap_seed: int = 0, replicate: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys_path = YS_DIR / dataset_name / f"n{n}_bootstrap{bootstrap_seed}_rep{replicate}.npy"
    if not ys_path.exists():
        raise FileNotFoundError(f"Could not find saved y values at {ys_path}")
    arr = np.load(ys_path)
    if arr.ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"Unexpected saved y array shape {arr.shape} at {ys_path}")

    return arr[2].astype(float), arr[0].astype(float), arr[1].astype(float)


def find_closest_90pct_n(summary_df: pd.DataFrame) -> int:
    summary = summary_df.dropna(subset=["r2"]).copy()
    if summary.empty:
        raise ValueError("No p-WHT rows available to compute threshold")

    grouped = summary.groupby("n")["r2"].mean().reset_index()
    last_row = grouped.loc[grouped["n"].idxmax()]
    target = 0.95 * float(last_row["r2"])
    closest_row = grouped.iloc[(grouped["r2"] - target).abs().argmin()]
    return int(closest_row["n"])


def plot_scatter_panel(
    ax: plt.Axes, 
    empirical: np.ndarray, 
    predicted: np.ndarray, 
    basis_label: str, 
    r2_label: float, 
    color: str, 
    n_samples: int,
    limits: list[float]
) -> None:
    ax.scatter(empirical, predicted, s=3, alpha=0.8, color=color, marker="o", edgecolors='none')
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_box_aspect(1) 
    
    locator = ticker.MaxNLocator(integer=True, nbins=5, steps=[1, 2, 3, 5, 10])
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    
    if len(empirical) > 1:
        slope, intercept = np.polyfit(empirical, predicted, 1)
        xs = np.array([limits[0], limits[1]])
        ax.plot(xs, slope * xs + intercept, color="black", linewidth=LINE_WIDTH, linestyle="--")

    ax.text(0.05, 0.95, f"{n_samples} samples", transform=ax.transAxes, ha="left", va="top", fontsize=font_size)
    ax.set_xlabel("Empirical fitness", fontsize=font_size, fontfamily=font_name, labelpad=2)
    ax.set_ylabel(f"{basis_label} fitness", fontsize=font_size, fontfamily=font_name, labelpad=0)
    ax.text(0.95, 0.05, f"$R^2 = {r2_label:.2f}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=font_size)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", width=LINE_WIDTH, labelsize=font_size)
    ax.tick_params(axis="y", width=LINE_WIDTH, labelsize=font_size)


def plot_dataset(csv_path: Path, output_dir: Path) -> Path:
    dataset_name = csv_path.stem
    summary_df = pd.read_csv(csv_path)

    fig = plt.figure(figsize=(FIGURE_WIDTH_PTS / 72, FIGURE_HEIGHT_PTS / 72))
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1, 1, 1], wspace=0.45, left=0.12, right=0.98, bottom=0.25, top=0.9)

    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    for ax in axes:
        ax.tick_params(axis='x', pad=1, labelsize=font_size, width=LINE_WIDTH)
        ax.tick_params(axis='y', pad=1, labelsize=font_size, width=LINE_WIDTH)
        ax.yaxis.set_label_coords(-0.28, 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Plot performance distributions
    for basis, color, label in [("p-WHT", COLOR_PWHT, "eWHT"), ("WHT", COLOR_WHT, "WHT")]:
        sub = summary_df[summary_df["basis_type"] == basis].dropna(subset=["r2"])
        if not sub.empty:
            grouped = sub.groupby("n")["r2"].agg(["mean", "std"]).reset_index()
            axes[0].plot(grouped["n"], grouped["mean"], color=color, marker='o', markersize=np.sqrt(3), label=label)
            axes[0].fill_between(grouped["n"], grouped["mean"] - grouped["std"], grouped["mean"] + grouped["std"], color=color, alpha=0.2)

    axes[0].set_xlabel("Number of samples", fontsize=font_size, labelpad=2)
    axes[0].set_ylabel("$R^2$", fontsize=font_size, labelpad=2)
    axes[0].set_ylim(0.0, 1.1)
    axes[0].set_xlim(left=0)
    axes[0].set_yticks([0.0, 0.5, 1.0])
    
    # Legend explicitly set to match base font_size parameter constraints
    legend = axes[0].legend(fontsize=font_size, prop={'size': font_size}, frameon=True, edgecolor='#d6d6d6', handlelength=0.8, handletextpad=0.5, loc='lower right')
    legend.set_zorder(200)
    legend.get_frame().set_linewidth(LINE_WIDTH)

    # Threshold guidelines
    pwh_summary = summary_df[summary_df["basis_type"] == "p-WHT"].dropna(subset=["r2"])
    selected_n = find_closest_90pct_n(pwh_summary)
    pwh_grouped = pwh_summary.groupby("n")["r2"].agg(["mean", "std"]).reset_index()
    selected_mean = float(pwh_grouped.loc[pwh_grouped["n"] == selected_n, "mean"].iloc[0])

    axes[0].plot([selected_n, selected_n], [0.0, selected_mean], linestyle="--", linewidth=LINE_WIDTH, color="black", alpha=0.7, zorder=1)
    axes[0].hlines(y=selected_mean, xmin=0, xmax=selected_n, linestyle="--", linewidth=LINE_WIDTH, color="black", alpha=0.7, zorder=1)
    axes[0].scatter([selected_n], [selected_mean], s=3, color="black", zorder=100, marker="o")

    # Load, calculate centered metrics, and plot scatters
    actual, pwht_pred, wht_pred = load_saved_y_values(dataset_name, selected_n, bootstrap_seed=0, replicate=0)
    wht_train_mean = np.mean(actual)  # Coordinate reference tracking

    actual_pwht_r2 = float(r2_score(actual - wht_train_mean, pwht_pred - wht_train_mean))
    actual_wht_r2 = float(r2_score(actual - wht_train_mean, wht_pred - wht_train_mean))

    all_vals = np.concatenate([actual, pwht_pred, wht_pred])
    mn, mx = all_vals.min(), all_vals.max()
    pad = (mx - mn) * 0.05
    shared_limits = [mn - pad, mx + pad]

    plot_scatter_panel(axes[1], actual, pwht_pred, "eWHT", actual_pwht_r2, COLOR_PWHT, len(actual), shared_limits)
    plot_scatter_panel(axes[2], actual, wht_pred, "WHT", actual_wht_r2, COLOR_WHT, len(actual), shared_limits)
    
    fig.align_labels()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset_name}_prediction.pdf"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=CS_MSA_DIR / "plots")
    parser.add_argument("--csv-dir", type=Path, default=CS_MSA_DIR)
    args = parser.parse_args()

    csv_paths = sorted([p for p in args.csv_dir.glob("*.csv") if p.name != "test.csv"])
    for csv_path in csv_paths:
        try:
            plot_dataset(csv_path, args.output_dir)
        except Exception as exc:
            print(f"Skipped {csv_path.name}: {exc}")


if __name__ == '__main__':
    main()