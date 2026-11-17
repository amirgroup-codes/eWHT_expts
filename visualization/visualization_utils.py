"""Helpers for three-panel summary PDFs (no viz_cov_rbd dependency)."""
from __future__ import annotations

import re
import sys
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "pwht"))
from gen_utils import sampling_function_df

mpl.rcParams.update(
    {
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.titlesize": 6,
        "axes.titlepad": 2,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

FIGURE_WIDTH_PTS = 480
FIGURE_HEIGHT_PTS = 105
VIZ_GRIDSPEC_WSPACE = 0.3
VIZ_SUBPLOT_ADJUST = {"left": 0.08, "right": 0.99, "top": 0.99, "bottom": 0.08}
SUMMARY_PANEL_WIDTH_RATIOS = [3, 0.88, 1.18]
SUMMARY_ORDER_BAR_SLOTS = 6

COLOR_PWHT = "#1f77b4"
COLOR_WHT = "#d62728"
PAIRWISE_HEATMAP_COLORBAR_LABEL = "Interaction"
CMAP_PAIRWISE = mpl.colors.LinearSegmentedColormap.from_list(
    "pairwise_theme",
    ["#faa808", "#f2f2f2", "#088408"],
    N=256,
)


def save_viz_summary_pdf(
    fig,
    output_path: Path,
    *,
    pad_inches: float | tuple[float, float, float, float] = 0.1,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=pad_inches)


def incremental_r2_from_runs(r2_runs: np.ndarray, k_max: int) -> np.ndarray:
    k_max = min(k_max, r2_runs.shape[1])
    mean_cum = np.mean(r2_runs[:, :k_max], axis=0)
    prev = np.concatenate(([0.0], mean_cum[:-1]))
    return mean_cum - prev


def decode_interaction(idx: int, labels: list[str]) -> list[str]:
    L = len(labels)
    return [labels[i] for i in range(L) if idx & (1 << (L - 1 - i))]


def prepare_centered_y(df: pd.DataFrame, L: int, mode: str = "WHT", ps: np.ndarray | None = None) -> np.ndarray:
    y = sampling_function_df(df, L).copy()
    if np.isnan(y).any():
        fill_value = float(np.nanmean(y)) if not np.all(np.isnan(y)) else 0.0
        y = np.where(np.isnan(y), fill_value, y)
    if mode == "WHT":
        return y - np.mean(y)
    if mode == "p-WHT":
        if ps is None:
            raise ValueError("ps vector must be provided for p-WHT weighted mean calculation.")
        probs = np.array([1.0])
        for p in ps:
            probs = np.kron(probs, [1 - p, p])
        weighted_mean = np.sum(y * probs)
        return y - weighted_mean
    return y


_MUTATION_LABEL_RE = re.compile(r"^([A-Z])([0-9]+)([A-Z])$")


def format_plot_position(pos: float | int) -> str:
    if isinstance(pos, float) and pos != int(pos):
        return str(pos)
    return str(int(pos))


def mutation_label_at_plot_position(analysis_label: str, plot_pos: float | int) -> str:
    """Same WT/MUT as analysis label (e.g. F29S), with position replaced for plotting."""
    m = _MUTATION_LABEL_RE.match(analysis_label.strip())
    if not m:
        return analysis_label
    return f"{m.group(1)}{format_plot_position(plot_pos)}{m.group(3)}"


def display_mutation_labels(
    analysis_labels: list[str],
    plot_positions: tuple[float | int, ...] | None,
) -> list[str]:
    if plot_positions is None:
        return analysis_labels
    if len(plot_positions) != len(analysis_labels):
        raise ValueError(
            f"plot_positions length {len(plot_positions)} != labels length {len(analysis_labels)}"
        )
    return [
        mutation_label_at_plot_position(lab, p)
        for lab, p in zip(analysis_labels, plot_positions, strict=True)
    ]


def parse_single_mutant_label(mutant: str) -> str | None:
    if not isinstance(mutant, str) or mutant == "WT" or ";" in mutant:
        return None
    m = re.match(r"^([A-Z])([0-9]+)([A-Z])$", mutant)
    return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else None


def _his3_wt_local_sequence(df: pd.DataFrame, L: int) -> str:
    """First row local sequence (WT alleles at assayed sites)."""
    wt_seq = str(df.iloc[0]["genotype_sequence"]).strip()
    if len(wt_seq) != L:
        raise ValueError(
            f"his3 genotype_sequence length {len(wt_seq)} != number of positions {L}"
        )
    return wt_seq


def build_position_labels_from_genotype_sequence(
    df: pd.DataFrame,
    positions: list[int],
    *,
    label_overrides: dict[Hashable, str] | None = None,
) -> dict[int, str]:
    """Per-site labels from his3-style genotype + genotype_sequence (row 0 = WT local seq)."""
    overrides = label_overrides or {}
    L = len(positions)
    work = df.copy()
    work["genotype"] = work["genotype"].astype(str).str.strip().str.zfill(L)
    wt_seq = _his3_wt_local_sequence(work, L)
    labels: dict[int, str] = {}
    for j, pos in enumerate(positions):
        if pos in overrides:
            labels[pos] = overrides[pos]
            continue
        label = None
        for _, row in work.iterrows():
            g = row["genotype"]
            if g.count("1") == 1 and g[j] == "1":
                seq = str(row["genotype_sequence"]).strip()
                label = f"{wt_seq[j]}{pos}{seq[j]}"
                break
        if label is None:
            for _, row in work.iterrows():
                if row["genotype"][j] == "1":
                    seq = str(row["genotype_sequence"]).strip()
                    label = f"{wt_seq[j]}{pos}{seq[j]}"
                    break
        labels[pos] = label or str(pos)
    return labels


def build_position_labels(
    df: pd.DataFrame,
    positions: list[Hashable],
    *,
    label_overrides: dict[Hashable, str] | None = None,
    label_from_genotype_sequence: bool = False,
) -> dict[Hashable, str]:
    if label_from_genotype_sequence:
        int_positions = [int(p) for p in positions]
        return build_position_labels_from_genotype_sequence(
            df, int_positions, label_overrides=label_overrides,
        )
    overrides = label_overrides or {}
    labels: dict[Hashable, str] = {}
    single_mutants = df["mutant"].dropna().astype(str)
    for pos in positions:
        if pos in overrides:
            labels[pos] = overrides[pos]
            continue
        if isinstance(pos, str) and not str(pos).isdigit():
            label = None
            for mutant in single_mutants.unique():
                if mutant == pos:
                    label = pos
                    break
            labels[pos] = label or str(pos)
            continue
        pos_int = int(pos)
        label = None
        for mutant in single_mutants.unique():
            parsed = parse_single_mutant_label(mutant)
            if parsed is None:
                continue
            if int(re.search(r"\d+", parsed).group(0)) == pos_int:
                label = parsed
                break
        labels[pos] = label or str(pos)
    return labels


def get_2nd_order_matrix(coeffs: np.ndarray, L: int) -> np.ndarray:
    matrix = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            idx = (1 << (L - 1 - i)) | (1 << (L - 1 - j))
            matrix[i, j] = matrix[j, i] = coeffs[idx]
    return matrix


def combined_pairwise_coefficient_matrix(
    c_wht: np.ndarray,
    mask_wht: np.ndarray,
    c_pwht: np.ndarray,
    mask_pwht: np.ndarray,
    L: int,
) -> np.ndarray:
    mat_w2 = get_2nd_order_matrix(c_wht * mask_wht, L)
    mat_p2 = get_2nd_order_matrix(c_pwht * mask_pwht, L)
    return np.tril(mat_w2, k=-1) + np.triu(mat_p2, k=1)


def _pairwise_heatmap_norm(values: np.ndarray) -> mpl.colors.TwoSlopeNorm:
    """Diverging scale: data min/max, white fixed at 0 (no symmetric |max| rescaling)."""
    if values.size == 0:
        return mpl.colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        pad = max(abs(vmin) * 0.1, 1e-8)
        vmin -= pad
        vmax += pad
    # TwoSlopeNorm requires vmin < vcenter < vmax so 0 stays the colormap midpoint.
    if vmin >= 0:
        vmin = -max(vmax * 0.05, 1e-8)
    if vmax <= 0:
        vmax = max(abs(vmin) * 0.05, 1e-8)
    return mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def plot_pairwise_coefficient_heatmap(
    ax,
    combined: np.ndarray,
    labels: list[str],
    *,
    colorbar_label: str | None = None,
) -> None:
    cbar_label = colorbar_label or PAIRWISE_HEATMAP_COLORBAR_LABEL
    norm = _pairwise_heatmap_norm(combined)
    hm = sns.heatmap(
        combined,
        ax=ax,
        cmap=CMAP_PAIRWISE,
        norm=norm,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0.0,
        cbar_kws={"label": cbar_label, "shrink": 0.85, "pad": 0.02},
    )
    cbar = hm.collections[0].colorbar
    cbar.set_ticks([])
    cbar.set_label(cbar_label, rotation=270, labelpad=10)


def draw_summary_pairwise_heatmap_panel(
    ax,
    combined: np.ndarray,
    labels: list[str],
    L: int,
    *,
    tick_rotation: float = 45,
) -> None:
    plot_pairwise_coefficient_heatmap(ax, combined, labels)
    ax.plot([0, L], [0, L], color="black", lw=1.0, ls="--", zorder=4, clip_on=False)
    ax.text(L * 0.9, L * 0.05, r"eWHT", color="black", fontsize=6, ha="center", va="center")
    ax.text(L * 0.09, L * 0.95, r"WHT", color="black", fontsize=6, ha="center", va="center")
    ax.set_title("")
    ax.set_xlabel("Mutation 1")
    ax.set_ylabel("Mutation 2")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=tick_rotation, ha="right")
    ax.tick_params(axis="both", length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)


def _add_wht_pwht_legend(ax, loc: str = "upper right") -> None:
    handles = [
        plt.Line2D(
            [0], [0], color=COLOR_PWHT, marker="s", markersize=6, linestyle="None",
            markeredgecolor="black", markeredgewidth=0.5, label=r"eWHT",
        ),
        plt.Line2D(
            [0], [0], color=COLOR_WHT, marker="s", markersize=6, linestyle="None",
            markeredgecolor="black", markeredgewidth=0.5, label=r"WHT",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        fontsize=6,
        frameon=True,
        edgecolor="#d6d6d6",
        handlelength=1.0,
        handletextpad=0.5,
        loc=loc,
    )
    legend.get_frame().set_linewidth(0.5)


def orders_with_positive_credit(
    values_pwht: dict[int, float],
    values_wht: dict[int, float],
    magnitude_orders: list[int],
) -> list[int]:
    """Keep only epistasis orders with nonzero incremental R² in either basis (drop empty slots)."""
    active: list[int] = []
    for o in magnitude_orders:
        a = float(values_pwht.get(o, 0.0))
        b = float(values_wht.get(o, 0.0))
        if abs(a) + abs(b) > 1e-15:
            active.append(o)
    return active if active else list(magnitude_orders)


def build_bar_axes(
    ax,
    values_pwht: dict[int, float],
    values_wht: dict[int, float],
    orders: list[int],
    ylabel: str,
    *,
    y_max: float | None = None,
    legend_loc: str = "upper right",
    bar_slot_count: int | None = None,
    keep_full_slot_xlim: bool = False,
    order_tick_labels: dict[int, str] | None = None,
    order_tick_show: dict[int, bool] | None = None,
) -> None:
    if bar_slot_count is None:
        x = np.arange(len(orders))
        xlim = None
    else:
        x = np.array([order - 1 for order in orders], dtype=float)
        if len(orders) < bar_slot_count:
            xlim = (-0.5, bar_slot_count - 0.5) if keep_full_slot_xlim else (
                float(np.min(x)) - 0.5,
                float(np.max(x)) + 0.5,
            )
        else:
            xlim = (-0.5, bar_slot_count - 0.5)
    values_pwht_arr = np.array([values_pwht[o] for o in orders], dtype=float)
    values_wht_arr = np.array([values_wht[o] for o in orders], dtype=float)
    width = 0.35
    valid_pwht = ~np.isnan(values_pwht_arr)
    valid_wht = ~np.isnan(values_wht_arr)
    if valid_pwht.any():
        ax.bar(
            x[valid_pwht] - width / 2,
            values_pwht_arr[valid_pwht],
            width,
            label=r"eWHT",
            color=COLOR_PWHT,
            edgecolor="black",
            linewidth=0.5,
        )
    if valid_wht.any():
        ax.bar(
            x[valid_wht] + width / 2,
            values_wht_arr[valid_wht],
            width,
            label=r"WHT",
            color=COLOR_WHT,
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    raw_labels = [
        (order_tick_labels.get(order, str(order)) if order_tick_labels else str(order))
        for order in orders
    ]
    tick_labels = [lab if lab != "" else " " for lab in raw_labels]
    ax.set_xticklabels(tick_labels)
    # Bottom x-axis only (otherwise Matplotlib draws duplicate ticks on the top spine).
    ax.tick_params(axis="x", which="both", top=False, labeltop=False)
    if order_tick_show is not None:
        for tick, order in zip(ax.xaxis.get_major_ticks(), orders, strict=True):
            show = order_tick_show.get(order, True)
            tick.tick1line.set_visible(show)
            tick.label1.set_visible(show)
    ax.set_xlabel("Epistasis order")
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if y_max is not None:
        ax.set_ylim(0, y_max)
    else:
        finite_vals = (
            np.concatenate([values_pwht_arr[valid_pwht], values_wht_arr[valid_wht]])
            if (valid_pwht.any() or valid_wht.any())
            else np.array([1.0])
        )
        axis_limit = int(np.ceil(max(finite_vals.max(), 1.0) * 1.1))
        ax.set_ylim(0, axis_limit)
    _add_wht_pwht_legend(ax, loc=legend_loc)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def sum_k95_r2_by_order(
    delta_r2: np.ndarray,
    idx_sorted: np.ndarray,
    k_max: int,
    orders: list[int],
    *,
    high_order_bucket: int | None = None,
) -> dict[int, float]:
    totals = {o: 0.0 for o in orders}
    max_bucket = max(orders)
    k_use = min(k_max, len(delta_r2), len(idx_sorted))
    for j in range(k_use):
        contrib = float(delta_r2[j])
        if contrib == 0.0:
            continue
        order = int(idx_sorted[j]).bit_count()
        if high_order_bucket is not None and order >= high_order_bucket:
            bucket = high_order_bucket
        elif order > max_bucket:
            bucket = max_bucket
        elif order not in totals:
            continue
        else:
            bucket = order
        totals[bucket] += contrib
    return totals


def compute_k95_r2_order_dicts(
    r2_wht_runs: np.ndarray,
    r2_ewht_runs: np.ndarray,
    idx_sorted_wht: np.ndarray,
    idx_sorted_ewht: np.ndarray,
    k_wht: int,
    k_ewht: int,
    orders: list[int],
    *,
    high_order_bucket: int | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    delta_wht = incremental_r2_from_runs(r2_wht_runs, k_wht)
    delta_ewht = incremental_r2_from_runs(r2_ewht_runs, k_ewht)
    totals_wht = sum_k95_r2_by_order(
        delta_wht, idx_sorted_wht, k_wht, orders, high_order_bucket=high_order_bucket,
    )
    totals_ewht = sum_k95_r2_by_order(
        delta_ewht, idx_sorted_ewht, k_ewht, orders, high_order_bucket=high_order_bucket,
    )
    return totals_ewht, totals_wht


def incremental_r2_by_index(
    delta_r2: np.ndarray,
    idx_sorted: np.ndarray,
    k_max: int,
) -> dict[int, float]:
    out: dict[int, float] = {}
    k_use = min(k_max, len(delta_r2), len(idx_sorted))
    for j in range(k_use):
        out[int(idx_sorted[j])] = float(delta_r2[j])
    return out


def interaction_label(idx: int, label_list: list[str]) -> str:
    parts = decode_interaction(idx, label_list)
    text = "×".join(parts) if parts else str(idx)
    if len(text) > 22:
        return text[:20] + "…"
    return text


def top_positive_negative(
    map_wht: dict[int, float],
    map_ewht: dict[int, float],
    top_n: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Panel C ranking: Δ(idx) = incr R²_eWHT(idx) − incr R²_WHT(idx).

    ``map_*`` are sparse: only ranks on each path’s k95 greedy list. Any index missing from
    either map is treated as **0** incremental R² for that basis. Candidates are **all indices
    that appear on at least one path** (union), not only where both paths share a coefficient.
    """
    all_idx = set(map_wht) | set(map_ewht)
    deltas = [
        (idx, float(map_ewht.get(idx, 0.0)) - float(map_wht.get(idx, 0.0)))
        for idx in all_idx
    ]
    pos = sorted([(i, d) for i, d in deltas if d > 0], key=lambda t: t[1], reverse=True)[:top_n]
    neg = sorted([(i, d) for i, d in deltas if d < 0], key=lambda t: t[1])[:top_n]
    return pos, neg


@dataclass(frozen=True)
class R2DeltaMapsConfig:
    dataset: str
    data_path: Path
    ps_path: Path
    summary_path: Path
    r2_wht_path: Path
    r2_pwht_path: Path
    positions: list[Hashable]
    label_overrides: dict[Hashable, str] | None = None
    label_from_genotype_sequence: bool = False


def load_k95_delta_maps(cfg: R2DeltaMapsConfig, pfwht) -> tuple[dict[int, float], dict[int, float], list[str]]:
    df = pd.read_csv(cfg.data_path, dtype={"genotype": str})
    ps = np.load(cfg.ps_path)
    summary_df = pd.read_csv(cfg.summary_path)
    row = summary_df.loc[summary_df["dataset"] == cfg.dataset].iloc[0]
    k_wht = int(row["k95_wht_r2"])
    k_ewht = int(row["k95_pwht_r2"])

    L = len(ps)
    df = df.copy()
    df["genotype"] = df["genotype"].astype(str).str.strip().str.zfill(L)
    y_wht = prepare_centered_y(df, L, mode="WHT")
    y_pwht = prepare_centered_y(df, L, mode="p-WHT", ps=ps)
    c_wht = pfwht(y_wht.copy(), ps=np.full(L, 0.5), norm="expectation")
    c_ewht = pfwht(y_pwht.copy(), ps=ps, norm="expectation")
    idx_wht = np.argsort(np.square(c_wht))[::-1]
    idx_ewht = np.argsort(np.square(c_ewht))[::-1]

    r2_wht = np.load(cfg.r2_wht_path)
    r2_ewht = np.load(cfg.r2_pwht_path)
    delta_wht = incremental_r2_from_runs(r2_wht, k_wht)
    delta_ewht = incremental_r2_from_runs(r2_ewht, k_ewht)

    map_wht = incremental_r2_by_index(delta_wht, idx_wht, k_wht)
    map_ewht = incremental_r2_by_index(delta_ewht, idx_ewht, k_ewht)

    pos_labels = build_position_labels(
        df,
        list(cfg.positions),
        label_overrides=cfg.label_overrides,
        label_from_genotype_sequence=cfg.label_from_genotype_sequence,
    )
    label_list = [pos_labels[p] for p in cfg.positions]
    return map_wht, map_ewht, label_list


def create_r2_summary_figure(
    panel_width_ratios: list[float] | None = None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    fig = plt.figure(figsize=(FIGURE_WIDTH_PTS / 72.0, FIGURE_HEIGHT_PTS / 72.0), dpi=72)
    ratios = panel_width_ratios if panel_width_ratios is not None else SUMMARY_PANEL_WIDTH_RATIOS
    gs = fig.add_gridspec(
        ncols=3,
        width_ratios=ratios,
        wspace=VIZ_GRIDSPEC_WSPACE,
    )
    ax_pair = fig.add_subplot(gs[0, 0])
    ax_r2_order = fig.add_subplot(gs[0, 1])
    ax_delta = fig.add_subplot(gs[0, 2])
    return fig, ax_pair, ax_r2_order, ax_delta


def _delta_r2_axis_scale(values: list[float]) -> tuple[float, float]:
    max_abs = max(abs(v) for v in values) if values else 0.0
    if max_abs >= 0.05:
        return 0.1, 0.12
    return 0.01, 0.012


def adjust_summary_panel_positions(ax_r2_order: plt.Axes, ax_delta: plt.Axes) -> None:
    p2 = ax_r2_order.get_position()
    ax_r2_order.set_position([p2.x0 - 0.03, p2.y0, p2.width, p2.height])
    p3 = ax_delta.get_position()
    narrow_w = p3.width * 0.86
    ax_delta.set_position([p3.x0 + (p3.width - narrow_w) * 0.15, p3.y0, narrow_w, p3.height])


def draw_r2_credit_by_order_panel(
    ax,
    totals_ewht: dict[int, float],
    totals_wht: dict[int, float],
    orders: list[int],
    *,
    bar_slot_count: int,
    keep_full_slot_xlim: bool = True,
    order_tick_labels: dict[int, str] | None = None,
    order_tick_show: dict[int, bool] | None = None,
    legend_loc: str = "upper left",
) -> None:
    build_bar_axes(
        ax,
        totals_ewht,
        totals_wht,
        orders=orders,
        ylabel=r"$R^2$",
        bar_slot_count=bar_slot_count,
        keep_full_slot_xlim=keep_full_slot_xlim,
        legend_loc=legend_loc,
        order_tick_labels=order_tick_labels,
        order_tick_show=order_tick_show,
        y_max=1.0,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])


def draw_diverging_top5_panel(
    ax,
    map_wht: dict[int, float],
    map_ewht: dict[int, float],
    label_list: list[str],
    *,
    top_n: int = 5,
) -> None:
    # Δ = incr_R²_eWHT − incr_R²_WHT over union(k95 indices); missing side → 0 (see top_positive_negative).
    pos, neg = top_positive_negative(map_wht, map_ewht, top_n)
    combined = [(idx, d) for idx, d in neg] + [(idx, d) for idx, d in pos]
    combined.sort(key=lambda t: t[1])
    if not combined:
        return
    labs = [interaction_label(idx, label_list) for idx, _ in combined]
    vals = [d for _, d in combined]
    y = np.arange(len(combined))
    colors = [COLOR_PWHT if v >= 0 else COLOR_WHT for v in vals]
    ax.barh(y, vals, color=colors, height=0.55, edgecolor="black", linewidth=0.25)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=6)
    for tick_label, (idx, _) in zip(ax.get_yticklabels(), combined, strict=True):
        if int(idx).bit_count() > 3:
            tick_label.set_fontsize(5)
    step, lim = _delta_r2_axis_scale(vals)
    ax.set_xlim(-lim, lim)
    ax.set_xticks([-step, 0.0, step])
    ax.set_xlabel(r"$\Delta R^2$ eWHT $-$ $\Delta R^2$ WHT", fontsize=6)
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_r2_summary_figure(
    output_path: Path,
    *,
    setup_pairwise_panel: Callable[[plt.Axes], None],
    totals_ewht_by_order: dict[int, float],
    totals_wht_by_order: dict[int, float],
    map_wht: dict[int, float],
    map_ewht: dict[int, float],
    label_list: list[str],
    magnitude_orders: list[int],
    order_tick_labels: dict[int, str] | None = None,
    order_tick_show: dict[int, bool] | None = None,
    order_bar_fixed_slots: int | None = None,
    bar_slot_count: int = SUMMARY_ORDER_BAR_SLOTS,
    panel_width_ratios: list[float] | None = None,
    r2_legend_loc: str = "upper left",
    diverging_top_n: int = 5,
) -> None:
    fig, ax_pair, ax_r2_order, ax_delta = create_r2_summary_figure(
        panel_width_ratios=panel_width_ratios,
    )
    setup_pairwise_panel(ax_pair)
    if order_bar_fixed_slots is not None:
        bar_orders = list(magnitude_orders)
        use_slots = order_bar_fixed_slots
        keep_wide_xlim = True
    else:
        bar_orders = orders_with_positive_credit(
            totals_ewht_by_order, totals_wht_by_order, magnitude_orders
        )
        n_bar = len(bar_orders)
        use_slots = bar_slot_count if n_bar >= bar_slot_count else n_bar
        keep_wide_xlim = n_bar >= bar_slot_count
    draw_r2_credit_by_order_panel(
        ax_r2_order,
        totals_ewht_by_order,
        totals_wht_by_order,
        orders=bar_orders,
        bar_slot_count=use_slots,
        keep_full_slot_xlim=keep_wide_xlim,
        order_tick_labels=order_tick_labels,
        order_tick_show=order_tick_show,
        legend_loc=r2_legend_loc,
    )
    draw_diverging_top5_panel(ax_delta, map_wht, map_ewht, label_list, top_n=diverging_top_n)
    fig.subplots_adjust(**VIZ_SUBPLOT_ADJUST)
    adjust_summary_panel_positions(ax_r2_order, ax_delta)
    save_viz_summary_pdf(fig, output_path)
    plt.close(fig)
