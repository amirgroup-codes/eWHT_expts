from __future__ import annotations
from math import comb
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Constants
FIGURE_WIDTH_PTS = 495
FIGURE_HEIGHT_PTS = 100
LINE_WIDTH = 0.5
COLOR_PWHT = "#1f77b4"
COLOR_WHT = "#d62728"

# Order Colors
ORDER_COLORS = {
    1: "#ef2b25",
    2: "#f59720",
    3: "#1cb04b",
    4: "#5466ac",
    5: "#ad2b8f",
    6: "#b3b3b3"
}

ROOT = Path(__file__).resolve().parents[1]
font_path = ROOT / 'Helvetica.ttf'
try:
    font_prop = fm.FontProperties(fname=str(font_path))
    fm.fontManager.addfont(str(font_path))
    font_name = font_prop.get_name()
except Exception:
    font_name = 'sans-serif'

plt.rcParams['font.family'] = font_name
font_size = 6
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

def interaction_order_array(L: int) -> np.ndarray:
    N = 2**L
    orders = np.zeros(N, dtype=int)
    for i in range(N):
        orders[i] = i.bit_count()
    return orders

def order_boundaries(L: int) -> list[int]:
    """Return interaction order (popcount) for coefficient indices 0..2^L-1."""
    boundaries: list[int] = []
    running = 0
    for k in range(0, L + 1):
        running += comb(L, k)
        if running < 2**L:
            boundaries.append(running)
    return boundaries

def normalize_empirical_squared_magnitudes(coeffs: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=float)
    denom = float(np.sqrt(np.sum(np.square(coeffs)))) if coeffs.size else 0.0
    return coeffs / denom if denom > 0 else np.zeros_like(coeffs)

def plot_variance_panel(ax, coeffs, orders, title_prefix, x_max, threshold=95.0):
    """Plots cumulative variance with priority given to higher-order points."""
    mags = np.square(coeffs)
    total_var = np.sum(mags)
    
    idx_sorted = np.argsort(mags)[::-1]
    sorted_mags = mags[idx_sorted]
    sorted_orders = orders[idx_sorted]
    
    cum_var = 100.0 * np.cumsum(sorted_mags) / total_var
    cum_var = np.concatenate(([0.0], cum_var))
    x_axis = np.arange(len(cum_var))
    
    # 1. Background cumulative line
    ax.plot(x_axis, cum_var, color="gray", linewidth=LINE_WIDTH, zorder=1)
    
    # 2. 95% threshold guide lines
    closest_idx = int(np.argmin(np.abs(cum_var - threshold)))
    x_mark = x_axis[closest_idx]
    y_mark = cum_var[closest_idx]
    
    ax.plot([x_mark, x_mark], [0.0, y_mark], linestyle="--", 
            linewidth=LINE_WIDTH, color="black", alpha=0.7, zorder=99)
    ax.plot([0, x_mark], [y_mark, y_mark], linestyle="--", 
            linewidth=LINE_WIDTH, color="black", alpha=0.7, zorder=99)
    ax.scatter([x_mark], [y_mark], s=3, color="black", zorder=100, marker="o")

    # 3. Plot interaction coefficients
    for order in sorted(ORDER_COLORS.keys()):
        color = ORDER_COLORS[order]
        mask = (sorted_orders == order)
        x_points = np.where(mask)[0] + 1
        y_points = cum_var[x_points]
        
        # Filter for current view
        visible = x_points <= x_max
        ax.scatter(x_points[visible], y_points[visible], 
                   color=color, s=3, label=f"{order}st" if order == 1 else f"{order}nd" if order == 2 else f"{order}rd" if order == 3 else f"{order}th", zorder=10 + order, marker="o")

    ax.set_xlabel(f"Number of largest {title_prefix} coefficients")
    ax.set_ylabel("% variance explained")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 105)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(
            fontsize=font_size,
            prop={'size': font_size},
            frameon=True,
            edgecolor='#d6d6d6',
            handlelength=0.4,
            handletextpad=0.5,
            loc='lower right',
            ncol=2
        )
    legend.set_zorder(200)
    legend.get_frame().set_zorder(199)
    legend.get_frame().set_linewidth(LINE_WIDTH)
    
    return int(x_mark)

def plot_fourier(
    coeff_pwht: np.ndarray,
    coeff_wht: np.ndarray,
    out_pdf: Path,
    threshold: float = 95.0,
) -> tuple[int, int]:
    n_coeffs = len(coeff_wht)
    L = int(round(np.log2(n_coeffs)))
    orders = interaction_order_array(L)
    
    c_pwht_norm = normalize_empirical_squared_magnitudes(coeff_pwht)
    c_wht_norm = normalize_empirical_squared_magnitudes(coeff_wht)

    # Compute x limit based on the max 75% mark of both methods
    def get_x_mark(norm_coeffs):
        mags = np.square(norm_coeffs)
        cum = 100.0 * np.cumsum(np.sort(mags)[::-1])
        cum = np.concatenate(([0.0], cum))
        return int(np.argmin(np.abs(cum - threshold)))

    x_mark_pwht = get_x_mark(c_pwht_norm)
    x_mark_wht = get_x_mark(c_wht_norm)
    
    if (x_mark_pwht < 1000 and x_mark_wht > 1000) or (x_mark_wht < 1000 and x_mark_pwht > 1000):
        x_max_var = max(int(np.ceil((4.0 / 3.0) * min(x_mark_pwht, x_mark_wht))), 20)
    else:
        x_max_var = max(int(np.ceil((4.0 / 3.0) * max(x_mark_pwht, x_mark_wht))), 20)

    fig = plt.figure(figsize=(FIGURE_WIDTH_PTS / 72, FIGURE_HEIGHT_PTS / 72), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1])
    
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_var_pwht = fig.add_subplot(gs[0, 1])
    ax_var_wht = fig.add_subplot(gs[0, 2])

    # Plot spectrum
    perm = np.argsort(orders, kind="stable")
    pwht_s = np.abs(c_pwht_norm[perm])
    wht_s = np.abs(c_wht_norm[perm])
    x_spec = np.arange(n_coeffs)

    # Filter for stem plot to avoid overcrowding
    STEM_THRESHOLD = 0
    mask_p = pwht_s >= STEM_THRESHOLD
    mask_w = wht_s >= STEM_THRESHOLD

    if np.any(mask_p):
        markerline_p, stemlines_p, _ = ax_spec.stem(
            x_spec[mask_p], pwht_s[mask_p], 
            linefmt=COLOR_PWHT, markerfmt=' ', label="eWHT", basefmt=" "
        )
        plt.setp(markerline_p, markersize=1, color=COLOR_PWHT)
        plt.setp(stemlines_p, linewidth=LINE_WIDTH - 0.2)

    if np.any(mask_w):
        markerline_w, stemlines_w, _ = ax_spec.stem(
            x_spec[mask_w], -wht_s[mask_w], 
            linefmt=COLOR_WHT, markerfmt=' ', label="WHT", basefmt=" "
        )
        plt.setp(stemlines_w, linewidth=LINE_WIDTH - 0.2)
        plt.setp(markerline_w, markersize=1, color=COLOR_WHT)

    ax_spec.axhline(0, color="black", linewidth=LINE_WIDTH)

    orders = interaction_order_array(L)
    order_2_mask = (orders == 2)
    # Direct indexing on the original array is the only way to ensure order alignment
    max_order_2 = np.abs(c_pwht_norm[order_2_mask]).max()
    print(f"Correct Max magnitude for order 2: {max_order_2:.4f}")
    
    # Boundary logic for interaction orders
    max_ord_spec = min(5, L)
    x_limit_spec = int(sum(comb(L, k) for k in range(0, max_ord_spec + 1)))
    for b in order_boundaries(L):
        if 0 <= b < x_limit_spec:
            ax_spec.axvline(b, linestyle="--", color="gray", alpha=0.3, zorder=0)
            
    ax_spec.set_xlim(0, x_limit_spec)
    ax_spec.set_ylabel("Coefficient magnitude")
    ax_spec.set_xticks([])
    ax_spec.legend(loc='upper right', frameon=False, handlelength=1.0, fontsize=font_size, prop={'size': font_size})
    ax_spec.spines["top"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)
    ax_spec.spines["bottom"].set_visible(True)
    ax_spec.spines["bottom"].set_color('none')
    ax_spec.spines["left"].set_linewidth(LINE_WIDTH)
    ax_spec.spines["bottom"].set_linewidth(LINE_WIDTH)
    legend_elements = [
        Line2D([0], [0], color=COLOR_PWHT, lw=LINE_WIDTH, label='eWHT'),
        Line2D([0], [0], color=COLOR_WHT, lw=LINE_WIDTH, label='WHT')
    ]
    legend = ax_spec.legend(
            handles=legend_elements,
            fontsize=font_size,
            prop={'size': font_size},
            frameon=True,
            edgecolor='#d6d6d6',
            handlelength=0.8,
            handletextpad=0.5,
            loc='upper right',
        )
    legend.get_frame().set_linewidth(LINE_WIDTH)

    _ = plot_variance_panel(ax_var_pwht, c_pwht_norm, orders, "eWHT", x_max_var, threshold)
    k95_wht_idx = plot_variance_panel(ax_var_wht, c_wht_norm, orders, "WHT", x_max_var, threshold)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    
    return k95_wht_idx, x_mark_pwht


def generate_and_render_r2_plots(name: str, L: int, r2_pwht: np.ndarray, r2_wht: np.ndarray, 
                                 coeffs_pwht_full: np.ndarray, coeffs_wht_full: np.ndarray, 
                                 ord_pwht: np.ndarray, ord_wht: np.ndarray, 
                                 k_cross_pwht: int, k_cross_wht: int, x_max: int, out_pdf: Path):
    """
    Renders the exact multi-panel visual profile style mapping to the custom runtime x_max bound.

    coeffs_pwht_full and coeffs_wht_full are the full Fourier coefficient arrays used for
    the left-spectrum panel; ord_pwht and ord_wht are the sorted interaction orders
    for the adaptive R^2 reconstruction trace.
    """
    x_max = int(x_max)
    mean_pwht = np.mean(r2_pwht, axis=0)
    mean_wht = np.mean(r2_wht, axis=0)
    
    fig = plt.figure(figsize=(FIGURE_WIDTH_PTS / 72, FIGURE_HEIGHT_PTS / 72), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.15, 1.15])
    
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_var_pwht = fig.add_subplot(gs[0, 1])
    ax_var_wht = fig.add_subplot(gs[0, 2])

    # 1. Left Subplot Panel: match the old Fourier left-panel style exactly
    c_pwht_norm = normalize_empirical_squared_magnitudes(coeffs_pwht_full)
    c_wht_norm = normalize_empirical_squared_magnitudes(coeffs_wht_full)
    orders = interaction_order_array(L)
    perm = np.argsort(orders, kind="stable")

    pwht_s = np.abs(c_pwht_norm[perm])
    wht_s = np.abs(c_wht_norm[perm])
    x_spec = np.arange(len(c_pwht_norm))

    # Filter for stem plot to avoid overcrowding
    STEM_THRESHOLD = 0
    mask_p = pwht_s >= STEM_THRESHOLD
    mask_w = wht_s >= STEM_THRESHOLD

    if np.any(mask_p):
        markerline_p, stemlines_p, _ = ax_spec.stem(
            x_spec[mask_p], pwht_s[mask_p], 
            linefmt=COLOR_PWHT, markerfmt=' ', label="eWHT", basefmt=" "
        )
        plt.setp(markerline_p, markersize=1, color=COLOR_PWHT)
        plt.setp(stemlines_p, linewidth=LINE_WIDTH - 0.2)

    if np.any(mask_w):
        markerline_w, stemlines_w, _ = ax_spec.stem(
            x_spec[mask_w], -wht_s[mask_w], 
            linefmt=COLOR_WHT, markerfmt=' ', label="WHT", basefmt=" "
        )
        plt.setp(stemlines_w, linewidth=LINE_WIDTH - 0.2)
        plt.setp(markerline_w, markersize=1, color=COLOR_WHT)

    ax_spec.axhline(0, color="black", linewidth=LINE_WIDTH)

    # Boundary logic for interaction orders
    max_ord_spec = min(5, L)
    x_limit_spec = int(sum(comb(L, k) for k in range(0, max_ord_spec + 1)))
    for b in order_boundaries(L):
        if 0 <= b < x_limit_spec:
            ax_spec.axvline(b + 0.5, linestyle="--", color="gray", alpha=0.3, zorder=0)

    ax_spec.set_xlim(0, x_limit_spec)
    max_y = 0.0
    if np.any(mask_p):
        max_y = max(max_y, float(np.max(pwht_s[mask_p])))
    if np.any(mask_w):
        max_y = max(max_y, float(np.max(wht_s[mask_w])))
    if max_y > 0.0:
        y_limit = max_y * 1.05
        ax_spec.set_ylim(-y_limit, y_limit)
    ax_spec.set_ylabel("Coefficient magnitude")
    ax_spec.set_xticks([])
    ax_spec.spines["top"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)
    ax_spec.spines["bottom"].set_visible(True)
    ax_spec.spines["bottom"].set_color('none')
    ax_spec.spines["left"].set_linewidth(LINE_WIDTH)
    ax_spec.spines["bottom"].set_linewidth(LINE_WIDTH)
    y_ticks = ax_spec.get_yticks()
    ax_spec.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
    ax_spec.set_yticklabels([f"{abs(t):.2f}" for t in y_ticks])
    legend_elements = [
        Line2D([0], [0], color=COLOR_PWHT, lw=LINE_WIDTH, label='eWHT'),
        Line2D([0], [0], color=COLOR_WHT, lw=LINE_WIDTH, label='WHT')
    ]
    legend = ax_spec.legend(
            handles=legend_elements,
            fontsize=font_size,
            prop={'size': font_size},
            frameon=True,
            edgecolor='#d6d6d6',
            handlelength=0.8,
            handletextpad=0.5,
            loc='upper right',
        )
    legend.get_frame().set_linewidth(LINE_WIDTH)

    # 2. Right Panels: Plot tracking loops with shaded error clouds + order dots
    for idx, (r2_grid, mean_track, ord_arr, ax, title, color) in enumerate([
        (r2_pwht, mean_pwht, ord_pwht, ax_var_pwht, "eWHT", COLOR_PWHT),
        (r2_wht, mean_wht, ord_wht, ax_var_wht, "WHT", COLOR_WHT)
    ]):
        r2_clamped = np.maximum(0.0, r2_grid[:, :x_max])
        mean_ext = np.mean(r2_clamped, axis=0)
        std_ext = np.std(r2_clamped, axis=0)
        max_len = min(x_max, len(mean_ext), len(std_ext))
        if max_len == 0:
            continue
        x_range = np.arange(1, max_len + 1)
        mean_ext = mean_ext[:max_len]
        std_ext = std_ext[:max_len]
        lower = np.maximum(0.0, mean_ext - std_ext)
        upper = np.minimum(1.0, mean_ext + std_ext)

        ax.plot(x_range, mean_ext, color="black", alpha=0.7, linewidth=LINE_WIDTH, zorder=3)
        ax.fill_between(x_range, lower, upper, color="black", alpha=0.15, linewidth=0, edgecolor='none', zorder=2)
        
        # Horizontal Guide Lines at 0.95 threshold matching original design
        k95_mark = k_cross_pwht if idx == 0 else k_cross_wht
        if 0 < k95_mark <= x_max:
            y_mark = mean_ext[k95_mark - 1]
            ax.plot([k95_mark, k95_mark], [0.0, y_mark], linestyle="--", linewidth=LINE_WIDTH, color="black", alpha=0.7, zorder=99)
            ax.plot([0, k95_mark], [y_mark, y_mark], linestyle="--", linewidth=LINE_WIDTH, color="black", alpha=0.7, zorder=99)
            ax.scatter([k95_mark], [y_mark], s=3, color="black", zorder=100, marker="o")

        # Layer order dots on top matching original plot style rules
        for order in sorted(ORDER_COLORS.keys()):
            mask = (ord_arr[:max_len] == order)
            x_pts = np.where(mask)[0] + 1
            if len(x_pts) > 0:
                y_pts = mean_ext[x_pts - 1]
                ax.scatter(
                    x_pts, 
                    y_pts, 
                    color=ORDER_COLORS[order], 
                    s=3, 
                    label='_nolegend_', 
                    zorder=10 + order, 
                    marker="o"
                )

        def order_label(order: int) -> str:
            if order == 1:
                return '1st'
            if order == 2:
                return '2nd'
            if order == 3:
                return '3rd'
            return f'{order}th'

        order_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=ORDER_COLORS[order], markersize=4, linestyle='None', label=order_label(order))
            for order in sorted(ORDER_COLORS.keys())
        ]

        ax.set_xlabel(f"Number of largest {title} epistatic interactions")
        ax.set_ylabel("$R^2$")
        ax.set_xlim(0, x_max + 1)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        leg = ax.legend(handles=order_handles, fontsize=font_size, prop={'size': font_size}, frameon=True, edgecolor='#d6d6d6', handlelength=0.4, handletextpad=0.5, loc='lower right', ncol=2)
        leg.set_zorder(200)
        leg.get_frame().set_linewidth(LINE_WIDTH)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)