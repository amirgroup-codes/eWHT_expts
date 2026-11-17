"""Figure style and the truncation figure (coefficient spectrum + eWHT / WHT R^2 curves)."""
from __future__ import annotations
from math import comb, cos, radians, sin
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as mticker
import numpy as np

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


def order_region_x_bounds(L: int, r: int) -> tuple[float, float]:
    """Start (inclusive) and end (exclusive) x indices for interaction order r in the sorted spectrum."""
    x_lo = sum(comb(L, k) for k in range(r))
    x_hi = sum(comb(L, k) for k in range(r + 1))
    return float(x_lo), float(x_hi)


def spectrum_order_label_arrow_orders(dataset_name: str | None, L: int) -> set[int]:
    """Orders that use an arrow:
    - none for beta
    - r=1, 2, 3 for cr9114
    - r=1, 2 for his3, mtagbfp, cov_rbd, and default
    """
    ds = (dataset_name or "").lower()
    if "beta" in ds:
        return set()
    if "cr9114" in ds or "cov_rbd" in ds:
        max_r = min(3, L)
        return {r for r in (1, 2, 3) if r <= max_r}
    max_r = min(2, L)
    return {r for r in (1, 2) if r <= max_r}


def _data_units_per_display_point(ax) -> tuple[float, float]:
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if bbox.width <= 0 or bbox.height <= 0:
        return 1.0, 1.0
    return (xlim[1] - xlim[0]) / bbox.width, (ylim[1] - ylim[0]) / bbox.height



def _draw_short_down_right_arrow_label(
    ax,
    x_line: float,
    y_anchor: float,
    text: str,
    *,
    arrow_length_pt: float = 14.0,
    angle_deg: float = 78.0,
    text_gap_pt: float = 0.0,
    text_drop_pt: float = 1.0,
) -> None:
    """Down-right arrow where the start of the text ('r') aligns right under the tip."""
    ptx, pty = _data_units_per_display_point(ax)

    rad = radians(angle_deg)
    dx = arrow_length_pt * cos(rad) * ptx
    dy = arrow_length_pt * sin(rad) * pty

    tip_x = x_line + dx
    tip_y = y_anchor - dy

    arrow = FancyArrowPatch(
        (x_line, y_anchor),
        (tip_x, tip_y),
        arrowstyle="-|>",
        mutation_scale=3.0,
        linewidth=LINE_WIDTH,
        color="black",
        clip_on=False,
        zorder=26,
    )
    ax.add_patch(arrow)

    # ha="left" starts the text string directly under the arrowhead's x-coordinate
    ax.text(
        tip_x + text_gap_pt * ptx,
        tip_y - text_drop_pt * pty,
        text,
        fontsize=font_size,
        ha="left",
        va="top",
        clip_on=False,
        zorder=25,
    )


def add_spectrum_order_r_labels(
    ax,
    L: int,
    x_limit_spec: float,
    y_limit: float | None = None,
    *,
    dataset_name: str | None = None,
) -> None:
    """Label interaction-order blocks with dataset-specific arrow configurations."""
    ymin, ymax = ax.get_ylim()
    if y_limit is None:
        y_limit = max(abs(ymin), abs(ymax), 1e-12)
    neg_ext = abs(ymin) if ymin < 0 else y_limit * 0.1
    pos_ext = ymax if ymax > 0 else y_limit * 0.1

    max_ord_spec = min(5, L)
    arrow_orders = spectrum_order_label_arrow_orders(dataset_name, L)

    base_inline_y = -neg_ext * 0.88 if ymin < 0 else pos_ext * 0.08
    x_right_margin = max(1.2, 0.04 * x_limit_spec)
    y_anchor = -neg_ext * 0.80 if ymin < 0 else pos_ext * 0.08

    ds = (dataset_name or "").lower()

    # Dataset-specific arrow configs:
    if "cr9114" in ds:
        arrow_configs = {
            1: {"arrow_length_pt": 28.0, "angle_deg": 89.0},
            2: {"arrow_length_pt": 18.0, "angle_deg": 87.0},
            3: {"arrow_length_pt": 10.0, "angle_deg": 84.0},
        }
    elif "his3" in ds:
        arrow_configs = {
            1: {"arrow_length_pt": 26.0, "angle_deg": 84.0},
            2: {"arrow_length_pt": 15.0, "angle_deg": 74.0},
        }
    elif "mtagbfp" in ds:
        arrow_configs = {
            1: {"arrow_length_pt": 24.0, "angle_deg": 88.0},
            2: {"arrow_length_pt": 13.0, "angle_deg": 85.0},
        }
    elif "cov_rbd" in ds:
        arrow_configs = {
            1: {"arrow_length_pt": 28.0, "angle_deg": 89.0},
            2: {"arrow_length_pt": 18.0, "angle_deg": 87.0},
            3: {"arrow_length_pt": 10.0, "angle_deg": 84.0},
        }
    else:
        # Default configuration
        arrow_configs = {
            1: {"arrow_length_pt": 26.0, "angle_deg": 82.0},
            2: {"arrow_length_pt": 15.0, "angle_deg": 74.0},
        }

    for r in range(1, max_ord_spec + 1):
        x_lo, x_hi = order_region_x_bounds(L, r)
        if x_lo >= x_limit_spec:
            continue
        x_hi_clamp = min(x_hi, x_limit_spec)
        text = rf"$r = {r}$"

        if r in arrow_orders:
            cfg = arrow_configs[r]
            x_line = 0.0 if r == 1 else (x_lo - 0.5)

            _draw_short_down_right_arrow_label(
                ax,
                x_line,
                y_anchor,
                text,
                arrow_length_pt=cfg["arrow_length_pt"],
                angle_deg=cfg["angle_deg"],
                text_gap_pt=0.0,
                text_drop_pt=1.0,
            )
        else:
            region_width = x_hi_clamp - x_lo
            
            if "cr9114" in ds:
                inset = min(40.0, max(8.0, 0.05 * region_width))
            else:
                inset = min(0.45, max(0.2, 0.06 * region_width))

            # Non-arrow inline labels remain horizontally level
            inline_y = base_inline_y

            ax.text(
                x_lo + inset,
                inline_y,
                text,
                fontsize=font_size,
                ha="left",
                va="top",
                clip_on=False,
                zorder=25,
            )

    ax.set_xlim(0, x_limit_spec + x_right_margin)


def spectrum_transform_coefficients(
    y_wht: np.ndarray,
    y_pwht: np.ndarray,
    ps: np.ndarray,
    L: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Unnormalized forward transform for spectrum display (norm=None)."""
    from pbias import pfwht

    ps_wht = np.full(L, 0.5)
    spec_pwht = pfwht(np.asarray(y_pwht, dtype=float).copy(), ps=ps, norm=None)
    spec_wht = pfwht(np.asarray(y_wht, dtype=float).copy(), ps=ps_wht, norm=None)
    return spec_pwht, spec_wht


def plot_spectrum_panel(
    ax_spec: plt.Axes,
    coeff_pwht: np.ndarray,
    coeff_wht: np.ndarray,
    L: int,
    *,
    dataset_name: str | None = None,
) -> None:
    """Fourier magnitude stem plot: raw coefficients, autoscale y, no y tick labels."""
    n_coeffs = len(coeff_wht)
    c_pwht = np.asarray(coeff_pwht, dtype=float)
    c_wht = np.asarray(coeff_wht, dtype=float)
    orders = interaction_order_array(L)
    perm = np.argsort(orders, kind="stable")
    pwht_s = np.abs(c_pwht[perm])
    wht_s = np.abs(c_wht[perm])
    x_spec = np.arange(n_coeffs)

    mask_p = pwht_s >= 0
    mask_w = wht_s >= 0

    if np.any(mask_p):
        markerline_p, stemlines_p, _ = ax_spec.stem(
            x_spec[mask_p],
            pwht_s[mask_p],
            linefmt=COLOR_PWHT,
            markerfmt=" ",
            label="eWHT",
            basefmt=" ",
        )
        plt.setp(markerline_p, markersize=1, color=COLOR_PWHT)
        plt.setp(stemlines_p, linewidth=LINE_WIDTH - 0.2)

    if np.any(mask_w):
        markerline_w, stemlines_w, _ = ax_spec.stem(
            x_spec[mask_w],
            -wht_s[mask_w],
            linefmt=COLOR_WHT,
            markerfmt=" ",
            label="WHT",
            basefmt=" ",
        )
        plt.setp(stemlines_w, linewidth=LINE_WIDTH - 0.2)
        plt.setp(markerline_w, markersize=1, color=COLOR_WHT)

    ax_spec.axhline(0, color="black", linewidth=LINE_WIDTH)

    max_ord_spec = min(5, L)
    x_limit_spec = int(sum(comb(L, k) for k in range(0, max_ord_spec + 1)))
    for b in order_boundaries(L):
        if 0 < b < x_limit_spec:
            ax_spec.axvline(b - 0.5, linestyle="--", color="gray", alpha=0.3, zorder=0)

    ax_spec.set_xlim(0, x_limit_spec)
    max_pos = float(np.max(pwht_s[mask_p])) if np.any(mask_p) else 0.0
    max_neg = float(np.max(wht_s[mask_w])) if np.any(mask_w) else 0.0
    y_limit = max(max_pos, max_neg, 1e-12) * 1.05
    ax_spec.set_ylim(-y_limit, y_limit)
    add_spectrum_order_r_labels(
        ax_spec, L, x_limit_spec, y_limit, dataset_name=dataset_name,
    )

    ax_spec.set_ylabel("Coefficient magnitude")
    ax_spec.set_xticks([])
    ax_spec.set_yticks([])
    ax_spec.tick_params(axis="y", which="both", length=0, labelleft=False)
    ax_spec.spines["top"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)
    ax_spec.spines["bottom"].set_visible(True)
    ax_spec.spines["bottom"].set_color("none")
    ax_spec.spines["left"].set_linewidth(LINE_WIDTH)
    ax_spec.spines["bottom"].set_linewidth(LINE_WIDTH)
    legend_elements = [
        Line2D([0], [0], color=COLOR_PWHT, lw=LINE_WIDTH, label="eWHT"),
        Line2D([0], [0], color=COLOR_WHT, lw=LINE_WIDTH, label="WHT"),
    ]
    legend = ax_spec.legend(
        handles=legend_elements,
        fontsize=font_size,
        prop={"size": font_size},
        frameon=True,
        edgecolor="#d6d6d6",
        handlelength=0.8,
        handletextpad=0.5,
        loc="upper right",
    )
    legend.get_frame().set_linewidth(LINE_WIDTH)


def generate_and_render_r2_plots(
    name: str,
    L: int,
    r2_pwht: np.ndarray,
    r2_wht: np.ndarray,
    coeffs_pwht_full: np.ndarray,
    coeffs_wht_full: np.ndarray,
    ord_pwht: np.ndarray,
    ord_wht: np.ndarray,
    k_cross_pwht: int,
    k_cross_wht: int,
    x_max: int,
    out_pdf: Path,
    *,
    y_wht: np.ndarray | None = None,
    y_pwht: np.ndarray | None = None,
    ps: np.ndarray | None = None,
):
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

    if y_wht is not None and y_pwht is not None and ps is not None:
        spec_pwht, spec_wht = spectrum_transform_coefficients(y_wht, y_pwht, ps, L)
    else:
        spec_pwht = np.asarray(coeffs_pwht_full, dtype=float)
        spec_wht = np.asarray(coeffs_wht_full, dtype=float)

    plot_spectrum_panel(ax_spec, spec_pwht, spec_wht, L, dataset_name=name)

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
            if order == 6:
                return '6th+'
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