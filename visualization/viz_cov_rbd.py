from __future__ import annotations
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from Bio.PDB import MMCIFParser
from matplotlib.patches import Rectangle

mpl.rcParams.update({
    'font.size': 6,
    'axes.titlesize': 6,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.titlesize': 6,
    'axes.titlepad': 2,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed_data" / "cov_rbd.csv"
PS_PATH = ROOT / "data" / "ps" / "msa" / "cov_rbd.npy"
SUMMARY_PATH = ROOT / "fitness_landscapes" / "plots" / "msa" / "k95_centered_r2_summary.csv"
CIF_PATH = ROOT / "visualization" / "6LZG.cif"
OUTPUT_PATH = ROOT / "visualization" / "cov_rbd.pdf"

sys.path.append(str(ROOT / 'pwht'))
from gen_utils import sampling_function_df
from pbias import pfwht

FIGURE_WIDTH_PTS = 480
FIGURE_HEIGHT_PTS = 105
MAGNITUDE_BAR_ORDERS = [1, 2, 3, 4, 5]
POSITIONS = [339, 371, 373, 375, 417, 440, 446, 477, 478, 484, 493, 496, 498, 501, 505]
CIF_CHAIN_ID = "B"
TARGET_CHAIN_ID = "A"
# Heavy-atom contact threshold for antigen–target interface proximity.
INTERFACE_CONTACT_THRESHOLD_A = 4.0
# Internal RBD Cα–Cα contact threshold used in printed spatial summaries.
RBD_CONTACT_THRESHOLD_A = 8.0
COLOR_PWHT = "#1f77b4"
COLOR_WHT = "#d62728"
CMAP_PAIRWISE = mpl.colors.LinearSegmentedColormap.from_list(
    "pairwise_theme",
    ["#faa808", "#f2f2f2", "#088408"],
    N=256,
)
CMAP_HIGHER = "PuOr"


def normalize_empirical_squared_magnitudes(coeffs: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=float)
    denom = float(np.sqrt(np.sum(np.square(coeffs)))) if coeffs.size else 0.0
    return coeffs / denom if denom > 0 else np.zeros_like(coeffs)


def count_top_k_orders(mask: np.ndarray, max_order: int = 5) -> dict:
    counts = {order: 0 for order in range(1, max_order + 1)}
    for idx in np.flatnonzero(mask):
        order = idx.bit_count()
        if 1 <= order <= max_order:
            counts[order] += 1
    return counts


def sum_top_k_magnitudes_by_order(mask: np.ndarray, coeffs: np.ndarray, max_order: int = 5) -> dict:
    # Use normalized coefficients to sum absolute magnitudes per order
    magnitudes = {order: 0.0 for order in range(1, max_order + 1)}
    for idx in np.flatnonzero(mask):
        order = idx.bit_count()
        if 1 <= order <= max_order:
            magnitudes[order] += float(np.abs(coeffs[idx]))
    return magnitudes


def draw_mismatch_boxes(ax, mask_wht: np.ndarray, mask_pwht: np.ndarray, L: int):
    for i in range(L):
        for j in range(i + 1, L):
            idx = (1 << (L - 1 - i)) | (1 << (L - 1 - j))
            wht_present = bool(mask_wht[idx])
            pwht_present = bool(mask_pwht[idx])
            if wht_present == pwht_present:
                continue
            if wht_present and not pwht_present:
                rect = Rectangle((j, i), 1, 1, fill=False, edgecolor='black', linewidth=0.5, zorder=3, clip_on=False)
            elif pwht_present and not wht_present:
                rect = Rectangle((i, j), 1, 1, fill=False, edgecolor='black', linewidth=0.5, zorder=3, clip_on=False)
            else:
                continue
            ax.add_patch(rect)


def get_2nd_order_matrix(coeffs: np.ndarray, L: int) -> np.ndarray:
    matrix = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            idx = (1 << (L - 1 - i)) | (1 << (L - 1 - j))
            matrix[i, j] = matrix[j, i] = coeffs[idx]
    return matrix


def get_ca_distance_matrix(positions: list[int], cif_path: Path, chain_id: str = CIF_CHAIN_ID) -> np.ndarray:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    chain = structure[0][chain_id]
    coords = []
    for pos in positions:
        residue = chain[pos]
        if "CA" not in residue:
            raise ValueError(f"Residue {pos} in chain {chain_id} has no CA atom.")
        coords.append(residue["CA"].coord.astype(float))
    coords = np.asarray(coords)
    L = len(coords)
    dist = np.zeros((L, L), dtype=float)
    for i in range(L):
        for j in range(i + 1, L):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            dist[i, j] = dist[j, i] = d
    return dist


def interaction_site_indices(idx: int, L: int) -> list[int]:
    return [i for i in range(L) if idx & (1 << (L - 1 - i))]


def mean_pairwise_distance_for_interaction(site_indices: list[int], dist_matrix: np.ndarray) -> float:
    if len(site_indices) < 2:
        raise ValueError("Interaction must include at least two sites.")
    pair_distances = [
        float(dist_matrix[i, j])
        for a, i in enumerate(site_indices)
        for j in site_indices[a + 1 :]
    ]
    return float(np.mean(pair_distances))


def collect_interaction_distances_by_order(
    mask: np.ndarray,
    L: int,
    dist_matrix: np.ndarray,
    min_order: int = 2,
    max_order: int = 5,
) -> dict[int, list[float]]:
    distances = {order: [] for order in range(min_order, max_order + 1)}
    for idx in np.flatnonzero(mask):
        order = idx.bit_count()
        if order < min_order or order > max_order:
            continue
        site_indices = interaction_site_indices(idx, L)
        distances[order].append(mean_pairwise_distance_for_interaction(site_indices, dist_matrix))
    return distances



def _heavy_atom_coords(chain) -> np.ndarray:
    coords = [
        atom.coord.astype(float)
        for residue in chain
        for atom in residue
        if atom.element != "H"
    ]
    if not coords:
        raise ValueError(f"No heavy atoms found in chain {chain.id}.")
    return np.asarray(coords)


def _residue_heavy_atom_coords(residue) -> np.ndarray:
    coords = [atom.coord.astype(float) for atom in residue if atom.element != "H"]
    if not coords:
        raise ValueError(f"Residue {residue.id} has no heavy atoms.")
    return np.asarray(coords)


def get_site_target_distances(
    pdb_positions: list[int],
    cif_path: Path,
    antigen_chain_id: str = CIF_CHAIN_ID,
    target_chain_id: str = TARGET_CHAIN_ID,
) -> np.ndarray:
    """Minimum heavy-atom distance from each assayed site to the target chain."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    model = structure[0]
    antigen_chain = model[antigen_chain_id]
    target_coords = _heavy_atom_coords(model[target_chain_id])

    distances = np.zeros(len(pdb_positions), dtype=float)
    for i, pos in enumerate(pdb_positions):
        residue = antigen_chain[pos]
        residue_coords = _residue_heavy_atom_coords(residue)
        distances[i] = float(
            np.min(np.linalg.norm(residue_coords[:, None, :] - target_coords[None, :, :], axis=2))
        )
    return distances


def mean_site_target_distance_for_interaction(
    site_indices: list[int],
    site_target_distances: np.ndarray,
) -> float:
    return float(np.mean([site_target_distances[i] for i in site_indices]))


def collect_interaction_mean_site_target_distances_by_order(
    mask: np.ndarray,
    L: int,
    site_target_distances: np.ndarray,
    min_order: int = 1,
    max_order: int = 5,
) -> dict[int, list[float]]:
    distances = {order: [] for order in range(min_order, max_order + 1)}
    for idx in np.flatnonzero(mask):
        order = idx.bit_count()
        if order < min_order or order > max_order:
            continue
        site_indices = interaction_site_indices(idx, L)
        distances[order].append(
            mean_site_target_distance_for_interaction(site_indices, site_target_distances)
        )
    return distances


def interface_contact_percentage_by_order(
    mask: np.ndarray,
    L: int,
    site_target_distances: np.ndarray,
    threshold_a: float,
    min_order: int = 1,
    max_order: int = 5,
) -> dict[int, float]:
    interaction_means = collect_interaction_mean_site_target_distances_by_order(
        mask, L, site_target_distances, min_order=min_order, max_order=max_order
    )
    percentages: dict[int, float] = {}
    for order, values in interaction_means.items():
        if not values:
            percentages[order] = float("nan")
        else:
            percentages[order] = 100.0 * float(np.mean(np.asarray(values) <= threshold_a))
    return percentages


def print_site_target_distances(
    positions: list[int],
    pdb_positions: list[int],
    labels: list[str],
    site_target_distances: np.ndarray,
) -> None:
    print("Per-site minimum heavy-atom distance to target chain:")
    for pos, pdb_pos, label, dist in zip(positions, pdb_positions, labels, site_target_distances):
        print(f"  {label} (position {pos}, PDB {pdb_pos}): {dist:.2f} Å")


def print_interface_contact_stats(
    name: str,
    contact_percentages: dict[int, float],
    threshold_a: float,
) -> None:
    print(f"{name} interface contact percentage (heavy-atom distance <= {threshold_a:.1f} Å):")
    for order in sorted(contact_percentages):
        value = contact_percentages[order]
        if np.isnan(value):
            print(f"  order {order}: n/a")
        else:
            print(f"  order {order}: {value:.1f}%")

def contact_fraction_by_order(
    mask: np.ndarray,
    L: int,
    dist_matrix: np.ndarray,
    threshold_a: float,
    min_order: int = 2,
    max_order: int = 5,
) -> dict[int, float]:
    fractions: dict[int, float] = {}
    for order in range(min_order, max_order + 1):
        within = 0
        total = 0
        for idx in np.flatnonzero(mask):
            if idx.bit_count() != order:
                continue
            site_indices = interaction_site_indices(idx, L)
            for a, i in enumerate(site_indices):
                for j in site_indices[a + 1 :]:
                    total += 1
                    if dist_matrix[i, j] <= threshold_a:
                        within += 1
        fractions[order] = (within / total) if total else float("nan")
    return fractions


def pooled_contact_fraction(
    mask: np.ndarray,
    L: int,
    dist_matrix: np.ndarray,
    threshold_a: float,
    min_order: int = 2,
) -> float:
    within = 0
    total = 0
    for idx in np.flatnonzero(mask):
        if idx.bit_count() < min_order:
            continue
        site_indices = interaction_site_indices(idx, L)
        for a, i in enumerate(site_indices):
            for j in site_indices[a + 1 :]:
                total += 1
                if dist_matrix[i, j] <= threshold_a:
                    within += 1
    return (within / total) if total else float("nan")


def print_spatial_proximity_stats(
    name: str,
    distances_by_order: dict[int, list[float]],
    contact_fractions: dict[int, float],
    pooled_contact: float,
    threshold_a: float,
) -> None:
    print(f"{name} spatial proximity (mean pairwise Cα distance per interaction):")
    for order in sorted(distances_by_order):
        values = distances_by_order[order]
        if values:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            print(f"  order {order}: mean={mean:.2f} Å, std={std:.2f} Å, n={len(values)}")
        else:
            print(f"  order {order}: n=0")

    print(f"{name} fraction of within-interaction residue pairs within {threshold_a:.1f} Å:")
    for order, fraction in contact_fractions.items():
        if np.isnan(fraction):
            print(f"  order {order}: n/a")
        else:
            print(f"  order {order}: {100.0 * fraction:.1f}%")
    if np.isnan(pooled_contact):
        print(f"  order >= 2 pooled: n/a")
    else:
        print(f"  order >= 2 pooled: {100.0 * pooled_contact:.1f}%")


def decode_interaction(idx: int, labels: list[str]) -> list[str]:
    L = len(labels)
    return [labels[i] for i in range(L) if idx & (1 << (L - 1 - i))]


def prepare_centered_y(df: pd.DataFrame, L: int, mode: str = 'WHT', ps: np.ndarray = None) -> np.ndarray:
    y = sampling_function_df(df, L).copy()
    if np.isnan(y).any():
        fill_value = float(np.nanmean(y)) if not np.all(np.isnan(y)) else 0.0
        y = np.where(np.isnan(y), fill_value, y)
    if mode == 'WHT':
        return y - np.mean(y)
    if mode == 'p-WHT':
        if ps is None:
            raise ValueError("ps vector must be provided for p-WHT weighted mean calculation.")
        probs = np.array([1.0])
        for p in ps:
            probs = np.kron(probs, [1 - p, p])
        weighted_mean = np.sum(y * probs)
        return y - weighted_mean
    return y


def parse_single_mutant_label(mutant: str):
    if not isinstance(mutant, str) or mutant == 'WT' or ';' in mutant:
        return None
    m = re.match(r'^([A-Z])([0-9]+)([A-Z])$', mutant)
    return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else None


def build_position_labels(df: pd.DataFrame, positions):
    labels = {}
    single_mutants = df['mutant'].dropna().astype(str)
    for pos in positions:
        label = None
        for mutant in single_mutants.unique():
            parsed = parse_single_mutant_label(mutant)
            if parsed is None:
                continue
            if int(re.search(r'\d+', parsed).group(0)) == pos:
                label = parsed
                break
        labels[pos] = label or str(pos)
    return labels


def get_pairwise_coeff_indices(L: int) -> list[int]:
    indices: list[int] = []
    for i in range(L):
        for j in range(i + 1, L):
            indices.append((1 << (L - 1 - i)) | (1 << (L - 1 - j)))
    return indices


def print_closest_pairwise_interactions(
    name: str,
    mask: np.ndarray,
    coeffs: np.ndarray,
    labels: list[str],
    dist_matrix: np.ndarray,
    L: int,
    n: int = 5,
) -> None:
    pairwise_idxs = [idx for idx in get_pairwise_coeff_indices(L) if mask[idx]]
    ranked: list[tuple[float, list[str], float]] = []
    for idx in pairwise_idxs:
        site_indices = interaction_site_indices(idx, L)
        dist = float(dist_matrix[site_indices[0], site_indices[1]])
        interaction = decode_interaction(idx, labels)
        ranked.append((dist, interaction, float(np.abs(coeffs[idx]))))
    ranked.sort(key=lambda item: item[0])
    print(f"{name} top {n} closest pairwise interactions (Cα–Cα distance):")
    if not ranked:
        print("  n/a")
        return
    for rank, (dist, interaction, coeff) in enumerate(ranked[:n], start=1):
        print(f"  {rank}. {interaction}: {dist:.2f} Å (|coeff|={coeff:.4f})")


def print_closest_third_order_interactions_to_interface(
    name: str,
    mask: np.ndarray,
    coeffs: np.ndarray,
    labels: list[str],
    site_target_distances: np.ndarray,
    L: int,
    n: int = 3,
) -> None:
    third_idxs = [idx for idx in np.flatnonzero(mask) if idx.bit_count() == 3]
    ranked: list[tuple[float, list[str], float]] = []
    for idx in third_idxs:
        site_indices = interaction_site_indices(idx, L)
        dist = mean_site_target_distance_for_interaction(site_indices, site_target_distances)
        interaction = decode_interaction(idx, labels)
        ranked.append((dist, interaction, float(np.abs(coeffs[idx]))))
    ranked.sort(key=lambda item: item[0])
    print(f"{name} top {n} 3rd-order interactions closest to target interface:")
    if not ranked:
        print("  n/a")
        return
    for rank, (dist, interaction, coeff) in enumerate(ranked[:n], start=1):
        print(f"  {rank}. {interaction}: {dist:.2f} Å (|coeff|={coeff:.4f})")


def print_pairwise_and_higher_order_stats(
    mask_wht: np.ndarray,
    mask_pwht: np.ndarray,
    c_wht_norm: np.ndarray,
    c_pwht_norm: np.ndarray,
    L: int,
) -> None:
    pairwise_idxs = get_pairwise_coeff_indices(L)

    wht_pair_idxs = [idx for idx in pairwise_idxs if mask_wht[idx]]
    pwht_pair_idxs = [idx for idx in pairwise_idxs if mask_pwht[idx]]
    shared_pair_idxs = [idx for idx in pairwise_idxs if mask_wht[idx] and mask_pwht[idx]]

    print("Pairwise (2nd-order) coefficient counts:")
    print(f"  WHT top-k pairwise:   {len(wht_pair_idxs)}")
    print(f"  p-WHT top-k pairwise: {len(pwht_pair_idxs)}")
    print(f"  Shared in both:       {len(shared_pair_idxs)}")

    if len(shared_pair_idxs) >= 2:
        wht_vals = np.array([c_wht_norm[idx] for idx in shared_pair_idxs], dtype=float)
        pwht_vals = np.array([c_pwht_norm[idx] for idx in shared_pair_idxs], dtype=float)
        pearson_r = float(np.corrcoef(wht_vals, pwht_vals)[0, 1])
        print(f"Pearson correlation (shared pairwise coefficients): {pearson_r:.4f}")
    elif len(shared_pair_idxs) == 1:
        print("Pearson correlation (shared pairwise coefficients): n/a (only 1 shared coefficient)")
    else:
        print("Pearson correlation (shared pairwise coefficients): n/a (no shared coefficients)")

    def sum_magnitude_order_ge(mask: np.ndarray, coeffs: np.ndarray, min_order: int) -> float:
        total = 0.0
        for idx in np.flatnonzero(mask):
            order = idx.bit_count()
            if order >= min_order:
                total += float(np.abs(coeffs[idx]))
        return total

    mag_wht_3plus = sum_magnitude_order_ge(mask_wht, c_wht_norm, min_order=3)
    mag_pwht_3plus = sum_magnitude_order_ge(mask_pwht, c_pwht_norm, min_order=3)
    print("Magnitude of 3rd+ order coefficients in top-k (normalized |coeff| sum):")
    print(f"  WHT:   {mag_wht_3plus:.4f}")
    print(f"  p-WHT: {mag_pwht_3plus:.4f}")


def mean_distance_by_order(distances_by_order: dict[int, list[float]]) -> dict[int, float]:
    return {
        order: (float(np.mean(values)) if values else float("nan"))
        for order, values in distances_by_order.items()
    }


def print_mean_contact_distance_stats(name: str, mean_distances: dict[int, float]) -> None:
    print(f"{name} mean contact distance (Cα–Cα):")
    for order in sorted(mean_distances):
        value = mean_distances[order]
        if np.isnan(value):
            print(f"  order {order}: n/a")
        else:
            print(f"  order {order}: {value:.2f} Å")


def build_bar_axes(
    ax,
    values_pwht,
    values_wht,
    orders=None,
    ylabel="Magnitude",
    *,
    y_max: float | None = None,
    legend_loc: str = 'upper right',
    bar_slot_count: int | None = None,
    xlim_right_pad: float = 0.0,
    keep_full_slot_xlim: bool = False,
):
    if orders is None:
        orders = MAGNITUDE_BAR_ORDERS
    if bar_slot_count is None:
        x = np.arange(len(orders))
        xlim = None
    else:
        x = np.array([order - 1 for order in orders], dtype=float)
        if len(orders) < bar_slot_count:
            if keep_full_slot_xlim:
                xlim = (-0.5, bar_slot_count - 0.5 + xlim_right_pad)
            else:
                xlim = (float(np.min(x)) - 0.5, float(np.max(x)) + 0.5 + xlim_right_pad)
        else:
            xlim = (-0.5, bar_slot_count - 0.5)
    values_pwht = np.array([values_pwht[o] for o in orders], dtype=float)
    values_wht = np.array([values_wht[o] for o in orders], dtype=float)
    width = 0.35

    valid_pwht = ~np.isnan(values_pwht)
    valid_wht = ~np.isnan(values_wht)
    if valid_pwht.any():
        ax.bar(
            x[valid_pwht] - width / 2,
            values_pwht[valid_pwht],
            width,
            label=r'eWHT',
            color=COLOR_PWHT,
            edgecolor='black',
            linewidth=0.5,
        )
    if valid_wht.any():
        ax.bar(
            x[valid_wht] + width / 2,
            values_wht[valid_wht],
            width,
            label=r'WHT',
            color=COLOR_WHT,
            edgecolor='black',
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f'{order}' for order in orders])
    ax.set_xlabel('Epistasis order')
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if y_max is not None:
        ax.set_ylim(0, y_max)
    else:
        finite_vals = np.concatenate([values_pwht[valid_pwht], values_wht[valid_wht]]) if (
            valid_pwht.any() or valid_wht.any()
        ) else np.array([1.0])
        axis_limit = int(np.ceil(max(finite_vals.max(), 1.0) * 1.1))
        ax.set_ylim(0, axis_limit)
    _add_wht_pwht_legend(ax, loc=legend_loc)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _add_wht_pwht_legend(ax, loc: str = 'upper right'):
    handles = [
        plt.Line2D([0], [0], color=COLOR_PWHT, marker='s', markersize=6, linestyle='None', markeredgecolor='black', markeredgewidth=0.5, label=r'eWHT'),
        plt.Line2D([0], [0], color=COLOR_WHT, marker='s', markersize=6, linestyle='None', markeredgecolor='black', markeredgewidth=0.5, label=r'WHT'),
    ]
    legend = ax.legend(
        handles=handles,
        fontsize=6,
        frameon=True,
        edgecolor='#d6d6d6',
        handlelength=1.0,
        handletextpad=0.5,
        loc=loc,
    )
    legend.get_frame().set_linewidth(0.5)


def main():
    df = pd.read_csv(DATA_PATH, dtype={"genotype": str})
    ps_actual = np.load(PS_PATH)
    summary_df = pd.read_csv(SUMMARY_PATH)
    summary_row = summary_df.loc[summary_df['dataset'] == 'cov_rbd']
    if summary_row.empty:
        raise ValueError('Dataset cov_rbd not found in summary file.')

    k_wht = int(summary_row['k95_wht_r2'].item())
    k_pwht = int(summary_row['k95_pwht_r2'].item())

    position_labels = build_position_labels(df, POSITIONS)
    labels = [position_labels[pos] for pos in POSITIONS]

    y_wht = prepare_centered_y(df, len(POSITIONS), mode='WHT')
    C_WHT = pfwht(y_wht, ps=np.full(len(POSITIONS), 0.5))

    y_pwht = prepare_centered_y(df, len(POSITIONS), mode='p-WHT', ps=ps_actual)
    C_PWHT = pfwht(y_pwht, ps=ps_actual)

    # 1. Initialize tracking masks exactly once
    mask_wht = np.zeros_like(C_WHT, dtype=bool)
    mask_pwht = np.zeros_like(C_PWHT, dtype=bool)

    # 2. Compute full-scale squared energies explicitly
    wht_energy = np.square(C_WHT)
    pwht_energy = np.square(C_PWHT)

    # 3. Extract un-shifted greedy rank arrays across the full 2^L frame
    idx_sorted_wht = np.argsort(wht_energy)[::-1]
    idx_sorted_pwht = np.argsort(pwht_energy)[::-1]

    # 4. Slice the top-k entries strictly from these validated positions
    mask_wht[idx_sorted_wht[:k_wht]] = True
    mask_pwht[idx_sorted_pwht[:k_pwht]] = True
    
    counts_wht = count_top_k_orders(mask_wht, max_order=5)
    counts_pwht = count_top_k_orders(mask_pwht, max_order=5)

    # Normalize coefficients the same way used for the pairwise heatmap.
    c_wht_norm = normalize_empirical_squared_magnitudes(C_WHT)
    c_pwht_norm = normalize_empirical_squared_magnitudes(C_PWHT)
    magnitude_wht = sum_top_k_magnitudes_by_order(mask_wht, c_wht_norm, max_order=5)
    magnitude_pwht = sum_top_k_magnitudes_by_order(mask_pwht, c_pwht_norm, max_order=5)

    print("WHT top-k order counts:")
    for order, count in counts_wht.items():
        print(f"  {order}th order: {count}")
    print("p-WHT top-k order counts:")
    for order, count in counts_pwht.items():
        print(f"  {order}th order: {count}")

    print("WHT top-k order magnitude sums:")
    for order, mag in magnitude_wht.items():
        print(f"  {order}th order: {mag:.4f}")
    print("p-WHT top-k order magnitude sums:")
    for order, mag in magnitude_pwht.items():
        print(f"  {order}th order: {mag:.4f}")

    def print_third_order_interactions(name: str, mask: np.ndarray, coeffs: np.ndarray) -> None:
        third_idxs = [idx for idx in np.flatnonzero(mask) if idx.bit_count() == 3]
        print(f"{name} 3rd-order interactions ({len(third_idxs)}):")
        for idx in third_idxs:
            interaction = decode_interaction(idx, labels)
            print(f"  {interaction}: {abs(coeffs[idx]):.4f}")

    print_third_order_interactions('WHT', mask_wht, c_wht_norm)
    print_third_order_interactions('p-WHT', mask_pwht, c_pwht_norm)

    print_pairwise_and_higher_order_stats(mask_wht, mask_pwht, c_wht_norm, c_pwht_norm, len(POSITIONS))

    L = len(POSITIONS)
    if len(POSITIONS) != L:
        raise ValueError(
            f"POSITIONS length ({len(POSITIONS)}) must match POSITIONS length ({L})."
        )
    dist_matrix = get_ca_distance_matrix(POSITIONS, CIF_PATH)
    site_target_distances = get_site_target_distances(POSITIONS, CIF_PATH)
    print_site_target_distances(POSITIONS, POSITIONS, labels, site_target_distances)
    print_closest_third_order_interactions_to_interface(
        'WHT', mask_wht, c_wht_norm, labels, site_target_distances, L, n=3
    )
    print_closest_third_order_interactions_to_interface(
        'p-WHT', mask_pwht, c_pwht_norm, labels, site_target_distances, L, n=3
    )

    distances_wht = collect_interaction_distances_by_order(mask_wht, L, dist_matrix)
    distances_pwht = collect_interaction_distances_by_order(mask_pwht, L, dist_matrix)
    interface_orders = [1, 2, 3, 4, 5]
    interface_contact_wht = interface_contact_percentage_by_order(
        mask_wht, L, site_target_distances, INTERFACE_CONTACT_THRESHOLD_A
    )
    interface_contact_pwht = interface_contact_percentage_by_order(
        mask_pwht, L, site_target_distances, INTERFACE_CONTACT_THRESHOLD_A
    )

    contact_wht = contact_fraction_by_order(mask_wht, L, dist_matrix, RBD_CONTACT_THRESHOLD_A)
    contact_pwht = contact_fraction_by_order(mask_pwht, L, dist_matrix, RBD_CONTACT_THRESHOLD_A)
    pooled_contact_wht = pooled_contact_fraction(mask_wht, L, dist_matrix, RBD_CONTACT_THRESHOLD_A)
    pooled_contact_pwht = pooled_contact_fraction(mask_pwht, L, dist_matrix, RBD_CONTACT_THRESHOLD_A)

    print_spatial_proximity_stats("WHT", distances_wht, contact_wht, pooled_contact_wht, RBD_CONTACT_THRESHOLD_A)
    print_spatial_proximity_stats("p-WHT", distances_pwht, contact_pwht, pooled_contact_pwht, RBD_CONTACT_THRESHOLD_A)
    print_interface_contact_stats("WHT", interface_contact_wht, INTERFACE_CONTACT_THRESHOLD_A)
    print_interface_contact_stats("p-WHT", interface_contact_pwht, INTERFACE_CONTACT_THRESHOLD_A)

    mat_w2 = get_2nd_order_matrix(c_wht_norm * mask_wht, L)
    mat_p2 = get_2nd_order_matrix(c_pwht_norm * mask_pwht, L)
    combined = np.tril(mat_w2, k=-1) + np.triu(mat_p2, k=1)

    fig = plt.figure(figsize=(FIGURE_WIDTH_PTS / 72.0, FIGURE_HEIGHT_PTS / 72.0), dpi=72)
    gs = fig.add_gridspec(ncols=3, width_ratios=[3, 1, 1], wspace=0.3)
    ax_pair = fig.add_subplot(gs[0, 0])
    ax_mag = fig.add_subplot(gs[0, 1])
    ax_interface = fig.add_subplot(gs[0, 2])

    abs_max_pair = np.max(np.abs(combined)) if combined.size else 1.0
    abs_max_pair = float(abs_max_pair) if abs_max_pair > 0 else 1.0
    hm = sns.heatmap(
        combined,
        ax=ax_pair,
        cmap=CMAP_PAIRWISE,
        center=0,
        vmin=-abs_max_pair,
        vmax=abs_max_pair,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0.0,
        cbar_kws={
            'label': 'Pairwise interaction',
            'shrink': 0.85,
            'pad': 0.02,
        }
    )
    cbar = hm.collections[0].colorbar
    cbar.set_ticks([-abs_max_pair, 0.0, abs_max_pair])
    cbar.set_ticklabels([f"{-abs_max_pair:.1f}", "0.0", f"{abs_max_pair:.1f}"])
    cbar.set_label('Interaction', rotation=270, labelpad=10)
    size = len(POSITIONS)
    ax_pair.plot(
        [0, size],
        [0, size],
        color='black',
        lw=1.0,
        ls='--',
        zorder=4,
        clip_on=False
    )
    ax_pair.text(
        size * 0.9,
        size * 0.05,
        r'eWHT',
        color='black',
        fontsize=6,
        ha='center',
        va='center'
    )
    ax_pair.text(
        size * 0.09,
        size * 0.95,
        r'WHT',
        color='black',
        fontsize=6,
        ha='center',
        va='center'
    )
    ax_pair.set_title('')
    ax_pair.set_xlabel('Mutation 1')
    ax_pair.set_ylabel('Mutation 2')
    ax_pair.set_xticklabels(ax_pair.get_xticklabels(), rotation=45, ha='right')
    ax_pair.tick_params(axis='both', length=2)
    ax_pair.spines['top'].set_visible(False)
    ax_pair.spines['right'].set_visible(False)
    ax_pair.spines['bottom'].set_visible(True)
    ax_pair.spines['left'].set_visible(True)

    build_bar_axes(ax_mag, magnitude_pwht, magnitude_wht)
    build_bar_axes(
        ax_interface,
        interface_contact_pwht,
        interface_contact_wht,
        orders=interface_orders,
        ylabel=r"Interface contact percentage (%) $\uparrow$",
        y_max=100.0,
        legend_loc='upper right',
    )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.08)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUTPUT_PATH}')


if __name__ == '__main__':
    main()