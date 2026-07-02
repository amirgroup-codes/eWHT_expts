from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import viz_cov_rbd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'pwht'))
from gen_utils import sampling_function_df
from pbias import pfwht

from viz_cov_rbd import (
    CMAP_PAIRWISE,
    FIGURE_HEIGHT_PTS,
    FIGURE_WIDTH_PTS,
    MAGNITUDE_BAR_ORDERS,
    build_bar_axes,
    build_position_labels,
    collect_interaction_distances_by_order,
    count_top_k_orders,
    decode_interaction,
    get_2nd_order_matrix,
    get_ca_distance_matrix,
    mean_distance_by_order,
    normalize_empirical_squared_magnitudes,
    prepare_centered_y,
    print_mean_contact_distance_stats,
    print_closest_pairwise_interactions,
    print_pairwise_and_higher_order_stats,
    sum_top_k_magnitudes_by_order,
)

# Project paths
DATA_PATH = ROOT / "data" / "processed_data" / "cov2.csv"
PS_PATH = ROOT / "data" / "ps" / "msa" / "cov2.npy"
SUMMARY_PATH = ROOT / "fitness_landscapes" / "plots" / "msa" / "k95_centered_r2_summary.csv"
CIF_PATH = ROOT / "visualization" / "7LKA.cif"
OUTPUT_PATH = ROOT / "visualization" / "cov2.pdf"

POSITIONS = [26, 27, 28, 31, 35, 50, 53, 56, 57, 58]
CIF_CHAIN_ID = "A"
CONTACT_ORDERS = [2, 3]
MAGNITUDE_ORDERS = [1, 2, 3]


def main():
    df = pd.read_csv(DATA_PATH, dtype={"genotype": str})
    ps_actual = np.load(PS_PATH)
    summary_df = pd.read_csv(SUMMARY_PATH)
    summary_row = summary_df.loc[summary_df['dataset'] == 'cov2']
    if summary_row.empty:
        raise ValueError('Dataset cov2 not found in summary file.')

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
    dist_matrix = get_ca_distance_matrix(POSITIONS, CIF_PATH, chain_id=CIF_CHAIN_ID)
    print_closest_pairwise_interactions(
        'p-WHT', mask_pwht, c_pwht_norm, labels, dist_matrix, L, n=5
    )

    distances_wht = collect_interaction_distances_by_order(
        mask_wht, L, dist_matrix, min_order=CONTACT_ORDERS[0], max_order=CONTACT_ORDERS[-1]
    )
    distances_pwht = collect_interaction_distances_by_order(
        mask_pwht, L, dist_matrix, min_order=CONTACT_ORDERS[0], max_order=CONTACT_ORDERS[-1]
    )
    contact_distance_wht = mean_distance_by_order(distances_wht)
    contact_distance_pwht = mean_distance_by_order(distances_pwht)

    print_mean_contact_distance_stats("WHT", contact_distance_wht)
    print_mean_contact_distance_stats("p-WHT", contact_distance_pwht)

    mat_w2 = get_2nd_order_matrix(c_wht_norm * mask_wht, L)
    mat_p2 = get_2nd_order_matrix(c_pwht_norm * mask_pwht, L)
    combined = np.tril(mat_w2, k=-1) + np.triu(mat_p2, k=1)

    fig = plt.figure(figsize=(FIGURE_WIDTH_PTS / 72.0, FIGURE_HEIGHT_PTS / 72.0), dpi=72)
    gs = fig.add_gridspec(ncols=3, width_ratios=[3, 1, 1], wspace=0.3)
    ax_pair = fig.add_subplot(gs[0, 0])
    ax_mag = fig.add_subplot(gs[0, 1])
    ax_contact = fig.add_subplot(gs[0, 2])

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
    # draw_mismatch_boxes(ax_pair, mask_wht, mask_pwht, len(POSITIONS))
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

    build_bar_axes(
        ax_mag,
        magnitude_pwht,
        magnitude_wht,
        orders=MAGNITUDE_ORDERS,
        bar_slot_count=len(MAGNITUDE_BAR_ORDERS),
        keep_full_slot_xlim=True,
    )
    build_bar_axes(
        ax_contact,
        contact_distance_pwht,
        contact_distance_wht,
        orders=CONTACT_ORDERS,
        ylabel=r"$C_\alpha$ distance (Å) $\downarrow$",
        bar_slot_count=len(MAGNITUDE_BAR_ORDERS),
        keep_full_slot_xlim=True,
        legend_loc='upper right',
    )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.08)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
