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
    build_bar_axes,
    build_position_labels,
    collect_interaction_distances_by_order,
    contact_fraction_by_order,
    count_top_k_orders,
    decode_interaction,
    get_2nd_order_matrix,
    get_ca_distance_matrix,
    get_site_antibody_distances,
    interface_contact_percentage_by_order,
    normalize_empirical_squared_magnitudes,
    pooled_contact_fraction,
    prepare_centered_y,
    print_interface_contact_stats,
    print_closest_third_order_interactions_to_interface,
    print_pairwise_and_higher_order_stats,
    print_site_antibody_distances,
    print_spatial_proximity_stats,
    sum_top_k_magnitudes_by_order,
)

# Project paths
DATA_PATH = ROOT / "data" / "processed_data" / "cr6261_h1_mean.csv"
PS_PATH = ROOT / "data" / "ps" / "msa" / "cr6261.npy"
SUMMARY_PATH = ROOT / "fitness_landscapes" / "plots" / "msa" / "k95_centered_r2_summary.csv"
CIF_PATH = ROOT / "visualization" / "3GBN.cif"
OUTPUT_PATH = ROOT / "visualization" / "cr6261_h1_mean.pdf"

POSITIONS = [28, 30, 58, 59, 62, 74, 75, 76, 77, 79, 104]
PDB_POSITIONS = [28, 30, 57, 58, 61, 73, 74, 75, 76, 78, 100]
CIF_CHAIN_ID = "H"
ANTIBODY_CHAIN_ID = "B"
# Heavy-atom contact threshold for antigen–antibody interface proximity.
INTERFACE_CONTACT_THRESHOLD_A = 4.0
# Internal Cα–Cα contact threshold used in printed spatial summaries.
RBD_CONTACT_THRESHOLD_A = 8.0


def main():
    df = pd.read_csv(DATA_PATH, dtype={"genotype": str})
    ps_actual = np.load(PS_PATH)
    summary_df = pd.read_csv(SUMMARY_PATH)
    summary_row = summary_df.loc[summary_df['dataset'] == 'cr6261_h1_mean']
    if summary_row.empty:
        raise ValueError('Dataset cr6261_h1_mean not found in summary file.')

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
    site_antibody_distances = get_site_antibody_distances(
        POSITIONS,
        CIF_PATH,
        antigen_chain_id=CIF_CHAIN_ID,
        antibody_chain_id=ANTIBODY_CHAIN_ID,
    )
    print_site_antibody_distances(POSITIONS, POSITIONS, labels, site_antibody_distances)
    print_closest_third_order_interactions_to_interface(
        'WHT', mask_wht, c_wht_norm, labels, site_antibody_distances, L, n=3
    )
    print_closest_third_order_interactions_to_interface(
        'p-WHT', mask_pwht, c_pwht_norm, labels, site_antibody_distances, L, n=3
    )

    distances_wht = collect_interaction_distances_by_order(mask_wht, L, dist_matrix)
    distances_pwht = collect_interaction_distances_by_order(mask_pwht, L, dist_matrix)
    interface_orders = [1, 2, 3, 4, 5]
    interface_contact_wht = interface_contact_percentage_by_order(
        mask_wht, L, site_antibody_distances, INTERFACE_CONTACT_THRESHOLD_A
    )
    interface_contact_pwht = interface_contact_percentage_by_order(
        mask_pwht, L, site_antibody_distances, INTERFACE_CONTACT_THRESHOLD_A
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
