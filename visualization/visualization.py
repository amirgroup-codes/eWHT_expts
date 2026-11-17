"""Unified three-panel summary PDFs: interaction heatmap, k95 R² by order, top-5 ΔR²."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VIZ_DIR))
PLOTS_DIR = VIZ_DIR / "plots"
FITNESS_PLOTS = ROOT / "fitness_landscapes" / "plots" / "esm_bin"  # from fitness_landscapes/r2_truncation.py
SUMMARY_PATH = FITNESS_PLOTS / "k95_centered_r2_summary.csv"
DATA_DIR = ROOT / "data" / "processed_data"
PS_DIR = ROOT / "data" / "ps" / "esm_wildtype_marginal"

sys.path.append(str(ROOT / "pwht"))
from pbias import pfwht

from visualization_utils import (
    R2DeltaMapsConfig,
    SUMMARY_ORDER_BAR_SLOTS,
    build_position_labels,
    combined_pairwise_coefficient_matrix,
    compute_k95_r2_order_dicts,
    draw_summary_pairwise_heatmap_panel,
    display_mutation_labels,
    load_k95_delta_maps,
    prepare_centered_y,
    save_r2_summary_figure,
)

# Assay positions from data/processed_data/process_data.py
COV2_POSITIONS = [26, 27, 28, 31, 35, 50, 53, 56, 57, 58]
CR9114_POSITIONS = [29, 30, 31, 52, 57, 58, 59, 71, 74, 75, 76, 77, 84, 87, 95, 106]
# FASTA numbering for heatmap axes only (analysis still uses CR9114_POSITIONS).
CR9114_PLOT_POSITIONS: list[float | int] = [
    30, 35, 36, 57, 64, 65, 66, 79, 82, 83, 84, 85, 92, 95, 103, 113,
]
CR6261_POSITIONS = [28, 30, 58, 59, 62, 74, 75, 76, 77, 79, 104]
CR6261_PLOT_POSITIONS: list[float | int] = [
    29, 35, 65, 66, 69, 82, 83, 84, 85, 87, 112.1,
]
HIS3_POSITIONS = [145, 147, 148, 151, 152, 154, 164, 165, 168, 169, 170]
MTAGBFP_POSITIONS = [20, 45, 63, 127, 143, 158, 168, 172, 174, 197, 206, 207, 231]
BETA_POSITIONS: list[Hashable] = ["NC", 42, 104, 182, 238]
COV_RBD_POSITIONS = [339, 371, 373, 375, 417, 440, 446, 477, 478, 484, 493, 496, 498, 501, 505]
BETA_LABEL_OVERRIDES = {"NC": "gNCa"}


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    positions: list[Hashable]
    max_order: int = 5
    high_order_bucket: int | None = None
    order_tick_labels: dict[int, str] | None = None
    position_label_overrides: dict[Hashable, str] = field(default_factory=dict)
    label_from_genotype_sequence: bool = False
    diverging_top_n: int = 5
    plot_positions: tuple[float | int, ...] | None = None
    r2_legend_loc: str = "upper left"
    heatmap_tick_rotation: float = 45
    # Optional gridspec width_ratios [pairwise, R²-by-order, ΔR²]; figure width unchanged.
    summary_panel_width_ratios: tuple[float, float, float] | None = None
    # Fixed bar x-span (slot count) like cr6261; None = auto from active orders.
    order_bar_fixed_slots: int | None = None
    # Per-order: show x tick mark and numeric label (bar positions unchanged).
    order_tick_show: dict[int, bool] | None = None

    @property
    def magnitude_orders(self) -> list[int]:
        return list(range(1, self.max_order + 1))

    def resolved_high_order_bucket(self) -> int:
        return self.high_order_bucket if self.high_order_bucket is not None else self.max_order


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "cov2": DatasetConfig(
        dataset="cov2",
        positions=COV2_POSITIONS,
        order_tick_labels={1: "1", 2: "2"},
        order_bar_fixed_slots=SUMMARY_ORDER_BAR_SLOTS,
        order_tick_show={1: True, 2: True, 3: False, 4: False, 5: False},
        r2_legend_loc="upper right",
    ),
    "cr9114_h1_mean": DatasetConfig(
        dataset="cr9114_h1_mean",
        positions=CR9114_POSITIONS,
        max_order=5,
        plot_positions=tuple(CR9114_PLOT_POSITIONS),
    ),
    "cr6261_h1_mean": DatasetConfig(
        dataset="cr6261_h1_mean",
        positions=CR6261_POSITIONS,
        max_order=6,
        high_order_bucket=6,
        order_tick_labels={6: "6+"},
        plot_positions=tuple(CR6261_PLOT_POSITIONS),
    ),
    "cr6261_h9_mean": DatasetConfig(
        dataset="cr6261_h9_mean",
        positions=CR6261_POSITIONS,
        max_order=6,
        high_order_bucket=6,
        order_tick_labels={6: "6+"},
        plot_positions=tuple(CR6261_PLOT_POSITIONS),
    ),
    "cov_rbd": DatasetConfig(
        dataset="cov_rbd",
        positions=COV_RBD_POSITIONS,
        order_tick_labels={1: "1", 2: "2", 3: "3"},
        order_bar_fixed_slots=SUMMARY_ORDER_BAR_SLOTS,
        order_tick_show={1: True, 2: True, 3: True, 4: False, 5: False},
        r2_legend_loc="upper right",
    ),
    "mtagbfp": DatasetConfig(
        dataset="mtagbfp",
        positions=MTAGBFP_POSITIONS,
        order_tick_labels={1: "1", 2: "2", 3: "3", 4: "4", 5: ""},
    ),
    "his3": DatasetConfig(
        dataset="his3",
        positions=HIS3_POSITIONS,
        max_order=6,
        high_order_bucket=6,
        order_tick_labels={6: "6+"},
        label_from_genotype_sequence=True,
    ),
    "beta": DatasetConfig(
        dataset="beta",
        positions=BETA_POSITIONS,
        order_tick_labels={1: "1", 2: "2", 3: "3", 4: "4", 5: ""},
        position_label_overrides=BETA_LABEL_OVERRIDES,
    ),
}

DEFAULT_DATASETS = tuple(DATASET_CONFIGS.keys())


def _load_k_values(dataset: str) -> tuple[int, int]:
    summary_df = pd.read_csv(SUMMARY_PATH)
    row = summary_df.loc[summary_df["dataset"] == dataset]
    if row.empty:
        raise ValueError(f"Dataset {dataset} not found in {SUMMARY_PATH}")
    return int(row["k95_wht_r2"].item()), int(row["k95_pwht_r2"].item())


def render_summary_pdf(cfg: DatasetConfig, output_path: Path) -> None:
    data_path = DATA_DIR / f"{cfg.dataset}.csv"
    ps_path = PS_DIR / f"{cfg.dataset}.npy"
    k_wht, k_ewht = _load_k_values(cfg.dataset)

    df = pd.read_csv(data_path, dtype={"genotype": str})
    ps = np.load(ps_path)
    L = len(cfg.positions)
    df = df.copy()
    df["genotype"] = df["genotype"].astype(str).str.strip().str.zfill(L)

    position_labels = build_position_labels(
        df,
        list(cfg.positions),
        label_overrides=cfg.position_label_overrides or None,
        label_from_genotype_sequence=cfg.label_from_genotype_sequence,
    )
    labels = [position_labels[p] for p in cfg.positions]

    y_wht = prepare_centered_y(df, L, mode="WHT")
    y_ewht = prepare_centered_y(df, L, mode="p-WHT", ps=ps)
    c_wht = pfwht(y_wht, ps=np.full(L, 0.5), norm="expectation")
    c_ewht = pfwht(y_ewht, ps=ps, norm="expectation")

    mask_wht = np.zeros_like(c_wht, dtype=bool)
    mask_ewht = np.zeros_like(c_ewht, dtype=bool)
    idx_sorted_wht = np.argsort(np.square(c_wht))[::-1]
    idx_sorted_ewht = np.argsort(np.square(c_ewht))[::-1]
    mask_wht[idx_sorted_wht[:k_wht]] = True
    mask_ewht[idx_sorted_ewht[:k_ewht]] = True

    combined = combined_pairwise_coefficient_matrix(c_wht, mask_wht, c_ewht, mask_ewht, L)

    overrides = cfg.position_label_overrides or None
    r2_cfg = R2DeltaMapsConfig(
        dataset=cfg.dataset,
        data_path=data_path,
        ps_path=ps_path,
        summary_path=SUMMARY_PATH,
        r2_wht_path=FITNESS_PLOTS / f"{cfg.dataset}_wht.npy",
        r2_pwht_path=FITNESS_PLOTS / f"{cfg.dataset}_pwht.npy",
        positions=list(cfg.positions),
        label_overrides=overrides,
        label_from_genotype_sequence=cfg.label_from_genotype_sequence,
    )
    map_wht, map_ewht, _label_list_analysis = load_k95_delta_maps(r2_cfg, pfwht)
    display_labels = display_mutation_labels(labels, cfg.plot_positions)
    r2_wht_runs = np.load(r2_cfg.r2_wht_path)
    r2_ewht_runs = np.load(r2_cfg.r2_pwht_path)
    totals_ewht, totals_wht = compute_k95_r2_order_dicts(
        r2_wht_runs,
        r2_ewht_runs,
        idx_sorted_wht,
        idx_sorted_ewht,
        k_wht,
        k_ewht,
        cfg.magnitude_orders,
        high_order_bucket=cfg.resolved_high_order_bucket(),
    )

    def setup_pairwise(ax) -> None:
        draw_summary_pairwise_heatmap_panel(
            ax,
            combined,
            display_labels,
            L,
            tick_rotation=cfg.heatmap_tick_rotation,
        )

    save_r2_summary_figure(
        output_path,
        setup_pairwise_panel=setup_pairwise,
        totals_ewht_by_order=totals_ewht,
        totals_wht_by_order=totals_wht,
        map_wht=map_wht,
        map_ewht=map_ewht,
        label_list=display_labels,
        magnitude_orders=cfg.magnitude_orders,
        order_tick_labels=cfg.order_tick_labels,
        order_tick_show=cfg.order_tick_show,
        order_bar_fixed_slots=cfg.order_bar_fixed_slots,
        panel_width_ratios=(
            list(cfg.summary_panel_width_ratios)
            if cfg.summary_panel_width_ratios is not None
            else None
        ),
        r2_legend_loc=cfg.r2_legend_loc,
        diverging_top_n=cfg.diverging_top_n,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate three-panel summary PDFs.")
    parser.add_argument(
        "datasets",
        nargs="*",
        metavar="dataset",
        help=f"Datasets to render (default: all {len(DEFAULT_DATASETS)} configured)",
    )
    args = parser.parse_args(argv)
    names = args.datasets if args.datasets else list(DEFAULT_DATASETS)
    unknown = set(names) - set(DATASET_CONFIGS)
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        cfg = DATASET_CONFIGS[name]
        out = PLOTS_DIR / f"{cfg.dataset}.pdf"
        render_summary_pdf(cfg, out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
