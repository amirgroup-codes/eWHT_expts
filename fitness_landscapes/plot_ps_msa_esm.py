from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PS_DIR = ROOT / "data" / "ps"
ESM_DIR = PS_DIR / "esm"
MSA_DIR = PS_DIR / "msa"
OUTPUT_DIR = ROOT / "fitness_landscapes" / "ps_plots"

FIGURE_WIDTH_PTS = 120
FIGURE_HEIGHT_PTS = 100
LINE_WIDTH = 0.5
COLOR_MSA = "#faa808"
COLOR_ESM = "#088408"

DATASETS = {
    "cov2": "COV107-23",
    "cr6261": "CR6261",
    "cov_rbd": "Hu-1",
    "cr9114": "CR9114",
    "mtagbfp": "TagBFP",
    "his3": "His3p",
    "beta": "β-lactamase",
}
NPY_STEM_TO_DATASET_KEY = {
    "mTagBFP2": "mtagbfp",
}

font_path = ROOT / "Helvetica.ttf"
try:
    font_prop = fm.FontProperties(fname=str(font_path))
    fm.fontManager.addfont(str(font_path))
    font_name = font_prop.get_name()
except Exception:
    font_name = "sans-serif"

plt.rcParams["font.family"] = font_name
font_size = 6
plt.rcParams["font.size"] = font_size
plt.rcParams["lines.linewidth"] = LINE_WIDTH
plt.rcParams["axes.linewidth"] = LINE_WIDTH
plt.rcParams["xtick.major.width"] = LINE_WIDTH
plt.rcParams["ytick.major.width"] = LINE_WIDTH
plt.rcParams["grid.linewidth"] = LINE_WIDTH
plt.rcParams["pdf.compression"] = False


def _dataset_title(npy_stem: str) -> str:
    dataset_key = NPY_STEM_TO_DATASET_KEY.get(npy_stem, npy_stem)
    return DATASETS.get(dataset_key, npy_stem)


def _shared_datasets() -> list[str]:
    esm_files = {path.stem for path in ESM_DIR.glob("*.npy")}
    msa_files = {path.stem for path in MSA_DIR.glob("*.npy")}
    return sorted(esm_files & msa_files)


def plot_dataset(npy_stem: str) -> Path:
    msa_ps = np.load(MSA_DIR / f"{npy_stem}.npy")
    esm_ps = np.load(ESM_DIR / f"{npy_stem}.npy")
    if len(msa_ps) != len(esm_ps):
        raise ValueError(
            f"MSA and ESM ps lengths differ for {npy_stem}: "
            f"{len(msa_ps)} vs {len(esm_ps)}"
        )

    x_positions = np.arange(1, len(msa_ps) + 1)
    tick_positions = x_positions if len(msa_ps) <= 5 else x_positions[::2]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_PTS / 72, FIGURE_HEIGHT_PTS / 72))
    ax.plot(
        x_positions,
        msa_ps,
        color=COLOR_MSA,
        marker=".",
        markersize=4,
        linewidth=LINE_WIDTH,
        label="MSA",
    )
    ax.plot(
        x_positions,
        esm_ps,
        color=COLOR_ESM,
        marker=".",
        markersize=4,
        linewidth=LINE_WIDTH,
        label="ESM2",
    )

    ax.set_title(_dataset_title(npy_stem), fontsize=8, fontfamily=font_name)
    ax.set_xlabel("Position", fontsize=font_size, fontfamily=font_name, labelpad=2)
    ax.set_ylabel("$p_i$", fontsize=font_size, fontfamily=font_name, labelpad=2)
    ax.set_xlim(0.5, len(msa_ps) + 0.5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(i) for i in tick_positions], fontfamily=font_name, fontsize=font_size)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=font_size, fontfamily=font_name)
    ax.tick_params(axis="x", pad=1, labelsize=font_size, width=LINE_WIDTH)
    ax.tick_params(axis="y", pad=1, labelsize=font_size, width=LINE_WIDTH)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        fontsize=font_size,
        prop={"size": font_size},
        frameon=True,
        edgecolor="#d6d6d6",
        handlelength=0.4,
        handletextpad=0.5,
        loc="upper right",
    )
    legend.get_frame().set_linewidth(LINE_WIDTH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{npy_stem}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    datasets = _shared_datasets()
    if not datasets:
        print(f"No shared .npy datasets found in {ESM_DIR} and {MSA_DIR}")
        return

    for npy_stem in datasets:
        out_path = plot_dataset(npy_stem)
        print(f"Saved {npy_stem} to {out_path}")


if __name__ == "__main__":
    main()
