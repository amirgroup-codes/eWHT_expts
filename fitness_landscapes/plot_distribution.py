"""ESM-bin distribution figure (4 panels) for each landscape:

  sequence logo (WT / mutant heights 1 - p_i, p_i) | UMAP of ESM2-650M embeddings (ESM-bin sequences)
  | p_i per site (eWHT vs WHT's 0.5) | WHT Gram matrix of the order <= 2 basis under mu_p

UMAP: mean-pooled ESM2-650M embeddings of every measured sequence (cached), PCA to 50 dims, cosine UMAP fit on
all sequences; 500 random ESM-bin sequences (DMS_score_bin_esm == 1) are plotted.

Outputs (fitness_landscapes/distribution/):
  embeddings/{dataset}_esm2_mean_pooled.npz   embedding cache (recomputed only if the sequence set changes)
  umap/{dataset}_distribution.pdf

  python fitness_landscapes/plot_distribution.py [--datasets cov2 his3] [--device 0] [--force-embed]
"""
from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import logomaker
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

FITNESS_DIR = Path(__file__).resolve().parent
ROOT = FITNESS_DIR.parent
PROCESSED_DIR = ROOT / "data" / "processed_data"
PS_DIR = ROOT / "data" / "ps" / "esm_wildtype_marginal"
ESM_DATA_DIR = PS_DIR / "esm_data"
EMBED_DIR = FITNESS_DIR / "distribution" / "embeddings"
OUT_DIR = FITNESS_DIR / "distribution" / "umap"

FIGURE_WIDTH_PTS, FIGURE_HEIGHT_PTS = 480, 105
LINE_WIDTH = 0.5
font_size = 6
COLOR_WT = "#088408"
COLOR_MUT = "#faa808"
COLOR_EWHT, COLOR_WHT = "#1f77b4", "#d62728"
GRAM_CMAP_POS = LinearSegmentedColormap.from_list("gram_positive", [(0.0, "#ffffff"), (1.0, COLOR_WT)])
GRAM_CMAP_DIV = LinearSegmentedColormap.from_list(
    "gram_diverging", [(0.0, "#2c6fbb"), (0.5, "#ffffff"), (1.0, COLOR_WT)])
GRAM_NORM_DIV = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
N_PLOT = 500                 # ESM-bin sequences drawn in the UMAP panel
EMBED_BATCH = 16
ESM_REPR_LAYER = 33
HF_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
UMAP_METRIC = "cosine"

# Mutated positions per landscape (1-indexed into the wildtype; "NC" = beta's non-coding site).
POSITIONS = {
    "his3": [145, 147, 148, 151, 152, 154, 164, 165, 168, 169, 170],
    "mtagbfp": [20, 45, 63, 127, 143, 158, 168, 172, 174, 197, 206, 207, 231],
    "beta": ["NC", 42, 104, 182, 238],
    "cov2": [26, 27, 28, 31, 35, 50, 53, 56, 57, 58],
    "cr9114": [29, 30, 31, 52, 57, 58, 59, 71, 74, 75, 76, 77, 84, 87, 95, 106],
    "cr6261": [28, 30, 58, 59, 62, 74, 75, 76, 77, 79, 104],
    "cov_rbd": [339, 371, 373, 375, 417, 440, 446, 477, 478, 484, 493, 496, 498, 501, 505],
}

try:
    fm.fontManager.addfont(str(ROOT / "Helvetica.ttf"))
    font_name = fm.FontProperties(fname=str(ROOT / "Helvetica.ttf")).get_name()
except Exception:
    font_name = "sans-serif"
plt.rcParams.update({"font.family": font_name, "font.size": font_size, "lines.linewidth": LINE_WIDTH,
                     "axes.linewidth": LINE_WIDTH, "xtick.major.width": LINE_WIDTH,
                     "ytick.major.width": LINE_WIDTH, "grid.linewidth": LINE_WIDTH, "pdf.compression": False})


def positions_for(stem: str) -> list[int | str]:
    return POSITIONS[stem.split("_")[0] if stem.startswith(("cr6261_", "cr9114_")) else stem]


# ---------------------------------------------------------------- WT / mutant residues
def wt_mut_residues(stem: str, positions: list[int | str]) -> tuple[list[str], list[str]]:
    """Wildtype and mutant residue at each site, read from the processed data.

    Datasets whose genotype_sequence is a 0/1 string list mutations in a 'mutant' column ('A123B;...');
    otherwise the all-zero genotype's sequence is the wildtype and the other allele is the mutant.
    beta's non-coding site is labelled g (WT) / a (mutant).
    """
    df = pd.read_csv(PROCESSED_DIR / f"{stem}.csv", dtype={"genotype": str, "genotype_sequence": str},
                     keep_default_na=False)
    seqs = df["genotype_sequence"].astype(str).tolist()
    binary = all(set(s) <= {"0", "1"} for s in seqs)
    mapping: dict[int, tuple[str, str]] = {}
    if binary:
        for token in ";".join(df["mutant"].astype(str)).split(";"):
            m = re.fullmatch(r"([A-Za-z])(\d+)([A-Za-z])", token.strip())
            if m:
                mapping[int(m.group(2))] = (m.group(1), m.group(3))
    else:
        L = len(str(df["genotype"].iloc[0]))
        wt_rows = df[df["genotype"].astype(str).str.strip() == "0" * L]
        wt_seq = str(wt_rows.iloc[0]["genotype_sequence"]) if len(wt_rows) else seqs[0]
    wt_aas, mut_aas = [], []
    for idx, pos in enumerate(positions):
        if pos == "NC":
            wt, mut = "g", "a"
        elif binary:
            wt, mut = mapping[pos]
        else:
            wt = wt_seq[idx]
            others = {s[idx] for s in seqs} - {wt}
            mut = others.pop() if others else wt
        wt_aas.append(wt)
        mut_aas.append(mut)
    return wt_aas, mut_aas


# ---------------------------------------------------------------- sequence logo
class FixedWTLogo(logomaker.Logo):
    """Logo whose stacks are drawn in a fixed order (wildtype at the bottom)."""

    def __init__(self, df, position_order, **kwargs):
        self.position_order = position_order
        super().__init__(df, **kwargs)

    def _compute_glyphs(self):
        glyph_df = pd.DataFrame(index=self.ps, columns=self.cs, dtype=object)
        for p in self.ps:
            vs = np.array(self.df.loc[p, :])
            order = self.position_order.get(p, np.arange(len(vs))[::-1])
            vs, cs = vs[order], [str(c) for c in self.cs[order]]
            floor = sum((vs - self.vsep) * (vs < 0)) + self.vsep / 2.0
            for v, c in zip(vs, cs):
                ceiling = floor + abs(v)
                glyph_df.loc[p, c] = logomaker.Glyph(
                    p, c, ax=self.ax, floor=floor, ceiling=ceiling, color=self.rgb_dict[c],
                    flip=(v < 0 and self.flip_below), zorder=self.zorder, font_name=self.font_name,
                    alpha=self.alpha, vpad=self.vpad, **self.glyph_kwargs)
                floor = ceiling + self.vsep
        self.glyph_df = glyph_df
        self.glyph_list = [g for g in glyph_df.values.ravel() if isinstance(g, logomaker.Glyph)]


def draw_logo(ax: plt.Axes, positions: list, wt_aas: list[str], mut_aas: list[str], ps: np.ndarray) -> None:
    """Stack heights exactly 1 - p_i (wildtype, green) and p_i (mutant, orange)."""
    rows = []
    for pos, wt, mut, p in zip(positions, wt_aas, mut_aas, np.clip(ps, 0.0, 1.0)):
        rows.append({wt: 1.0} if (pos != "NC" and wt == mut) else {wt: 1.0 - float(p), mut: float(p)})
    df = pd.DataFrame(rows).fillna(0.0)
    df.index = np.arange(1, len(df) + 1)
    cols = list(df.columns)
    order = {}
    for p, wt in zip(df.index, wt_aas):
        others = sorted((c for c in cols if c != wt), key=lambda c: df.loc[p, c], reverse=True)
        order[p] = np.array([cols.index(wt)] + [cols.index(c) for c in others], dtype=int)
    logo = FixedWTLogo(df, position_order=order, ax=ax, color_scheme="gray", width=0.8, vpad=0.05,
                       stack_order="fixed")
    logo.style_glyphs(color=COLOR_MUT)
    logo.style_glyphs_in_sequence(sequence="".join(wt_aas), color=COLOR_WT)
    ticks = np.arange(1, len(positions) + 1)
    ticks = ticks if len(positions) <= 5 else ticks[::2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(i) for i in ticks], fontfamily=font_name, fontsize=font_size)
    ax.set_xlim(0.5, len(positions) + 0.5)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Position", fontsize=font_size, fontfamily=font_name, labelpad=2)
    ax.set_ylabel("Probability", fontsize=font_size, fontfamily=font_name)
    ax.tick_params(axis="both", width=LINE_WIDTH, labelsize=font_size)
    logo.style_spines(visible=False)
    logo.style_spines(spines=("left", "bottom"), visible=True, linewidth=LINE_WIDTH)


# ---------------------------------------------------------------- ESM2 embeddings + UMAP
def load_sequences(stem: str, L: int) -> pd.DataFrame:
    """Measured sequences with their ESM-bin label, in processed-data order."""
    proc = pd.read_csv(PROCESSED_DIR / f"{stem}.csv", dtype={"genotype": str, "genotype_sequence": str})
    esm = pd.read_csv(ESM_DATA_DIR / f"{stem}.csv", dtype={"genotype": str})
    for d in (proc, esm):
        d["genotype"] = d["genotype"].astype(str).str.strip().str.zfill(L)
    df = proc[["genotype", "genotype_sequence"]].merge(esm[["genotype", "DMS_score_bin_esm"]], on="genotype",
                                                       how="left")
    df["DMS_score_bin_esm"] = df["DMS_score_bin_esm"].fillna(0).astype(int)
    df["genotype_sequence"] = df["genotype_sequence"].astype(str).str.strip()
    return df


@lru_cache(maxsize=1)
def _hf_bundle(device_id: int | None):
    import torch
    from transformers import AutoTokenizer, EsmModel

    device = torch.device(f"cuda:{device_id}" if device_id is not None else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    return EsmModel.from_pretrained(HF_ESM_MODEL).eval().to(device), AutoTokenizer.from_pretrained(HF_ESM_MODEL), device


def embed_mean_pooled(sequences: list[str], device_id: int | None) -> np.ndarray:
    """Mean-pooled last-layer ESM2-650M embeddings: fair-esm with the local weights if available,
    otherwise the Hugging Face checkpoint."""
    import torch

    try:
        sys.path.insert(0, str(PS_DIR))
        from esm_scoring import load_esm_bundle

        model, alphabet, batch_converter, device = load_esm_bundle(device_id)
        rows = []
        for start in range(0, len(sequences), EMBED_BATCH):
            chunk = sequences[start:start + EMBED_BATCH]
            _, _, tokens = batch_converter([(f"s{i}", s) for i, s in enumerate(chunk, start=start)])
            tokens = tokens.to(device)
            with torch.no_grad():
                rep = model(tokens, repr_layers=[ESM_REPR_LAYER], return_contacts=False)["representations"][
                    ESM_REPR_LAYER]
            mask = (tokens != alphabet.padding_idx).unsqueeze(-1)
            rows.append(((rep * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
        return np.vstack(rows)
    except (ImportError, FileNotFoundError):
        model, tokenizer, device = _hf_bundle(device_id)
        rows = []
        for start in range(0, len(sequences), EMBED_BATCH):
            enc = tokenizer(sequences[start:start + EMBED_BATCH], return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            rows.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
        return np.vstack(rows)


def embeddings_for(stem: str, df: pd.DataFrame, device_id: int | None, force: bool) -> np.ndarray:
    cache = EMBED_DIR / f"{stem}_esm2_mean_pooled.npz"
    genotypes = df["genotype"].astype(str).tolist()
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        if [str(g) for g in z["genotypes"].tolist()] == genotypes:
            return z["embeddings"]
    print(f"  Embedding {len(df)} sequences for {stem}...")
    emb = embed_mean_pooled(df["genotype_sequence"].tolist(), device_id).astype(np.float32)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, embeddings=emb, is_esm_bin=df["DMS_score_bin_esm"].to_numpy(np.uint8),
                        genotypes=np.array(genotypes, dtype=object))
    return emb


def umap_coords(embeddings: np.ndarray) -> np.ndarray:
    import umap

    n, d = embeddings.shape
    x = embeddings if (d <= 50 and n <= 4096) else PCA(n_components=min(50, d, n - 1),
                                                         random_state=42).fit_transform(embeddings)
    return umap.UMAP(n_neighbors=min(30, max(2, x.shape[0] - 1)), min_dist=0.1, metric=UMAP_METRIC,
                     random_state=42).fit_transform(x)


def draw_umap(ax: plt.Axes, coords: np.ndarray) -> None:
    ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.85, color=COLOR_WT, edgecolors="none", rasterized=True)
    ax.set_xlabel("UMAP 1", fontsize=font_size, fontfamily=font_name)
    ax.set_ylabel("UMAP 2", fontsize=font_size, fontfamily=font_name)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    handle = Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_WT, markersize=4, linestyle="None",
                    label="ESM2-650M")
    leg = ax.legend(handles=[handle], fontsize=font_size, prop={"size": font_size}, frameon=True,
                    edgecolor="#d6d6d6", handlelength=0.4, handletextpad=0.5, loc="upper right")
    leg.get_frame().set_linewidth(LINE_WIDTH)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------- p_i and Gram panels
def draw_ps(ax: plt.Axes, ps: np.ndarray) -> None:
    x = np.arange(1, len(ps) + 1)
    ticks = x if len(ps) <= 5 else x[::2]
    ax.plot(x, ps, color=COLOR_EWHT, marker=".", markersize=4, linewidth=LINE_WIDTH, label="eWHT")
    ax.plot(x, [0.5] * len(ps), color=COLOR_WHT, marker=".", markersize=4, linewidth=LINE_WIDTH, label="WHT")
    ax.set_xlabel("Position", fontsize=font_size, fontfamily=font_name)
    ax.set_ylabel("$p_i$", fontsize=font_size, fontfamily=font_name)
    ax.set_xlim(0.5, len(ps) + 0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(i) for i in ticks], fontfamily=font_name, fontsize=font_size)
    ax.tick_params(axis="both", width=LINE_WIDTH, labelsize=font_size)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=font_size, fontfamily=font_name)
    leg = ax.legend(fontsize=font_size, prop={"size": font_size}, frameon=True, edgecolor="#d6d6d6",
                    handlelength=0.4, handletextpad=0.5, loc="upper right")
    leg.get_frame().set_linewidth(LINE_WIDTH)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def wht_gram_order2(ps: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Gram matrix of the order <= 2 WHT basis under mu_p: E[chi_S chi_T] = prod_{k in S ^ T} (1 - 2 p_k).
    Also returns the block bounds [0, end of order 0, end of order 1, end of order 2]."""
    mu = 1.0 - 2.0 * np.asarray(ps, dtype=float)
    L = len(mu)
    sets = [()] + [(i,) for i in range(L)] + list(combinations(range(L), 2))
    g = np.array([[float(np.prod([mu[k] for k in set(a) ^ set(b)])) if set(a) ^ set(b) else 1.0
                   for b in sets] for a in sets])
    bounds = [0]
    for order in (0, 1, 2):
        bounds.append(bounds[-1] + sum(1 for s in sets if len(s) == order))
    return g, bounds


def draw_gram(ax: plt.Axes, g: np.ndarray, bounds: list[int]) -> None:
    if np.any(g < 0):
        im = ax.imshow(g, aspect="auto", cmap=GRAM_CMAP_DIV, norm=GRAM_NORM_DIV, interpolation="nearest")
        ticks, labels = [-1.0, 1.0], ["-1", "1"]
    else:
        im = ax.imshow(g, aspect="auto", cmap=GRAM_CMAP_POS, vmin=0.0, vmax=1.0, interpolation="nearest")
        ticks, labels = [0.0, 1.0], ["0", "1"]
    starts = [bounds[1], bounds[2]]  # first-order and second-order blocks
    for setter_ticks, setter_labels in ((ax.set_xticks, ax.set_xticklabels), (ax.set_yticks, ax.set_yticklabels)):
        setter_ticks(starts)
        setter_labels(["1", "2"], fontfamily=font_name, fontsize=font_size)
    ax.set_xlabel("WHT epistasis order", fontsize=font_size, fontfamily=font_name)
    ax.set_ylabel("WHT epistasis order", fontsize=font_size, fontfamily=font_name)
    ax.tick_params(axis="both", width=LINE_WIDTH, labelsize=font_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    cb.ax.set_yticklabels(labels, fontsize=font_size)
    cb.ax.yaxis.set_minor_locator(mticker.NullLocator())


# ---------------------------------------------------------------- figure
def plot_dataset(stem: str, device_id: int | None = None, force_embed: bool = False) -> Path:
    ps = np.load(PS_DIR / f"{stem}.npy").astype(float)
    positions = positions_for(stem)
    if len(positions) != len(ps):
        raise ValueError(f"{stem}: {len(ps)} ps but {len(positions)} positions")
    wt_aas, mut_aas = wt_mut_residues(stem, positions)

    df = load_sequences(stem, len(ps))
    is_esm = (df["DMS_score_bin_esm"] == 1).to_numpy()
    print(f"  UMAP on {len(df)} embeddings (metric={UMAP_METRIC})...")
    coords = umap_coords(embeddings_for(stem, df, device_id, force_embed))
    esm_idx = np.flatnonzero(is_esm)
    pick = np.random.default_rng(42).choice(esm_idx, size=min(N_PLOT, len(esm_idx)), replace=False)

    fig, (ax_logo, ax_umap, ax_ps, ax_gram) = plt.subplots(
        1, 4, figsize=(FIGURE_WIDTH_PTS / 72, FIGURE_HEIGHT_PTS / 72), facecolor="none", constrained_layout=True)
    draw_logo(ax_logo, positions, wt_aas, mut_aas, ps)
    draw_umap(ax_umap, coords[pick])
    draw_ps(ax_ps, ps)
    draw_gram(ax_gram, *wht_gram_order2(ps))

    fig.patch.set_alpha(0.0)
    for ax in fig.get_axes():
        ax.patch.set_facecolor("none")
        ax.patch.set_alpha(0.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{stem}_distribution.pdf"
    fig.savefig(out, dpi=300, transparent=True)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", help="default: every dataset with ps and esm_data")
    ap.add_argument("--device", type=int, default=None, help="CUDA device id")
    ap.add_argument("--force-embed", action="store_true", help="recompute embeddings even if cached")
    args = ap.parse_args()
    stems = args.datasets or sorted(p.stem for p in PS_DIR.glob("*.npy")
                                    if (PROCESSED_DIR / f"{p.stem}.csv").exists()
                                    and (ESM_DATA_DIR / f"{p.stem}.csv").exists())
    for stem in stems:
        print(f"Saved {plot_dataset(stem, args.device, args.force_embed)}")


if __name__ == "__main__":
    main()
