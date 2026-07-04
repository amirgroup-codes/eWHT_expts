from __future__ import annotations
import re
import sys
from itertools import combinations, product
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import logomaker
from sklearn.preprocessing import OneHotEncoder
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

try:
    import umap
except ImportError:
    umap = None

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed_data"
PROCESSED_MSA_DIR = ROOT / "data" / "processed_msa"
PS_DIR = ROOT / "data" / "ps" / "msa"
OUTPUT_DIR = ROOT / "fitness_landscapes" / "distribution" / "msa"
FIGURE_WIDTH_PTS = 480
FIGURE_HEIGHT_PTS = 105
LINE_WIDTH = 0.5
COLOR_WT = "#088408"  # Green
COLOR_MUT = "#faa808"  # Orange
COLOR_EWHT_BLUE = "#2c6fbb"
GRAM_CMAP_POS = LinearSegmentedColormap.from_list(
    "gram_positive",
    [(0.0, "#ffffff"), (1.0, COLOR_WT)],
)
GRAM_CMAP_DIV = LinearSegmentedColormap.from_list(
    "gram_diverging",
    [(0.0, COLOR_EWHT_BLUE), (0.5, "#ffffff"), (1.0, COLOR_WT)],
)
GRAM_NORM_DIV = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

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
plt.rcParams['lines.linewidth'] = LINE_WIDTH
plt.rcParams['axes.linewidth'] = LINE_WIDTH
plt.rcParams['xtick.major.width'] = LINE_WIDTH
plt.rcParams['ytick.major.width'] = LINE_WIDTH
plt.rcParams['grid.linewidth'] = LINE_WIDTH
plt.rcParams['pdf.compression'] = False

PROTEIN_METADATA = {
    "his3": ([145, 147, 148, 151, 152, 154, 164, 165, 168, 169, 170], "his3"),
    "mTagBFP2": ([20, 45, 63, 127, 143, 158, 168, 172, 174, 197, 206, 207, 231], "mTagBFP2"),
    "beta": (["NC", 42, 104, 182, 238], "beta"),
    "cov2": ([26, 27, 28, 31, 35, 50, 53, 56, 57, 58], "cov2"),
    "cr9114": ([29, 30, 31, 52, 57, 58, 59, 71, 74, 75, 76, 77, 84, 87, 95, 106], "cr9114"),
    "cr6261": ([28, 30, 58, 59, 62, 74, 75, 76, 77, 79, 104], "cr6261"),
    "cov_rbd": ([339, 371, 373, 375, 417, 440, 446, 477, 478, 484, 493, 496, 498, 501, 505], "cov_rbd"),
}


def dataset_key_from_csv(csv_path: Path) -> str:
    stem = csv_path.stem
    if stem.startswith("cr6261_"):
        return "cr6261"
    if stem.startswith("cr9114_"):
        return "cr9114"
    if stem == "mtagbfp":
        return "mTagBFP2"
    return stem

PROCESSED_DATA_FILEMAP = {
    "his3": "his3",
    "mTagBFP2": "mtagbfp",
    "beta": "beta",
    "cov2": "cov2",
    "cr9114": "cr9114",
    "cr6261": "cr6261",
}


def _read_a2m(path: Path) -> list[str]:
    sequences: list[str] = []
    current_seq: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
    if current_seq:
        sequences.append("".join(current_seq))
    return sequences


def _resolve_processed_csv_path(dataset_key: str) -> Path:
    base_name = PROCESSED_DATA_FILEMAP.get(dataset_key, dataset_key)
    candidates = sorted(PROCESSED_DIR.glob(f"{base_name}*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find processed_data CSV for dataset key '{dataset_key}' using prefix '{base_name}'."
        )
    return candidates[0]


def _is_binary_genotype_sequence(seqs: list[str]) -> bool:
    if not seqs:
        return False
    lengths = {len(seq) for seq in seqs}
    if len(lengths) != 1:
        return False
    return all(all(ch in "01" for ch in seq) for seq in seqs)


def _parse_mutant_token(token: str):
    token = token.strip()
    if not token or token.upper() == "WT":
        return None
    if token == "NC":
        return ("NC", None, None)

    m = re.fullmatch(r"([A-Za-z])(\d+)([A-Za-z])", token)
    if not m:
        raise ValueError(f"Could not parse mutant token '{token}'")
    return m.group(1), int(m.group(2)), m.group(3)


def _count_wt_mut_in_msa(sequences: list[str], pos_1indexed: int, wt_aa: str, mut_aa: str) -> tuple[int, int]:
    wt_upper = wt_aa.upper()
    mut_upper = mut_aa.upper()
    wt_count = 0
    mut_count = 0

    for seq in sequences:
        if pos_1indexed - 1 >= len(seq):
            continue
        ch = seq[pos_1indexed - 1]
        if not ch.isalpha():
            continue
        ch = ch.upper()
        if ch == wt_upper:
            wt_count += 1
        elif ch == mut_upper:
            mut_count += 1

    return wt_count, mut_count


def _infer_wt_mut_from_processed_data(csv_path: Path, sequences: list[str], positions: list[int | str]) -> tuple[list[dict[str, float]], list[str], list[str], list[float]]:
    df = pd.read_csv(csv_path, dtype={"genotype": str, "genotype_sequence": str}, keep_default_na=False)
    if df.empty:
        raise ValueError(f"Processed CSV {csv_path} is empty.")

    seqs = df["genotype_sequence"].astype(str).tolist()
    if len({len(seq) for seq in seqs}) != 1:
        raise ValueError(f"All genotype_sequence rows in {csv_path} must have the same length.")

    binary = _is_binary_genotype_sequence(seqs)
    mapping: dict[int, tuple[str, str]] = {}
    wt_sequence = None
    if binary:
        if "mutant" not in df.columns:
            raise ValueError(f"Processed CSV {csv_path} must contain a 'mutant' column for binary genotype data.")
        for mutant_str in df["mutant"].astype(str).tolist():
            for token in mutant_str.split(";"):
                parsed = _parse_mutant_token(token)
                if parsed is None:
                    continue
                if parsed[0] == "NC":
                    continue
                wt_aa, pos, mut_aa = parsed
                if pos in mapping and mapping[pos] != (wt_aa, mut_aa):
                    raise ValueError(
                        f"Conflicting mutant mapping at position {pos}: existing={mapping[pos]}, new=({wt_aa}, {mut_aa})"
                    )
                mapping[pos] = (wt_aa, mut_aa)

    else:
        zero_genotype = None
        if "genotype" in df.columns:
            genotype_length = len(str(df.loc[df.index[0], "genotype"]))
            zero_genotype = "0" * genotype_length
            zero_rows = df.loc[df["genotype"].astype(str).str.strip() == zero_genotype]
        else:
            zero_rows = df.iloc[0:0]

        if len(zero_rows) > 0:
            wt_sequence = str(zero_rows.iloc[0]["genotype_sequence"])
        else:
            wt_sequence = seqs[0]
            print(
                f"Warning: no all-zero genotype row found in {csv_path}, using first row as wildtype sequence"
            )

    pos_data: list[dict[str, float]] = []
    wt_aas: list[str] = []
    mut_aas: list[str] = []
    ps_values: list[float] = []

    for idx, pos in enumerate(positions):
        if pos == "NC":
            pos_data.append({"g": 0.5, "a": 0.5})
            wt_aas.append("g")
            mut_aas.append("a")
            ps_values.append(0.5)
            continue

        if binary:
            if pos not in mapping:
                raise ValueError(f"Could not infer WT/MUT mapping for position {pos} in {csv_path}")
            wt_aa, mut_aa = mapping[pos]
        else:
            if wt_sequence is None:
                raise AssertionError("wt_sequence should be set for non-binary genotype_sequence data")
            wt_aa = wt_sequence[idx]
            alleles = {seq[idx] for seq in seqs}
            alleles.discard("")
            if wt_aa not in alleles:
                raise ValueError(f"Wildtype amino acid '{wt_aa}' not observed at position {pos}")
            if len(alleles) == 1:
                mut_aa = wt_aa
            elif len(alleles) == 2:
                mut_aa = next(a for a in alleles if a != wt_aa)
            else:
                raise ValueError(
                    f"Position {pos} has more than two alleles in processed data: {sorted(alleles)}"
                )

        wt_count, mut_count = _count_wt_mut_in_msa(sequences, int(pos), wt_aa, mut_aa) if isinstance(pos, int) else (0, 0)
        total = wt_count + mut_count
        if total == 0:
            wt_freq = 0.0
            mut_freq = 0.0
            p_value = 0.5
        else:
            wt_freq = wt_count / total
            mut_freq = mut_count / total
            p_value = mut_count / total

        if wt_aa == mut_aa:
            pos_data.append({wt_aa: wt_freq})
        else:
            pos_data.append({wt_aa: wt_freq, mut_aa: mut_freq})
        wt_aas.append(wt_aa)
        mut_aas.append(mut_aa)
        ps_values.append(p_value)

    return pos_data, wt_aas, mut_aas, ps_values


def _get_aa_distribution(sequences: list[str], pos_1indexed: int) -> tuple[str, dict[str, float]]:
    wt_seq = sequences[0]
    pos_0indexed = pos_1indexed - 1
    wt_aa = wt_seq[pos_0indexed].upper()

    counts = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
    total = 0
    for seq in sequences:
        if pos_0indexed < len(seq):
            aa = seq[pos_0indexed].upper()
            if aa in counts:
                counts[aa] += 1
                total += 1

    freqs = {aa: (count / total if total > 0 else 0.0) for aa, count in counts.items()}
    return wt_aa, freqs


def _build_fixed_wt_order(df: pd.DataFrame, wt_sequence: str) -> dict:
    columns = list(df.columns)
    ordered_indices = {}
    for p, wt in zip(df.index, wt_sequence):
        row = df.loc[p]
        if wt in columns:
            wt_idx = columns.index(wt)
            other_columns = [c for c in columns if c != wt]
            other_columns.sort(key=lambda c: row[c], reverse=True)
            ordered_indices[p] = np.array([wt_idx] + [columns.index(c) for c in other_columns], dtype=int)
        else:
            ordered_indices[p] = np.arange(len(columns))[::-1]
    return ordered_indices


def _enforce_min_wt_height(distributions: list[dict], wt_sequence: str, min_height: float = 0.05) -> list[dict]:
    adjusted = []
    for dist, wt_aa in zip(distributions, wt_sequence):
        freq = dist.copy()
        wt_val = float(freq.get(wt_aa, 0.0))
        if wt_val < min_height:
            othersum = sum(value for key, value in freq.items() if key != wt_aa)
            freq[wt_aa] = min_height
            if othersum > 0:
                scale = (1.0 - min_height) / othersum
                for key in freq:
                    if key != wt_aa:
                        freq[key] = float(freq[key]) * scale
        adjusted.append(freq)
    return adjusted


class FixedWTLogo(logomaker.Logo):
    def __init__(self, df, position_order, **kwargs):
        self.position_order = position_order
        super().__init__(df, **kwargs)

    def _compute_glyphs(self):
        glyph_df = pd.DataFrame(index=self.ps, columns=self.cs, dtype=object)
        for p in self.ps:
            vs = np.array(self.df.loc[p, :])
            if self.stack_order == 'big_on_top':
                ordered_indices = np.argsort(vs)
            elif self.stack_order == 'small_on_top':
                tmp_vs = np.zeros(len(vs))
                indices = (vs != 0)
                tmp_vs[indices] = 1.0 / vs[indices]
                ordered_indices = np.argsort(tmp_vs)
            elif self.stack_order == 'fixed':
                ordered_indices = self.position_order.get(p, np.arange(len(vs))[::-1])
            else:
                assert False, 'This line of code should never be called.'

            vs = vs[ordered_indices]
            cs = [str(c) for c in self.cs[ordered_indices]]
            floor = sum((vs - self.vsep) * (vs < 0)) + self.vsep / 2.0
            for v, c in zip(vs, cs):
                ceiling = floor + abs(v)
                this_color = self.rgb_dict[c]
                flip = (v < 0 and self.flip_below)
                glyph = logomaker.Glyph(
                    p,
                    c,
                    ax=self.ax,
                    floor=floor,
                    ceiling=ceiling,
                    color=this_color,
                    flip=flip,
                    zorder=self.zorder,
                    font_name=self.font_name,
                    alpha=self.alpha,
                    vpad=self.vpad,
                    **self.glyph_kwargs,
                )
                glyph_df.loc[p, c] = glyph
                floor = ceiling + self.vsep

        self.glyph_df = glyph_df
        self.glyph_list = [g for g in self.glyph_df.values.ravel() if isinstance(g, logomaker.Glyph)]


def draw_sequence_logo(ax, pos_labels, distributions, wt_aas):
    wt_sequence = ''.join(wt_aas)
    adjusted_distributions = _enforce_min_wt_height(distributions, wt_sequence)
    logo_df = pd.DataFrame(adjusted_distributions).fillna(0.0)
    logo_df.index = np.arange(1, len(logo_df) + 1)
    position_order = _build_fixed_wt_order(logo_df, wt_sequence)

    logo = FixedWTLogo(
        logo_df,
        position_order=position_order,
        ax=ax,
        color_scheme='gray',
        width=0.8,
        vpad=0.05,
        stack_order='fixed',
    )

    logo.style_glyphs(color=COLOR_MUT)
    logo.style_glyphs_in_sequence(sequence=wt_sequence, color=COLOR_WT)

    ax.set_xlabel("Position", fontsize=font_size, fontfamily=font_name, labelpad=2)
    ax.set_ylabel("Probability", fontsize=font_size, fontfamily=font_name)
    positions = np.arange(1, len(pos_labels) + 1)
    tick_positions = positions if len(pos_labels) <= 5 else positions[::2]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(i) for i in tick_positions], fontfamily=font_name, ha='center', va='top', fontsize=font_size)
    ax.set_xlim(0.5, len(pos_labels) + 0.5)
    ax.tick_params(axis='x', width=LINE_WIDTH, labelsize=font_size)
    ax.tick_params(axis='y', width=LINE_WIDTH, labelsize=font_size)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1.0])
    logo.style_spines(visible=False)
    logo.style_spines(spines=('left', 'bottom'), visible=True, linewidth=LINE_WIDTH)


def compute_msa_umap_embeddings(
    sequences: list[str],
    positions: list[int | str],
    wt_aas: list[str],
    mut_aas: list[str],
) -> np.ndarray | None:
    if umap is None:
        return None

    # 1. Only accept sequence if it has the WT or Mut AA at any position.
    criteria = {}
    for i, pos in enumerate(positions):
        if pos == "NC":
            continue
        wt = wt_aas[i].upper()
        mut = mut_aas[i].upper()
        # Build set of allowed AAs for this position
        allowed = {wt}
        if mut == 'X':
            allowed.update({'V', 'L', 'I'})
        else:
            allowed.add(mut)
        criteria[pos] = allowed

    # 2. Filter sequences based on criteria
    filtered_rows = []
    for seq in sequences:
        # Check if sequence is long enough to cover positions
        if any(isinstance(p, int) and len(seq) < p for p in positions):
            continue
        is_relevant = False
        row = []
        # Extract row and check against criteria
        for i, p in enumerate(positions):
            if p == "NC":
                row.append("NC_PLACEHOLDER")
            else:
                aa = seq[p - 1].upper()
                row.append(aa)
                # Check if this residue matches the library's WT or Mutant definition
                if aa in criteria.get(p, set()):
                    is_relevant = True
        if is_relevant:
            filtered_rows.append(row)
    if not filtered_rows:
        return None
    
    # 3. NC Expansion
    expanded_rows = []
    nc_indices = [idx for idx, pos in enumerate(positions) if pos == "NC"]
    for row in filtered_rows:
        if not nc_indices:
            expanded_rows.append(row)
            continue
        for nc_combo in product(['g', 'a'], repeat=len(nc_indices)):
            new_row = list(row)
            for j, allele in enumerate(nc_combo):
                new_row[nc_indices[j]] = allele
            expanded_rows.append(new_row)

    # 4. Sampling cap for UMAP
    max_seqs = 1000
    if len(expanded_rows) > max_seqs:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(expanded_rows), size=max_seqs, replace=False)
        final_rows = [expanded_rows[i] for i in indices]
    else:
        final_rows = expanded_rows

    if len(final_rows) <= 3:
        return None

    # 5. One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X = encoder.fit_transform(final_rows)
    print(f"UMAP Input Shape: {X.shape} (N={X.shape[0]}, Dim={X.shape[1]})")

    # 6. UMAP
    reducer = umap.UMAP(
        metric='hamming', 
        random_state=42
    )
    return reducer.fit_transform(X)

def parse_sequences(df: pd.DataFrame) -> np.ndarray:
    if "genotype" not in df.columns:
        raise ValueError("CSV must contain a 'genotype' column")
    seqs = np.stack([np.fromiter((int(ch) for ch in str(seq).strip()), dtype=np.int8) for seq in df["genotype"]])
    return seqs


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, dtype={"genotype": str, "genotype_sequence": str})
    seqs = parse_sequences(df)
    dataset_key = dataset_key_from_csv(csv_path)
    ps_path = PS_DIR / f"{dataset_key}.npy"
    if not ps_path.exists():
        raise FileNotFoundError(f"Could not find probability file for dataset key '{dataset_key}'")

    ps = np.load(ps_path).astype(float)
    if ps.ndim != 1:
        raise ValueError(f"Loaded ps must be a 1D array, got shape {ps.shape}")
    if len(ps) != seqs.shape[1]:
        raise ValueError(f"ps length {len(ps)} does not match genotype length {seqs.shape[1]} for {csv_path.name}")
    return seqs, ps


def _order2_interaction_sets(L: int) -> list[tuple[int, ...]]:
    sets: list[tuple[int, ...]] = [()]
    sets += [(i,) for i in range(L)]
    sets += list(combinations(range(L), 2))
    return sets


def build_order2_gram_matrices(ps: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Gram matrices for interactions up to 2nd order under the evolutionary measure D_p.
    eWHT basis is orthonormal (identity); WHT parity basis has off-diagonal coupling.
    """
    p = np.asarray(ps, dtype=float)
    mu = 1.0 - 2.0 * p
    sets = _order2_interaction_sets(len(p))
    n = len(sets)

    g_ewht = np.eye(n)
    g_wht = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = set(sets[i]) ^ set(sets[j])
            g_wht[i, j] = float(np.prod([mu[k] for k in diff])) if diff else 1.0

    bounds = [0]
    for order in (0, 1, 2):
        bounds.append(bounds[-1] + sum(1 for s in sets if len(s) == order))
    return g_ewht, g_wht, bounds


def _order_tick_starts(bounds: list[int]) -> list[int]:
    """Tick positions at the start of 1st- and 2nd-order blocks (order 0 unlabeled)."""
    return [bounds[1], bounds[2]]


def draw_gram_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    matrix: np.ndarray,
    label: str,
    bounds: list[int],
    *,
    diverging: bool,
    cbar_pad: float,
    cbar_width: float,
    y_label_x_pos: float,
) -> None:
    if diverging:
        im = ax.imshow(
            matrix,
            aspect='auto',
            cmap=GRAM_CMAP_DIV,
            norm=GRAM_NORM_DIV,
            interpolation='nearest',
        )
        cbar_ticks = [-1.0, 1.0]
        cbar_labels = ['-1', '1']
    else:
        im = ax.imshow(
            matrix,
            aspect='auto',
            cmap=GRAM_CMAP_POS,
            vmin=0.0,
            vmax=1.0,
            interpolation='nearest',
        )
        cbar_ticks = [0.0, 1.0]
        cbar_labels = ['0', '1']

    starts = _order_tick_starts(bounds)
    ax.set_xticks(starts)
    ax.set_xticklabels(['1', '2'], fontfamily=font_name, fontsize=font_size)
    ax.set_yticks(starts)
    ax.set_yticklabels(['1', '2'], fontfamily=font_name, fontsize=font_size)
    ax.set_xlabel(f'{label} epistasis order', fontsize=font_size, fontfamily=font_name, labelpad=2)
    ax.set_ylabel(f'{label} epistasis order', fontsize=font_size, fontfamily=font_name, labelpad=2)
    ax.yaxis.set_label_coords(y_label_x_pos, 0.5)
    ax.tick_params(axis='both', width=LINE_WIDTH, labelsize=font_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax_pos = ax.get_position()
    cbar_ax = fig.add_axes([ax_pos.x1 + cbar_pad, ax_pos.y0, cbar_width, ax_pos.height])
    cb = fig.colorbar(im, cax=cbar_ax, ticks=cbar_ticks)
    cb.ax.yaxis.set_major_locator(mticker.FixedLocator(cbar_ticks))
    cb.ax.set_yticklabels(cbar_labels, fontsize=font_size)
    cb.ax.yaxis.set_minor_locator(mticker.NullLocator())


def _set_transparent_background(fig: plt.Figure) -> None:
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    for ax in fig.get_axes():
        ax.patch.set_facecolor('none')
        ax.patch.set_alpha(0.0)


def plot_dataset(csv_path: Path) -> Path | None:
    dataset_key = dataset_key_from_csv(csv_path)
    if dataset_key not in PROTEIN_METADATA:
        print(f"Skipping {csv_path.name}: no sequence metadata for dataset key '{dataset_key}'")
        return None

    positions, msa_name = PROTEIN_METADATA[dataset_key]
    msa_path = PROCESSED_MSA_DIR / f"{msa_name}.a2m"
    if not msa_path.exists():
        print(f"Skipping {csv_path.name}: missing MSA file {msa_path}")
        return None

    sequences = _read_a2m(msa_path)
    csv_path_for_inference = _resolve_processed_csv_path(dataset_key)
    pos_data, wt_aas, mut_aas, _ = _infer_wt_mut_from_processed_data(csv_path_for_inference, sequences, positions)

    _, ps = load_dataset(csv_path)
    g_ewht_evo, g_wht_evo, gram_bounds = build_order2_gram_matrices(ps)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(FIGURE_WIDTH_PTS / 72, FIGURE_HEIGHT_PTS / 72), facecolor='none')
    # Layout Constants
    left = 0.05
    bottom = 0.18
    main_width = 0.135
    main_height = 0.68
    Y_LABEL_X_POS = -0.22

    # Define variable paddings
    pad_small = 0.05  # Between 1-2 and 2-3 and 3-4
    pad_large = 0.075  # Leading into and between the matrices (4-5)

    # Calculate explicit x-positions
    x0 = left                                     # Logo
    x1 = x0 + main_width + pad_small              # UMAP
    x2 = x1 + main_width + pad_small              # PS (p-values)
    x3 = x2 + main_width + pad_small              # WHT Gram matrix
    x4 = x3 + main_width + pad_large              # eWHT Gram matrix

    logo_ax = fig.add_axes([x0, bottom, main_width, main_height])
    umap_ax = fig.add_axes([x1, bottom, main_width, main_height])
    ps_ax   = fig.add_axes([x2, bottom, main_width, main_height])
    wht_ax  = fig.add_axes([x3, bottom, main_width, main_height])
    ewht_ax = fig.add_axes([x4, bottom, main_width, main_height])
    for ax in [logo_ax, umap_ax, ps_ax, wht_ax, ewht_ax]:
        ax.tick_params(axis='x', pad=1, labelsize=font_size)
        ax.tick_params(axis='y', pad=1, labelsize=font_size)

    draw_sequence_logo(logo_ax, positions, pos_data, wt_aas)

    umap_embedded = compute_msa_umap_embeddings(sequences, positions, wt_aas, mut_aas)
    if umap_embedded is not None:
        umap_ax.scatter(umap_embedded[:, 0], umap_embedded[:, 1], s=8, alpha=0.8, color=COLOR_WT, edgecolors='none', label='MSA')
        umap_ax.set_xlabel('UMAP 1', fontsize=font_size, fontfamily=font_name, labelpad=2)
        umap_ax.set_ylabel('UMAP 2', fontsize=font_size, fontfamily=font_name, labelpad=2)
        umap_ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, integer=True))
        umap_ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, integer=True))
        umap_ax.spines['top'].set_visible(False)
        umap_ax.spines['right'].set_visible(False)
        legend = umap_ax.legend(
            fontsize=font_size,
            prop={'size': font_size},
            frameon=True,
            edgecolor='#d6d6d6',
            handlelength=0.4,
            handletextpad=0.5,
            loc='upper right',
        )
        umap_ax.yaxis.set_label_coords(Y_LABEL_X_POS, 0.5)
        legend.get_frame().set_linewidth(LINE_WIDTH)
    else:
        umap_ax.axis('off')
        umap_ax.set_title('UMAP unavailable', fontsize=font_size)

    x_positions = np.arange(1, len(ps) + 1)
    ps_ax.plot(x_positions, ps, color='#1f77b4', marker='.', markersize=4, linewidth=LINE_WIDTH, label='eWHT')
    ps_ax.plot(x_positions, [0.5] * len(ps), color='#d62728', marker='.', markersize=4, linewidth=LINE_WIDTH, label='WHT')
    ps_ax.set_xlabel('Position', fontsize=font_size, fontfamily=font_name, labelpad=2)
    ps_ax.set_ylabel('$p_i$', fontsize=font_size, fontfamily=font_name, labelpad=2)
    tick_positions = x_positions if len(ps) <= 5 else x_positions[::2]
    ps_ax.set_xlim(0.5, len(ps) + 0.5)
    ps_ax.set_xticks(tick_positions)
    ps_ax.set_xticklabels([str(i) for i in tick_positions], fontfamily=font_name, fontsize=font_size)
    ps_ax.tick_params(axis='x', width=LINE_WIDTH, labelsize=font_size)
    ps_ax.tick_params(axis='y', width=LINE_WIDTH, labelsize=font_size)
    ps_ax.set_ylim(0, 1.1)
    ps_ax.set_yticks([0, 0.5, 1.0])
    ps_ax.set_yticklabels(['0', '0.5', '1'], fontsize=font_size, fontfamily=font_name)
    legend = ps_ax.legend(
        fontsize=font_size,
        prop={'size': font_size},
        frameon=True,
        edgecolor='#d6d6d6',
        handlelength=0.4,
        handletextpad=0.5,
        loc='upper right',
    )
    legend.get_frame().set_linewidth(LINE_WIDTH)
    ps_ax.spines['top'].set_visible(False)
    ps_ax.spines['right'].set_visible(False)

    CBAR_PAD = 0.005
    CBAR_WIDTH = 0.005

    wht_diverging = bool(np.any(g_wht_evo < 0))

    for ax, matrix, label, diverging in [
        (wht_ax, g_wht_evo, 'WHT', wht_diverging),
        (ewht_ax, g_ewht_evo, 'eWHT', False),
    ]:
        draw_gram_heatmap(
            fig,
            ax,
            matrix,
            label,
            gram_bounds,
            diverging=diverging,
            cbar_pad=CBAR_PAD,
            cbar_width=CBAR_WIDTH,
            y_label_x_pos=Y_LABEL_X_POS,
        )

    out_path = OUTPUT_DIR / f'{csv_path.stem}_distribution.pdf'
    _set_transparent_background(fig)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)
    return out_path


def main() -> None:
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"Processed data directory not found: {PROCESSED_DIR}")

    csv_paths = sorted([p for p in PROCESSED_DIR.glob('*.csv') if p.name != 'test.csv'], key=lambda p: p.name)
    if not csv_paths:
        print('No CSV datasets found in', PROCESSED_DIR)
        return

    for csv_path in csv_paths:
        try:
            out_path = plot_dataset(csv_path)
            if out_path:
                print(f"Saved distribution figure for {csv_path.name} to {out_path}")
        except Exception as exc:
            print(f"Skipped {csv_path.name}: {exc}")


if __name__ == '__main__':
    main()
