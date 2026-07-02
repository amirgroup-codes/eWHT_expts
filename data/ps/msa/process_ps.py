from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CURR_DIR = Path(__file__).resolve().parent
DATA_DIR = CURR_DIR.parent
PROCESSED_MSA_DIR = DATA_DIR / "../processed_msa"
PROCESSED_DATA_DIR = DATA_DIR.parent / "processed_data"

# Mapping from protein/msa names to processed_data file prefixes.
PROCESSED_DATA_FILEMAP = {
    "his3": "his3",
    "mTagBFP2": "mtagbfp",
    "beta": "beta",
    "cov2": "cov2",
    "cr9114": "cr9114",
    "cr6261": "cr6261",
    "cov_rbd": "cov_rbd",
}

# Hyperparameters
LOWER_P = 0.05
UPPER_P = 0.95

# Protein metadata: name -> (positions, msa_filename)
PROTEIN_METADATA = {
    "his3": ([145, 147, 148, 151, 152, 154, 164, 165, 168, 169, 170], "his3"),
    "mTagBFP2": ([20, 45, 63, 127, 143, 158, 168, 172, 174, 197, 206, 207, 231], "mTagBFP2"),
    "beta": (["NC", 42, 104, 182, 238], "beta"),
    "cov2": ([26, 27, 28, 31, 35, 50, 53, 56, 57, 58], "cov2"),
    "cr9114": ([29, 30, 31, 52, 57, 58, 59, 71, 74, 75, 76, 77, 84, 87, 95, 106], "cr9114"),
    "cr6261": ([28, 30, 58, 59, 62, 74, 75, 76, 77, 79, 104], "cr6261"),
    "cov_rbd": ([339, 371, 373, 375, 417, 440, 446, 477, 478, 484, 493, 496, 498, 501, 505], "cov_rbd"),
}

def _read_a2m(path: Path) -> list[str]:
    """Read A2M/FASTA file and return list of sequences (gaps and lowercase preserved)."""
    sequences: list[str] = []
    current_seq: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)

    if current_seq:
        sequences.append("".join(current_seq))

    return sequences


def _resolve_processed_csv_path(protein_name: str, msa_filename: str) -> Path:
    base_name = PROCESSED_DATA_FILEMAP.get(protein_name, msa_filename)
    candidates = sorted(PROCESSED_DATA_DIR.glob(f"{base_name}*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find processed_data CSV for protein '{protein_name}' using prefix '{base_name}'."
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


def _count_wt_mut_in_msa(sequences: list[str], pos_1indexed: int, wt_aa: str, mut_aa: str) -> float:
    """
    Count WT and Mut occurrences. 
    If mut_aa is 'X', treats V, L, and I as the mutant set.
    """
    wt_upper = wt_aa.upper()
    # Define mutant set. If X, include V, L, I (for cov2 dataset). Otherwise, just the provided AA.
    if mut_aa.upper() == 'X':
        mut_set = {'V', 'L', 'I'}
    else:
        mut_set = {mut_aa.upper()}
        
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
        elif ch in mut_set:
            mut_count += 1

    total = wt_count + mut_count
    if total == 0:
        return 0.5
    return mut_count / total


def _get_ps_from_processed_data(protein_name: str, msa_filename: str) -> tuple[list[float], str]:
    csv_path = _resolve_processed_csv_path(protein_name, msa_filename)
    df = pd.read_csv(csv_path, dtype={"genotype": str, "genotype_sequence": str}, keep_default_na=False)

    if "genotype_sequence" not in df.columns:
        raise ValueError(f"Processed CSV {csv_path} must contain a 'genotype_sequence' column.")
    if "genotype" not in df.columns:
        raise ValueError(f"Processed CSV {csv_path} must contain a 'genotype' column.")

    if df.empty:
        raise ValueError(f"Processed CSV {csv_path} is empty.")

    seqs = df["genotype_sequence"].astype(str).tolist()
    if len({len(seq) for seq in seqs}) != 1:
        raise ValueError(f"All genotype_sequence rows in {csv_path} must have the same length.")

    positions = PROTEIN_METADATA[protein_name][0]
    msa_path = PROCESSED_MSA_DIR / f"{msa_filename}.a2m"
    if not msa_path.exists():
        raise FileNotFoundError(f"MSA not found: {msa_path}")
    msa_seqs = _read_a2m(msa_path)

    if _is_binary_genotype_sequence(seqs):
        # Binary processed data: infer wildtype/mutant amino acids from mutant annotations.
        mapping: dict[int, tuple[str, str]] = {}
        for mutant_str in df["mutant"].astype(str).tolist():
            for token in mutant_str.split(";"):
                parsed = _parse_mutant_token(token)
                if parsed is None:
                    continue
                if parsed[0] == "NC":
                    continue
                wt_aa, pos, mut_aa = parsed
                if pos in mapping:
                    if mapping[pos] != (wt_aa, mut_aa):
                        raise ValueError(
                            f"Conflicting mutant mapping at position {pos}: "
                            f"existing={mapping[pos]}, new=({wt_aa}, {mut_aa})"
                        )
                else:
                    mapping[pos] = (wt_aa, mut_aa)

        wt_seq = str(df.loc[df.index[0], "mutated_sequence"]).strip()
        ps: list[float] = []
        for pos in positions:
            if pos == "NC":
                ps.append(0.5)
                continue
            if pos not in mapping:
                raise ValueError(f"Could not infer WT/MUT mapping for position {pos} in {csv_path}")
            wt_aa, mut_aa = mapping[pos]
            ps.append(min(max(_count_wt_mut_in_msa(msa_seqs, pos, wt_aa, mut_aa), LOWER_P), UPPER_P))

    else:
        zero_genotype = None
        if "genotype" in df.columns:
            genotype_length = len(str(df.loc[df.index[0], "genotype"]))
            zero_genotype = "0" * genotype_length
            zero_rows = df.loc[df["genotype"].astype(str).str.strip() == zero_genotype]
        else:
            zero_rows = df.iloc[0:0]

        if len(zero_rows) > 0:
            wt_seq = str(zero_rows.iloc[0]["genotype_sequence"])
        else:
            wt_seq = seqs[0]
            print(
                f"Warning: no all-zero genotype row found in {csv_path}, using first row as wildtype sequence"
            )

        ps = []
        for idx, pos in enumerate(positions):
            if pos == "NC":
                ps.append(0.5)
                continue
            wt_aa = wt_seq[idx]
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
            ps.append(min(max(_count_wt_mut_in_msa(msa_seqs, pos, wt_aa, mut_aa), LOWER_P), UPPER_P))

    return ps, wt_seq


def process_msa_to_ps() -> None:
    """Compute and save binary mutation probability distributions for each protein."""
    for protein_name, (positions, msa_filename) in PROTEIN_METADATA.items():
        try:
            ps, wt_seq = _get_ps_from_processed_data(protein_name, msa_filename)
        except FileNotFoundError as exc:
            print(f"Processed data CSV not found for {protein_name}: {exc}")
            continue
        except Exception as exc:
            print(f"Error processing {protein_name}: {exc}")
            continue

        output_path = CURR_DIR / f"{protein_name}.npy"
        np.save(output_path, np.array(ps, dtype=np.float32))
        print(
            f"Processing {protein_name}: inferred WT sequence '{wt_seq}' and saved p(mutant) to {output_path}"
        )


def plot_distributions() -> None:
    """Load PS arrays and create visualizations."""
    dist_dir = CURR_DIR / "distribution"
    dist_dir.mkdir(exist_ok=True)
    
    npy_files = sorted(CURR_DIR.glob("*.npy"))
    
    for npy_path in npy_files:
        protein_name = npy_path.stem
        ps = np.load(npy_path)
        
        print(f"Plotting {protein_name}...")
        
        positions = np.arange(1, len(ps) + 1)
        
        # Create 2-subplot figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Top plot: p vs position
        ax_top = axes[0]
        ax_top.plot(positions, ps, marker='o', linestyle='-', linewidth=2, markersize=6)
        ax_top.set_xlabel("Position", fontsize=12)
        ax_top.set_ylabel("p", fontsize=12)
        ax_top.set_title(f"{protein_name}: Mutation Probability by Position", fontsize=14)
        ax_top.grid(True, alpha=0.3)
        ax_top.set_ylim([0, 1.0])
        
        # Bottom plot: sqrt(p/(1-p)) vs position
        # Avoid division by zero / log of zero
        ps_safe = np.clip(ps, 1e-10, 1.0 - 1e-10)
        odds_ratio = np.sqrt(ps_safe / (1.0 - ps_safe))
        
        ax_bottom = axes[1]
        ax_bottom.plot(positions, odds_ratio, marker='s', linestyle='-', linewidth=2, markersize=6, color='orange')
        ax_bottom.set_xlabel("Position", fontsize=12)
        ax_bottom.set_ylabel("sqrt(p/(1-p))", fontsize=12)
        ax_bottom.set_title(f"{protein_name}: Odds Ratio by Position", fontsize=14)
        ax_bottom.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        png_path = dist_dir / f"{protein_name}.png"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {png_path}")


def main() -> None:
    process_msa_to_ps()
    plot_distributions()

if __name__ == "__main__":
    main()
