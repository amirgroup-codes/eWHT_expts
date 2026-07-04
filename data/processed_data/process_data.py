from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO

CURR_DIR = Path(__file__).resolve().parent
DATA_DIR = CURR_DIR.parent

def infer_mutant(mutated_sequence: str, wt_sequence: str) -> str:
    """Infer mutation strings in WTPOSMUT format, e.g. A320C.

    Multiple substitutions are joined with ";". If no substitutions are present,
    returns "WT".
    """
    mutated_sequence = mutated_sequence.strip().strip('"')
    wt_sequence = wt_sequence.strip().strip('"')

    if len(mutated_sequence) != len(wt_sequence):
        raise ValueError(
            "Sequence length mismatch: "
            f"mutated={len(mutated_sequence)}, wt={len(wt_sequence)}"
        )

    mutations: list[str] = []
    for i, (wt_aa, mut_aa) in enumerate(zip(wt_sequence, mutated_sequence), start=1):
        if wt_aa != mut_aa:
            mutations.append(f"{wt_aa}{i}{mut_aa}")

    return ";".join(mutations) if mutations else "WT"


def process_his3() -> None:
    """
    Process raw HIS3 data into standardized columns.

    Input:  data/raw_data/his3p_raw_data.csv
    Output: data/processed_data/his3.csv

    Paper: https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1008079
    """
    HIS3 = "MTEQKALVKRITNETKIQIAISLKGGPLAIEHSIFPEKEAEAVAEQATQSQVINVHTGIGFLDHMIHALAKHSGWSLIVECIGDLHIDDHHTTEDCGIALGQAFKEALGAVRGVKRFGSGFAPLDEALSRAVVDLSNRPYAVVELGLQREKVGDLSCEMIPHFLESFAEASRITLHVDCLRGKNDHHRSESAFKALAVAIREATSPNGTNDVPSTKGVLM"
    POSITIONS = [145, 147, 148, 151, 152, 154, 164, 165, 168, 169, 170]

    input_path = DATA_DIR / "raw_data" / "his3p_raw_data.csv"
    output_path = CURR_DIR / "his3.csv"

    df = pd.read_csv(input_path)
    df = df.loc[df["seq"].notna()].copy()
    df["full_sequence"] = df["seq"].astype(str).str.strip().str.strip('"')
    df = df.loc[df["full_sequence"] != ""].copy()
    df["log_fitness"] = pd.to_numeric(df["log_fitness"], errors="coerce")
    df = df.loc[df["log_fitness"].notna()].copy()

    # Extract binary sequence
    df["genotype_sequence"] = df["full_sequence"].apply(
        lambda x: "".join(x[i - 1] for i in POSITIONS)
    )

    # At each of 11 positions, use the 2 most frequent residues as binary 0/1 alleles.
    split_cols = pd.DataFrame(df["genotype_sequence"].apply(list).tolist())
    best = []
    for col in range(len(POSITIONS)):
        counts = split_cols[col].value_counts()
        if len(counts.index) < 2:
            raise ValueError(
                f"Position {POSITIONS[col]} has fewer than 2 observed alleles."
            )
        best.append([counts.index[0], counts.index[1]])

    records = []
    for bits in itertools.product((0, 1), repeat=len(POSITIONS)):
        genotype = "".join(str(b) for b in bits)
        local_seq = "".join(best[i][bits[i]] for i in range(len(bits)))
        matches = df.loc[df["genotype_sequence"] == local_seq]

        if matches.empty:
            fitness = np.nan
        else:
            # Match previous behavior: average in linear space, then map back to log space.
            fitness = float(np.log(np.mean(np.exp(matches["log_fitness"].to_numpy()))))

        seq_chars = list(HIS3)
        for aa, pos in zip(local_seq, POSITIONS):
            seq_chars[pos - 1] = aa
        mutated_sequence = "".join(seq_chars)

        records.append(
            {
                "mutant": infer_mutant(mutated_sequence, wt_sequence=HIS3),
                "mutated_sequence": mutated_sequence,
                "genotype": genotype,
                "genotype_sequence": local_seq,
                "fitness": fitness,
            }
        )

    out_df = pd.DataFrame.from_records(
        records,
        columns=["mutant", "mutated_sequence", "genotype", "genotype_sequence", "fitness"],
    )
    out_df["estimated_fitness"] = 0
    out_df = fill_nan_from_subset_of_mutations(out_df, "fitness", "his3")
    out_df.to_csv(output_path, index=False)


def process_mtagbfp() -> None:
    """
    Process raw mTagBFP2 data into standardized columns.

    Input:  data/raw_data/mtagbfp_raw_data.csv
    Output: data/processed_data/mtagbfp.csv

    Paper: https://www.nature.com/articles/s41467-019-12130-8#Abs1
    """
    GFP = "MNSLIKENMRMMVVMEGSVDGYQFKCTGEGDGNPYMGTQTMRIKVVEGGPLPFAFDILATSFLYGSKTFIKHTKGIPDFFKQSFPEGFTWERVTRYEDGGVFTVMQDTSLEDGCLVYHAKVTGVNFTSNGAVMQKKTKGWEPFTEMLYPADGGLRGYNQMALNVDGGSYLSASAETTYRSKKTVENFKMPGFHFVDYRLERLEESNNEMFVVQHEHAVAKFCDLPSKLGRK"
    POSITIONS = [20, 45, 63, 127, 143, 158, 168, 172, 174, 197, 206, 207, 231]

    input_path = DATA_DIR / "raw_data" / "mtagbfp_raw_data.csv"
    output_path = CURR_DIR / "mtagbfp.csv"

    df = pd.read_csv(input_path)

    # Drop metadata row and keep rows with a true local amino-acid sequence.
    df = df.loc[df["amino acid"].notna()].copy()
    df["genotype_sequence"] = df["amino acid"].astype(str).str.strip().str.strip('"')
    df = df.loc[df["genotype_sequence"].str.lower() != "sequence"].copy()
    wt_local = "".join(GFP[pos - 1] for pos in POSITIONS)

    def build_global_sequence(local_sequence: str) -> str:
        if len(local_sequence) != len(POSITIONS):
            raise ValueError(
                "Local sequence length mismatch: "
                f"expected={len(POSITIONS)}, got={len(local_sequence)}"
            )

        seq_chars = list(GFP)
        for aa, pos in zip(local_sequence, POSITIONS):
            seq_chars[pos - 1] = aa
        return "".join(seq_chars)

    def to_genotype(local_sequence: str) -> str:
        if len(local_sequence) != len(wt_local):
            raise ValueError(
                "Local sequence length mismatch for binary encoding: "
                f"expected={len(wt_local)}, got={len(local_sequence)}"
            )
        return "".join("0" if aa == wt_aa else "1" for aa, wt_aa in zip(local_sequence, wt_local))

    # Build the full mutated sequence and binary genotype encoding
    df["mutated_sequence"] = df["genotype_sequence"].apply(build_global_sequence)
    df["mutant"] = df["mutated_sequence"].apply(
        lambda seq: infer_mutant(seq, wt_sequence=GFP)
    )
    df["genotype"] = df["genotype_sequence"].apply(to_genotype)
    df["fitness"] = pd.to_numeric(df["brightness.2"], errors="coerce")

    out_df = df[
        [
            "mutant",
            "mutated_sequence",
            "genotype",
            "genotype_sequence",
            "fitness",
        ]
    ].copy()
    out_df["estimated_fitness"] = 0
    out_df.to_csv(output_path, index=False)


def process_beta() -> None:
    """
    Process raw beta-lactamase data into standardized columns.

    Input:  data/raw_data/weinreich.json
    Output: data/processed_data/beta.csv

    Paper: https://www.science.org/doi/10.1126/science.1123539
    """
    BETA = "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARAGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYEPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPMAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGEGGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    POSITIONS = ["NC", 42, 104, 182, 238]
    TARGET_AA = {42: "G", 104: "K", 182: "T", 238: "S"}

    input_path = DATA_DIR / "raw_data" / "weinreich.json"
    output_path = CURR_DIR / "beta.csv"

    with input_path.open("r") as infile:
        raw = json.load(infile)

    genotypes = raw.get("genotypes", [])
    phenotypes = raw.get("phenotypes", [])

    if len(genotypes) != len(phenotypes):
        raise ValueError(
            "Genotype/phenotype length mismatch: "
            f"genotypes={len(genotypes)}, phenotypes={len(phenotypes)}"
        )

    records = []
    for genotype, phenotype in zip(genotypes, phenotypes):
        genotype = str(genotype).strip()
        if len(genotype) != len(POSITIONS) or any(bit not in {"0", "1"} for bit in genotype):
            raise ValueError(f"Invalid genotype format: {genotype}")

        seq_chars = list(BETA)
        mutant_parts: list[str] = []

        if genotype[0] == "1":
            mutant_parts.append("NC")

        for bit, pos in zip(genotype[1:], POSITIONS[1:]):
            if bit == "1":
                wt_aa = seq_chars[pos - 1]
                mut_aa = TARGET_AA[pos]
                seq_chars[pos - 1] = mut_aa
                mutant_parts.append(f"{wt_aa}{pos}{mut_aa}")

        records.append(
            {
                "mutant": ";".join(mutant_parts) if mutant_parts else "WT",
                "mutated_sequence": "".join(seq_chars),
                "genotype": genotype,
                "genotype_sequence": genotype,
                "fitness": float(phenotype),
                "estimated_fitness": 0,
            }
        )

    pd.DataFrame.from_records(
        records,
        columns=["mutant", "mutated_sequence", "genotype", "genotype_sequence", "fitness", "estimated_fitness"],
    ).to_csv(output_path, index=False)

def process_cov2() -> None:
    """
    Process raw COV2 data into standardized columns.

    Input:  data/raw_data/cov2_raw_data.csv
    Output: data/processed_data/cov2.csv

    Paper: https://www.pnas.org/doi/10.1073/pnas.2413884122#executive-summary-abstract
    """
    COV2 = "EVQLVESGGGLIQPGGSLRLSCAASGFTVSSNYMSWVRQAPGKGLEWVSVIYSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDLGTGLFDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC"
    POSITIONS = [26, 27, 28, 31, 35, 50, 53, 56, 57, 58]
    TARGET_AA = {26: "E", 27: "X", 28: "I", 31: "R", 35: "T", 50: "L", 53: "P", 56: "T", 57: "A", 58: "F"} # X=V/L/I

    input_path = DATA_DIR / "raw_data" / "cov2_raw_data.csv"
    output_path = CURR_DIR / "cov2.csv"

    df = pd.read_csv(input_path)

    df["genotype"] = (
        df["seq"]
        .astype(str)
        .str.findall(r"[01]")
        .apply(lambda bits: "".join(bits))
    )

    expected_len = len(POSITIONS)
    if not (df["genotype"].str.len() == expected_len).all():
        bad_lengths = sorted(df.loc[df["genotype"].str.len() != expected_len, "genotype"].str.len().unique())
        raise ValueError(
            "COV2 genotype length mismatch: "
            f"expected={expected_len}, found={bad_lengths}"
        )

    def build_sequence_and_mutant(genotype: str) -> tuple[str, str]:
        seq_chars = list(COV2)
        mutant_parts: list[str] = []

        for bit, pos in zip(genotype, POSITIONS):
            if bit == "1":
                wt_aa = seq_chars[pos - 1]
                mut_aa = TARGET_AA[pos]
                seq_chars[pos - 1] = mut_aa
                mutant_parts.append(f"{wt_aa}{pos}{mut_aa}")

        mutant = ";".join(mutant_parts) if mutant_parts else "WT"
        return "".join(seq_chars), mutant

    seq_mut = df["genotype"].apply(build_sequence_and_mutant)
    df["mutated_sequence"] = seq_mut.apply(lambda x: x[0])
    df["mutant"] = seq_mut.apply(lambda x: x[1])
    def get_amino_acids_at_positions(seq: str) -> str:
        return "".join([seq[pos - 1] for pos in POSITIONS])
    df["genotype_sequence"] = df["mutated_sequence"].apply(get_amino_acids_at_positions)
    df["fitness"] = pd.to_numeric(df["F1_model"], errors="coerce")

    out_df = df[["mutant", "mutated_sequence", "genotype", "genotype_sequence", "fitness"]].copy()
    out_df = out_df.loc[out_df["fitness"].notna()].copy()
    out_df["estimated_fitness"] = 0
    out_df.to_csv(output_path, index=False)
    
def process_cr9114() -> None:
    """
    Process raw CR9114 data into standardized columns.

    Different fitness functions: h1_mean, h3_mean
    Input:  data/raw_data/cr9114_raw_data.csv
    Output: data/processed_data/cr9114_{fitness_function}.csv

    Paper: https://elifesciences.org/articles/71393#abstract
    """
    CR9114 = "QVQLVQSGAEVKKPGSSVKVSCKSSGGTFSSYAISWVRQAPGQGLDWMGGIIPIFGTANYAQKFQGRVTITADKSTSTAYMELSSLRSEDTAVYYCARHGNYYYYYGMDVWGQGTTVTVSS"
    POSITIONS = [29, 30, 31, 52, 57, 58, 59, 71, 74, 75, 76, 77, 84, 87, 95, 106]
    TARGET_AA = {29: "S", 30: "N", 31: "N", 52: "S", 57: "S", 58: "T", 59: "A", 71: "S", 74: "I", 75: "F", 76: "S", 77: "N", 84: "N", 87: "T", 95: "F", 106: "S"} 
    # In the paper they use IMGT unique numbering. Here, the positions are numbered sequentially.

    input_path = DATA_DIR / "raw_data" / "cr9114_raw_data.csv"
    df = pd.read_csv(input_path)

    pos_columns = [f"pos{i}" for i in range(1, len(POSITIONS) + 1)]
    df["genotype"] = (
        df[pos_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .agg("".join, axis=1)
    )

    expected_len = len(POSITIONS)
    if not (df["genotype"].str.len() == expected_len).all():
        bad_lengths = sorted(df.loc[df["genotype"].str.len() != expected_len, "genotype"].str.len().unique())
        raise ValueError(
            "CR9114 genotype length mismatch: "
            f"expected={expected_len}, found={bad_lengths}"
        )

    def build_sequence(genotype: str) -> str:
        seq_chars = list(CR9114)
        for bit, pos in zip(genotype, POSITIONS):
            if bit == "1":
                seq_chars[pos - 1] = TARGET_AA[pos]
        return "".join(seq_chars)

    df["mutated_sequence"] = df["genotype"].apply(build_sequence)
    df["mutant"] = df["mutated_sequence"].apply(lambda seq: infer_mutant(seq, wt_sequence=CR9114))
    df["genotype_sequence"] = df["mutated_sequence"].apply(
        lambda seq: "".join(seq[pos - 1] for pos in POSITIONS)
    )

    for fitness_column in ["h1_mean", "h3_mean"]:
        df_temp = df[["mutant", "mutated_sequence", "genotype", "genotype_sequence"]].copy()
        df_temp["fitness"] = pd.to_numeric(df[fitness_column], errors="coerce")
        df_temp["estimated_fitness"] = 0
        # Fill NaN fitness values from combinatorial subsets of mutations
        df_temp = fill_nan_from_subset_of_mutations(df_temp, "fitness", "cr9114")
        output_path = CURR_DIR / f"cr9114_{fitness_column}.csv"
        df_temp.to_csv(output_path, index=False)

def process_cr6261() -> None:
    """
    Process raw CR6261 data into standardized columns.

    Different fitness functions: h1_mean, h9_mean
    Input:  data/raw_data/cr6261_raw_data.csv
    Output: data/processed_data/cr6261_{fitness_function}.csv

    Paper: https://elifesciences.org/articles/71393#abstract
    """
    CR6261 = "QVQLVESGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGPEWMGGIIPIFGTANYAQKFQGRVTITADKSTSTAYMELSSLRSEDTAMYYCAKHMGYQLRETMDVWGQGTTVTVSS"
    POSITIONS = [28, 30, 58, 59, 62, 74, 75, 76, 77, 79, 104]
    TARGET_AA = {28: "P", 30: "R", 58: "T", 59: "K", 62: "P", 74: "D", 75: "F", 76: "A", 77: "G", 79: "V", 104: "V"}
    # In the paper they use IMGT unique numbering. Here, the positions are numbered sequentially.

    input_path = DATA_DIR / "raw_data" / "cr6261_raw_data.csv"
    df = pd.read_csv(input_path)

    pos_columns = [f"pos{i}" for i in range(1, len(POSITIONS) + 1)]
    df["genotype"] = (
        df[pos_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .agg("".join, axis=1)
    )

    expected_len = len(POSITIONS)
    if not (df["genotype"].str.len() == expected_len).all():
        bad_lengths = sorted(df.loc[df["genotype"].str.len() != expected_len, "genotype"].str.len().unique())
        raise ValueError(
            "CR6261 genotype length mismatch: "
            f"expected={expected_len}, found={bad_lengths}"
        )

    def build_sequence(genotype: str) -> str:
        seq_chars = list(CR6261)
        for bit, pos in zip(genotype, POSITIONS):
            if bit == "1":
                seq_chars[pos - 1] = TARGET_AA[pos]
        return "".join(seq_chars)

    # Expand to the complete 2^11 genotype grid so missing genotypes can be imputed.
    all_genotypes = ["".join(map(str, bits)) for bits in itertools.product((0, 1), repeat=expected_len)]
    observed = df[["genotype", "h1_mean", "h9_mean"]].copy()
    observed["h1_mean"] = pd.to_numeric(observed["h1_mean"], errors="coerce")
    observed["h9_mean"] = pd.to_numeric(observed["h9_mean"], errors="coerce")
    observed = observed.groupby("genotype", as_index=False)[["h1_mean", "h9_mean"]].mean()
    df_full = pd.DataFrame({"genotype": all_genotypes}).merge(observed, on="genotype", how="left")

    def build_sequence(genotype: str) -> str:
        seq_chars = list(CR6261)
        for bit, pos in zip(genotype, POSITIONS):
            if bit == "1":
                seq_chars[pos - 1] = TARGET_AA[pos]
        return "".join(seq_chars)

    df_full["mutated_sequence"] = df_full["genotype"].apply(build_sequence)
    df_full["mutant"] = df_full["mutated_sequence"].apply(lambda seq: infer_mutant(seq, wt_sequence=CR6261))
    df_full["genotype_sequence"] = df_full["mutated_sequence"].apply(
        lambda seq: "".join(seq[pos - 1] for pos in POSITIONS)
    )

    for fitness_column in ["h1_mean", "h9_mean"]:
        df_temp = df_full[["mutant", "mutated_sequence", "genotype", "genotype_sequence", fitness_column]].copy()
        df_temp = df_temp.rename(columns={fitness_column: "fitness"})
        df_temp["estimated_fitness"] = 0
        # Fill NaN fitness values from combinatorial subsets of mutations
        df_temp = fill_nan_from_subset_of_mutations(df_temp, "fitness", "cr6261")
        output_path = CURR_DIR / f"cr6261_{fitness_column}.csv"
        df_temp.to_csv(output_path, index=False)

def process_cov_rbd() -> None:
    """
    Process raw RBD combinatorial data into standardized columns.
    
    Input:  data/raw_data/cov_rbd_raw_data.tsv
    Output: data/processed_data/cov_rbd.csv

    Paper: https://www.nature.com/articles/s41467-022-34506-z
    """
    # 1. Load Wildtype Sequence from FASTA
    fasta_path = Path("../fasta/cov_rbd.fasta")
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found at {fasta_path}")
    
    record = SeqIO.read(fasta_path, "fasta")
    WT_SEQ = str(record.seq)
    
    POSITIONS = [339, 371, 373, 375, 417, 440, 446, 477, 478, 484, 493, 496, 498, 501, 505]
    TARGET_AA = {
        339: "D", 371: "L", 373: "P", 375: "F", 417: "N", 
        440: "K", 446: "S", 477: "N", 478: "K", 484: "A", 
        493: "R", 496: "S", 498: "R", 501: "Y", 505: "H"
    }

    input_path = Path("../raw_data/cov_rbd_raw_data.tsv")
    output_path = Path("../processed_data/cov_rbd.csv")

    # Read observed data
    df_raw = pd.read_csv(input_path, sep="\t")
    # Normalize genotype format
    df_raw["genotype"] = df_raw["geno"].astype(str).str.zfill(len(POSITIONS))
    df_raw["fitness"] = pd.to_numeric(df_raw["log10Kd"], errors="coerce")

    # 2. Build Full Combinatorial Grid (2^15)
    all_genotypes = ["".join(map(str, bits)) for bits in itertools.product((0, 1), repeat=len(POSITIONS))]
    df_full = pd.DataFrame({"genotype": all_genotypes})

    # Merge observed data into the full grid
    df_full = df_full.merge(df_raw[["genotype", "fitness"]], on="genotype", how="left")

    # 3. Build Sequences and Mutants for ALL genotypes
    def build_sequence_and_mutant(genotype: str) -> tuple[str, str, str]:
        seq_chars = list(WT_SEQ)
        mutant_parts: list[str] = []
        for bit, pos in zip(genotype, POSITIONS):
            if bit == "1":
                wt_aa = seq_chars[pos - 1]
                mut_aa = TARGET_AA.get(pos, "X")
                seq_chars[pos - 1] = mut_aa
                mutant_parts.append(f"{wt_aa}{pos}{mut_aa}")
        
        mutated_seq = "".join(seq_chars)
        mutant = ";".join(mutant_parts) if mutant_parts else "WT"
        genotype_seq = "".join([mutated_seq[pos - 1] for pos in POSITIONS])
        return mutated_seq, mutant, genotype_seq

    results = df_full["genotype"].apply(build_sequence_and_mutant)
    df_full["mutated_sequence"] = [r[0] for r in results]
    df_full["mutant"] = [r[1] for r in results]
    df_full["genotype_sequence"] = [r[2] for r in results]

    # 4. Impute Missing Fitness Values
    df_full["estimated_fitness"] = 0
    df_processed = fill_nan_from_subset_of_mutations(df_full, "fitness", "cov_rbd")

    # 5. Save output
    out_cols = ["mutant", "mutated_sequence", "genotype", "genotype_sequence", "fitness", "estimated_fitness"]
    df_processed[out_cols].to_csv(output_path, index=False)
    print(f"Processed and imputed data saved to {output_path}")

def fill_nan_from_subset_of_mutations(df: pd.DataFrame, fitness_col: str, dataset: str) -> pd.DataFrame:
    """
    Fill NaN fitness values by averaging the effects of all combinatorial mutations present in the genotype.
    If no subsets exist in the data, use 6 as fallback if the dataset is CR9114 and 7 as the fallback for CR6261. Else, use the empirical mean.    
    """
    df = df.copy()
    if "estimated_fitness" not in df.columns:
        df["estimated_fitness"] = 0

    genotypes_with_nan = df.loc[df[fitness_col].isna(), "genotype"].unique()

    genotype_to_fitness = dict(zip(df["genotype"], df[fitness_col]))
    
    for geno in genotypes_with_nan:
        # Find positions that are 1 (mutated)
        one_positions = [i for i, bit in enumerate(geno) if bit == "1"]
        
        # Generate all combinatorial subsets of these positions
        subset_fitness_values = []
        
        # Iterate through all combinations of the one_positions (from 0 to all)
        for r in range(len(one_positions) + 1):
            for combo in itertools.combinations(one_positions, r):
                # Build a genotype with only positions in combo set to 1
                subset_geno_list = ["0"] * len(geno)
                for pos in combo:
                    subset_geno_list[pos] = "1"
                subset_geno = "".join(subset_geno_list)
                
                if subset_geno in genotype_to_fitness:
                    fit_val = genotype_to_fitness[subset_geno]
                    if pd.notna(fit_val):
                        subset_fitness_values.append(fit_val)
        
        # Fill with mean if subsets exist
        if subset_fitness_values:
            fill_value = np.mean(subset_fitness_values)
        elif dataset == "cr9114":
            fill_value = 6.0
        elif dataset == "cr6261":
            fill_value = 7.0
        else:
            fill_value = np.mean(df[fitness_col].dropna())
        fill_mask = (df["genotype"] == geno) & (df[fitness_col].isna())
        df.loc[fill_mask, fitness_col] = fill_value
        df.loc[fill_mask, "estimated_fitness"] = 1
    
    df["estimated_fitness"] = df["estimated_fitness"].astype(int)
    return df

def main() -> None:
    process_his3()
    process_mtagbfp()
    process_beta()
    process_cov2()
    process_cr9114()
    process_cr6261()
    process_cov_rbd()


if __name__ == "__main__":
    main()

