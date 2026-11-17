"""ESM2 wildtype-marginal scoring (zero-shot LLR, esm-variants style) and the dataset metadata it needs.

One unmasked forward pass on the wildtype sequence gives a position x amino-acid log-likelihood-ratio
(LLR) matrix; a variant's score is the sum of the LLRs of its substitutions.

get_wt_LLR and the tiling helpers (for sequences longer than ESM2's 1022-residue window, e.g. the full
SARS-CoV-2 spike for cov_rbd) are adapted from esm-variants, esm_variants_utils.py
(MIT License, Copyright (c) 2023 Nadav Brandes and Vasilis Ntranos).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LOCAL_ESM_WEIGHTS = ROOT / "models" / "esm2_t33_650M_UR50D.pt"
AA_ORDER = ["K", "R", "H", "E", "D", "N", "Q", "T", "S", "C", "G", "A", "V", "L", "I", "M", "P", "Y", "F", "W"]
X_MUTANT_AAS = ("V", "L", "I")  # cov2 position 27: "X" = any of V, L, I

# Wildtype sequence, mutated positions (1-indexed into wt; "NC" = beta's non-coding site) and mutant residue.
DATASETS_METADATA = {
    "beta": {
        "wt": "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARAGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYEPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPMAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGEGGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW",
        "positions": ["NC", 42, 104, 182, 238],
        "targets": {42: "G", 104: "K", 182: "T", 238: "S"},
    },
    "cov2": {
        "wt": "EVQLVESGGGLIQPGGSLRLSCAASGFTVSSNYMSWVRQAPGKGLEWVSVIYSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDLGTGLFDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC",
        "positions": [26, 27, 28, 31, 35, 50, 53, 56, 57, 58],
        "targets": {26: "E", 27: "X", 28: "I", 31: "R", 35: "T", 50: "L", 53: "P", 56: "T", 57: "A", 58: "F"},
    },
    "cov_rbd": {
        "wt": "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSE",
        "positions": [339, 371, 373, 375, 417, 440, 446, 477, 478, 484, 493, 496, 498, 501, 505],
        "targets": {339: "D", 371: "L", 373: "P", 375: "F", 417: "N", 440: "K", 446: "S", 477: "N", 478: "K",
                    484: "A", 493: "R", 496: "S", 498: "R", 501: "Y", 505: "H"},
    },
    "cr6261": {
        "wt": "QVQLVESGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGPEWMGGIIPIFGTANYAQKFQGRVTITADKSTSTAYMELSSLRSEDTAMYYCAKHMGYQLRETMDVWGQGTTVTVSS",
        "positions": [28, 30, 58, 59, 62, 74, 75, 76, 77, 79, 104],
        "targets": {28: "P", 30: "R", 58: "T", 59: "K", 62: "P", 74: "D", 75: "F", 76: "A", 77: "G", 79: "V", 104: "V"},
    },
    "cr9114": {
        "wt": "QVQLVQSGAEVKKPGSSVKVSCKSSGGTFSSYAISWVRQAPGQGLDWMGGIIPIFGTANYAQKFQGRVTITADKSTSTAYMELSSLRSEDTAVYYCARHGNYYYYYGMDVWGQGTTVTVSS",
        "positions": [29, 30, 31, 52, 57, 58, 59, 71, 74, 75, 76, 77, 84, 87, 95, 106],
        "targets": {29: "S", 30: "N", 31: "N", 52: "S", 57: "S", 58: "T", 59: "A", 71: "S", 74: "I", 75: "F", 76: "S",
                    77: "N", 84: "N", 87: "T", 95: "F", 106: "S"},
    },
    "his3": {
        "wt": "MTEQKALVKRITNETKIQIAISLKGGPLAIEHSIFPEKEAEAVAEQATQSQVINVHTGIGFLDHMIHALAKHSGWSLIVECIGDLHIDDHHTTEDCGIALGQAFKEALGAVRGVKRFGSGFAPLDEALSRAVVDLSNRPYAVVELGLQREKVGDLSCEMIPHFLESFAEASRITLHVDCLRGKNDHHRSESAFKALAVAIREATSPNGTNDVPSTKGVLM",
        "positions": [145, 147, 148, 151, 152, 154, 164, 165, 168, 169, 170],
        "targets": {145: "I", 147: "L", 148: "Q", 151: "M", 152: "V", 154: "E", 164: "M", 165: "H", 168: "S", 169: "E",
                    170: "S"},
    },
    "mTagBFP2": {
        "wt": "MNSLIKENMRMMVVMEGSVDGYQFKCTGEGDGNPYMGTQTMRIKVVEGGPLPFAFDILATSFLYGSKTFIKHTKGIPDFFKQSFPEGFTWERVTRYEDGGVFTVMQDTSLEDGCLVYHAKVTGVNFTSNGAVMQKKTKGWEPFTEMLYPADGGLRGYNQMALNVDGGSYLSASAETTYRSKKTVENFKMPGFHFVDYRLERLEESNNEMFVVQHEHAVAKFCDLPSKLGRK",
        "positions": [20, 45, 63, 127, 143, 158, 168, 172, 174, 197, 206, 207, 231],
        "targets": {20: "N", 45: "A", 63: "M", 127: "P", 143: "S", 158: "A", 168: "G", 172: "C", 174: "L", 197: "R",
                    206: "D", 207: "K", 231: "R"},
    },
}


def metadata_key(stem: str) -> str:
    """Dataset stem -> DATASETS_METADATA key (several landscapes share one protein)."""
    if stem.startswith("cr6261_"):
        return "cr6261"
    if stem.startswith("cr9114_"):
        return "cr9114"
    if stem == "mtagbfp":
        return "mTagBFP2"
    return stem


def mutable_positions(meta: dict) -> list[int]:
    return [int(p) for p in meta["positions"] if p != "NC"]


# ---------------------------------------------------------------- model
def _resolve_device(device_id: int | None):
    import torch

    if device_id is not None:
        return torch.device(f"cuda:{device_id}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=4)
def load_esm_bundle(device_id: int | None = None):
    """(model, alphabet, batch_converter, device) for ESM2-650M from the local weights in models/."""
    import esm
    import torch

    if not LOCAL_ESM_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing local ESM weights: {LOCAL_ESM_WEIGHTS}")
    device = _resolve_device(device_id)
    print(f"  Loading ESM2 weights from {LOCAL_ESM_WEIGHTS} on {device}...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    checkpoint = torch.load(LOCAL_ESM_WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    return model.eval().to(device), alphabet, alphabet.get_batch_converter(), device


# ---------------------------------------------------------------- WT LLR matrix (esm-variants)
def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _chop(L, min_overlap=511, max_len=1022):
    return L[max_len - min_overlap:-max_len + min_overlap]


def _intervals(L, min_overlap=511, max_len=1022, parts=None):
    if parts is None:
        parts = []
    if len(L) <= max_len:
        if parts[-2][-1] - parts[-1][0] < min_overlap:
            return parts + [np.arange(L[int(len(L) / 2)] - int(max_len / 2), L[int(len(L) / 2)] + int(max_len / 2))]
        return parts
    parts += [L[:max_len], L[-max_len:]]
    return _intervals(_chop(L, min_overlap, max_len), min_overlap, max_len, parts=parts)


def _intervals_and_weights(seq_len, min_overlap=511, max_len=1022, s=16):
    """Overlapping windows covering the sequence and per-window sigmoid blending weights."""
    ints = _intervals(np.arange(seq_len), min_overlap=min_overlap, max_len=max_len)
    ints = [ints[i] for i in np.argsort([i[0] for i in ints])]
    a = int(np.round(min_overlap / 2))
    t = np.arange(max_len)
    f = np.ones(max_len)
    f[:a] = 1 / (1 + np.exp(-(t[:a] - a / 2) / s))
    f[max_len - a:] = 1 / (1 + np.exp((t[:a] - a / 2) / s))
    f0 = np.ones(max_len)
    f0[max_len - a:] = 1 / (1 + np.exp((t[:a] - a / 2) / s))
    fn = np.ones(max_len)
    fn[:a] = 1 / (1 + np.exp(-(t[:a] - a / 2) / s))
    filt = [f0] + [f for _ in ints[1:-1]] + [fn]
    M = np.zeros((len(ints), seq_len))
    for k, i in enumerate(ints):
        M[k, i] = filt[k]
    return ints, M, M / M.sum(0)


def _llr_frame(log_probs: np.ndarray, alphabet, seq: str) -> pd.DataFrame:
    """Rows = amino acids (AA_ORDER), columns = '{wt_aa} {pos}'; log p(aa) - log p(wt aa)."""
    logits = pd.DataFrame(log_probs, columns=alphabet.all_toks, index=list(seq)).T.iloc[4:24].loc[AA_ORDER]
    logits.columns = [j.split(".")[0] + " " + str(i + 1) for i, j in enumerate(logits.columns)]
    wt_norm = np.diag(logits.loc[[c.split(" ")[0] for c in logits.columns]])
    return logits - wt_norm


def compute_wt_llr_matrix(wt_sequence: str, model, alphabet, batch_converter, device) -> pd.DataFrame:
    """Position x amino-acid LLR matrix from the WT sequence (tiled windows if longer than 1022)."""
    import torch

    if len(wt_sequence) <= 1022:
        _, _, tokens = batch_converter([("WT", wt_sequence)])
        with torch.no_grad():
            out = torch.log_softmax(model(tokens.to(device), repr_layers=[33], return_contacts=False)["logits"], dim=-1)
        return _llr_frame(out[0].cpu().numpy()[1:-1, :], alphabet, wt_sequence)

    ints, _M, M_norm = _intervals_and_weights(len(wt_sequence), min_overlap=512, max_len=1022, s=20)
    windows = [(f"WT_{i}", "".join(np.array(list(wt_sequence))[idx])) for i, idx in enumerate(ints)]
    parts = []
    for batch in _chunks(windows, 20):
        _, _, tokens = batch_converter(batch)
        with torch.no_grad():
            out = torch.log_softmax(model(tokens.to(device), repr_layers=[33], return_contacts=False)["logits"], dim=-1)
        parts += [out[i].cpu().numpy()[1:-1, :] for i in range(out.shape[0])]
    full = np.zeros((len(wt_sequence), 33))
    for i, idx in enumerate(ints):
        part = np.zeros((len(wt_sequence), 33))
        part[idx] = parts[i]
        full += (part.T * M_norm[i, :]).T
    return _llr_frame(full, alphabet, wt_sequence)


# ---------------------------------------------------------------- variant scores
def llr_column_for_position(llr_df: pd.DataFrame, pos: int) -> str:
    for col in llr_df.columns:
        if int(col.split()[1]) == pos:
            return col
    raise KeyError(f"No LLR column for position {pos} in WT matrix")


def site_llr(llr_df: pd.DataFrame, col: str, target_aa: str) -> float:
    """LLR of one substitution (cov2 'X' = log-sum-exp over V, L, I)."""
    if target_aa == "X":
        llrs = np.array([float(llr_df.loc[aa, col]) for aa in X_MUTANT_AAS], dtype=np.float64)
        return float(np.log(np.exp(llrs).sum()))
    return float(llr_df.loc[target_aa, col])


def score_variants_from_wt_llr(llr_df: pd.DataFrame, positions: list[int], targets: dict[int, str],
                               genotypes: list[str]) -> np.ndarray:
    """Sum of per-site LLRs over the mutated positions of each genotype bitstring (bit j <-> positions[j])."""
    col_by_pos = {pos: llr_column_for_position(llr_df, pos) for pos in positions}
    site = {pos: site_llr(llr_df, col_by_pos[pos], targets[pos]) for pos in positions}
    return np.array([sum(site[pos] for j, pos in enumerate(positions) if g[j] == "1") for g in genotypes],
                    dtype=np.float64)
