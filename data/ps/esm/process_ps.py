from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

# Hyperparameters
LOWER_P = 0.1
UPPER_P = 0.9

# Protein metadata
DATASETS_METADATA = {
    "beta": {
        "wt": "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARAGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYEPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPMAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGEGGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW",
        "positions": ["NC", 42, 104, 182, 238],
        "targets": {42: "G", 104: "K", 182: "T", 238: "S"}
    },
    "cov2": {
        "wt": "EVQLVESGGGLIQPGGSLRLSCAASGFTVSSNYMSWVRQAPGKGLEWVSVIYSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDLGTGLFDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC",
        "positions": [26, 27, 28, 31, 35, 50, 53, 56, 57, 58],
        "targets": {26: "E", 27: "X", 28: "I", 31: "R", 35: "T", 50: "L", 53: "P", 56: "T", 57: "A", 58: "F"} # X=V/L/I
    },
    "cov_rbd": {
        "wt": "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSE",  # Populated dynamically via dataframe row parsing
        "positions": [339, 371, 373, 375, 417, 440, 446, 477, 478, 484, 493, 496, 498, 501, 505],
        "targets": {339: "D", 371: "L", 373: "P", 375: "F", 417: "N", 440: "K", 446: "S", 477: "N", 478: "K", 484: "A", 493: "R", 496: "S", 498: "R", 501: "Y", 505: "H"}
    },
    "cr6261": {
        "wt": "QVQLVESGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGPEWMGGIIPIFGTANYAQKFQGRVTITADKSTSTAYMELSSLRSEDTAMYYCAKHMGYQLRETMDVWGQGTTVTVSS",
        "positions": [28, 30, 58, 59, 62, 74, 75, 76, 77, 79, 104],
        "targets": {28: "P", 30: "R", 58: "T", 59: "K", 62: "P", 74: "D", 75: "F", 76: "A", 77: "G", 79: "V", 104: "V"}
    },
    "cr9114": {
        "wt": "QVQLVQSGAEVKKPGSSVKVSCKSSGGTFSSYAISWVRQAPGQGLDWMGGIIPIFGTANYAQKFQGRVTITADKSTSTAYMELSSLRSEDTAVYYCARHGNYYYYYGMDVWGQGTTVTVSS",
        "positions": [29, 30, 31, 52, 57, 58, 59, 71, 74, 75, 76, 77, 84, 87, 95, 106],
        "targets": {29: "S", 30: "N", 31: "N", 52: "S", 57: "S", 58: "T", 59: "A", 71: "S", 74: "I", 75: "F", 76: "S", 77: "N", 84: "N", 87: "T", 95: "F", 106: "S"}
    },
    "his3": {
        "wt": "MTEQKALVKRITNETKIQIAISLKGGPLAIEHSIFPEKEAEAVAEQATQSQVINVHTGIGFLDHMIHALAKHSGWSLIVECIGDLHIDDHHTTEDCGIALGQAFKEALGAVRGVKRFGSGFAPLDEALSRAVVDLSNRPYAVVELGLQREKVGDLSCEMIPHFLESFAEASRITLHVDCLRGKNDHHRSESAFKALAVAIREATSPNGTNDVPSTKGVLM",
        "positions": [145, 147, 148, 151, 152, 154, 164, 165, 168, 169, 170],
        "starting_targets": {145: "L", 147: "F", 148: "R", 151: "K", 152: "I", 154: "Q", 164: "L", 165: "D", 168: "A", 169: "G", 170: "G"},
        "targets": {145: "I", 147: "L", 148: "Q", 151: "M", 152: "V", 154: "E", 164: "M", 165: "H", 168: "S", 169: "E", 170: "S"}
    },
    "mTagBFP2": {
        "wt": "MNSLIKENMRMMVVMEGSVDGYQFKCTGEGDGNPYMGTQTMRIKVVEGGPLPFAFDILATSFLYGSKTFIKHTKGIPDFFKQSFPEGFTWERVTRYEDGGVFTVMQDTSLEDGCLVYHAKVTGVNFTSNGAVMQKKTKGWEPFTEMLYPADGGLRGYNQMALNVDGGSYLSASAETTYRSKKTVENFKMPGFHFVDYRLERLEESNNEMFVVQHEHAVAKFCDLPSKLGRK",
        "positions": [20, 45, 63, 127, 143, 158, 168, 172, 174, 197, 206, 207, 231],
        "targets": {20: "N", 45: "A", 63: "M", 127: "P", 143: "S", 158: "A", 168: "G", 172: "C", 174: "L", 197: "R", 206: "D", 207: "K", 231: "R"}
    }
}
# cov2 position 27 uses X to denote any of V, L, or I.
X_MUTANT_AAS = ("V", "L", "I")

CURR_DIR = Path(__file__).resolve().parent


def _aa_token_id(tokenizer: AutoTokenizer, aa: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(aa.upper())
    if token_id == tokenizer.unk_token_id:
        raise ValueError(f"Unknown amino acid token '{aa}' in ESM vocabulary.")
    return token_id


def _clip_p(prob_mut: float) -> float:
    return min(max(prob_mut, LOWER_P), UPPER_P)


def _mutation_prob_from_logits(
    logits: torch.Tensor,
    tokenizer: AutoTokenizer,
    wt_aa: str,
    mut_aa: str,
) -> tuple[float, float, float]:
    """
    Softmax logits at a masked site, then renormalize {wt, mutant} to a binary distribution.
    Returns (p_wt_renorm, p_mut_renorm, p_mut_renorm) — last value is p(mutant) for ps.
    For X, mutant mass is the sum over V, L, and I.
    """
    probs = torch.softmax(logits, dim=-1)
    wt_prob = probs[_aa_token_id(tokenizer, wt_aa)].item()

    if mut_aa.upper() == "X":
        mut_prob = sum(probs[_aa_token_id(tokenizer, aa)].item() for aa in X_MUTANT_AAS)
    else:
        mut_prob = probs[_aa_token_id(tokenizer, mut_aa)].item()

    total = wt_prob + mut_prob
    if total == 0.0:
        return 0.5, 0.5, 0.5
    p_wt = wt_prob / total
    p_mut = mut_prob / total
    return p_wt, p_mut, p_mut


@torch.no_grad()
def get_site_probabilities(
    wt_sequence: str,
    positions: list[int | str],
    targets: dict[int, str],
    model: EsmForMaskedLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    starting_targets: dict[int, str] | None = None,
) -> np.ndarray:
    """Mask each site, read ESM logits, renormalize wt vs mutant, return p(mutant)."""
    tokens = tokenizer(wt_sequence, return_tensors="pt")
    base_input_ids = tokens["input_ids"].to(device)

    ps: list[float] = []
    for pos in positions:
        if pos == "NC":
            ps.append(0.5)
            continue

        pos = int(pos)
        if starting_targets is not None:
            wt_aa = starting_targets[pos]
        else:
            wt_aa = wt_sequence[pos - 1]
        mut_aa = targets[pos]

        masked_ids = base_input_ids.clone()
        masked_ids[0, pos] = tokenizer.mask_token_id
        pos_logits = model(masked_ids).logits[0, pos]

        _, _, prob_mut = _mutation_prob_from_logits(pos_logits, tokenizer, wt_aa, mut_aa)
        ps.append(_clip_p(prob_mut))

    return np.array(ps, dtype=np.float32)


def process_esm_to_ps(model_name: str = "facebook/esm2_t33_650M_UR50D") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading transformer model '{model_name}' onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    for name, meta in DATASETS_METADATA.items():
        print(f"Computing ps for {name}...")
        ps = get_site_probabilities(
            meta["wt"],
            meta["positions"],
            meta["targets"],
            model,
            tokenizer,
            device,
            starting_targets=meta.get("starting_targets"),
        )
        out_path = CURR_DIR / f"{name}.npy"
        np.save(out_path, ps)


def main() -> None:
    process_esm_to_ps()


if __name__ == "__main__":
    main()
