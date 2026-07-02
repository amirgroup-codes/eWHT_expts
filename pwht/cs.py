from itertools import combinations
import numpy as np

def p_walsh_hadamard_from_seqs(bin_seqs, ps, norm="expectation"):
    """
    Compute p-biased Walsh-Hadamard design matrix for compressed sensing experiments.
    
    Parameters:
    - bin_seqs: (N, L) array of 0/1 (0=Wildtype, 1=Mutation)
    - ps: (L,) array of mutation probabilities
    - norm: "expectation" or "sqrt" (Default: "expectation")
    """
    bin_seqs = np.asarray(bin_seqs, dtype=np.int8)
    ps = np.asarray(ps, dtype=np.float64)
    N, L = bin_seqs.shape
    M_val = 1 << L

    # 1. Basis factors mapping: 0 -> sqrt(p/q), 1 -> -sqrt(q/p)
    coeff0 = np.sqrt(ps / (1.0 - ps)) 
    coeff1 = -np.sqrt((1.0 - ps) / ps)

    # 2. Iterative Kronecker Construction
    if norm == "stable":
        p_norms = np.sqrt(coeff0**2 + coeff1**2)
        coeff0 = coeff0 / p_norms
        coeff1 = coeff1 / p_norms

    X_lex = np.empty((N, M_val), dtype=np.float64)
    X_lex[:, 0] = 1.0
    span = 1
    for j in range(L):
        # Column vector of coefficients for site j
        c = np.where(bin_seqs[:, [j]] == 1, coeff1[j], coeff0[j])
        # Multiply all existing interaction columns by the new site's basis
        X_lex[:, span : 2 * span] = X_lex[:, :span] * c
        span <<= 1

    # 3. Reorder Columns by Degree
    # We map the order of combinations directly to binary indices
    # e.g., (0, 1) -> 2^0 + 2^1 = 3
    degree_idx = [
        sum(1 << i for i in combo) 
        for r in range(L + 1) 
        for combo in combinations(range(L), r)
    ]
    X = X_lex[:, degree_idx]

    # 4. Normalization Logic
    if norm == "expectation":
        return X
    elif norm == "sqrt":
        return X / np.sqrt(M_val)
    elif norm == "stable":
        return X, p_norms
    else:        
        raise ValueError(f"norm must be one of {{'expectation', 'sqrt', 'stable'}}, got {norm}")


def walsh_hadamard_from_seqs(bin_seqs, normalize=False):
    """
    Compute Walsh-Hadamard design matrix for compressed sensing experiments.
    seqs: (n_samples, L) array of 0/1
	Returns: (n_samples, 2**L) design matrix
    """
    S = np.asarray(bin_seqs, dtype=np.int8)  # N x L
    N, L = S.shape
    M_val = 2**L
    
    # 1. Generate the Mask Matrix in the same order as powerset()
    # This ensures the columns match your original implementation
    masks = []
    for r in range(L + 1):
        for combo in combinations(range(L), r):
            m = np.zeros(L, dtype=np.int8)
            m[list(combo)] = 1
            masks.append(m)
    
    MaskMatrix = np.array(masks) # 2^L x L
    
    # 2. Compute binary parity: (S @ MaskMatrix.T) % 2
    parity = (S @ MaskMatrix.T) % 2
    
    # 3. Map parity 0 -> 1 and parity 1 -> -1
    X = (1 - 2 * parity).astype(float)
    
    if normalize:
        X = X / np.sqrt(M_val)

    return X