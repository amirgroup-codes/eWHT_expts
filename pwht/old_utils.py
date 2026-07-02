import numpy as np
import torch
import itertools

def get_ps(wt, seqs):
    '''
    Get the empirical probabilities vector (ps) from a set of sequences.
    wt: wild-type sequence
    seqs: list of sequences (should still include wt!)
    '''
    mut_mask = np.array([[aa != wt[i] for i, aa in enumerate(seq)]
                     for seq in seqs], dtype=bool)   # shape (N, 237)

    p_mut = mut_mask.mean(axis=0)   # length-237 vector of empirical P(mutated)
    p_mut = np.where(p_mut == 0, 1e-8, p_mut)
    return p_mut

def get_all_interactions(L):
    """Get all possible interaction sets for length L"""
    all_U = []
    for r in range(L + 1):
        for U in itertools.combinations(range(L), r):
            all_U.append(list(U))
    return all_U

def convert_01_bin_seqs(bin_seqs):
    """Convert 0/1 sequences to -1/1 for Walsh-Hadamard"""
    return 2 * bin_seqs - 1

def walsh_hadamard_from_seqs(bin_seqs, norm='sqrt'):
    """
    Returns an N x 2^L array containing the Walsh-Hadamard encodings of
    a given list of N binary ({0,1}) sequences.
    """
    bin_seqs_ = convert_01_bin_seqs(np.array(bin_seqs))
    L = len(bin_seqs_[0])
    all_U = get_all_interactions(L)
    M = 2**L
    N = len(bin_seqs)
    X = np.zeros((N, len(all_U)))
    for i, U in enumerate(all_U):
        if len(U) == 0:
            X[:, i] = 1
        else:
            X[:, i] = np.prod(bin_seqs_[:, U], axis=1)
    if norm == 'sqrt':
        X = X / np.sqrt(M)
    elif norm == 'full':
        X = X / M
    elif norm == 'none':
        pass
    else:
        raise ValueError(f"Invalid norm: {norm}")
    return X

def p_walsh_hadamard_from_seqs(bin_seqs, ps, add_bias=False):
    """
    Returns an N x 2^L array containing the p-biased Walsh-Hadamard encodings of
    a given list of N binary ({0,1}) sequences.
    """
    bin_seqs = np.array(bin_seqs)
    ps = np.array(ps)
    L = len(bin_seqs[0])
    all_U = get_all_interactions(L)
    N = len(bin_seqs)
    X = np.zeros((N, len(all_U)))
    X[:, 0] = 1
    all_U = all_U[1:]
    
    # Precompute coefficients for each position
    coeff_0 = np.sqrt(ps / (1 - ps))
    coeff_1 = -np.sqrt((1 - ps) / ps)
    
    for ind, seq in enumerate(bin_seqs):
        for i, U in enumerate(all_U, start=1):
            prod = 1
            for j in U:
                if seq[j] == 1:
                    prod *= coeff_1[j]
                else:
                    prod *= coeff_0[j]
            X[ind, i] = prod
    # Vectorized Bernoulli measure
    if add_bias:
        probs = np.where(bin_seqs == 0, ps, 1 - ps)
        weights = np.prod(probs, axis=1)
        X *= weights[:, None]
    
    return X

def dec_to_bin(x, num_bits):
    assert x < 2**num_bits, "number of bits are not enough"
    u = bin(x)[2:].zfill(num_bits)
    u = list(u)
    u = [int(i) for i in u]
    return np.array(u)

def p_bias_matrix(p):
    '''
    Gives the 2x2 p-biased matrix (for an individual AA site), where p is p(mutation).
    '''
    q = 1.0 - p
    return np.array([
        [1.0,               1.0              ],
        [np.sqrt(p / q),  -np.sqrt(q / p)]
    ])

def big_p_bias_matrix(ps, add_bias=False):
    '''
    Gives the full N x N (where N = 2^n, n = len(ps)) p-biased matrix for the entire function.
    If add_bias is True, then for each column with index i (in 0, ..., 2^n-1), the column is
    scaled by prod_{j=0}^{n-1}(ps[j] if the jth bit of i is 1 else (1-ps[j])).
    '''
    H = p_bias_matrix(ps[0])
    for pi in ps[1:]:
        H = np.kron(H, p_bias_matrix(pi))

    if add_bias:                           # multiply each column by P(x)
        n = len(ps)
        for col in range(1 << n):
            bits = [(col >> (n-1-i)) & 1 for i in range(n)]   # big-endian
            P    = np.prod([ps[i] if b else 1-ps[i]
                            for i, b in enumerate(bits)])
            H[:, col] *= P
    return H



def pfwht(a, ps, add_bias=False):
    if add_bias:
        n = len(ps)
        N = len(a)
        for idx in range(N):
            binary = dec_to_bin(idx, n)
            bias_factor = np.prod([ps[j] if bit == 1 else (1 - ps[j]) for j, bit in enumerate(binary)])
            a[idx] *= bias_factor
    h = 1
    while h < len(a):
        p = ps[-int(np.log2(h)) - 1]
        alpha = np.sqrt(p / (1 - p))
        beta = np.sqrt((1 - p) / p)
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = (alpha * x) - (beta * y)
        h *= 2
    return a


def pfwht_torch(a, ps, add_bias=False, device='cuda'):
    """
    PyTorch version of pfwht on GPU.
    """
    a = torch.as_tensor(a, dtype=torch.float64, device=device).clone()
    ps = torch.as_tensor(ps, dtype=torch.float64, device=device)
    if add_bias:
        n = len(ps)
        N = len(a)
        idxs = torch.arange(N, device=device)
        binaries = ((idxs[:, None] & (1 << torch.arange(n-1, -1, -1, device=device))) > 0).long()
        bias_factors = torch.prod(torch.where(binaries == 1, ps, 1 - ps), dim=1)
        a = a * bias_factors
    h = 1
    n = len(ps)
    while h < len(a):
        p = ps[-int(np.log2(h)) - 1]
        alpha = torch.sqrt(p / (1 - p))
        beta = torch.sqrt((1 - p) / p)
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j].clone()
                y = a[j + h].clone()
                a[j] = x + y
                a[j + h] = (alpha * x) - (beta * y)
        h *= 2
    return a

def bernoulli_weights(x_bits, p, use_log=True, eps=1e-8):
    """
    x_bits : (B, n) 0/1   – batch of mutation masks
    p      : (n,)          – tensor of per-site mutation probs
    returns: (B,) weights  w(x)=∏ p_i^{x_i}(1-p_i)^{1-x_i}
    """
    if use_log:
        p  = torch.clamp(p, eps, 1.0 - eps)
        logw = x_bits * torch.log(p) + (1 - x_bits) * torch.log1p(-p)
        return torch.exp(logw.sum(dim=1))
    else:
        return (p**x_bits * (1.0 - p)**(1.0 - x_bits)).prod(dim=1)

def bernoulli_weights_numpy(x_bits, p, use_log=True, eps=1e-8):
    """
    Numpy version of bernoulli_weights function.
    x_bits : (B, n) 0/1   – batch of mutation masks
    p      : (n,)          – array of per-site mutation probs
    returns: (B,) weights  w(x)=∏ p_i^{x_i}(1-p_i)^{1-x_i}
    """
    if use_log:
        p = np.clip(p, eps, 1.0 - eps)
        logw = x_bits * np.log(p) + (1 - x_bits) * np.log1p(-p)
        return np.exp(logw.sum(axis=1))
    else:
        return (p**x_bits * (1.0 - p)**(1.0 - x_bits)).prod(axis=1)

class BernoulliMSE(torch.nn.Module):
    def __init__(self, p, reduction="mean", log_stable=True):
        super().__init__()
        self.register_buffer("p", torch.as_tensor(p, dtype=torch.float32))
        self.reduction = reduction
        self.log_stable = log_stable

    def forward(self, pred, target, x_bits):
        w   = bernoulli_weights(x_bits.float(), self.p,
                                use_log=self.log_stable)            # (B,)
        se  = (pred - target).pow(2).view(w.size(0), -1).mean(1)    # (B,)
        loss = w * se
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss            # 'none'


def inverse_pwht(coeffs, ps):
    """True inverse of regular pWHT (add_bias=True) via linear solve."""
    H_fwd = big_p_bias_matrix(ps, add_bias=True)
    return np.linalg.solve(H_fwd, np.asarray(coeffs, dtype=float))


def run_self_tests_pwht_inverse_roundtrip():
    """Sanity test: y -> pWHT -> inverse pWHT should recover y."""
    rng = np.random.default_rng(0)
    L = 5
    N = 2**L
    y = rng.normal(size=N)
    ps = np.array([0.2, 0.4, 0.6, 0.7, 0.3], dtype=float)

    coeffs = pfwht(y.copy(), ps, add_bias=True)
    y_recovered = inverse_pwht(coeffs, ps)

    assert np.allclose(y_recovered, y, atol=1e-10)
    return {
        "max_abs_roundtrip_error": float(np.max(np.abs(y_recovered - y))),
        "L": L,
    }

if __name__ == "__main__":
    test_results = run_self_tests_pwht_inverse_roundtrip()
    print("Self-test results:", test_results)