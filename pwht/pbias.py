from __future__ import annotations
from itertools import product
import numpy as np
import pandas as pd
try:
	import torch
except ImportError: 
	torch = None
try:
    from pwht.gen_utils import _validate_transform_inputs
except ImportError: 
    from gen_utils import _validate_transform_inputs

"""
p-fWHT
"""
def pfwht(y: np.ndarray | torch.Tensor, ps: np.ndarray | torch.Tensor | None = None, norm: str | None = "expectation") -> np.ndarray | torch.Tensor:
	"""Forward p-biased WHT (add_bias=True)."""
	return compute_pfwht(y, ps=ps, add_bias=True, norm=norm)


def ipfwht(a, ps):
    '''
    Simple implementation of inverse p-biased FFT. Assumes input signal a has been forward
    transformed with the p-biased normalization (i.e., expectation). 
    '''
    h = 1
    while h < len(a):
        p = ps[-int(np.log2(h)) - 1]
        alpha = np.sqrt(p / (1 - p))
        beta = np.sqrt((1 - p) / p)
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + (alpha * y)
                a[j + h] = x - (beta * y)
        h *= 2
    return a

# def ipfwht(y: np.ndarray, ps: np.ndarray | None = None, norm: str | None = "expectation") -> np.ndarray:
# 	"""Inverse p-biased WHT (add_bias=False)."""
# 	raise NotImplementedError("ipfwht is not currently implemented; use np.linalg.solve with the p-bias matrix for exact inversion.")
# 	return compute_pfwht(y, ps=ps, add_bias=False, norm=norm)


def _norm_multiplier(N: int, norm: str | None) -> float:
    if norm in ["expectation", "stable"]:
        return 1.0
    if norm == "sqrt":
        return float(np.sqrt(N))
    if norm is None:
        return float(N)
    raise ValueError("norm must be one of {'expectation', 'sqrt', 'stable', None}.")


def compute_pfwht(
	y: np.ndarray,
	ps: np.ndarray | None = None,
	add_bias: bool = False,
	norm: str | None = "expectation",
) -> np.ndarray:
	"""
	If add_bias=False, this corresponds to the inverse FFT convention used in old_utils.
	If add_bias=True, this corresponds to the regular p-biased FFT convention used in old_utils.
	"""
	input_is_torch = torch is not None and torch.is_tensor(y)
	if input_is_torch:
		return _compute_pfwht_torch(y, ps, add_bias=add_bias, norm=norm, device=y.device, return_numpy=False)
	yv, p, _, _ = _validate_transform_inputs(y, ps)
	# Switch to torch implementation at larger sizes
	if len(p) > 14 and torch is not None and torch.cuda.is_available():
		return _compute_pfwht_torch(yv, p, add_bias=add_bias, norm=norm, device="cuda", return_numpy=True)
	return _compute_pfwht_numpy(yv, p, add_bias=add_bias, norm=norm)


def _compute_pfwht_numpy(y: np.ndarray, ps: np.ndarray, add_bias: bool = False, norm: str | None = "expectation") -> np.ndarray:
	y = y.copy()
	n = len(ps)
	N = len(y)
	p_norms = np.ones(n, dtype=np.float64)

	if add_bias:
		norm_scale = _norm_multiplier(N, norm)
		# Vectorized bias application
		idxs = np.arange(N)
		binaries = ((idxs[:, None] & (1 << np.arange(n-1, -1, -1))) > 0)
		bias_factors = np.prod(np.where(binaries, ps, 1 - ps), axis=1) * norm_scale
		y *= bias_factors

	h = 1
	while h < N:
		level_idx = n - 1 - int(np.log2(h))
		p = ps[level_idx]
		alpha, beta = np.sqrt(p / (1-p)), np.sqrt((1-p) / p)

		if norm == "stable":
			pn = np.sqrt(alpha**2 + beta**2)
			p_norms[level_idx] = pn
			alpha /= pn
			beta /= pn

		y = y.reshape(-1, 2, h)
		x, a = y[:, 0, :].copy(), y[:, 1, :].copy()
		y[:, 0, :] = x + a
		y[:, 1, :] = (alpha * x) - (beta * a)
		h *= 2

	res = y.flatten()
	return (res, p_norms) if norm == "stable" else res

def _compute_pfwht_torch(
	y: np.ndarray | torch.Tensor, 
	ps: np.ndarray | torch.Tensor, 
	add_bias: bool = False, 
	norm: str | None = "expectation", 
	device: str | torch.device = "cuda",
	return_numpy: bool = False
) -> np.ndarray | torch.Tensor:
	"""
    PyTorch version of pfwht on GPU.
    """
	# Ensure inputs are tensors on the correct device
	if isinstance(y, np.ndarray):
		y_t = torch.as_tensor(y, dtype=torch.float64, device=device)
	else:
		y_t = y.to(dtype=torch.float64, device=device)
	if not return_numpy:
		y_t = y_t.clone()

	ps_t = torch.as_tensor(ps, dtype=torch.float64, device=device)
	n = len(ps_t)
	N = len(y_t)
	p_norms_t = torch.ones(n, dtype=torch.float64, device=device)
    
	if add_bias:
		norm_scale = _norm_multiplier(N, norm)
		idxs = torch.arange(N, device=device)
		binaries = ((idxs[:, None] & (1 << torch.arange(n-1, -1, -1, device=device))) > 0)
		bias_factors = torch.prod(torch.where(binaries, ps_t, 1 - ps_t), dim=1) * norm_scale
		y_t *= bias_factors

	h = 1
	while h < N:
		level_idx = n - 1 - int(np.log2(h))
		p = ps_t[level_idx]
		alpha, beta = torch.sqrt(p / (1-p)), torch.sqrt((1-p) / p)

		if norm == "stable":
			pn = torch.sqrt(alpha**2 + beta**2)
			p_norms_t[level_idx] = pn
			alpha /= pn
			beta /= pn

		y_t = y_t.view(-1, 2, h)
		x, a = y_t[:, 0, :].clone(), y_t[:, 1, :].clone()
		y_t[:, 0, :] = x + a
		y_t[:, 1, :] = (alpha * x) - (beta * a)
		y_t = y_t.view(-1)
		h *= 2
		
	if norm == "stable":
		return (y_t.cpu().numpy(), p_norms_t.cpu().numpy()) if return_numpy else (y_t, p_norms_t)
	return y_t.cpu().numpy() if return_numpy else y_t



"""
p-WHT
"""
def pwht(y: np.ndarray, ps: np.ndarray | None = None, norm: str | None = "expectation") -> np.ndarray:
	"""Matrix-form forward p-WHT"""
	yv, p, _, _ = _validate_transform_inputs(y, ps)
	out = _big_p_bias_matrix(p, add_bias=True, norm=norm)
	if norm == "stable":
		H, p_norms = out
		return H @ yv, p_norms
	return out @ yv


def ipwht(y: np.ndarray, ps: np.ndarray | None = None, norm: str | None = "expectation") -> np.ndarray:
	"""Matrix-form inverse p-WHT"""
	yv, p, _, _ = _validate_transform_inputs(y, ps)
	if norm == "stable":
		H, p_norms = _big_p_bias_matrix(p, add_bias=False, norm=norm)
		return (H.T @ yv, p_norms)
	H = _big_p_bias_matrix(p, add_bias=False, norm=norm)
	return H.T @ yv


def _p_bias_matrix(p: float, stable: bool = False) -> np.ndarray:
	"""
	Gives the 2x2 p-biased matrix (for an individual AA site), where p is p(mutation).
	If stable=True, normalizes the functional (second) row by its Frobenius norm.
	"""
	q = 1.0 - p
	c0, c1 = np.sqrt(p / q), -np.sqrt(q / p)
	p_norm = 1.0
	if stable:
		p_norm = np.sqrt(c0**2 + c1**2)
		c0 /= p_norm
		c1 /= p_norm
	return np.array([[1.0, 1.0], [c0, c1]], dtype=float), p_norm


def _big_p_bias_matrix(ps: np.ndarray, add_bias: bool = False, norm: str | None = "expectation") -> np.ndarray:
	"""
	Gives the full N x N (where N = 2^n, n = len(ps)) p-biased matrix for the entire function.
    If add_bias is True, then for each column with index i (in 0, ..., 2^n-1), the column is
    scaled by prod_{j=0}^{n-1}(ps[j] if the jth bit of i is 1 else (1-ps[j])).
	"""
	is_stable = (norm == "stable")
	n = len(ps)
	p_norms = np.ones(n)

	# 1. Build the Kronecker Product
	H, pn = _p_bias_matrix(float(ps[0]), stable=is_stable)
	p_norms[0] = pn
	for i in range(1, n):
		mat, pn = _p_bias_matrix(float(ps[i]), stable=is_stable)
		p_norms[i] = pn
		H = np.kron(H, mat)

	# 2. Apply Expectation Weights (add_bias)
	if add_bias:
		n = len(ps)
		norm_scale = _norm_multiplier(1 << n, norm)
		for col in range(1 << n):
			bits = [(col >> (n - 1 - i)) & 1 for i in range(n)]
			P = np.prod([ps[i] if b else 1.0 - ps[i] for i, b in enumerate(bits)]) * norm_scale
			H[:, col] *= P
	return H, p_norms if is_stable else H



"""
Stable --> Expectation norm
"""
def get_stable_scaling_factors(p_norms):
    """Vectorized Kronecker scaling factors (Stable -> Expectation)"""
    L = len(p_norms)
    if torch.is_tensor(p_norms):
        scaling = torch.ones(1, dtype=torch.float64, device=p_norms.device)
        for i in range(L - 1, -1, -1):
            scaling = torch.cat([scaling, scaling * p_norms[i]])
        return scaling
    else:
        scaling = np.array([1.0], dtype=np.float64)
        for pn in reversed(p_norms):
            scaling = np.concatenate([scaling, scaling * pn])
        return scaling



"""
Vanilla WHT
"""
def walsh_hadamard_matrix(L: int = 13, normalize: bool = False) -> np.ndarray:
	"""Compute the WHT matrix for domain dimension L."""
	H1 = np.asarray([[1.0, 1.0], [1.0, -1.0]])
	H = np.asarray([1.0])
	for _ in range(L):
		H = np.kron(H, H1)
	if normalize:
		H = H / np.sqrt(2**L)
	return H


"""
p-bias utils
"""
def dec_to_bin(x, num_bits):
    assert x < 2**num_bits, "number of bits are not enough"
    u = bin(x)[2:].zfill(num_bits)
    u = list(u)
    u = [int(i) for i in u]
    return np.array(u)









def run_stability_tests():
    L = 3
    ps = np.array([0.1, 0.4, 0.9], dtype=float)
    landscape = np.random.default_rng(42).normal(size=2**L)

    print("--- 1. Testing Fast Transform (pfwht) ---")
    # Expectation (Single return)
    Y_exp = pfwht(landscape, ps=ps, norm="expectation")
    assert isinstance(Y_exp, np.ndarray), "Expectation should return ndarray"

    landscape_restored = ipfwht(Y_exp.copy(), ps)
    err_ipfwht = np.max(np.abs(landscape - landscape_restored))
    print(f"ipfwht roundtrip error: {err_ipfwht:.6g}")
    assert err_ipfwht < 1e-12
    
    # Stable (Tuple return)
    Y_stable, p_norms = pfwht(landscape, ps=ps, norm="stable")
    assert isinstance(Y_stable, np.ndarray) and isinstance(p_norms, np.ndarray), "Stable should return tuple"
    
    # Verify Restoration
    scaling = get_stable_scaling_factors(p_norms)
    Y_restored = Y_stable * scaling
    err_pfwht = np.max(np.abs(Y_exp - Y_restored))
    print(f"pfwht restoration error: {err_pfwht:.6g}")
    assert err_pfwht < 1e-12

    print("\n--- 2. Testing Matrix Transform (pwht) ---")
    Y_m_exp = pwht(landscape, ps=ps, norm="expectation")
    Y_m_stable, p_m_norms = pwht(landscape, ps=ps, norm="stable")
    
    scaling_m = get_stable_scaling_factors(p_m_norms)
    Y_m_restored = Y_m_stable * scaling_m
    err_pwht = np.max(np.abs(Y_m_exp - Y_m_restored))
    print(f"pwht restoration error: {err_pwht:.6g}")
    assert err_pwht < 1e-12

    if torch is not None:
        print("\n--- 3. Testing Torch Implementation ---")
        l_t = torch.as_tensor(landscape, device="cuda" if torch.cuda.is_available() else "cpu")
        ps_t = torch.as_tensor(ps, device=l_t.device)
        
        # Expectation
        Y_t_exp = pfwht(l_t, ps=ps_t, norm="expectation")
        
        # Stable
        Y_t_stable, p_t_norms = pfwht(l_t, ps=ps_t, norm="stable")
        
        # Restoration
        s_t = get_stable_scaling_factors(p_t_norms)
        Y_t_restored = Y_t_stable * s_t
        err_torch = torch.max(torch.abs(Y_t_exp - Y_t_restored)).item()
        print(f"Torch restoration error: {err_torch:.6g}")
        assert err_torch < 1e-12

    print("\nAll stability restoration tests passed.")

if __name__ == "__main__":
    run_stability_tests()
