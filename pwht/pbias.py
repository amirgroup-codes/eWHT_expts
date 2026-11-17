"""p-biased Walsh-Hadamard transform (eWHT) and its inverse.

With ps = 0.5 at every site this is the ordinary WHT. Coefficients use the "expectation" normalisation
by default: the basis is orthonormal under the product measure mu_p, so sum(c^2) = Var_{mu_p}(y).

Bit convention: site 0 is the most significant bit of the genotype index (as in gen_utils.get_genotypes).
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from pwht.gen_utils import _validate_transform_inputs
except ImportError:
    from gen_utils import _validate_transform_inputs

NORMS = ("expectation", "sqrt", None)


def pfwht(y, ps=None, norm: str | None = "expectation"):
    """Forward p-biased WHT of a full landscape y (length 2^L). Accepts numpy arrays or torch tensors."""
    return compute_pfwht(y, ps=ps, add_bias=True, norm=norm)


def ipfwht(a: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """Inverse of pfwht(..., norm="expectation"). Returns a new array; the input is not modified."""
    a = np.array(a, dtype=float)
    h = 1
    while h < len(a):
        p = float(ps[-int(np.log2(h)) - 1])
        v = a.reshape(-1, 2, h)
        x, y = v[:, 0, :].copy(), v[:, 1, :]
        v[:, 0, :] = x + np.sqrt(p / (1 - p)) * y
        v[:, 1, :] = x - np.sqrt((1 - p) / p) * y
        h *= 2
    return a


def _norm_multiplier(N: int, norm: str | None) -> float:
    if norm == "expectation":
        return 1.0
    if norm == "sqrt":
        return float(np.sqrt(N))
    if norm is None:
        return float(N)
    raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")


def compute_pfwht(y, ps=None, add_bias: bool = False, norm: str | None = "expectation"):
    """add_bias=True is the forward p-biased transform; add_bias=False the unweighted butterfly."""
    if torch is not None and torch.is_tensor(y):
        return _compute_pfwht_torch(y, ps, add_bias=add_bias, norm=norm, device=y.device, return_numpy=False)
    yv, p, _, _ = _validate_transform_inputs(y, ps)
    if len(p) > 14 and torch is not None and torch.cuda.is_available():
        return _compute_pfwht_torch(yv, p, add_bias=add_bias, norm=norm, device="cuda", return_numpy=True)
    return _compute_pfwht_numpy(yv, p, add_bias=add_bias, norm=norm)


def _compute_pfwht_numpy(y: np.ndarray, ps: np.ndarray, add_bias: bool, norm: str | None) -> np.ndarray:
    y = y.copy()
    n, N = len(ps), len(y)
    if add_bias:
        bits = (np.arange(N)[:, None] & (1 << np.arange(n - 1, -1, -1))) > 0
        y *= np.prod(np.where(bits, ps, 1 - ps), axis=1) * _norm_multiplier(N, norm)
    h = 1
    while h < N:
        p = ps[n - 1 - int(np.log2(h))]
        alpha, beta = np.sqrt(p / (1 - p)), np.sqrt((1 - p) / p)
        y = y.reshape(-1, 2, h)
        x, a = y[:, 0, :].copy(), y[:, 1, :].copy()
        y[:, 0, :] = x + a
        y[:, 1, :] = alpha * x - beta * a
        h *= 2
    return y.flatten()


def _compute_pfwht_torch(y, ps, add_bias: bool, norm: str | None, device="cuda", return_numpy: bool = False):
    """Same as _compute_pfwht_numpy, on a torch device (used automatically for L > 14 when CUDA is available)."""
    if isinstance(y, np.ndarray):
        y_t = torch.as_tensor(y, dtype=torch.float64, device=device)
    else:
        y_t = y.to(dtype=torch.float64, device=device)
    if not return_numpy:
        y_t = y_t.clone()
    ps_t = torch.as_tensor(ps, dtype=torch.float64, device=device)
    n, N = len(ps_t), len(y_t)
    if add_bias:
        idxs = torch.arange(N, device=device)
        bits = (idxs[:, None] & (1 << torch.arange(n - 1, -1, -1, device=device))) > 0
        y_t *= torch.prod(torch.where(bits, ps_t, 1 - ps_t), dim=1) * _norm_multiplier(N, norm)
    h = 1
    while h < N:
        p = ps_t[n - 1 - int(np.log2(h))]
        alpha, beta = torch.sqrt(p / (1 - p)), torch.sqrt((1 - p) / p)
        y_t = y_t.view(-1, 2, h)
        x, a = y_t[:, 0, :].clone(), y_t[:, 1, :].clone()
        y_t[:, 0, :] = x + a
        y_t[:, 1, :] = alpha * x - beta * a
        y_t = y_t.view(-1)
        h *= 2
    return y_t.cpu().numpy() if return_numpy else y_t


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    for ps in (np.array([0.1, 0.4, 0.9]), np.full(3, 0.5)):
        y = rng.normal(size=8)
        err = np.abs(ipfwht(pfwht(y, ps=ps), ps) - y).max()
        print(f"ps={ps}: ipfwht(pfwht(y)) round-trip error {err:.2e}")
        assert err < 1e-12
    print("ok")
