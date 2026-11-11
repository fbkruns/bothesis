"""Helpers for gp.py: data conversion, bounds, kernels, comparison formatting, and a Bernoulli likelihood with
the logit link (GPyTorch only ships the probit one)."""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit._script")

import numpy as np
import torch
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood

DTYPE = torch.float64 


# data conversion
def to_numpy(a):
    """List, numpy array or torch tensor -> float numpy array."""
    return np.asarray(a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a, float)


def training_inputs(X, n):
    """Training inputs as an (n, d) float64 tensor; a 1-d array of length n is read as n one-dimensional points."""
    A = to_numpy(X)
    rows = A.shape[0] if A.ndim >= 1 else 1
    if rows != n:
        raise ValueError(f"X has {rows} rows but there are {n} observations (one row per observation expected)")
    return torch.as_tensor(A.reshape(n, -1), dtype=DTYPE)


def query_inputs(X, d):
    """Inputs for prediction as an (m, d) float64 tensor.

    A 1-d array is read as m points when d == 1 and as one point otherwise; a scalar is one 1-d point."""
    X = to_numpy(X)
    if X.ndim == 0:
        X = X.reshape(1, 1)
    elif X.ndim == 1:
        X = X.reshape(-1, 1) if d in (None, 1) else X.reshape(1, -1)
    if d is not None and X.shape[1] != d:
        raise ValueError(f"expected {d} input columns, got {X.shape[1]}")
    return torch.as_tensor(X, dtype=DTYPE)


def as_2d(A):
    """(m, d) numpy array; a 1-d array is read as m one-dimensional points."""
    A = to_numpy(A)
    return A.reshape(-1, 1) if A.ndim == 1 else A
####

# bounds for Botorch and plotting
def bounds_tensor(bounds, d):
    """Bounds given as (d, 2) [one (low, high) row per input] or (2, d) -> BoTorch's (2, d) tensor; None stays None."""
    if bounds is None:
        return None
    b = torch.as_tensor(np.asarray(bounds, float), dtype=DTYPE)
    if b.shape == (d, 2):
        b = b.T
    if b.shape != (2, d):
        raise ValueError(f"bounds must have shape (d, 2) or (2, d) with d = {d}, got {tuple(b.shape)}")
    return b

def plot_box(X, bounds, d):
    """(d, 2) box for plotting grids: the given bounds, or the data range padded by 5%."""
    if bounds is not None:
        return bounds_tensor(bounds, d).T.numpy()
    lo, hi = X.min(0), X.max(0)
    pad = 0.05 * np.where(hi > lo, hi - lo, 1.0)
    return np.stack([lo - pad, hi + pad], 1)

######

def make_kernel(kind, d, scaled):
    """kind: 'matern52' | 'rbf' | a GPyTorch Kernel.  ARD over all d inputs, BoTorch's dimension-scaled prior.

    scaled=True wraps it in a ScaleKernel: needed when there is no outcome standardisation (pairwise, binary),
    because the latent scale must then be learned."""
    if isinstance(kind, Kernel):
        return kind
    if kind not in ("matern52", "rbf"):
        raise ValueError(f"kernel must be 'matern52', 'rbf' or a gpytorch Kernel, got {kind!r}")
    base = get_covar_module_with_dim_scaled_prior(ard_num_dims=d, use_rbf_kernel=(kind == "rbf"))
    return ScaleKernel(base) if scaled else base

def format_comparisons(xa, xb, a_wins=None, decimals=10):
    """Pairs of designs -> (unique datapoints, comparisons) in PairwiseGP's format.

    xa, xb:   (m, d) arrays, row j holds the two designs of comparison j
    a_wins:   (m,) bools, True if xa[j] was preferred; None means xa[j] is always the winner
    returns:  X (n, d) unique designs, C (m, 2) int64 with C[j] = [index of winner, index of loser]
    Designs that appear in several comparisons get a single index (rows equal up to `decimals`)."""
    xa, xb = np.atleast_2d(np.asarray(xa, float)), np.atleast_2d(np.asarray(xb, float))
    if xa.shape != xb.shape:
        raise ValueError(f"xa and xb must have the same shape, got {xa.shape} and {xb.shape}")
    wins = np.ones(len(xa), bool) if a_wins is None else np.asarray(a_wins, bool).ravel()
    if len(wins) != len(xa):
        raise ValueError("a_wins must have one entry per comparison")
    X, inverse = np.unique(np.round(np.vstack([xa, xb]), decimals), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    ia, ib = inverse[: len(xa)], inverse[len(xa):]
    C = np.where(wins[:, None], np.stack([ia, ib], 1), np.stack([ib, ia], 1))
    return X, C.astype(np.int64)


class LogitBernoulliLikelihood(_OneDimensionalLikelihood):
    """P(y = 1 | g) = sigmoid(g).  GPyTorch only has the probit version (BernoulliLikelihood); the
    expectations needed by the ELBO and the predictive are computed by Gauss-Hermite quadrature."""

    def forward(self, function_samples, *args, **kwargs):
        return torch.distributions.Bernoulli(logits=function_samples)

    def marginal(self, function_dist, *args, **kwargs):
        probs = self.quadrature(torch.sigmoid, function_dist)
        return torch.distributions.Bernoulli(probs=probs)

