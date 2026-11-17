"""Acquisition for the BO loop: EI of the objective times the probability of feasibility of each constraint
times the decision maker's acceptance probability, computed in log space:

    log alpha(x) = log EI_f(x | best_f)  +  sum_i log P(c_i(x) <= 0)  +  log pi(x)

    EI_f      ContinuousGP of the objective (BoTorch's LogExpectedImprovement: no underflow far from the incumbent)
    P(c_i<=0) Phi(-mean_i / sd_i) from the ContinuousGP of constraint i, as torch.special.log_ndtr (stable for
              very negative arguments, where log(Phi(z)) would give log(0) = -inf)
    pi(x)     the decision maker's 'preference function': E[Phi(u(x))], the probit of the latent utility u of a
              PreferenceGP fitted to pairwise comparisons (PreferenceGP.prob_torch)

Every model takes inputs in original units (the transforms live inside the models).  best_f is the best value
among feasible designs; when there is none yet, EI is undefined and the acquisition is log PoF alone (find a
feasible design first).  The objective is MAXIMISED.
"""

from .gp_utils import DTYPE, bounds_tensor

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import t_batch_mode_transform


class LogEIProduct(AcquisitionFunction):
    """log [EI(x) * prod_i PoF_i(x) * pi(x)], pi(x) = E[Phi(u(x))] or, with landmarks, the soft-Copeland score C(x).

    f_gp       fitted ContinuousGP of the objective
    c_gps      fitted ContinuousGPs, one per constraint c_i(x) <= 0 (may be empty)
    pref_gp    fitted PreferenceGP of the decision maker's comparisons (None: no preference factor)
    best_f     best observed value among feasible designs (the EI threshold); None = there is none yet
    copeland   a prepared copeland.SoftCopeland (pref_gp and fixed landmarks): the preference factor is then the
               soft-Copeland score C(x) (González et al. 2017, eq. 5) instead of pi(x); None: pi(x)
    """

    def __init__(self, f_gp, c_gps=(), pref_gp=None, best_f=None, copeland=None):
        super().__init__(model=f_gp.model)
        self.c_models, self.pref_gp, self.copeland = [c.model for c in c_gps], pref_gp, copeland
        self.log_ei = None if best_f is None else LogExpectedImprovement(f_gp.model, best_f=best_f)

    def log_pof(self, X):
        """sum_i log P(c_i(X) <= 0), shape (b,), for X of shape (b, 1, d)."""
        total = torch.zeros(X.shape[0], dtype=X.dtype)
        for c_model in self.c_models:
            posterior = c_model.posterior(X)
            mean, sd = posterior.mean.view(-1), posterior.variance.clamp_min(1e-12).sqrt().view(-1)
            total = total + torch.special.log_ndtr(-mean / sd)
        return total

    def log_pref(self, X):
        """log of the preference factor (pi or Copeland), shape (b,), for X of shape (b, 1, d); 0 without a preference model."""
        if self.pref_gp is None:
            return torch.zeros(X.shape[0], dtype=X.dtype)
        if self.copeland is None:
            score = self.pref_gp.prob_torch(X)
        else:
            score = self.copeland(X.reshape(-1, X.shape[-1]))
        return score.view(-1).clamp_min(1e-300).log()

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        log_probs = self.log_pof(X) + self.log_pref(X)
        return log_probs if self.log_ei is None else self.log_ei(X) + log_probs


def known_constraint_values(constraint, X):
    """c(X) as a numpy array (n,) for numpy designs X (n, d).  A known constraint is a cheap exact function of a torch tensor
    (n, d) -> (n,), feasible where <= 0, written with torch operations so that BoTorch can differentiate it (no GP).
    This is its numpy-in, numpy-out use (plots, feasibility of evaluated designs)."""
    with torch.no_grad():
        return constraint(torch.as_tensor(np.atleast_2d(X), dtype=DTYPE)).detach().numpy()


def known_feasible(X, known_constraints):
    """Bool array (n,) for the designs X (n, d): every known constraint c(X) <= 0 holds."""
    return np.all([known_constraint_values(constraint, X) <= 0 for constraint in known_constraints], axis=0)


def optimize(acq, bounds, d, num_restarts=10, raw_samples=512, seed=0, known_constraints=()):
    """Maximise acq over the box with BoTorch's multi-start L-BFGS-B: evaluate raw_samples random points, start
    L-BFGS-B from the num_restarts best of them, keep the best end point.  Returns (x as numpy (d,), acq value).

    known_constraints  cheap exact constraints (torch functions c(X) <= 0, see known_constraint_values): only designs where
                       all hold are considered.  They are passed to optimize_acqf as nonlinear_inequality_constraints, which
                       BoTorch solves with SLSQP; it needs starting points that already satisfy them, so the num_restarts
                       starts are the best of raw_samples random known-feasible designs."""
    box = bounds_tensor(bounds, d)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        if not known_constraints:
            x, value = optimize_acqf(acq, bounds=box, q=1, num_restarts=num_restarts, raw_samples=raw_samples)
            return x.detach().squeeze(0).numpy(), float(value)
        starts = _sample_known_feasible(box, known_constraints, raw_samples)
        if not acq(starts[:1].clone().requires_grad_(True).unsqueeze(1)).requires_grad:
            # no EI, no black-box constraint and no preference model yet (nothing feasible so far): the acquisition does not
            # depend on x, there is no gradient to follow and any known-feasible design is as good as another
            return starts[0].numpy(), float(acq(starts[:1].unsqueeze(1)))
        with torch.no_grad():
            starts = starts[torch.topk(acq(starts.unsqueeze(1)), min(num_restarts, len(starts))).indices]
        x, value = optimize_acqf(acq, bounds=box, q=1, num_restarts=len(starts),
                                 nonlinear_inequality_constraints=[(_for_botorch(c), True) for c in known_constraints],
                                 batch_initial_conditions=starts.unsqueeze(1))
    x = x.detach().squeeze(0)
    return _into_known_region(x, starts[0], known_constraints).numpy(), float(value)


def _for_botorch(constraint):
    """Our convention c(X) <= 0 on an (n, d) tensor -> BoTorch's: callable(x) >= 0 on one design x of shape (d,), a scalar."""
    return lambda x: -constraint(x.unsqueeze(0)).squeeze()


def _into_known_region(x, anchor, known_constraints):
    """SLSQP can end a hair outside the region (a violation of 1e-9 on its edge).  The recommendation and the incumbent check
    the constraints exactly, so move x towards a known-feasible anchor in tiny steps until it is feasible: it stays on the edge."""
    for t in [0.0] + [10.0 ** -k for k in range(10, -1, -1)]:
        candidate = x + t * (anchor - x)
        if known_feasible(candidate.numpy(), known_constraints)[0]:
            return candidate
    return anchor


def _sample_known_feasible(box, known_constraints, n, max_rounds=200):
    """n random designs in the box that satisfy the known constraints (rejection sampling), as a tensor (n, d)."""
    low, high = box[0], box[1]
    kept = torch.empty(0, box.shape[1], dtype=box.dtype)
    for _ in range(max_rounds):
        X = low + torch.rand(n, box.shape[1], dtype=box.dtype) * (high - low)
        kept = torch.cat([kept, X[torch.as_tensor(known_feasible(X.numpy(), known_constraints))]])
        if len(kept) >= n:
            return kept[:n]
    raise ValueError(f"only {len(kept)} known-feasible designs in {max_rounds * n} random ones: the region where all "
                     "known constraints hold is empty or tiny")
