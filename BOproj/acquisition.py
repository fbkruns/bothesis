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

from .gp_utils import bounds_tensor

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import t_batch_mode_transform


class LogEIProduct(AcquisitionFunction):
    """log [EI(x) * prod_i PoF_i(x) * pi(x)].

    f_gp       fitted ContinuousGP of the objective
    c_gps      fitted ContinuousGPs, one per constraint c_i(x) <= 0 (may be empty)
    pref_gp    fitted PreferenceGP of the decision maker's comparisons (None: no preference factor)
    best_f     best observed value among feasible designs (the EI threshold); None = there is none yet
    """

    def __init__(self, f_gp, c_gps=(), pref_gp=None, best_f=None):
        super().__init__(model=f_gp.model)
        self.c_models, self.pref_gp = [c.model for c in c_gps], pref_gp
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
        """log pi(X), shape (b,), for X of shape (b, 1, d); 0 without a preference model."""
        if self.pref_gp is None:
            return torch.zeros(X.shape[0], dtype=X.dtype)
        return self.pref_gp.prob_torch(X).view(-1).clamp_min(1e-300).log()

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        log_probs = self.log_pof(X) + self.log_pref(X)
        return log_probs if self.log_ei is None else self.log_ei(X) + log_probs


def optimize(acq, bounds, d, num_restarts=10, raw_samples=512, seed=0):
    """Maximise acq over the box with BoTorch's multi-start L-BFGS-B: evaluate raw_samples random points, start
    L-BFGS-B from the num_restarts best of them, keep the best end point.  Returns (x as numpy (d,), acq value)."""
    box = bounds_tensor(bounds, d)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        x, value = optimize_acqf(acq, bounds=box, q=1, num_restarts=num_restarts, raw_samples=raw_samples)
    return x.detach().squeeze(0).numpy(), float(value)
