"""Acquisition for the BO loop: the expected improvement of the product g(x) = f(x) * pi(x) of the objective and the
decision maker's preference score, times the probability of feasibility of each constraint, in log space:

    log alpha(x) = log E[max{f(x) pi(x) - g*, 0}]  +  sum_i log P(c_i(x) <= 0)

    f         ContinuousGP of the objective, f(x) ~ N(mean, sd^2)
    pi(x)     the preference score, known once the preference model is fitted: the soft-Copeland score C(x) (with
              landmarks) or E[Phi(u(x))], the probit of the latent utility u of a PreferenceGP; 1 without a DM
    g*        the incumbent product, max_i y_i pi(x_i) over the feasible evaluations
    P(c_i<=0) Phi(-mean_i / sd_i) from the ContinuousGP of constraint i, as torch.special.log_ndtr (stable for
              very negative arguments, where log(Phi(z)) would give log(0) = -inf)

pi(x) is not random, so  E[max{f pi - g*, 0}] = pi(x) * E[max{f - t(x), 0}],  t(x) = g* / pi(x): the usual EI of f with a
threshold that depends on x (the less the DM likes x, the higher the bar), computed with BoTorch's stable helper.  An
evaluated x with y pi(x) <= g* has nothing left to gain, so the loop does not keep resampling it.  Without a DM pi = 1
and this is BoTorch's LogExpectedImprovement with best_f = the best y.  The product needs f > 0 (otherwise a smaller
pi would make f pi larger).

Every model takes inputs in original units (the transforms live inside the models).  When there is no feasible
evaluation yet, EI is undefined and the acquisition is log PoF + log pi (find a feasible design first).  The objective
is MAXIMISED.
"""

from .gp_utils import DTYPE, bounds_tensor

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import _log_ei_helper
from botorch.optim import optimize_acqf
from botorch.utils.transforms import t_batch_mode_transform


class LogEIOfProduct(AcquisitionFunction):
    """log E[max{f(x) pi(x) - g*, 0}] + sum_i log PoF_i(x), pi(x) the preference score (see the module docstring).

    f_gp       fitted ContinuousGP of the objective
    c_gps      fitted ContinuousGPs, one per constraint c_i(x) <= 0 (may be empty)
    pref_gp    fitted PreferenceGP of the decision maker's comparisons (None: no preference factor, pi = 1)
    best_g     the incumbent product g* = max_i y_i pi(x_i) over the feasible evaluations; None = there is none yet
    copeland   a prepared copeland.SoftCopeland (pref_gp and fixed landmarks): the preference score is then the
               soft-Copeland score C(x) (González et al. 2017, eq. 5) instead of pi(x) = E[Phi(u(x))]; None: pi(x)

    Its terms, each of shape (b,) for X of shape (b, 1, d), add up to the acquisition:
        log_ei(X)    log E[max{f(x) - g* / pi(x), 0}]   (None while best_g is None)
        log_pref(X)  log pi(x)
        log_pof(X)   sum_i log P(c_i(x) <= 0)
    """

    def __init__(self, f_gp, c_gps=(), pref_gp=None, best_g=None, copeland=None):
        super().__init__(model=f_gp.model)
        self.c_models, self.pref_gp, self.copeland = [c.model for c in c_gps], pref_gp, copeland
        self.best_g = best_g
        self.log_ei = None if best_g is None else self._log_ei

    def log_pof(self, X):
        """sum_i log P(c_i(X) <= 0), shape (b,), for X of shape (b, 1, d)."""
        total = torch.zeros(X.shape[0], dtype=X.dtype)
        for c_model in self.c_models:
            posterior = c_model.posterior(X)
            mean, sd = posterior.mean.view(-1), posterior.variance.clamp_min(1e-12).sqrt().view(-1)
            total = total + torch.special.log_ndtr(-mean / sd)
        return total

    def log_pref(self, X):
        """log of the preference score (pi or Copeland), shape (b,), for X of shape (b, 1, d); 0 without a preference model."""
        if self.pref_gp is None:
            return torch.zeros(X.shape[0], dtype=X.dtype)
        if self.copeland is None:
            score = self.pref_gp.prob_torch(X)
        else:
            score = self.copeland(X.reshape(-1, X.shape[-1]))
        return score.view(-1).clamp_min(1e-300).log()

    def _log_ei(self, X):
        """log E[max{f(x) - t(x), 0}], t(x) = g* / pi(x), shape (b,), for X of shape (b, 1, d)."""
        posterior = self.model.posterior(X)
        mean = posterior.mean.view(-1)
        sd = posterior.variance.clamp_min(1e-12).sqrt().view(-1)
        threshold = self.best_g / self.log_pref(X).exp()
        return sd.log() + _log_ei_helper((mean - threshold) / sd)

    def value(self, X):
        """log alpha for X of shape (b, 1, d) -> (b,).  A plain method, so that IPython's autoreload picks up edits (it
        cannot update forward, whose decorator keeps the original function)."""
        log_probs = self.log_pof(X) + self.log_pref(X)
        return log_probs if self.log_ei is None else self.log_ei(X) + log_probs

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        return self.value(X)


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
