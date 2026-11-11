"""Wrapping of BoTorch GP models for continuous, preference and binary data, making it simpler to input data, update, plot 

Common interface:
    model.fit(...)                       returns the model, so  m = ContinuousGP().fit(X, y)  works
    model.update(...)                    ContinuousGP and BinaryGP: add new data and refit on all of it (returns the model)
    model.predict(X) -> (mean, var)      latent posterior, numpy arrays of shape (n,)
    model.posterior(X)                   the BoTorch posterior (torch), for acquisition functions
    model.model                          the underlying BoTorch model
    model.plot(...)                      see gp_plotting.py; show=True opens a window, save="f.png" writes a file
Inputs X can be lists, numpy arrays or torch tensors; a 1-d array is read as n points when the model is
1-dimensional and as one point otherwise.

Files:  gp.py (the models), gp_utils.py (conversion, bounds, kernels, comparisons), gp_plotting.py (plots).
"""

import math
from .gp_utils import (DTYPE, LogitBernoulliLikelihood, as_2d, bounds_tensor, format_comparisons, make_kernel,
                      plot_box, query_inputs, to_numpy, training_inputs)

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, PairwiseGP, SingleTaskGP
from botorch.models.approximate_gp import SingleTaskVariationalGP
from botorch.models.likelihoods.pairwise import PairwiseLogitLikelihood, PairwiseProbitLikelihood
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO

from .gp_plotting import plot_binary, plot_continuous, plot_preference


class _BaseGP:
    """Prediction shared by the three models."""

    model = None
    d = None
    plot_bounds = None              

    def _X(self, X):
        return query_inputs(X, self.d) #returning the inputs the way BoTorch needs, a tensor of shape (n, d)

    def _check_fitted(self): 
        if self.model is None:
            raise RuntimeError("call fit(...) before using the model") #just checking if the model has been fitted, if not, raise an error

    def posterior(self, X, **kw): # 
        """BoTorch posterior of the latent function at X."""
        self._check_fitted()
        self.model.eval()
        return self.model.posterior(self._X(X), **kw)

    def predict(self, X, full_cov=False):
        """Posterior mean and variance (or covariance) of the latent function, as numpy arrays."""
        with torch.no_grad():
            post = self.posterior(X)
            mean = post.mean.squeeze(-1).cpu().numpy()
            if full_cov:
                return mean, post.distribution.covariance_matrix.cpu().numpy()
            return mean, post.variance.squeeze(-1).clamp_min(0).cpu().numpy()

    def sample(self, X, n=1, seed=None):
        """n joint posterior samples of the latent function at X, shape (n, len(X))."""
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            return self.posterior(X).rsample(torch.Size([n])).squeeze(-1).cpu().numpy()


class ContinuousGP(_BaseGP):
    """f(x) from noisy evaluations y = f(x) + eps.

    bounds       (d, 2) input box, used by the Normalize transform and the plots; None = learned from the data
    kernel       'matern52' (default) or 'rbf', both ARD; or any GPyTorch Kernel
    noise        None = learned; a float = known noise variance for every point (fixed-noise GP)
    cat_dims     indices of categorical inputs -> MixedSingleTaskGP (kernel argument then ignored)
    normalize_x, standardize_y   BoTorch input / outcome transforms (both on by default)
    """

    def __init__(self, bounds=None, kernel="matern52", noise=None, cat_dims=None, normalize_x=True,
                 standardize_y=True):
        self.bounds, self.kernel, self.fixed_noise = bounds, kernel, noise
        self.cat_dims = list(cat_dims) if cat_dims else []
        self.normalize_x, self.standardize_y = normalize_x, standardize_y
        self.model = self.mll = None

    def fit(self, X, y, y_var=None):
        """X (n, d), y (n,).  y_var: optional per-point noise variances (overrides `noise`)."""
        Yt = torch.as_tensor(to_numpy(y), dtype=DTYPE).reshape(-1, 1)
        Xt = training_inputs(X, len(Yt))
        self.d = Xt.shape[1]
        cont = [i for i in range(self.d) if i not in self.cat_dims]
        b = bounds_tensor(self.bounds, self.d)
        in_tf = None
        if self.normalize_x and cont:
            in_tf = Normalize(d=self.d, indices=cont if self.cat_dims else None,
                              bounds=b[:, cont] if (b is not None and self.cat_dims) else b)
        out_tf = Standardize(m=1) if self.standardize_y else None
        yvar = None
        if y_var is not None:
            yvar = torch.as_tensor(np.broadcast_to(to_numpy(y_var), (len(Yt),)).copy(), dtype=DTYPE).reshape(-1, 1)
        elif self.fixed_noise is not None:
            yvar = torch.full_like(Yt, float(self.fixed_noise))
        if self.cat_dims:
            self.model = MixedSingleTaskGP(Xt, Yt, cat_dims=self.cat_dims, train_Yvar=yvar,
                                           input_transform=in_tf, outcome_transform=out_tf)
        else:
            self.model = SingleTaskGP(Xt, Yt, train_Yvar=yvar, covar_module=make_kernel(self.kernel, self.d, False),
                                      input_transform=in_tf, outcome_transform=out_tf)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)
        self.model.eval()
        self.train_X, self.train_y = Xt.numpy(), Yt.squeeze(-1).numpy()
        self.train_yvar = yvar.squeeze(-1).numpy() if y_var is not None else None
        self.plot_bounds = plot_box(self.train_X, self.bounds, self.d)
        return self

    def update(self, X_new, y_new, y_var_new=None):
        """Add new observations and refit on all the data (the hyperparameters are learned again).

        X_new (m, d) and y_new (m,); y_var_new only if the model was fitted with per-point y_var."""
        self._check_fitted()
        y_new = to_numpy(y_new).ravel()
        X_new = query_inputs(X_new, self.d).numpy()       # one point may be a flat array of length d
        if len(X_new) != len(y_new):
            raise ValueError(f"X_new has {len(X_new)} rows but there are {len(y_new)} new observations")
        X_all, y_all = np.vstack([self.train_X, X_new]), np.r_[self.train_y, y_new]
        if self.train_yvar is None:
            if y_var_new is not None:
                raise ValueError("this model was fitted without y_var, so y_var_new is not allowed")
            return self.fit(X_all, y_all)
        if y_var_new is None:
            raise ValueError("this model was fitted with y_var, so give y_var_new for the new points")
        v_new = np.broadcast_to(to_numpy(y_var_new), (len(y_new),))
        return self.fit(X_all, y_all, np.r_[self.train_yvar, v_new])

    def noise(self):
        """Observation noise variance in the units of y (mean over points for a fixed-noise model)."""
        self._check_fitted()
        nv = self.model.likelihood.noise.detach().mean()
        if self.standardize_y:
            nv = nv * self.model.outcome_transform.stdvs.squeeze() ** 2
        return float(nv)

    def lengthscales(self):
        """ARD lengthscales (in normalised input units if normalize_x)."""
        self._check_fitted()
        k = self.model.covar_module
        while not hasattr(k, "lengthscale") and hasattr(k, "base_kernel"):
            k = k.base_kernel
        return k.lengthscale.detach().cpu().numpy().ravel()

    def predict(self, X, full_cov=False, observation_noise=False):
        """Mean and variance of f(x); observation_noise=True gives the variance of a new y instead."""
        mean, var = super().predict(X, full_cov=full_cov)
        if observation_noise:
            var = var + (self.noise() * np.eye(len(mean)) if full_cov else self.noise())
        return mean, var

    def plot(self, **kwargs):
        """Mean, 95% band and data (1-d) or mean / sd maps (2-d).  Arguments: gp_plotting.plot_continuous."""
        return plot_continuous(self, **kwargs)


class PreferenceGP(_BaseGP):
    """Latent utility u(x) from pairwise comparisons (BoTorch PairwiseGP, Laplace approximation).
    Essentially the preference model of Chu and Ghahramani (2005)
    """

    def __init__(self, bounds=None, kernel="matern52", link="probit", normalize_x=True):
        if link not in ("probit", "logit"):
            raise ValueError("link must be 'probit' or 'logit'")
        self.bounds, self.kernel, self.link, self.normalize_x = bounds, kernel, link, normalize_x
        self.model = self.mll = None

    def fit(self, xa, xb, a_wins=None):
        """Row j of xa and xb is one comparison; a_wins[j] is True if xa[j] won (None: xa always wins)."""
        Xn, C = format_comparisons(as_2d(xa), as_2d(xb), a_wins)
        self.d = Xn.shape[1]
        b = bounds_tensor(self.bounds, self.d)
        in_tf = Normalize(d=self.d, bounds=b) if self.normalize_x else None
        lik = PairwiseProbitLikelihood() if self.link == "probit" else PairwiseLogitLikelihood()
        covar = None if self.kernel == "default" else make_kernel(self.kernel, self.d, True)
        self.model = PairwiseGP(torch.as_tensor(Xn, dtype=DTYPE), torch.as_tensor(C, dtype=torch.long),
                                likelihood=lik, covar_module=covar, input_transform=in_tf)
        self.mll = PairwiseLaplaceMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)
        self.model.eval()
        self.train_X, self.comparisons = Xn, C
        self.plot_bounds = plot_box(Xn, self.bounds, self.d)
        return self

    def prob_torch(self, X):
        """E[link(u(x))] for every design x in X, keeping gradients: the probit (Phi) or logit (sigmoid) of the latent
        utility, with its posterior uncertainty integrated out in closed form.

        X      designs in original units, shape (..., d): d inputs per design, any number of leading dimensions.
        Returns one probability per design: X's shape without the last axis, i.e. (n,) for (n, d), (b, 1) for (b, 1, d).

        Probit: Phi(m / sqrt(1 + v)).  Logit: sigmoid(m / sqrt(1 + pi v / 8)), the usual approximation.
        The zero of a utility learned from comparisons is not fixed by the data: it comes from the GP's prior mean,
        so 0.5 means 'as good as the typical design', not 'acceptable'."""
        self._check_fitted()
        X = torch.as_tensor(X, dtype=DTYPE)
        posterior = self.model.posterior(X.reshape(-1, 1, X.shape[-1]))
        mean, var = posterior.mean.reshape(-1), posterior.variance.reshape(-1)
        if self.link == "probit":
            prob = torch.special.ndtr(mean / torch.sqrt(1 + var))
        elif self.link == "logit":
            prob = torch.sigmoid(mean / torch.sqrt(1 + math.pi * var / 8))
        return prob.reshape(X.shape[:-1])

    def prob_better_torch(self, X, x_ref):
        """P(the DM prefers x to x_ref) for every design x in X, keeping gradients.

        X      designs in original units, shape (..., d): d inputs per design, any number of leading dimensions
               (e.g. (n, d) for n designs, or (b, 1, d) for a BoTorch batch).  x_ref: one design, shape (d,).
        Returns one probability per design: X's shape without the last axis, i.e. (n,) for (n, d), (b, 1) for (b, 1, d).

        Probit: Phi(m / sqrt(2 + v)), m and v = posterior mean and variance of u(x) - u(x_ref), so the GP's uncertainty
        about the difference is integrated out in closed form.  Logit: sigmoid(m / sqrt(1 + pi v / 8)), the usual
        approximation."""
        X = torch.as_tensor(X, dtype=DTYPE)
        reference = torch.as_tensor(x_ref, dtype=X.dtype).reshape(1, -1)
        prob = self.prob_better_matrix_torch(X.reshape(-1, X.shape[-1]), reference)
        return prob.reshape(X.shape[:-1])

    def prob_better_matrix_torch(self, X, X_ref, chunk_size=500):
        """P(the DM prefers X[i] to X_ref[k]) for every pair (i, k), keeping gradients.

        X      n designs in original units, shape (n, d).   X_ref  M reference designs, shape (M, d).
        Returns an (n, M) matrix.  One joint posterior over a block of designs and X_ref gives the mean and variance of
        every difference u(X[i]) - u(X_ref[k]), including the covariance between the two, so this is far cheaper than
        n * M separate two-point posteriors.  Row i does not depend on the other designs, so X is cut into blocks of
        chunk_size only to bound the size of the covariance matrix (n <= chunk_size: a single block).
        Link as in prob_better_torch."""
        self._check_fitted()
        X = torch.as_tensor(X, dtype=DTYPE)
        X_ref = torch.as_tensor(X_ref, dtype=DTYPE)
        blocks = [self._prob_better_block(X[start:start + chunk_size], X_ref) for start in range(0, len(X), chunk_size)]
        return torch.cat(blocks)

    def _prob_better_block(self, X, X_ref):
        """prob_better_matrix_torch for one block of designs: a single joint posterior over X and X_ref."""
        n = len(X)
        posterior = self.model.posterior(torch.cat([X, X_ref]).unsqueeze(0))
        mean = posterior.mean.reshape(-1)
        covariance = posterior.mvn.covariance_matrix.reshape(len(mean), len(mean))
        variance = covariance.diagonal()
        diff_mean = mean[:n, None] - mean[None, n:]
        diff_var = (variance[:n, None] + variance[None, n:] - 2 * covariance[:n, n:]).clamp_min(0)
        if self.link == "probit":
            return torch.special.ndtr(diff_mean / torch.sqrt(2 + diff_var))
        return torch.sigmoid(diff_mean / torch.sqrt(1 + math.pi * diff_var / 8))

    def soft_copeland_torch(self, X, landmarks=None):
        """Soft-Copeland score of Gonzalez et al. (2017), eq. 5, keeping gradients: C(x) = mean over the landmarks x_k
        of P(the DM prefers x to x_k), the average probability that x wins a duel.  Its maximiser is the Condorcet winner.

        X          n designs in original units, shape (n, d).
        landmarks  M designs standing in for the integral over the box, shape (M, d); None: X itself (e.g. a grid).
        Returns one score in [0, 1] per design, shape (n,)."""
        X = torch.as_tensor(X, dtype=DTYPE)
        landmarks = X if landmarks is None else landmarks
        return self.prob_better_matrix_torch(X, landmarks).mean(dim=1)

    def plot(self, **kwargs):
        """Latent utility with arrows from loser to winner.  Arguments: gp_plotting.plot_preference."""
        return plot_preference(self, **kwargs)


class BinaryGP(_BaseGP): #NOT USED SO FAR. JUST IN CASE IT'S NEEDED LATER, IF WE WANT TO HAVE THE OPTION TO HAVE THE DM ACCEPT/REJECT SOMETHING
    """GP for the acceptability a(x) of a design: P(a(x) = 1 | data) from accept / reject labels.

    BoTorch SingleTaskVariationalGP with a Bernoulli likelihood
    Inputs are scaled to [0, 1]^d with `bounds` (or the data range) before the model sees them, as specified in 
    BoTorch documentation; to use the
    BoTorch model directly (e.g. in an acquisition function), pass it inputs scaled the same way. """

    DEFAULT_FIT_KWARGS = {"optimizer_kwargs": {"options": {"maxiter": 200}}}

    def __init__(self, bounds=None, kernel="matern52", link="probit", normalize_x=True, inducing="data",
                 fit_kwargs=None, seed=0):
        if link not in ("probit", "logit"):
            raise ValueError("link must be 'probit' or 'logit'")
        self.bounds, self.kernel, self.link, self.normalize_x = bounds, kernel, link, normalize_x
        self.inducing, self.seed = inducing, seed
        self.fit_kwargs = self.DEFAULT_FIT_KWARGS if fit_kwargs is None else fit_kwargs
        self.model = self.mll = None

    def fit(self, X, labels):
        """X (n, d); labels: bools, {0, 1} or {-1, +1}."""
        lab = to_numpy(labels).ravel()
        Xt = training_inputs(X, len(lab))
        self.d = Xt.shape[1]
        Yt = torch.as_tensor((lab > 0).astype(float), dtype=DTYPE).reshape(-1, 1)
        b = bounds_tensor(self.bounds, self.d)
        if b is None:
            b = torch.stack([Xt.min(0).values, Xt.max(0).values])
        span = b[1] - b[0]
        self._lo, self._span = b[0], torch.where(span > 0, span, torch.ones_like(span))
        Xu = self._unit(Xt)
        lik = BernoulliLikelihood() if self.link == "probit" else LogitBernoulliLikelihood()
        if self.inducing == "data":
            kw = dict(inducing_points=Xu.clone(), learn_inducing_points=False)
        else:
            kw = dict(inducing_points=int(self.inducing), learn_inducing_points=True)
        # The variational set-up draws random numbers: seed a private copy of torch's generator so the same data
        # always gives the same model, without touching the caller's random state.
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            self.model = SingleTaskVariationalGP(Xu, Yt, likelihood=lik,
                                                 covar_module=make_kernel(self.kernel, self.d, True), **kw)
            self.mll = VariationalELBO(self.model.likelihood, self.model.model, num_data=len(Yt))
            fit_gpytorch_mll(self.mll, **self.fit_kwargs)
        self.model.eval()
        self.train_X, self.labels = Xt.numpy(), Yt.squeeze(-1).numpy().astype(bool)
        self.plot_bounds = plot_box(self.train_X, self.bounds, self.d)
        return self

    def update(self, X_new, labels_new):
        """Add new labelled designs and refit on all the data.  Same formats as fit(X, labels)."""
        self._check_fitted()
        lab_new = to_numpy(labels_new).ravel()
        X_new = query_inputs(X_new, self.d).numpy()       # one point may be a flat array of length d
        if len(X_new) != len(lab_new):
            raise ValueError(f"X_new has {len(X_new)} rows but there are {len(lab_new)} new labels")
        return self.fit(np.vstack([self.train_X, X_new]), np.r_[self.labels.astype(float), lab_new])

    def _unit(self, Xt):
        return (Xt - self._lo) / self._span if self.normalize_x else Xt

    def posterior(self, X, **kw):
        """BoTorch posterior of the latent g at X (given in original units; scaled to the unit cube here)."""
        self._check_fitted()
        self.model.eval()
        return self.model.posterior(self._unit(self._X(X)), **kw)

    def prob(self, X):
        """P(y = 1 | data) at each row of X, with the latent uncertainty integrated out:
        E[Phi(g)] (probit) or E[sigmoid(g)] (logit, the 'binary softmax'), g ~ posterior of the latent."""
        with torch.no_grad():
            return self.prob_torch(self._X(X)).reshape(-1).cpu().numpy()

    def prob_torch(self, X):
        """prob() for a torch tensor X of shape (..., d) in original units, keeping gradients (for acquisition
        functions, e.g. BoTorch's (b, 1, d) batches).  Returns shape (...)."""
        self._check_fitted()
        self.model.eval()
        post = self.model.posterior(self._unit(X))
        return self.model.likelihood.marginal(post.distribution).probs

    def plot(self, **kwargs):
        """Latent g and P(y = 1) (1-d) or a probability map with the boundary (2-d).
        Arguments: gp_plotting.plot_binary."""
        return plot_binary(self, **kwargs)