"""Soft-Copeland score C(x) = 1/M sum_k P(x > x_k) of a fitted PreferenceGP, with the landmark side computed once.

The preference model is fitted once and the landmarks are fixed, so everything that only involves the landmarks
(their posterior mean and variance, and their link to the training data) is the same at every evaluation.  Only the
candidate side is recomputed, and only what the score needs: each candidate's posterior mean and variance and its
covariance with every landmark.  The n x n covariance among the candidates, which PairwiseGP.posterior also builds,
is never formed.

The formulas are those of PairwiseGP.posterior (Laplace approximation, eval mode), written out so that the landmark
parts can be cached; PreferenceGP.soft_copeland_torch gives the same numbers and stays as the reference."""
import math

import torch

from .gp_utils import DTYPE


class SoftCopeland:
    """C(x) for a fitted PreferenceGP and fixed landmarks.  Use: SoftCopeland(preference_gp, landmarks).prepare(), then
    call the object on designs X of shape (n, d) in original units: it returns C of shape (n,), keeping gradients."""

    def __init__(self, preference_gp, landmarks):
        self.preference_gp = preference_gp
        self.landmarks = torch.as_tensor(landmarks, dtype=DTYPE)
        self.is_prepared = False

    def prepare(self):
        """Compute and store everything that depends on the landmarks only.  Call again after refitting the GP."""
        model = self.preference_gp.model
        with torch.no_grad():
            model.posterior(self.landmarks[:1])                   # PairwiseGP refreshes its derived values lazily
            self._train_inputs = model.transform_inputs(model.datapoints)
            self._weights = torch.cholesky_solve(
                (model.utility - model._prior_mean(self._train_inputs)).unsqueeze(-1), model.covar_chol)
            self._link_to_data = torch.linalg.solve(model.hlcov_eye, model.likelihood_hess)
            landmarks = model.transform_inputs(self.landmarks)
            self._landmarks = landmarks
            landmark_data_covariance = model.covar_module(landmarks, self._train_inputs).to_dense()
            self._landmark_mean = (landmark_data_covariance @ self._weights).squeeze(-1) + model._prior_mean(landmarks)
            self._landmark_projection = self._link_to_data @ landmark_data_covariance.T
            self._landmark_variance = (model.covar_module(landmarks, diag=True)
                                       - (landmark_data_covariance * self._landmark_projection.T).sum(-1))
        self.is_prepared = True
        return self

    def __call__(self, X):
        """C(x) for each row of X (n, d); gradients flow to X."""
        if not self.is_prepared:
            raise RuntimeError("call prepare() first")
        model = self.preference_gp.model
        X = model.transform_inputs(torch.as_tensor(X, dtype=DTYPE))
        data_covariance = model.covar_module(X, self._train_inputs).to_dense()                 # (n, N)
        mean = (data_covariance @ self._weights).squeeze(-1) + model._prior_mean(X)           # (n,)
        variance = model.covar_module(X, diag=True) - (data_covariance * (data_covariance @ self._link_to_data.T)).sum(-1)
        landmark_covariance = (model.covar_module(X, self._landmarks).to_dense()
                               - data_covariance @ self._landmark_projection)                  # (n, M)
        diff_mean = mean[:, None] - self._landmark_mean[None, :]
        diff_variance = (variance[:, None] + self._landmark_variance[None, :] - 2 * landmark_covariance).clamp_min(0)
        if self.preference_gp.link == "probit":
            return torch.special.ndtr(diff_mean / torch.sqrt(2 + diff_variance)).mean(dim=1)
        return torch.sigmoid(diff_mean / torch.sqrt(1 + math.pi * diff_variance / 8)).mean(dim=1)
