"""Test problems with known answers for the BO loop: objectives, constraints, and simulated decision makers.

Objectives are MAXIMISED (the loop's convention), so minimisation benchmarks come negated.
Every function takes X of shape (n, d) and returns (n,).  A utility u(x) defines a decision maker who knows u and
prefers the design with the larger u; x is acceptable where u(x) > 0.  The optimiser never sees u, only the answers.
"""

import numpy as np

ROSEN_BOUNDS = [[-2.0, 2.0], [-2.0, 2.0]]


#objectives
def rosenbrock(X):
    """f(x) = sum_i 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2, minimum 0 at (1, ..., 1)."""
    return np.sum(100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2, axis=1)


def neg_rosenbrock(X):
    return -rosenbrock(X)


def neg_log_rosenbrock(X):
    return -np.log1p(rosenbrock(X))


def ball(X):
    """Constraint |x|^2 - d <= 0; Rosenbrock's optimum (1, ..., 1) is exactly on the boundary."""
    return (X ** 2).sum(1) - X.shape[1]


#utilities
# 2-d utilities on [-2, 2]^2, each chosen so that Rosenbrock's optimum (1, 1) is NOT acceptable: the preference
# moves the answer, and we can check that the BO points follow it.
def u_linear(X):
    """Accept x1 < 0.5 (a half plane)."""
    return 0.5 - X[:, 0]


def u_polynomial(X):
    """Accept a disc of radius 0.8 around (-1, 1)."""
    return 1 - ((X[:, 0] + 1) ** 2 + (X[:, 1] - 1) ** 2) / 0.64


def u_exponential(X):
    """Accept x1 + x2 < log 2 (a half plane with an exponentially growing penalty)."""
    return 2 - np.exp(X[:, 0] + X[:, 1])


UTILITIES = {"linear": u_linear, "polynomial": u_polynomial, "exponential": u_exponential}


class PairwiseLabeler:
    """Simulated decision maker who knows the utility: xa wins against xb exactly when u(xa) > u(xb)."""

    def __init__(self, utility):
        self.utility = utility

    def __call__(self, xa, xb):
        return self.utility(xa) > self.utility(xb)


def grid_optimum(objective, bounds, accept=None, n=801):
    """Best point of objective on an n x n grid (2-d), among points where accept(X) is True if given.
    Returns (x*, f*): the reference answer for regret and distance to the optimal location."""
    box = np.asarray(bounds, float)
    grid_x1, grid_x2 = np.meshgrid(np.linspace(*box[0], n), np.linspace(*box[1], n))
    X = np.c_[grid_x1.ravel(), grid_x2.ravel()]
    values = objective(X)
    if accept is not None:
        values = np.where(accept(X), values, -np.inf)
    i = np.argmax(values)
    return X[i], values[i]
