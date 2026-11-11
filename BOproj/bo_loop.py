"""Bayesian optimisation with black-box constraints and a decision maker (DM) who compares designs.

    opt = BayesianOptimizer(objective, bounds, constraints=[c1], comparer=dm, X_pref=X_pref, run_dir="runs/x")
    opt.run(n_iter=40)
    opt.plot_points(); opt.plot_convergence(); opt.plot_diagnostics(); opt.plot_acquisition(n=15)

The problem
    objective(X)    (n, d) -> (n,), MAXIMISED
    constraint(X)   (n, d) -> (n,), feasible where <= 0; each constraint gets its own ContinuousGP
    bounds          (d, 2) the search box (fixed)

The decision maker (optional)
    comparer(XA, XB)   (n, d), (n, d) -> (n,) bools, True where the XA row wins.  Model: PreferenceGP (probit link) of
                       the comparisons, a latent utility u;  pi(x) = E[Phi(u(x))]
    X_pref             designs shown to the DM before the start, never evaluated: len(X_pref) random pairs are compared
The DM is only asked these comparisons: the preference model is fitted once, before the first evaluation.

One BO step
    propose():   fit the GPs on the data so far (once: they are kept in opt.models), maximise
                 log EI + sum_i log PoF_i + log pi  (acquisition.LogEIProduct)
    evaluate():  call the objective and the constraints at the chosen x
    then the step is recorded (one row of history.csv) and saved.
The incumbent (best y, the EI threshold) is the best feasible evaluation.  The preference steers the search through
pi; it does not decide which design is the best.

Files in run_dir, rewritten after every evaluation
    config.json          settings, seed, library versions
    history.csv          one row per evaluation: x, y, c, feasible, best y so far, regret, distance from the best x
                         to the optimum; BO rows also: the objective GP's prediction at x before evaluating it,
                         acquisition value, PoF, pi, the objective GP's lengthscales and noise, timings
    data.npz             X, y, C, X_pref and the comparisons (pairs_a vs pairs_b, who won)
    models/iter_###.pt   the GPs that chose evaluation ###:  load_models(run_dir, ###)
    summary.json         best point and value, regret, distance to the optimum, number of evaluations
"""

import csv
import json
import time
import warnings
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from . import bo_plotting
from .acquisition import LogEIProduct, optimize
from .gp import ContinuousGP, PreferenceGP
from .gp_utils import DTYPE

# The GPs of one step: objective (ContinuousGP), constraints (list of ContinuousGP), preference (PreferenceGP or None).
#  A named tuple, so  objective_gp, constraint_gps, preference_gp = models  also works.
Models = namedtuple("Models", "objective constraints preference")


def initial_design(bounds, n, method="sobol", seed=27):
    """n points inside bounds (d, 2). Possible methods: 'sobol' (default), 'lhs' or 'random'."""
    box = np.asarray(bounds, float)
    if method == "sobol":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            unit_points = qmc.Sobol(d=len(box), scramble=True, seed=seed).random(n)
    elif method == "lhs":
        unit_points = qmc.LatinHypercube(d=len(box), seed=seed).random(n)
    elif method == "random":
        unit_points = np.random.default_rng(seed).random((n, len(box)))
    else:
        raise ValueError("method must be 'sobol', 'lhs' or 'random'")
    return box[:, 0] + unit_points * (box[:, 1] - box[:, 0])


class BayesianOptimizer:
    """
    objective, bounds, constraints, comparer, X_pref:  see the module docstring
    init          initial design: 'sobol', 'lhs' or 'random', n_init points (default 2 (d + 1))
    strategy      'logei' (the BO step above) or 'random' (uniform points: a baseline with the same saved format)
    f_optimum     known best value (benchmarks): regret = f_optimum - best y
    x_optimum     known best location: dist_to_opt = |best x - x_optimum|
    """

    num_restarts, raw_samples = 10, 512        # the acquisition optimiser's L-BFGS-B starts and random candidates

    def __init__(self, objective, bounds, constraints=(), comparer=None, X_pref=None, init="sobol", n_init=None,
                 strategy="logei", seed=0, run_dir=None, f_optimum=None, x_optimum=None):
        self.objective, self.constraints = objective, list(constraints)
        self.bounds = np.asarray(bounds, float)
        self.d, self.m = len(self.bounds), len(self.constraints)
        self.comparer = comparer
        self.X_pref = np.empty((0, self.d)) if X_pref is None else np.asarray(X_pref, float)
        self.init, self.n_init = init, n_init or 2 * (self.d + 1)
        self.X_init = initial_design(self.bounds, self.n_init, init, seed)
        self.strategy, self.seed, self.rng = strategy, seed, np.random.default_rng(seed)
        self.run_dir = None if run_dir is None else Path(run_dir)
        self.f_optimum = f_optimum
        self.x_optimum = None if x_optimum is None else np.asarray(x_optimum, float)
        # the data, filled by run()
        self.X, self.y, self.C = np.empty((0, self.d)), np.empty(0), np.empty((0, self.m))
        self.pairs_a = self.pairs_b = np.empty((0, self.d))       # the DM's comparisons: pairs_a[j] vs pairs_b[j] ...
        self.pairs_a_wins = np.empty(0, bool)                     # ... True if pairs_a[j] won
        self.preference_gp = None                                 # fitted once, on those comparisons
        self.records, self.models = [], {}      # rows of history.csv;  n -> the GPs fitted on the first n evaluations

    #the data
    @property
    def feasible(self):
        """Per evaluation: every constraint <= 0 (always True without constraints)."""
        return np.all(self.C <= 0, axis=1)

    def best(self, n=None):
        """(x, y) of the best feasible evaluation among the first n (default all), or (None, nan)."""
        candidates = np.flatnonzero(self.feasible[:n])
        if not len(candidates):
            return None, np.nan
        i = candidates[np.argmax(self.y[candidates])]
        return self.X[i], self.y[i]

    def history(self, column):
        """One column of history.csv as a float array (nan where a row has no value, e.g. model columns of the
        initial design)."""
        return np.array([record.get(column, np.nan) for record in self.records], float)

    #the loop
    def run(self, n_iter):
        """Evaluate the initial design (on the first call), then n_iter steps.  Saves after every evaluation."""
        if not self.records:                       
            self._t_start = time.time()
            self._compare_before_start()
            self._save_config()
        for x in self.X_init[len(self.y):]:
            self._step(x, "init")
        for _ in range(n_iter):
            if self.strategy == "random":
                self._step(self.bounds[:, 0] + self.rng.random(self.d) * np.ptp(self.bounds, axis=1), "rand")
            else:
                x, proposal = self.propose()
                self._step(x, "bo", proposal)
        return self

    def propose(self):
        """The next design: maximise the acquisition built on the GPs of all evaluations so far.
        Returns x and what the models said about x before it is evaluated (columns of history.csv)."""
        n = len(self.y)
        t_start = time.perf_counter()
        models = self.models_at(n)
        t_fit = time.perf_counter() - t_start
        acquisition = self.acquisition(n)
        x, acq_value = optimize(acquisition, self.bounds, self.d, self.num_restarts, self.raw_samples,
                                seed=self.seed * 100_003 + n)
        mean, var = models.objective.predict(x)                     # gp.py reads a (d,) array as one design
        with torch.no_grad():
            x_batch = torch.as_tensor(x[None, None], dtype=DTYPE)
            pof = float(acquisition.log_pof(x_batch).exp()[0])
            pi = float(acquisition.log_pref(x_batch).exp()[0])
        proposal = {"pred_mean": mean[0], "pred_sd": np.sqrt(var[0]), "acq_value": acq_value, "pof": pof, "pi": pi,
                    **{f"ls{i + 1}": value for i, value in enumerate(models.objective.lengthscales())},
                    "noise": models.objective.noise(), "t_fit": t_fit, "t_acq": time.perf_counter() - t_start - t_fit}
        return x, proposal

    def evaluate(self, x):
        """Evaluate the objective and the constraints at x and append the results to the data."""
        y = self.objective(x[None])[0]
        constraint_values = [constraint(x[None])[0] for constraint in self.constraints]
        self.X, self.y = np.vstack([self.X, x]), np.r_[self.y, y]
        self.C = np.vstack([self.C, [constraint_values]])

    def _step(self, x, phase, proposal=None):
        """Evaluate x, then record and save that evaluation."""
        self.evaluate(x)
        self.records.append(self._record(phase, proposal or {}))
        self.save()

    #the decision maker
    def _compare_before_start(self):
        """Let the DM compare len(X_pref) random pairs of X_pref, then fit the preference model on the answers."""
        if self.comparer is None:
            return
        if len(self.X_pref) < 2:
            warnings.warn("a comparer is given but X_pref has fewer than 2 designs: no preference model is fitted, "
                          "the acquisition has no pi term")
            return
        n = len(self.X_pref)
        first = self.rng.integers(0, n, n)
        second = (first + self.rng.integers(1, n, n)) % n               # never a design against itself
        self.pairs_a, self.pairs_b = self.X_pref[first], self.X_pref[second]
        self.pairs_a_wins = self.comparer(self.pairs_a, self.pairs_b)
        with seeded(self.seed):
            self.preference_gp = PreferenceGP(bounds=self.bounds, link="probit").fit(
                self.pairs_a, self.pairs_b, self.pairs_a_wins)

    #models
    def models_at(self, n=None):
        """The GPs fitted on the first n evaluations (default all): the models that chose evaluation n.
        Each set is fitted once and kept in self.models, so plots reuse the models the loop used."""
        n = len(self.y) if n is None else n
        if n not in self.models:
            with seeded(self.seed * 100_003 + n):
                self.models[n] = self._fit_models(n)
        return self.models[n]

    def _fit_models(self, n):
        X = self.X[:n]
        objective_gp = ContinuousGP(bounds=self.bounds).fit(X, self.y[:n])
        constraint_gps = [ContinuousGP(bounds=self.bounds).fit(X, self.C[:n, i]) for i in range(self.m)]
        return Models(objective_gp, constraint_gps, self.preference_gp)

    def acquisition(self, n=None):
        """The acquisition after the first n evaluations (default all), i.e. the one that chose evaluation n."""
        models = self.models_at(n)
        _, best_y = self.best(n)
        return LogEIProduct(models.objective, models.constraints, models.preference,
                            best_f=None if np.isnan(best_y) else best_y)

    #recording and saving
    def _record(self, phase, proposal):
        """The history.csv row of the last evaluation."""
        i = len(self.y) - 1
        best_x, best_y = self.best()
        regret = np.nan if self.f_optimum is None else self.f_optimum - best_y
        distance = np.nan if self.x_optimum is None or best_x is None else float(np.linalg.norm(best_x - self.x_optimum))
        return {"eval": i, "phase": phase, **{f"x{j + 1}": value for j, value in enumerate(self.X[i])}, "y": self.y[i],
                **{f"c{j + 1}": value for j, value in enumerate(self.C[i])}, "feasible": bool(self.feasible[i]),
                "best_y": best_y, "regret": regret, "dist_to_opt": distance, **proposal,
                "wall_time": time.time() - self._t_start}

    def _save_config(self):
        if self.run_dir is None:
            return
        import botorch
        (self.run_dir / "models").mkdir(parents=True, exist_ok=True)
        config = {"objective": _name(self.objective), "constraints": [_name(c) for c in self.constraints],
                  "bounds": self.bounds.tolist(), "init": self.init, "n_init": self.n_init,
                  "strategy": self.strategy, "seed": self.seed,
                  "decision_maker": None if self.comparer is None else _name(getattr(self.comparer, "utility",
                                                                                     self.comparer)),
                  "n_pref": len(self.X_pref),
                  "f_optimum": self.f_optimum, "x_optimum": None if self.x_optimum is None else self.x_optimum.tolist(),
                  "num_restarts": self.num_restarts, "raw_samples": self.raw_samples,
                  "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "versions": {"numpy": np.__version__, "torch": torch.__version__, "botorch": botorch.__version__}}
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    def save(self):
        """Write history.csv, data.npz, summary.json and the models that chose the last evaluation to run_dir."""
        if self.run_dir is None or not self.records:
            return
        columns = list(dict.fromkeys(key for record in self.records for key in record))  # init rows lack model columns
        with open(self.run_dir / "history.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, columns)
            writer.writeheader()
            writer.writerows(self.records)
        np.savez(self.run_dir / "data.npz", X=self.X, y=self.y, C=self.C, X_pref=self.X_pref, pairs_a=self.pairs_a,
                 pairs_b=self.pairs_b, pairs_a_wins=self.pairs_a_wins)
        last = len(self.y) - 1
        if last in self.models:                                         # a BO step: keep the models that chose it
            torch.save(self.models[last], self.run_dir / "models" / f"iter_{last:03d}.pt")
        best_x, best_y = self.best()
        record = self.records[-1]
        (self.run_dir / "summary.json").write_text(json.dumps({
            "n_evals": len(self.y), "n_feasible": int(self.feasible.sum()), "n_comparisons": len(self.pairs_a_wins),
            "best_x": None if best_x is None else best_x.tolist(), "best_y": best_y,
            "regret": record["regret"], "dist_to_opt": record["dist_to_opt"],
            "time_s": time.time() - self._t_start}, indent=2))

    #plots
    def plot_points(self, **kwargs):
        """Where the optimiser sampled.  Arguments: bo_plotting.plot_points."""
        return bo_plotting.plot_points(self, **kwargs)

    def plot_convergence(self, **kwargs):
        """Regret and distance to the optimum against evaluations.  Arguments: bo_plotting.plot_convergence."""
        return bo_plotting.plot_convergence(self, **kwargs)

    def plot_diagnostics(self, **kwargs):
        """Predictions vs observations, hyperparameters, acquisition values, timings.  bo_plotting.plot_diagnostics."""
        return bo_plotting.plot_diagnostics(self, **kwargs)

    def plot_acquisition(self, n=None, **kwargs):
        """Maps of log EI, log PoF, log pi and their sum after n evaluations.  bo_plotting.plot_acquisition."""
        return bo_plotting.plot_acquisition(self, n=n, **kwargs)


def load_models(run_dir, n):
    """The GPs that chose evaluation n of a saved run, exactly as fitted then: Models(objective, constraints, preference)."""
    return torch.load(Path(run_dir) / "models" / f"iter_{n:03d}.pt", weights_only=False)


@contextmanager
def seeded(seed):
    """Run the block with torch's and numpy's global random generators seeded, then restore them.  Fitting draws from
    both (BoTorch retries a failed fit from random hyperparameters; PairwiseGP perturbs the start of its utility
    search with np.random), so without this the same run seed would not give the same run."""
    numpy_state = np.random.get_state()
    np.random.seed(seed)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            yield
    finally:
        np.random.set_state(numpy_state)


def _name(function):
    return getattr(function, "__name__", type(function).__name__)
