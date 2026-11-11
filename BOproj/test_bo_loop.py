"""Checks of acquisition.py, bo_loop.py and BinaryGP.prob_torch on small problems.  About two minutes.

    python -m pytest test_bo_loop.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # lets pytest find the package
from BOproj.acquisition import LogEIProduct
from BOproj.benchmarks import grid_optimum, u_linear
from BOproj.bo_loop import BayesianOptimizer, initial_design, load_models
from BOproj.gp import BinaryGP, ContinuousGP

BOUNDS = [[-2.0, 2.0]] * 2
class FastOptimizer(BayesianOptimizer):
    num_restarts, raw_samples = 4, 128      # a cheaper acquisition search: these tests check logic, not the optimum


def bowl(X):
    """Maximum 0 at (0.5, -0.5)."""
    X = np.atleast_2d(X)
    return -((X[:, 0] - 0.5) ** 2 + (X[:, 1] + 0.5) ** 2)


def half_plane(X):
    """Feasible where x1 <= 0."""
    return np.atleast_2d(X)[:, 0]


def fitted(n=12, seed=0):
    X = np.random.default_rng(seed).uniform(-2, 2, (n, 2))
    return X, ContinuousGP(bounds=BOUNDS).fit(X, bowl(X)), ContinuousGP(bounds=BOUNDS).fit(X, half_plane(X))


#acquisition
def test_without_constraints_it_is_botorch_log_ei():
    _, f, _ = fitted()
    Xq = torch.as_tensor(np.random.default_rng(1).uniform(-2, 2, (20, 1, 2)))
    with torch.no_grad():
        assert torch.allclose(LogEIProduct(f, [], best_f=-0.5)(Xq), LogExpectedImprovement(f.model, best_f=-0.5)(Xq))


def test_pof_is_phi_of_minus_mean_over_sd():
    _, f, c = fitted()
    Xq = np.random.default_rng(2).uniform(-2, 2, (20, 2))
    m, v = c.predict(Xq)
    with torch.no_grad():
        lp = LogEIProduct(f, [c], best_f=-0.5).log_pof(torch.as_tensor(Xq[:, None]))
    assert np.allclose(np.exp(lp.numpy()), norm.cdf(-m / np.sqrt(v)), atol=1e-6)


def test_nothing_feasible_means_pof_only():
    _, f, c = fitted()
    acq = LogEIProduct(f, [c], best_f=None)
    Xq = torch.as_tensor(np.random.default_rng(3).uniform(-2, 2, (10, 1, 2)))
    with torch.no_grad():
        assert acq.log_ei is None and torch.allclose(acq(Xq), acq.log_pof(Xq))


def test_prob_torch_matches_prob_and_has_gradients():
    X = np.random.default_rng(5).uniform(-2, 2, (30, 2))
    g = BinaryGP(bounds=BOUNDS, link="logit").fit(X, X[:, 0] < 0.5)
    Xq = np.random.default_rng(6).uniform(-2, 2, (8, 2))
    Xt = torch.as_tensor(Xq[:, None, :]).requires_grad_(True)          # BoTorch's (b, 1, d) batch
    p = g.prob_torch(Xt)
    p.sum().backward()
    assert np.allclose(p.detach().numpy().ravel(), g.prob(Xq)) and torch.all(torch.isfinite(Xt.grad))


#optimiser
@pytest.mark.parametrize("method", ["sobol", "lhs", "random"])
def test_initial_design_stays_in_the_box(method):
    X = initial_design([[0.0, 2.0], [-1.0, 1.0]], 16, method, seed=0)
    assert X.shape == (16, 2) and X[:, 0].min() >= 0 and X[:, 0].max() <= 2 and np.abs(X[:, 1]).max() <= 1
    if method == "lhs":                                                 # one point in each of 16 slices per input
        assert sorted(np.floor(X[:, 0] / 2 * 16).astype(int)) == list(range(16))


def test_pairwise_run_on_rosenbrock_saves_and_reloads_its_preference_model(tmp_path):
    """Rosenbrock with a DM who prefers smaller x1 (u = 0.5 - x1): 30 pairs before the start, and the saved models of a
    step hold the preference model fitted on exactly those."""
    from BOproj.benchmarks import ROSEN_BOUNDS, PairwiseLabeler, neg_rosenbrock
    opt = FastOptimizer(neg_rosenbrock, ROSEN_BOUNDS, comparer=PairwiseLabeler(u_linear),
                        X_pref=initial_design(ROSEN_BOUNDS, 30, seed=9), run_dir=tmp_path).run(12)
    assert np.all((opt.history("pi")[opt.n_init:] > 0) & (opt.history("pi")[opt.n_init:] < 1))
    n = len(opt.y) - 1
    models = load_models(tmp_path, n)
    assert len(models.preference.comparisons) == 30


@pytest.fixture(scope="module")
def con_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("con")
    return FastOptimizer(bowl, BOUNDS, constraints=[half_plane], seed=0, run_dir=d, f_optimum=-0.25,
                  x_optimum=[0, -0.5]).run(8), d


def test_optimizer_finds_the_constrained_optimum(con_run):
    """Best feasible point of bowl with x1 <= 0 is (0, -0.5), value -0.25."""
    opt, _ = con_run
    best_x, best_y = opt.best()
    assert best_x[0] <= 0 and best_y > -0.25 - 0.05


def test_every_file_is_written_and_consistent(con_run):
    opt, d = con_run
    n = len(opt.y)
    assert n == opt.n_init + 8
    for f in ("config.json", "history.csv", "data.npz", "summary.json"):
        assert (d / f).exists(), f
    assert len(list((d / "models").glob("iter_*.pt"))) == 8
    lines = (d / "history.csv").read_text().splitlines()
    assert len(lines) == n + 1 and lines[0].startswith("eval,phase,x1,x2,y,c1,feasible")
    data = np.load(d / "data.npz")
    assert data["X"].shape == (n, 2) and data["C"].shape == (n, 1) and np.allclose(data["y"], bowl(data["X"]))
    h = np.genfromtxt(d / "history.csv", delimiter=",", names=True, dtype=None, encoding=None)
    assert np.array_equal(h["feasible"], data["X"][:, 0] <= 0)
    best = np.maximum.accumulate(np.where(h["feasible"], h["y"], -np.inf))
    assert np.allclose(h["best_y"][np.isfinite(best)], best[np.isfinite(best)])
    assert np.allclose(h["dist_to_opt"][-1], np.linalg.norm(opt.best()[0] - [0, -0.5]))
    bo = h["phase"] == "bo"
    assert np.all(np.isfinite(h["pred_mean"][bo])) and np.all((h["pof"][bo] >= 0) & (h["pof"][bo] <= 1))
    s = json.loads((d / "summary.json").read_text())
    assert s["n_evals"] == n and np.isclose(s["best_y"], opt.best()[1])


def test_load_models_gives_back_the_same_gp(con_run):
    """opt.models[k] chose evaluation k; the saved copy predicts the same."""
    opt, d = con_run
    k = len(opt.y) - 1
    f, (c,), pref = load_models(d, k)
    assert pref is None
    Xq = np.random.default_rng(4).uniform(-2, 2, (15, 2))
    assert len(f.train_y) == k
    assert np.allclose(f.predict(Xq)[0], opt.models[k].objective.predict(Xq)[0], atol=1e-6)
    assert np.allclose(c.predict(Xq)[1], opt.models[k].constraints[0].predict(Xq)[1], atol=1e-6)


def test_models_are_fitted_once_per_step(con_run):
    """The acquisition of a past step (e.g. for plot_acquisition) reuses the models the loop fitted then."""
    opt, _ = con_run
    assert sorted(opt.models) == list(range(opt.n_init, len(opt.y)))        # one set per BO step
    k = opt.n_init + 2
    assert opt.acquisition(k).model is opt.models[k].objective.model


def test_random_strategy_stays_in_the_box_and_saves_no_models(tmp_path):
    opt = FastOptimizer(bowl, BOUNDS, strategy="random", seed=0, run_dir=tmp_path).run(20)
    assert np.all((opt.X >= -2) & (opt.X <= 2)) and not list((tmp_path / "models").iterdir())


def test_same_seed_same_run():
    first = FastOptimizer(bowl, BOUNDS, seed=3).run(3)
    second = FastOptimizer(bowl, BOUNDS, seed=3).run(3)
    assert np.allclose(first.X, second.X)


#pairwise preferences
@pytest.mark.parametrize("link", ["probit", "logit"])
def test_prob_better_matches_sampling_and_is_one_half_against_itself(link):
    from BOproj.benchmarks import PairwiseLabeler, u_polynomial
    from BOproj.gp import PreferenceGP
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (30, 2))
    first = rng.integers(0, 30, 60)
    second = (first + rng.integers(1, 30, 60)) % 30
    wins = PairwiseLabeler(u_polynomial)(X[first], X[second])
    gp = PreferenceGP(bounds=BOUNDS, link=link).fit(X[first], X[second], wins)
    reference = np.array([1.0, 1.0])
    X_test = rng.uniform(-2, 2, (6, 2))
    closed_form = gp.prob_better_torch(torch.tensor(X_test), reference).detach().numpy()
    draws = gp.sample(np.vstack([X_test, reference]), n=4000, seed=0)       # joint draws of u at the 6 designs and x_ref
    diff = draws[:, :6] - draws[:, 6:]
    sampled = (norm.cdf(diff / np.sqrt(2)) if link == "probit" else 1 / (1 + np.exp(-diff))).mean(0)
    assert np.allclose(closed_form, sampled, atol=0.05)
    assert gp.prob_better_torch(torch.tensor(reference[None]), reference).item() == pytest.approx(0.5)


def test_the_dm_is_only_asked_the_pairs_of_x_pref_and_the_model_is_fitted_once(tmp_path):
    from BOproj.benchmarks import PairwiseLabeler
    opt = FastOptimizer(bowl, BOUNDS, comparer=PairwiseLabeler(u_linear),
                        X_pref=initial_design(BOUNDS, 10, "sobol", seed=1), run_dir=tmp_path).run(3)
    assert len(opt.pairs_a) == 10 and len(opt.preference_gp.comparisons) == 10
    assert all(models.preference is opt.preference_gp for models in opt.models.values())
    h = np.genfromtxt(tmp_path / "history.csv", delimiter=",", names=True, dtype=None, encoding=None)
    assert np.all((h["pi"][opt.n_init:] > 0) & (h["pi"][opt.n_init:] < 1))
    assert np.load(tmp_path / "data.npz")["pairs_a"].shape == (10, 2)


def test_pairwise_prob_torch_is_a_probability_with_gradients():
    from BOproj.benchmarks import PairwiseLabeler, u_linear
    from BOproj.gp import PreferenceGP
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (20, 2))
    first = rng.integers(0, 20, 40)
    second = (first + rng.integers(1, 20, 40)) % 20
    wins = PairwiseLabeler(u_linear)(X[first], X[second])
    gp = PreferenceGP(bounds=BOUNDS, link="logit").fit(X[first], X[second], wins)
    X_test = torch.tensor(rng.uniform(-2, 2, (5, 1, 2)), requires_grad=True)
    prob = gp.prob_torch(X_test)
    assert prob.shape == (5, 1) and ((prob > 0) & (prob < 1)).all()
    prob.sum().backward()
    assert torch.isfinite(X_test.grad).all()


def test_pairwise_run_is_reproducible_whatever_the_global_random_state():
    """PairwiseGP draws from np.random and BoTorch's fit retries draw from torch: the loop seeds both per step."""
    from BOproj.benchmarks import PairwiseLabeler
    runs = []
    for global_seed in (1, 2):
        torch.manual_seed(global_seed)
        np.random.seed(global_seed)
        runs.append(FastOptimizer(bowl, BOUNDS, comparer=PairwiseLabeler(u_linear), seed=3,
                                  X_pref=initial_design(BOUNDS, 10, "sobol", seed=1)).run(4).X)
    assert np.array_equal(runs[0], runs[1])
