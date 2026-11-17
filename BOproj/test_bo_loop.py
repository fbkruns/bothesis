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


@pytest.mark.parametrize("score", ["probability", "copeland"])
def test_recommended_is_improvement_times_preference_score_and_best_is_untouched(score):
    from BOproj.benchmarks import PairwiseLabeler
    plain = FastOptimizer(bowl, BOUNDS, seed=0).run(3)
    assert plain.preference_scores(plain.X) is None
    assert np.array_equal(plain.recommended()[0], plain.best()[0])      # no preference model: the plain best

    opt = FastOptimizer(bowl, BOUNDS, comparer=PairwiseLabeler(u_linear), seed=0, preference_score=score, n_landmarks=16,
                        X_pref=initial_design(BOUNDS, 10, "sobol", seed=1)).run(3)
    scores = opt.preference_scores(opt.X)
    assert scores.shape == (len(opt.y),) and np.all((scores >= 0) & (scores <= 1))
    weighted = (opt.y - opt.y.min()) * scores                           # bowl has no constraints: every evaluation counts
    _, recommended_y = opt.recommended()
    assert np.isclose(recommended_y, opt.y[np.argmax(weighted)])
    assert np.isclose(opt.best()[1], opt.y.max())                       # the incumbent is still the highest y


#known constraints (exact, no GP)


def known_cut(X):
    """Known constraint, a torch function of an (n, 2) tensor: feasible where x1 <= 0.  The maximum of bowl, at x1 = 0.5, is cut off."""
    return torch.as_tensor(X, dtype=torch.float64).reshape(-1, 2)[:, 0]


def values(constraint, X):
    from BOproj.acquisition import known_constraint_values
    return known_constraint_values(constraint, X)


def test_initial_design_with_accept_keeps_only_accepted_designs_and_is_unchanged_without_it():
    plain = initial_design(BOUNDS, 8, "sobol", seed=3)
    assert np.array_equal(initial_design(BOUNDS, 8, "sobol", seed=3, accept=None), plain)
    kept = initial_design(BOUNDS, 8, "sobol", seed=3, accept=lambda X: values(known_cut, X) <= 0)
    assert kept.shape == (8, 2) and np.all(kept[:, 0] <= 0)
    with pytest.raises(ValueError):
        initial_design(BOUNDS, 4, "random", seed=0, accept=lambda X: np.zeros(len(X), bool))


def test_optimize_with_known_constraints_returns_a_known_feasible_point_that_beats_random_feasible_ones():
    from BOproj.acquisition import known_feasible, optimize
    X, f_gp, _ = fitted()
    acquisition = LogEIProduct(f_gp, best_f=float(bowl(X).max()))
    x, value = optimize(acquisition, BOUNDS, 2, known_constraints=[known_cut])
    assert values(known_cut, x)[0] <= 0                                 # exactly feasible, also when the optimum is on the edge
    rng = np.random.default_rng(0)
    random_points = rng.uniform(-2, 2, (400, 2))
    random_points = random_points[known_feasible(random_points, [known_cut])]
    with torch.no_grad():
        random_values = acquisition(torch.as_tensor(random_points[:, None, :])).numpy()
    assert value >= random_values.max() - 1e-9
    with pytest.raises(ValueError):                                     # nothing is known-feasible
        optimize(acquisition, BOUNDS, 2, known_constraints=[lambda X: torch.ones(len(X), dtype=torch.float64)])


def test_optimize_with_a_curved_known_constraint_stays_inside_it():
    from BOproj.acquisition import optimize
    X, f_gp, _ = fitted()
    acquisition = LogEIProduct(f_gp, best_f=float(bowl(X).max()))

    def outside_disc(Z):                                                # feasible only OUTSIDE the unit disc around (0.5, -0.5)
        return 1.0 - (torch.as_tensor(Z, dtype=torch.float64) - torch.tensor([0.5, -0.5], dtype=torch.float64)).pow(2).sum(dim=1)

    for seed in range(3):
        x, _ = optimize(acquisition, BOUNDS, 2, seed=seed, known_constraints=[outside_disc])
        assert values(outside_disc, x)[0] <= 0


def test_known_constraints_are_exact_filters_with_no_gp_and_the_search_stays_inside(tmp_path):
    opt = FastOptimizer(bowl, BOUNDS, known_constraints=[known_cut], seed=0, run_dir=tmp_path).run(6)
    assert opt.m == 0 and opt.C.shape == (len(opt.y), 0)                 # no black-box constraint, nothing stored in C
    assert all(len(models.constraints) == 0 for models in opt.models.values())      # and no GP fitted for the known one
    assert np.all(values(known_cut, opt.X[opt.n_init:]) <= 0)            # every BO point (the initial design covers the whole box)
    assert values(known_cut, opt.best()[0])[0] <= 0
    assert json.loads((tmp_path / "config.json").read_text())["known_constraints"] == ["known_cut"]


def test_random_strategy_respects_known_constraints():
    opt = FastOptimizer(bowl, BOUNDS, known_constraints=[known_cut], strategy="random", seed=1).run(10)
    assert np.all(values(known_cut, opt.X[opt.n_init:]) <= 0)


def test_initial_design_outside_the_known_region_is_allowed_but_never_the_incumbent():
    opt = FastOptimizer(bowl, BOUNDS, known_constraints=[known_cut], seed=0).run(4)
    outside = values(known_cut, opt.X) > 0
    assert outside[:opt.n_init].any()                                    # the initial design does cover the whole box
    assert not np.any(outside[opt.n_init:])                              # the search stays inside
    assert not opt.feasible[outside].any()
    assert values(known_cut, opt.best()[0])[0] <= 0 and values(known_cut, opt.recommended()[0])[0] <= 0
    assert opt.best()[1] < opt.y[outside].max()                          # a better y outside did not become the incumbent


def test_no_initial_design_point_in_the_known_region_the_search_finds_it():
    start = initial_design(BOUNDS, 6, "sobol", seed=0)                   # what the optimiser evaluates first (n_init = 6, seed 0)
    corner = torch.tensor([-1.9, -1.9], dtype=torch.float64)
    radius = 0.5 * np.linalg.norm(start - corner.numpy(), axis=1).min()

    def tiny_disc(X):                                                    # known-feasible only near the corner, away from every initial point
        return (torch.as_tensor(X, dtype=torch.float64) - corner).pow(2).sum(dim=1) - radius ** 2

    assert np.all(values(tiny_disc, start) > 0)
    opt = FastOptimizer(bowl, BOUNDS, known_constraints=[tiny_disc], seed=0).run(3)
    assert opt.best(opt.n_init)[0] is None                               # nothing feasible after the initial design
    assert np.all(values(tiny_disc, opt.X[opt.n_init:]) <= 0)            # the search goes straight into the region
    assert opt.best()[0] is not None and values(tiny_disc, opt.best()[0])[0] <= 0
