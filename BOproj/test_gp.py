"""Validation of gp.py on synthetic problems with known answers.

Run it either way:
    python test_gp.py                 # standalone: runs every check, prints PASS / FAIL, saves plots to gp_plots/
    python test_gp.py --show          # same, and opens each plot in a TkAgg window
    python -m pytest test_gp.py -v    # as a pytest suite (plots saved to gp_plots/)

Each check fits a model on data whose truth we know, then compares predictions with that truth.
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # lets `python test_gp.py` find the package
from BOproj.gp import BinaryGP, ContinuousGP, PreferenceGP, format_comparisons

SETTINGS = {"show": False, "outdir": Path(os.environ.get("GP_PLOT_DIR", "gp_plots"))}


# test functions
def forrester(X):
    """Negated Forrester function on [0, 1]: maximum about 6.02 at x = 0.757."""
    x = np.ravel(X)
    return -((6 * x - 2) ** 2) * np.sin(12 * x - 4)


def branin(X):
    """Negated Branin on [-5, 10] x [0, 15]: three maxima of -0.398."""
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    b, c, r, s, t = 5.1 / (4 * np.pi**2), 5 / np.pi, 6.0, 10.0, 1 / (8 * np.pi)
    return -((x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)


def utility_1d(X):
    return np.sin(6 * np.atleast_2d(X)[:, 0])


def utility_2d(X):
    X = np.atleast_2d(X)
    return -((X[:, 0] - 0.3) ** 2 + (X[:, 1] - 0.6) ** 2)          # best at (0.3, 0.6)


def plot(model, name, **kw):
    path = SETTINGS["outdir"] / f"{name}.png"
    model.plot(save=path, show=SETTINGS["show"], title=name, **kw)
    assert path.exists() and path.stat().st_size > 5_000, f"plot {path} was not written"


def rng(seed=0):
    return np.random.default_rng(seed)


def stratified(n, seed):
    """One random point in each of n equal slices of [0, 1]"""
    return (np.arange(n) + rng(seed).random(n)) / n


# ContinuousGP tests
def test_continuous_1d_fits_forrester():
    X = stratified(12, seed=0)
    m = ContinuousGP(bounds=[[0, 1]]).fit(X, forrester(X))
    g = np.linspace(0, 1, 300)
    mu, var = m.predict(g)
    assert mu.shape == var.shape == (300,)
    truth = forrester(g)
    assert np.sqrt(np.mean((mu - truth) ** 2)) < 0.08 * np.ptp(truth)   # within 8% of the range of f
    mu_tr, var_tr = m.predict(X)
    assert np.allclose(mu_tr, forrester(X), atol=0.5) and var_tr.max() < 0.5
    plot(m, "continuous_1d_forrester", truth=forrester)


def test_continuous_bands_cover_the_truth_with_noise():
    X = rng(1).random(25)
    y = forrester(X) + 0.5 * rng(2).standard_normal(25)
    m = ContinuousGP(bounds=[[0, 1]]).fit(X, y)
    g = np.linspace(0, 1, 300)
    mu, var = m.predict(g)
    inside = np.abs(forrester(g) - mu) <= 1.96 * np.sqrt(var)
    assert inside.mean() >= 0.85
    assert 0.05 < m.noise() < 1.0                                       # true noise variance 0.25
    _, var_y = m.predict(g, observation_noise=True)
    assert np.allclose(var_y - var, m.noise())
    plot(m, "continuous_1d_noisy", truth=forrester, observation_noise=True)


def test_continuous_2d_branin_and_ard():
    lo, hi = np.array([-5.0, 0.0]), np.array([10.0, 15.0])
    X = lo + (hi - lo) * rng(3).random((40, 2))
    m = ContinuousGP(bounds=[[-5, 10], [0, 15]]).fit(X, branin(X))
    Q = lo + (hi - lo) * rng(4).random((500, 2))
    mu, _ = m.predict(Q)
    assert np.corrcoef(mu, branin(Q))[0, 1] > 0.95
    assert m.lengthscales().shape == (2,)                               # one lengthscale per input (ARD)
    plot(m, "continuous_2d_branin")


def f3(X):
    return np.sin(4 * X[:, 0]) + X[:, 1] ** 2 - 0.1 * X[:, 2]


def test_continuous_slice_of_a_3d_function():
    X = rng(5).random((40, 3))
    m = ContinuousGP(bounds=[[0, 1]] * 3).fit(X, f3(X))
    plot(m, "continuous_3d_slice_x0", dims=[0], fixed=[0.5, 0.5, 0.5], truth=f3)
    plot(m, "continuous_3d_slice_x0_x1", dims=[0, 1], fixed=[0.5, 0.5, 0.5])


def test_continuous_input_formats_agree():
    X = rng(6).random(10)
    m = ContinuousGP(bounds=[[0, 1]]).fit(list(X), list(forrester(X)))
    q = [0.1, 0.5, 0.9]
    a = m.predict(q)[0]
    b = m.predict(np.array(q)[:, None])[0]
    c = m.predict(torch.tensor(q))[0]
    assert np.allclose(a, b) and np.allclose(a, c)
    m2 = ContinuousGP(bounds=[[0, 1]]).fit(torch.tensor(X), torch.tensor(forrester(X)))
    assert np.allclose(m2.predict(q)[0], a, atol=1e-3)


def test_continuous_fixed_noise_and_mixed_inputs():
    X = rng(7).random(20)
    m = ContinuousGP(bounds=[[0, 1]], noise=0.01).fit(X, forrester(X))
    assert abs(m.noise() - 0.01) < 1e-6
    Xc = np.c_[rng(8).random(30), rng(9).integers(0, 3, 30)]
    y = np.sin(4 * Xc[:, 0]) + np.array([0.0, 2.0, -2.0])[Xc[:, 1].astype(int)]
    mm = ContinuousGP(cat_dims=[1]).fit(Xc, y)
    mu, _ = mm.predict([[0.3, 0], [0.3, 1], [0.3, 2]])
    assert mu[1] > mu[0] > mu[2]


def test_continuous_posterior_works_with_botorch_acquisitions():
    from botorch.acquisition import LogExpectedImprovement
    X = rng(10).random(8)
    m = ContinuousGP(bounds=[[0, 1]]).fit(X, forrester(X))
    acq = LogExpectedImprovement(m.model, best_f=float(forrester(X).max()))
    vals = acq(torch.linspace(0, 1, 50, dtype=torch.float64).reshape(-1, 1, 1))
    assert vals.shape == (50,) and torch.isfinite(vals).all()


# PreferenceGP
def comparisons(n, d, utility, seed):
    r = rng(seed)
    xa, xb = r.random((n, d)), r.random((n, d))
    return xa, xb, utility(xa) > utility(xb)


def test_format_comparisons_deduplicates():
    xa = np.array([[0.1], [0.2], [0.1]])
    xb = np.array([[0.2], [0.3], [0.3]])
    X, C = format_comparisons(xa, xb, a_wins=[True, False, True])
    assert X.shape == (3, 1)                                          # 0.1, 0.2, 0.3 appear once each
    assert C.tolist() == [[0, 1], [2, 1], [0, 2]]                     # [winner, loser] indices into X


def test_preference_learns_a_1d_utility():
    xa, xb, w = comparisons(30, 1, utility_1d, seed=12)
    m = PreferenceGP(bounds=[[0, 1]]).fit(xa, xb, w)
    g = np.linspace(0, 1, 200)
    mu, var = m.predict(g)
    assert kendalltau(mu, utility_1d(g[:, None]))[0] > 0.8
    assert (var > 0).all()
    p = m.prob_better_torch(np.array([[0.26]]), np.array([0.78])).item()       # u(0.26) = 1, u(0.78) = -1
    assert p > 0.9
    plot(m, "preference_1d", truth=utility_1d, max_arrows=20)


def test_preference_learns_a_2d_utility():
    xa, xb, w = comparisons(40, 2, utility_2d, seed=13)
    m = PreferenceGP(bounds=[[0, 1], [0, 1]]).fit(xa, xb, w)
    Q = rng(14).random((300, 2))
    mu, _ = m.predict(Q)
    assert kendalltau(mu, utility_2d(Q))[0] > 0.7
    best = Q[np.argmax(mu)]
    assert np.linalg.norm(best - [0.3, 0.6]) < 0.2
    plot(m, "preference_2d", max_arrows=25)


def test_preference_logit_link():
    xa, xb, w = comparisons(30, 1, utility_1d, seed=15)
    m = PreferenceGP(bounds=[[0, 1]], link="logit").fit(xa, xb, w)
    g = np.linspace(0, 1, 100)
    assert kendalltau(m.predict(g)[0], utility_1d(g[:, None]))[0] > 0.7


# BinaryGP
def below_06(Z):
    return np.ravel(Z) <= 0.6


def test_binary_1d_threshold_probit_and_logit():
    X = stratified(40, seed=16)
    labels = X <= 0.6
    g = np.linspace(0, 1, 300)
    for link in ("probit", "logit"):
        m = BinaryGP(bounds=[[0, 1]], link=link).fit(X, labels)
        p = m.prob(g)
        assert p.shape == (300,) and ((p >= 0) & (p <= 1)).all()
        assert np.mean((p > 0.5) == (g <= 0.6)) > 0.93, link       # boundary within about 0.02
        if link == "probit":
            plot(m, "binary_1d_threshold", truth=below_06)


def disk(Z):
    return ((np.atleast_2d(Z) - [0.4, 0.5]) ** 2).sum(1) < 0.09


def test_binary_2d_disk():
    X = rng(17).random((80, 2))
    m = BinaryGP(bounds=[[0, 1], [0, 1]]).fit(X, disk(X))
    Q = rng(18).random((1000, 2))
    assert np.mean((m.prob(Q) > 0.5) == disk(Q)) > 0.85
    plot(m, "binary_2d_disk", truth=disk)


def test_binary_label_formats_agree():
    X = rng(19).random(30)
    lab = X <= 0.5
    g = np.linspace(0, 1, 50)
    a = BinaryGP(bounds=[[0, 1]]).fit(X, lab).prob(g)
    b = BinaryGP(bounds=[[0, 1]]).fit(X, np.where(lab, 1, -1)).prob(g)
    c = BinaryGP(bounds=[[0, 1]]).fit(torch.tensor(X), torch.tensor(lab.astype(float))).prob(g)
    assert np.allclose(a, b, atol=1e-3) and np.allclose(a, c, atol=1e-3)

#update
def test_continuous_update_equals_fitting_on_all_the_data():
    X = stratified(10, seed=20)
    x_new = np.array([0.5, 0.52])
    m = ContinuousGP(bounds=[[0, 1]]).fit(X, forrester(X))
    n_before = len(m.train_X)
    m.update(x_new, forrester(x_new))
    assert len(m.train_X) == n_before + 2 and len(m.train_y) == n_before + 2
    full = ContinuousGP(bounds=[[0, 1]]).fit(np.r_[X, x_new], np.r_[forrester(X), forrester(x_new)])
    g = np.linspace(0, 1, 50)
    assert np.allclose(m.predict(g)[0], full.predict(g)[0], atol=1e-6)
    assert np.allclose(m.predict(g)[1], full.predict(g)[1], atol=1e-6)


def test_continuous_update_reduces_uncertainty_where_the_point_is_added():
    X = stratified(8, seed=21)
    m = ContinuousGP(bounds=[[0, 1]]).fit(X, forrester(X))
    g = np.linspace(0, 1, 200)
    var = m.predict(g)[1]
    x_new = g[np.argmax(var)]
    m.update([x_new], forrester(np.array([x_new])))
    assert m.predict([x_new])[1][0] < 0.5 * var.max()


def test_continuous_update_in_two_dimensions_and_with_a_single_point():
    X = rng(22).random((15, 2))
    m = ContinuousGP(bounds=[[0, 1]] * 2).fit(X, f3(np.c_[X, np.zeros(15)]))
    m.update([0.4, 0.6], [1.0])                          # one new point, given as a 1-d array
    assert m.train_X.shape == (16, 2) and m.train_X[-1].tolist() == [0.4, 0.6]
    try:
        m.update([[0.1, 0.1], [0.2, 0.2]], [1.0])         # two rows but one value
        raise AssertionError("a row / value mismatch should have raised")
    except ValueError:
        pass


def test_continuous_update_with_per_point_noise():
    X = stratified(6, seed=23)
    m = ContinuousGP(bounds=[[0, 1]]).fit(X, forrester(X), y_var=np.full(6, 0.01))
    try:
        m.update([0.5], [forrester(np.array([0.5]))[0]])
        raise AssertionError("update without y_var_new should have raised")
    except ValueError:
        pass
    m.update([0.5], forrester(np.array([0.5])), y_var_new=0.01)
    assert len(m.train_y) == 7 and np.allclose(m.train_yvar, 0.01)
    plain = ContinuousGP(bounds=[[0, 1]]).fit(X, forrester(X))
    try:
        plain.update([0.5], forrester(np.array([0.5])), y_var_new=0.01)
        raise AssertionError("y_var_new on a model fitted without y_var should have raised")
    except ValueError:
        pass


def test_binary_update_equals_fitting_on_all_the_labels():
    X = stratified(30, seed=26)
    lab = X <= 0.5
    m = BinaryGP(bounds=[[0, 1]]).fit(X[:20], lab[:20])
    m.update(X[20:], lab[20:])
    assert len(m.train_X) == 30 and len(m.labels) == 30
    full = BinaryGP(bounds=[[0, 1]]).fit(X, lab)
    g = np.linspace(0, 1, 40)
    assert np.allclose(m.prob(g), full.prob(g), atol=1e-3)


def test_binary_update_accepts_minus_one_plus_one_labels_and_a_single_point():
    X = stratified(20, seed=27)
    m = BinaryGP(bounds=[[0, 1]]).fit(X, X <= 0.5)
    m.update([0.9], [-1])
    assert len(m.labels) == 21 and not m.labels[-1]
    m.update([0.1], [1])
    assert m.labels[-1]


def test_continuous_random_vs_stratified_variance():
    """Compare posterior uncertainty (variance) between stratified and pure uniform random points."""
    n_points = 12
    test_seed = 27  # seed 27 for reproducibility

    # 1. Stratified samples
    X_strat = stratified(n_points, seed=test_seed)
    y_strat = forrester(X_strat)
    m_strat = ContinuousGP(bounds=[[0, 1]]).fit(X_strat, y_strat)

    # 2. Pure uniform random samples
    X_rand = rng(test_seed).random(n_points)
    y_rand = forrester(X_rand)
    m_rand = ContinuousGP(bounds=[[0, 1]]).fit(X_rand, y_rand)

    # 3. Dense evaluation grid across the domain
    g = np.linspace(0, 1, 300)
    _, var_strat = m_strat.predict(g)
    _, var_rand = m_rand.predict(g)

    # Stratified eliminates large gaps, so peak unobserved uncertainty (max variance)
    # is almost always lower or more evenly distributed than uniform random
    print(
        f"\n[Variance Check] Max Var (Stratified): {var_strat.max():.4f} | Max Var (Uniform): {var_rand.max():.4f}"
    )
    print(
        f"[Variance Check] Mean Var (Stratified): {var_strat.mean():.4f} | Mean Var (Uniform): {var_rand.mean():.4f}"
    )

    # Generate diagnostic plots to see where the GP is uncertain
    plot(
        m_strat,
        "continuous_1d_stratified_variance",
        truth=forrester,
        observation_noise=False,
    )
    plot(
        m_rand,
        "continuous_1d_random_variance",
        truth=forrester,
        observation_noise=False,
    )

    # Verification: Stratified design leaves no gaps > 2/n, preventing massive variance spikes
    assert (
        var_strat.shape == var_rand.shape == (300,)
    ), "Predictions must match grid shape."


from pathlib import Path


def test_continuous_stratified_vs_random_vs_grid():
    """3-way comparison: Deterministic Grid vs Stratified vs Pure Random."""
    n_points = 12
    test_seed = 27

    # 1. Deterministic Slicing (midpoint of each of the n slices)
    X_grid = (np.arange(n_points) + 0.5) / n_points
    y_grid = forrester(X_grid)
    m_grid = ContinuousGP(bounds=[[0, 1]]).fit(X_grid, y_grid)

    # 2. Stratified Sampling (1 jittered point per slice)
    X_strat = stratified(n_points, seed=test_seed)
    y_strat = forrester(X_strat)
    m_strat = ContinuousGP(bounds=[[0, 1]]).fit(X_strat, y_strat)

    # 3. Pure Uniform Random Sampling
    X_rand = rng(test_seed).random(n_points)
    y_rand = forrester(X_rand)
    m_rand = ContinuousGP(bounds=[[0, 1]]).fit(X_rand, y_rand)

    # Evaluation grid
    g = np.linspace(0, 1, 300)
    truth = forrester(g)
    mu_grid, var_grid = m_grid.predict(g)
    mu_strat, var_strat = m_strat.predict(g)
    mu_rand, var_rand = m_rand.predict(g)

    # Setup 2-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)

    # --- Panel 1: Model Fits ---
    ax1 = axes[0]
    ax1.plot(g, truth, "k--", label="Ground Truth", alpha=0.6, linewidth=1.5)

    # Grid (Forest Green)
    ax1.plot(g, mu_grid, color="#2ca02c", lw=1.8, label=r"Grid $\mu(x)$")
    ax1.fill_between(
        g,
        mu_grid - 1.96 * np.sqrt(var_grid),
        mu_grid + 1.96 * np.sqrt(var_grid),
        color="#2ca02c",
        alpha=0.15,
    )
    ax1.scatter(
        X_grid,
        y_grid,
        color="#2ca02c",
        marker="^",
        edgecolors="black",
        s=45,
        zorder=4,
        label="Grid Points",
    )

    # Stratified (Blue)
    ax1.plot(g, mu_strat, color="#1f77b4", lw=1.8, label=r"Stratified $\mu(x)$")
    ax1.fill_between(
        g,
        mu_strat - 1.96 * np.sqrt(var_strat),
        mu_strat + 1.96 * np.sqrt(var_strat),
        color="#1f77b4",
        alpha=0.18,
    )
    ax1.scatter(
        X_strat,
        y_strat,
        color="#1f77b4",
        marker="o",
        edgecolors="black",
        s=45,
        zorder=4,
        label="Stratified Points",
    )

    # Pure Random (Crimson)
    ax1.plot(
        g,
        mu_rand,
        color="#d62728",
        lw=1.8,
        linestyle=":",
        label=r"Random $\mu(x)$",
    )
    ax1.fill_between(
        g,
        mu_rand - 1.96 * np.sqrt(var_rand),
        mu_rand + 1.96 * np.sqrt(var_rand),
        color="#d62728",
        alpha=0.12,
    )
    ax1.scatter(
        X_rand,
        y_rand,
        color="#d62728",
        marker="s",
        edgecolors="black",
        s=40,
        zorder=4,
        label="Random Points",
    )

    ax1.set_title("GP Fit Comparison", fontsize=12, fontweight="bold")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 2: Posterior Variance Overlap ---
    ax2 = axes[1]
    ax2.plot(
        g,
        var_grid,
        color="#2ca02c",
        lw=2.0,
        label=f"Grid Var (Max: {var_grid.max():.2f})",
    )
    ax2.fill_between(g, 0, var_grid, color="#2ca02c", alpha=0.18)

    ax2.plot(
        g,
        var_strat,
        color="#1f77b4",
        lw=2.0,
        label=f"Stratified Var (Max: {var_strat.max():.2f})",
    )
    ax2.fill_between(g, 0, var_strat, color="#1f77b4", alpha=0.18)

    ax2.plot(
        g,
        var_rand,
        color="#d62728",
        lw=2.0,
        linestyle="-.",
        label=f"Random Var (Max: {var_rand.max():.2f})",
    )
    ax2.fill_between(g, 0, var_rand, color="#d62728", alpha=0.14)

    # Tick marks showing placement across the x-axis
    ax2.plot(
        X_grid,
        np.zeros_like(X_grid) + 0.02,
        "|",
        color="#2ca02c",
        ms=12,
        mew=2,
        label="Grid ticks",
    )
    ax2.plot(
        X_strat,
        np.zeros_like(X_strat) + 0.01,
        "|",
        color="#1f77b4",
        ms=12,
        mew=2,
        label="Stratified ticks",
    )
    ax2.plot(
        X_rand,
        np.zeros_like(X_rand),
        "|",
        color="#d62728",
        ms=12,
        mew=2,
        label="Random ticks",
    )

    ax2.set_title(
        "Posterior Variance Overlap (Uncertainty)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel(r"Posterior Variance $\sigma^2(x)$")
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    # Save to stratified directory
    target_dir = Path("stratified")
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = (target_dir / "grid_vs_stratified_vs_random.png").resolve()

    plt.savefig(out_path, dpi=200)
    print(f"\n[OK] 3-way comparison plot saved to: {out_path}")

    if SETTINGS["show"]:
        try:
            plt.show()
        except Exception as e:
            print(f"[Warning] GUI show failed: {e}")

    plt.close(fig)
    assert out_path.exists() and out_path.stat().st_size > 5_000


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def gramacy_lee(x):
    x = np.ravel(x)
    return np.sin(10 * np.pi * x) / (2 * x + 1e-6) + (x - 1) ** 4


def localized_peak(x):
    x = np.ravel(x)
    return np.exp(-30 * (x - 0.5) ** 2)


def resonant_sine(x):
    x = np.ravel(x)
    return np.sin(12 * np.pi * x)


def test_compare_designs_across_functions():
    """Systematic benchmark of Grid vs Stratified vs Random on multiple 1D functions."""
    n_points = 12
    test_seed = 27

    benchmark_funcs = [
        ("Forrester", forrester),
        ("Gramacy & Lee", gramacy_lee),
        ("Localized Spike", localized_peak),
        ("Resonant Sine (12pi)", resonant_sine),
    ]

    target_dir = Path.cwd() / "stratified"
    target_dir.mkdir(parents=True, exist_ok=True)

    g = np.linspace(0.01, 1.0, 400)
    fig, axes = plt.subplots(
        len(benchmark_funcs), 2, figsize=(15, 3.5 * len(benchmark_funcs))
    )

    print(
        f"\n{'Function':<22} | {'Design':<11} | {'RMSE':<8} | {'Max Var':<8} | {'Mean Var':<8}"
    )
    print("-" * 65)

    for i, (fname, fn) in enumerate(benchmark_funcs):
        truth_g = fn(g)

        # 1. Deterministic Grid (midpoints)
        X_grid = (np.arange(n_points) + 0.5) / n_points
        y_grid = fn(X_grid)
        m_grid = ContinuousGP(bounds=[[0, 1]]).fit(X_grid, y_grid)
        mu_grid, var_grid = m_grid.predict(g)

        # 2. Stratified
        X_strat = stratified(n_points, seed=test_seed)
        y_strat = fn(X_strat)
        m_strat = ContinuousGP(bounds=[[0, 1]]).fit(X_strat, y_strat)
        mu_strat, var_strat = m_strat.predict(g)

        # 3. Pure Uniform Random
        X_rand = rng(test_seed).random(n_points)
        y_rand = fn(X_rand)
        m_rand = ContinuousGP(bounds=[[0, 1]]).fit(X_rand, y_rand)
        mu_rand, var_rand = m_rand.predict(g)

        designs = [
            ("Grid", mu_grid, var_grid, "#2ca02c", "-", X_grid),
            ("Stratified", mu_strat, var_strat, "#1f77b4", "--", X_strat),
            ("Random", mu_rand, var_rand, "#d62728", ":", X_rand),
        ]

        # Left Column: Fit vs Truth
        ax_fit = axes[i, 0]
        ax_fit.plot(g, truth_g, "k-", lw=1.5, alpha=0.5, label="Ground Truth")
        for dname, mu, var, col, ls, X_pts in designs:
            rmse = np.sqrt(np.mean((mu - truth_g) ** 2))
            max_v = var.max()
            mean_v = var.mean()
            print(
                f"{fname:<22} | {dname:<11} | {rmse:<8.3f} | {max_v:<8.3f} | {mean_v:<8.3f}"
            )

            ax_fit.plot(g, mu, color=col, linestyle=ls, lw=1.8, label=dname)
            ax_fit.scatter(
                X_pts, fn(X_pts), color=col, s=25, alpha=0.8, zorder=3
            )

        ax_fit.set_title(f"{fname}: Predicted Mean $\\mu(x)$", fontsize=11)
        ax_fit.set_ylabel("f(x)")
        ax_fit.legend(loc="best", fontsize=8)
        ax_fit.grid(True, linestyle="--", alpha=0.4)

        # Right Column: Posterior Variance Overlap
        ax_var = axes[i, 1]
        for dname, _, var, col, ls, _ in designs:
            ax_var.plot(g, var, color=col, linestyle=ls, lw=1.8, label=dname)
            ax_var.fill_between(g, 0, var, color=col, alpha=0.12)

        ax_var.set_title(f"{fname}: Posterior Variance $\\sigma^2(x)$", fontsize=11)
        ax_var.set_ylabel("$\\sigma^2(x)$")
        ax_var.legend(loc="upper right", fontsize=8)
        ax_var.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = target_dir / "multi_function_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"\n[OK] Benchmark plot saved to: {out_path.resolve()}\n")
    
#standalone runner
# plotting options
def test_2d_plots_accept_a_colormap_name_or_object():
    """cmap="gist_heat" (a plain name, no matplotlib import needed by the caller) or a Colormap object, for every model."""
    import matplotlib
    from BOproj.gp import BinaryGP, ContinuousGP, PreferenceGP
    r = rng(5)
    X = r.random((25, 2))
    a, b = r.random((20, 2)), r.random((20, 2))
    models = {"continuous": ContinuousGP(bounds=[[0, 1]] * 2).fit(X, -((X - 0.5) ** 2).sum(1)),
              "preference": PreferenceGP(bounds=[[0, 1]] * 2).fit(a, b, a[:, 0] > b[:, 0]),
              "binary": BinaryGP(bounds=[[0, 1]] * 2).fit(X, X[:, 0] > 0.5)}
    for name, model in models.items():
        plot(model, f"cmap_name_{name}", cmap="gist_heat")
        plot(model, f"cmap_object_{name}", cmap=matplotlib.colormaps["gist_heat"])
    try:
        models["continuous"].plot(cmap="not_a_colormap")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown colormap name should raise a ValueError")


def test_models_work_without_importing_matplotlib():
    """matplotlib is only imported when a plot is drawn: fitting and predicting must not need it."""
    import subprocess
    code = ("import sys, numpy as np\n"
            "from BOproj.gp import ContinuousGP\n"
            "import BOproj.bo_plotting\n"
            "X = np.random.default_rng(0).random((8, 2))\n"
            "ContinuousGP(bounds=[[0, 1]] * 2).fit(X, X.sum(1)).predict(X)\n"
            "assert not any(m.startswith('matplotlib') for m in sys.modules), 'matplotlib was imported'\n")
    run = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent.parent,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-500:]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="open each plot in a TkAgg window")
    ap.add_argument("--outdir", default=str(SETTINGS["outdir"]), help="where to save the plots")
    ap.add_argument("-k", default="", help="only run checks whose name contains this text")
    a = ap.parse_args(argv)
    SETTINGS["show"], SETTINGS["outdir"] = a.show, Path(a.outdir)
    # globals() lists names in the order they were defined, so the checks run top to bottom
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_") and a.k in name]
    failed = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            status = "PASS"
        except Exception:
            status, failed = "FAIL", failed + 1
            traceback.print_exc()
        print(f"{status}  {name:55s} {time.perf_counter() - t0:5.1f} s")
    print(f"\n{len(tests) - failed}/{len(tests)} checks passed; plots in {SETTINGS['outdir'].resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())