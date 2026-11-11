"""Plots for the Bayesian optimiser (bo_loop.py).  

Common arguments, as in gp_plotting.py:
    dims    which two inputs to draw (maps); the others are held at `fixed` (default: the best design)
    ax      an existing matplotlib axis to draw into (plot_points only); the figure is then left open and untouched
    show    open an interactive TkAgg window;  save   path of a file to write;  title   figure title
    cmap    colour map of the maps: a matplotlib name such as "gist_heat" or a Colormap object (matplotlib is only
            imported when a plot is drawn)
"""

import numpy as np
import torch

from .gp_plotting import _figure, _finish, _use_backend
from .gp_utils import DTYPE


def _dims_fixed(opt, dims, fixed):
    dims = list(range(min(opt.d, 2))) if dims is None else [int(i) for i in np.atleast_1d(dims)]
    if fixed is None:
        best_x, _ = opt.best()
        fixed = best_x if best_x is not None else opt.bounds.mean(1)
    return dims, np.asarray(fixed, float).ravel()


def _grid(opt, dims, fixed, n):
    """(grid_x1, grid_x2, X_grid): an n x n grid over two inputs, the others held at `fixed`."""
    (low1, high1), (low2, high2) = opt.bounds[dims[0]], opt.bounds[dims[1]]
    grid_x1, grid_x2 = np.meshgrid(np.linspace(low1, high1, n), np.linspace(low2, high2, n))
    X_grid = np.tile(fixed, (n * n, 1))
    X_grid[:, dims[0]], X_grid[:, dims[1]] = grid_x1.ravel(), grid_x2.ravel()
    return grid_x1, grid_x2, X_grid


def _as_runs(runs):
    """BayesianOptimizer | list of them | {label: list} -> {label: list}."""
    if isinstance(runs, dict):
        return {label: list(v) if isinstance(v, (list, tuple)) else [v] for label, v in runs.items()}
    runs = list(runs) if isinstance(runs, (list, tuple)) else [runs]
    return {runs[0].strategy: runs}


#where it sampled
def plot_points(opt, dims=None, fixed=None, background=None, background_label="f(x)", n=150, ax=None, show=False,
                save=None, title=None, cmap=None, background_scale="linear"):
    """The evaluated designs over a map of `background` (a callable X -> values, e.g. the true objective).
    background_scale 'linear' (default), or 'log': log-spaced colours (a display choice only, the values are untouched;
    values below 1e-4 of the maximum share the lowest colour) so that a function spanning orders of magnitude keeps
    its contrast.  Backgrounds with negative values always use linear colours.

    initial design: white squares;  BO points: coloured by step;  infeasible: hollow.  Also drawn: constraint
    boundaries c_i = 0 (white dashed), the true boundary u = 0 when the comparer has a .utility (magenta dotted), the
    known optimum (black star) and the best design found (cyan X).  1-d problems: f against x."""
    _use_backend(show)
    fig, (axis,) = _figure(ax, size=(6.4, 5.2))
    X, n_init = opt.X, opt.n_init
    bo_steps = np.arange(len(X) - n_init)
    if opt.d == 1:
        if background is not None:
            x_line = np.linspace(*opt.bounds[0], 400)[:, None]
            axis.plot(x_line[:, 0], background(x_line), "k--", lw=1, label=background_label)
        axis.scatter(X[:n_init, 0], opt.y[:n_init], marker="s", c="w", edgecolors="k", label="initial design")
        scatter = axis.scatter(X[n_init:, 0], opt.y[n_init:], c=bo_steps, cmap="autumn", edgecolors="k", label="BO")
        fig.colorbar(scatter, ax=axis, label="BO step")
        axis.set(xlabel="x1", ylabel="y")
        axis.legend(fontsize=8)
        axis.set_title(title or f"{len(X)} evaluations ({n_init} initial)")
        return _finish(fig, show, save, keep_open=ax is not None)

    dims, fixed = _dims_fixed(opt, dims, fixed)
    grid_x1, grid_x2, X_grid = _grid(opt, dims, fixed, n)
    on_grid = lambda values: np.asarray(values).reshape(grid_x1.shape)
    if background is not None:
        values = on_grid(background(X_grid))
        levels, norm, extend = 30, None, "neither"
        if background_scale == "log" and values.min() >= 0 and values.max() > 0:
            from matplotlib.colors import LogNorm
            floor = max(values.min(), values.max() * 1e-4)
            levels, norm = np.geomspace(floor, values.max(), 30), LogNorm(floor, values.max())
            extend = "min" if values.min() < floor else "neither"
        contours = axis.contourf(grid_x1, grid_x2, values, levels=levels, norm=norm, extend=extend,
                                 cmap=("viridis_r" if cmap is None else cmap))
        if norm is None:
            fig.colorbar(contours, ax=axis, label=background_label)
        else:                                                              # ticks at powers of ten, plain numbers
            decades = 10.0 ** np.arange(np.ceil(np.log10(floor)), np.floor(np.log10(values.max())) + 1)
            bar = fig.colorbar(contours, ax=axis, label=background_label + " (log colours)", ticks=decades, format="%g")
            bar.minorticks_off()
    for i, constraint in enumerate(opt.constraints):
        axis.contour(grid_x1, grid_x2, on_grid(constraint(X_grid)), levels=[0], colors="w", linestyles="--",
                     linewidths=1.3)
        if i == 0:
            axis.plot([], [], "w--", label="constraint boundary")
    if hasattr(opt.comparer, "utility"):                                  # draw where the true utility is 0
        axis.contour(grid_x1, grid_x2, on_grid(opt.comparer.utility(X_grid)), levels=[0.0], colors="magenta",
                     linestyles=":", linewidths=1.8)
        axis.plot([], [], ":", color="magenta", label="boundary of the acceptable region (truth, u = 0)")
    points, feasible = X[:, dims], opt.feasible
    init_points, init_feasible = points[:n_init], feasible[:n_init]
    bo_points, bo_feasible = points[n_init:], feasible[n_init:]
    axis.scatter(*init_points[init_feasible].T, marker="s", c="w", edgecolors="k", s=35, label="initial design")
    axis.scatter(*init_points[~init_feasible].T, marker="s", facecolors="none", edgecolors="k", s=35)
    scatter = axis.scatter(*bo_points[bo_feasible].T, c=bo_steps[bo_feasible], cmap="autumn", vmin=0,
                           vmax=max(len(bo_steps) - 1, 1), edgecolors="k", s=35, label="BO points")
    axis.scatter(*bo_points[~bo_feasible].T, facecolors="none", edgecolors="k", s=35)
    if len(bo_steps):
        fig.colorbar(scatter, ax=axis, label="BO step")
    if opt.x_optimum is not None:
        axis.scatter(*opt.x_optimum[dims], marker="*", c="k", s=220, label="optimum")
    best_x, _ = opt.best()
    if best_x is not None:
        axis.scatter(*best_x[dims], marker="X", c="cyan", edgecolors="k", s=130, label="best found")
    axis.set(xlim=opt.bounds[dims[0]], ylim=opt.bounds[dims[1]], xlabel=f"x{dims[0] + 1}", ylabel=f"x{dims[1] + 1}")
    axis.set_aspect("equal")
    axis.legend(fontsize=7, loc="lower left")
    axis.set_title(title or f"{len(X)} evaluations ({n_init} initial)")   # on the axis: works inside subplot grids
    return _finish(fig, show, save, keep_open=ax is not None)


#convergence
def plot_convergence(runs, transform=None, ylabel="regret  (f* - best y)", show=False, save=None, title=None):
    """Regret and distance from the best design to the optimum, against the number of evaluations.

    runs       a BayesianOptimizer, a list of them (median and quartiles over seeds), or {label: list} to compare
    transform  callable applied to the regret before drawing (e.g. np.expm1 for a log-transformed objective)"""
    _use_backend(show)
    runs = _as_runs(runs)
    first = next(iter(runs.values()))[0]
    has_distance = first.x_optimum is not None
    fig, axes = _figure(None, 1, 2 if has_distance else 1, size=(6, 4))
    colours = ["tab:blue", "tab:gray", "tab:orange", "tab:green", "tab:purple", "tab:red"]
    for (label, optimizers), colour in zip(runs.items(), colours):
        for axis, column in zip(axes, ("regret", "dist_to_opt")):
            curves = np.array([opt.history(column) for opt in optimizers])     # one row per run
            if column == "regret" and transform is not None:
                curves = transform(curves)
            curves = np.maximum(curves, 1e-12)                                 # log scale: no zeros
            evals = np.arange(1, curves.shape[1] + 1)
            if len(curves) == 1:
                axis.plot(evals, curves[0], color=colour, label=label)
            else:
                axis.plot(evals, np.nanmedian(curves, 0), color=colour, label=f"{label}, median of {len(curves)}")
                axis.fill_between(evals, *np.nanpercentile(curves, [25, 75], 0), color=colour, alpha=0.2)
    for axis, axis_label in zip(axes, (ylabel, "|best x - x*|")):
        axis.axvline(first.n_init + 0.5, color="k", ls=":", lw=1, label="end of initial design")
        axis.set(yscale="log", xlabel="evaluations", ylabel=axis_label)
        axis.legend(fontsize=8)
    return _finish(fig, show, save, title)


#diagnostics
def plot_diagnostics(opt, show=False, save=None, title=None):
    """Six panels over the evaluations:
    y and the incumbent | GP prediction before evaluating vs the observed y (calibration) | standardised error
    (y - mean) / sd, which should mostly stay within +-2 | lengthscales of the objective GP | acquisition value,
    PoF and pi at the chosen point | time to fit the GPs and to optimise the acquisition."""
    _use_backend(show)
    fig, axes = _figure(None, 2, 3, size=(6.6, 4.8))       # roomy: six panels with their own legends
    evals = np.arange(len(opt.y))
    is_bo = np.array([record["phase"] == "bo" for record in opt.records])
    mean, sd = opt.history("pred_mean"), opt.history("pred_sd")

    axis = axes[0]
    axis.scatter(evals[~is_bo], opt.y[~is_bo], c="gray", s=15, label="initial / random")
    axis.scatter(evals[is_bo], opt.y[is_bo], c="tab:blue", s=15, label="BO")
    axis.scatter(evals[~opt.feasible], opt.y[~opt.feasible], marker="x", c="red", s=25, label="infeasible")
    axis.plot(evals, opt.history("best_y"), "k", lw=1.5, drawstyle="steps-post", label="incumbent")
    axis.set(xlabel="evaluation", ylabel="y", title="observations")
    axis.legend(fontsize=7)

    axis = axes[1]
    axis.errorbar(mean[is_bo], opt.y[is_bo], xerr=2 * sd[is_bo], fmt="o", ms=3, alpha=0.6, elinewidth=0.8)
    if is_bo.any():
        both = np.r_[mean[is_bo], opt.y[is_bo]]
        axis.plot([both.min(), both.max()], [both.min(), both.max()], "k--", lw=1)
    axis.set(xlabel="predicted mean (+- 2 sd)", ylabel="observed y", title="calibration")

    axis = axes[2]
    z = (opt.y - mean) / sd
    axis.axhspan(-2, 2, color="gray", alpha=0.2)
    axis.plot(evals[is_bo], z[is_bo], "o-", ms=3)
    axis.set(xlabel="evaluation", ylabel="(y - mean) / sd",
             title=f"{np.mean(np.abs(z[is_bo]) <= 2):.0%} within +-2" if is_bo.any() else "standardised error")

    axis = axes[3]
    for i in range(opt.d):
        axis.plot(evals[is_bo], opt.history(f"ls{i + 1}")[is_bo], "o-", ms=3, label=f"lengthscale x{i + 1}")
    axis.set(xlabel="evaluation", ylabel="normalised units", yscale="log", title="objective GP hyperparameters")
    axis.legend(fontsize=7)

    axis = axes[4]
    axis.plot(evals[is_bo], opt.history("acq_value")[is_bo], "o-", ms=3, c="k", label="log acquisition")
    axis.set(xlabel="evaluation", ylabel="log alpha(x_next)", title="acquisition at the chosen point")
    prob_axis = axis.twinx()
    if opt.m:
        prob_axis.plot(evals[is_bo], opt.history("pof")[is_bo], c="tab:orange", lw=1, label="PoF")
    if opt.comparer is not None:
        prob_axis.plot(evals[is_bo], opt.history("pi")[is_bo], c="tab:green", lw=1, label="pi")
    prob_axis.set_ylim(-0.05, 1.05)
    prob_axis.set_ylabel("probability")
    handles, labels = axis.get_legend_handles_labels()
    handles2, labels2 = prob_axis.get_legend_handles_labels()
    axis.legend(handles + handles2, labels + labels2, fontsize=7)

    axis = axes[5]
    t_fit, t_acq = opt.history("t_fit")[is_bo], opt.history("t_acq")[is_bo]
    axis.bar(evals[is_bo], t_fit, label="fit GPs")
    axis.bar(evals[is_bo], t_acq, bottom=t_fit, label="optimise acquisition")
    axis.set(xlabel="evaluation", ylabel="seconds", title="time per step")
    axis.legend(fontsize=7)
    return _finish(fig, show, save, title)


#acquisition maps
def plot_acquisition(opt, n=None, dims=None, fixed=None, grid=70, floor=20.0, show=False, save=None, title=None,
                     cmap=None):
    """Maps of the terms of log alpha = log EI + log PoF + log pi, and their sum, after the first n evaluations
    (default all: the map that will choose the next point), drawn with the models the loop fitted at that step.
    Each map is clipped at (its max - floor) so the interesting region is visible (in the corners log EI can be hundreds of units lower).
    Marked: the evaluations so far (white), the point the optimiser chose next if it is known (red star).
    Grid points are evaluated as separate q = 1 batches, so a fine grid costs no n x n covariance."""
    _use_backend(show)
    n = len(opt.y) if n is None else int(n)
    acq = opt.acquisition(n)
    dims, fixed = _dims_fixed(opt, dims, fixed)
    grid_x1, grid_x2, X_grid = _grid(opt, dims, fixed, grid)
    X_batch = torch.as_tensor(X_grid[:, None, :], dtype=DTYPE)
    with torch.no_grad():
        terms = {}
        if acq.log_ei is not None:
            terms["log EI"] = acq.log_ei(X_batch)
        if opt.m:
            terms["log PoF"] = acq.log_pof(X_batch)
        if opt.comparer is not None:
            terms["log pi"] = acq.log_pref(X_batch)
        if len(terms) != 1:                                      # with a single term the sum is the same picture
            terms["log alpha (sum)"] = acq(X_batch)
    fig, axes = _figure(None, 1, len(terms), size=(4.6, 4.2))
    data_points = opt.X[:n][:, dims]
    for axis, (name, values) in zip(axes, terms.items()):
        values = values.numpy().reshape(grid_x1.shape)
        values = np.maximum(values, values.max() - floor)
        contours = axis.contourf(grid_x1, grid_x2, values, levels=25, cmap=("magma" if cmap is None else cmap))
        fig.colorbar(contours, ax=axis)
        axis.scatter(*data_points.T, c="w", edgecolors="k", s=18)
        if n < len(opt.y):
            axis.scatter(*opt.X[n, dims], marker="*", c="red", edgecolors="w", s=220, label="chosen next")
        else:
            total = terms["log alpha (sum)"] if "log alpha (sum)" in terms else next(iter(terms.values()))
            best_on_grid = int(np.argmax(total.numpy()))
            axis.scatter(*X_grid[best_on_grid, dims], marker="*", c="red", edgecolors="w", s=220, label="grid argmax")
        axis.set(title=name, xlabel=f"x{dims[0] + 1}", ylabel=f"x{dims[1] + 1}")
        axis.set_aspect("equal")
    axes[0].legend(fontsize=7, loc="lower left")
    return _finish(fig, show, save, title or f"acquisition after {n} evaluations")
