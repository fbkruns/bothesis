"""Plots for the models in gp.py.  Each model's .plot(...) calls the matching function here.

Common arguments:
    dims    which inputs to plot (one or two); the others are held at `fixed`
    fixed   a full design whose values are used for the inputs not plotted (default: chosen per model)
    truth   optional callable X -> values, drawn dashed for comparison
    ax      an existing matplotlib axis to draw into (1-d plots only); the figure is then left open and untouched
    show    open an interactive TkAgg window (WSL/Linux needs: sudo apt install python3-tk)
    save    path of a file to write (png, pdf, svg, ...)
    title   figure title
    cmap    colour map of the 2-d maps: a matplotlib name such as "gist_heat" or a Colormap object (default: per plot).
            A name is enough, there is no need to import matplotlib yourself; matplotlib itself is only imported
            when a plot is drawn, so the models work without it.
"""

import sys
import warnings
from pathlib import Path

import numpy as np


#helpers
def _use_backend(show):
    """Without a window to open, draw off-screen (works without a display)."""
    import matplotlib
    if not show and "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")


def _figure(ax, nrows=1, ncols=1, size=(6.5, 4.2)):
    import matplotlib.pyplot as plt
    if ax is not None:
        return ax.figure, np.atleast_1d(ax)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False)
    return fig, axes.ravel()


def _finish(fig, show, save, title=None, dpi=150, keep_open=False):
    """Title, layout, save, show.  keep_open=True when the caller passed its own axis: then the figure is theirs, so it
    is neither re-laid-out nor closed (they may add more panels, call plt.show() or plt.tight_layout() themselves)."""
    import matplotlib.pyplot as plt
    if title:
        fig.suptitle(title)
    if not keep_open:
        fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        try:
            plt.switch_backend("TkAgg")
            plt.show()
        except Exception as e:
            warnings.warn(f"could not open a TkAgg window ({e}); use save=... instead")
    elif not keep_open:
        plt.close(fig)
    return fig


def _slice(model, dims, fixed, default_point):
    """Validate dims / fixed; default dims are the first one or two inputs."""
    if dims is None:
        dims = list(range(min(model.d, 2)))
    dims = [int(i) for i in np.atleast_1d(dims)]
    if not 1 <= len(dims) <= 2:
        raise ValueError("plot dims must name one or two input dimensions")
    fixed = np.asarray(default_point if fixed is None else fixed, float).ravel()
    if len(fixed) != model.d:
        raise ValueError(f"fixed must give a value for each of the {model.d} inputs")
    return dims, fixed


def _grid_1d(model, dim, n, fixed):
    lo, hi = model.plot_bounds[dim]
    g = np.linspace(lo, hi, n)
    Xg = np.tile(fixed, (n, 1))
    Xg[:, dim] = g
    return g, Xg


def _grid_2d(model, dims, n, fixed):
    (lo0, hi0), (lo1, hi1) = model.plot_bounds[dims[0]], model.plot_bounds[dims[1]]
    g0, g1 = np.meshgrid(np.linspace(lo0, hi0, n), np.linspace(lo1, hi1, n))
    Xg = np.tile(fixed, (n * n, 1))
    Xg[:, dims[0]], Xg[:, dims[1]] = g0.ravel(), g1.ravel()
    return g0, g1, Xg


def _band(a, g, m, v, color, label):
    s = np.sqrt(v)
    a.fill_between(g, m - 1.96 * s, m + 1.96 * s, alpha=0.25, color=color, label="95% band")
    a.plot(g, m, color=color, lw=2, label=label)


#ContinuousGP
def plot_continuous(model, dims=None, fixed=None, n=200, observation_noise=False, truth=None, ax=None,
                    show=False, save=None, title=None, cmap=None):
    """1-d: mean, 95% band (+-1.96 sd) and the data.  2-d: maps of the mean and of the sd, with the data.
    fixed defaults to the best observed point; observation_noise=True widens the band to a new y.
    In a slice, observations close to the slice are drawn solid and the others faded."""
    model._check_fitted()
    _use_backend(show)
    X, y = model.train_X, model.train_y
    dims, fixed = _slice(model, dims, fixed, X[int(np.argmax(y))])
    others = [i for i in range(model.d) if i not in dims]
    near = np.ones(len(X), bool) if not others else \
        np.all(np.abs(X[:, others] - fixed[others]) <= 0.1 * np.ptp(model.plot_bounds[others], 1), 1)
    if len(dims) == 1:
        fig, (a,) = _figure(ax)
        g, Xg = _grid_1d(model, dims[0], n, fixed)
        m, v = model.predict(Xg, observation_noise=observation_noise)
        _band(a, g, m, v, "tab:blue", "posterior mean")
        if observation_noise:
            a.collections[-1].set_label("95% band (new y)")
        if truth is not None:
            a.plot(g, np.asarray(truth(Xg)).ravel(), "k--", lw=1, label="truth")
        a.scatter(X[near, dims[0]], y[near], c="k", s=25, zorder=3, label="observations")
        if (~near).any():
            a.scatter(X[~near, dims[0]], y[~near], c="k", s=12, alpha=0.2, zorder=2)
        a.set_xlabel(f"x{dims[0]}")
        a.set_ylabel("f(x)")
        a.legend(fontsize=8)
    else:
        fig, axes = _figure(None, 1, 2, size=(5.2, 4.4))
        g0, g1, Xg = _grid_2d(model, dims, 60, fixed)
        m, v = model.predict(Xg, observation_noise=observation_noise)
        for a, z, lab in ((axes[0], m, "posterior mean"), (axes[1], np.sqrt(v), "posterior sd")):
            cs = a.contourf(g0, g1, z.reshape(g0.shape), levels=20, cmap=("viridis" if cmap is None else cmap))
            fig.colorbar(cs, ax=a)
            a.scatter(X[near, dims[0]], X[near, dims[1]], c="w", edgecolors="k", s=25)
            a.set_xlabel(f"x{dims[0]}")
            a.set_ylabel(f"x{dims[1]}")
            a.set_title(lab)
    keep = ax is not None and len(dims) == 1                  # drawn into the caller's axis: the figure stays theirs
    if keep and title:
        a.set_title(title)
    return _finish(fig, show, save, None if keep else title, keep_open=keep)


#PreferenceGP
def plot_preference(model, dims=None, fixed=None, n=200, truth=None, ax=None, show=False, save=None,
                    title=None, max_arrows=25, cmap=None, probability=False):
    """Latent utility (mean, 95% band) with the compared designs; an arrow goes from loser to winner.
    The utility has no fixed level or scale, so `truth` is rescaled to the model's mean before drawing.
    probability=True (2-d): colour by the score E[Phi(u)] (E[sigmoid(u)] for the logit link) in [0, 1],
    model.prob_torch, instead of the utility."""
    model._check_fitted()
    _use_backend(show)
    X = model.train_X
    u_train, _ = model.predict(X)
    dims, fixed = _slice(model, dims, fixed, X[int(np.argmax(u_train))])
    C = model.comparisons[:max_arrows]
    if len(dims) == 1:
        fig, (a,) = _figure(ax)
        g, Xg = _grid_1d(model, dims[0], n, fixed)
        m, v = model.predict(Xg)
        _band(a, g, m, v, "tab:purple", "latent utility (mean)")
        if truth is not None:
            t = np.asarray(truth(Xg)).ravel()
            a.plot(g, (t - t.mean()) / (t.std() + 1e-12) * m.std() + m.mean(), "k--", lw=1, label="truth (rescaled)")
        x0 = X[:, dims[0]]
        a.scatter(x0, u_train, c="k", s=20, zorder=3, label="compared designs")
        for w, l in C:
            a.annotate("", xy=(x0[w], u_train[w]), xytext=(x0[l], u_train[l]),
                       arrowprops=dict(arrowstyle="->", color="tab:orange", alpha=0.6, connectionstyle="arc3,rad=0.25"))
        a.plot([], [], color="tab:orange", label="loser -> winner")
        a.set_xlabel(f"x{dims[0]}")
        a.set_ylabel("u(x)")
        a.legend(fontsize=8)
    else:
        fig, (a,) = _figure(None, size=(6.2, 5.0))
        g0, g1, Xg = _grid_2d(model, dims, 70, fixed)
        if probability:
            score = model.prob_torch(Xg).detach().numpy()
            cs = a.contourf(g0, g1, score.reshape(g0.shape), levels=np.linspace(0, 1, 21),
                            cmap=("RdYlBu" if cmap is None else cmap))
            fig.colorbar(cs, ax=a, label="score E[Phi(u)]" if model.link == "probit" else "score E[sigmoid(u)]")
        else:
            m, _ = model.predict(Xg)
            cs = a.contourf(g0, g1, m.reshape(g0.shape), levels=20, cmap=("viridis" if cmap is None else cmap))
            fig.colorbar(cs, ax=a, label="latent utility (mean)")
        P = X[:, dims]
        for w, l in C:
            a.annotate("", xy=P[w], xytext=P[l], arrowprops=dict(arrowstyle="->", color="w", alpha=0.8, lw=1))
        a.scatter(P[:, 0], P[:, 1], c="w", edgecolors="k", s=22, zorder=3)
        a.set_xlabel(f"x{dims[0]}")
        a.set_ylabel(f"x{dims[1]}")
        a.set_title("arrows: loser -> winner")
    keep = ax is not None and len(dims) == 1
    if keep and title:
        a.set_title(title)
    return _finish(fig, show, save, None if keep else title, keep_open=keep)


#BinaryGP
def plot_binary(model, dims=None, fixed=None, n=200, truth=None, ax=None, show=False, save=None, title=None,
                cmap=None):
    """1-d: latent g with its 95% band (top) and P(y = 1) with the labels (bottom).
    2-d: map of P(y = 1), the decision boundary P = 0.5, accepted (o) and rejected (x) designs.
    truth: optional callable X -> bool; its boundary is drawn dashed."""
    model._check_fitted()
    _use_backend(show)
    X, lab = model.train_X, model.labels
    dims, fixed = _slice(model, dims, fixed, X[lab][0] if lab.any() else X[0])
    if len(dims) == 1:
        fig, axes = _figure(None, 2, 1, size=(6.5, 3.0))
        g, Xg = _grid_1d(model, dims[0], n, fixed)
        m, v = model.predict(Xg)
        a = axes[0]
        _band(a, g, m, v, "tab:green", "latent g (mean)")
        a.axhline(0, color="gray", lw=0.8)
        a.set_ylabel("g(x)")
        a.legend(fontsize=8)
        a = axes[1]
        a.plot(g, model.prob(Xg), color="tab:green", lw=2, label="P(y = 1)")
        a.axhline(0.5, color="gray", lw=0.8, ls=":")
        x0 = X[:, dims[0]]
        a.scatter(x0[lab], np.ones(lab.sum()), marker="o", c="tab:blue", label="y = 1", zorder=3)
        a.scatter(x0[~lab], np.zeros((~lab).sum()), marker="x", c="tab:red", label="y = 0", zorder=3)
        if truth is not None:
            a.plot(g, np.asarray(truth(Xg), float).ravel(), "k--", lw=1, label="truth")
        a.set_ylim(-0.08, 1.08)
        a.set_xlabel(f"x{dims[0]}")
        a.set_ylabel("probability")
        a.legend(fontsize=8)
    else:
        fig, (a,) = _figure(None, size=(6.2, 5.0))
        g0, g1, Xg = _grid_2d(model, dims, 70, fixed)
        p = model.prob(Xg).reshape(g0.shape)
        cs = a.contourf(g0, g1, p, levels=np.linspace(0, 1, 21), cmap=("RdYlBu" if cmap is None else cmap))
        fig.colorbar(cs, ax=a, label="P(y = 1)")
        a.contour(g0, g1, p, levels=[0.5], colors="k", linewidths=2)
        if truth is not None:
            a.contour(g0, g1, np.asarray(truth(Xg), float).reshape(g0.shape), levels=[0.5], colors="k",
                      linestyles="--", linewidths=1)
        P = X[:, dims]
        a.scatter(P[lab, 0], P[lab, 1], marker="o", c="tab:blue", edgecolors="k", label="y = 1")
        a.scatter(P[~lab, 0], P[~lab, 1], marker="x", c="k", label="y = 0")
        a.set_xlabel(f"x{dims[0]}")
        a.set_ylabel(f"x{dims[1]}")
        a.set_title("solid: P = 0.5" + (", dashed: truth" if truth is not None else ""))
        a.legend(fontsize=8, loc="upper right")
    return _finish(fig, show, save, title)
