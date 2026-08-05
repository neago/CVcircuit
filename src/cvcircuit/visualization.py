"""Phase-space and covariance visualization.

Matplotlib conventions follow the user's own tools (``lcg_plus/plotting.py``,
``qopt/tomography_plotting.py``): ``imshow`` with bilinear interpolation, a
symmetric ``Normalize(-scale, scale)`` for signed quasiprobabilities,
``ax.set_aspect("equal")``, ``(..., ax=None)`` defaulting to ``plt.gca()``,
and no ``plt.show()`` or global rcParams.  The default diverging colormap
``CMWIG1`` is the user's own hand-tuned blue-white-red map from
``qopt/tomography_plotting.py``; pass ``cmap=`` to override.  We use
``origin="lower"`` throughout, so (unlike ``lcg_plus``) no ``-pvec``
meshgrid flip is needed to match the phase-space y-axis convention.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

CMWIG1 = mpl.colors.LinearSegmentedColormap.from_list(
    "cmwig1",
    [
        (0.00, "#0047b2"), (0.10, "#0c50b7"), (0.20, "#2765c2"), (0.30, "#5889d3"),
        (0.40, "#a2bee8"), (0.49, "#ffffff"), (0.51, "#ffffff"), (0.60, "#e8a2a2"),
        (0.70, "#d35858"), (0.80, "#c22727"), (0.90, "#b70c0c"), (1.00, "#b20000"),
    ],
)

_S_NAME = {0.0: "Wigner", -1.0: "Husimi Q", 1.0: "P"}


def _auto_range(sub_state, s, pad_sigma=5.0):
    cov_s = sub_state.cov - s * (sub_state.hbar / 2) * np.eye(2)
    eigs = np.linalg.eigvalsh(cov_s)
    spread_cov = cov_s if np.all(eigs > 0) else sub_state.cov
    std = float(np.sqrt(np.linalg.eigvalsh(spread_cov).max()))
    pad = pad_sigma * std + 0.5
    mean = sub_state.mean
    return (mean[0] - pad, mean[0] + pad), (mean[1] - pad, mean[1] + pad)


# ---------------------------------------------------------------------------
# quasiprobability heatmaps
# ---------------------------------------------------------------------------


def plot_quasiprobability(state, mode=0, s=0.0, ax=None, n=200, xrange=None, prange=None,
                           cmap=None, colorbar=True, title=None):
    """Heatmap of the s-ordered quasiprobability of ``mode`` (s=0 Wigner, s=-1 Husimi
    Q, s=+1 P; raises for the P function of a non-classical state)."""
    ax = ax or plt.gca()
    sub = state.reduced([mode])
    auto_x, auto_p = _auto_range(sub, s)
    xrange = xrange or auto_x
    prange = prange or auto_p
    xvec = np.linspace(*xrange, n)
    pvec = np.linspace(*prange, n)
    x_grid, p_grid = np.meshgrid(xvec, pvec)
    values = state.quasiprobability(x_grid, p_grid, s=s, mode=mode)

    signed = s == 0.0
    if cmap is None:
        cmap = CMWIG1 if signed else "viridis"
    scale = float(np.max(np.abs(values))) or 1.0
    norm = mpl.colors.Normalize(-scale, scale) if signed else mpl.colors.Normalize(0, scale)

    extent = (xrange[0], xrange[1], prange[0], prange[1])
    image = ax.imshow(values, extent=extent, origin="lower", cmap=cmap, norm=norm,
                       interpolation="bilinear", aspect="equal")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(title if title is not None else f"{_S_NAME.get(s, f's={s:g}')} function, mode {mode}")
    if colorbar:
        plt.colorbar(image, ax=ax)
    return ax


def plot_wigner(state, mode=0, ax=None, **kwargs):
    """Wigner function of ``mode``."""
    return plot_quasiprobability(state, mode=mode, s=0.0, ax=ax, **kwargs)


def plot_husimi(state, mode=0, ax=None, **kwargs):
    """Husimi Q function of ``mode``."""
    return plot_quasiprobability(state, mode=mode, s=-1.0, ax=ax, **kwargs)


def plot_p(state, mode=0, ax=None, **kwargs):
    """Glauber-Sudarshan P function of ``mode`` (raises if the state is non-classical)."""
    return plot_quasiprobability(state, mode=mode, s=1.0, ax=ax, **kwargs)


def plot_wigner_grid(state, n_cols=None, figsize=None, **kwargs):
    """A grid of single-mode Wigner functions, one subplot per mode.

    This is a figure-level function (it needs several Axes, not one), so it
    returns ``(fig, axes)`` rather than a single ``Axes``.
    """
    n = state.num_modes
    n_cols = n_cols or min(n, 4)
    n_rows = int(np.ceil(n / n_cols))
    figsize = figsize or (3.2 * n_cols, 3.0 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    for m in range(n):
        plot_wigner(state, mode=m, ax=axes[m // n_cols, m % n_cols], colorbar=False, **kwargs)
    for m in range(n, n_rows * n_cols):
        axes[m // n_cols, m % n_cols].axis("off")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# marginals
# ---------------------------------------------------------------------------


def plot_marginal(state, mode, angle=0.0, ax=None, xvec=None, label=None, **plot_kwargs):
    """Plot the 1-D probability density of the quadrature at ``angle`` on ``mode``."""
    ax = ax or plt.gca()
    xvec, pdf = state.marginal(mode, angle=angle, xvec=xvec)
    ax.plot(xvec, pdf, label=label, **plot_kwargs)
    ax.set_xlabel(f"quadrature (mode {mode}, angle={angle:.3g})")
    ax.set_ylabel("probability density")
    if label:
        ax.legend()
    return ax


def plot_joint_marginal(state, quad_a, quad_b, ax=None, n=200, xrange=None, yrange=None,
                         cmap="viridis", colorbar=True, n_sigma=5.0):
    """Joint probability density of two quadrature combinations ``(mode, angle)``.

    The two quadratures may be on the same or different modes -- this is
    the tool for EPR-style correlation plots, e.g. mode-0's x against
    mode-1's p.
    """
    ax = ax or plt.gca()
    mode_a, angle_a = quad_a
    mode_b, angle_b = quad_b
    n_modes = state.num_modes

    dir_a = np.zeros(2 * n_modes)
    dir_a[2 * mode_a : 2 * mode_a + 2] = [np.cos(angle_a), np.sin(angle_a)]
    dir_b = np.zeros(2 * n_modes)
    dir_b[2 * mode_b : 2 * mode_b + 2] = [np.cos(angle_b), np.sin(angle_b)]

    mean_a, mean_b = dir_a @ state.mean, dir_b @ state.mean
    var_a, var_b = dir_a @ state.cov @ dir_a, dir_b @ state.cov @ dir_b
    cov_ab = dir_a @ state.cov @ dir_b
    cov2 = np.array([[var_a, cov_ab], [cov_ab, var_b]])
    inv, det = np.linalg.inv(cov2), np.linalg.det(cov2)

    if xrange is None:
        pad = n_sigma * np.sqrt(var_a)
        xrange = (mean_a - pad, mean_a + pad)
    if yrange is None:
        pad = n_sigma * np.sqrt(var_b)
        yrange = (mean_b - pad, mean_b + pad)
    x_grid, y_grid = np.meshgrid(np.linspace(*xrange, n), np.linspace(*yrange, n))
    dx, dy = x_grid - mean_a, y_grid - mean_b
    exponent = -0.5 * (inv[0, 0] * dx**2 + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy**2)
    pdf = np.exp(exponent) / (2 * np.pi * np.sqrt(det))

    image = ax.imshow(pdf, extent=(*xrange, *yrange), origin="lower", cmap=cmap,
                       interpolation="bilinear", aspect="equal")
    ax.set_xlabel(f"mode {mode_a} quadrature (angle={angle_a:.3g})")
    ax.set_ylabel(f"mode {mode_b} quadrature (angle={angle_b:.3g})")
    if colorbar:
        plt.colorbar(image, ax=ax)
    return ax


# ---------------------------------------------------------------------------
# uncertainty ellipses and covariance heatmap
# ---------------------------------------------------------------------------


def plot_ellipse(states, mode=0, n_sigma=1.0, ax=None, labels=None, colors=None, fill=False):
    """Overlay n-sigma uncertainty ellipses for one or more states' ``mode``.

    ``states`` may be a single :class:`~cvcircuit.state.GaussianState` or a
    list, for comparing several states (e.g. before/after a channel) on one
    axes.
    """
    ax = ax or plt.gca()
    if not isinstance(states, (list, tuple)):
        states = [states]
    if labels is None:
        labels = [None] * len(states)
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, one_state in enumerate(states):
        sub = one_state.reduced([mode])
        eigvals, eigvecs = np.linalg.eigh(sub.cov)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * n_sigma * np.sqrt(np.maximum(eigvals, 0))
        color = colors[i % len(colors)]
        ellipse = Ellipse(
            sub.mean, width, height, angle=angle,
            facecolor=color if fill else "none", edgecolor=color,
            linewidth=1.5, alpha=0.8 if fill else 1.0, label=labels[i],
        )
        ax.add_patch(ellipse)

    ax.relim()
    ax.autoscale_view()
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_aspect("equal")
    if any(labels):
        ax.legend()
    return ax


def plot_covariance(state, ax=None, cmap=None, colorbar=True):
    """Heatmap of the full covariance matrix, with per-mode x/p tick labels and gridlines."""
    ax = ax or plt.gca()
    cov = state.cov
    scale = float(np.max(np.abs(cov))) or 1.0
    cmap = cmap or CMWIG1
    norm = mpl.colors.Normalize(-scale, scale)
    image = ax.imshow(cov, cmap=cmap, norm=norm, interpolation="nearest")

    n = state.num_modes
    tick_labels = [f"{'x' if i % 2 == 0 else 'p'}{i // 2}" for i in range(2 * n)]
    ax.set_xticks(range(2 * n))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(range(2 * n))
    ax.set_yticklabels(tick_labels, fontsize=8)
    for m in range(1, n):
        ax.axhline(2 * m - 0.5, color="gray", linewidth=0.5)
        ax.axvline(2 * m - 0.5, color="gray", linewidth=0.5)
    if colorbar:
        plt.colorbar(image, ax=ax)
    return ax
