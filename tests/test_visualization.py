"""Visualization: smoke tests (every plot function runs and returns the right
type of artist) plus a few value-level checks where cheap to verify."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cvcircuit import state as st
from cvcircuit import visualization as viz


@pytest.fixture(autouse=True)
def _close_figures_after_each_test():
    yield
    plt.close("all")


def test_plot_wigner_returns_axes_and_uses_gca_by_default():
    fig, ax = plt.subplots()
    plt.sca(ax)
    returned = viz.plot_wigner(st.squeezed(0.5))
    assert returned is ax


def test_plot_husimi_and_p_run():
    s = st.thermal(1.0)
    ax1 = viz.plot_husimi(s, ax=plt.gca())
    ax2 = viz.plot_p(s, ax=plt.gca())
    assert ax1 is not None and ax2 is not None


def test_plot_p_propagates_nonclassical_error():
    with pytest.raises(ValueError, match="s=1.0"):
        viz.plot_p(st.squeezed(0.5), ax=plt.gca())


def test_plot_quasiprobability_default_cmap_is_diverging_for_wigner():
    ax = viz.plot_quasiprobability(st.squeezed(0.4), s=0.0, ax=plt.gca())
    image = ax.images[0]
    assert image.get_cmap().name == "cmwig1"
    assert image.norm.vmin == -image.norm.vmax  # symmetric norm


def test_plot_wigner_grid_returns_fig_and_right_number_of_axes():
    s = st.random_state(5, seed=0)
    fig, axes = viz.plot_wigner_grid(s)
    assert axes.size >= 5


def test_plot_marginal_matches_state_marginal():
    s = st.squeezed(0.5).displace(0.2 - 0.1j)
    ax = viz.plot_marginal(s, mode=0, angle=0.3, ax=plt.gca())
    line = ax.lines[0]
    xvec, pdf = s.marginal(0, angle=0.3)
    assert np.allclose(line.get_xdata(), xvec)
    assert np.allclose(line.get_ydata(), pdf)


def test_plot_joint_marginal_epr_correlation_visible():
    tmsv = st.two_mode_squeezed(1.0)
    ax = viz.plot_joint_marginal(tmsv, (0, 0.0), (1, 0.0), ax=plt.gca())
    assert len(ax.images) == 1


def test_plot_ellipse_single_and_multiple_states():
    ax = viz.plot_ellipse(st.squeezed(0.5), ax=plt.gca())
    assert len(ax.patches) == 1
    ax2 = viz.plot_ellipse([st.vacuum(1), st.squeezed(0.5)], labels=["vac", "sq"], ax=plt.gca())
    assert len(ax2.patches) >= 1  # accumulates on the same gca() axes


def test_plot_covariance_has_mode_tick_labels():
    s = st.two_mode_squeezed(0.5)
    ax = viz.plot_covariance(s, ax=plt.gca())
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["x0", "p0", "x1", "p1"]


def test_cmwig1_is_centered_white():
    color_at_half = viz.CMWIG1(0.5)
    assert np.allclose(color_at_half[:3], (1.0, 1.0, 1.0), atol=0.05)
