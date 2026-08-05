"""General-dyne, homodyne, heterodyne: cross-path agreement, statistics, invariants."""

import numpy as np
import pytest

from cvcircuit import measure
from cvcircuit import ops
from cvcircuit import state as st


def test_generaldyne_with_squeezed_povm_converges_to_homodyne():
    s = st.two_mode_squeezed(0.9)
    # a strongly p-squeezed POVM approximates an x-homodyne projection
    povm = ops.squeezing(8.0, phi=0.0) @ (s.hbar / 2 * np.eye(2)) @ ops.squeezing(8.0, phi=0.0).T
    _, cond_gd = s.generaldyne([0], povm, seed=1)
    _, cond_ho = s.homodyne(mode=0, angle=0.0, seed=1)
    assert np.allclose(cond_gd.reduced([1]).cov, cond_ho.reduced([1]).cov, atol=1e-3)


def test_heterodyne_is_generaldyne_with_vacuum_povm():
    s = st.squeezed(0.6) & st.thermal(0.4)
    povm = (s.hbar / 2) * np.eye(2)
    out_gd, cond_gd = s.generaldyne([0], povm, seed=7)
    out_het, cond_het = s.heterodyne(mode=0, seed=7)
    from cvcircuit import conversions

    assert np.allclose(conversions.xp_to_alpha(out_gd, hbar=s.hbar)[0], out_het)
    assert np.allclose(cond_gd.cov, cond_het.cov)
    assert np.allclose(cond_gd.mean, cond_het.mean)


def test_generaldyne_two_mode_povm():
    s = st.random_state(3, seed=0)
    povm = (s.hbar / 2) * np.eye(4)  # heterodyne on 2 modes at once
    outcome, cond = s.generaldyne([0, 2], povm, seed=3)
    assert outcome.shape == (4,)
    assert np.allclose(cond.reduced([0, 2]).cov, (s.hbar / 2) * np.eye(4))
    assert cond.num_modes == 3


def test_generaldyne_rejects_duplicate_modes():
    s = st.vacuum(2)
    with pytest.raises(ValueError, match="distinct"):
        measure.generaldyne(s.mean, s.cov, s.hbar, [0, 0], np.eye(2))


def test_generaldyne_wrong_povm_shape():
    s = st.vacuum(2)
    with pytest.raises(ValueError, match="cov_povm has shape"):
        measure.generaldyne(s.mean, s.cov, s.hbar, [0], np.eye(4))


def test_outcome_distribution_matches_sampling_statistics():
    s = st.squeezed(0.5) & st.vacuum(1)
    povm = (s.hbar / 2) * np.eye(2)
    mean_b, outcome_cov = measure.outcome_distribution(s.mean, s.cov, [0], povm)
    outcomes = np.array(
        [measure.generaldyne(s.mean, s.cov, s.hbar, [0], povm, seed=i)[0] for i in range(4000)]
    )
    assert np.allclose(outcomes.mean(axis=0), mean_b, atol=0.1)
    assert np.allclose(np.cov(outcomes.T), outcome_cov, atol=0.15)


def test_generaldyne_logpdf_matches_scipy_style_formula():
    mean_b = np.array([0.3, -0.2])
    cov = np.array([[1.2, 0.1], [0.1, 0.9]])
    outcome = np.array([0.5, 0.1])
    logpdf = measure.generaldyne_logpdf(outcome, mean_b, cov)
    delta = outcome - mean_b
    expected = -0.5 * (2 * np.log(2 * np.pi) + np.log(np.linalg.det(cov)) + delta @ np.linalg.inv(cov) @ delta)
    assert logpdf == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# homodyne
# ---------------------------------------------------------------------------


def test_homodyne_rotated_angle_matches_pre_rotation():
    # homodyne at angle theta on s == homodyne at angle 0 on s rotated by -theta,
    # then rotate the conditional state back by theta
    s = st.squeezed(0.7).displace(0.4 - 0.2j)
    theta = 0.9
    _, cond_direct = s.homodyne(mode=0, angle=theta, seed=5)
    pre_rotated = s.rotate(-theta, mode=0)
    _, cond_indirect = pre_rotated.homodyne(mode=0, angle=0.0, seed=5)
    cond_indirect_rotated = cond_indirect.rotate(theta, mode=0)
    assert np.allclose(cond_direct.cov, cond_indirect_rotated.cov, atol=1e-8)


def test_homodyne_efficiency_adds_loss_before_projection():
    s = st.squeezed(0.6)
    eta = 0.8
    _, cond_ineff = s.homodyne(mode=0, angle=0.0, efficiency=eta, seed=0)
    lossy = s.loss(eta, mode=0)
    _, cond_direct = lossy.homodyne(mode=0, angle=0.0, seed=0)
    assert np.allclose(cond_ineff.cov, cond_direct.cov, atol=1e-8)


def test_homodyne_electronic_noise_increases_outcome_variance():
    s = st.vacuum(1)
    outcomes_clean = [s.homodyne(mode=0, seed=i)[0] for i in range(2000)]
    outcomes_noisy = [s.homodyne(mode=0, electronic_noise=0.5, seed=i)[0] for i in range(2000)]
    assert np.var(outcomes_noisy) > np.var(outcomes_clean)


def test_homodyne_invalid_efficiency():
    with pytest.raises(ValueError, match="efficiency"):
        st.vacuum(1).homodyne(efficiency=0.0)
    with pytest.raises(ValueError, match="efficiency"):
        st.vacuum(1).homodyne(efficiency=1.5)


def test_homodyne_invalid_electronic_noise():
    with pytest.raises(ValueError, match="electronic_noise"):
        st.vacuum(1).homodyne(electronic_noise=-0.1)


def test_homodyne_sampling_matches_analytic_gaussian_5sigma():
    s = st.squeezed(0.4)
    shots = 3000
    outcomes = np.array([s.homodyne(mode=0, angle=0.0, seed=i)[0] for i in range(shots)])
    analytic_var = s.quad_variance(0, angle=0.0)
    sigma_of_var_estimate = analytic_var * np.sqrt(2 / shots)  # approx std of sample variance
    assert abs(np.var(outcomes) - analytic_var) < 5 * sigma_of_var_estimate + 1e-6
    assert abs(np.mean(outcomes)) < 5 * np.sqrt(analytic_var / shots) + 1e-6


def test_seeded_reproducibility():
    s = st.squeezed(0.5)
    out1, _ = s.homodyne(mode=0, seed=123)
    out2, _ = s.homodyne(mode=0, seed=123)
    out3, _ = s.homodyne(mode=0, seed=124)
    assert out1 == out2
    assert out1 != out3
