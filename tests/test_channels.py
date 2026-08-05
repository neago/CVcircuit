"""Gaussian channels: loss, amplifier, noise, and the exact-moment phase-noise map."""

import numpy as np
import pytest

from cvcircuit import channels
from cvcircuit import state as st

HBARS = (1, 2)


def test_loss_identity_at_eta_one():
    ch = channels.loss(1.0, hbar=2)
    assert np.allclose(ch.X, np.eye(2))
    assert np.allclose(ch.Y, 0)


@pytest.mark.parametrize("hbar", HBARS)
def test_loss_squeezed_variance_formula(hbar):
    r, eta = 0.8, 0.7
    s = st.squeezed(r, hbar=hbar).loss(eta, mode=0)
    var_x = s.quad_variance(0, angle=0.0)
    # in SNU: var = eta * exp(-2r) + (1-eta)
    expected_snu = eta * np.exp(-2 * r) + (1 - eta)
    assert var_x / (hbar / 2) == pytest.approx(expected_snu, abs=1e-8)


def test_loss_invalid_eta():
    with pytest.raises(ValueError, match="eta"):
        channels.loss(1.5)
    with pytest.raises(ValueError, match="eta"):
        channels.loss(-0.1)


def test_amplifier_gain_one_is_identity():
    ch = channels.amplifier(1.0)
    assert np.allclose(ch.X, np.eye(2))
    assert np.allclose(ch.Y, 0)


def test_amplifier_invalid_gain():
    with pytest.raises(ValueError, match="g must be"):
        channels.amplifier(0.5)


def test_amplifier_increases_variance():
    s = st.vacuum(1).amplify(2.0, mode=0)
    assert s.cov[0, 0] > 1.0  # hbar=2 vacuum starts at 1


def test_excess_noise_adds_isotropic_noise():
    xi = 0.3
    s = st.vacuum(1, hbar=2).add_noise(xi, mode=0)
    assert np.allclose(s.cov, (1 + xi) * np.eye(2))


def test_excess_noise_invalid():
    with pytest.raises(ValueError, match="xi"):
        channels.excess_noise(-1.0)


def test_channel_not_completely_positive_rejected():
    with pytest.raises(ValueError, match="completely positive"):
        channels.GaussianChannel(np.eye(2), -np.eye(2), hbar=2)


def test_channel_compose_matches_sequential_application():
    ch1 = channels.loss(0.8, hbar=2)
    ch2 = channels.excess_noise(0.1, hbar=2)
    composed = ch1.compose(ch2)
    s = st.squeezed(0.5)
    step1 = s.apply_channel(ch1, [0])
    step2 = step1.apply_channel(ch2, [0])
    combined = s.apply_channel(composed, [0])
    assert np.allclose(step2.mean, combined.mean, atol=1e-10)
    assert np.allclose(step2.cov, combined.cov, atol=1e-10)


def test_channel_matmul_operator():
    ch1 = channels.loss(0.8, hbar=2)
    ch2 = channels.excess_noise(0.1, hbar=2)
    composed_matmul = ch2 @ ch1  # apply ch1 first, then ch2
    composed_method = ch1.compose(ch2)
    assert np.allclose(composed_matmul.X, composed_method.X)
    assert np.allclose(composed_matmul.Y, composed_method.Y)


def test_classical_noise_channel():
    cov_noise = np.diag([0.2, 0.4])
    ch = channels.classical_noise(cov_noise, hbar=2)
    assert np.allclose(ch.X, np.eye(2))
    assert np.allclose(ch.Y, cov_noise)


def test_thermal_channel_resets_to_thermal():
    ch = channels.thermal_channel(1.5, hbar=2)
    s = st.squeezed(2.0).apply_channel(ch, [0])
    expected = st.thermal(1.5)
    assert np.allclose(s.cov, expected.cov, atol=1e-10)
    assert np.allclose(s.mean, 0, atol=1e-10)


def test_thermal_channel_invalid():
    with pytest.raises(ValueError, match="nbar"):
        channels.thermal_channel(-1.0)


# ---------------------------------------------------------------------------
# phase noise
# ---------------------------------------------------------------------------


def test_phase_noise_zero_sigma_is_pure_rotation():
    s = st.coherent(1 + 0.5j)
    mean_phase = 0.3
    noisy = s.phase_noise(0.0, mode=0, mean_phase=mean_phase)
    rotated = s.rotate(mean_phase, mode=0)
    assert np.allclose(noisy.mean, rotated.mean, atol=1e-8)
    assert np.allclose(noisy.cov, rotated.cov, atol=1e-8)


def test_phase_noise_large_sigma_averages_mean_to_zero():
    s = st.coherent(1 + 0.5j)
    noisy = s.phase_noise(50.0, mode=0)  # effectively uniform phase
    assert np.allclose(noisy.mean, 0, atol=1e-6)


def test_phase_noise_preserves_uncertainty_relation():
    s = st.squeezed(0.7).displace(0.5 - 0.2j)
    noisy = s.phase_noise(0.4, mode=0, mean_phase=0.9)
    from cvcircuit import ops

    omega = ops.symplectic_form(1)
    eigs = np.linalg.eigvalsh(noisy.cov + 1j * (noisy.hbar / 2) * omega)
    assert eigs.min() >= -1e-8


def test_phase_noise_monte_carlo():
    # cross-check the exact-moment formula against direct Monte Carlo averaging
    rng = np.random.default_rng(0)
    hbar = 2.0
    mean = np.array([1.3, -0.7])
    cov = np.array([[0.6, 0.15], [0.15, 1.9]])
    sigma, mean_phase = 0.35, 0.4

    new_mean, new_cov = channels.phase_noise(mean, cov, hbar, sigma, mode=0, mean_phase=mean_phase)

    n_samples = 300_000
    phis = rng.normal(mean_phase, sigma, size=n_samples)
    samples = rng.multivariate_normal(mean, cov, size=n_samples)
    c, s_ = np.cos(phis), np.sin(phis)
    rotated = np.stack([c * samples[:, 0] - s_ * samples[:, 1], s_ * samples[:, 0] + c * samples[:, 1]], axis=1)

    assert np.allclose(new_mean, rotated.mean(axis=0), atol=0.02)
    assert np.allclose(new_cov, np.cov(rotated.T), atol=0.02)


def test_phase_noise_invalid_sigma():
    with pytest.raises(ValueError, match="sigma"):
        channels.phase_noise(np.zeros(2), np.eye(2), 2, -1.0, mode=0)


def test_phase_noise_mode_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        channels.phase_noise(np.zeros(2), np.eye(2), 2, 0.1, mode=5)
