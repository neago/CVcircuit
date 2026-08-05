"""Gaussian measurements: general-dyne, homodyne, heterodyne.

Every function here operates on raw ``(mean, cov, hbar)`` arrays for an
n-mode state and returns ``(outcome, new_mean, new_cov)`` for the **full**
n-mode system: measured modes are reset to vacuum and kept in place, so
mode indices never shift (matching Strawberry Fields' convention).
:mod:`cvcircuit.state` wraps these as :class:`~cvcircuit.state.GaussianState`
methods.

General-dyne is the single primitive (Brask, arXiv:2102.05748, Sec. 4;
Weedbrook et al., RMP 84, 621 (2012), Sec. V); homodyne is implemented as
its own exact rank-1 update (not a squeezed-POVM limit) for numerical
robustness, following the same reference.
"""

from __future__ import annotations

import numpy as np

from . import channels as _channels
from . import conversions
from . import ops


def _validate_modes(modes, n):
    modes = np.atleast_1d(np.asarray(modes, dtype=int))
    for m in modes:
        if not 0 <= m < n:
            raise ValueError(f"mode {m} out of range for {n}-mode state")
    if len(set(modes.tolist())) != modes.size:
        raise ValueError(f"modes must be distinct, got {modes.tolist()}")
    return modes


# ---------------------------------------------------------------------------
# general-dyne
# ---------------------------------------------------------------------------


def outcome_distribution(mean, cov, modes, cov_povm):
    """Mean and covariance of the general-dyne outcome, without sampling.

    The outcome of measuring ``modes`` against a Gaussian POVM of
    covariance ``cov_povm`` (2k x 2k, k = len(modes)) is itself Gaussian
    distributed with mean = mean of the measured modes and covariance
    sigma_B + cov_povm.
    """
    n = mean.size // 2
    modes = _validate_modes(modes, n)
    idx_b = ops.mode_indices(modes)
    cov_povm = np.asarray(cov_povm, dtype=float)
    k = modes.size
    if cov_povm.shape != (2 * k, 2 * k):
        raise ValueError(f"cov_povm has shape {cov_povm.shape}, expected {(2 * k, 2 * k)} for {k} mode(s)")
    mean_b = mean[idx_b]
    sigma_b = cov[np.ix_(idx_b, idx_b)]
    return mean_b, sigma_b + cov_povm


def generaldyne_logpdf(outcome, mean_b, outcome_cov):
    """Log-density of a general-dyne outcome under its Gaussian distribution."""
    outcome = np.asarray(outcome, dtype=float)
    delta = outcome - np.asarray(mean_b, dtype=float)
    sign, logdet = np.linalg.slogdet(outcome_cov)
    if sign <= 0:
        raise ValueError("outcome_cov is not positive definite")
    quad = delta @ np.linalg.solve(outcome_cov, delta)
    return -0.5 * (outcome.size * np.log(2 * np.pi) + logdet + quad)


def generaldyne(mean, cov, hbar, modes, cov_povm, seed=None):
    """General-dyne measurement of ``modes`` against a Gaussian POVM of
    covariance ``cov_povm`` (2k x 2k).  ``cov_povm = (hbar/2) I`` is
    heterodyne; ``cov_povm -> 0`` in one quadrature approaches homodyne
    (use :func:`homodyne` directly for an exact, non-singular result).

    Returns ``(outcome, new_mean, new_cov)`` for the full n-mode system,
    with the measured modes reset to vacuum.
    """
    n = mean.size // 2
    modes = _validate_modes(modes, n)
    mean_b, outcome_cov = outcome_distribution(mean, cov, modes, cov_povm)

    idx_b = ops.mode_indices(modes)
    idx_a = np.setdiff1d(np.arange(2 * n), idx_b)
    sigma_a = cov[np.ix_(idx_a, idx_a)]
    c_ab = cov[np.ix_(idx_a, idx_b)]
    mean_a = mean[idx_a]

    rng = np.random.default_rng(seed)
    outcome = rng.multivariate_normal(mean_b, outcome_cov)

    gain = c_ab @ np.linalg.inv(outcome_cov)
    new_mean_a = mean_a + gain @ (outcome - mean_b)
    new_sigma_a = sigma_a - gain @ c_ab.T

    new_mean = np.zeros(2 * n)
    new_cov = np.zeros((2 * n, 2 * n))
    new_mean[idx_a] = new_mean_a
    new_cov[np.ix_(idx_a, idx_a)] = new_sigma_a
    new_cov[np.ix_(idx_b, idx_b)] = (hbar / 2) * np.eye(idx_b.size)
    return outcome, new_mean, new_cov


# ---------------------------------------------------------------------------
# heterodyne
# ---------------------------------------------------------------------------


def heterodyne(mean, cov, hbar, mode, seed=None):
    """Heterodyne (dual/eight-port homodyne) measurement of one mode.

    General-dyne with cov_povm = (hbar/2) I (a vacuum-noise POVM); the
    outcome is returned as a complex amplitude alpha.
    """
    n = mean.size // 2
    _validate_modes(mode, n)
    cov_povm = (hbar / 2) * np.eye(2)
    outcome, new_mean, new_cov = generaldyne(mean, cov, hbar, [mode], cov_povm, seed=seed)
    alpha = conversions.xp_to_alpha(outcome, hbar=hbar)[0]
    return alpha, new_mean, new_cov


# ---------------------------------------------------------------------------
# homodyne
# ---------------------------------------------------------------------------


def _detector_channel(efficiency, electronic_noise, hbar):
    channel = None
    if efficiency < 1:
        channel = _channels.loss(efficiency, hbar=hbar)
    if electronic_noise > 0:
        noise_channel = _channels.additive_noise(electronic_noise, electronic_noise, hbar=hbar)
        channel = noise_channel if channel is None else channel.compose(noise_channel)
    return channel


def homodyne(mean, cov, hbar, mode, angle=0.0, efficiency=1.0, electronic_noise=0.0, seed=None):
    """Homodyne measurement of the quadrature at ``angle`` on ``mode``.

    Implemented exactly, not as a general-dyne limit: rotate the mode by
    ``-angle`` so the target quadrature is the local x, apply the rank-1
    Brask (arXiv:2102.05748, Eq. 20-21) update with the scalar 1/B_11, then
    rotate back.  ``efficiency`` (0, 1] and ``electronic_noise`` (shot-noise
    units, >= 0) model a lossy, noisy detector as a channel applied to the
    mode just before the projection.
    """
    n = mean.size // 2
    _validate_modes(mode, n)
    if not 0 < efficiency <= 1:
        raise ValueError(f"efficiency must be in (0, 1], got {efficiency}")
    if electronic_noise < 0:
        raise ValueError(f"electronic_noise must be >= 0, got {electronic_noise}")

    mean = np.array(mean, dtype=float)
    cov = np.array(cov, dtype=float)
    detector = _detector_channel(efficiency, electronic_noise, hbar)
    if detector is not None:
        x_full = ops.expand(detector.X, [mode], n)
        y_full = np.zeros((2 * n, 2 * n))
        idx = ops.mode_indices([mode])
        y_full[np.ix_(idx, idx)] = detector.Y
        mean = x_full @ mean
        cov = x_full @ cov @ x_full.T + y_full

    rot_full = ops.expand(ops.rotation(-angle), [mode], n)
    r = rot_full @ mean
    s = rot_full @ cov @ rot_full.T

    idx_b = ops.mode_indices([mode])
    idx_a = np.setdiff1d(np.arange(2 * n), idx_b)
    sigma_a = s[np.ix_(idx_a, idx_a)]
    sigma_b = s[np.ix_(idx_b, idx_b)]
    c_ab = s[np.ix_(idx_a, idx_b)]
    mean_a = r[idx_a]
    mean_b = r[idx_b]

    b11 = sigma_b[0, 0]
    if b11 <= 1e-14:
        raise ValueError("degenerate homodyne measurement (zero variance in the measured quadrature)")

    rng = np.random.default_rng(seed)
    outcome = rng.normal(mean_b[0], np.sqrt(b11))

    pi_proj = np.array([[1.0, 0.0], [0.0, 0.0]])
    u_vec = np.array([outcome, 0.0])
    new_sigma_a = sigma_a - (1.0 / b11) * c_ab @ pi_proj @ c_ab.T
    new_mean_a = mean_a - (1.0 / b11) * c_ab @ pi_proj @ (mean_b - u_vec)

    new_r = np.zeros(2 * n)
    new_s = np.zeros((2 * n, 2 * n))
    new_r[idx_a] = new_mean_a
    new_s[np.ix_(idx_a, idx_a)] = new_sigma_a
    new_s[np.ix_(idx_b, idx_b)] = (hbar / 2) * np.eye(2)

    new_mean = rot_full.T @ new_r
    new_cov = rot_full.T @ new_s @ rot_full.T.T
    return outcome, new_mean, new_cov
