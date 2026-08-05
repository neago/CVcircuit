"""Gaussian channels: loss, amplification, excess noise, and phase noise.

A Gaussian channel maps mean and covariance as::

    r_bar -> X r_bar + d          sigma -> X sigma X^T + Y

:class:`GaussianChannel` bundles (X, Y, d) for a fixed number of modes;
:func:`loss`, :func:`amplifier`, :func:`excess_noise` and
:func:`classical_noise` build the common single-mode cases.  Complete
positivity requires Y + i(hbar/2)(Omega - X Omega X^T) >= 0 (Weedbrook et
al., RMP 84, 621 (2012), Eq. 4.4); the constructor checks it.

**Phase noise is not a Gaussian channel** and is deliberately not built as
one here -- see :func:`phase_noise` below.
"""

from __future__ import annotations

import numpy as np

from .ops import symplectic_form

_ATOL = 1e-8


class GaussianChannel:
    """A Gaussian channel (X, Y, d) acting on a fixed number of modes.

    Parameters
    ----------
    X, Y:
        The linear and noise maps, sigma -> X sigma X^T + Y.  Y must be
        symmetric positive semi-definite.
    d:
        Displacement added to the mean after the X map; defaults to zero.
    hbar:
        The hbar the channel's complete-positivity bound was checked
        against (informational; the channel itself is a fixed numeric map).
    """

    def __init__(self, X, Y, d=None, hbar=2, atol=_ATOL):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n = X.shape[-1] // 2
        if X.shape != (2 * n, 2 * n) or Y.shape != (2 * n, 2 * n):
            raise ValueError(f"X and Y must be square 2n x 2n matrices, got {X.shape} and {Y.shape}")
        if not np.allclose(Y, Y.T, atol=atol):
            raise ValueError("Y must be symmetric")
        omega = symplectic_form(n)
        cp_matrix = Y + 1j * (hbar / 2) * (omega - X @ omega @ X.T)
        min_eig = np.linalg.eigvalsh(cp_matrix).min()
        if min_eig < -atol:
            raise ValueError(
                f"channel is not completely positive at hbar={hbar} "
                f"(Y + i(hbar/2)(Omega - X Omega X^T) has eigenvalue {min_eig:.3g} < 0)"
            )
        self.X = X
        self.Y = Y
        self.d = np.zeros(2 * n) if d is None else np.asarray(d, dtype=float)
        self.hbar = float(hbar)
        self.num_modes = n

    def compose(self, other):
        """Channel obtained by applying ``self`` first, then ``other``."""
        if self.num_modes != other.num_modes:
            raise ValueError(f"cannot compose channels on {self.num_modes} and {other.num_modes} modes")
        X = other.X @ self.X
        Y = other.X @ self.Y @ other.X.T + other.Y
        d = other.X @ self.d + other.d
        return GaussianChannel(X, Y, d, hbar=self.hbar)

    def __matmul__(self, other):
        """``a @ b`` applies ``b`` first, then ``a`` (matching matrix-product order)."""
        return other.compose(self)

    def __repr__(self):
        return f"GaussianChannel({self.num_modes} mode(s), hbar={self.hbar:g})"


# ---------------------------------------------------------------------------
# single-mode channel constructors
# ---------------------------------------------------------------------------


def loss(eta, nbar_env=0.0, hbar=2):
    """Pure loss / attenuation channel: transmittance eta, environment mean photon number nbar_env.

    X = sqrt(eta) I,  Y = (1-eta)(hbar/2)(2 nbar_env + 1) I.  eta=1 is the
    identity channel; nbar_env=0 (vacuum environment) is ordinary optical
    loss.
    """
    if not 0 <= eta <= 1:
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    if nbar_env < 0:
        raise ValueError(f"nbar_env must be >= 0, got {nbar_env}")
    X = np.sqrt(eta) * np.eye(2)
    Y = (1 - eta) * (hbar / 2) * (2 * nbar_env + 1) * np.eye(2)
    return GaussianChannel(X, Y, hbar=hbar)


def amplifier(g, nbar_env=0.0, hbar=2):
    """Phase-insensitive amplifier channel: gain g >= 1, environment mean photon number nbar_env.

    X = sqrt(g) I,  Y = (g-1)(hbar/2)(2 nbar_env + 1) I.
    """
    if g < 1:
        raise ValueError(f"g must be >= 1, got {g}")
    if nbar_env < 0:
        raise ValueError(f"nbar_env must be >= 0, got {nbar_env}")
    X = np.sqrt(g) * np.eye(2)
    Y = (g - 1) * (hbar / 2) * (2 * nbar_env + 1) * np.eye(2)
    return GaussianChannel(X, Y, hbar=hbar)


def excess_noise(xi, hbar=2):
    """Isotropic additive (excess) noise channel, xi in shot-noise units: X=I, Y=xi(hbar/2)I.

    In CV-QKD the line noise referred to the channel output, xi, relates to
    the input-referred excess noise epsilon via the total line-added noise
    chi_line = (1-eta)/eta + epsilon (Grosshans-Grangier / GG02 protocol
    convention, hbar=2, shot-noise units); combine with :func:`loss` via
    :meth:`GaussianChannel.compose` to build a full line model.
    """
    if xi < 0:
        raise ValueError(f"xi must be >= 0, got {xi}")
    X = np.eye(2)
    Y = xi * (hbar / 2) * np.eye(2)
    return GaussianChannel(X, Y, hbar=hbar)


def additive_noise(xi_x, xi_p, hbar=2):
    """Anisotropic additive noise: independent noise xi_x, xi_p (shot-noise units) on x, p."""
    if xi_x < 0 or xi_p < 0:
        raise ValueError(f"xi_x, xi_p must be >= 0, got {xi_x}, {xi_p}")
    X = np.eye(2)
    Y = (hbar / 2) * np.diag([xi_x, xi_p])
    return GaussianChannel(X, Y, hbar=hbar)


def classical_noise(cov_noise, hbar=2):
    """Generic classical (additive Gaussian) noise channel with noise covariance ``cov_noise``."""
    cov_noise = np.asarray(cov_noise, dtype=float)
    return GaussianChannel(np.eye(cov_noise.shape[0]), cov_noise, hbar=hbar)


def thermal_channel(nbar, hbar=2):
    """Channel that resets a mode to a thermal state with mean photon number nbar (X=0, Y=vacuum-like)."""
    if nbar < 0:
        raise ValueError(f"nbar must be >= 0, got {nbar}")
    X = np.zeros((2, 2))
    Y = hbar * (nbar + 0.5) * np.eye(2)
    return GaussianChannel(X, Y, hbar=hbar)


# ---------------------------------------------------------------------------
# phase noise -- exact first- and second-moment update, NOT a GaussianChannel
# ---------------------------------------------------------------------------


def phase_noise(mean, cov, hbar, sigma, mode, mean_phase=0.0):
    """Apply Gaussian phase diffusion phi ~ N(mean_phase, sigma^2) to one mode.

    Averaging a Gaussian state over a randomly rotated phase produces a
    genuinely **non-Gaussian** mixture of rotated Gaussians (its Wigner
    function is a phase-space average of rotated Gaussians, not itself
    Gaussian) whenever the state is displaced or anisotropically squeezed.
    Because the map depends on the state's own mean, it cannot be written
    as a fixed (X, Y) :class:`GaussianChannel`.

    This function instead computes the **exact** first and second moments
    of the true (non-Gaussian) output state, using
    E[R(phi)] = exp(-sigma^2/2) R(mean_phase) and the analogous closed form
    for E[R(phi) M R(phi)^T] (Gaussian-averaged rotation of a symmetric
    2x2 M), then returns the Gaussian state with those moments -- the
    standard "Gaussian channel with phase noise" approximation used
    throughout the CV-QKD literature (e.g. Lodewyck et al., PRA 76, 042305
    (2007)), made explicit rather than silently assumed.

    Operates on raw (mean, cov) arrays and returns (new_mean, new_cov);
    :meth:`cvcircuit.state.GaussianState.phase_noise` wraps this.
    """
    mean = np.asarray(mean, dtype=float).copy()
    cov = np.asarray(cov, dtype=float).copy()
    n = mean.size // 2
    if not 0 <= mode < n:
        raise ValueError(f"mode {mode} out of range for {n}-mode state")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")

    i = 2 * mode
    r = mean[i : i + 2]
    sub = cov[i : i + 2, i : i + 2]

    c, s = np.cos(mean_phase), np.sin(mean_phase)
    rot = np.array([[c, -s], [s, c]])
    decay = np.exp(-sigma**2 / 2)
    e_rot = decay * rot  # E[R(phi)]

    # E[R M R^T] for symmetric 2x2 M: average = (tr M / 2) I + decay^4 * (rotated traceless part)
    def averaged_conjugate(M):
        half_trace = 0.5 * np.trace(M)
        traceless = M - half_trace * np.eye(2)
        return half_trace * np.eye(2) + np.exp(-2 * sigma**2) * (rot @ traceless @ rot.T)

    new_r = e_rot @ r
    new_sub = averaged_conjugate(sub) + averaged_conjugate(np.outer(r, r)) - np.outer(new_r, new_r)

    mean[i : i + 2] = new_r
    cov[i : i + 2, i : i + 2] = new_sub
    return mean, cov
