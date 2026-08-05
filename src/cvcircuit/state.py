"""GaussianState: the core value object, plus its constructors.

A GaussianState is fully described by a mean vector and covariance matrix
in **xpxp** order, r = (x1, p1, ..., xn, pn), together with the value of
hbar used to define x, p and the uncertainty bound.  See
:mod:`docs/conventions.md` for the full formalism.

GaussianState is **immutable**: every method below (``squeeze``,
``beamsplitter``, ``loss``, ``homodyne``, ...) returns a *new* state and
leaves ``self`` untouched, even the ones that read like in-place verbs.
"""

from __future__ import annotations

import numpy as np

from . import channels as _channels
from . import conversions
from . import decompositions
from . import measure as _measure
from . import ops

_ATOL = 1e-8


class GaussianState:
    """A multimode Gaussian state, given by its mean and covariance (xpxp order).

    Parameters
    ----------
    mean:
        Length-2n vector (x1, p1, ..., xn, pn).
    cov:
        2n x 2n symmetric covariance matrix satisfying the uncertainty
        relation cov + i(hbar/2) Omega >= 0 (vacuum: cov = (hbar/2) I).
    hbar:
        [x, p] = i hbar.  Default 2 (vacuum variance 1, shot-noise units);
        pass ``hbar=1`` to match the Brask/Ferraro-Olivares-Paris convention
        used in much of the theory literature.
    """

    def __init__(self, mean, cov, hbar=2):
        mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if mean.ndim != 1 or mean.size % 2 != 0:
            raise ValueError(f"mean must be a 1-D vector of even length, got shape {mean.shape}")
        n = mean.size // 2
        if cov.shape != (2 * n, 2 * n):
            raise ValueError(f"cov has shape {cov.shape}, expected {(2 * n, 2 * n)} to match {n}-mode mean")
        if not np.allclose(cov, cov.T, atol=_ATOL):
            raise ValueError("cov must be symmetric")
        omega = ops.symplectic_form(n)
        uncertainty = cov + 1j * (hbar / 2) * omega
        min_eig = np.linalg.eigvalsh(uncertainty).min()
        if min_eig < -_ATOL:
            raise ValueError(
                f"cov violates the uncertainty relation sigma + i(hbar/2)Omega >= 0 "
                f"(minimum eigenvalue {min_eig:.3g}); is hbar={hbar} correct for this cov?"
            )
        self.mean = mean
        self.cov = cov
        self.hbar = float(hbar)
        self.num_modes = n

    # -- validation -----------------------------------------------------

    def _check_mode(self, mode):
        if not 0 <= mode < self.num_modes:
            raise ValueError(f"mode {mode} out of range for {self.num_modes}-mode state")

    def _check_modes(self, modes):
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        for m in modes:
            self._check_mode(m)
        if len(set(modes.tolist())) != modes.size:
            raise ValueError(f"modes must be distinct, got {modes.tolist()}")
        return modes

    # -- combining and reducing -----------------------------------------

    def __and__(self, other):
        """Tensor product: ``a & b`` puts b's modes after a's."""
        if not isinstance(other, GaussianState):
            return NotImplemented
        if self.hbar != other.hbar:
            raise ValueError(f"cannot combine states with different hbar ({self.hbar} vs {other.hbar})")
        mean = np.concatenate([self.mean, other.mean])
        n1, n2 = self.num_modes, other.num_modes
        cov = np.zeros((2 * (n1 + n2), 2 * (n1 + n2)))
        cov[: 2 * n1, : 2 * n1] = self.cov
        cov[2 * n1 :, 2 * n1 :] = other.cov
        return GaussianState(mean, cov, hbar=self.hbar)

    def reduced(self, modes):
        """The reduced state of ``modes`` (a partial trace over the rest)."""
        modes = self._check_modes(modes)
        idx = ops.mode_indices(modes)
        return GaussianState(self.mean[idx], self.cov[np.ix_(idx, idx)], hbar=self.hbar)

    partial_trace = reduced

    def permute(self, order):
        """Reorder modes according to ``order`` (a permutation of range(num_modes))."""
        order = self._check_modes(order)
        if sorted(order.tolist()) != list(range(self.num_modes)):
            raise ValueError(f"order must be a permutation of range({self.num_modes}), got {order.tolist()}")
        idx = ops.mode_indices(order)
        return GaussianState(self.mean[idx], self.cov[np.ix_(idx, idx)], hbar=self.hbar)

    # -- unitary (symplectic) operations ---------------------------------

    def _apply_symplectic(self, local_matrix, modes, local_d=None):
        modes = self._check_modes(modes)
        s_full = ops.expand(local_matrix, modes, self.num_modes)
        mean = s_full @ self.mean
        if local_d is not None:
            mean = mean + ops.expand_vector(local_d, modes, self.num_modes)
        cov = s_full @ self.cov @ s_full.T
        return GaussianState(mean, cov, hbar=self.hbar)

    def displace(self, alpha, mode=0):
        """Displace ``mode`` by complex amplitude alpha: D(alpha)."""
        d = ops.displacement(alpha, hbar=self.hbar)
        return self._apply_symplectic(np.eye(2), [mode], local_d=d)

    def displace_xp(self, x, p, mode=0):
        """Displace ``mode`` by (x, p) directly (rather than a complex alpha)."""
        return self._apply_symplectic(np.eye(2), [mode], local_d=np.array([x, p], dtype=float))

    def rotate(self, phi, mode=0):
        """Phase rotation R(phi) on ``mode`` (a -> a exp(i phi))."""
        return self._apply_symplectic(ops.rotation(phi), [mode])

    def fourier(self, mode=0):
        """Fourier gate (90-degree rotation) on ``mode``."""
        return self._apply_symplectic(ops.fourier(), [mode])

    def squeeze(self, r, phi=0.0, mode=0):
        """Single-mode squeezing S(r, phi) on ``mode``."""
        return self._apply_symplectic(ops.squeezing(r, phi), [mode])

    def shear(self, s=1.0, mode=0):
        """Shear/P-gate on ``mode``: p -> p + s x."""
        return self._apply_symplectic(ops.shear(s), [mode])

    def beamsplitter(self, theta, modes=(0, 1), phi=0.0):
        """Beamsplitter with mixing angle theta (T = cos^2(theta)) on ``modes``."""
        return self._apply_symplectic(ops.beamsplitter(theta, phi), modes)

    def two_mode_squeeze(self, r, modes=(0, 1), phi=0.0):
        """Two-mode squeezing on ``modes``."""
        return self._apply_symplectic(ops.two_mode_squeezing(r, phi), modes)

    def interferometer(self, unitary, modes=None):
        """Apply a passive n-mode interferometer (unitary matrix) to ``modes``."""
        unitary = np.asarray(unitary)
        if modes is None:
            modes = range(unitary.shape[0])
        return self._apply_symplectic(ops.interferometer(unitary), modes)

    def cz(self, s=1.0, modes=(0, 1)):
        """CV controlled-Z gate, edge weight s, on ``modes``."""
        return self._apply_symplectic(ops.cz(s), modes)

    def cx(self, s=1.0, modes=(0, 1)):
        """CV controlled-X / SUM gate, gain s, on ``modes``."""
        return self._apply_symplectic(ops.cx(s), modes)

    def apply_symplectic(self, matrix, modes=None, d=None):
        """Apply an arbitrary symplectic ``matrix`` (and optional displacement ``d``)."""
        matrix = np.asarray(matrix, dtype=float)
        if modes is None:
            modes = range(matrix.shape[0] // 2)
        if not ops.is_symplectic(matrix):
            raise ValueError("matrix is not symplectic (S Omega S^T != Omega)")
        return self._apply_symplectic(matrix, modes, local_d=d)

    # -- channels ---------------------------------------------------------

    def apply_channel(self, channel, modes):
        """Apply a :class:`~cvcircuit.channels.GaussianChannel` to ``modes``."""
        modes = self._check_modes(modes)
        if channel.num_modes != modes.size:
            raise ValueError(f"channel acts on {channel.num_modes} mode(s), got {modes.size} target mode(s)")
        x_full = ops.expand(channel.X, modes, self.num_modes)
        idx = ops.mode_indices(modes)
        y_full = np.zeros_like(self.cov)
        y_full[np.ix_(idx, idx)] = channel.Y
        d_full = ops.expand_vector(channel.d, modes, self.num_modes)
        mean = x_full @ self.mean + d_full
        cov = x_full @ self.cov @ x_full.T + y_full
        return GaussianState(mean, cov, hbar=self.hbar)

    def loss(self, eta, mode=0, nbar_env=0.0):
        """Pure loss channel (transmittance eta) on ``mode``."""
        return self.apply_channel(_channels.loss(eta, nbar_env, hbar=self.hbar), [mode])

    def amplify(self, g, mode=0, nbar_env=0.0):
        """Phase-insensitive amplifier channel (gain g) on ``mode``."""
        return self.apply_channel(_channels.amplifier(g, nbar_env, hbar=self.hbar), [mode])

    def add_noise(self, xi, mode=0):
        """Isotropic additive (excess) noise channel, xi in shot-noise units, on ``mode``."""
        return self.apply_channel(_channels.excess_noise(xi, hbar=self.hbar), [mode])

    def phase_noise(self, sigma, mode=0, mean_phase=0.0):
        """Gaussian phase diffusion phi ~ N(mean_phase, sigma^2) on ``mode``.

        Tracks the exact first and second moments of the resulting
        (non-Gaussian) state; see :func:`cvcircuit.channels.phase_noise`.
        """
        mean, cov = _channels.phase_noise(self.mean, self.cov, self.hbar, sigma, mode, mean_phase=mean_phase)
        return GaussianState(mean, cov, hbar=self.hbar)

    # -- measurements -------------------------------------------------------

    def homodyne(self, mode=0, angle=0.0, efficiency=1.0, electronic_noise=0.0, seed=None):
        """Homodyne measurement of the quadrature at ``angle`` on ``mode``.

        Returns ``(outcome, conditional_state)``.  The measured mode is
        reset to vacuum and kept, so mode indices are unchanged.
        """
        outcome, mean, cov = _measure.homodyne(
            self.mean, self.cov, self.hbar, mode, angle, efficiency=efficiency,
            electronic_noise=electronic_noise, seed=seed,
        )
        return outcome, GaussianState(mean, cov, hbar=self.hbar)

    def heterodyne(self, mode=0, seed=None):
        """Heterodyne measurement on ``mode``; returns ``(alpha, conditional_state)``."""
        outcome, mean, cov = _measure.heterodyne(self.mean, self.cov, self.hbar, mode, seed=seed)
        return outcome, GaussianState(mean, cov, hbar=self.hbar)

    def generaldyne(self, modes, cov_povm, seed=None):
        """General-dyne measurement of ``modes`` against POVM covariance ``cov_povm``."""
        outcome, mean, cov = _measure.generaldyne(self.mean, self.cov, self.hbar, modes, cov_povm, seed=seed)
        return outcome, GaussianState(mean, cov, hbar=self.hbar)

    # -- scalar properties ----------------------------------------------

    def symplectic_eigenvalues(self):
        """Symplectic eigenvalues, natural units (vacuum: hbar/2 per mode)."""
        return decompositions.symplectic_eigenvalues(self.cov)

    def symplectic_eigenvalues_snu(self):
        """Symplectic eigenvalues in shot-noise units (vacuum: 1 per mode)."""
        return decompositions.symplectic_eigenvalues_snu(self.cov, self.hbar)

    def williamson(self):
        """Williamson decomposition (S, D) of the covariance matrix; see :func:`cvcircuit.decompositions.williamson`."""
        return decompositions.williamson(self.cov)

    def purity(self):
        """Tr(rho^2) = 1 / prod(symplectic eigenvalues in SNU)."""
        return float(1.0 / np.prod(self.symplectic_eigenvalues_snu()))

    def is_pure(self, atol=1e-6):
        """Whether the state is (numerically) pure."""
        return bool(abs(self.purity() - 1.0) < atol)

    def is_classical(self, atol=1e-8):
        """Whether cov - (hbar/2) I >= 0 (a proper P-representation exists)."""
        eigs = np.linalg.eigvalsh(self.cov - (self.hbar / 2) * np.eye(2 * self.num_modes))
        return bool(eigs.min() >= -atol)

    def von_neumann_entropy(self, base=2):
        """Von Neumann entropy S = -Tr(rho log rho), in units of ``base`` (default bits)."""
        nu = self.symplectic_eigenvalues_snu()
        p, m = (nu + 1) / 2, (nu - 1) / 2
        safe_m = np.where(m > 1e-12, m, 1.0)  # avoid log(0); the where() below discards it anyway
        term_m = np.where(m > 1e-12, m * np.log(safe_m), 0.0)
        entropy_nats = np.sum(p * np.log(p) - term_m)
        return float(entropy_nats / np.log(base))

    def renyi_entropy(self, alpha=2, base=2):
        """Renyi-alpha entropy; alpha=2 reduces to -log(purity)."""
        if alpha == 1:
            return self.von_neumann_entropy(base=base)
        nu = self.symplectic_eigenvalues_snu()
        p, m = (nu + 1) / 2, (nu - 1) / 2
        term = p**alpha - np.where(m > 0, m**alpha, 0.0)
        entropy_nats = np.sum(np.log(term)) / (alpha - 1)
        return float(entropy_nats / np.log(base))

    def mean_photon_number(self, mode=None):
        """Mean photon number of ``mode`` (or the total, if ``mode`` is None)."""
        if mode is None:
            return float(sum(self.mean_photon_number(m) for m in range(self.num_modes)))
        self._check_mode(mode)
        sub = self.reduced([mode])
        return float((np.trace(sub.cov) + sub.mean @ sub.mean) / (2 * self.hbar) - 0.5)

    def photon_number_variance(self, mode=0):
        """Variance of the photon number operator for ``mode``.

        Var(n) = nbar'(nbar'+1) + |m'|^2 + 2 Re(alpha*^2 m') + |alpha|^2 (2 nbar' + 1),
        where nbar', m' are the mean-subtracted number and anomalous
        (squeezing) moments and alpha is the coherent displacement.
        Reduces to Var(n) = nbar(nbar+1) for thermal states and Var(n) =
        <n> for coherent states (Poissonian).
        """
        sub = self.reduced([mode])
        hbar = self.hbar
        xx, pp, xp = sub.cov[0, 0], sub.cov[1, 1], sub.cov[0, 1]
        nbar_fluct = (xx + pp) / (2 * hbar) - 0.5
        m = ((xx - pp) + 2j * xp) / (2 * hbar)
        alpha = (sub.mean[0] + 1j * sub.mean[1]) / np.sqrt(2 * hbar)
        var = (
            nbar_fluct * (nbar_fluct + 1)
            + abs(m) ** 2
            + 2 * np.real(np.conj(alpha) ** 2 * m)
            + abs(alpha) ** 2 * (2 * nbar_fluct + 1)
        )
        return float(var)

    def quad_mean(self, mode, angle=0.0):
        """Mean of the quadrature at ``angle`` on ``mode``."""
        sub = self.reduced([mode])
        direction = np.array([np.cos(angle), np.sin(angle)])
        return float(direction @ sub.mean)

    def quad_variance(self, mode, angle=0.0):
        """Variance of the quadrature at ``angle`` on ``mode``."""
        sub = self.reduced([mode])
        direction = np.array([np.cos(angle), np.sin(angle)])
        return float(direction @ sub.cov @ direction)

    def squeezing_db(self, mode=0):
        """Squeezing of ``mode``'s best quadrature relative to vacuum, in dB (negative = squeezed)."""
        sub_cov = self.reduced([mode]).cov
        min_var = np.linalg.eigvalsh(sub_cov).min()
        return float(10 * np.log10(min_var / (self.hbar / 2)))

    # -- phase space ------------------------------------------------------

    def quasiprobability(self, x, p, s=0.0, mode=0):
        """The s-ordered quasiprobability distribution of ``mode`` at (x, p).

        s=0: Wigner function (default).  s=-1: Husimi Q function.  s=+1:
        Glauber-Sudarshan P function -- only a genuine function when
        cov - s(hbar/2)I is positive definite (i.e. the state is
        "classical" in Mandel's sense); raises otherwise.
        """
        sub = self.reduced([mode])
        sigma_s = sub.cov - s * (self.hbar / 2) * np.eye(2)
        det = np.linalg.det(sigma_s)
        if det <= 0:
            raise ValueError(
                f"the s={s} quasiprobability is not a valid function for this state "
                f"(cov - s*hbar/2*I is not positive definite); this always happens for "
                f"s=1 (the P function) unless the state is classical (see is_classical())"
            )
        inv = np.linalg.inv(sigma_s)
        x, p = np.asarray(x, dtype=float), np.asarray(p, dtype=float)
        dx, dp = x - sub.mean[0], p - sub.mean[1]
        exponent = -0.5 * (inv[0, 0] * dx**2 + 2 * inv[0, 1] * dx * dp + inv[1, 1] * dp**2)
        norm = 1.0 / (2 * np.pi * np.sqrt(det))
        return norm * np.exp(exponent)

    def wigner(self, x, p, mode=0):
        """The Wigner function of ``mode`` at (x, p)."""
        return self.quasiprobability(x, p, s=0.0, mode=mode)

    def husimi(self, x, p, mode=0):
        """The Husimi Q function of ``mode`` at (x, p)."""
        return self.quasiprobability(x, p, s=-1.0, mode=mode)

    def p_function(self, x, p, mode=0):
        """The Glauber-Sudarshan P function of ``mode`` at (x, p) (raises if non-classical)."""
        return self.quasiprobability(x, p, s=1.0, mode=mode)

    def marginal(self, mode, angle=0.0, xvec=None):
        """The 1-D probability density of the quadrature at ``angle`` on ``mode``.

        Returns ``(xvec, pdf)``; if ``xvec`` is not given, a default range
        spanning +/-6 standard deviations around the mean is used.
        """
        mean_theta = self.quad_mean(mode, angle)
        var_theta = self.quad_variance(mode, angle)
        if xvec is None:
            std = np.sqrt(var_theta)
            xvec = np.linspace(mean_theta - 6 * std, mean_theta + 6 * std, 401)
        else:
            xvec = np.asarray(xvec, dtype=float)
        pdf = np.exp(-((xvec - mean_theta) ** 2) / (2 * var_theta)) / np.sqrt(2 * np.pi * var_theta)
        return xvec, pdf

    # -- interop ------------------------------------------------------------

    def to_xxpp(self):
        """Return (mean, cov) reordered to xxpp block order."""
        return conversions.xpxp_to_xxpp(self.mean), conversions.xpxp_to_xxpp(self.cov)

    @classmethod
    def from_xxpp(cls, mean, cov, hbar=2):
        """Construct from (mean, cov) given in xxpp block order."""
        return cls(conversions.xxpp_to_xpxp(mean), conversions.xxpp_to_xpxp(cov), hbar=hbar)

    def copy(self):
        """A copy of this state (states are immutable, so this is rarely needed)."""
        return GaussianState(self.mean.copy(), self.cov.copy(), hbar=self.hbar)

    def __repr__(self):
        return f"GaussianState({self.num_modes} mode(s), hbar={self.hbar:g}, purity={self.purity():.4g})"


# ---------------------------------------------------------------------------
# constructors
# ---------------------------------------------------------------------------


def vacuum(n=1, hbar=2):
    """The n-mode vacuum state."""
    return GaussianState(np.zeros(2 * n), (hbar / 2) * np.eye(2 * n), hbar=hbar)


def coherent(alpha, hbar=2):
    """A coherent state |alpha> (or a product of coherent states, if alpha is an array)."""
    mean = conversions.alpha_to_xp(alpha, hbar=hbar)
    n = mean.size // 2
    return GaussianState(mean, (hbar / 2) * np.eye(2 * n), hbar=hbar)


def squeezed(r, phi=0.0, hbar=2):
    """A single-mode squeezed vacuum state S(r, phi)|0>."""
    s_mat = ops.squeezing(r, phi)
    cov = s_mat @ ((hbar / 2) * np.eye(2)) @ s_mat.T
    return GaussianState(np.zeros(2), cov, hbar=hbar)

def squeezed_coherent(alpha, r, phi=0.0, hbar=2):
    """A displaced squeezed vacuum state D(alpha) S(r, phi)|0>."""
    s_mat = ops.squeezing(r, phi)
    cov = s_mat @ ((hbar / 2) * np.eye(2)) @ s_mat.T
    mean = conversions.alpha_to_xp(alpha, hbar=hbar)
    return GaussianState(mean, cov, hbar=hbar)


def thermal(nbar, hbar=2):
    """A thermal state with mean photon number nbar (or an array, for several modes)."""
    nbar = np.atleast_1d(np.asarray(nbar, dtype=float))
    n = nbar.size
    cov = hbar * np.diag(np.repeat(nbar + 0.5, 2))
    return GaussianState(np.zeros(2 * n), cov, hbar=hbar)


def two_mode_squeezed(r, phi=0.0, hbar=2):
    """A two-mode squeezed vacuum (TMSV) state, entangling two fresh modes."""
    s_mat = ops.two_mode_squeezing(r, phi)
    cov = s_mat @ ((hbar / 2) * np.eye(4)) @ s_mat.T
    return GaussianState(np.zeros(4), cov, hbar=hbar)


def epr(r, hbar=2):
    """An EPR pair: alias for :func:`two_mode_squeezed` with phi=0 (approaches a
    perfect EPR state as r -> infinity)."""
    return two_mode_squeezed(r, phi=0.0, hbar=hbar)


def tensor(*states):
    """Tensor product of several :class:`GaussianState` objects, in order."""
    if len(states) == 1 and not isinstance(states[0], GaussianState):
        states = tuple(states[0])
    if not states:
        raise ValueError("tensor() requires at least one state")
    result = states[0]
    for state in states[1:]:
        result = result & state
    return result


def graph_state(adjacency, r=10.0, hbar=2):
    """An (approximate, finite-squeezing) CV graph/cluster state.

    Built as the standard Menicucci et al. (PRL 97, 110501 (2006))
    construction: p-squeezed vacua (squeezing r; p-squeezing is
    ``squeeze(r, phi=pi)`` in this library's convention) connected by CZ
    gates with weights taken from ``adjacency``.  Exact graph states
    require r -> infinity; finite r gives the usual finite-squeezing
    approximation used in all real proposals and experiments.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    if adjacency.shape != (n, n) or not np.allclose(adjacency, adjacency.T):
        raise ValueError("adjacency must be a square symmetric matrix")
    state = tensor(*[squeezed(r, phi=np.pi, hbar=hbar) for _ in range(n)])
    for i in range(n):
        for j in range(i + 1, n):
            weight = adjacency[i, j]
            if weight != 0:
                state = state.cz(weight, modes=(i, j))
    return state


cluster_state = graph_state


def random_state(n, pure=False, seed=None, hbar=2, max_squeezing=1.0, displacement_scale=1.0):
    """A random n-mode Gaussian state, for tests and demos (not physically motivated)."""
    rng = np.random.default_rng(seed)
    s_mat = ops.random_symplectic(n, seed=rng.integers(2**31), max_squeezing=max_squeezing)
    if pure:
        base_cov = (hbar / 2) * np.eye(2 * n)
    else:
        nbar = rng.uniform(0, 1, size=n)
        base_cov = hbar * np.diag(np.repeat(nbar + 0.5, 2))
    cov = s_mat @ base_cov @ s_mat.T
    mean = rng.normal(scale=displacement_scale * np.sqrt(hbar), size=2 * n)
    return GaussianState(mean, cov, hbar=hbar)
