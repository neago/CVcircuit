"""An independent, truncated-Fock-space Gaussian-state simulator.

Not a test file itself (no ``test_`` functions) -- a deliberately
different implementation used *by* the tests to cross-check the Gaussian
(covariance-matrix) formalism in :mod:`cvcircuit`.  Everything here is
built from explicit annihilation/creation operator matrices at a Fock
cutoff and matrix-exponentiated unitary generators; agreement between the
first/second moments computed this way and the Gaussian formalism's own
covariance matrices is strong evidence of correctness, since the two
methods share essentially no code.

Generator conventions (verified by hand against ``cvcircuit.ops`` during
development; also re-checked by ``test_reference.py``):

* rotation(phi):        U = exp(i phi a_dag a)
* squeezing(r, phi):     U = exp(0.5 (xi* a^2 - xi a_dag^2)), xi = r exp(i phi)
* displacement(alpha):   U = exp(alpha a_dag - alpha* a)
* beamsplitter(theta, phi): U = exp(theta (e^{i phi} a b_dag - e^{-i phi} a_dag b))
* two_mode_squeezing(r, phi): U = exp(r (e^{i phi} a_dag b_dag - e^{-i phi} a b))
"""

import numpy as np


def fock_operators(cutoff):
    """Annihilation operator a (cutoff x cutoff) and the identity, single mode."""
    a = np.diag(np.sqrt(np.arange(1, cutoff)), k=1)
    return a, np.eye(cutoff)


def _expm_anti_hermitian(generator):
    """exp(A) for anti-Hermitian A, via eigh on the Hermitian A/i (numerically robust)."""
    hermitian = generator / 1j
    if not np.allclose(hermitian, hermitian.conj().T, atol=1e-9):
        raise ValueError("generator must be anti-Hermitian")
    eigvals, eigvecs = np.linalg.eigh(hermitian)
    return (eigvecs * np.exp(1j * eigvals)) @ eigvecs.conj().T


class FockSimulator:
    """A single- or two-mode Fock-space state, with the operations needed for cross-checks.

    Parameters
    ----------
    num_modes:
        1 or 2.
    cutoff:
        Fock-space truncation dimension per mode; state vectors have
        length ``cutoff**num_modes``.
    hbar:
        Must match the :mod:`cvcircuit` state being compared against.
    """

    def __init__(self, num_modes, cutoff=40, hbar=2):
        if num_modes not in (1, 2):
            raise ValueError(f"num_modes must be 1 or 2, got {num_modes}")
        self.num_modes = num_modes
        self.cutoff = cutoff
        self.hbar = hbar
        a1, i1 = fock_operators(cutoff)
        if num_modes == 1:
            self.a = [a1]
        else:
            self.a = [np.kron(a1, i1), np.kron(i1, a1)]
        self.adag = [op.conj().T for op in self.a]
        self.dim = cutoff**num_modes
        self.identity = np.eye(self.dim)
        self.psi = np.zeros(self.dim, dtype=complex)
        self.psi[0] = 1.0  # vacuum

    # -- state preparation -------------------------------------------------

    def displace(self, alpha, mode=0):
        gen = alpha * self.adag[mode] - np.conj(alpha) * self.a[mode]
        self.psi = _expm_anti_hermitian(gen) @ self.psi
        return self

    def rotate(self, phi, mode=0):
        n_op = self.adag[mode] @ self.a[mode]
        gen = 1j * phi * n_op
        self.psi = _expm_anti_hermitian(gen) @ self.psi
        return self

    def squeeze(self, r, phi=0.0, mode=0):
        xi = r * np.exp(1j * phi)
        a, adag = self.a[mode], self.adag[mode]
        gen = 0.5 * (np.conj(xi) * (a @ a) - xi * (adag @ adag))
        self.psi = _expm_anti_hermitian(gen) @ self.psi
        return self

    def beamsplitter(self, theta, phi=0.0):
        if self.num_modes != 2:
            raise ValueError("beamsplitter requires a 2-mode simulator")
        a, b = self.a
        adag, bdag = self.adag
        gen = theta * (np.exp(1j * phi) * a @ bdag - np.exp(-1j * phi) * adag @ b)
        self.psi = _expm_anti_hermitian(gen) @ self.psi
        return self

    def two_mode_squeeze(self, r, phi=0.0):
        if self.num_modes != 2:
            raise ValueError("two_mode_squeeze requires a 2-mode simulator")
        a, b = self.a
        adag, bdag = self.adag
        gen = r * (np.exp(1j * phi) * adag @ bdag - np.exp(-1j * phi) * a @ b)
        self.psi = _expm_anti_hermitian(gen) @ self.psi
        return self

    def density_matrix(self):
        return np.outer(self.psi, self.psi.conj())

    # -- moments and reduction ----------------------------------------------

    def _quad_ops(self, mode):
        a, adag = self.a[mode], self.adag[mode]
        x_op = np.sqrt(self.hbar / 2) * (a + adag)
        p_op = -1j * np.sqrt(self.hbar / 2) * (a - adag)
        return x_op, p_op

    def moments(self):
        """(mean, cov) in xpxp order, from the current (pure) state vector."""
        quad_ops = []
        for mode in range(self.num_modes):
            quad_ops.extend(self._quad_ops(mode))
        mean = np.array([np.real(self.psi.conj() @ op @ self.psi) for op in quad_ops])
        n = 2 * self.num_modes
        cov = np.zeros((n, n))
        centered = [op - mean[i] * self.identity for i, op in enumerate(quad_ops)]
        for i in range(n):
            for j in range(n):
                anticomm = centered[i] @ centered[j] + centered[j] @ centered[i]
                cov[i, j] = np.real(self.psi.conj() @ (0.5 * anticomm) @ self.psi)
        return mean, cov

    def reduced_moments(self, mode):
        """(mean, cov) of a single mode of a 2-mode state (partial trace, mixed in general)."""
        if self.num_modes != 2:
            raise ValueError("reduced_moments requires a 2-mode simulator")
        psi_mat = self.psi.reshape(self.cutoff, self.cutoff)
        rho = psi_mat @ psi_mat.conj().T if mode == 0 else psi_mat.T @ psi_mat.conj()
        a1, i1 = fock_operators(self.cutoff)
        adag1 = a1.conj().T
        x_op = np.sqrt(self.hbar / 2) * (a1 + adag1)
        p_op = -1j * np.sqrt(self.hbar / 2) * (a1 - adag1)
        mean_x = np.real(np.trace(rho @ x_op))
        mean_p = np.real(np.trace(rho @ p_op))
        dx, dp = x_op - mean_x * i1, p_op - mean_p * i1
        cov = np.array(
            [
                [np.real(np.trace(rho @ (dx @ dx))), np.real(np.trace(rho @ (0.5 * (dx @ dp + dp @ dx))))],
                [np.real(np.trace(rho @ (0.5 * (dx @ dp + dp @ dx)))), np.real(np.trace(rho @ (dp @ dp)))],
            ]
        )
        return np.array([mean_x, mean_p]), cov

    def loss(self, eta, mode=0, ancilla_cutoff=None):
        """Pure loss on ``mode`` via a beamsplitter with a vacuum ancilla + partial trace.

        Only supported starting from a 1-mode simulator (the ancilla
        becomes mode 1); returns the reduced (mean, cov) of the signal
        mode after tracing out the ancilla.
        """
        if self.num_modes != 1:
            raise ValueError("loss() cross-check requires a 1-mode simulator (ancilla becomes mode 1)")
        two_mode = FockSimulator(2, cutoff=self.cutoff, hbar=self.hbar)
        two_mode.psi = np.kron(self.psi, np.eye(self.cutoff)[0])  # signal (x) vacuum ancilla
        theta = np.arccos(np.sqrt(eta))
        two_mode.beamsplitter(theta, phi=0.0)
        return two_mode.reduced_moments(0)
