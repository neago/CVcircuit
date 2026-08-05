"""NumPy-only symplectic and interferometer decompositions.

No SciPy: the Williamson and Bloch-Messiah decompositions below are built
entirely from ``numpy.linalg.eigh``/``svd``, using the standard trick that a
real antisymmetric generator A becomes Hermitian once multiplied by i, so
its canonical block form can be read off from ``eigh(1j * A)`` instead of a
general (SciPy) Schur decomposition.  All three decompositions are verified
to machine precision by ``tests/test_decompositions.py`` against random
symplectic matrices and against reconstruction of the original input.

References: Weedbrook et al., RMP 84, 621 (2012), Sec. II.C.2 (Williamson)
and III.E (Bloch-Messiah); Reck et al., PRL 73, 58 (1994) (interferometer
mesh) -- see also Clements et al., Optica 3, 1460 (2016) for the
depth-optimal rectangular variant, which this module does not implement.
"""

from __future__ import annotations

import numpy as np

from .ops import symplectic_form

_ATOL = 1e-9


# ---------------------------------------------------------------------------
# symplectic eigenvalues
# ---------------------------------------------------------------------------


def symplectic_eigenvalues(cov, hbar=None):
    """Symplectic eigenvalues nu_k of a covariance matrix, sorted ascending.

    Computed as the positive members of the eigenvalues of i*Omega*cov,
    which are real for any physical covariance matrix.  In the same units
    as ``cov`` (vacuum gives nu_k = hbar/2 for every mode); ``hbar`` is
    unused and accepted only for a symmetric call signature with
    :func:`symplectic_eigenvalues_snu`.
    """
    del hbar
    n = cov.shape[-1] // 2
    omega = symplectic_form(n)
    eigvals = np.linalg.eigvals(1j * omega @ cov)
    eigvals = np.sort(np.abs(eigvals))
    return eigvals[0::2]


def symplectic_eigenvalues_snu(cov, hbar=2):
    """Symplectic eigenvalues in shot-noise units (vacuum = 1)."""
    return 2 * symplectic_eigenvalues(cov) / hbar


# ---------------------------------------------------------------------------
# Williamson decomposition
# ---------------------------------------------------------------------------


def williamson(cov, atol=_ATOL):
    """Williamson decomposition cov = S @ D @ S.T, S symplectic.

    D = diag(nu_1, nu_1, ..., nu_n, nu_n) holds the symplectic eigenvalues
    (xpxp order, same units as ``cov``).  Construction: with M = cov^(1/2)
    (symmetric square root via ``eigh``), the real antisymmetric generator
    A = M^-1 Omega M^-1 is block-diagonalized by a real orthogonal O built
    from the eigenvectors of the Hermitian matrix i*A; S = M @ O @ diag(sqrt(mu)).
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[-1] // 2
    if cov.shape != (2 * n, 2 * n):
        raise ValueError(f"cov must be square with even dimension, got shape {cov.shape}")
    if not np.allclose(cov, cov.T, atol=atol):
        raise ValueError("cov must be symmetric")

    w, v = np.linalg.eigh(cov)
    if np.any(w <= atol):
        raise ValueError("cov must be positive definite for a Williamson decomposition")
    msqrt = v @ np.diag(np.sqrt(w)) @ v.T
    minv = v @ np.diag(1 / np.sqrt(w)) @ v.T

    omega = symplectic_form(n)
    a_gen = minv @ omega @ minv  # real antisymmetric

    mu_all, u = np.linalg.eigh(1j * a_gen)
    order = np.argsort(mu_all)[::-1]
    u = u[:, order[:n]]  # eigenvectors for the n positive eigenvalues

    o = np.zeros((2 * n, 2 * n))
    o[:, 0::2] = np.sqrt(2) * u.real
    o[:, 1::2] = np.sqrt(2) * u.imag

    # fix the per-block sign so O.T A O has the +Omega orientation, i.e.
    # block[2k, 2k+1] > 0, by swapping the (x, p)-like columns if needed
    a_block = o.T @ a_gen @ o
    for k in range(n):
        if a_block[2 * k, 2 * k + 1] < 0:
            o[:, [2 * k, 2 * k + 1]] = o[:, [2 * k + 1, 2 * k]]
    a_block = o.T @ a_gen @ o
    mu = np.array([a_block[2 * k, 2 * k + 1] for k in range(n)])

    scale = np.repeat(np.sqrt(mu), 2)
    s_mat = (msqrt @ o) * scale[np.newaxis, :]
    nu = 1.0 / mu
    d_mat = np.diag(np.repeat(nu, 2))
    return s_mat, d_mat


# ---------------------------------------------------------------------------
# Bloch-Messiah / Euler decomposition
# ---------------------------------------------------------------------------


def bloch_messiah(matrix, atol=_ATOL):
    """Bloch-Messiah decomposition S = O1 @ Z @ O2 of a symplectic matrix.

    O1, O2 are orthogonal-symplectic (interferometers); Z is diagonal,
    holding reciprocal pairs (exp(r_k), exp(-r_k)) -- single-mode squeezers
    in the Williamson eigenbasis.  Construction: polar decomposition
    S = O_polar @ P (via SVD) gives the orthogonal part directly and a
    symmetric-positive-definite, symplectic P; P's eigenvectors (via
    ``eigh``) pair up as reciprocal-eigenvalue partners, which after a sign
    fix (so each pair is Omega-oriented) block-diagonalize P into Z.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[-1] // 2
    if matrix.shape != (2 * n, 2 * n):
        raise ValueError(f"matrix must be square with even dimension, got shape {matrix.shape}")

    u, sv, wt = np.linalg.svd(matrix)
    o_polar = u @ wt
    p_mat = wt.T @ np.diag(sv) @ wt  # symmetric, positive definite, symplectic

    d, v = np.linalg.eigh(p_mat)
    order = np.argsort(d)[::-1]
    d, v = d[order], v[:, order]

    omega = symplectic_form(n)
    o2_cols = np.zeros((2 * n, 2 * n))
    z = np.zeros(2 * n)
    for k in range(n):
        u_k, w_k = v[:, k], v[:, 2 * n - 1 - k]
        if u_k @ omega @ w_k < 0:
            w_k = -w_k
        o2_cols[:, 2 * k] = u_k
        o2_cols[:, 2 * k + 1] = w_k
        z[2 * k] = d[k]
        z[2 * k + 1] = d[2 * n - 1 - k]

    o1 = o_polar @ o2_cols
    o2 = o2_cols.T
    return o1, np.diag(z), o2


# ---------------------------------------------------------------------------
# interferometer mesh (Reck-style triangular decomposition)
# ---------------------------------------------------------------------------


def interferometer_mesh(unitary, atol=1e-9):
    """Decompose an n-mode unitary into a triangular mesh of beamsplitters
    and phase shifters (Reck et al., PRL 73, 58 (1994)).

    Returns a list of elements in physical application order, each either
    ``("phase", mode, angle)`` or ``("bs", mode1, mode2, theta, phi)``,
    where ``theta, phi`` are exactly the arguments expected by
    :func:`cvcircuit.ops.beamsplitter`.  Applying the elements in order
    (phases first, then beamsplitters) reconstructs ``unitary``:

        U == interferometer(compose of the returned elements)

    This is a triangular (Reck), not depth-optimal rectangular (Clements),
    mesh -- simpler to construct correctly and equally general; only the
    circuit depth differs.
    """
    u_mat = np.array(unitary, dtype=complex)
    n = u_mat.shape[0]
    if u_mat.shape != (n, n):
        raise ValueError(f"unitary must be square, got shape {u_mat.shape}")
    if not np.allclose(u_mat @ u_mat.conj().T, np.eye(n), atol=atol):
        raise ValueError("unitary must satisfy U U^dagger = I")

    def block(row1, row2, theta, phi):
        c, s = np.cos(theta), np.sin(theta)
        t_mat = np.eye(n, dtype=complex)
        t_mat[row1, row1] = c
        t_mat[row1, row2] = -s * np.exp(-1j * phi)
        t_mat[row2, row1] = s * np.exp(1j * phi)
        t_mat[row2, row2] = c
        return t_mat

    eliminations = []
    working = u_mat.copy()
    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            a, b = working[row - 1, col], working[row, col]
            if abs(b) < atol:
                theta, phi = 0.0, 0.0
            else:
                theta = np.arctan2(abs(b), abs(a))
                phi = np.angle(b) - np.angle(a) + np.pi
            working = block(row - 1, row, theta, phi) @ working
            eliminations.append((row - 1, row, theta, phi))
    output_phases = np.angle(np.diag(working))

    elements = [("phase", mode, float(angle)) for mode, angle in enumerate(output_phases)]
    for row1, row2, theta, phi in reversed(eliminations):
        # physical realization applies T^dagger = T(theta, phi + pi)
        elements.append(("bs", row1, row2, float(theta), float(phi + np.pi)))
    return elements
