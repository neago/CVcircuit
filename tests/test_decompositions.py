"""Williamson, Bloch-Messiah and interferometer-mesh decompositions: reconstruction
and structural round-trips against random symplectics/unitaries."""

import numpy as np
import pytest

from cvcircuit import decompositions as dec
from cvcircuit import ops


def random_cov(n, hbar=2, seed=0):
    rng = np.random.default_rng(seed)
    s_mat = ops.random_symplectic(n, seed=rng.integers(2**31), max_squeezing=1.2)
    nu = rng.uniform(hbar / 2, 2 * hbar, size=n)
    nu = np.maximum(nu, hbar / 2)
    d_mat = np.diag(np.repeat(nu, 2))
    return s_mat @ d_mat @ s_mat.T, nu


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_symplectic_eigenvalues_vacuum(n):
    hbar = 2.0
    nu = dec.symplectic_eigenvalues(hbar / 2 * np.eye(2 * n))
    assert np.allclose(nu, hbar / 2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_symplectic_eigenvalues_match_construction(n):
    cov, nu_true = random_cov(n, seed=n)
    nu = dec.symplectic_eigenvalues(cov)
    assert np.allclose(np.sort(nu), np.sort(nu_true), atol=1e-8)


def test_symplectic_eigenvalues_snu_vacuum_is_one():
    for hbar in (1, 2):
        nu_snu = dec.symplectic_eigenvalues_snu(hbar / 2 * np.eye(4), hbar=hbar)
        assert np.allclose(nu_snu, 1.0)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_williamson_reconstructs_and_is_symplectic(n):
    cov, _ = random_cov(n, seed=100 + n)
    s_mat, d_mat = dec.williamson(cov)
    assert np.allclose(s_mat @ d_mat @ s_mat.T, cov, atol=1e-8)
    assert ops.is_symplectic(s_mat, atol=1e-7)
    assert np.all(np.diag(d_mat) > 0)


def test_williamson_vacuum_gives_hbar_half():
    cov = 1.0 * np.eye(4)  # hbar=2 vacuum, 2 modes
    _, d_mat = dec.williamson(cov)
    assert np.allclose(np.diag(d_mat), 1.0)


def test_williamson_not_positive_definite_raises():
    with pytest.raises(ValueError, match="positive definite"):
        dec.williamson(np.zeros((2, 2)))


def test_williamson_not_square_raises():
    with pytest.raises(ValueError, match="square"):
        dec.williamson(np.eye(3))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_bloch_messiah_reconstructs(n):
    s_mat = ops.random_symplectic(n, seed=200 + n, max_squeezing=1.5)
    o1, z_mat, o2 = dec.bloch_messiah(s_mat)
    assert np.allclose(o1 @ z_mat @ o2, s_mat, atol=1e-8)
    assert ops.is_symplectic(o1, atol=1e-7)
    assert ops.is_symplectic(o2, atol=1e-7)
    assert np.allclose(o1 @ o1.T, np.eye(2 * n), atol=1e-7)
    assert np.allclose(o2 @ o2.T, np.eye(2 * n), atol=1e-7)


def test_bloch_messiah_identity():
    o1, z_mat, o2 = dec.bloch_messiah(np.eye(4))
    assert np.allclose(o1 @ z_mat @ o2, np.eye(4), atol=1e-8)
    assert np.allclose(np.diag(z_mat), 1.0)


def test_bloch_messiah_pure_squeezing():
    s_mat = np.eye(4)
    s_mat[0:2, 0:2] = ops.squeezing(0.8)
    s_mat[2:4, 2:4] = ops.squeezing(0.3)
    o1, z_mat, o2 = dec.bloch_messiah(s_mat)
    assert np.allclose(o1 @ z_mat @ o2, s_mat, atol=1e-8)
    z_sorted = np.sort(np.diag(z_mat))
    expected = np.sort([np.exp(-0.8), np.exp(0.8), np.exp(-0.3), np.exp(0.3)])
    assert np.allclose(z_sorted, expected, atol=1e-8)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_interferometer_mesh_reconstructs(n):
    u_mat = ops.random_interferometer(n, seed=300 + n)
    x_idx, p_idx = np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)
    u_complex = u_mat[np.ix_(x_idx, x_idx)] + 1j * u_mat[np.ix_(p_idx, x_idx)]

    elements = dec.interferometer_mesh(u_complex)
    s_total = np.eye(2 * n)
    for el in elements:
        if el[0] == "phase":
            _, mode, angle = el
            local = ops.rotation(angle)
        else:
            _, m1, m2, theta, phi = el
            local = ops.beamsplitter(theta, phi)
            mode = (m1, m2)
        modes = [mode] if isinstance(mode, int) else list(mode)
        s_total = ops.expand(local, modes, n) @ s_total

    assert np.allclose(s_total, ops.interferometer(u_complex), atol=1e-7)


def test_interferometer_mesh_element_count():
    n = 4
    u_mat = ops.random_interferometer(n, seed=1)
    x_idx, p_idx = np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)
    u_complex = u_mat[np.ix_(x_idx, x_idx)] + 1j * u_mat[np.ix_(p_idx, x_idx)]
    elements = dec.interferometer_mesh(u_complex)
    n_phase = sum(1 for e in elements if e[0] == "phase")
    n_bs = sum(1 for e in elements if e[0] == "bs")
    assert n_phase == n
    assert n_bs == n * (n - 1) // 2


def test_interferometer_mesh_requires_unitary():
    with pytest.raises(ValueError, match="unitary"):
        dec.interferometer_mesh(np.array([[1, 1], [0, 1]], dtype=complex))
