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


def random_unitary_complex(n, seed=0):
    """A random n x n unitary, extracted from ops.random_interferometer's xpxp embedding."""
    u_mat = ops.random_interferometer(n, seed=seed)
    x_idx, p_idx = np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)
    return u_mat[np.ix_(x_idx, x_idx)] + 1j * u_mat[np.ix_(p_idx, x_idx)]


def mesh_to_symplectic(elements, n):
    """Compose a mesh (interferometer_mesh/clements_decomposition output) into its xpxp symplectic."""
    s_total = np.eye(2 * n)
    for el in elements:
        if el[0] == "phase":
            _, mode, angle = el
            local, modes = ops.rotation(angle), [mode]
        else:
            _, m1, m2, theta, phi = el
            local, modes = ops.beamsplitter(theta, phi), [m1, m2]
        s_total = ops.expand(local, modes, n) @ s_total
    return s_total


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
    u_complex = random_unitary_complex(n, seed=300 + n)
    elements = dec.interferometer_mesh(u_complex)
    s_total = mesh_to_symplectic(elements, n)
    assert np.allclose(s_total, ops.interferometer(u_complex), atol=1e-7)


def test_interferometer_mesh_element_count():
    n = 4
    elements = dec.interferometer_mesh(random_unitary_complex(n, seed=1))
    n_phase = sum(1 for e in elements if e[0] == "phase")
    n_bs = sum(1 for e in elements if e[0] == "bs")
    assert n_phase == n
    assert n_bs == n * (n - 1) // 2


def test_interferometer_mesh_requires_unitary():
    with pytest.raises(ValueError, match="unitary"):
        dec.interferometer_mesh(np.array([[1, 1], [0, 1]], dtype=complex))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 8, 10, 15])
def test_clements_decomposition_reconstructs(n):
    u_complex = random_unitary_complex(n, seed=400 + n)
    elements = dec.clements_decomposition(u_complex)
    s_total = mesh_to_symplectic(elements, n)
    assert np.allclose(s_total, ops.interferometer(u_complex), atol=1e-7)


def test_clements_decomposition_element_count():
    n = 6
    elements = dec.clements_decomposition(random_unitary_complex(n, seed=2))
    n_phase = sum(1 for e in elements if e[0] == "phase")
    n_bs = sum(1 for e in elements if e[0] == "bs")
    assert n_phase == n
    assert n_bs == n * (n - 1) // 2


def test_clements_decomposition_requires_unitary():
    with pytest.raises(ValueError, match="unitary"):
        dec.clements_decomposition(np.array([[1, 1], [0, 1]], dtype=complex))


@pytest.mark.parametrize("n", [4, 6, 8, 10, 15])
def test_clements_is_shallower_than_reck(n):
    """The whole point of Clements over Reck: a genuinely rectangular (depth ~n,
    not ~2n) mesh -- checked via Circuit's own frontier-based depth(), the
    same allocator diagrams use, so this also exercises real integration
    with the rest of the library rather than re-deriving depth by hand."""
    from cvcircuit.circuit import Circuit

    u_complex = random_unitary_complex(n, seed=500 + n)

    def circuit_from_mesh(elements):
        c = Circuit(n)
        for el in elements:
            if el[0] == "phase":
                c.rotate(el[2], mode=el[1])
            else:
                c.beamsplitter(el[3], modes=(el[1], el[2]), phi=el[4])
        return c

    depth_clements = circuit_from_mesh(dec.clements_decomposition(u_complex)).depth()
    depth_reck = circuit_from_mesh(dec.interferometer_mesh(u_complex)).depth()

    assert depth_clements <= n + 1
    assert depth_reck >= 2 * n - 3
    assert depth_clements < depth_reck
