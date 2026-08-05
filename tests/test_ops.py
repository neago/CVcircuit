"""Symplectic matrix factories: symplectic invariant, embedding, known values."""

import numpy as np
import pytest

from cvcircuit import ops

ALL_1MODE = [
    ("rotation", ops.rotation(0.4)),
    ("squeezing", ops.squeezing(0.7, 0.3)),
    ("shear", ops.shear(0.5)),
    ("fourier", ops.fourier()),
]
ALL_2MODE = [
    ("beamsplitter", ops.beamsplitter(0.5, 0.2)),
    ("two_mode_squeezing", ops.two_mode_squeezing(0.6, 0.1)),
    ("cz", ops.cz(0.8)),
    ("cx", ops.cx(1.1)),
]


@pytest.mark.parametrize("name,matrix", ALL_1MODE + ALL_2MODE, ids=lambda v: v if isinstance(v, str) else "")
def test_all_gates_are_symplectic(name, matrix):
    assert ops.is_symplectic(matrix), f"{name} is not symplectic"


def test_symplectic_form_structure():
    omega = ops.symplectic_form(2)
    assert omega.shape == (4, 4)
    assert np.allclose(omega, [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    assert np.allclose(omega @ omega, -np.eye(4))


def test_rotation_known_values():
    r = ops.rotation(np.pi / 2)
    # x -> -p, p -> x  at phi=pi/2
    assert np.allclose(r @ [1, 0], [0, 1])
    assert np.allclose(r @ [0, 1], [-1, 0])


def test_squeezing_at_phi_zero_is_diagonal():
    r = 0.6
    s = ops.squeezing(r, phi=0.0)
    assert np.allclose(s, np.diag([np.exp(-r), np.exp(r)]))


def test_beamsplitter_transmittance():
    theta = 0.3
    bs = ops.beamsplitter(theta)
    t_expected = np.cos(theta) ** 2
    # power transmitted from mode 1 to mode 1 output, at phi=0
    assert bs[0, 0] ** 2 == pytest.approx(t_expected)


def test_beamsplitter_t_roundtrip():
    for transmittance in (0.0, 0.3, 0.5, 0.99, 1.0):
        bs = ops.beamsplitter_t(transmittance)
        assert bs[0, 0] ** 2 == pytest.approx(transmittance, abs=1e-10)


def test_beamsplitter_t_invalid():
    with pytest.raises(ValueError, match="transmittance"):
        ops.beamsplitter_t(1.5)


def test_beamsplitter_vacuum_is_identity_like():
    # a 50/50 BS applied twice at phi=0 is not identity in general, but
    # T=1 (theta=0) beamsplitter must be the identity
    bs = ops.beamsplitter_t(1.0)
    assert np.allclose(bs, np.eye(4))


def test_interferometer_matches_beamsplitter():
    theta, phi = 0.4, 0.9
    bs = ops.beamsplitter(theta, phi)
    c, s = np.cos(theta), np.sin(theta)
    u_bs = np.array([[c, -s * np.exp(-1j * phi)], [s * np.exp(1j * phi), c]])
    assert np.allclose(ops.interferometer(u_bs), bs)


def test_interferometer_requires_unitary():
    with pytest.raises(ValueError, match="unitary"):
        ops.interferometer(np.array([[1, 1], [0, 1]], dtype=complex))


def test_displacement_hbar_scaling():
    alpha = 1 + 0.5j
    d1 = ops.displacement(alpha, hbar=1)
    d2 = ops.displacement(alpha, hbar=2)
    assert np.allclose(d2, d1 * np.sqrt(2))


def test_expand_embeds_into_larger_system():
    s_local = ops.squeezing(0.5)
    full = ops.expand(s_local, [1], 3)
    assert full.shape == (6, 6)
    assert np.allclose(full[2:4, 2:4], s_local)
    assert np.allclose(full[0:2, 0:2], np.eye(2))
    assert np.allclose(full[4:6, 4:6], np.eye(2))
    # off-diagonal blocks (coupling to other modes) must be zero
    assert np.allclose(full[0:2, 2:4], 0)


def test_expand_two_mode_out_of_order():
    bs = ops.beamsplitter(0.4)
    full = ops.expand(bs, [2, 0], 3)
    idx = ops.mode_indices([2, 0])
    assert np.allclose(full[np.ix_(idx, idx)], bs)


def test_expand_wrong_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        ops.expand(np.eye(2), [0, 1], 3)


def test_expand_vector():
    d = np.array([1.0, 2.0])
    full = ops.expand_vector(d, [1], 2)
    assert np.allclose(full, [0, 0, 1, 2])


def test_random_interferometer_is_symplectic_and_passive():
    for n in (1, 2, 4):
        u = ops.random_interferometer(n, seed=n)
        assert ops.is_symplectic(u)
        assert np.allclose(u @ u.T, np.eye(2 * n))  # orthogonal: passive


def test_random_symplectic_is_symplectic():
    for n in (1, 3):
        s = ops.random_symplectic(n, seed=n)
        assert ops.is_symplectic(s)


def test_mode_indices():
    assert np.array_equal(ops.mode_indices([0, 2]), [0, 1, 4, 5])
    assert np.array_equal(ops.mode_indices(1), [2, 3])
