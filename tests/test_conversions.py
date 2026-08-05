"""Ordering, dB and complex-amplitude conversions: round-trips and known values."""

import numpy as np
import pytest

from cvcircuit import conversions as conv


def test_xpxp_xxpp_roundtrip_vector():
    rng = np.random.default_rng(0)
    for n in (1, 2, 5):
        vec = rng.normal(size=2 * n)
        assert np.allclose(conv.xxpp_to_xpxp(conv.xpxp_to_xxpp(vec)), vec)


def test_xpxp_xxpp_roundtrip_matrix():
    rng = np.random.default_rng(1)
    for n in (1, 2, 5):
        mat = rng.normal(size=(2 * n, 2 * n))
        assert np.allclose(conv.xxpp_to_xpxp(conv.xpxp_to_xxpp(mat)), mat)


def test_xpxp_to_xxpp_known_case():
    # xpxp (x1,p1,x2,p2) -> xxpp (x1,x2,p1,p2)
    vec = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(conv.xpxp_to_xxpp(vec), [1.0, 3.0, 2.0, 4.0])


def test_r_to_db_known_value():
    # r=1 nepers -> -20/ln(10) dB = -8.6858896...
    assert conv.r_to_db(1.0) == pytest.approx(-8.685889638, abs=1e-8)


def test_r_db_roundtrip():
    for r in (0.0, 0.1, 0.5, 1.0, 2.0):
        assert conv.db_to_r(conv.r_to_db(r)) == pytest.approx(r, abs=1e-10)


def test_r_to_db_sign_convention():
    # squeezing (r > 0) is reported as negative dB
    assert conv.r_to_db(0.5) < 0
    assert conv.r_to_db(0.0) == pytest.approx(0.0, abs=1e-10)


def test_eta_db_roundtrip():
    for eta in (0.01, 0.3, 0.9, 1.0):
        assert conv.db_to_eta(conv.eta_to_db(eta)) == pytest.approx(eta, abs=1e-10)


def test_eta_to_db_invalid():
    with pytest.raises(ValueError, match=r"eta must be in"):
        conv.eta_to_db(0.0)
    with pytest.raises(ValueError, match=r"eta must be in"):
        conv.eta_to_db(1.5)


def test_distance_to_eta_matches_db_per_km():
    eta = conv.distance_to_eta(10.0, db_per_km=0.2)
    assert eta == pytest.approx(conv.db_to_eta(2.0))


def test_alpha_xp_roundtrip_single_mode():
    for hbar in (1, 2):
        for alpha in (0.0, 1 + 1j, -0.5 + 0.3j):
            xp = conv.alpha_to_xp(alpha, hbar=hbar)
            assert conv.xp_to_alpha(xp, hbar=hbar)[0] == pytest.approx(alpha, abs=1e-10)


def test_alpha_xp_roundtrip_multimode():
    alphas = np.array([1 + 1j, -0.3 + 0.2j, 0.0])
    xp = conv.alpha_to_xp(alphas, hbar=2)
    assert xp.shape == (6,)
    recovered = conv.xp_to_alpha(xp, hbar=2)
    assert np.allclose(recovered, alphas)


def test_alpha_to_xp_known_value_hbar2():
    # hbar=2: d = sqrt(2*2)*(Re a, Im a) = 2*(Re a, Im a)
    xp = conv.alpha_to_xp(1 + 0.5j, hbar=2)
    assert np.allclose(xp, [2.0, 1.0])
