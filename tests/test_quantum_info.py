"""Fidelity, entropies, and entanglement/steering measures: known values and invariants."""

import numpy as np
import pytest

from cvcircuit import quantum_info as qi
from cvcircuit import state as st


# ---------------------------------------------------------------------------
# fidelity
# ---------------------------------------------------------------------------


def test_fidelity_self_is_one():
    for s in (st.vacuum(2), st.squeezed(0.6), st.thermal(1.2), st.two_mode_squeezed(0.5)):
        assert qi.fidelity(s, s) == pytest.approx(1.0, abs=1e-8)


def test_fidelity_coherent_states_known_value():
    a, b = 1 + 1j, 0.3 - 0.2j
    assert qi.fidelity(st.coherent(a), st.coherent(b)) == pytest.approx(np.exp(-abs(a - b) ** 2), abs=1e-8)


def test_fidelity_squeezed_vacuum_known_value():
    # |<0|S(r)|0>|^2 = 1/cosh(r)
    r = 0.5
    assert qi.fidelity(st.squeezed(r), st.vacuum(1)) == pytest.approx(1 / np.cosh(r), abs=1e-8)


def test_fidelity_symmetric():
    a, b = st.squeezed(0.4), st.coherent(0.5 + 0.2j)
    assert qi.fidelity(a, b) == pytest.approx(qi.fidelity(b, a), abs=1e-10)


def test_fidelity_hbar_independent():
    a1, b1 = st.coherent(0.6 + 0.2j, hbar=1), st.coherent(0.1 - 0.3j, hbar=1)
    a2, b2 = st.coherent(0.6 + 0.2j, hbar=2), st.coherent(0.1 - 0.3j, hbar=2)
    assert qi.fidelity(a1, b1) == pytest.approx(qi.fidelity(a2, b2), abs=1e-8)


def test_fidelity_mismatched_modes_raises():
    with pytest.raises(ValueError, match="modes"):
        qi.fidelity(st.vacuum(1), st.vacuum(2))


def test_fidelity_mismatched_hbar_raises():
    with pytest.raises(ValueError, match="hbar"):
        qi.fidelity(st.vacuum(1, hbar=1), st.vacuum(1, hbar=2))


def test_bures_distance_and_angle_self_zero():
    # near F=1 the sqrt(1-F) in the Bures distance amplifies floating-point
    # error in the fidelity computation, so this needs a looser tolerance
    # than the other "self" checks above.
    s = st.squeezed(0.3)
    assert qi.bures_distance(s, s) == pytest.approx(0.0, abs=1e-4)
    assert qi.bures_angle(s, s) == pytest.approx(0.0, abs=1e-4)


def test_trace_distance_bounds_ordering():
    a, b = st.coherent(0.0), st.coherent(1.0)
    lower, upper = qi.trace_distance_bounds(a, b)
    assert 0 <= lower <= upper <= 1


# ---------------------------------------------------------------------------
# single-state wrappers
# ---------------------------------------------------------------------------


def test_wrappers_match_state_methods():
    s = st.random_state(2, seed=1)
    assert qi.purity(s) == s.purity()
    assert qi.von_neumann_entropy(s) == s.von_neumann_entropy()
    assert qi.renyi_entropy(s, alpha=3) == s.renyi_entropy(alpha=3)


# ---------------------------------------------------------------------------
# mutual information
# ---------------------------------------------------------------------------


def test_mutual_information_product_state_is_zero():
    s = st.squeezed(0.5) & st.thermal(0.3)
    assert qi.mutual_information(s, [0]) == pytest.approx(0.0, abs=1e-8)


def test_mutual_information_pure_tmsv_is_twice_entropy():
    tmsv = st.two_mode_squeezed(0.8)
    mi = qi.mutual_information(tmsv, [0])
    entropy = tmsv.reduced([0]).von_neumann_entropy()
    assert mi == pytest.approx(2 * entropy, abs=1e-8)


def test_mutual_information_requires_proper_subset():
    with pytest.raises(ValueError, match="proper subset"):
        qi.mutual_information(st.vacuum(2), [0, 1])


# ---------------------------------------------------------------------------
# entanglement: negativity and separability
# ---------------------------------------------------------------------------


def test_log_negativity_tmsv_known_value():
    for r in (0.2, 0.8, 1.5):
        assert qi.logarithmic_negativity(st.two_mode_squeezed(r), [0]) == pytest.approx(
            np.log2(np.exp(2 * r)), abs=1e-6
        )


def test_log_negativity_product_state_is_zero():
    s = st.squeezed(0.5) & st.thermal(0.3)
    assert qi.logarithmic_negativity(s, [0]) == pytest.approx(0.0, abs=1e-8)


def test_negativity_relation_to_log_negativity():
    s = st.two_mode_squeezed(0.7)
    en = qi.logarithmic_negativity(s, [0])
    n = qi.negativity(s, [0])
    assert n == pytest.approx((2**en - 1) / 2, abs=1e-8)


def test_is_separable_product_and_entangled():
    product = st.squeezed(0.5) & st.vacuum(1)
    entangled = st.two_mode_squeezed(0.5)
    assert qi.is_separable(product, [0])
    assert not qi.is_separable(entangled, [0])
    assert qi.simon_criterion is qi.is_separable


# ---------------------------------------------------------------------------
# Duan / EPR criterion
# ---------------------------------------------------------------------------


def test_duan_criterion_and_epr_variance_analytic():
    for r in (0.0, 0.3, 1.0):
        tmsv = st.two_mode_squeezed(r)
        ev = qi.epr_variance(tmsv)
        assert ev == pytest.approx(2 * tmsv.hbar * np.exp(-2 * r), abs=1e-8)
    assert not qi.duan_criterion(st.two_mode_squeezed(0.0))  # r=0: product state
    assert qi.duan_criterion(st.two_mode_squeezed(0.5))


def test_duan_false_does_not_crash_on_product_state():
    s = st.squeezed(0.5) & st.thermal(1.0)
    assert qi.duan_criterion(s) is False


# ---------------------------------------------------------------------------
# steering
# ---------------------------------------------------------------------------


def test_steerability_zero_at_zero_squeezing():
    assert qi.steerability(st.two_mode_squeezed(0.0), [0], 1) == pytest.approx(0.0, abs=1e-8)


def test_steerability_increases_with_squeezing():
    values = [qi.steerability(st.two_mode_squeezed(r), [0], 1) for r in (0.1, 0.5, 1.0, 2.0)]
    assert all(b >= a - 1e-10 for a, b in zip(values, values[1:]))


def test_steerability_symmetric_for_tmsv():
    tmsv = st.two_mode_squeezed(0.9)
    assert qi.steerability(tmsv, [0], 1) == pytest.approx(qi.steerability(tmsv, [1], 0), abs=1e-8)


def test_steerability_nonnegative():
    s = st.random_state(3, seed=4)
    assert qi.steerability(s, [0], 1) >= 0


# ---------------------------------------------------------------------------
# entanglement of formation and discord (pure-state scope)
# ---------------------------------------------------------------------------


def test_eof_pure_state_matches_reduced_entropy():
    tmsv = st.two_mode_squeezed(0.9)
    assert qi.entanglement_of_formation(tmsv, [0]) == pytest.approx(tmsv.reduced([0]).von_neumann_entropy())


def test_eof_rejects_mixed_state():
    mixed = st.thermal(1.0) & st.vacuum(1)
    with pytest.raises(ValueError, match="pure"):
        qi.entanglement_of_formation(mixed, [0])


def test_discord_pure_state_matches_eof():
    tmsv = st.two_mode_squeezed(0.6)
    assert qi.gaussian_discord(tmsv, [0], 1) == pytest.approx(qi.entanglement_of_formation(tmsv, [0]))


def test_discord_rejects_mixed_state():
    mixed = st.thermal(1.0) & st.vacuum(1)
    with pytest.raises(ValueError, match="pure"):
        qi.gaussian_discord(mixed, [0], 1)
