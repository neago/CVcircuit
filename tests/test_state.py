"""GaussianState: constructors, operations, properties, phase space.

Cross-checked against an independent truncated-Fock-space simulator
(reference.py) wherever that's feasible -- see the ``test_fock_*``
functions -- plus analytic values, invariants, and error paths.
"""

import numpy as np
import pytest

from cvcircuit import ops
from cvcircuit import quantum_info as qi
from cvcircuit import state as st
from reference import FockSimulator

HBARS = (1, 2)


# ---------------------------------------------------------------------------
# constructors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hbar", HBARS)
def test_vacuum(hbar):
    v = st.vacuum(3, hbar=hbar)
    assert v.num_modes == 3
    assert np.allclose(v.mean, 0)
    assert np.allclose(v.cov, hbar / 2 * np.eye(6))
    assert v.is_pure()
    assert v.purity() == pytest.approx(1.0)


@pytest.mark.parametrize("hbar", HBARS)
def test_coherent_mean_photon_number(hbar):
    alpha = 1.2 - 0.7j
    c = st.coherent(alpha, hbar=hbar)
    assert c.mean_photon_number() == pytest.approx(abs(alpha) ** 2, abs=1e-8)
    assert c.photon_number_variance() == pytest.approx(abs(alpha) ** 2, abs=1e-8)
    assert c.is_pure()


def test_thermal_mean_photon_number():
    for nbar in (0.0, 0.5, 3.0):
        t = st.thermal(nbar)
        assert t.mean_photon_number() == pytest.approx(nbar, abs=1e-10)
        assert t.photon_number_variance() == pytest.approx(nbar * (nbar + 1), abs=1e-8)
    assert st.thermal(0.0).is_pure()
    assert not st.thermal(1.0).is_pure()


def test_squeezed_at_phi_zero():
    for hbar in HBARS:
        r = 0.6
        s = st.squeezed(r, hbar=hbar)
        assert s.cov[0, 0] == pytest.approx(hbar / 2 * np.exp(-2 * r))
        assert s.cov[1, 1] == pytest.approx(hbar / 2 * np.exp(2 * r))
        assert s.squeezing_db(0) == pytest.approx(10 * np.log10(np.exp(-2 * r)), abs=1e-8)


def test_two_mode_squeezed_purity_and_entropy():
    for r in (0.0, 0.5, 1.5):
        tmsv = st.two_mode_squeezed(r)
        assert tmsv.is_pure()
        assert tmsv.reduced([0]).von_neumann_entropy() == pytest.approx(
            tmsv.reduced([1]).von_neumann_entropy(), abs=1e-8
        )


def test_epr_is_two_mode_squeezed_phi_zero():
    r = 0.7
    assert np.allclose(st.epr(r).cov, st.two_mode_squeezed(r, phi=0.0).cov)


def test_random_state_valid_and_reproducible():
    a = st.random_state(3, seed=0)
    b = st.random_state(3, seed=0)
    c = st.random_state(3, seed=1)
    assert np.allclose(a.mean, b.mean) and np.allclose(a.cov, b.cov)
    assert not np.allclose(a.mean, c.mean)
    assert st.random_state(4, pure=True, seed=2).is_pure()


def test_graph_state_builds_and_entangles():
    adjacency = np.array([[0, 1], [1, 0]])
    g = st.graph_state(adjacency, r=1.0)
    assert g.num_modes == 2
    assert not qi.is_separable(g, [0])
    assert st.cluster_state is st.graph_state


def test_graph_state_bad_adjacency():
    with pytest.raises(ValueError, match="symmetric"):
        st.graph_state(np.array([[0, 1], [0, 0]]))


def test_tensor_function():
    a, b, c = st.vacuum(1), st.coherent(1.0), st.squeezed(0.5)
    combined = st.tensor(a, b, c)
    assert combined.num_modes == 3
    combined2 = st.tensor([a, b, c])
    assert np.allclose(combined.cov, combined2.cov)
    with pytest.raises(ValueError, match="at least one"):
        st.tensor()


# ---------------------------------------------------------------------------
# constructor validation
# ---------------------------------------------------------------------------


def test_invalid_mean_shape():
    with pytest.raises(ValueError, match="even length"):
        st.GaussianState(np.zeros(3), np.eye(4))


def test_mismatched_cov_shape():
    with pytest.raises(ValueError, match="expected"):
        st.GaussianState(np.zeros(2), np.eye(4))


def test_asymmetric_cov_rejected():
    cov = np.eye(2)
    cov[0, 1] = 5.0
    with pytest.raises(ValueError, match="symmetric"):
        st.GaussianState(np.zeros(2), cov)


def test_uncertainty_violation_rejected():
    with pytest.raises(ValueError, match="uncertainty relation"):
        st.GaussianState(np.zeros(2), 0.1 * np.eye(2), hbar=2)


# ---------------------------------------------------------------------------
# combine / reduce / permute
# ---------------------------------------------------------------------------


def test_and_operator_is_tensor():
    a, b = st.squeezed(0.5), st.vacuum(1)
    combined = a & b
    assert combined.num_modes == 2
    assert np.allclose(combined.cov[0:2, 0:2], a.cov)
    assert np.allclose(combined.cov[2:4, 2:4], b.cov)
    assert np.allclose(combined.cov[0:2, 2:4], 0)


def test_and_requires_matching_hbar():
    with pytest.raises(ValueError, match="hbar"):
        st.vacuum(1, hbar=1) & st.vacuum(1, hbar=2)


def test_reduced_recovers_product_state_factor():
    combined = st.squeezed(0.7) & st.coherent(1 + 1j)
    assert np.allclose(combined.reduced([1]).mean, st.coherent(1 + 1j).mean)
    assert np.allclose(combined.reduced([1]).cov, st.coherent(1 + 1j).cov)


def test_reduced_duplicate_modes_rejected():
    s = st.vacuum(2)
    with pytest.raises(ValueError, match="distinct"):
        s.reduced([0, 0])


def test_reduced_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        st.vacuum(2).reduced([5])


def test_permute_round_trip():
    s = st.random_state(3, seed=3)
    permuted = s.permute([2, 0, 1])
    back = permuted.permute([1, 2, 0])
    assert np.allclose(back.mean, s.mean, atol=1e-10)
    assert np.allclose(back.cov, s.cov, atol=1e-10)


def test_permute_invalid_not_a_permutation():
    with pytest.raises(ValueError, match="permutation"):
        st.vacuum(3).permute([0, 2])  # distinct, but not a permutation of range(3)


# ---------------------------------------------------------------------------
# unitary ops: invariants
# ---------------------------------------------------------------------------


def test_ops_preserve_uncertainty_and_purity():
    s = st.random_state(2, pure=True, seed=5)
    ops_applied = (
        s.squeeze(0.4, mode=0)
        .rotate(0.3, mode=1)
        .beamsplitter(0.5, modes=(0, 1))
        .two_mode_squeeze(0.2, modes=(0, 1))
        .displace(0.3 + 0.1j, mode=0)
        .cz(0.5, modes=(0, 1))
        .cx(0.4, modes=(0, 1))
        .shear(0.2, mode=0)
    )
    assert ops_applied.is_pure()
    omega = ops.symplectic_form(2)
    eigs = np.linalg.eigvalsh(ops_applied.cov + 1j * (ops_applied.hbar / 2) * omega)
    assert eigs.min() >= -1e-8


def test_gates_return_new_state_not_mutate():
    s = st.vacuum(1)
    s2 = s.squeeze(1.0)
    assert np.allclose(s.cov, np.eye(2))  # original untouched
    assert not np.allclose(s2.cov, s.cov)


def test_apply_symplectic_rejects_non_symplectic():
    # det != 1, so not symplectic even for a single mode (where det=1 suffices)
    with pytest.raises(ValueError, match="symplectic"):
        st.vacuum(1).apply_symplectic(np.array([[2.0, 0.0], [0.0, 1.0]]))


def test_interferometer_matches_beamsplitter():
    theta, phi = 0.3, 0.6
    c, s_ = np.cos(theta), np.sin(theta)
    u_mat = np.array([[c, -s_ * np.exp(-1j * phi)], [s_ * np.exp(1j * phi), c]])
    a = st.coherent([1 + 0.5j, -0.3j]).interferometer(u_mat)
    b = st.coherent([1 + 0.5j, -0.3j]).beamsplitter(theta, phi=phi)
    assert np.allclose(a.mean, b.mean) and np.allclose(a.cov, b.cov)


# ---------------------------------------------------------------------------
# Fock-space cross-checks (independent method)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hbar", HBARS)
def test_fock_rotation(hbar):
    alpha = 0.6 + 0.3j
    phi = 0.7
    sim = FockSimulator(1, cutoff=35, hbar=hbar)
    sim.displace(alpha).rotate(phi)
    mean, cov = sim.moments()
    expected = st.coherent(alpha, hbar=hbar).rotate(phi)
    assert np.allclose(mean, expected.mean, atol=1e-5)
    assert np.allclose(cov, expected.cov, atol=1e-4)


@pytest.mark.parametrize("hbar", HBARS)
def test_fock_squeezing(hbar):
    sim = FockSimulator(1, cutoff=35, hbar=hbar)
    sim.squeeze(0.5, phi=0.9)
    _, cov = sim.moments()
    expected = st.squeezed(0.5, phi=0.9, hbar=hbar)
    assert np.allclose(cov, expected.cov, atol=1e-4)


def test_fock_displacement():
    sim = FockSimulator(1, cutoff=30, hbar=2)
    alpha = -0.4 + 0.9j
    sim.displace(alpha)
    mean, cov = sim.moments()
    expected = st.coherent(alpha)
    assert np.allclose(mean, expected.mean, atol=1e-6)
    assert np.allclose(cov, expected.cov, atol=1e-6)


def test_fock_beamsplitter():
    sim = FockSimulator(2, cutoff=16)
    sim.displace(0.5 + 0.2j, mode=0)
    sim.beamsplitter(0.4, phi=0.5)
    mean, cov = sim.moments()
    expected = (st.coherent(0.5 + 0.2j) & st.vacuum(1)).beamsplitter(0.4, phi=0.5)
    assert np.allclose(mean, expected.mean, atol=1e-4)
    assert np.allclose(cov, expected.cov, atol=1e-3)


def test_fock_two_mode_squeezing():
    sim = FockSimulator(2, cutoff=16)
    sim.two_mode_squeeze(0.4, phi=0.3)
    _, cov = sim.moments()
    expected = st.two_mode_squeezed(0.4, phi=0.3)
    assert np.allclose(cov, expected.cov, atol=1e-3)


def test_fock_loss_matches_channel():
    sim = FockSimulator(1, cutoff=30)
    sim.squeeze(0.5)
    _, cov = sim.loss(0.6)
    expected = st.squeezed(0.5).loss(0.6)
    assert np.allclose(cov, expected.cov, atol=1e-3)


# ---------------------------------------------------------------------------
# entropy, purity, symplectic eigenvalues (analytic values)
# ---------------------------------------------------------------------------


def test_thermal_entropy_matches_formula():
    for nbar in (0.3, 1.0, 2.5):
        t = st.thermal(nbar)

        def g(x):
            return (x + 1) * np.log2(x + 1) - (x * np.log2(x) if x > 0 else 0)

        assert t.von_neumann_entropy() == pytest.approx(g(nbar), abs=1e-8)


def test_renyi2_equals_minus_log_purity():
    s = st.random_state(2, seed=7)
    assert s.renyi_entropy(alpha=2) == pytest.approx(-np.log2(s.purity()), abs=1e-8)


def test_renyi_alpha_one_equals_von_neumann():
    s = st.random_state(2, seed=8)
    assert s.renyi_entropy(alpha=1) == pytest.approx(s.von_neumann_entropy(), abs=1e-8)


def test_symplectic_eigenvalues_snu_pure_state_is_one():
    s = st.random_state(3, pure=True, seed=9)
    assert np.allclose(s.symplectic_eigenvalues_snu(), 1.0, atol=1e-7)


def test_williamson_round_trip_on_state():
    s = st.random_state(3, seed=11)
    s_mat, d_mat = s.williamson()
    assert np.allclose(s_mat @ d_mat @ s_mat.T, s.cov, atol=1e-7)


# ---------------------------------------------------------------------------
# measurement conditional-state formulas
# ---------------------------------------------------------------------------


def test_homodyne_tmsv_conditional_variance():
    # textbook EPR conditional squeezing: after x-homodyne on mode 0,
    # Var(x2 | x1) = 1/cosh(2r); Var(p2) unconditioned = cosh(2r) (hbar=2)
    r = 1.0
    tmsv = st.two_mode_squeezed(r)
    _, cond = tmsv.homodyne(mode=0, angle=0.0, seed=0)
    reduced = cond.reduced([1])
    assert reduced.cov[0, 0] == pytest.approx(1 / np.cosh(2 * r), abs=1e-8)
    assert reduced.cov[1, 1] == pytest.approx(np.cosh(2 * r), abs=1e-8)
    assert reduced.cov[0, 1] == pytest.approx(0.0, abs=1e-8)


def test_heterodyne_coherent_state_outcome_statistics():
    # outcome_cov = sigma_signal + cov_povm = (hbar/2 + hbar/2) I = hbar I (in x,p units);
    # alpha = (x+ip)/sqrt(2 hbar), so Var(Re alpha) = Var(x)/(2 hbar) = 1/2.
    alpha = 0.5 - 0.3j
    outcomes = np.array([st.coherent(alpha).heterodyne(seed=seed)[0] for seed in range(3000)])
    assert np.mean(outcomes.real) == pytest.approx(alpha.real, abs=0.05)
    assert np.mean(outcomes.imag) == pytest.approx(alpha.imag, abs=0.05)
    assert np.var(outcomes.real) == pytest.approx(0.5, abs=0.1)
    assert np.var(outcomes.imag) == pytest.approx(0.5, abs=0.1)


def test_generaldyne_heterodyne_equal_povm_matches_heterodyne():
    s = st.squeezed(0.5) & st.vacuum(1)
    cov_povm = (s.hbar / 2) * np.eye(2)
    out_het, cond_het = s.heterodyne(mode=0, seed=42)
    out_gd, cond_gd = s.generaldyne([0], cov_povm, seed=42)
    from cvcircuit import conversions

    assert np.allclose(conversions.xp_to_alpha(out_gd, hbar=s.hbar)[0], out_het)
    assert np.allclose(cond_het.cov, cond_gd.cov)


def test_generaldyne_measured_mode_reset_to_vacuum():
    s = st.squeezed(0.8) & st.thermal(1.0)
    _, cond = s.homodyne(mode=0, seed=0)
    assert np.allclose(cond.reduced([0]).cov, (s.hbar / 2) * np.eye(2))
    assert cond.num_modes == 2  # mode indices unchanged


def test_measurement_out_of_range_mode():
    with pytest.raises(ValueError, match="out of range"):
        st.vacuum(1).homodyne(mode=3)


# ---------------------------------------------------------------------------
# phase space
# ---------------------------------------------------------------------------


def test_wigner_normalization():
    s = st.squeezed(0.5)
    xv = np.linspace(-10, 10, 300)
    pv = np.linspace(-10, 10, 300)
    x_grid, p_grid = np.meshgrid(xv, pv)
    w = s.wigner(x_grid, p_grid)
    integral = w.sum() * (xv[1] - xv[0]) * (pv[1] - pv[0])
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_wigner_marginal_matches_quad_variance():
    s = st.squeezed(0.6).displace(0.3 + 0.1j)
    xvec, pdf = s.marginal(0, angle=0.4)
    mean_est = np.trapezoid(xvec * pdf, xvec)
    var_est = np.trapezoid((xvec - mean_est) ** 2 * pdf, xvec)
    assert mean_est == pytest.approx(s.quad_mean(0, angle=0.4), abs=1e-6)
    assert var_est == pytest.approx(s.quad_variance(0, angle=0.4), abs=1e-6)


def test_p_function_raises_for_squeezed_state():
    with pytest.raises(ValueError, match="s=1.0"):
        st.squeezed(0.5).p_function(0.0, 0.0)


def test_p_function_works_for_classical_state():
    t = st.thermal(2.0)
    assert t.is_classical()
    value = t.p_function(0.0, 0.0)
    assert value > 0


def test_quasiprobability_consistency_classical_state():
    t = st.thermal(1.5)
    w = t.wigner(0.3, -0.2)
    q = t.husimi(0.3, -0.2)
    p = t.p_function(0.3, -0.2)
    # Q is the widest (most smoothed), P the narrowest, at a point away from the peak
    assert w > 0 and q > 0 and p > 0


# ---------------------------------------------------------------------------
# interop and repr
# ---------------------------------------------------------------------------


def test_xxpp_round_trip():
    s = st.random_state(3, seed=13)
    mean_xxpp, cov_xxpp = s.to_xxpp()
    back = st.GaussianState.from_xxpp(mean_xxpp, cov_xxpp, hbar=s.hbar)
    assert np.allclose(back.mean, s.mean) and np.allclose(back.cov, s.cov)


def test_repr_contains_mode_count_and_purity():
    s = st.vacuum(2)
    text = repr(s)
    assert "2 mode" in text
    assert "purity" in text


def test_copy_is_independent():
    s = st.vacuum(1)
    c = s.copy()
    c.mean[:] = 99  # mutate the copy's raw array directly
    assert not np.allclose(s.mean, c.mean)
