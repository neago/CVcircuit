"""Circuit builder: state()/run() agreement with state-first calls, to_symplectic/to_channel,
feedforward, manipulation methods, error paths."""

import numpy as np
import pytest

from cvcircuit import circuit as ci
from cvcircuit import ops
from cvcircuit import quantum_info as qi
from cvcircuit import state as st


def test_circuit_state_matches_state_first_application():
    circuit = ci.Circuit(2)
    circuit.squeeze(0.8, phi=0.2, mode=0).beamsplitter(0.5, modes=(0, 1)).displace(0.3 + 0.1j, mode=1)
    from_circuit = circuit.state()
    from_state = st.vacuum(2).squeeze(0.8, phi=0.2, mode=0).beamsplitter(0.5, modes=(0, 1)).displace(0.3 + 0.1j, mode=1)
    assert np.allclose(from_circuit.mean, from_state.mean)
    assert np.allclose(from_circuit.cov, from_state.cov)


def test_circuit_with_channels_matches_state_first():
    circuit = ci.Circuit(1)
    circuit.squeeze(0.5).loss(0.7, nbar_env=0.2).noise(0.05)
    from_circuit = circuit.state()
    from_state = st.squeezed(0.5).loss(0.7, nbar_env=0.2).add_noise(0.05)
    assert np.allclose(from_circuit.cov, from_state.cov)


def test_circuit_custom_initial_state():
    initial = st.squeezed(0.4)
    circuit = ci.Circuit(1, initial=initial)
    circuit.rotate(0.5)
    assert np.allclose(circuit.state().cov, initial.rotate(0.5).cov)


def test_circuit_initial_state_mismatch_rejected():
    with pytest.raises(ValueError, match="mode"):
        ci.Circuit(2, initial=st.vacuum(1))
    with pytest.raises(ValueError, match="hbar"):
        ci.Circuit(1, hbar=1, initial=st.vacuum(1, hbar=2))


def test_mode_validation():
    circuit = ci.Circuit(2)
    with pytest.raises(ValueError, match="out of range"):
        circuit.squeeze(0.5, mode=5)
    with pytest.raises(ValueError, match="distinct"):
        circuit.beamsplitter(0.5, modes=(0, 0))


# ---------------------------------------------------------------------------
# to_symplectic / to_channel
# ---------------------------------------------------------------------------


def test_to_symplectic_matches_direct_application():
    circuit = ci.Circuit(2)
    circuit.squeeze(0.6, mode=0).rotate(0.3, mode=1).beamsplitter(0.4, modes=(0, 1)).cz(0.5, modes=(0, 1))
    s_mat, d_vec = circuit.to_symplectic()
    expected = st.vacuum(2).squeeze(0.6, mode=0).rotate(0.3, mode=1).beamsplitter(0.4, modes=(0, 1)).cz(
        0.5, modes=(0, 1)
    )
    assert np.allclose(s_mat @ np.zeros(4) + d_vec, expected.mean)
    assert np.allclose(s_mat @ (expected.hbar / 2 * np.eye(4)) @ s_mat.T, expected.cov)
    assert ops.is_symplectic(s_mat)


def test_to_symplectic_rejects_channel():
    circuit = ci.Circuit(1)
    circuit.squeeze(0.5).loss(0.8)
    with pytest.raises(ValueError, match="unitary"):
        circuit.to_symplectic()


def test_to_symplectic_rejects_feedforward():
    circuit = ci.Circuit(2)
    circuit.homodyne(0, label="x")
    circuit.displace(lambda m: m["x"], mode=1)
    with pytest.raises(ValueError, match="unitary"):
        circuit.to_symplectic()


def test_to_channel_matches_direct_application():
    circuit = ci.Circuit(1)
    circuit.squeeze(0.5).loss(0.7, nbar_env=0.1).amplify(1.3, nbar_env=0.2).noise(0.05)
    x_mat, y_mat, d_vec = circuit.to_channel()
    expected = st.squeezed(0.5).loss(0.7, nbar_env=0.1).amplify(1.3, nbar_env=0.2).add_noise(0.05)
    assert np.allclose(x_mat @ np.zeros(2) + d_vec, expected.mean)
    assert np.allclose(x_mat @ (1 * np.eye(2)) @ x_mat.T + y_mat, expected.cov)


def test_to_channel_rejects_measurement():
    circuit = ci.Circuit(1)
    circuit.homodyne(0)
    with pytest.raises(ValueError, match="cannot include"):
        circuit.to_channel()


# ---------------------------------------------------------------------------
# feedforward
# ---------------------------------------------------------------------------


def test_teleportation_fidelity_improves_with_squeezing():
    alpha_in = 1.0 + 0.5j

    def teleport(r, seed):
        circuit = ci.Circuit(3)
        circuit.displace(alpha_in, mode=0)
        circuit.two_mode_squeeze(r, modes=(1, 2))
        circuit.beamsplitter(np.pi / 4, modes=(0, 1))
        circuit.homodyne(0, angle=0.0, label="xA")
        circuit.homodyne(1, angle=np.pi / 2, label="pB")
        circuit.displace(lambda m: (m["xA"] + 1j * m["pB"]) / np.sqrt(2), mode=2)
        final_state, _ = circuit._execute(seed=seed)
        return final_state.reduced([2])

    ideal = st.coherent(alpha_in)
    fidelities = [qi.fidelity(ideal, teleport(r, seed=123)) for r in (0.1, 1.0, 3.0)]
    assert fidelities[0] < fidelities[1] < fidelities[2]
    assert fidelities[2] > 0.99


def test_feedforward_record_labels():
    circuit = ci.Circuit(2)
    circuit.homodyne(0, label="a")
    circuit.heterodyne(1, label="b")
    _, records = circuit._execute(seed=0)
    assert set(records.keys()) == {"a", "b"}


def test_feedforward_default_labels_are_deterministic():
    circuit = ci.Circuit(2)
    circuit.homodyne(0)
    circuit.homodyne(1)
    _, records1 = circuit._execute(seed=1)
    _, records2 = circuit._execute(seed=1)
    assert set(records1.keys()) == set(records2.keys())
    assert len(records1) == 2


# ---------------------------------------------------------------------------
# run() / Result
# ---------------------------------------------------------------------------


def test_run_sampling_matches_analytic_distribution():
    circuit = ci.Circuit(1)
    circuit.squeeze(0.4).homodyne(0, angle=0.0, label="x")
    result = circuit.run(shots=3000, seed=0)
    analytic_var = st.squeezed(0.4).quad_variance(0, angle=0.0)
    assert abs(result.var("x") - analytic_var) < 5 * analytic_var * np.sqrt(2 / 3000) + 1e-6
    assert abs(result.mean("x")) < 0.1


def test_run_invalid_shots():
    with pytest.raises(ValueError, match="shots"):
        ci.Circuit(1).homodyne(0).run(shots=0)


def test_result_getitem_and_repr():
    circuit = ci.Circuit(1)
    circuit.homodyne(0, label="x")
    result = circuit.run(shots=10, seed=0)
    assert result["x"].shape == (10,)
    assert "x" in repr(result)
    with pytest.raises(KeyError):
        result["nonexistent"]


def test_result_cov_and_histogram():
    circuit = ci.Circuit(2)
    circuit.two_mode_squeeze(1.0, modes=(0, 1))
    circuit.homodyne(0, angle=0.0, label="x0")
    circuit.homodyne(1, angle=0.0, label="x1")
    result = circuit.run(shots=2000, seed=0)
    # EPR correlation: x0, x1 should be positively correlated for phi=0 TMSV
    assert result.cov("x0", "x1") > 0
    counts, edges = result.histogram("x0", bins=20)
    assert counts.sum() == 2000


# ---------------------------------------------------------------------------
# circuit manipulation
# ---------------------------------------------------------------------------


def test_compose_same_modes():
    a = ci.Circuit(2).squeeze(0.5, mode=0)
    b = ci.Circuit(2).rotate(0.3, mode=1)
    a.compose(b)
    assert len(a) == 2
    assert np.allclose(a.state().cov, st.vacuum(2).squeeze(0.5, mode=0).rotate(0.3, mode=1).cov)


def test_compose_with_mode_remap():
    a = ci.Circuit(3)
    b = ci.Circuit(2).beamsplitter(0.4, modes=(0, 1))
    a.compose(b, modes=[1, 2])
    expected = st.vacuum(3).beamsplitter(0.4, modes=(1, 2))
    assert np.allclose(a.state().cov, expected.cov)


def test_compose_mismatched_hbar():
    a = ci.Circuit(1, hbar=2)
    b = ci.Circuit(1, hbar=1)
    with pytest.raises(ValueError, match="hbar"):
        a.compose(b)


def test_inverse_undoes_circuit():
    circuit = ci.Circuit(2)
    circuit.squeeze(0.5, mode=0).beamsplitter(0.4, modes=(0, 1)).displace(0.3 + 0.2j, mode=1)
    round_trip = circuit.copy().compose(circuit.inverse())
    result = round_trip.state()
    assert np.allclose(result.mean, 0, atol=1e-8)
    assert np.allclose(result.cov, np.eye(4), atol=1e-8)


def test_copy_is_independent():
    a = ci.Circuit(1).squeeze(0.5)
    b = a.copy()
    b.rotate(0.3)
    assert len(a) == 1
    assert len(b) == 2


def test_depth_and_count_ops():
    circuit = ci.Circuit(3)
    circuit.squeeze(0.5, mode=0)
    circuit.squeeze(0.5, mode=1)
    circuit.beamsplitter(0.4, modes=(0, 1))
    circuit.rotate(0.2, mode=2)
    assert circuit.depth() == 2  # squeeze(0), squeeze(1) in parallel, then BS; mode 2's rotate is depth 1
    assert circuit.count_ops() == {"squeeze": 2, "beamsplitter": 1, "rotate": 1}
    assert len(circuit) == 4


def test_repr():
    circuit = ci.Circuit(2, hbar=1)
    assert "2 modes" in repr(circuit)
    assert "hbar=1" in repr(circuit)
