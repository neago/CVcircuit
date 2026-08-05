"""Comparisons and correlation measures between/within Gaussian states.

Single-state quantities (purity, entropy) are also available as
:class:`~cvcircuit.state.GaussianState` methods; the free functions here
exist for a uniform functional-style call (``cv.purity(s)``) and are thin
wrappers.  Everything else here genuinely needs two states or a
bipartition and lives only here.

References: Weedbrook et al., RMP 84, 621 (2012); Adesso & Illuminati,
J. Phys. A 40, 7821 (2007) (entanglement measures); Banchi, Braunstein &
Pirandola, PRL 115, 260501 (2015) (fidelity); Simon, PRL 84, 2726 (2000)
and Duan, Giedke, Cirac & Zoller, PRL 84, 2722 (2000) (separability);
Kogias, Lee, Ragy & Adesso, PRL 114, 060403 (2015) (Gaussian steering).
"""

from __future__ import annotations

import numpy as np

from . import ops
from .decompositions import symplectic_eigenvalues_snu

# ---------------------------------------------------------------------------
# single-state wrappers
# ---------------------------------------------------------------------------


def purity(state):
    """Tr(rho^2); see :meth:`cvcircuit.state.GaussianState.purity`."""
    return state.purity()


def von_neumann_entropy(state, base=2):
    """See :meth:`cvcircuit.state.GaussianState.von_neumann_entropy`."""
    return state.von_neumann_entropy(base=base)


def renyi_entropy(state, alpha=2, base=2):
    """See :meth:`cvcircuit.state.GaussianState.renyi_entropy`."""
    return state.renyi_entropy(alpha=alpha, base=base)


# ---------------------------------------------------------------------------
# fidelity and distance measures
# ---------------------------------------------------------------------------


def _matrix_sqrt_general(matrix):
    """Square root of a general (not necessarily symmetric) real matrix via eigendecomposition."""
    w, v = np.linalg.eig(matrix)
    return (v * np.sqrt(w.astype(complex))) @ np.linalg.inv(v)


def fidelity(state_a, state_b):
    """Fidelity F = |<a|b>|^2-convention fidelity between two Gaussian states.

    Uses the Banchi-Braunstein-Pirandola formula (PRL 115, 260501 (2015),
    Eqs. 9-15) for arbitrary multimode Gaussian states, rescaled to their
    hbar=1 convention internally.  Returns the **squared** fidelity (the
    convention this package shares with ``qcircsim.quantum_info.state_fidelity``):
    for two coherent states, ``fidelity(coherent(a), coherent(b)) == exp(-|a-b|**2)``.
    """
    if state_a.num_modes != state_b.num_modes:
        raise ValueError(f"states have different numbers of modes ({state_a.num_modes} vs {state_b.num_modes})")
    if state_a.hbar != state_b.hbar:
        raise ValueError(f"states have different hbar ({state_a.hbar} vs {state_b.hbar})")
    n = state_a.num_modes
    hbar = state_a.hbar
    omega = ops.symplectic_form(n)

    v1, v2 = state_a.cov / hbar, state_b.cov / hbar
    delta = (state_b.mean - state_a.mean) / np.sqrt(hbar)
    v_sum = v1 + v2
    v_sum_inv = np.linalg.inv(v_sum)

    v_aux = omega.T @ v_sum_inv @ (omega / 4 + v2 @ omega @ v1)
    m = v_aux @ omega
    m_inv_sq = np.linalg.inv(m @ m)
    inner = np.eye(2 * n) + m_inv_sq / 4
    sqrt_inner = _matrix_sqrt_general(inner)
    f_tot4 = np.linalg.det(2 * (sqrt_inner + np.eye(2 * n)) @ v_aux)
    f_tot = np.real(f_tot4) ** 0.25
    f0 = f_tot / np.linalg.det(v_sum) ** 0.25
    f_unsquared = f0 * np.exp(-0.25 * delta @ v_sum_inv @ delta)
    return float(np.clip(f_unsquared**2, 0.0, 1.0))


def bures_distance(state_a, state_b):
    """Bures distance D_B = sqrt(2(1 - sqrt(F))), F the squared fidelity."""
    f_unsquared = np.sqrt(fidelity(state_a, state_b))
    return float(np.sqrt(max(0.0, 2 * (1 - f_unsquared))))


def bures_angle(state_a, state_b):
    """Bures angle arccos(sqrt(F)), F the squared fidelity."""
    f_unsquared = np.sqrt(fidelity(state_a, state_b))
    return float(np.arccos(np.clip(f_unsquared, -1.0, 1.0)))


def trace_distance_bounds(state_a, state_b):
    """Fuchs-van de Graaf bounds (lower, upper) on the trace distance, from the fidelity."""
    f_unsquared = np.sqrt(fidelity(state_a, state_b))
    lower = 1 - f_unsquared
    upper = np.sqrt(max(0.0, 1 - f_unsquared**2))
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# mutual information
# ---------------------------------------------------------------------------


def mutual_information(state, modes_a, base=2):
    """Quantum mutual information I(A:B) = S(A) + S(B) - S(AB) for the (modes_a | rest) split."""
    modes_a = list(np.atleast_1d(modes_a))
    modes_b = [m for m in range(state.num_modes) if m not in modes_a]
    if not modes_b:
        raise ValueError("modes_a must be a proper subset of the state's modes")
    s_a = state.reduced(modes_a).von_neumann_entropy(base=base)
    s_b = state.reduced(modes_b).von_neumann_entropy(base=base)
    s_ab = state.von_neumann_entropy(base=base)
    return float(s_a + s_b - s_ab)


def conditional_entropy(state, modes_x, modes_y, base=2):
    """Conditional entropy S(X|Y) = S(XY) - S(Y)."""
    modes_x = list(np.atleast_1d(modes_x))
    modes_y = list(np.atleast_1d(modes_y))
    combined = sorted(set(modes_x) | set(modes_y))
    s_xy = state.reduced(combined).von_neumann_entropy(base=base)
    s_y = state.reduced(modes_y).von_neumann_entropy(base=base) if modes_y else 0.0
    return float(s_xy - s_y)


# ---------------------------------------------------------------------------
# entanglement: negativity and separability
# ---------------------------------------------------------------------------


def _partial_transpose_cov(state, modes_a):
    """cov with p -> -p on every mode NOT in modes_a (phase-space partial transpose)."""
    modes_a = set(np.atleast_1d(modes_a).tolist())
    n = state.num_modes
    lam = np.ones(2 * n)
    for m in range(n):
        if m not in modes_a:
            lam[2 * m + 1] = -1
    return np.diag(lam) @ state.cov @ np.diag(lam)


def logarithmic_negativity(state, modes_a):
    """Logarithmic negativity E_N = sum_k max(0, -log2(nu_tilde_k)) across (modes_a | rest).

    nu_tilde_k are the symplectic eigenvalues (SNU) of the partially
    transposed covariance matrix (Vidal & Werner, PRA 65, 032314 (2002);
    Adesso & Illuminati, J. Phys. A 40, 7821 (2007)).
    """
    cov_pt = _partial_transpose_cov(state, modes_a)
    nu_tilde = symplectic_eigenvalues_snu(cov_pt, state.hbar)
    return float(np.sum(np.maximum(0.0, -np.log2(nu_tilde))))


def negativity(state, modes_a):
    """Negativity N = (||rho^Tb||_1 - 1)/2 = (2^E_N - 1)/2."""
    return float((2 ** logarithmic_negativity(state, modes_a) - 1) / 2)


def is_separable(state, modes_a, atol=1e-8):
    """PPT (positive partial transpose) test across (modes_a | rest).

    Necessary for separability of any bipartite state; also **sufficient**
    when one side is a single mode (Simon, PRL 84, 2726 (2000); Werner &
    Wolf, PRA 64, 021802(R) (2001)).  For larger, larger bipartitions a
    True result is only the necessary PPT condition, not a proof of
    separability (bound-entangled Gaussian states exist beyond 1xN).
    """
    cov_pt = _partial_transpose_cov(state, modes_a)
    nu_tilde = symplectic_eigenvalues_snu(cov_pt, state.hbar)
    return bool(np.all(nu_tilde >= 1 - atol))


simon_criterion = is_separable


def epr_variance(state, modes=(0, 1)):
    """Duan et al. EPR-like combination Var(x1-x2) + Var(p1+p2) for a mode pair."""
    mode1, mode2 = modes
    sub = state.reduced([mode1, mode2])
    u_vec = np.array([1.0, 0.0, -1.0, 0.0])
    v_vec = np.array([0.0, 1.0, 0.0, 1.0])
    return float(u_vec @ sub.cov @ u_vec + v_vec @ sub.cov @ v_vec)


def duan_criterion(state, modes=(0, 1)):
    """Duan, Giedke, Cirac & Zoller sufficient inseparability test (PRL 84, 2722 (2000)).

    True (Var(x1-x2) + Var(p1+p2) < 2 hbar) confirms entanglement; False
    does not imply separability (this is sufficient, not necessary).
    """
    return bool(epr_variance(state, modes) < 2 * state.hbar)


# ---------------------------------------------------------------------------
# steering
# ---------------------------------------------------------------------------


def steerability(state, modes_a, mode_b, base=np.e):
    """Gaussian steerability G^(A->B) of mode_b by modes_a (Kogias, Lee, Ragy & Adesso, PRL 114, 060403 (2015)).

    G = sum over symplectic eigenvalues nu_k (SNU) of the Schur complement
    sigma_B|A = sigma_B - sigma_AB^T sigma_A^-1 sigma_AB that fall below 1,
    of -log(nu_k); zero (no steering) if none do.  Not symmetric in
    general: call with (modes_a, mode_b) and (mode_b's-set, one of modes_a)
    separately to compare both directions.  ``base=e`` (nats) by default.
    """
    modes_a = list(np.atleast_1d(modes_a))
    sub = state.reduced(modes_a + [mode_b])
    k = len(modes_a)
    idx_a = ops.mode_indices(range(k))
    idx_b = ops.mode_indices([k])
    sigma_a = sub.cov[np.ix_(idx_a, idx_a)]
    sigma_b = sub.cov[np.ix_(idx_b, idx_b)]
    sigma_ab = sub.cov[np.ix_(idx_a, idx_b)]
    schur = sigma_b - sigma_ab.T @ np.linalg.inv(sigma_a) @ sigma_ab
    nu = symplectic_eigenvalues_snu(schur, state.hbar)
    g_nats = np.sum(np.maximum(0.0, -np.log(nu)))
    return float(g_nats / np.log(base))


# ---------------------------------------------------------------------------
# entanglement of formation and discord -- pure states only
# ---------------------------------------------------------------------------


def entanglement_of_formation(state, modes_a, base=2):
    """Entanglement of formation across (modes_a | rest), for **pure** global states only.

    For any pure bipartite state (Gaussian or not) EoF exactly equals the
    entanglement entropy of either reduced state -- the well-defined,
    exactly-computable case.  Mixed-state EoF requires a convex-roof
    optimization this package does not implement; use
    :func:`logarithmic_negativity` or :func:`negativity` for mixed states.
    """
    if not state.is_pure():
        raise ValueError(
            "entanglement_of_formation is only implemented for pure global states "
            "(where it equals the reduced-state entropy); for mixed states use "
            "logarithmic_negativity() or negativity() instead"
        )
    return state.reduced(list(np.atleast_1d(modes_a))).von_neumann_entropy(base=base)


def gaussian_discord(state, modes_a, mode_b, base=2):
    """Quantum discord D(modes_a : mode_b), for **pure** global states only.

    For a pure bipartite state, discord in either direction exactly equals
    the entanglement entropy (same value as :func:`entanglement_of_formation`).
    The general mixed-state closed form (Giorda & Paris, PRL 105, 020503
    (2010); Adesso & Datta, PRL 105, 030501 (2010)) is not implemented; use
    :func:`logarithmic_negativity` for mixed states.
    """
    if not state.is_pure():
        raise ValueError(
            "gaussian_discord is only implemented for pure global states "
            "(where it equals the reduced-state entropy); for mixed states use "
            "logarithmic_negativity() instead"
        )
    del mode_b  # direction-independent for pure states
    return state.reduced(list(np.atleast_1d(modes_a))).von_neumann_entropy(base=base)
