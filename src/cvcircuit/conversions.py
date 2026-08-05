"""Unit and ordering conversions.

Three independent conventions meet in this module, each documented in
:mod:`docs/conventions.md`:

* **Quadrature ordering** -- this package stores phase-space vectors and
  covariance matrices in **xpxp** order, r = (x1, p1, x2, p2, ..., xn, pn).
  :func:`xpxp_to_xxpp` / :func:`xxpp_to_xpxp` convert to/from the **xxpp**
  block order r = (x1, ..., xn, p1, ..., pn) used by Strawberry Fields,
  TheWalrus and the Banchi-Braunstein-Pirandola fidelity paper.
* **Squeezing units** -- :func:`r_to_db` / :func:`db_to_r` convert the
  squeezing parameter r (nepers, as it enters S(r) = diag(e^-r, e^r)) to and
  from decibels, using the sign convention r_dB = 10 log10(exp(-2r)), so
  squeezing (r > 0) is reported as **negative** dB.  This matches
  ``lcg_plus.conversions`` and everyday lab usage ("-3 dB of squeezing");
  some references (e.g. TN_Sampling) use the opposite sign.
* **Complex amplitude <-> quadratures** -- :func:`alpha_to_xp` /
  :func:`xp_to_alpha` convert between the complex amplitude alpha = (x +
  ip)/sqrt(2 hbar) and an xpxp displacement vector, for whichever hbar the
  state uses.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# xpxp <-> xxpp reordering
# ---------------------------------------------------------------------------


def xpxp_to_xxpp_order(n):
    """Index array that reorders an xpxp vector of n modes into xxpp order."""
    return np.concatenate([np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)])


def xxpp_to_xpxp_order(n):
    """Index array that reorders an xxpp vector of n modes into xpxp order."""
    return np.argsort(xpxp_to_xxpp_order(n))


def _reorder(array, order):
    array = np.asarray(array)
    if array.ndim == 1:
        return array[order]
    if array.ndim == 2:
        return array[np.ix_(order, order)]
    raise ValueError(f"expected a 1-D vector or 2-D matrix, got shape {array.shape}")


def xpxp_to_xxpp(array):
    """Reorder a vector (2n,) or matrix (2n, 2n) from xpxp to xxpp order."""
    n = array.shape[-1] // 2
    return _reorder(array, xpxp_to_xxpp_order(n))


def xxpp_to_xpxp(array):
    """Reorder a vector (2n,) or matrix (2n, 2n) from xxpp to xpxp order."""
    n = array.shape[-1] // 2
    return _reorder(array, xxpp_to_xpxp_order(n))


# ---------------------------------------------------------------------------
# squeezing <-> decibels
# ---------------------------------------------------------------------------


def r_to_db(r):
    """Squeezing parameter r (nepers) to squeezing level in dB.

    r_dB = 10 log10(exp(-2r)), so squeezing (r > 0, reduced variance in the
    squeezed quadrature) is reported as negative dB, e.g. r=1 -> -8.686 dB.
    """
    return 10 * np.log10(np.exp(-2 * r))


def db_to_r(db):
    """Inverse of :func:`r_to_db`: squeezing level in dB to r (nepers)."""
    return -0.5 * np.log(10 ** (db / 10))


# ---------------------------------------------------------------------------
# transmittance <-> dB (fibre / free-space loss)
# ---------------------------------------------------------------------------


def eta_to_db(eta):
    """Power transmittance eta in (0, 1] to loss in dB (positive for eta<1)."""
    if not 0 < eta <= 1:
        raise ValueError(f"eta must be in (0, 1], got {eta}")
    return -10 * np.log10(eta)


def db_to_eta(db):
    """Inverse of :func:`eta_to_db`: loss in dB to power transmittance eta."""
    return 10 ** (-db / 10)


def distance_to_eta(length_km, db_per_km=0.2):
    """Fibre transmittance over ``length_km`` at ``db_per_km`` attenuation.

    The default 0.2 dB/km is standard telecom fibre at 1550 nm.
    """
    return db_to_eta(db_per_km * length_km)


# ---------------------------------------------------------------------------
# complex amplitude <-> quadratures
# ---------------------------------------------------------------------------


def alpha_to_xp(alpha, hbar=2):
    """Complex amplitude(s) alpha to an xpxp displacement vector.

    alpha = (x + ip) / sqrt(2 hbar), so d = sqrt(2 hbar) (Re a, Im a).
    Accepts a scalar (one mode) or a 1-D array of amplitudes (n modes) and
    returns a length-2n vector.
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=complex))
    scale = np.sqrt(2 * hbar)
    vec = np.empty(2 * alpha.size)
    vec[0::2] = scale * alpha.real
    vec[1::2] = scale * alpha.imag
    return vec


def xp_to_alpha(vec, hbar=2):
    """Inverse of :func:`alpha_to_xp`: an xpxp vector to complex amplitude(s)."""
    vec = np.atleast_1d(np.asarray(vec, dtype=float))
    if vec.size % 2 != 0:
        raise ValueError(f"expected an even-length xpxp vector, got size {vec.size}")
    x, p = vec[0::2], vec[1::2]
    return (x + 1j * p) / np.sqrt(2 * hbar)
