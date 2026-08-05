"""Circuit: a chainable builder for Gaussian circuits, mirroring qcircsim's QuantumCircuit.

>>> from cvcircuit import Circuit
>>> c = Circuit(2)
>>> c.squeeze(1.0, mode=0).beamsplitter(np.pi / 4, modes=(0, 1))    # doctest: +SKIP
>>> c.state().symplectic_eigenvalues()                              # doctest: +SKIP

Every builder method (``squeeze``, ``beamsplitter``, ``loss``, ``homodyne``,
...) appends an :class:`Instruction` and returns ``self``, so calls chain.
Numeric parameters may be a plain value or a **callable** taking the dict of
measurement outcomes so far -- this is how feedforward (e.g. CV
teleportation) is expressed; see :mod:`docs/guide.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import channels as _channels
from . import decompositions
from . import ops
from . import state as _state

_UNITARY_KINDS = (
    "displace", "rotate", "squeeze", "beamsplitter", "two_mode_squeeze",
    "interferometer", "cz", "cx", "shear", "symplectic",
)
_CHANNEL_KINDS = ("loss", "amplify", "noise")
_MEASUREMENT_KINDS = ("homodyne", "heterodyne", "generaldyne")


@dataclass(frozen=True)
class Instruction:
    """One operation in a :class:`Circuit`.

    ``params`` values may be plain numbers/arrays or callables (feedforward)
    taking the dict of measurement outcomes recorded so far.
    """

    kind: str
    modes: tuple
    params: dict = field(default_factory=dict)
    label: str | None = None


class Circuit:
    """A chainable builder for a Gaussian circuit on ``num_modes`` modes.

    Parameters
    ----------
    num_modes:
        Number of modes.
    hbar:
        [x, p] = i hbar for every state this circuit produces.
    initial:
        A :class:`~cvcircuit.state.GaussianState` to start from instead of
        vacuum (must match ``num_modes`` and ``hbar``).
    """

    def __init__(self, num_modes, hbar=2, initial=None):
        if num_modes < 1:
            raise ValueError(f"num_modes must be >= 1, got {num_modes}")
        if initial is not None:
            if initial.num_modes != num_modes:
                raise ValueError(f"initial state has {initial.num_modes} mode(s), expected {num_modes}")
            if initial.hbar != hbar:
                raise ValueError(f"initial state has hbar={initial.hbar}, expected {hbar}")
        self.num_modes = num_modes
        self.hbar = float(hbar)
        self.initial = initial
        self.instructions = []

    def _check_modes(self, modes):
        modes = tuple(int(m) for m in np.atleast_1d(modes))
        for m in modes:
            if not 0 <= m < self.num_modes:
                raise ValueError(f"mode {m} out of range for {self.num_modes}-mode circuit")
        if len(set(modes)) != len(modes):
            raise ValueError(f"modes must be distinct, got {modes}")
        return modes

    def _append(self, kind, modes, params, label=None):
        self.instructions.append(Instruction(kind, self._check_modes(modes), params, label))
        return self

    # -- unitary operations ----------------------------------------------

    def displace(self, alpha, mode=0):
        """Displace ``mode`` by complex amplitude alpha."""
        return self._append("displace", [mode], {"alpha": alpha})

    def rotate(self, phi, mode=0):
        """Phase rotation R(phi) on ``mode``."""
        return self._append("rotate", [mode], {"phi": phi})

    def squeeze(self, r, phi=0.0, mode=0):
        """Single-mode squeezing S(r, phi) on ``mode``."""
        return self._append("squeeze", [mode], {"r": r, "phi": phi})

    def beamsplitter(self, theta, modes=(0, 1), phi=0.0):
        """Beamsplitter (T = cos^2(theta)) on ``modes``."""
        return self._append("beamsplitter", modes, {"theta": theta, "phi": phi})

    def two_mode_squeeze(self, r, modes=(0, 1), phi=0.0):
        """Two-mode squeezing on ``modes``."""
        return self._append("two_mode_squeeze", modes, {"r": r, "phi": phi})

    def interferometer(self, unitary, modes=None, mesh="clements"):
        """Passive interferometer (unitary matrix) on ``modes`` (default: the first len(unitary) modes).

        By default (``mesh="clements"``) this decomposes the unitary into
        its physical mesh of beamsplitters and phase shifters (the
        depth-optimal rectangular mesh, Clements et al., Optica 3, 1460
        (2016)) and appends *those* as ordinary instructions -- so it draws,
        counts toward ``depth()``, and feeds ``to_symplectic()``/
        ``to_channel()`` exactly like a hand-built mesh would.  Pass
        ``mesh="reck"`` for the simpler but deeper (2n-3) triangular mesh,
        or ``mesh=None`` to append a single opaque ``"interferometer"``
        instruction instead (matching
        :meth:`~cvcircuit.state.GaussianState.interferometer`'s direct,
        undecomposed application -- one box in diagrams, one line of depth,
        cheaper for large n).
        """
        unitary = np.asarray(unitary)
        if modes is None:
            modes = range(unitary.shape[0])
        modes = self._check_modes(modes)
        if unitary.shape[0] != len(modes):
            raise ValueError(f"unitary has shape {unitary.shape}, expected {len(modes)} modes")

        if mesh is None:
            return self._append("interferometer", modes, {"unitary": unitary})
        if mesh == "clements":
            elements = decompositions.clements_decomposition(unitary)
        elif mesh == "reck":
            elements = decompositions.interferometer_mesh(unitary)
        else:
            raise ValueError(f"mesh must be 'clements', 'reck', or None, got {mesh!r}")

        for element in elements:
            if element[0] == "phase":
                _, local_mode, angle = element
                self.rotate(angle, mode=modes[local_mode])
            else:
                _, local_m1, local_m2, theta, phi = element
                self.beamsplitter(theta, modes=(modes[local_m1], modes[local_m2]), phi=phi)
        return self

    def cz(self, s=1.0, modes=(0, 1)):
        """CV controlled-Z gate, edge weight s, on ``modes``."""
        return self._append("cz", modes, {"s": s})

    def cx(self, s=1.0, modes=(0, 1)):
        """CV controlled-X / SUM gate, gain s, on ``modes``."""
        return self._append("cx", modes, {"s": s})

    def shear(self, s=1.0, mode=0):
        """Shear/P-gate on ``mode``: p -> p + s x."""
        return self._append("shear", [mode], {"s": s})

    def symplectic(self, matrix, modes=None, d=None):
        """Apply an arbitrary symplectic ``matrix`` (and optional displacement ``d``) to ``modes``."""
        matrix = np.asarray(matrix, dtype=float)
        if modes is None:
            modes = range(matrix.shape[0] // 2)
        return self._append("symplectic", modes, {"matrix": matrix, "d": d})

    # -- channels -----------------------------------------------------------

    def loss(self, eta, mode=0, nbar_env=0.0):
        """Pure loss channel (transmittance eta) on ``mode``."""
        return self._append("loss", [mode], {"eta": eta, "nbar_env": nbar_env})

    def amplify(self, g, mode=0, nbar_env=0.0):
        """Phase-insensitive amplifier channel (gain g) on ``mode``."""
        return self._append("amplify", [mode], {"g": g, "nbar_env": nbar_env})

    def noise(self, xi, mode=0):
        """Isotropic additive noise channel, xi in shot-noise units, on ``mode``."""
        return self._append("noise", [mode], {"xi": xi})

    def phase_noise(self, sigma, mode=0, mean_phase=0.0):
        """Gaussian phase diffusion phi ~ N(mean_phase, sigma^2) on ``mode``."""
        return self._append("phase_noise", [mode], {"sigma": sigma, "mean_phase": mean_phase})

    # -- measurements ------------------------------------------------------

    def homodyne(self, mode=0, angle=0.0, efficiency=1.0, electronic_noise=0.0, label=None):
        """Homodyne measurement of the quadrature at ``angle`` on ``mode``."""
        return self._append(
            "homodyne", [mode],
            {"angle": angle, "efficiency": efficiency, "electronic_noise": electronic_noise},
            label=label,
        )

    def heterodyne(self, mode=0, label=None):
        """Heterodyne measurement on ``mode``."""
        return self._append("heterodyne", [mode], {}, label=label)

    def generaldyne(self, modes, cov_povm, label=None):
        """General-dyne measurement of ``modes`` against POVM covariance ``cov_povm``."""
        return self._append("generaldyne", modes, {"cov_povm": cov_povm}, label=label)

    # -- execution -----------------------------------------------------------

    @staticmethod
    def _resolve(params, records):
        return {k: (v(records) if callable(v) else v) for k, v in params.items()}

    def _measurement_label(self, instr, counter):
        return instr.label if instr.label is not None else f"{instr.kind}_{instr.modes}_{counter}"

    def _execute(self, seed=None):
        """Run once; returns (final_state, records) where records maps measurement labels to outcomes."""
        rng = np.random.default_rng(seed)
        current = self.initial.copy() if self.initial is not None else _state.vacuum(self.num_modes, hbar=self.hbar)
        records = {}
        counter = 0
        for instr in self.instructions:
            params = self._resolve(instr.params, records)
            kind, modes = instr.kind, instr.modes
            if kind == "displace":
                current = current.displace(params["alpha"], mode=modes[0])
            elif kind == "rotate":
                current = current.rotate(params["phi"], mode=modes[0])
            elif kind == "squeeze":
                current = current.squeeze(params["r"], phi=params["phi"], mode=modes[0])
            elif kind == "beamsplitter":
                current = current.beamsplitter(params["theta"], modes=modes, phi=params["phi"])
            elif kind == "two_mode_squeeze":
                current = current.two_mode_squeeze(params["r"], modes=modes, phi=params["phi"])
            elif kind == "interferometer":
                current = current.interferometer(params["unitary"], modes=modes)
            elif kind == "cz":
                current = current.cz(params["s"], modes=modes)
            elif kind == "cx":
                current = current.cx(params["s"], modes=modes)
            elif kind == "shear":
                current = current.shear(params["s"], mode=modes[0])
            elif kind == "symplectic":
                current = current.apply_symplectic(params["matrix"], modes=modes, d=params.get("d"))
            elif kind == "loss":
                current = current.loss(params["eta"], mode=modes[0], nbar_env=params["nbar_env"])
            elif kind == "amplify":
                current = current.amplify(params["g"], mode=modes[0], nbar_env=params["nbar_env"])
            elif kind == "noise":
                current = current.add_noise(params["xi"], mode=modes[0])
            elif kind == "phase_noise":
                current = current.phase_noise(params["sigma"], mode=modes[0], mean_phase=params["mean_phase"])
            elif kind in _MEASUREMENT_KINDS:
                seed_k = int(rng.integers(2**31))
                if kind == "homodyne":
                    outcome, current = current.homodyne(
                        mode=modes[0], angle=params["angle"], efficiency=params["efficiency"],
                        electronic_noise=params["electronic_noise"], seed=seed_k,
                    )
                elif kind == "heterodyne":
                    outcome, current = current.heterodyne(mode=modes[0], seed=seed_k)
                else:
                    outcome, current = current.generaldyne(modes, params["cov_povm"], seed=seed_k)
                records[self._measurement_label(instr, counter)] = outcome
                counter += 1
            else:
                raise RuntimeError(f"unknown instruction kind {kind!r}")  # pragma: no cover
        return current, records

    def state(self, seed=None):
        """Run once and return the final :class:`~cvcircuit.state.GaussianState`."""
        final_state, _ = self._execute(seed=seed)
        return final_state

    def run(self, shots=1000, seed=None):
        """Run ``shots`` times and return a :class:`Result` of the measurement outcomes."""
        if shots < 1:
            raise ValueError(f"shots must be >= 1, got {shots}")
        rng = np.random.default_rng(seed)
        records_list = []
        for _ in range(shots):
            _, records = self._execute(seed=int(rng.integers(2**31)))
            records_list.append(records)
        return Result(records_list)

    # -- unitary/channel extraction -----------------------------------------

    def _local_symplectic(self, instr):
        kind, params = instr.kind, instr.params
        if any(callable(v) for v in params.values()):
            raise ValueError(f"instruction {kind!r} has a feedforward (callable) parameter; cannot extract a fixed matrix")
        if kind == "displace":
            return np.eye(2), ops.displacement(params["alpha"], hbar=self.hbar)
        if kind == "rotate":
            return ops.rotation(params["phi"]), None
        if kind == "squeeze":
            return ops.squeezing(params["r"], params["phi"]), None
        if kind == "beamsplitter":
            return ops.beamsplitter(params["theta"], params["phi"]), None
        if kind == "two_mode_squeeze":
            return ops.two_mode_squeezing(params["r"], params["phi"]), None
        if kind == "interferometer":
            return ops.interferometer(params["unitary"]), None
        if kind == "cz":
            return ops.cz(params["s"]), None
        if kind == "cx":
            return ops.cx(params["s"]), None
        if kind == "shear":
            return ops.shear(params["s"]), None
        if kind == "symplectic":
            return np.asarray(params["matrix"], dtype=float), params.get("d")
        raise ValueError(f"{kind!r} is not a unitary instruction")

    def to_symplectic(self):
        """The overall (S, d) of this circuit's unitary part; raises if it has channels or measurements."""
        for instr in self.instructions:
            if instr.kind not in _UNITARY_KINDS:
                raise ValueError(f"to_symplectic() requires a purely unitary circuit; found a {instr.kind!r} instruction")
        n = self.num_modes
        s_total, d_total = np.eye(2 * n), np.zeros(2 * n)
        for instr in self.instructions:
            s_local, d_local = self._local_symplectic(instr)
            s_full = ops.expand(s_local, instr.modes, n)
            s_total = s_full @ s_total
            d_total = s_full @ d_total
            if d_local is not None:
                d_total = d_total + ops.expand_vector(d_local, instr.modes, n)
        return s_total, d_total

    def to_channel(self):
        """The overall (X, Y, d) of this circuit; raises on measurements or phase noise."""
        for instr in self.instructions:
            if instr.kind not in _UNITARY_KINDS and instr.kind not in _CHANNEL_KINDS:
                raise ValueError(f"to_channel() cannot include a {instr.kind!r} instruction")
        n = self.num_modes
        x_total, y_total, d_total = np.eye(2 * n), np.zeros((2 * n, 2 * n)), np.zeros(2 * n)
        for instr in self.instructions:
            kind, params = instr.kind, instr.params
            if kind in _UNITARY_KINDS:
                s_local, d_local = self._local_symplectic(instr)
                x_full = ops.expand(s_local, instr.modes, n)
                y_add = np.zeros((2 * n, 2 * n))
                if d_local is not None:
                    d_local_full = ops.expand_vector(d_local, instr.modes, n)
                else:
                    d_local_full = 0.0
            else:
                if any(callable(v) for v in params.values()):
                    raise ValueError(f"instruction {kind!r} has a feedforward (callable) parameter")
                if kind == "loss":
                    ch = _channels.loss(params["eta"], params["nbar_env"], hbar=self.hbar)
                elif kind == "amplify":
                    ch = _channels.amplifier(params["g"], params["nbar_env"], hbar=self.hbar)
                else:  # noise
                    ch = _channels.excess_noise(params["xi"], hbar=self.hbar)
                x_full = ops.expand(ch.X, instr.modes, n)
                idx = ops.mode_indices(instr.modes)
                y_add = np.zeros((2 * n, 2 * n))
                y_add[np.ix_(idx, idx)] = ch.Y
                d_local_full = 0.0
            x_total = x_full @ x_total
            y_total = x_full @ y_total @ x_full.T + y_add
            d_total = x_full @ d_total + d_local_full
        return x_total, y_total, d_total

    # -- circuit manipulation -------------------------------------------------

    def compose(self, other, modes=None):
        """Append ``other``'s instructions, optionally remapped onto ``modes``."""
        if other.hbar != self.hbar:
            raise ValueError(f"cannot compose circuits with different hbar ({self.hbar} vs {other.hbar})")
        if modes is None:
            if other.num_modes != self.num_modes:
                raise ValueError(f"cannot compose a {other.num_modes}-mode circuit without specifying modes")
            mapping = {i: i for i in range(other.num_modes)}
        else:
            modes = self._check_modes(modes)
            if len(modes) != other.num_modes:
                raise ValueError(f"modes has {len(modes)} entries, expected {other.num_modes}")
            mapping = dict(enumerate(modes))
        for instr in other.instructions:
            new_modes = tuple(mapping[m] for m in instr.modes)
            self.instructions.append(Instruction(instr.kind, new_modes, dict(instr.params), instr.label))
        return self

    def inverse(self):
        """A new circuit implementing the inverse of this circuit's unitary part."""
        s_mat, d_vec = self.to_symplectic()
        s_inv = np.linalg.inv(s_mat)
        new_circuit = Circuit(self.num_modes, hbar=self.hbar)
        new_circuit.symplectic(s_inv, modes=range(self.num_modes), d=-s_inv @ d_vec)
        return new_circuit

    def copy(self):
        """A shallow copy of this circuit (a new instruction list, same instructions)."""
        new_circuit = Circuit(self.num_modes, hbar=self.hbar, initial=self.initial)
        new_circuit.instructions = [Instruction(i.kind, i.modes, dict(i.params), i.label) for i in self.instructions]
        return new_circuit

    def _column_layout(self):
        """Per-instruction column index via a per-mode frontier allocator (used by depth() and diagrams)."""
        frontier = [0] * self.num_modes
        columns = []
        for instr in self.instructions:
            col = max((frontier[m] for m in instr.modes), default=0)
            columns.append(col)
            for m in instr.modes:
                frontier[m] = col + 1
        return columns, frontier

    def depth(self):
        """Circuit depth: the longest chain of operations sharing a mode."""
        _, frontier = self._column_layout()
        return max(frontier) if frontier else 0

    def count_ops(self):
        """Dict of {instruction kind: count}."""
        counts = {}
        for instr in self.instructions:
            counts[instr.kind] = counts.get(instr.kind, 0) + 1
        return counts

    def __len__(self):
        return len(self.instructions)

    # -- diagrams -------------------------------------------------------------

    def draw_text(self):
        """Unicode text-art circuit diagram."""
        from . import diagram

        return diagram.render_text(diagram.layout(self))

    def draw(self, ax=None, style=None):
        """A styled matplotlib circuit diagram; returns the ``Axes``."""
        from . import diagram

        return diagram.render_mpl(diagram.layout(self), ax=ax, style=style)

    def save(self, path, dpi=200):
        """Save a styled circuit diagram to ``path`` (format inferred from the suffix: .svg/.png/.pdf)."""
        from . import diagram

        diagram.save(self, path, dpi=dpi)

    def __str__(self):
        return self.draw_text()

    def _repr_svg_(self):
        from . import diagram

        return diagram.repr_svg(self)

    def __repr__(self):
        return f"Circuit({self.num_modes} modes, hbar={self.hbar:g}, {len(self.instructions)} instructions)"


class Result:
    """Outcomes of running a :class:`Circuit` for several shots.

    Internally a list of per-shot ``{label: outcome}`` dicts (outcomes may
    be scalars, complex heterodyne amplitudes, or vectors, so a single
    rectangular array isn't a natural fit); ``result[label]`` gives the
    array of that label's outcomes across shots.
    """

    def __init__(self, records):
        self.records = records
        self.shots = len(records)
        self.labels = list(records[0].keys()) if records else []

    def __getitem__(self, label):
        if label not in self.labels:
            raise KeyError(f"no measurement labelled {label!r}; available labels: {self.labels}")
        return np.array([r[label] for r in self.records])

    def mean(self, label=None):
        """Mean outcome for ``label`` (or a dict of means for every label, if ``label`` is None)."""
        if label is not None:
            return np.mean(self[label], axis=0)
        return {lbl: self.mean(lbl) for lbl in self.labels}

    def var(self, label=None):
        """Variance of the outcomes for ``label`` (or a dict, if ``label`` is None)."""
        if label is not None:
            return np.var(self[label], axis=0)
        return {lbl: self.var(lbl) for lbl in self.labels}

    def cov(self, label_x, label_y):
        """Covariance between two scalar-valued labels' outcomes across shots."""
        return float(np.cov(self[label_x], self[label_y])[0, 1])

    def histogram(self, label, bins=50):
        """Histogram (counts, bin_edges) of a scalar label's outcomes."""
        return np.histogram(self[label], bins=bins)

    def __repr__(self):
        return f"Result({self.shots} shots, labels={self.labels})"
