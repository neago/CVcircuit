# User Guide

- [Installation](#installation)
- [Two ways to work](#two-ways-to-work)
- [States](#states)
- [Operations](#operations)
- [Channels and noise](#channels-and-noise)
- [Measurements](#measurements)
- [Circuits and feedforward](#circuits-and-feedforward)
- [Phase space and diagrams](#phase-space-and-diagrams)
- [Quantum information](#quantum-information)
- [Conventions](#conventions)

## Installation

```bash
pip install git+https://github.com/neago/CVcircuit.git
# or, for local development:
git clone https://github.com/neago/CVcircuit.git && cd CVcircuit && uv sync
```

## Two ways to work

`GaussianState` is an **immutable** value object — every method returns a
*new* state and leaves the original untouched, even ones that read like
verbs (`s.squeeze(...)` doesn't squeeze `s`, it returns a squeezed copy).
Build states by chaining:

```python
from cvcircuit import squeezed, vacuum
import numpy as np

s = squeezed(1.0) & vacuum()          # tensor product: 2 modes
s = s.beamsplitter(np.pi / 4, modes=(0, 1)).loss(0.1, mode=0)
```

`Circuit` is a **mutable builder** that records a sequence of operations
without running them; call `.state()` or `.run()` to execute:

```python
from cvcircuit import Circuit

c = Circuit(2)
c.squeeze(1.0, mode=0).beamsplitter(np.pi / 4, modes=(0, 1)).loss(0.1, mode=0)
s = c.state(seed=0)                   # same GaussianState as above
```

Use `GaussianState` for quick interactive exploration; use `Circuit` when
you want a reproducible, drawable program, shot sampling, or feedforward
(measurement outcomes feeding later gates — see [below](#circuits-and-feedforward)).

## States

Constructors (all take `hbar=2` by default):

| constructor | state |
|---|---|
| `vacuum(n)` | $n$-mode vacuum |
| `coherent(alpha)` | coherent state(s); `alpha` may be an array, for several modes at once |
| `squeezed(r, phi)` | single-mode squeezed vacuum |
| `squeezed_coherent(alpha, r, phi)` | displaced squeezed vacuum |
| `thermal(nbar)` | thermal state(s) |
| `two_mode_squeezed(r, phi)` / `epr(r)` | a two-mode squeezed vacuum (TMSV) pair |
| `graph_state(adjacency, r)` / `cluster_state` | CV cluster/graph state (finite squeezing $r$) |
| `random_state(n, pure=False, seed=None)` | random state, for tests/demos |

Combine states with `&` (tensor product) and pull modes back out with
`.reduced(modes)`:

```python
from cvcircuit import coherent, squeezed, thermal

joint = squeezed(0.8) & coherent(1 + 1j) & thermal(0.5)   # 3 modes
mode1 = joint.reduced([1])                                # just the coherent one
```

Every state carries scalar properties as methods:
`.purity()`, `.von_neumann_entropy(base=2)`, `.mean_photon_number(mode=None)`,
`.squeezing_db(mode)`, `.symplectic_eigenvalues()`, `.is_pure()`, `.is_classical()`,
`.williamson()` — see [`docs/api.md`](api.md) for the full list.

## Operations

Every symplectic gate is both an `ops.<name>` matrix factory and a
`GaussianState`/`Circuit` method:

```python
s = vacuum(2)
s = s.squeeze(1.0, mode=0)                    # single-mode squeezing
s = s.rotate(np.pi / 6, mode=1)               # phase rotation
s = s.beamsplitter(np.pi / 4, modes=(0, 1))   # T = cos^2(theta)
s = s.two_mode_squeeze(0.5, modes=(0, 1))     # entangling
s = s.cz(0.8, modes=(0, 1))                   # CV cluster-state gate
s = s.displace(1 + 0.5j, mode=0)
```

For an arbitrary passive interferometer, pass a unitary matrix:
`s.interferometer(U, modes=(0, 1, 2))` applies it directly as one
transform. `Circuit.interferometer(U, modes=..., mesh="clements")`
(the default) instead **decomposes** it into the physical mesh of
beamsplitters and phase shifters and appends those as ordinary
instructions, so it draws, counts toward `depth()`, and feeds
`to_symplectic()`/`to_channel()` like a hand-built mesh would — pass
`mesh="reck"` for the simpler but deeper triangular mesh, or `mesh=None`
for a single opaque instruction (one box, cheaper for large n). The
underlying decompositions are also directly callable:
`decompositions.clements_decomposition(U)` (depth-optimal rectangular
mesh, Clements et al., Optica 3, 1460 (2016)) and
`decompositions.interferometer_mesh(U)` (simpler triangular mesh, Reck et
al., PRL 73, 58 (1994)) — see demo 02.

## Channels and noise

```python
s = squeezed(1.0).loss(0.2, mode=0)           # 80% transmission
s = s.amplify(1.5, mode=0)                    # phase-insensitive amplifier
s = s.add_noise(0.05, mode=0)                 # isotropic excess noise, shot-noise units
s = s.phase_noise(0.1, mode=0)                # Gaussian phase diffusion (see below)
```

`loss`, `amplify` and `add_noise` are genuine `GaussianChannel`s (fixed
$X,Y$ maps) — see `docs/conventions.md` for the exact matrices, and
`channels.GaussianChannel` if you want to build or compose your own.
`phase_noise` is **not** a `GaussianChannel`: it tracks the exact first two
moments of a state's phase averaged over $\varphi\sim\mathcal N(\bar\varphi,\sigma^2)$,
which is genuinely non-Gaussian for a displaced or squeezed state. See
`docs/conventions.md` for why, and demo 03 for a direct comparison against
a fixed-phase-rotation channel.

## Measurements

```python
outcome, conditional_state = s.homodyne(mode=0, angle=0.0)
alpha, conditional_state = s.heterodyne(mode=0)
outcome, conditional_state = s.generaldyne(modes=[0, 1], cov_povm=some_2x2_or_4x4_cov)
```

All three return `(outcome, conditional_state)`; the measured mode(s) are
reset to vacuum and kept (mode indices never shift — `.reduced()` drops
them if you don't need them). `homodyne` takes `efficiency` (0,1] and
`electronic_noise` (shot-noise units) to model a lossy, noisy detector.
`generaldyne` is the general primitive — heterodyne is
`cov_povm=(hbar/2)*I`; a strongly squeezed `cov_povm` approaches homodyne
in one quadrature, but `homodyne` computes that limit exactly rather than
as a numerically singular special case.

## Circuits and feedforward

Any numeric parameter in a `Circuit` builder call may be a **callable**
instead of a plain number — it receives the dict of measurement outcomes
recorded so far. This is how CV teleportation, measurement-based gates,
and adaptive protocols are expressed:

```python
c = Circuit(3)
c.displace(1 + 0.5j, mode=0)                     # state to teleport
c.two_mode_squeeze(2.0, modes=(1, 2))            # EPR resource
c.beamsplitter(np.pi / 4, modes=(0, 1))
c.homodyne(0, angle=0.0, label="xA")
c.homodyne(1, angle=np.pi / 2, label="pB")
c.displace(lambda m: (m["xA"] + 1j * m["pB"]) / np.sqrt(2), mode=2)

teleported = c.state(seed=0).reduced([2])
```

`c.run(shots=10_000)` repeats the whole circuit (including feedforward)
`shots` times and returns a `Result`; `result["label"]`, `.mean()`, `.var()`,
`.cov()` and `.histogram()` summarize the outcomes. `c.to_symplectic()` and
`c.to_channel()` collapse a purely-unitary (resp. unitary+channel) circuit
into a single $(S,d)$ or $(X,Y,d)$, useful for inspecting or reusing the
overall transform.

## Phase space and diagrams

```python
s.wigner(x, p, mode=0)      # or .husimi(...) / .p_function(...) / .quasiprobability(..., s=0.5)
```

`cvcircuit.visualization` has the matplotlib plotting functions:
`plot_wigner`, `plot_husimi`, `plot_p`, `plot_marginal`, `plot_joint_marginal`
(for EPR-style correlation plots), `plot_ellipse` (overlay uncertainty
ellipses from several states), `plot_covariance`, `plot_wigner_grid`.

`Circuit` diagrams: `print(c)` or `c.draw_text()` for Unicode text art;
`c.draw(ax=None)` for a styled matplotlib diagram (`Axes`); `c.save(path)`
writes SVG/PNG/PDF (format from the suffix); a bare `c` in a Jupyter cell
renders the styled diagram inline via `_repr_svg_`.

## Quantum information

```python
from cvcircuit import fidelity, logarithmic_negativity, is_separable, duan_criterion

fidelity(state_a, state_b)                 # squared overlap; see docs/conventions.md
logarithmic_negativity(entangled, [0])     # across the (modes_a | rest) bipartition
is_separable(entangled, [0])               # PPT test
duan_criterion(entangled, modes=(0, 1))
```

See `docs/conventions.md` for exact formulas and literature references for
every measure, including the deliberate scope restriction on
`entanglement_of_formation`/`gaussian_discord` to pure global states.

## Conventions

ħ, quadrature ordering, every symplectic matrix and channel, the
measurement update formulas, the dB sign convention, and a cross-reference
table against Brask/Weedbrook/thewalrus notation all live in
[`docs/conventions.md`](conventions.md) — read it once before comparing
numbers against another tool or paper.
