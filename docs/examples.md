# Worked Examples

Self-contained, runnable snippets, each with real output as a comment or a
fenced block below it.

- [Squeezing degradation under loss](#squeezing-degradation-under-loss)
- [EPR entanglement and homodyne conditioning](#epr-entanglement-and-homodyne-conditioning)
- [Entanglement across a range of squeezing](#entanglement-across-a-range-of-squeezing)
- [CV teleportation with feedforward](#cv-teleportation-with-feedforward)
- [Circuit diagrams](#circuit-diagrams)
- [Cross-checking against an independent Fock-space simulator](#cross-checking-against-an-independent-fock-space-simulator)

## Squeezing degradation under loss

```python
from cvcircuit import squeezed

for eta in [1.0, 0.9, 0.7, 0.5]:
    s = squeezed(1.0).loss(eta, mode=0)
    print(f"eta={eta}: squeezing = {s.squeezing_db(0):.3f} dB")
```

```text
eta=1.0: squeezing = -8.686 dB
eta=0.9: squeezing = -6.540 dB
eta=0.7: squeezing = -4.037 dB
eta=0.5: squeezing = -2.459 dB
```

## EPR entanglement and homodyne conditioning

Measuring one half of a two-mode squeezed vacuum (TMSV) pair conditionally
squeezes the other half — the basis of CV quantum key distribution and
teleportation:

```python
from cvcircuit import two_mode_squeezed

epr = two_mode_squeezed(1.5)
outcome, conditional = epr.homodyne(mode=0, angle=0.0, seed=0)
print(f"outcome: {outcome:.4f}")
print(f"mode 1 conditional Var(x): {conditional.reduced([1]).quad_variance(0, 0.0):.4f}")
```

```text
outcome: 0.3989
mode 1 conditional Var(x): 0.0993
```

(Vacuum is 1.0 at `hbar=2`, so mode 1 is squeezed almost 10 dB below shot
noise, *conditioned on* mode 0's measured outcome — the textbook EPR
conditional-squeezing result, $\mathrm{Var}(x_2|x_1) = 1/\cosh(2r)$.)

## Entanglement across a range of squeezing

```python
from cvcircuit import two_mode_squeezed, logarithmic_negativity, is_separable, duan_criterion

for r in [0.3, 1.0, 2.0]:
    tmsv = two_mode_squeezed(r)
    print(f"r={r}: log_neg={logarithmic_negativity(tmsv, [0]):.4f}  "
          f"separable={is_separable(tmsv, [0])}  duan={duan_criterion(tmsv)}")
```

```text
r=0.3: log_neg=0.8656  separable=False  duan=True
r=1.0: log_neg=2.8854  separable=False  duan=True
r=2.0: log_neg=5.7708  separable=False  duan=True
```

## CV teleportation with feedforward

Any numeric gate parameter may be a callable receiving the dict of
measurement outcomes recorded so far — here it implements the
Braunstein-Kimble feedforward correction:

```python
import numpy as np
from cvcircuit import Circuit, coherent, fidelity

alpha_in = 1.0 + 0.5j
c = Circuit(3)
c.displace(alpha_in, mode=0)                    # the state to teleport
c.two_mode_squeeze(2.0, modes=(1, 2))            # EPR resource, modes 1 & 2
c.beamsplitter(np.pi / 4, modes=(0, 1))
c.homodyne(0, angle=0.0, label="xA")
c.homodyne(1, angle=np.pi / 2, label="pB")
c.displace(lambda m: (m["xA"] + 1j * m["pB"]) / np.sqrt(2), mode=2)

final_state, records = c._execute(seed=0)
teleported = final_state.reduced([2])
print("records:", {k: round(v, 4) for k, v in records.items()})
print("fidelity:", fidelity(coherent(alpha_in), teleported))
```

```text
records: {'xA': 8.4717, 'pB': -3.7413}
fidelity: 0.9559693068318507
```

(`c.state(seed=...)` is the usual public entry point when you don't need
the raw outcome records; `_execute` is used here just to print them.
Fidelity → 1 as the resource state's squeezing $r\to\infty$ — see demo
`07_teleportation.ipynb` for the full curve.)

## Circuit diagrams

```python
from cvcircuit import Circuit
import numpy as np

c = Circuit(2)
c.squeeze(1.0, mode=0).beamsplitter(np.pi / 4, modes=(0, 1)).loss(0.1, mode=0).homodyne(0, label="x")
print(c)
```

```text
mode 0: [S 1][BS π/4][η 0.1][x(0)]
                │
mode 1: ─────[BS π/4]─────────────
```

`c.draw()` renders the same circuit as a styled matplotlib figure (colour-
coded by operation type); `c.save("circuit.svg")` writes it to disk, and a
bare `c` in a Jupyter cell shows it inline.

## Cross-checking against an independent Fock-space simulator

The test suite's `tests/reference.py` is a from-scratch, truncated-Fock-space
simulator (explicit creation/annihilation matrices, unitaries built via
matrix exponentials of the relevant Hamiltonians) used to independently
verify every symplectic operation and channel:

```python
import sys
sys.path.insert(0, "tests")   # only needed to import the test helper directly
from reference import FockSimulator
from cvcircuit import squeezed

sim = FockSimulator(1, cutoff=35, hbar=2)
sim.squeeze(0.5, phi=0.9)
_, cov_fock = sim.moments()

cov_gaussian = squeezed(0.5, phi=0.9).cov
print("agreement:", (abs(cov_fock - cov_gaussian) < 1e-4).all())
```

```text
agreement: True
```
