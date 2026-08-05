# API Reference

- [GaussianState](#gaussianstate)
- [State constructors](#state-constructors)
- [Circuit](#circuit)
- [Result](#result)
- [ops](#ops)
- [channels](#channels)
- [measure](#measure)
- [decompositions](#decompositions)
- [conversions](#conversions)
- [quantum_info](#quantum_info)
- [visualization](#visualization)

---

## GaussianState

`GaussianState(mean, cov, hbar=2)` — immutable; every method below returns
a *new* state.

**Combine / reduce**

| member | description |
|---|---|
| `a & b` | tensor product |
| `.reduced(modes)` / `.partial_trace(modes)` | reduced state on `modes` |
| `.permute(order)` | reorder modes |
| `.copy()` | independent copy |

**Unitary operations** (all return a new state)

| member | description |
|---|---|
| `.displace(alpha, mode=0)` | displacement by complex amplitude |
| `.displace_xp(x, p, mode=0)` | displacement by (x, p) directly |
| `.rotate(phi, mode=0)` | phase rotation |
| `.fourier(mode=0)` | 90° rotation |
| `.squeeze(r, phi=0.0, mode=0)` | single-mode squeezing |
| `.shear(s=1.0, mode=0)` | shear / P-gate |
| `.beamsplitter(theta, modes=(0,1), phi=0.0)` | $T=\cos^2\theta$ |
| `.two_mode_squeeze(r, modes=(0,1), phi=0.0)` | |
| `.interferometer(unitary, modes=None)` | passive n-mode unitary |
| `.cz(s=1.0, modes=(0,1))` | CV controlled-Z |
| `.cx(s=1.0, modes=(0,1))` | CV controlled-X / SUM |
| `.apply_symplectic(matrix, modes=None, d=None)` | arbitrary symplectic |

**Channels**

| member | description |
|---|---|
| `.apply_channel(channel, modes)` | apply a `GaussianChannel` |
| `.loss(eta, mode=0, nbar_env=0.0)` | |
| `.amplify(g, mode=0, nbar_env=0.0)` | |
| `.add_noise(xi, mode=0)` | isotropic excess noise, SNU |
| `.phase_noise(sigma, mode=0, mean_phase=0.0)` | exact-moment (non-Gaussian) map |

**Measurements** — each returns `(outcome, conditional_state)`

| member | description |
|---|---|
| `.homodyne(mode=0, angle=0.0, efficiency=1.0, electronic_noise=0.0, seed=None)` | |
| `.heterodyne(mode=0, seed=None)` | |
| `.generaldyne(modes, cov_povm, seed=None)` | |

**Scalar properties**

| member | description |
|---|---|
| `.symplectic_eigenvalues()` / `.symplectic_eigenvalues_snu()` | natural units / vacuum=1 |
| `.williamson()` | `(S, D)` with `cov = S @ D @ S.T` |
| `.purity()` | $\mathrm{Tr}(\rho^2)$ |
| `.is_pure(atol=1e-6)` / `.is_classical(atol=1e-8)` | |
| `.von_neumann_entropy(base=2)` / `.renyi_entropy(alpha=2, base=2)` | |
| `.mean_photon_number(mode=None)` / `.photon_number_variance(mode=0)` | |
| `.quad_mean(mode, angle=0.0)` / `.quad_variance(mode, angle=0.0)` | |
| `.squeezing_db(mode=0)` | negative = squeezed |

**Phase space**

| member | description |
|---|---|
| `.quasiprobability(x, p, s=0.0, mode=0)` | general $s$-ordered function |
| `.wigner(x, p, mode=0)` / `.husimi(x, p, mode=0)` / `.p_function(x, p, mode=0)` | $s=0,-1,+1$ |
| `.marginal(mode, angle=0.0, xvec=None)` | returns `(xvec, pdf)` |

**Interop**

| member | description |
|---|---|
| `.to_xxpp()` / `GaussianState.from_xxpp(mean, cov, hbar=2)` | xpxp ↔ xxpp |

---

## State constructors

`vacuum(n=1, hbar=2)`, `coherent(alpha, hbar=2)`, `squeezed(r, phi=0.0, hbar=2)`,
`squeezed_coherent(alpha, r, phi=0.0, hbar=2)`, `thermal(nbar, hbar=2)`,
`two_mode_squeezed(r, phi=0.0, hbar=2)`, `epr(r, hbar=2)`,
`graph_state(adjacency, r=10.0, hbar=2)` (alias `cluster_state`),
`random_state(n, pure=False, seed=None, hbar=2, max_squeezing=1.0, displacement_scale=1.0)`,
`tensor(*states)`.

---

## Circuit

`Circuit(num_modes, hbar=2, initial=None)` — mutable builder; every gate
method returns `self`. Same op set as `GaussianState`'s unitary/channel
methods above, plus `.symplectic(matrix, modes=None, d=None)`, and
measurements take an optional `label=`:
`.homodyne(...)`, `.heterodyne(...)`, `.generaldyne(...)`.

| member | description |
|---|---|
| `.state(seed=None)` | run once, return the final `GaussianState` |
| `.run(shots=1000, seed=None)` | run `shots` times, return a `Result` |
| `.to_symplectic()` | overall `(S, d)`; raises on channels/measurements |
| `.to_channel()` | overall `(X, Y, d)`; raises on measurements/phase noise |
| `.compose(other, modes=None)` | append another circuit's instructions |
| `.inverse()` | a circuit implementing the unitary part's inverse |
| `.copy()` | independent copy |
| `.depth()` / `.count_ops()` / `len(c)` | |
| `.draw_text()` / `str(c)` | Unicode text-art diagram |
| `.draw(ax=None, style=None)` | styled matplotlib diagram, returns `Axes` |
| `.save(path, dpi=200)` | SVG/PNG/PDF, format from suffix |

---

## Result

Outcomes of `Circuit.run()`.

| member | description |
|---|---|
| `result[label]` | array of that label's outcomes across shots |
| `.mean(label=None)` / `.var(label=None)` | one label, or a dict for all |
| `.cov(label_x, label_y)` | covariance between two scalar labels |
| `.histogram(label, bins=50)` | `(counts, bin_edges)` |
| `.labels` / `.shots` | |

---

## ops

Symplectic matrix factories (all hbar-independent except `displacement`):
`rotation(phi)`, `fourier()`, `squeezing(r, phi=0.0)`, `shear(s=1.0)`,
`displacement(alpha, hbar=2)`, `two_mode_squeezing(r, phi=0.0)`,
`beamsplitter(theta, phi=0.0)`, `beamsplitter_t(transmittance, phi=0.0)`,
`cz(s=1.0)`, `cx(s=1.0)`, `interferometer(unitary)`.

Utilities: `symplectic_form(n)`, `is_symplectic(matrix, atol=1e-8)`,
`expand(matrix, modes, n)`, `expand_vector(vector, modes, n)`,
`mode_indices(modes)`, `random_interferometer(n, seed=None)`,
`random_symplectic(n, seed=None, passive=False, max_squeezing=1.0)`.

---

## channels

`GaussianChannel(X, Y, d=None, hbar=2, atol=1e-8)` — `.compose(other)`,
`a @ b` (apply `b` then `a`).

Constructors: `loss(eta, nbar_env=0.0, hbar=2)`, `amplifier(g, nbar_env=0.0, hbar=2)`,
`excess_noise(xi, hbar=2)`, `additive_noise(xi_x, xi_p, hbar=2)`,
`classical_noise(cov_noise, hbar=2)`, `thermal_channel(nbar, hbar=2)`.

`phase_noise(mean, cov, hbar, sigma, mode, mean_phase=0.0)` — free function
on raw arrays (not a `GaussianChannel`); returns `(new_mean, new_cov)`.

---

## measure

Free functions on raw `(mean, cov, hbar)` arrays; return `(outcome, new_mean, new_cov)`
for the full n-mode system.

`generaldyne(mean, cov, hbar, modes, cov_povm, seed=None)`,
`heterodyne(mean, cov, hbar, mode, seed=None)`,
`homodyne(mean, cov, hbar, mode, angle=0.0, efficiency=1.0, electronic_noise=0.0, seed=None)`,
`outcome_distribution(mean, cov, modes, cov_povm)`,
`generaldyne_logpdf(outcome, mean_b, outcome_cov)`.

---

## decompositions

`symplectic_eigenvalues(cov)`, `symplectic_eigenvalues_snu(cov, hbar=2)`,
`williamson(cov, atol=1e-9)` → `(S, D)`,
`bloch_messiah(matrix, atol=1e-9)` → `(O1, Z, O2)`,
`clements_decomposition(unitary, atol=1e-9)` and
`interferometer_mesh(unitary, atol=1e-9)` → ordered list of
`("phase", mode, angle)` / `("bs", mode1, mode2, theta, phi)` elements
(Clements rectangular mesh, depth n, recommended; Reck triangular mesh,
depth 2n-3 — same return format, interchangeable).

---

## conversions

Ordering: `xpxp_to_xxpp(array)`, `xxpp_to_xpxp(array)` (vector or matrix).
Squeezing: `r_to_db(r)`, `db_to_r(db)`. Loss: `eta_to_db(eta)`,
`db_to_eta(db)`, `distance_to_eta(length_km, db_per_km=0.2)`. Amplitude:
`alpha_to_xp(alpha, hbar=2)`, `xp_to_alpha(vec, hbar=2)`.

---

## quantum_info

| function | description |
|---|---|
| `fidelity(state_a, state_b)` | squared overlap, arbitrary multimode Gaussian states |
| `bures_distance` / `bures_angle(state_a, state_b)` | |
| `trace_distance_bounds(state_a, state_b)` | Fuchs-van de Graaf `(lower, upper)` |
| `purity` / `von_neumann_entropy` / `renyi_entropy(state, ...)` | wrap the `GaussianState` methods |
| `mutual_information(state, modes_a, base=2)` | |
| `conditional_entropy(state, modes_x, modes_y, base=2)` | |
| `logarithmic_negativity` / `negativity(state, modes_a)` | |
| `is_separable(state, modes_a, atol=1e-8)` (alias `simon_criterion`) | PPT test |
| `duan_criterion(state, modes=(0,1))` / `epr_variance(state, modes=(0,1))` | |
| `steerability(state, modes_a, mode_b, base=e)` | Gaussian steering, one direction |
| `entanglement_of_formation(state, modes_a, base=2)` | **pure global states only** |
| `gaussian_discord(state, modes_a, mode_b, base=2)` | **pure global states only** |

---

## visualization

All take `(..., ax=None, ...)` and return the `Axes` (default `plt.gca()`),
except `plot_wigner_grid` which is figure-level and returns `(fig, axes)`.

`plot_quasiprobability(state, mode=0, s=0.0, ax=None, n=200, xrange=None, prange=None, cmap=None, colorbar=True, title=None)`,
`plot_wigner` / `plot_husimi` / `plot_p(state, mode=0, ax=None, **kwargs)`,
`plot_wigner_grid(state, n_cols=None, figsize=None, **kwargs)`,
`plot_marginal(state, mode, angle=0.0, ax=None, xvec=None, label=None, **plot_kwargs)`,
`plot_joint_marginal(state, quad_a, quad_b, ax=None, n=200, xrange=None, yrange=None, cmap="viridis", colorbar=True, n_sigma=5.0)`,
`plot_ellipse(states, mode=0, n_sigma=1.0, ax=None, labels=None, colors=None, fill=False)`,
`plot_covariance(state, ax=None, cmap=None, colorbar=True)`.

`CMWIG1` — the default diverging colormap for signed quasiprobabilities.
