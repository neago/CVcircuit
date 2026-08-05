"""cvcircuit -- a small, readable Gaussian CV quantum optics simulator built on NumPy.

>>> import numpy as np
>>> from cvcircuit import squeezed, vacuum
>>> s = squeezed(1.0) & vacuum()
>>> s = s.beamsplitter(np.pi / 4, modes=(0, 1))
>>> s.symplectic_eigenvalues_snu()                       # doctest: +SKIP
array([2.92044936, 0.34241831])

Two ways to build a state: chain :class:`GaussianState` methods directly
for quick interactive work, or build a :class:`Circuit` for a reproducible,
drawable, sampleable program. See ``docs/guide.md`` and ``docs/conventions.md``
(hbar, quadrature ordering, every symplectic matrix and channel) for the
full picture.
"""

from . import channels, conversions, decompositions, diagram, measure, ops, quantum_info, visualization
from .channels import (
    GaussianChannel, additive_noise, amplifier, classical_noise, excess_noise, loss,
    phase_noise, thermal_channel,
)
from .circuit import Circuit, Instruction, Result
from .quantum_info import (
    bures_angle, bures_distance, conditional_entropy, duan_criterion,
    entanglement_of_formation, epr_variance, fidelity, gaussian_discord, is_separable,
    logarithmic_negativity, mutual_information, negativity, purity, renyi_entropy,
    simon_criterion, steerability, trace_distance_bounds, von_neumann_entropy,
)
from .state import (
    GaussianState, cluster_state, coherent, epr, graph_state, random_state, squeezed,
    squeezed_coherent, tensor, thermal, two_mode_squeezed, vacuum,
)
from .visualization import (
    plot_covariance, plot_ellipse, plot_husimi, plot_joint_marginal, plot_marginal,
    plot_p, plot_quasiprobability, plot_wigner, plot_wigner_grid,
)

__version__ = "0.1.0"

__all__ = [
    "GaussianState", "Circuit", "Instruction", "Result", "GaussianChannel",
    "channels", "conversions", "decompositions", "diagram", "measure", "ops", "quantum_info", "visualization",
    "vacuum", "coherent", "squeezed", "squeezed_coherent", "thermal",
    "two_mode_squeezed", "epr", "graph_state", "cluster_state", "random_state", "tensor",
    "loss", "amplifier", "excess_noise", "additive_noise", "classical_noise",
    "thermal_channel", "phase_noise",
    "fidelity", "bures_distance", "bures_angle", "trace_distance_bounds",
    "purity", "von_neumann_entropy", "renyi_entropy",
    "mutual_information", "conditional_entropy",
    "logarithmic_negativity", "negativity", "is_separable", "simon_criterion",
    "duan_criterion", "epr_variance", "steerability",
    "entanglement_of_formation", "gaussian_discord",
    "plot_wigner", "plot_husimi", "plot_p", "plot_quasiprobability", "plot_wigner_grid",
    "plot_marginal", "plot_joint_marginal", "plot_ellipse", "plot_covariance",
]
