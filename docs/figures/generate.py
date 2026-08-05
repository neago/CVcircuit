"""Regenerate every SVG figure embedded in the README and docs/guide.md.

Run with ``uv run python docs/figures/generate.py`` from the repo root.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cvcircuit import Circuit, squeezed, thermal, two_mode_squeezed, vacuum
from cvcircuit import visualization as viz

HERE = Path(__file__).parent


def circuit_teleportation():
    c = Circuit(3)
    c.displace(1 + 0.5j, mode=0)
    c.two_mode_squeeze(2.0, modes=(1, 2))
    c.beamsplitter(np.pi / 4, modes=(0, 1))
    c.homodyne(0, angle=0.0, label="xA")
    c.homodyne(1, angle=np.pi / 2, label="pB")
    c.displace(lambda m: (m["xA"] + 1j * m["pB"]) / np.sqrt(2), mode=2)
    c.save(HERE / "circuit_teleportation.svg")


def wigner_gallery():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    viz.plot_wigner(squeezed(0.8), ax=axes[0], title="squeezed vacuum")
    viz.plot_wigner(vacuum(1).displace(1.2 + 0.6j), ax=axes[1], title="coherent state")
    viz.plot_wigner(thermal(1.2), ax=axes[2], title="thermal state")
    fig.tight_layout()
    fig.savefig(HERE / "wigner_gallery.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def epr_correlation():
    tmsv = two_mode_squeezed(1.2)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    viz.plot_joint_marginal(tmsv, (0, 0.0), (1, 0.0), ax=ax)
    ax.set_title("EPR correlation: mode 0 vs mode 1 (x quadrature)")
    fig.tight_layout()
    fig.savefig(HERE / "epr_correlation.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    circuit_teleportation()
    wigner_gallery()
    epr_correlation()
    print(f"wrote figures to {HERE}")
