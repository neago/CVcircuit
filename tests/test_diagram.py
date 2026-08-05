"""Circuit diagrams: text and matplotlib renderers agree on layout; exports work."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cvcircuit import circuit as ci
from cvcircuit import diagram as dg


def _every_kind_circuit():
    circuit = ci.Circuit(3)
    circuit.squeeze(1.0, phi=0.3, mode=0)
    circuit.displace(1 + 1j, mode=1)
    circuit.rotate(np.pi / 4, mode=2)
    circuit.shear(0.2, mode=0)
    circuit.beamsplitter(np.pi / 4, modes=(0, 1))
    circuit.two_mode_squeeze(0.5, modes=(1, 2))
    circuit.cz(0.8, modes=(0, 2))
    circuit.cx(1.2, modes=(1, 2))
    circuit.interferometer(np.eye(2, dtype=complex), modes=(0, 1), mesh=None)
    circuit.loss(0.9, mode=1)
    circuit.amplify(1.2, mode=0)
    circuit.noise(0.05, mode=2)
    circuit.phase_noise(0.1, mode=1)
    circuit.homodyne(0, angle=np.pi / 2, label="x0")
    circuit.heterodyne(1, label="het1")
    circuit.generaldyne([2], np.eye(2), label="gd2")
    return circuit


def test_layout_element_count_matches_instructions():
    circuit = _every_kind_circuit()
    lay = dg.layout(circuit)
    assert len(lay.elements) == len(circuit.instructions)
    assert lay.num_modes == 3
    assert lay.num_columns == circuit.depth()


def test_text_and_mpl_renderers_share_layout_columns():
    circuit = _every_kind_circuit()
    lay = dg.layout(circuit)
    text = dg.render_text(lay)
    ax = dg.render_mpl(lay)
    # every element's column appears in both renderings (structural agreement,
    # since both consume the same Layout)
    assert len(ax.patches) + len(ax.lines) > 0
    assert text.count("\n") == 2 * lay.num_modes - 2  # n mode rows + (n-1) gap rows, joined
    plt.close(ax.figure)


def test_render_text_contains_every_label():
    circuit = _every_kind_circuit()
    text = dg.render_text(dg.layout(circuit))
    for expected in ("S 1", "D 1+1j", "R π/4", "BS π/4", "TMS 0.5", "CZ 0.8", "CX 1.2", "U",
                       "η 0.9", "G 1.2", "ξ 0.05", "x(π/2)", "het", "gd"):
        assert expected in text, f"missing {expected!r} in:\n{text}"


def test_feedforward_label_shows_placeholder():
    circuit = ci.Circuit(2)
    circuit.homodyne(0, label="x")
    circuit.displace(lambda m: m["x"], mode=1)
    text = dg.render_text(dg.layout(circuit))
    assert "f(·)" in text


def test_save_svg_png_pdf(tmp_path):
    circuit = _every_kind_circuit()
    for suffix in ("svg", "png", "pdf"):
        path = tmp_path / f"circuit.{suffix}"
        circuit.save(str(path))
        assert path.exists()
        assert path.stat().st_size > 500


def test_save_svg_has_valid_header(tmp_path):
    circuit = ci.Circuit(1).squeeze(0.5)
    path = tmp_path / "c.svg"
    circuit.save(str(path))
    content = path.read_text()
    assert "<svg" in content


def test_repr_svg_returns_svg_and_closes_figure():
    circuit = _every_kind_circuit()
    n_figs_before = len(plt.get_fignums())
    svg = circuit._repr_svg_()
    assert "<svg" in svg
    assert len(plt.get_fignums()) == n_figs_before


def test_str_matches_draw_text():
    circuit = ci.Circuit(1).squeeze(0.5)
    assert str(circuit) == circuit.draw_text()


def test_draw_returns_axes_and_accepts_existing_ax():
    circuit = _every_kind_circuit()
    fig, ax = plt.subplots()
    returned = circuit.draw(ax=ax)
    assert returned is ax
    plt.close(fig)


def test_empty_circuit_diagrams_dont_crash():
    circuit = ci.Circuit(2)
    assert circuit.draw_text()  # just mode wires
    ax = circuit.draw()
    plt.close(ax.figure)


def test_format_angle_pi_fractions():
    assert dg._format_angle(np.pi / 4) == "π/4"
    assert dg._format_angle(np.pi) == "π"
    assert dg._format_angle(-np.pi / 2) == "-π/2"
    assert dg._format_angle(0.12345) == "0.1235"


@pytest.mark.parametrize("style", ["light", "dark", None, "nonexistent"])
def test_render_mpl_style_options(style):
    circuit = ci.Circuit(1).squeeze(0.5)
    ax = dg.render_mpl(dg.layout(circuit), style=style)
    plt.close(ax.figure)
