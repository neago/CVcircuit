"""Circuit diagrams: one shared layout, a Unicode text renderer, and a
styled matplotlib renderer (SVG/PNG/PDF via ``fig.savefig``).

:func:`layout` assigns every instruction a column via the circuit's own
per-mode frontier allocator (:meth:`cvcircuit.circuit.Circuit._column_layout`),
so the text and vector pictures can never disagree with each other or with
:meth:`~cvcircuit.circuit.Circuit.depth`.  Access all of this through
:class:`~cvcircuit.circuit.Circuit`'s own ``draw_text()``, ``draw()``,
``save()`` and ``_repr_svg_`` -- this module is not usually imported
directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

CATEGORY_COLOR = {
    "unitary": "#5c7cfa",     # indigo: D, R, S, F, shear, symplectic, interferometer
    "two_mode": "#0ca678",    # teal: BS, TMS, CZ, CX
    "channel": "#f08c00",     # amber: loss, amplify, noise, phase noise
    "measurement": "#e03131", # rose: homodyne, heterodyne, generaldyne
}

STYLE = {
    "light": {"wire": "#adb5bd", "text": "#1a1a1a", "box_face": "#ffffff"},
    "dark": {"wire": "#5c6370", "text": "#e6e6e6", "box_face": "#2b2b2b"},
}

_CATEGORY_OF_KIND = {
    "displace": "unitary", "rotate": "unitary", "squeeze": "unitary",
    "shear": "unitary", "symplectic": "unitary", "interferometer": "unitary",
    "beamsplitter": "two_mode", "two_mode_squeeze": "two_mode", "cz": "two_mode", "cx": "two_mode",
    "loss": "channel", "amplify": "channel", "noise": "channel", "phase_noise": "channel",
    "homodyne": "measurement", "heterodyne": "measurement", "generaldyne": "measurement",
}


# ---------------------------------------------------------------------------
# label formatting
# ---------------------------------------------------------------------------


def _format_angle(value):
    """Format an angle, using multiples of pi when they are (nearly) exact."""
    if value == 0:
        return "0"
    fraction = Fraction(value / math.pi).limit_denominator(12)
    if abs(float(fraction) * math.pi - value) < 1e-9 and fraction != 0:
        num, den = fraction.numerator, fraction.denominator
        text = "π" if abs(num) == 1 else f"{abs(num)}π"
        if num < 0:
            text = "-" + text
        return text if den == 1 else f"{text}/{den}"
    return f"{value:.4g}"


def _fmt(value):
    if callable(value):
        return "f(·)"
    if isinstance(value, complex):
        return f"{value.real:.3g}{value.imag:+.3g}j"
    if isinstance(value, (int, float, np.floating, np.integer)):
        return f"{float(value):.4g}"
    return str(value)


def _fmt_angle(value):
    if callable(value):
        return "f(·)"
    return _format_angle(float(value))


def _label(instr):
    p = instr.params
    kind = instr.kind
    if kind == "displace":
        return f"D {_fmt(p['alpha'])}"
    if kind == "rotate":
        return f"R {_fmt_angle(p['phi'])}"
    if kind == "squeeze":
        phi = p.get("phi", 0.0)
        extra = "" if (not callable(phi) and phi == 0) else f",{_fmt_angle(phi)}"
        return f"S {_fmt(p['r'])}{extra}"
    if kind == "shear":
        return f"P {_fmt(p['s'])}"
    if kind in ("symplectic", "interferometer"):
        return "U"
    if kind == "beamsplitter":
        return f"BS {_fmt_angle(p['theta'])}"
    if kind == "two_mode_squeeze":
        return f"TMS {_fmt(p['r'])}"
    if kind == "cz":
        return f"CZ {_fmt(p['s'])}"
    if kind == "cx":
        return f"CX {_fmt(p['s'])}"
    if kind == "loss":
        return f"η {_fmt(p['eta'])}"
    if kind == "amplify":
        return f"G {_fmt(p['g'])}"
    if kind == "noise":
        return f"ξ {_fmt(p['xi'])}"
    if kind == "phase_noise":
        return f"φ±{_fmt(p['sigma'])}"
    if kind == "homodyne":
        return f"x({_fmt_angle(p['angle'])})"
    if kind == "heterodyne":
        return "het"
    if kind == "generaldyne":
        return "gd"
    return kind  # pragma: no cover


# ---------------------------------------------------------------------------
# shared layout
# ---------------------------------------------------------------------------


@dataclass
class Layout:
    num_modes: int
    num_columns: int
    elements: list = field(default_factory=list)


def layout(circuit):
    """Assign every instruction a column and a render-ready label/category."""
    columns, frontier = circuit._column_layout()
    elements = [
        {
            "kind": instr.kind,
            "modes": instr.modes,
            "column": col,
            "label": _label(instr),
            "category": _CATEGORY_OF_KIND.get(instr.kind, "unitary"),
        }
        for instr, col in zip(circuit.instructions, columns)
    ]
    num_columns = max(frontier) if frontier else 0
    return Layout(circuit.num_modes, num_columns, elements)


# ---------------------------------------------------------------------------
# text renderer
# ---------------------------------------------------------------------------


def render_text(lay):
    """Unicode text-art rendering of a :class:`Layout`."""
    n = lay.num_modes
    num_cols = lay.num_columns

    prefixes = [f"mode {m}: " for m in range(n)]
    pad_width = max((len(p) for p in prefixes), default=6)
    prefixes = [p.ljust(pad_width) for p in prefixes]

    col_width = {}
    for el in lay.elements:
        col_width[el["column"]] = max(col_width.get(el["column"], 4), len(el["label"]) + 2)
    widths = [col_width.get(c, 4) for c in range(num_cols)]

    mode_rows = [list(prefixes[m]) for m in range(n)]
    gap_rows = [list(" " * pad_width) for _ in range(n - 1)]

    for col in range(num_cols):
        w = widths[col]
        mode_cells = ["─" * w for _ in range(n)]
        gap_cells = [" " * w for _ in range(n - 1)]
        for el in (e for e in lay.elements if e["column"] == col):
            modes_sorted = sorted(el["modes"])
            token = f"[{el['label']}]".center(w, "─")
            for m in modes_sorted:
                mode_cells[m] = token
            for m in range(modes_sorted[0], modes_sorted[-1]):
                gap_cells[m] = "│".center(w)
        for m in range(n):
            mode_rows[m].extend(mode_cells[m])
        for m in range(n - 1):
            gap_rows[m].extend(gap_cells[m])

    lines = []
    for m in range(n):
        lines.append("".join(mode_rows[m]))
        if m < n - 1:
            lines.append("".join(gap_rows[m]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# matplotlib renderer
# ---------------------------------------------------------------------------


def _draw_box(ax, x, y, label, color, theme):
    from matplotlib.patches import FancyBboxPatch

    width = 0.15 * max(len(label), 2) + 0.24
    height = 0.36
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor=color, facecolor=theme["box_face"], zorder=5,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color=color,
             family="monospace", zorder=6)


def _draw_measurement(ax, x, y, label, color, theme):
    from matplotlib.patches import Wedge

    ax.add_patch(Wedge((x, y), 0.17, 0, 180, facecolor=color, edgecolor=color, zorder=5))
    ax.text(x, y - 0.32, label, ha="center", va="top", fontsize=8, color=color,
             family="monospace", zorder=6)


def render_mpl(lay, ax=None, style=None):
    """Styled matplotlib rendering of a :class:`Layout`; returns the ``Axes``.

    ``fig.savefig(path)`` on the returned Axes' figure writes clean vector
    SVG/PDF or rasterised PNG depending on ``path``'s suffix.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    theme = STYLE.get(style, STYLE["light"]) if style in STYLE else STYLE["light"]
    n = lay.num_modes
    num_cols = max(lay.num_columns, 1)

    if ax is None:
        fig_w = max(4.0, 1.05 * num_cols + 2.0)
        fig_h = max(1.4, 0.85 * n + 0.5)
        _, ax = plt.subplots(figsize=(fig_w, fig_h))

    def y_of(mode):
        return n - 1 - mode

    x_start, x_end = -0.6, (num_cols - 1) + 0.6

    for m in range(n):
        y = y_of(m)
        ax.add_line(Line2D([x_start, x_end], [y, y], color=theme["wire"], linewidth=1.2, zorder=1))
        ax.text(x_start - 0.15, y, f"mode {m}", ha="right", va="center", fontsize=9,
                 color=theme["text"], family="monospace")

    for el in lay.elements:
        x = el["column"]
        modes_sorted = sorted(el["modes"])
        color = CATEGORY_COLOR.get(el["category"], CATEGORY_COLOR["unitary"])

        if len(modes_sorted) >= 2:
            y1, y2 = y_of(modes_sorted[0]), y_of(modes_sorted[-1])
            ax.add_line(Line2D([x, x], [min(y1, y2), max(y1, y2)], color=color, linewidth=1.6, zorder=2))
            if el["kind"] == "cz":
                for yy in (y1, y2):
                    ax.plot([x], [yy], marker="o", markersize=6, color=color, zorder=4)
                continue
            if el["kind"] == "cx":
                ax.plot([x], [y1], marker="o", markersize=6, color=color, zorder=4)
                ax.plot([x], [y2], marker="o", markersize=15, markerfacecolor=theme["box_face"],
                         markeredgecolor=color, markeredgewidth=1.6, zorder=4)
                ax.plot([x, x], [y2 - 0.11, y2 + 0.11], color=color, linewidth=1.4, zorder=5)
                ax.plot([x - 0.11, x + 0.11], [y2, y2], color=color, linewidth=1.4, zorder=5)
                continue
            for yy in (y1, y2):
                ax.plot([x], [yy], marker="o", markersize=4, color=color, zorder=3)
            _draw_box(ax, x, (y1 + y2) / 2, el["label"], color, theme)
        else:
            y = y_of(modes_sorted[0])
            if el["category"] == "measurement":
                _draw_measurement(ax, x, y, el["label"], color, theme)
            elif el["kind"] == "loss":
                _draw_box(ax, x, y, el["label"], color, theme)
                ax.annotate(
                    "", xy=(x + 0.26, y - 0.36), xytext=(x + 0.05, y - 0.08),
                    arrowprops=dict(arrowstyle="->", color=color, linewidth=1.1), zorder=4,
                )
            else:
                _draw_box(ax, x, y, el["label"], color, theme)

    ax.set_xlim(x_start - 1.0, x_end + 0.3)
    ax.set_ylim(-0.8, (n - 1) + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def save(circuit, path, dpi=200):
    """Save a styled circuit diagram to ``path`` (format inferred from the suffix)."""
    import matplotlib.pyplot as plt

    ax = render_mpl(layout(circuit))
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(ax.figure)


def repr_svg(circuit):
    """In-memory SVG rendering, for Jupyter's ``_repr_svg_`` protocol."""
    import io

    import matplotlib.pyplot as plt

    ax = render_mpl(layout(circuit))
    buf = io.StringIO()
    ax.figure.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(ax.figure)
    return buf.getvalue()
