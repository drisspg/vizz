"""Tensor cells with independent numeric content, semantic state, and overlays."""

from __future__ import annotations

from enum import Enum
from math import isfinite

import numpy as np
from manim import Line, Rectangle, Text, VGroup

from vizz.presentations.theme import Theme


class CellState(Enum):
    RETAINED = "retained"
    INACTIVE = "inactive"
    MASKED = "masked"


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class TensorGrid(VGroup):
    """A centered grid indexed from the top left, initially retained.

    Each stable cell contains ``[frame, content, hatches]``; the latter two are
    stable VGroups. Masking hides, but does not delete, its value. Inactive
    values are muted; retained values are fully visible. Selection rectangles
    and tile lines are detached snapshots, not state changes or live updaters.
    Geometry follows affine transforms of the grid's cell frames.
    """

    def __init__(
        self, rows: int, cols: int, theme: Theme, cell_size: float = 0.5
    ) -> None:
        _positive_int(rows, "rows")
        _positive_int(cols, "cols")
        if isinstance(cell_size, bool) or not isinstance(cell_size, (int, float)):
            raise TypeError("cell_size must be a positive finite number")
        if not isfinite(cell_size) or cell_size <= 0:
            raise ValueError("cell_size must be a positive finite number")
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.theme = theme
        self._states = [CellState.RETAINED] * (rows * cols)
        for row in range(rows):
            for col in range(cols):
                frame = Rectangle(
                    width=cell_size,
                    height=cell_size,
                    stroke_color=theme.panel_stroke,
                    stroke_width=1,
                    fill_color=theme.accent_success,
                    fill_opacity=0.16,
                ).move_to(
                    [
                        (col - (cols - 1) / 2) * cell_size,
                        ((rows - 1) / 2 - row) * cell_size,
                        0,
                    ]
                )
                self.add(VGroup(frame, VGroup(), VGroup()))

    def cell(self, row: int, col: int) -> VGroup:
        for name, index, size in (("row", row, self.rows), ("col", col, self.cols)):
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= index < size:
                raise IndexError(f"{name} index {index} outside [0, {size})")
        return self[row * self.cols + col]

    def set_state(self, row: int, col: int, state: CellState) -> TensorGrid:
        frame, content, hatches = self.cell(row, col)
        if not isinstance(state, CellState):
            raise TypeError("state must be a CellState")
        self._states[row * self.cols + col] = state
        hatches.remove(*hatches.submobjects)
        frame.set_stroke(self.theme.panel_stroke, opacity=1)
        match state:
            case CellState.RETAINED:
                frame.set_fill(self.theme.accent_success, opacity=0.16)
                content.set_color(self.theme.text).set_opacity(1)
            case CellState.INACTIVE:
                frame.set_fill(self.theme.panel_fill, opacity=1)
                frame.set_stroke(opacity=0.4)
                content.set_color(self.theme.muted_text).set_opacity(0.45)
            case CellState.MASKED:
                frame.set_fill(self.theme.panel_fill, opacity=1)
                content.set_opacity(0)
                ur, ul, dl, _ = frame.get_vertices()
                right, up = ur - ul, ul - dl
                for offset in (-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75):
                    # Clip in cell-local coordinates, inset to keep stroke caps inside.
                    x0, y0 = max(0, -offset), max(0, offset)
                    x1, y1 = min(1, 1 - offset), min(1, 1 + offset)
                    hatches.add(
                        Line(
                            dl + (0.06 + 0.88 * x0) * right + (0.06 + 0.88 * y0) * up,
                            dl + (0.06 + 0.88 * x1) * right + (0.06 + 0.88 * y1) * up,
                            color=self.theme.muted_text,
                            stroke_width=1,
                            stroke_opacity=0.55,
                        )
                    )
        return self

    def set_value(self, row: int, col: int, text: str) -> TensorGrid:
        """Replace the literal label; empty/whitespace text clears it, never masks."""
        frame, content, _ = self.cell(row, col)
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        content.remove(*content.submobjects)
        if text.strip():
            label = Text(text, font=self.theme.mono_font, color=self.theme.text)
            if label.width > 0 and label.height > 0:
                label.scale(min(0.65 / label.width, 0.55 / label.height))
            ur, ul, dl, _ = frame.get_vertices()
            label.apply_matrix(np.column_stack((ur - ul, ul - dl, [0, 0, 1])))
            label.move_to(frame.get_center())
            content.add(label)
        state = self._states[row * self.cols + col]
        content.set_color(
            self.theme.muted_text if state is CellState.INACTIVE else self.theme.text
        ).set_opacity(
            0
            if state is CellState.MASKED
            else 0.45
            if state is CellState.INACTIVE
            else 1
        )
        return self

    def causal_mask(self) -> TensorGrid:
        """Mask columns strictly above the diagonal, retaining every other cell."""
        for row in range(self.rows):
            for col in range(self.cols):
                self.set_state(
                    row, col, CellState.MASKED if col > row else CellState.RETAINED
                )
        return self

    def region(self, r0: int, r1: int, c0: int, c1: int, *, color: str) -> Rectangle:
        """Outline nonempty half-open bounds without modifying the selected cells."""
        for name, bound, size in (
            ("r0", r0, self.rows),
            ("r1", r1, self.rows),
            ("c0", c0, self.cols),
            ("c1", c1, self.cols),
        ):
            if isinstance(bound, bool) or not isinstance(bound, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= bound <= size:
                raise IndexError(f"{name} bound {bound} outside [0, {size}]")
        if r0 >= r1 or c0 >= c1:
            raise ValueError("region must have nonempty, increasing bounds")
        ur = self.cell(r0, c1 - 1)[0].get_vertices()[0]
        ul = self.cell(r0, c0)[0].get_vertices()[1]
        dl = self.cell(r1 - 1, c0)[0].get_vertices()[2]
        dr = self.cell(r1 - 1, c1 - 1)[0].get_vertices()[3]
        return Rectangle(
            color=color, fill_opacity=0, stroke_width=3
        ).set_points_as_corners([ur, ul, dl, dr, ur])

    def tile_lines(self, row_step: int, col_step: int) -> VGroup:
        """Return internal tile boundaries; partial edge tiles need no extra line."""
        _positive_int(row_step, "row_step")
        _positive_int(col_step, "col_step")
        lines = VGroup()
        for row in range(row_step, self.rows, row_step):
            lines.add(
                Line(
                    self.cell(row, 0)[0].get_vertices()[1],
                    self.cell(row, self.cols - 1)[0].get_vertices()[0],
                    color=self.theme.muted_text,
                    stroke_width=2,
                )
            )
        for col in range(col_step, self.cols, col_step):
            lines.add(
                Line(
                    self.cell(0, col)[0].get_vertices()[1],
                    self.cell(self.rows - 1, col)[0].get_vertices()[2],
                    color=self.theme.muted_text,
                    stroke_width=2,
                )
            )
        return lines
