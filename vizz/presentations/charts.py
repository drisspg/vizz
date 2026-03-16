from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from manim import *

from vizz.presentations.theme import Theme


@dataclass
class Series:
    name: str
    values: list[float]
    color: str


def _mix(color_a: str, color_b: str, alpha: float) -> ManimColor:
    return interpolate_color(ManimColor(color_a), ManimColor(color_b), alpha)


def _chart_axis_color(theme: Theme) -> ManimColor:
    return _mix(theme.divider, theme.text, 0.4)


def _chart_grid_color(theme: Theme) -> ManimColor:
    return _mix(theme.divider, theme.text, 0.22)


def _chart_label_color(theme: Theme) -> ManimColor:
    return _mix(theme.muted_text, theme.text, 0.35)


def _series_stroke(color: str) -> ManimColor:
    return _mix(color, "#111111", 0.32)


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


_SHAPE_RE = re.compile(r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)")


def parse_shape(shape_str: str) -> tuple[int, int, int, int, int, int]:
    m = _SHAPE_RE.search(shape_str)
    if not m:
        raise ValueError(f"Cannot parse shape: {shape_str!r}")
    return tuple(int(g) for g in m.groups())


def seq_label(n: int) -> str:
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


def grouped_bar_chart(
    theme: Theme,
    series_list: list[Series],
    x_labels: list[str],
    *,
    y_label: str = "TFlops",
    x_length: float = 5.5,
    y_length: float = 3.5,
    y_max: float | None = None,
    bar_width: float = 0.18,
    bar_gap: float = 0.04,
    show_values: bool = False,
    value_font_size: int = 11,
    show_legend: bool = True,
    legend_font_size: int = 13,
) -> VGroup:
    n_series = len(series_list)
    n_points = len(x_labels)

    all_vals = [v for s in series_list for v in s.values]
    computed_y_max = y_max or max(all_vals) * 1.15

    y_step = _nice_step(computed_y_max, target_ticks=5)

    axes = Axes(
        x_range=[0, n_points, 1],
        y_range=[0, computed_y_max, y_step],
        x_length=x_length,
        y_length=y_length,
        tips=False,
        axis_config={
            "color": _chart_axis_color(theme),
            "stroke_width": 1.5,
            "include_ticks": True,
            "tick_size": 0.06,
        },
        y_axis_config={
            "include_numbers": True,
            "font_size": 14,
            "decimal_number_config": {
                "num_decimal_places": 0,
                "color": _chart_label_color(theme),
            },
        },
        x_axis_config={
            "include_numbers": False,
        },
    )

    grid_lines = VGroup()
    for tick_val in _tick_range(0, computed_y_max, y_step):
        if tick_val == 0:
            continue
        left = axes.c2p(0, tick_val)
        right = axes.c2p(n_points, tick_val)
        grid_lines.add(
            DashedLine(
                left,
                right,
                color=_chart_grid_color(theme),
                stroke_width=0.95,
                dash_length=0.04,
                dashed_ratio=0.4,
            )
        )

    x_axis_labels = VGroup()
    for i, label_text in enumerate(x_labels):
        pos = axes.c2p(i + 0.5, 0) + DOWN * 0.18
        lbl = Text(
            label_text,
            font=theme.mono_font,
            font_size=12,
            color=_chart_label_color(theme),
        )
        lbl.move_to(pos)
        x_axis_labels.add(lbl)

    y_axis_label = Text(
        y_label, font=theme.mono_font, font_size=12, color=_chart_label_color(theme)
    )
    y_axis_label.next_to(axes.y_axis, UP, buff=0.12)
    y_axis_label.align_to(axes.y_axis, LEFT)

    group_width = n_series * bar_width + (n_series - 1) * bar_gap
    bars = VGroup()
    value_labels = VGroup()

    for si, series in enumerate(series_list):
        for xi, val in enumerate(series.values):
            cx = xi + 0.5
            offset = -group_width / 2 + si * (bar_width + bar_gap) + bar_width / 2
            bar_center_x = cx + offset

            bottom = axes.c2p(bar_center_x, 0)
            top = axes.c2p(bar_center_x, val)
            bar_height = top[1] - bottom[1]

            bar = Rectangle(
                width=axes.x_length / n_points * bar_width,
                height=max(bar_height, 0.01),
                fill_color=series.color,
                fill_opacity=0.94,
                stroke_color=_series_stroke(series.color),
                stroke_width=0.7,
            )
            bar.move_to(bottom, aligned_edge=DOWN)
            bars.add(bar)

            if show_values:
                vl = Text(
                    f"{val:.0f}",
                    font=theme.mono_font,
                    font_size=value_font_size,
                    color=_chart_label_color(theme),
                )
                vl.next_to(bar, UP, buff=0.06)
                value_labels.add(vl)

    result = VGroup(grid_lines, axes, x_axis_labels, y_axis_label, bars)
    if show_values:
        result.add(value_labels)

    if show_legend:
        result.add(_build_legend(theme, series_list, legend_font_size))
        result[-1].next_to(axes, UP, buff=0.08).align_to(axes, RIGHT)

    return result


def cumulative_line_chart(
    theme: Theme,
    x_labels: list[str],
    values: list[float],
    *,
    line_color: str | None = None,
    y_label: str = "Repos",
    x_length: float = 5.5,
    y_length: float = 3.5,
    y_max: float | None = None,
    show_dots: bool = True,
    show_values: bool = True,
    value_font_size: int = 11,
    dot_radius: float = 0.045,
) -> VGroup:
    n_points = len(values)
    color = line_color or theme.accent_primary
    computed_y_max = y_max or max(values) * 1.15
    y_step = _nice_step(computed_y_max, target_ticks=5)

    axes = Axes(
        x_range=[0, n_points, 1],
        y_range=[0, computed_y_max, y_step],
        x_length=x_length,
        y_length=y_length,
        tips=False,
        axis_config={
            "color": _chart_axis_color(theme),
            "stroke_width": 1.5,
            "include_ticks": True,
            "tick_size": 0.06,
        },
        y_axis_config={
            "include_numbers": True,
            "font_size": 14,
            "decimal_number_config": {
                "num_decimal_places": 0,
                "color": _chart_label_color(theme),
            },
        },
        x_axis_config={"include_numbers": False},
    )

    grid_lines = VGroup()
    for tick_val in _tick_range(0, computed_y_max, y_step):
        if tick_val == 0:
            continue
        left = axes.c2p(0, tick_val)
        right = axes.c2p(n_points, tick_val)
        grid_lines.add(
            DashedLine(
                left,
                right,
                color=_chart_grid_color(theme),
                stroke_width=0.95,
                dash_length=0.04,
                dashed_ratio=0.4,
            )
        )

    x_axis_labels = VGroup()
    for i, label_text in enumerate(x_labels):
        pos = axes.c2p(i + 0.5, 0) + DOWN * 0.18
        lbl = Text(
            label_text,
            font=theme.mono_font,
            font_size=10,
            color=_chart_label_color(theme),
        )
        lbl.move_to(pos)
        x_axis_labels.add(lbl)

    y_axis_label = Text(
        y_label, font=theme.mono_font, font_size=12, color=_chart_label_color(theme)
    )
    y_axis_label.next_to(axes.y_axis, UP, buff=0.12)
    y_axis_label.align_to(axes.y_axis, LEFT)

    points = [axes.c2p(i + 0.5, v) for i, v in enumerate(values)]

    line_halo = VMobject(color=theme.panel_fill, stroke_width=5.8, stroke_opacity=0.98)
    line_halo.set_points_smoothly(points)

    line = VMobject(color=color, stroke_width=3.6)
    line.set_points_smoothly(points)

    fill = line.copy()
    left_bottom = axes.c2p(0.5, 0)
    right_bottom = axes.c2p(n_points - 0.5, 0)
    fill.add_line_to(right_bottom)
    fill.add_line_to(left_bottom)
    fill.close_path()
    fill.set_fill(color=color, opacity=0.12)
    fill.set_stroke(width=0)

    dots = VGroup()
    value_labels = VGroup()
    for i, (pt, val) in enumerate(zip(points, values)):
        if show_dots:
            dot = Dot(pt, radius=dot_radius, color=color)
            dot_inner = Dot(pt, radius=dot_radius * 0.5, color=theme.panel_fill)
            dots.add(dot, dot_inner)
        if show_values and (i % 3 == 0 or i == len(values) - 1):
            vl = Text(
                f"{int(val):,}",
                font=theme.mono_font,
                font_size=value_font_size,
                color=theme.text,
            )
            vl.next_to(pt, UP, buff=0.12)
            value_labels.add(vl)

    return VGroup(
        grid_lines,
        axes,
        x_axis_labels,
        y_axis_label,
        fill,
        line_halo,
        line,
        dots,
        value_labels,
    )


def bar_chart(
    theme: Theme,
    x_labels: list[str],
    values: list[float],
    *,
    bar_color: str | None = None,
    y_label: str = "",
    x_length: float = 5.5,
    y_length: float = 3.5,
    y_max: float | None = None,
    bar_width: float = 0.55,
    show_values: bool = True,
    value_font_size: int = 11,
) -> VGroup:
    n = len(values)
    color = bar_color or theme.accent_primary
    computed_y_max = y_max or max(values) * 1.15
    y_step = _nice_step(computed_y_max, target_ticks=5)

    axes = Axes(
        x_range=[0, n, 1],
        y_range=[0, computed_y_max, y_step],
        x_length=x_length,
        y_length=y_length,
        tips=False,
        axis_config={
            "color": _chart_axis_color(theme),
            "stroke_width": 1.5,
            "include_ticks": True,
            "tick_size": 0.06,
        },
        y_axis_config={
            "include_numbers": True,
            "font_size": 14,
            "decimal_number_config": {
                "num_decimal_places": 0,
                "color": _chart_label_color(theme),
            },
        },
        x_axis_config={"include_numbers": False},
    )

    grid_lines = VGroup()
    for tick_val in _tick_range(0, computed_y_max, y_step):
        if tick_val == 0:
            continue
        left = axes.c2p(0, tick_val)
        right = axes.c2p(n, tick_val)
        grid_lines.add(
            DashedLine(
                left,
                right,
                color=_chart_grid_color(theme),
                stroke_width=0.95,
                dash_length=0.04,
                dashed_ratio=0.4,
            )
        )

    x_axis_labels = VGroup()
    for i, label_text in enumerate(x_labels):
        pos = axes.c2p(i + 0.5, 0) + DOWN * 0.18
        lbl = Text(
            label_text,
            font=theme.mono_font,
            font_size=10,
            color=_chart_label_color(theme),
        )
        lbl.move_to(pos)
        x_axis_labels.add(lbl)

    y_axis_label_mob = VGroup()
    if y_label:
        yl = Text(
            y_label, font=theme.mono_font, font_size=12, color=_chart_label_color(theme)
        )
        yl.next_to(axes.y_axis, UP, buff=0.12).align_to(axes.y_axis, LEFT)
        y_axis_label_mob.add(yl)

    bars = VGroup()
    value_labels = VGroup()
    for i, val in enumerate(values):
        bottom = axes.c2p(i + 0.5, 0)
        top = axes.c2p(i + 0.5, val)
        bar = Rectangle(
            width=axes.x_length / n * bar_width,
            height=max(top[1] - bottom[1], 0.01),
            fill_color=color,
            fill_opacity=0.94,
            stroke_color=_series_stroke(color),
            stroke_width=0.7,
        )
        bar.move_to(bottom, aligned_edge=DOWN)
        bars.add(bar)

        if show_values and val > 0:
            vl = Text(
                str(int(val)),
                font=theme.mono_font,
                font_size=value_font_size,
                color=_chart_label_color(theme),
            )
            vl.next_to(bar, UP, buff=0.06)
            value_labels.add(vl)

    return VGroup(grid_lines, axes, x_axis_labels, y_axis_label_mob, bars, value_labels)


def _build_legend(theme: Theme, series_list: list[Series], font_size: int) -> VGroup:
    entries = VGroup()
    for s in series_list:
        swatch = Rectangle(
            width=0.18,
            height=0.12,
            fill_color=s.color,
            fill_opacity=0.94,
            stroke_color=_series_stroke(s.color),
            stroke_width=0.55,
        )
        label = Text(
            s.name, font=theme.mono_font, font_size=font_size, color=theme.text
        )
        entry = VGroup(swatch, label).arrange(RIGHT, buff=0.1)
        entries.add(entry)
    entries.arrange(RIGHT, buff=0.5)
    return entries


def _nice_step(y_max: float, target_ticks: int = 5) -> float:
    raw = y_max / target_ticks
    magnitude = 10 ** int(f"{raw:.0e}".split("e")[1])
    normalized = raw / magnitude
    if normalized <= 1:
        return magnitude
    if normalized <= 2:
        return 2 * magnitude
    if normalized <= 5:
        return 5 * magnitude
    return 10 * magnitude


def _tick_range(start: float, stop: float, step: float) -> list[float]:
    result = []
    val = start
    while val <= stop + step * 0.01:
        result.append(val)
        val += step
    return result
