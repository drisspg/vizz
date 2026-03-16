from __future__ import annotations

from typing import TYPE_CHECKING, Any

from manim import *

if TYPE_CHECKING:
    from vizz.presentations.components import SlideBase


DET_COLORS = ["#7a8f9d", "#9f7754", "#6f8b7b", "#8b7d9e"]


def _pill(
    text: str,
    *,
    fill: str,
    stroke: str,
    text_color: str,
    font_size: int = 18,
) -> Group:
    label = Text(text, font_size=font_size, color=text_color, weight=MEDIUM)
    bg = RoundedRectangle(
        corner_radius=0.16,
        width=label.width + 0.42,
        height=label.height + 0.26,
        fill_color=fill,
        fill_opacity=1.0,
        stroke_color=stroke,
        stroke_width=1.3,
    )
    label.move_to(bg)
    return Group(bg, label)


def _fit_panel_content(panel: Group, content: Mobject) -> Mobject:
    frame = panel[0]
    divider = panel[2]
    max_width = frame.width - 0.72
    max_height = divider.get_y() - frame.get_bottom()[1] - 0.38
    if content.width > max_width:
        content.scale_to_fit_width(max_width)
    if content.height > max_height:
        content.scale_to_fit_height(max_height)
    content.move_to(
        [frame.get_center()[0], (divider.get_y() + frame.get_bottom()[1]) / 2, 0]
    )
    return content


def _number_token(number: int, color: str) -> Group:
    circle = Circle(radius=0.2, stroke_width=2.0, stroke_color=color)
    circle.set_fill(color, opacity=0.18)
    label = Text(str(number), font_size=19, color=color, weight=SEMIBOLD)
    label.move_to(circle)
    token = Group(circle, label)
    token.number = number
    return token


def _ordered_slot(number: int, color: str) -> Group:
    rect = RoundedRectangle(
        width=1.12,
        height=0.42,
        corner_radius=0.1,
        stroke_width=1.4,
        stroke_color=color,
        fill_color=color,
        fill_opacity=0.07,
    )
    idx = Text(str(number), font_size=16, color=color, weight=MEDIUM)
    idx.move_to(rect.get_left() + RIGHT * 0.2)
    slot = Group(rect, idx)
    slot.target_point = rect.get_center() + RIGHT * 0.12
    return slot


def _task_tile(
    *,
    fill: str,
    opacity: float,
    width: float = 0.56,
    height: float = 0.34,
) -> RoundedRectangle:
    tile = RoundedRectangle(
        corner_radius=0.06,
        width=width,
        height=height,
        fill_color=fill,
        fill_opacity=opacity,
        stroke_width=0,
    )
    tile.set_fill(fill, opacity=opacity)
    return tile


def _deterministic_backward_card(scene: SlideBase) -> dict[str, Any]:
    t = scene.theme
    panel = scene.labeled_panel("Deterministic backward", width=6.0, height=4.9)

    source_tokens = Group(
        *[_number_token(number, DET_COLORS[number]) for number in [2, 0, 3, 1]]
    )
    source_tokens.arrange(DOWN, buff=0.22)
    source_tokens[0].shift(LEFT * 0.08 + UP * 0.05)
    source_tokens[1].shift(RIGHT * 0.03)
    source_tokens[2].shift(LEFT * 0.05 + DOWN * 0.03)
    source_tokens[3].shift(RIGHT * 0.08)
    source_label = scene.meta_text(
        "partial grads", font_size=13, color=t.accent_primary, uppercase=False
    )
    source_col = Group(source_label, source_tokens).arrange(DOWN, buff=0.14)

    structures = Group(
        _pill(
            "owners",
            fill="#ebe6dc",
            stroke="#b9ad97",
            text_color="#5f564a",
            font_size=15,
        ),
        _pill(
            "offsets",
            fill="#e7ecef",
            stroke="#92a4ae",
            text_color="#53646e",
            font_size=15,
        ),
        _pill(
            "slots",
            fill="#e5ece8",
            stroke="#8aa08e",
            text_color="#506256",
            font_size=15,
        ),
    )
    structures.arrange(DOWN, buff=0.15)
    structures_label = scene.meta_text(
        "extra ordering state", font_size=13, color=t.accent_primary, uppercase=False
    )
    structure_col = Group(structures_label, structures).arrange(DOWN, buff=0.14)

    ordered_slots = Group(
        *[_ordered_slot(number, DET_COLORS[number]) for number in range(4)]
    )
    ordered_slots.arrange(DOWN, buff=0.18)
    ordered_label = scene.meta_text(
        "stable reduction order", font_size=13, color=t.accent_primary, uppercase=False
    )
    ordered_col = Group(ordered_label, ordered_slots).arrange(DOWN, buff=0.14)

    visual = Group(source_col, structure_col, ordered_col).arrange(
        RIGHT, buff=0.34, aligned_edge=DOWN
    )
    caption = scene.body_text(
        "Extra data structures impose a reproducible accumulation order.",
        font_size=18,
        color=t.muted_text,
    )
    content = Group(visual, caption).arrange(DOWN, buff=0.3)
    content = _fit_panel_content(panel, content)
    group = Group(panel, content)

    return {
        "group": group,
        "tokens": source_tokens,
        "structures": structures,
        "ordered_slots": ordered_slots,
    }


def _clc_card(scene: SlideBase) -> dict[str, Any]:
    t = scene.theme
    cool = "#5f7684"
    warm = "#8a5a34"
    neutral = "#b7b1a6"

    panel = scene.labeled_panel("CLC dynamic work scheduling", width=6.2, height=4.9)

    busy_tiles = Group(
        *[_task_tile(fill=cool, opacity=0.9 if idx < 4 else 0.28) for idx in range(5)]
    )
    busy_tiles.arrange(DOWN, buff=0.08)
    busy_label = scene.meta_text(
        "busy queue", font_size=13, color=t.accent_primary, uppercase=False
    )
    busy_col = Group(busy_label, busy_tiles).arrange(DOWN, buff=0.14)

    helper_tiles = Group(
        _task_tile(fill=warm, opacity=0.9),
        _task_tile(fill=neutral, opacity=0.28),
        _task_tile(fill=neutral, opacity=0.28),
        _task_tile(fill=neutral, opacity=0.28),
        _task_tile(fill=neutral, opacity=0.28),
    )
    helper_tiles.arrange(DOWN, buff=0.08)
    helper_label = scene.meta_text(
        "idle queue", font_size=13, color=t.accent_primary, uppercase=False
    )
    helper_col = Group(helper_label, helper_tiles).arrange(DOWN, buff=0.14)

    columns = Group(busy_col, helper_col).arrange(RIGHT, buff=1.0, aligned_edge=DOWN)
    steal_arrow = CurvedArrow(
        busy_tiles[2].get_right() + RIGHT * 0.02,
        helper_tiles[2].get_left() + LEFT * 0.02,
        angle=-PI / 3,
        color=t.accent_primary,
        stroke_width=2.6,
    )
    steal_label = scene.meta_text(
        "steal work", font_size=13, color=t.accent_primary, uppercase=False
    )
    steal_label.next_to(steal_arrow, UP, buff=0.05)
    badge = _pill(
        "recently landed",
        fill="#ebe6dc",
        stroke="#b9ad97",
        text_color="#5f564a",
        font_size=16,
    )

    visual = Group(columns, steal_arrow, steal_label)
    caption = scene.body_text(
        "Idle blocks can pull tiles over when the workload gets skewed.",
        font_size=18,
        color=t.muted_text,
    )
    content = Group(visual, badge, caption).arrange(DOWN, buff=0.24)
    content = _fit_panel_content(panel, content)
    group = Group(panel, content)

    return {
        "group": group,
        "busy_tiles": busy_tiles,
        "helper_tiles": helper_tiles,
        "steal_arrow": steal_arrow,
        "steal_label": steal_label,
        "cool": cool,
        "warm": warm,
        "neutral": neutral,
    }


def build(scene: SlideBase) -> None:
    header = scene.section_header("More Things Coming")

    left = _deterministic_backward_card(scene)
    right = _clc_card(scene)
    content = Group(left["group"], right["group"]).arrange(RIGHT, buff=0.45)
    content.next_to(header, DOWN, buff=0.28)
    content.shift(DOWN * 0.04)

    scene.play(FadeIn(header[0], shift=UP * 0.15), Create(header[1]))
    scene.play(
        FadeIn(left["group"], shift=UP * 0.08),
        FadeIn(right["group"], shift=UP * 0.08),
    )

    slot_by_num = {int(slot[1].text): slot for slot in left["ordered_slots"]}
    moving_tokens = Group(*[token.copy() for token in left["tokens"]])
    det_paths = []
    for token in moving_tokens:
        slot = slot_by_num[token.number]
        end_point = slot[0].get_center() + RIGHT * 0.12
        det_paths.append(
            CubicBezier(
                token.get_center(),
                token.get_center() + RIGHT * 0.6,
                end_point + LEFT * 0.42,
                end_point,
            )
        )
    scene.add(*moving_tokens)
    for token in left["tokens"]:
        token.set_opacity(0.28)

    scene.play(
        AnimationGroup(
            *[
                ShowPassingFlash(
                    path.copy().set_stroke(color=DET_COLORS[token.number], width=3.0),
                    time_width=0.6,
                )
                for token, path in zip(moving_tokens, det_paths, strict=True)
            ],
            lag_ratio=0.12,
        ),
        LaggedStart(
            *[
                MoveAlongPath(token, path)
                for token, path in zip(moving_tokens, det_paths, strict=True)
            ],
            lag_ratio=0.12,
        ),
        AnimationGroup(
            *[
                Indicate(chip[0], color=scene.theme.accent_primary, scale_factor=1.03)
                for chip in left["structures"]
            ],
            lag_ratio=0.18,
        ),
        run_time=2.1,
    )

    transfer_path = CubicBezier(
        right["busy_tiles"][2].get_center(),
        right["busy_tiles"][2].get_center() + RIGHT * 0.75 + UP * 0.18,
        right["helper_tiles"][2].get_center() + LEFT * 0.7 + UP * 0.18,
        right["helper_tiles"][2].get_center(),
    )
    moving_tile = right["busy_tiles"][2].copy()
    moving_tile.set_fill(right["warm"], opacity=0.95)
    scene.add(moving_tile)
    scene.play(
        FadeToColor(right["busy_tiles"][2], right["neutral"]),
        right["busy_tiles"][2].animate.set_fill(right["neutral"], opacity=0.28),
        MoveAlongPath(moving_tile, transfer_path),
        Indicate(
            right["steal_arrow"], color=scene.theme.accent_primary, scale_factor=1
        ),
        FadeIn(right["steal_label"], shift=UP * 0.03),
        run_time=1.6,
    )
    scene.play(
        ReplacementTransform(moving_tile, right["helper_tiles"][2]),
        right["helper_tiles"][2].animate.set_fill(right["warm"], opacity=0.95),
        run_time=0.45,
    )
    scene.play(
        Indicate(
            right["helper_tiles"][1],
            color=scene.theme.accent_primary,
            scale_factor=1.03,
        ),
        Indicate(
            right["helper_tiles"][2],
            color=scene.theme.accent_primary,
            scale_factor=1.03,
        ),
        run_time=0.5,
    )

    scene.next_slide()
    scene.clear_stage()
