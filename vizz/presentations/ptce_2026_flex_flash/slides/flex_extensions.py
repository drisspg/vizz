from __future__ import annotations

from typing import TYPE_CHECKING

from manim import *

if TYPE_CHECKING:
    from vizz.presentations.components import SlideBase


def _fit_panel_content(panel: Group, content: Mobject) -> Mobject:
    frame = panel[0]
    divider = panel[2]
    max_width = frame.width - 0.7
    max_height = divider.get_y() - frame.get_bottom()[1] - 0.5
    if content.width > max_width:
        content.scale_to_fit_width(max_width)
    if content.height > max_height:
        content.scale_to_fit_height(max_height)
    content.move_to(
        [frame.get_center()[0], (divider.get_y() + frame.get_bottom()[1]) / 2, 0]
    )
    return content


def _score_example(scene: SlideBase, name: str, code_string: str) -> VGroup:
    label = scene.meta_text(
        name,
        font_size=16,
        color=scene.theme.accent_primary,
        uppercase=False,
    )
    code = scene.themed_code(code_string, font_size=16)
    code.scale_to_fit_width(4.9)
    return VGroup(label, code).arrange(DOWN, buff=0.18)


def _mask_example(scene: SlideBase, name: str, pattern: list[list[bool]]) -> VGroup:
    label = scene.meta_text(
        name,
        font_size=16,
        color=scene.theme.accent_success,
        uppercase=False,
    )
    cells = VGroup()
    for row in pattern:
        for active in row:
            square = Square(
                side_length=0.34,
                stroke_color=scene.theme.panel_stroke,
                stroke_width=1.4,
                fill_color=scene.theme.accent_success if active else "#e2e8f0",
                fill_opacity=0.82 if active else 0.28,
            )
            cells.add(square)
    cells.arrange_in_grid(rows=len(pattern), cols=len(pattern[0]), buff=0.05)
    return VGroup(label, cells).arrange(DOWN, buff=0.24)


def _causal_pattern(size: int) -> list[list[bool]]:
    return [[col <= row for col in range(size)] for row in range(size)]


def _sliding_window_pattern(size: int, window_size: int) -> list[list[bool]]:
    return [
        [0 <= row - col <= window_size for col in range(size)] for row in range(size)
    ]


def _prefix_lm_pattern(size: int, prefix_length: int) -> list[list[bool]]:
    return [
        [col < prefix_length or col <= row for col in range(size)]
        for row in range(size)
    ]


def _document_causal_pattern(lengths: tuple[int, ...]) -> list[list[bool]]:
    doc_ids = [doc_id for doc_id, length in enumerate(lengths) for _ in range(length)]
    size = len(doc_ids)
    return [
        [doc_ids[row] == doc_ids[col] and col <= row for col in range(size)]
        for row in range(size)
    ]


def build(scene: SlideBase) -> None:
    header = scene.section_header(
        "FlexAttention adds two things to vanilla FlashAttention"
    )
    mask_examples = [
        _mask_example(scene, "Causal", _causal_pattern(7)),
        _mask_example(scene, "Sliding window", _sliding_window_pattern(7, 2)),
        _mask_example(scene, "Prefix LM", _prefix_lm_pattern(7, 2)),
        _mask_example(scene, "Document mask", _document_causal_pattern((2, 3, 2))),
    ]
    left_group = scene.labeled_panel("1. Score modifications", width=6.0, height=4.7)
    right_group = scene.labeled_panel("2. Mask patterns", width=6.2, height=4.7)
    Group(left_group, right_group).arrange(RIGHT, buff=0.5).move_to(DOWN * 0.1)
    mask_examples = [
        _fit_panel_content(right_group, example) for example in mask_examples
    ]

    frame = left_group[0]
    divider = left_group[2]
    panel_cx = frame.get_center()[0]
    max_code_width = frame.width - 0.7
    label_y = divider.get_y() - 0.55
    code_cy = (divider.get_y() - 0.9 + frame.get_bottom()[1] + 0.25) / 2

    score_data = [
        (
            "ALiBi",
            "def alibi_mod(score, b, h, q_idx, kv_idx):\n"
            "    bias = (kv_idx - q_idx) * scale[h]\n"
            "    return score + bias",
        ),
        (
            "Soft cap",
            "def tanh_softcap(score, b, h, q_idx, kv_idx):\n"
            "    return soft_cap * tanh(score / soft_cap)",
        ),
        (
            "Activation wrapper",
            "def relu_score_mod(score, b, h, q_idx, kv_idx):\n"
            "    return log(relu(score) + 1.0)",
        ),
        (
            "Relative bias",
            "def relative_bias(score, b, h, q_idx, kv_idx):\n"
            "    return score + (q_idx - kv_idx)",
        ),
    ]
    score_labels = []
    score_code_texts = []
    max_bg_w = 0.0
    max_bg_h = 0.0
    for name, code_string in score_data:
        label = scene.meta_text(
            name, font_size=16, color=scene.theme.accent_primary, uppercase=False
        )
        label.move_to([panel_cx, label_y, 0])
        code_obj = scene.themed_code(code_string, font_size=16)
        if code_obj.width > max_code_width:
            code_obj.scale_to_fit_width(max_code_width)
        max_bg_w = max(max_bg_w, code_obj[0].width)
        max_bg_h = max(max_bg_h, code_obj[0].height)
        code_text = code_obj[1]
        code_text.move_to([panel_cx, code_cy, 0])
        score_labels.append(label)
        score_code_texts.append(code_text)

    t = scene.theme
    code_bg = RoundedRectangle(
        corner_radius=0.2,
        width=max_bg_w,
        height=max_bg_h,
        fill_color=t.code_background or t.panel_fill,
        fill_opacity=1.0,
        stroke_color=t.panel_stroke,
        stroke_width=1.0,
    )
    code_bg.move_to([panel_cx, code_cy, 0])

    scene.play(FadeIn(header[0], shift=UP * 0.15), Create(header[1]))
    scene.play(FadeIn(left_group[0]), FadeIn(left_group[1]), Create(left_group[2]))
    scene.play(FadeIn(right_group[0]), FadeIn(right_group[1]), Create(right_group[2]))
    scene.play(
        FadeIn(score_labels[0]),
        FadeIn(code_bg),
        FadeIn(score_code_texts[0]),
        FadeIn(mask_examples[0], shift=UP * 0.1),
    )
    scene.wait(0.2)
    cur_sl, cur_ct = score_labels[0], score_code_texts[0]
    current_mask = mask_examples[0]
    for nl, nct, next_mask in zip(
        score_labels[1:], score_code_texts[1:], mask_examples[1:]
    ):
        scene.next_slide()
        scene.play(
            TransformMatchingShapes(cur_sl, nl),
            TransformMatchingShapes(cur_ct, nct),
            ReplacementTransform(current_mask, next_mask),
        )
        scene.wait(0.2)
        cur_sl, cur_ct, current_mask = nl, nct, next_mask
    scene.next_slide()
    scene.clear_stage()
