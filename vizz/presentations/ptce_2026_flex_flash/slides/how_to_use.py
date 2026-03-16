from __future__ import annotations

from typing import TYPE_CHECKING

from manim import *

if TYPE_CHECKING:
    from vizz.presentations.components import SlideBase


def _usage_card(scene: SlideBase) -> Group:
    t = scene.theme
    frame = RoundedRectangle(
        corner_radius=t.panel_corner_radius,
        width=11.0,
        height=3.9,
        stroke_color=t.accent_primary,
        stroke_width=t.panel_stroke_width,
        fill_color=t.panel_fill,
        fill_opacity=1,
    )
    title = scene.meta_text(
        "Same FlexAttention API, one-line backend switch",
        font_size=16,
        color=t.accent_primary,
        uppercase=False,
    )
    title.move_to(frame.get_top() + DOWN * 0.34)
    title.align_to(frame.get_left() + RIGHT * 0.3, LEFT)
    code_obj = scene.themed_code(
        "from torch.nn.attention.flex_attention import flex_attention, create_block_mask\n"
        "\n"
        "block_mask = create_block_mask(mask_mod, B, H, S, S, BLOCK_SIZE=(256, 128))\n"
        "\n"
        "out = flex_attention(\n"
        "    q, k, v,\n"
        "    score_mod=score_mod,\n"
        "    block_mask=block_mask,\n"
        '    kernel_options={"BACKEND": "FLASH"},\n'
        ")",
        font_size=18,
    )
    code_obj.scale_to_fit_width(frame.width - 1.0)
    if code_obj.height > frame.height - 0.95:
        code_obj.scale_to_fit_height(frame.height - 0.95)
    code_text = code_obj[1]
    content_cy = (title.get_bottom()[1] - 0.15 + frame.get_bottom()[1] + 0.15) / 2
    code_text.move_to([frame.get_center()[0], content_cy, 0])
    return Group(frame, title, code_text)


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = scene.section_header("How to use the Flash backend")
    usage_card = _usage_card(scene)
    usage_card.next_to(header, DOWN, buff=0.18)

    bullets = scene.bullet_list(
        "Same FlexAttention API -- bring your existing score_mod and block_mask.",
        "On Blackwell, set BLOCK_SIZE=(256, 128) for create_block_mask.",
        'kernel_options={"BACKEND": "FLASH"} routes into the FA4 kernel.',
        font_size=22,
        color=t.text,
    ).to_edge(DOWN, buff=0.4)

    scene.play(FadeIn(header[0], shift=UP * 0.15), Create(header[1]))
    scene.play(
        FadeIn(usage_card[0]),
        FadeIn(usage_card[1]),
        FadeIn(usage_card[2], shift=UP * 0.08),
    )
    scene.play(
        LaggedStart(*[FadeIn(row, shift=UP * 0.08) for row in bullets], lag_ratio=0.12)
    )
    scene.next_slide()
    scene.clear_stage()
