from __future__ import annotations

from typing import TYPE_CHECKING

from manim import *

if TYPE_CHECKING:
    from vizz.presentations.components import SlideBase


def _stage_card(
    scene: SlideBase, title: str, code_string: str, color: str, language: str
) -> Group:
    t = scene.theme
    frame = RoundedRectangle(
        corner_radius=t.panel_corner_radius,
        width=11.2,
        height=3.25,
        stroke_color=color,
        stroke_width=t.panel_stroke_width,
        fill_color=t.panel_fill,
        fill_opacity=1,
    )
    title_text = scene.meta_text(
        title,
        font_size=16,
        color=color,
        uppercase=False,
    )
    title_text.move_to(frame.get_top() + DOWN * 0.34)
    title_text.align_to(frame.get_left() + RIGHT * 0.32, LEFT)
    code_obj = scene.themed_code(code_string, language=language, font_size=20)
    code_obj.scale_to_fit_width(frame.width - 1.15)
    if code_obj.height > frame.height - 0.95:
        code_obj.scale_to_fit_height(frame.height - 0.95)
    code_text = code_obj[1]
    content_cy = (title_text.get_bottom()[1] - 0.15 + frame.get_bottom()[1] + 0.15) / 2
    code_text.move_to([frame.get_center()[0], content_cy, 0])
    return Group(frame, title_text, code_text)


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = scene.section_header(
        "The key idea: lower Python score mods into CuTeDSL inside FA4"
    )
    stage_specs = [
        (
            "Python score_mod",
            t.accent_primary,
            """
def alibi_mod(score, b, h, q_idx, kv_idx):
    scale = torch.exp2(-((h + 1) * 8.0 / H))
    bias = (kv_idx - q_idx) * scale
    return score + bias
            """,
            "python",
        ),
        (
            "FX / Inductor pointwise IR",
            t.accent_secondary,
            """
tmp0 = h + 1
tmp1 = tmp0 * 8.0 / H
tmp2 = exp2(-tmp1)
tmp3 = kv_idx - q_idx
tmp4 = tmp3 * tmp2
return score + tmp4
            """,
            "python",
        ),
        (
            "CuTeDSL TensorSSA",
            t.accent_success,
            """
tmp0 = h + cute.full_like(h, 1)
tmp1 = tmp0 * cute.full_like(tmp0, 8.0 / H)
tmp2 = cute.math.exp2(-tmp1)
tmp3 = kv_idx - q_idx
tmp4 = tmp3 * tmp2
return score + tmp4
            """,
            "python",
        ),
        (
            "FA4 hook args",
            t.accent_danger,
            """
out, lse = _flash_attn_fwd(
    q, k, v,
    score_mod=score_mod,
    mask_mod=mask_mod,
    block_sparse_tensors=block_sparse_tensors,
    aux_tensors=aux_tensors,
)
            """,
            "python",
        ),
    ]

    cards = [
        _stage_card(scene, title, code_string, color, language)
        for title, color, code_string, language in stage_specs
    ]
    for card in cards:
        card.move_to(UP * 0.1)
    frame = cards[0][0].copy()
    current_title = cards[0][1].copy()
    current_code = cards[0][2].copy()

    stage_row = VGroup(
        *[
            scene.meta_text(title, font_size=14, color=color, uppercase=False)
            for title, color, _, _ in stage_specs
        ]
    ).arrange(RIGHT, buff=0.45)
    stage_row.to_edge(DOWN, buff=0.58)

    progress = Line(
        stage_row.get_left() + DOWN * 0.26,
        stage_row.get_right() + DOWN * 0.26,
        color=t.divider,
        stroke_width=3,
    )
    progress_dots = VGroup(
        *[
            Dot(point=label.get_center() + DOWN * 0.26, radius=0.08, color=color)
            for label, (_, color, _, _) in zip(stage_row, stage_specs)
        ]
    )
    progress_indicator = Dot(radius=0.11, color=t.text)
    progress_indicator.move_to(progress_dots[0].get_center())

    tagline = scene.body_text(
        "Same user intent, progressively lowered into the host FA4 kernel",
        font_size=22,
        color=t.muted_text,
    ).next_to(stage_row, UP, buff=0.18)
    detail_panel = scene.labeled_panel(
        "Added entrypoints for user-defined score_mods and mask_mods",
        width=11.1,
        height=2.8,
        content=scene.bullet_list(
            "fwd hooks: score_mod, mask_mod, block_sparse_tensors, aux_tensors",
            "bwd hooks: score_mod, score_mod_bwd, mask_mod, block_sparse_tensors,aux_tensors",
            font_size=20,
        ),
    )
    detail_panel[0].set_stroke(color=t.accent_danger)
    detail_panel[2].set_opacity(0)
    detail_panel.move_to(UP * 0.1)

    scene.play(FadeIn(header[0], shift=UP * 0.15), Create(header[1]))
    scene.play(
        FadeIn(frame, shift=UP * 0.15),
        FadeIn(current_title, shift=UP * 0.15),
        FadeIn(current_code, shift=UP * 0.15),
        Create(progress),
        FadeIn(stage_row, shift=UP * 0.08),
        FadeIn(progress_dots),
        FadeIn(progress_indicator),
    )
    scene.play(FadeIn(tagline, shift=UP * 0.08))
    scene.next_slide()

    for index, next_card in enumerate(cards[1:], start=1):
        scene.play(
            Transform(current_title, next_card[1].copy()),
            Transform(current_code, next_card[2].copy()),
            frame.animate.set_stroke(color=stage_specs[index][1]),
            progress_indicator.animate.move_to(progress_dots[index].get_center()),
            stage_row[index - 1].animate.set_color(t.muted_text),
            stage_row[index].animate.set_color(t.text),
            run_time=0.95,
        )
        scene.next_slide()

    scene.play(
        Transform(frame, detail_panel[0]),
        Transform(current_title, detail_panel[1]),
        FadeTransform(current_code, detail_panel[3]),
        FadeOut(tagline, shift=DOWN * 0.05),
        FadeOut(stage_row, shift=DOWN * 0.05),
        FadeOut(progress, shift=DOWN * 0.05),
        FadeOut(progress_dots, shift=DOWN * 0.05),
        FadeOut(progress_indicator, shift=DOWN * 0.05),
        run_time=0.9,
    )
    scene.next_slide()
    scene.clear_stage()
