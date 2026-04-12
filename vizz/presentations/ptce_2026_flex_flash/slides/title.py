from __future__ import annotations

from typing import TYPE_CHECKING

from manim import *

if TYPE_CHECKING:
    from vizz.presentations.components import SlideBase

from vizz.presentations.ptce_2026_flex_flash import ASSET_ROOT


def build(scene: SlideBase) -> None:
    t = scene.theme
    issue_mark = scene.meta_text(
        "PyTorch Conference Europe 2026 ⋅ Driss Guessous ⋅ Meta Superintelligence Labs",
        font_size=18,
        color=t.accent_primary,
    )
    heading = scene.title_text("FlexAttention\n+ FlashAttention-4", font_size=70)
    subheading = scene.body_text(
        "A faster backend for\ncustom attention patterns",
        font_size=38,
        color=t.muted_text,
    )
    takeaway = Text(
        "1.2x to 3.2x faster over existing Triton\nimplementation on compute-bound workloads",
        # font=t.sans_font,
        font_size=29,
        color=t.text,
        t2w={"1.2x to 3.2x": BOLD},
    )
    takeaway_group = VGroup(takeaway).arrange(DOWN, aligned_edge=LEFT, buff=0.16)

    left_column = VGroup(
        issue_mark,
        heading,
        subheading,
        takeaway_group,
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.26)
    left_column.set_width(min(left_column.width, 6.2))

    hero_frame = scene.panel(width=6.5, height=4.85)
    hero = scene.image_panel(str(ASSET_ROOT / "flex_flash.png"), width=6.18)
    if hero.height > hero_frame.height - 0.2:
        hero.scale_to_fit_height(hero_frame.height - 0.2)
    hero.move_to(hero_frame.get_center())
    hero_panel = Group(hero_frame, hero)

    content = Group(left_column, hero_panel).arrange(RIGHT, buff=0.6, aligned_edge=UP)
    content.move_to(UP * 0.12)

    scene.play(
        FadeIn(issue_mark, shift=UP * 0.1),
        FadeIn(heading, shift=UP * 0.16),
        FadeIn(subheading, shift=UP * 0.08),
    )
    scene.next_slide()
    scene.play(
        FadeIn(hero_panel, shift=LEFT * 0.12),
        FadeIn(takeaway_group, shift=UP * 0.06),
    )
    scene.next_slide()
    scene.clear_stage()
