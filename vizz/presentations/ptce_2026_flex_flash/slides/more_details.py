from __future__ import annotations

from typing import TYPE_CHECKING

from manim import *

if TYPE_CHECKING:
    from vizz.presentations.components import SlideBase


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = scene.section_header("Learn More")

    gym_qr = ImageMobject(
        "vizz/presentations/ptce_2026_flex_flash/assets/qrcode_attention_gym.png"
    ).scale_to_fit_height(3.7)
    blog_qr = ImageMobject(
        "vizz/presentations/ptce_2026_flex_flash/assets/qrcode_pytorch_blog.png"
    ).scale_to_fit_height(3.7)

    gym_title = scene.meta_text(
        "Examples @ the Attn Gym",
        font_size=20,
        color=t.accent_primary,
        uppercase=False,
    )
    blog_title = scene.meta_text(
        "All the details @ our pytorch blog",
        font_size=20,
        color=t.accent_primary,
        uppercase=False,
    )

    gym_blurb = scene.body_text(
        "Examples and benchmarks:\ngithub.com/meta-pytorch/attention-gym",
        font_size=20,
        color=t.muted_text,
    )
    blog_blurb = scene.body_text(
        "Detailed methodology and setup:\npytorch.org",
        font_size=20,
        color=t.muted_text,
    )

    gym_col = Group(gym_title, gym_qr, gym_blurb).arrange(DOWN, buff=0.26)
    blog_col = Group(blog_title, blog_qr, blog_blurb).arrange(DOWN, buff=0.26)
    ctas = Group(gym_col, blog_col).arrange(RIGHT, buff=1.0, aligned_edge=UP)
    ctas.next_to(header, DOWN, buff=0.35)

    scene.play(FadeIn(header[0], shift=UP * 0.15), Create(header[1]))
    scene.play(
        FadeIn(gym_title, shift=UP * 0.08),
        FadeIn(blog_title, shift=UP * 0.08),
        FadeIn(gym_qr, scale=0.9),
        FadeIn(blog_qr, scale=0.9),
        FadeIn(gym_blurb, shift=UP * 0.08),
        FadeIn(blog_blurb, shift=UP * 0.08),
    )
