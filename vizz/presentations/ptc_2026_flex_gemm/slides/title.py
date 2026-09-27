from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    DashedVMobject,
    FadeIn,
    ImageMobject,
    Line,
    RoundedRectangle,
    TransformFromCopy,
    VGroup,
)

from vizz.presentations.components import SlideBase

PLATE = Path(__file__).parents[1] / "assets" / "tile_plate.png"
# Pixel boxes in tile_plate.png (300 dpi), measured from its green strokes: the
# epilogue bead chain (beads centered at x=1154, r~53, y 629..1015) plus ~24px padding.
CHAIN_PX = (1078, 604, 1231, 1040)


def _px(plate: ImageMobject, width_px: int, x: float, y: float) -> np.ndarray:
    scale = plate.width / width_px
    return plate.get_corner(UP + LEFT) + np.array([x * scale, -y * scale, 0.0])


def build(scene: SlideBase) -> None:
    t = scene.theme
    kicker = scene.meta_text("PyTorch Conference NA 2026 · lightning talk")
    flex = scene.title_text("Flex", font_size=72).set_color(t.accent_success)
    gemm = scene.title_text("GEMM", font_size=72)
    title = VGroup(flex, gemm).arrange(RIGHT, buff=0.04, aligned_edge=DOWN)
    subtitle = scene.body_text(
        "Bringing flexible PyTorch\nepilogues to GEMM", font_size=32, color=t.muted_text
    )
    rule = Line(LEFT, RIGHT, color=t.divider, stroke_width=1.0)
    speaker = scene.body_text("Driss Guessous · PyTorch @ Meta", font_size=24)
    text = VGroup(kicker, title, subtitle, rule, speaker).arrange(
        DOWN, aligned_edge=LEFT, buff=0.3
    )
    rule.put_start_and_end_on(rule.get_left(), rule.get_left() + RIGHT * text.width)
    text.to_edge(LEFT, buff=0.7).shift(UP * 0.1)

    plate = ImageMobject(str(PLATE))
    width_px = plate.pixel_array.shape[1]
    plate.scale_to_fit_width(6.3).to_edge(RIGHT, buff=0.35)

    scene.play(FadeIn(text, shift=UP * 0.1), FadeIn(plate))
    scene.wait(0.2)
    scene.next_slide(
        notes="title — The plate is the whole talk: A and B panels sweep over k into one accumulator tile (amber). Everything in ink is the GEMM we already have; the green chain is the part you write. The dotted C is never written; D is the only thing that leaves the kernel."
    )

    x0, y0, x1, y1 = CHAIN_PX
    top_left = _px(plate, width_px, x0, y0)
    bottom_right = _px(plate, width_px, x1, y1)
    slot = DashedVMobject(
        RoundedRectangle(
            corner_radius=0.06,
            width=bottom_right[0] - top_left[0],
            height=top_left[1] - bottom_right[1],
            stroke_color=t.accent_success,
            stroke_width=1.6,
        ).move_to((top_left + bottom_right) / 2),
        num_dashes=36,
    )
    label = (
        scene.title_text("Flex", font_size=26)
        .set_color(t.accent_success)
        .next_to(slot, UP, buff=0.08)
    )
    scene.play(Create(slot), TransformFromCopy(flex, label), run_time=1.0)
    scene.wait(0.2)
    scene.next_slide(
        notes="title.flex — That green region is the Flex in FlexGEMM: a hook in the store path of a fixed, fast GEMM where your PyTorch runs on the accumulator tile. Same idea as score_mod in FlexAttention."
    )
