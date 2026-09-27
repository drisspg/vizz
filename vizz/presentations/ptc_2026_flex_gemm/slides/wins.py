from manim import DOWN, LEFT, RIGHT, UP, FadeIn, GrowFromEdge, Rectangle, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header
from vizz.presentations.ptc_2026_flex_gemm.slides.common import punchline, tint

# Schematic widths only; not measured data.
GEMM_W, TRAFFIC_W, PENALTY_W = 5.2, 2.2, 0.35


def _bar(scene: SlideBase, width: float, color: str, label: str) -> VGroup:
    # Illustrative, not data: outline + faint tint so it never reads as a measured bar.
    rect = Rectangle(
        width=width,
        height=0.6,
        stroke_color=color,
        stroke_width=1.4,
        fill_color=tint(scene, color, 0.16),
        fill_opacity=1,
    )
    text = scene.meta_text(label, font_size=13)
    if text.width > width - 0.1:
        text = scene.meta_text(label, font_size=13).next_to(rect, UP, buff=0.08)
    else:
        text.move_to(rect)
    return VGroup(rect, text)


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(scene, "Where the win comes from: bytes, not magic")

    equation = punchline(scene, "ΔBytes  =  2 · M · g · N · sizeof(C)", font_size=32)
    legend = scene.meta_text(
        "g = 1 same-shape epilogue · g = 2 SwiGLU gate/up", uppercase=False
    )
    top = (
        VGroup(equation, legend)
        .arrange(DOWN, buff=0.18)
        .next_to(header, DOWN, buff=0.35)
    )

    unfused_gemm = _bar(scene, GEMM_W, t.muted_text, "gemm")
    unfused_traffic = _bar(scene, TRAFFIC_W, t.accent_danger, "write C + read C")
    unfused = VGroup(unfused_gemm, unfused_traffic).arrange(RIGHT, buff=0)
    fused_gemm = _bar(scene, GEMM_W, t.muted_text, "gemm")
    fused_penalty = _bar(scene, PENALTY_W, t.accent_secondary, "")
    fused = VGroup(fused_gemm, fused_penalty).arrange(RIGHT, buff=0)
    labels = VGroup(
        scene.meta_text("unfused", uppercase=False),
        scene.meta_text("fused", uppercase=False),
    )
    bars = VGroup(unfused, fused).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
    for label, bar in zip(labels, bars):
        label.next_to(bar, LEFT, buff=0.3)
    chart = VGroup(labels, bars).move_to(DOWN * 0.2)
    penalty_note = (
        scene.meta_text(
            "epilogue cost hidden in the store path (if it is cheap enough)",
            uppercase=False,
            color=t.accent_secondary,
        )
        .next_to(fused, DOWN, buff=0.15)
        .align_to(fused, LEFT)
    )
    schematic = scene.meta_text(
        "schematic, not to scale", font_size=13, uppercase=False
    )
    schematic.next_to(chart, RIGHT, buff=0.3).align_to(chart, DOWN)

    scene.play(FadeIn(header), FadeIn(top))
    scene.play(
        FadeIn(labels),
        GrowFromEdge(unfused, LEFT),
        GrowFromEdge(fused, LEFT),
        FadeIn(penalty_note),
        FadeIn(schematic),
        run_time=0.8,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="wins.model — Fused time is GEMM time times one plus a small epilogue penalty; unfused adds the C round trip. Biggest wins when the separate epilogue is bandwidth-bound and the GEMM is not saturating compute, or when the output is much smaller than the accumulator (fp8/fp4)."
    )

    caveats = scene.bullet_list(
        "Small / skinny shapes are launch-bound: fusion can lose",
        "C may already be hot in L2: less traffic to remove",
        "Heavy epilogues cost registers and slow the mainloop",
        font_size=21,
    )
    caveat_head = scene.meta_text("when it loses", color=t.accent_danger)
    caveat_block = VGroup(caveat_head, caveats).arrange(
        DOWN, aligned_edge=LEFT, buff=0.15
    )
    caveat_block.next_to(penalty_note, DOWN, buff=0.4).align_to(chart, LEFT)
    scene.play(FadeIn(caveat_block, shift=UP * 0.1), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="wins.caveats — Fusion is not free. This is why older Triton epilogue fusion rarely beat cuBLAS plus a standalone kernel, and why dispatch should be able to say no."
    )
