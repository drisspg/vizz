from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    FadeIn,
    FadeOut,
    Rectangle,
    ReplacementTransform,
    Transform,
    Uncreate,
    VGroup,
)

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import (
    code_block,
    flow,
    kernel,
    tensor,
    wasted_flow,
)
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header

CODE = """
def mlp_up(a, b, bias):
    out = a @ b              # GEMM
    return relu(out + bias)  # epilogue
"""

TOP_Y, BAND_Y = -0.85, -2.55


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(
        scene, "Epilogues are everywhere for those with the eyes to see"
    )
    code = code_block(scene, "what users write", CODE, font_size=24)
    code.next_to(header, DOWN, buff=0.45).align_to(header, LEFT)
    examples = VGroup(
        scene.meta_text("the epilogue can be"),
        scene.body_text(
            "bias · activation · residual · α/β\naux outputs · fp8 scales · SwiGLU",
            font_size=20,
        ),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    examples.next_to(code, RIGHT, buff=0.6).align_to(code, UP)

    # Unfused: kernels on top; C detours down to off-chip HBM and back up.
    gemm = kernel(scene, "GEMM", width=2.6).move_to([-4.6, TOP_Y, 0])
    epi = kernel(scene, "epilogue", width=2.6).move_to([0.6, TOP_Y, 0])
    d_mem = tensor(scene, "D  [M × N]", width=2.2, height=0.75).move_to(
        [4.9, BAND_Y, 0]
    )
    band = Rectangle(
        width=12.4,
        height=1.15,
        stroke_color=t.divider,
        stroke_width=1.0,
        fill_color=t.panel_fill,
        fill_opacity=1,
    ).move_to([0.1, BAND_Y, 0])
    band_label = scene.meta_text("hbm · global memory (slower)", font_size=13)
    band_label.move_to(
        band.get_corner(DOWN + LEFT) + RIGHT * 0.2 + UP * 0.2, aligned_edge=DOWN + LEFT
    )
    c_mem = tensor(
        scene, "C  [M × N]", width=2.2, height=0.75, color=t.accent_danger
    ).move_to([-1.4, BAND_Y, 0])
    arrows = VGroup(
        wasted_flow(scene, gemm.get_bottom() + RIGHT * 0.4, c_mem.get_left()),
        wasted_flow(scene, c_mem.get_right(), epi.get_bottom() + LEFT * 0.4),
        flow(scene, epi.get_right(), d_mem.get_top()),
    )
    write_label = scene.meta_text("write C", font_size=13, color=t.accent_danger)
    write_label.next_to(arrows[0].get_center(), LEFT, buff=0.2)
    read_label = scene.meta_text("read C", font_size=13, color=t.accent_danger)
    read_label.next_to(arrows[1].get_center(), RIGHT, buff=0.2)
    round_trip = VGroup(write_label, read_label)
    unfused_label = scene.meta_text("today: eager, and often torch.compile")
    unfused_label.next_to(gemm, UP, buff=0.35).align_to(gemm, LEFT)

    scene.play(FadeIn(header), FadeIn(code), FadeIn(examples))
    scene.play(
        FadeIn(VGroup(band, band_label, gemm, epi, d_mem)),
        FadeIn(c_mem),
        *[Create(a) for a in arrows],
        FadeIn(round_trip),
        FadeIn(unfused_label),
        run_time=0.8,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="problem.unfused — Users write the epilogue as normal tensor code after mm. The GEMM output C leaves registers and shared memory: it is written out to HBM (global memory), then read all the way back up by a second kernel, and D is written again. Red dashed is traffic that exists only because we did not fuse."
    )

    # Fused, animated as the fusion itself: the round trip retracts, C is pulled
    # up out of HBM into the gap between the kernels, the kernels close in and
    # merge into one, and the HBM band is no longer needed.
    fused = kernel(
        scene, "GEMM + epilogue  (one launch)", width=7.8, color=t.accent_success
    ).move_to([-2.0, TOP_Y, 0])
    fused_arrow = flow(
        scene, fused.get_right(), d_mem.get_top(), color=t.accent_success
    )
    fused_label = scene.meta_text(
        "flexgemm: fused into the store path", color=t.accent_success
    ).move_to(unfused_label, aligned_edge=LEFT)
    saved = scene.body_text(
        "2 · M · N · sizeof(C) fewer HBM bytes, one fewer launch", font_size=24
    )
    saved.next_to(band, DOWN, buff=0.25).align_to(band, LEFT)
    caption = scene.meta_text(
        "C stays in registers: never written", font_size=13, color=t.accent_success
    ).move_to(c_mem)
    gap = (gemm.get_right()[0] + epi.get_left()[0]) / 2

    scene.play(
        Uncreate(arrows[0]),
        Uncreate(arrows[1]),
        FadeOut(round_trip),
        c_mem.animate.scale(0.55).move_to([gap, TOP_Y, 0]),
        run_time=0.7,
    )
    scene.play(
        gemm.animate.next_to(c_mem, LEFT, buff=0.05),
        epi.animate.next_to(c_mem, RIGHT, buff=0.05),
        run_time=0.5,
    )
    scene.play(
        ReplacementTransform(VGroup(gemm, c_mem, epi), fused),
        Transform(arrows[2], fused_arrow),
        FadeOut(unfused_label),
        FadeIn(fused_label),
        run_time=0.7,
    )
    scene.play(FadeIn(saved, shift=UP * 0.1), FadeIn(caption), run_time=0.5)
    scene.wait(0.2)
    scene.next_slide(
        notes="problem.fused — Fusing the epilogue into the store path: the round trip disappears, C never goes down to HBM, and two kernels become one. That is the entire source of the win; no magic mainloop speedup."
    )
