from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    FadeIn,
    FadeOut,
    ReplacementTransform,
    VGroup,
)

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import (
    box,
    code_block,
    empty_slot,
    flow,
    wasted_flow,
)
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header

CODE = """
def mlp_up(a, b, bias):
    out = a @ b              # GEMM
    return relu(out + bias)  # epilogue
"""


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
    ).arrange(DOWN, aligned_edge=[-1, 0, 0], buff=0.12)
    examples.next_to(code, RIGHT, buff=0.6).align_to(code, UP)

    # Unfused dataflow: GEMM -> C in HBM -> epilogue kernel -> D.
    gemm = box(scene, "GEMM kernel", width=2.6)
    c_mem = box(scene, "C  [M × N]", width=2.2, sublabel="hbm", color=t.accent_danger)
    epi = box(scene, "epilogue kernel", width=2.6)
    d_mem = box(scene, "D  [M × N]", width=2.2, sublabel="hbm")
    row = VGroup(gemm, c_mem, epi, d_mem).arrange(RIGHT, buff=0.95)
    row.scale_to_fit_width(12.4).move_to(DOWN * 1.35)
    arrows = VGroup(
        wasted_flow(scene, gemm.get_right(), c_mem.get_left()),
        wasted_flow(scene, c_mem.get_right(), epi.get_left()),
        flow(scene, epi.get_right(), d_mem.get_left()),
    )
    round_trip = scene.meta_text(
        "write C  ·  read C", font_size=13, color=t.accent_danger
    ).next_to(c_mem, DOWN, buff=0.18)
    unfused_label = scene.meta_text("today: eager, and often torch.compile")
    unfused_label.next_to(row, UP, buff=0.5).align_to(row, [-1, 0, 0])

    scene.play(FadeIn(header), FadeIn(code), FadeIn(examples))
    scene.play(
        FadeIn(row),
        *[Create(a) for a in arrows],
        FadeIn(round_trip),
        FadeIn(unfused_label),
        run_time=0.8,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="problem.unfused — Users write the epilogue as normal tensor code after mm. The accumulator C is written to HBM, read back by a second kernel, and D is written again. Red dashed is traffic that exists only because we did not fuse."
    )

    # Fused: the epilogue runs on the accumulator tile before the store.
    fused = box(
        scene, "GEMM + epilogue  (one kernel)", width=6.3, color=t.accent_success
    ).move_to(VGroup(gemm, epi))
    ghost = empty_slot(scene, "C · never written", width=2.4, height=0.6)
    ghost.next_to(fused, DOWN, buff=0.35)
    fused_arrow = flow(
        scene, fused.get_right(), d_mem.get_left(), color=t.accent_success
    )
    saved = scene.body_text(
        "2 · M · N · sizeof(C) fewer HBM bytes, one fewer launch", font_size=24
    ).next_to(ghost, DOWN, buff=0.4)
    fused_label = scene.meta_text(
        "flexgemm: fused into the store path", color=t.accent_success
    ).move_to(unfused_label, aligned_edge=[-1, 0, 0])

    scene.play(
        FadeOut(arrows),
        FadeOut(round_trip),
        ReplacementTransform(VGroup(gemm, epi), fused),
        ReplacementTransform(c_mem, ghost),
        Create(fused_arrow),
        FadeOut(unfused_label),
        FadeIn(fused_label),
        FadeIn(saved, shift=UP * 0.1),
        run_time=0.9,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="problem.fused — Fusing the epilogue into the store path removes the C write and the C read. That is the entire source of the win; no magic mainloop speedup."
    )
