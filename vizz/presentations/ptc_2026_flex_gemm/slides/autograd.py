from itertools import pairwise

from manim import DOWN, LEFT, RIGHT, UP, Create, FadeIn, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import (
    flow,
    kernel,
    tensor,
)
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header

# Source: ~/obsidian/Presentations/flex_gemm/flex_gemm_presentation_model_context.md
# ("Backward") and FLEX_GEMM_TALK.ipynb section 6 (TorchTitan integration).


def _row(scene: SlideBase, items: list, y: float) -> VGroup:
    row = VGroup(*items).arrange(RIGHT, buff=0.5)
    row.move_to([-4.6, y, 0], aligned_edge=LEFT)
    arrows = VGroup(
        *[flow(scene, a.get_right(), b.get_left()) for a, b in pairwise(row)]
    )
    return VGroup(row, arrows)


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(scene, "Why FlexGEMM is forward-only (for now)")

    problem = VGroup(
        scene.meta_text("the catch", color=t.accent_danger),
        scene.body_text(
            "The backward of an epilogue, E′(acc, dY), is an input to both gradient GEMMs:\n"
            "a prologue. Prologue fusion slows the mainloop, so automatic backward\n"
            "fusion would make the expensive part slower.",
            font_size=20,
        ),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    problem.next_to(header, DOWN, buff=0.3).align_to(header, LEFT)

    scene.play(FadeIn(header), FadeIn(problem))
    scene.wait(0.2)
    scene.next_slide(
        notes="autograd.catch — Differentiating the epilogue is easy; placing it is not. E prime needs the accumulator and dY and feeds the dgrad and wgrad GEMMs, so it is a prologue, and prologue fusion is the enemy of a fast mainloop."
    )

    fwd_label = scene.meta_text("forward", font_size=14)
    fwd = _row(
        scene,
        [
            kernel(scene, "mm₁ + relu", width=2.6, color=t.accent_success),
            tensor(scene, "h, pre-act", width=2.0, height=0.7),
            kernel(scene, "mm₂", width=2.0),
        ],
        0.35,
    )
    bwd_label = scene.meta_text("backward", font_size=14)
    bwd = _row(
        scene,
        [
            kernel(scene, "mm₂ dgrad + relu′", width=2.9, color=t.accent_success),
            tensor(scene, "d pre-act", width=2.0, height=0.7),
            kernel(scene, "mm₁ dgrad · wgrad", width=2.9),
        ],
        -1.1,
    )
    fwd_label.move_to([-6.25, fwd.get_y(), 0], aligned_edge=LEFT)
    bwd_label.move_to([-6.25, bwd.get_y(), 0], aligned_edge=LEFT)
    insight = (
        scene.body_text(
            "Instead: E′ becomes the epilogue of the GEMM that produces dY.",
            font_size=22,
            color=t.accent_success,
        )
        .next_to(bwd, DOWN, buff=0.35)
        .align_to(fwd_label, LEFT)
    )

    scene.play(FadeIn(fwd_label), FadeIn(fwd[0]), *[Create(a) for a in fwd[1]])
    scene.play(
        FadeIn(bwd_label),
        FadeIn(bwd[0]),
        *[Create(a) for a in bwd[1]],
        FadeIn(insight, shift=UP * 0.1),
        run_time=0.9,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="autograd.reroute — In an MLP, mm1's fused relu in the forward becomes relu-prime fused into the epilogue of mm2's dgrad in the backward, saving the pre-activation as an aux output. Every fused op stays an epilogue."
    )

    how = VGroup(
        scene.meta_text("how to use it today", font_size=14),
        scene.body_text(
            "explicit torch.autograd.Function: local, readable, owns the formulas\n"
            "post-autograd FX rewrite: keeps the ATen backward, global policy (TorchTitan prototype)\n"
            "compiled backward through flex_gemm raises instead of returning silent None grads",
            font_size=17,
        ),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    how.next_to(insight, DOWN, buff=0.35).align_to(header, LEFT)
    scene.play(FadeIn(how, shift=UP * 0.1), run_time=0.5)
    scene.wait(0.2)
    scene.next_slide(
        notes="autograd.today — Two integration paths: an explicit autograd.Function, or a post-autograd graph pass that rewrites the traced backward. Either way, nothing silently drops gradients."
    )
