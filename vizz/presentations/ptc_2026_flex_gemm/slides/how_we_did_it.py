from manim import LEFT, RIGHT, FadeIn, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header

# Hard part -> approach -> landed pytorch/pytorch PRs (main, Aug-Sep 2026).
SOLVED = [
    (
        "reductions in the tile",
        "N-group reductions; blocked / transposed outputs",
        "#188739 #191026 #192661",
    ),
    (
        "shape-changing outputs",
        "SwiGLU M×2N → M×N; packed NVFP4 outputs",
        "#190158 #191270",
    ),
    ("low-precision inputs", "block-scaled MXFP8 / NVFP4 mainloops", "#192839"),
    ("mixture of experts", "varlen-M grouped GEMM + grouped SwiGLU", "#196318 #196321"),
    ("picking a config", "Inductor-owned QuACK autotuning", "#196174"),
]
# Source: ~/agent_notes/findings/flex_gemm_dsv3_first_user_study.md (split-K),
# ~/agent_notes/findings/assets/flex_gemm_audit_20260907/feature_gaps.md.
UNSOLVED = [
    (
        "split-K",
        "no split-K with custom epilogues: skinny router 1.63× slower than cuBLAS",
    ),
    (
        "full-row reductions",
        "whole-row RMSNorm / LayerNorm: rejected past 512 columns",
    ),
    (
        "autograd",
        "forward-only; compiled backward raises, never silent None grads",
    ),
]


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(scene, "How we built it, and what we could not do")

    def table(items: list[tuple[str, ...]], color: str, top: float) -> VGroup:
        rows = VGroup()
        for index, (key, text, *tag) in enumerate(items):
            y = top - index * 0.5
            r = VGroup(
                scene.meta_text(key, font_size=16, color=color).move_to(
                    [-6.2, y, 0], aligned_edge=LEFT
                ),
                scene.body_text(text, font_size=20).move_to(
                    [-2.6, y, 0], aligned_edge=LEFT
                ),
            )
            if tag:
                r.add(
                    scene.meta_text(tag[0], font_size=13, uppercase=False).move_to(
                        [4.5, y, 0], aligned_edge=LEFT
                    )
                )
            rows.add(r)
        return rows

    solved = table(SOLVED, t.accent_success, 1.75)
    unsolved_head = scene.meta_text("could not (yet)", color=t.accent_danger)
    unsolved_head.move_to([-6.2, -1.0, 0], aligned_edge=LEFT)
    unsolved = table(UNSOLVED, t.accent_danger, -1.55)

    scene.play(
        FadeIn(header), FadeIn(solved, shift=RIGHT * 0.1, lag_ratio=0.15), run_time=1.2
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="how.solved — Each hard part was its own small landed PR: tile-local reductions and their output layouts, shape-changing outputs for SwiGLU and packed fp4, block-scaled inputs, MoE grouped GEMM, and letting Inductor own the autotuning."
    )

    scene.play(
        FadeIn(unsolved_head),
        FadeIn(unsolved, shift=RIGHT * 0.1, lag_ratio=0.15),
        run_time=0.9,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="how.unsolved — What we could not do: split-K with a custom epilogue does not exist in QuACK, so skinny GEMMs like an MoE router lose to cuBLAS. Full-row norms do not fit a tile. And there is no automatic backward yet; we made compiled backward raise rather than return silent None gradients."
    )
