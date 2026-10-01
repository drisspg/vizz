from manim import DOWN, LEFT, RIGHT, UP, FadeIn, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header
from vizz.presentations.ptc_2026_flex_gemm.slides.common import kernel

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

    def foundation(name: str, color: str, lines: str, tag: str) -> VGroup:
        card = kernel(scene, name, width=5.6, height=1.1, color=color, font_size=26)
        label = scene.meta_text(tag, font_size=13, color=color)
        body = scene.body_text(lines, font_size=20)
        return VGroup(label, card, body).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

    quack = foundation(
        "QuACK · EpiMod",
        t.accent_success,
        "QuACK ships an extensive epilogue-fusion API, EpiMod.\n"
        "FlexGEMM lowers your PyTorch epilogue into it:\n"
        "most of FlexGEMM is that lowering.",
        "built on",
    )
    nvgemm = foundation(
        "NVGEMM",
        t.accent_secondary,
        "NVIDIA's CuTeDSL operator API.\n"
        "Integration started: the same epilogue plan,\n"
        "vendor-provided mainloops.",
        "now integrating",
    )
    foundations = VGroup(quack, nvgemm).arrange(RIGHT, buff=0.6, aligned_edge=UP)
    foundations.next_to(header, DOWN, buff=0.45).align_to(header, LEFT)

    unsolved_head = scene.meta_text("could not (yet)", color=t.accent_danger)
    unsolved = VGroup()
    for key, text in UNSOLVED:
        unsolved.add(
            VGroup(
                scene.meta_text(key, font_size=15, color=t.accent_danger),
                scene.body_text(text, font_size=19),
            ).arrange(RIGHT, buff=0.35)
        )
    unsolved.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    block = VGroup(unsolved_head, unsolved).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    block.next_to(foundations, DOWN, buff=0.55).align_to(header, LEFT)

    scene.play(
        FadeIn(header),
        FadeIn(foundations, shift=RIGHT * 0.1, lag_ratio=0.3),
        run_time=1.0,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="how.built — QuACK from Tri Dao already has an extensive epilogue-fusion API called EpiMod, and we built most of FlexGEMM on it: the work is lowering an arbitrary PyTorch epilogue into EpiMod. We have also started integrating NVIDIA's CuTeDSL operator API, NVGEMM, so the same epilogue plan can ride on vendor mainloops."
    )

    scene.play(FadeIn(block, shift=RIGHT * 0.1), run_time=0.8)
    scene.wait(0.2)
    scene.next_slide(
        notes="how.unsolved — What we could not do: split-K with a custom epilogue does not exist in QuACK, so skinny GEMMs like an MoE router lose to cuBLAS. Full-row norms do not fit a tile. And there is no automatic backward yet; we made compiled backward raise rather than return silent None gradients."
    )
