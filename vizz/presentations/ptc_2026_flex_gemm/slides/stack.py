from itertools import pairwise

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    FadeIn,
    GrowArrow,
    LaggedStart,
    RoundedRectangle,
    VGroup,
)

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import box, kernel
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(
        scene, "Under the hood: trace, analyze, emit into a GEMM template"
    )

    hop = box(
        scene,
        "flex_gemm HOP",
        width=3.0,
        height=1.2,
        sublabel="Dynamo → FX epilogue graph\nclosures lifted + guarded",
    )
    analysis = box(
        scene,
        "Inductor lowering",
        width=3.4,
        height=1.2,
        sublabel="problem → analysis →\nlowering plan → selection",
    )
    template = kernel(
        scene, "GEMM + your epilogue", width=3.0, height=1.2, color=t.accent_success
    )
    pipeline = VGroup(hop, analysis, template).arrange(RIGHT, buff=0.7)
    pipeline.next_to(header, DOWN, buff=0.75)
    steps = VGroup()
    for number, stage, color in zip(
        "123",
        pipeline,
        (t.accent_secondary, t.accent_secondary, t.accent_success),
        strict=True,
    ):
        dot = scene.meta_text(number, font_size=14, color=t.background)
        disc = RoundedRectangle(
            corner_radius=0.17,
            width=0.34,
            height=0.34,
            stroke_width=0,
            fill_color=color,
            fill_opacity=1,
        )
        dot.move_to(disc)
        steps.add(VGroup(disc, dot).move_to(stage.get_corner(UP + LEFT) + LEFT * 0.12))
    for stage in (hop, analysis):
        stage[0].set_stroke(t.accent_secondary, width=1.6)
    template_note = scene.meta_text(
        "mainloop untouched · epilogue at the store", font_size=12
    ).next_to(template, DOWN, buff=0.12)
    arrows = VGroup(
        *[
            Arrow(
                a.get_right(),
                b.get_left(),
                buff=0.08,
                color=t.muted_text,
                stroke_width=3,
            )
            for a, b in pairwise(pipeline)
        ]
    )
    scene.play(
        FadeIn(header),
        FadeIn(pipeline),
        FadeIn(steps),
        FadeIn(template_note),
        *[GrowArrow(a) for a in arrows],
        run_time=0.8,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="stack.pipeline — flex_gemm is a higher-order op like flex_attention. Dynamo traces the epilogue to FX; Inductor builds a representation of the tile program, mostly figuring out which reductions are local; then it emits code into an existing high-performance GEMM template."
    )

    def backend(name: str, status: str, detail: str, color: str) -> VGroup:
        card = box(scene, name, width=3.6, height=1.35, color=color, font_size=24)
        chip_text = scene.meta_text(status, font_size=12, color=t.background)
        chip = RoundedRectangle(
            corner_radius=0.04,
            width=chip_text.width + 0.24,
            height=chip_text.height + 0.14,
            stroke_width=0,
            fill_color=color,
            fill_opacity=1,
        )
        chip_text.move_to(chip)
        badge = (
            VGroup(chip, chip_text)
            .next_to(card, UP, buff=-0.2)
            .align_to(card, LEFT)
            .shift(RIGHT * 0.15)
        )
        info = scene.body_text(detail, font_size=17, color=t.text)
        info.next_to(card, DOWN, buff=0.15)
        return VGroup(card, badge, info)

    backends = VGroup(
        backend(
            "QuACK",
            "shipping",
            "CuTeDSL · EpiMod epilogues\nvendored in PyTorch",
            t.accent_success,
        ),
        backend(
            "NVGEMM",
            "in progress",
            "NVIDIA's CuTeDSL API\nday-0 mainloops for new GPUs",
            t.accent_secondary,
        ),
        backend(
            "Triton",
            "needs love",
            "exists · the path to AMD\nneeds a faster mainloop\nand more robust fusion",
            t.accent_danger,
        ),
    ).arrange(RIGHT, buff=0.45, aligned_edge=UP)
    backends.next_to(pipeline, DOWN, buff=1.0)
    backend_label = scene.meta_text("gemm template backends", color=t.muted_text)
    backend_label.next_to(backends, UP, buff=0.35).align_to(backends, LEFT)
    scene.play(
        FadeIn(backend_label),
        LaggedStart(
            *[FadeIn(card, shift=UP * 0.1) for card in backends], lag_ratio=0.2
        ),
        run_time=0.9,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="stack.backends — QuACK from Tri Dao is the shipping backend and shares code with the FlexAttention/FlashAttention-4 integration. NVGEMM lets NVIDIA supply day-0 mainloops for new hardware. Triton exists and is the natural path to AMD, but it needs a faster mainloop and more robust fusion before it is competitive."
    )
