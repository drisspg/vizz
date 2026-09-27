from itertools import pairwise

from manim import DOWN, LEFT, RIGHT, UP, Arrow, FadeIn, GrowArrow, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import box
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(scene, "Under the hood: a HOP, an analysis, a template")

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
    template = box(
        scene,
        "GEMM template",
        width=3.0,
        height=1.2,
        sublabel="mainloop stays untouched;\nepilogue inlined at the store",
    )
    pipeline = VGroup(hop, analysis, template).arrange(RIGHT, buff=0.7)
    pipeline.next_to(header, DOWN, buff=0.6)
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
        FadeIn(header), FadeIn(pipeline), *[GrowArrow(a) for a in arrows], run_time=0.8
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="stack.pipeline — flex_gemm is a higher-order op like flex_attention. Dynamo traces the epilogue to FX; Inductor builds a representation of the tile program, mostly figuring out which reductions are local; then it emits code into an existing high-performance GEMM template."
    )

    backends = VGroup(
        box(
            scene,
            "QuACK (CuTeDSL)",
            width=3.3,
            sublabel="primary · EpiMod epilogues · vendored",
            color=t.accent_success,
        ),
        box(
            scene,
            "NVGEMM",
            width=3.3,
            sublabel="in progress · NVIDIA CuTeDSL API",
        ),
        box(
            scene,
            "Triton",
            width=3.3,
            sublabel="exists · needs love (AMD?)",
        ),
    ).arrange(RIGHT, buff=0.4)
    backends.next_to(pipeline, DOWN, buff=0.9)
    backend_label = scene.meta_text("gemm template backends", color=t.muted_text)
    backend_label.next_to(backends, UP, buff=0.25).align_to(backends, LEFT)
    escape = scene.body_text(
        "Escape hatch: inline_asm_elementwise lowers into the epilogue — e.g. the E8M0 scale cvt instruction",
        font_size=20,
        color=t.muted_text,
    ).next_to(backends, DOWN, buff=0.55)
    scene.play(FadeIn(backends), FadeIn(backend_label), FadeIn(escape), run_time=0.7)
    scene.wait(0.2)
    scene.next_slide(
        notes="stack.backends — QuACK from Tri Dao is the main backend today and shares code with the FlexAttention/FlashAttention-4 integration. NVGEMM lets vendors supply day-0 mainloops for new hardware. Inline PTX lets users, and their agents, hill-climb without waiting on us."
    )
