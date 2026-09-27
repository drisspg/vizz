from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    DashedLine,
    FadeIn,
    GrowFromEdge,
    Rectangle,
    SurroundingRectangle,
    VGroup,
)

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header

# Source: ~/meta/my_scripts/misc/FLEX_GEMM_TALK.ipynb baked outputs, NVIDIA B200,
# PyTorch 8f61c19 (2026-08-11). Baseline: torch.compile(fullgraph=True) of the
# same unfused program. Timing: fixed-pointer CUDA-graph replay, median of
# alternating rounds. Re-bake on the landing stack before the talk.
CASES = [
    ("bias + ReLU", "bf16  2048³", "24.1 → 20.9 µs", 1.15),
    (
        "GEMM → NVFP4 activation + blocked scales",
        "16384 × 4096 × 4096",
        "560 → 369 µs",
        1.52,
    ),
    (
        "MXFP8 MLP: GEMM → ReLU → MXFP8 → GEMM",
        "2048 · 4096 → 16384 → 4096",
        "256 → 239 µs",
        1.07,
    ),
]
BAR_SCALE = 2.6  # scene units per 1.0x speedup


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(scene, "Measured on B200 vs. torch.compile")

    rows = VGroup()
    for name, shape, times, speedup in CASES:
        label = VGroup(
            scene.body_text(name, font_size=21),
            scene.meta_text(f"{shape}  ·  {times}", font_size=14, uppercase=False),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        bar = Rectangle(
            width=speedup * BAR_SCALE,
            height=0.46,
            stroke_width=0,
            fill_color=t.accent_primary,
            fill_opacity=0.85,
        )
        value = scene.body_text(f"{speedup:.2f}×", font_size=24)
        rows.add(VGroup(label, bar, value))

    left_x = -6.3
    bar_x = left_x + max(row[0].width for row in rows) + 0.35
    for index, (label, bar, value) in enumerate(rows):
        y = 1.3 - index * 1.05
        label.move_to([left_x, y, 0], aligned_edge=LEFT)
        bar.move_to([bar_x, y, 0], aligned_edge=LEFT)
        value.next_to(bar, RIGHT, buff=0.2)
    speed_head = scene.meta_text("speedup ↑", font_size=13).move_to(
        [bar_x, 2.05, 0], aligned_edge=LEFT
    )
    baseline_x = bar_x + BAR_SCALE
    baseline = DashedLine(
        [baseline_x, rows.get_top()[1] + 0.2, 0],
        [baseline_x, rows.get_bottom()[1] - 0.2, 0],
        color=t.muted_text,
        stroke_width=2,
        dash_length=0.03,
        dashed_ratio=0.35,
    )
    baseline_label = scene.meta_text("torch.compile = 1.0×", font_size=13).next_to(
        baseline, UP, buff=0.08
    )
    footer = scene.meta_text(
        "B200 · CUDA-graph replay, median of alternating rounds · PyTorch 8f61c19 (Aug 2026)",
        font_size=13,
        uppercase=False,
    ).to_edge(DOWN, buff=0.3)

    scene.play(
        FadeIn(header),
        FadeIn(footer),
        Create(baseline),
        FadeIn(baseline_label),
        FadeIn(speed_head),
    )
    scene.play(
        *[FadeIn(row[0]) for row in rows],
        *[GrowFromEdge(row[1], LEFT) for row in rows],
        *[FadeIn(row[2]) for row in rows],
        run_time=0.9,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="results.speedups — Bias+ReLU is modest. The low-precision producer is the big one: the fused kernel writes 4-bit data plus scales instead of a bf16 accumulator, so the removed traffic is large. The full MXFP8 MLP is 1.07x end to end for the pair of GEMMs."
    )

    mx_row = rows[2]
    focus = SurroundingRectangle(
        mx_row, color=t.accent_secondary, buff=0.12, corner_radius=0.04
    )
    numerics = VGroup(
        scene.meta_text(
            "numerics vs FP32-accumulator reference", color=t.accent_secondary
        ),
        scene.body_text(
            "compiled PyTorch: 10.8% bit-exact    ·    FlexGEMM: 100% bit-exact",
            font_size=22,
        ),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    numerics.next_to(rows, DOWN, buff=0.55).align_to(rows, LEFT)
    scene.play(Create(focus), FadeIn(numerics, shift=UP * 0.1), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="results.numerics — The more interesting MXFP8 result is numerics: the scales are computed from the fp32 accumulator in registers, not from a rounded bf16 intermediate. That is only safe to promise because fusion is guaranteed or the compile fails."
    )

    swiglu = VGroup(
        scene.meta_text("honest caveat", color=t.accent_danger),
        scene.body_text(
            "SwiGLU MLP (Qwen3-8B shapes): ~1.10× in isolation, no end-to-end training win yet",
            font_size=21,
        ),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    swiglu.next_to(numerics, DOWN, buff=0.35).align_to(rows, LEFT)
    scene.play(FadeIn(swiglu, shift=UP * 0.1), run_time=0.5)
    scene.wait(0.2)
    scene.next_slide(
        notes="results.caveat — In TorchTitan the SwiGLU fusion wins as a microbenchmark but not end to end: Amdahl plus other effects we are still debugging."
    )
