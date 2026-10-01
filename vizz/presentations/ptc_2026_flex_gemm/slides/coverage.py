from manim import DOWN, LEFT, RIGHT, Create, FadeIn, Line, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header

# Source: torch/_higher_order_ops/flex_gemm.py on pytorch main (2026-09-23):
# FLEX_GEMM_OP_SPECS, flex_gemm_scaled_mm, flex_gemm_grouped_mm.
SUPPORTED = [
    ("torch.mm  ·  torch.addmm", "[M,K] @ [K,N]", "dense linear layers"),
    ("torch.bmm  ·  torch.baddbmm", "[B,M,K] @ [B,K,N]", "batched / per-head"),
    ("F.scaled_mm", "MXFP8 · NVFP4 block-scaled", "low-precision dense matmuls"),
    ("F.grouped_mm", "[ΣM,K] @ [E,K,N] + offs", "MoE experts (varlen-M)"),
]
NOT_YET = [
    ("F.scaled_grouped_mm", "prototyped, not landed"),
    ("grouped_mm 2-D × 2-D", "MoE weight gradient"),
]


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(scene, "One epilogue API over all the GEMMs")

    def row(api: str, shape: str, use: str, color: str) -> VGroup:
        return VGroup(
            scene.meta_text(api, font_size=20, color=t.text, uppercase=False),
            scene.meta_text(shape, font_size=17, color=t.text, uppercase=False),
            scene.body_text(use, font_size=20),
        )

    rows = VGroup(*[row(*item, t.accent_primary) for item in SUPPORTED])
    col_x = (-6.2, -1.9, 2.7)
    for index, r in enumerate(rows):
        y = 1.45 - index * 0.62
        for cell, x in zip(r, col_x, strict=True):
            cell.move_to([x, y, 0], aligned_edge=LEFT)
    heads = VGroup(
        *[
            scene.meta_text(text, font_size=14).move_to([x, 2.05, 0], aligned_edge=LEFT)
            for text, x in zip(("gemm", "operands", "used for"), col_x, strict=True)
        ]
    )
    rule = Line([-6.2, 1.8, 0], [6.3, 1.8, 0], color=t.divider, stroke_width=1.2)
    footer = (
        scene.body_text(
            "Same epilogue function, same semantics, all lowered to fused kernels",
            font_size=22,
            color=t.text,
        )
        .next_to(rows, DOWN, buff=0.45)
        .align_to(rows, LEFT)
    )

    scene.play(FadeIn(header), FadeIn(heads), Create(rule))
    scene.play(
        FadeIn(rows, shift=RIGHT * 0.1, lag_ratio=0.2), FadeIn(footer), run_time=1.0
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="coverage.supported — You do not rewrite your model around a new GEMM. flex_gemm takes the op you already call: mm, addmm, bmm, baddbmm, F.scaled_mm for MXFP8 and NVFP4, and F.grouped_mm for MoE experts. All of these are in PyTorch main."
    )

    missing_head = scene.meta_text("not yet")
    missing = VGroup(
        *[
            VGroup(
                scene.meta_text(api, font_size=17, color=t.text, uppercase=False),
                scene.body_text(why, font_size=19, color=t.muted_text),
            ).arrange(RIGHT, buff=0.35)
            for api, why in NOT_YET
        ]
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
    block = VGroup(missing_head, missing).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
    block.next_to(footer, DOWN, buff=0.45).align_to(rows, LEFT)
    scene.play(FadeIn(block, shift=RIGHT * 0.1), run_time=0.5)
    scene.wait(0.2)
    scene.next_slide(
        notes="coverage.gaps — Two gaps: scaled grouped GEMM is prototyped and bit-exact but not landed, and the MoE weight-gradient form of grouped_mm is not supported yet."
    )
