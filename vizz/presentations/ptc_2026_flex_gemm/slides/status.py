from manim import DOWN, UP, FadeIn, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header
from vizz.presentations.ptc_2026_flex_gemm.slides.common import punchline

# Landed on pytorch main Aug-Sep 2026 (#192662 ... #196321); verify the nightly
# spelling before 2026-10-13 (see ~/obsidian/flex_gemm/PTC_2026_Before_Conference_TODO.md).
STATUS = [
    "In PyTorch main today: from torch._higher_order_ops import flex_gemm",
    "QuACK GEMMs vendored into PyTorch; Blackwell (B200 / GB200)",
    "Next: scaled grouped GEMM, backward, more mainloops, a blog post",
]


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(scene, "Status and takeaways")
    takeaway = VGroup(
        scene.body_text("Write the epilogue in PyTorch.", font_size=34),
        scene.body_text("Get one kernel, or a clear error.", font_size=34),
        punchline(scene, "The win is the bytes you do not move.", font_size=34),
    ).arrange(DOWN, buff=0.22)
    takeaway.next_to(header, DOWN, buff=0.7).set_x(0)
    status = scene.bullet_list(*STATUS, font_size=22)
    status.next_to(takeaway, DOWN, buff=0.7).set_x(0)
    ask = scene.meta_text(
        "bring me your epilogues  ·  @drisspg", color=t.accent_secondary
    ).to_edge(DOWN, buff=0.45)

    scene.play(
        FadeIn(header), FadeIn(takeaway, shift=UP * 0.1, lag_ratio=0.3), run_time=1.0
    )
    scene.play(FadeIn(status), FadeIn(ask), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="status — Three takeaways. It is in main and the nightlies now, still a private-namespace API. Ask the audience for real epilogues from their models."
    )
