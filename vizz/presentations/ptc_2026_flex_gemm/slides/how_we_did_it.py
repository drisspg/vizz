from manim import DOWN, LEFT, RIGHT, UP, FadeIn, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header
from vizz.presentations.ptc_2026_flex_gemm.slides.common import kernel


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(scene, "How we built it")

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
    foundations.move_to([0, -0.4, 0])

    scene.play(
        FadeIn(header),
        FadeIn(foundations, shift=RIGHT * 0.1, lag_ratio=0.3),
        run_time=1.0,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="how.built — QuACK from Tri Dao already has an extensive epilogue-fusion API called EpiMod, and we built most of FlexGEMM on it: the work is lowering an arbitrary PyTorch epilogue into EpiMod. We have also started integrating NVIDIA's CuTeDSL operator API, NVGEMM, so the same epilogue plan can ride on vendor mainloops."
    )
