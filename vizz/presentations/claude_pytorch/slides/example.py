from manim import DOWN, LEFT, RIGHT, UP, FadeIn, VGroup

from vizz.presentations.claude_pytorch.slides.common import comment, header
from vizz.presentations.components import SlideBase

# research/pytorch_repo.md §7.2: pytorch/pytorch#176266 (external contributor,
# "Add scaled_mm_v2 cpu implementation"), run 22680756884, landed as d9e65e8cfed.
FINDINGS = [
    (
        "High",
        "compile error: MKLDNN branch uses undefined mat_a / mat_b",
    ),
    (
        "Medium",
        "copy-paste bug: scale_b_opt built from scale_a",
    ),
    (
        "Low",
        "*out_dtype dereferenced without a nullopt check",
    ),
]


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "@claude on a real PR")
    pr = (
        scene.meta_text(
            "pytorch/pytorch#176266 · external contributor · scaled_mm_v2 on CPU",
            uppercase=False,
        )
        .next_to(head, DOWN, buff=0.3)
        .align_to(head, LEFT)
    )

    ask = comment(
        scene,
        "drisspg",
        VGroup(
            scene.body_text("@claude look for any subtle bugs on this pr", font_size=28)
        ),
        color=t.accent_success,
        width=8.0,
        meta="maintainer",
    )
    ask.next_to(pr, DOWN, buff=0.45).align_to(head, LEFT)

    rows = VGroup()
    for severity, text in FINDINGS:
        color = t.accent_danger if severity != "Low" else t.muted_text
        rows.add(
            VGroup(
                scene.meta_text(severity, font_size=17, color=color),
                scene.body_text(text, font_size=24),
            ).arrange(RIGHT, buff=0.25)
        )
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.24)
    text_x = max(row[0].get_right()[0] for row in rows) + 0.25
    for row in rows:
        row[1].align_to([text_x, 0, 0], LEFT)
    reply = comment(
        scene,
        "claude[bot]",
        rows,
        color=t.accent_secondary,
        width=9.6,
        meta="4m 39s",
    )
    reply.next_to(ask, DOWN, buff=0.4).align_to(ask, LEFT).shift(RIGHT * 0.8)

    outcome = VGroup(
        scene.body_text(
            "✓ High + Medium fixed before landing", font_size=24, color=t.accent_success
        ),
        scene.body_text(
            "humans debated the Low one and chose value_or(...) · landed d9e65e8",
            font_size=18,
            color=t.muted_text,
        ),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    outcome.next_to(reply, DOWN, buff=0.4).align_to(reply, LEFT).shift(RIGHT * 0.25)

    scene.play(FadeIn(head), FadeIn(pr), FadeIn(ask, shift=UP * 0.1), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="example.ask — An external contributor's FP8 PR adding a CPU path for scaled_mm. I asked @claude to look for subtle bugs. This is the whole interface: a comment."
    )
    scene.play(FadeIn(reply, shift=UP * 0.1), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="example.reply — Four and a half minutes later: a compile error in the MKLDNN branch that only some platforms build, and a copy-paste bug where scale_b came from scale_a. Both real."
    )
    scene.play(FadeIn(outcome, shift=UP * 0.1), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="example.outcome — Both were fixed before landing. The low-severity note got pushed back on by humans, who picked their own fix. That is the shape we want: Claude finds things, maintainers decide."
    )
