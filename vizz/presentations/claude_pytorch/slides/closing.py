from manim import DOWN, LEFT, RIGHT, UP, FadeIn

from vizz.presentations.claude_pytorch.slides.common import (
    code_block,
    header,
    punchline,
)
from vizz.presentations.components import SlideBase

USAGE = """
@claude review this PR
@claude /pr-review
@claude fable effort=high why does this test fail on ROCm?
"""


def build(scene: SlideBase) -> None:
    head = header(scene, "Agent-shaped infra for an agent-shaped world")
    points = scene.bullet_list(
        "Scoped: smallest tools; untrusted input split from write tokens",
        "Auditable: public transcripts, usage rows, labeled bot edits",
        "Repo-aware: skills and CLAUDE.md carry the maintainers' bar",
        font_size=26,
    )
    points.next_to(head, DOWN, buff=0.5).to_edge(LEFT, buff=0.9)
    code = code_block(scene, "try it (needs write access)", USAGE, font_size=20)
    code.next_to(points, DOWN, buff=0.5).align_to(points, LEFT)
    final = punchline(
        scene, "Not replacing maintainers. Keeping the bar.", font_size=28
    )
    final.next_to(code, RIGHT, buff=0.6).align_to(code, DOWN)
    if final.get_right()[0] > 6.7:
        final.next_to(code, DOWN, buff=0.4).align_to(code, LEFT)
    thanks = (
        scene.meta_text(
            "thanks: Zain Rizvi · Jean Schmidt · Alban Desmaison · Nikita Shulga · Anshul Sinha · Richard Zou · Andrey Talman",
            font_size=13,
            uppercase=False,
        )
        .to_edge(DOWN, buff=0.3)
        .align_to(points, LEFT)
    )

    scene.play(FadeIn(head), FadeIn(points, lag_ratio=0.2), run_time=0.8)
    scene.play(FadeIn(code, shift=UP * 0.1), run_time=0.6)
    scene.play(FadeIn(thanks), FadeIn(final), run_time=0.5)
    scene.wait(0.2)
    scene.next_slide(
        notes="closing — Three properties: scoped, auditable, repo-aware. If you have write access, try @claude today, including /pr-review and the fable and effort knobs. If you own a pytorch or meta-pytorch repo, onboarding is one script. The goal is not to replace maintainers; it is to keep the bar while the volume doubles."
    )
