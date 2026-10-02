from manim import DOWN, LEFT, RIGHT, UP, Create, FadeIn, Line, VGroup

from vizz.presentations.claude_pytorch.slides.common import box, flow
from vizz.presentations.components import SlideBase

SOURCES = ("contributors + agents", "bots + agents", "everyone + agents")


def build(scene: SlideBase) -> None:
    t = scene.theme
    kicker = scene.meta_text("PyTorch Conference NA 2026 · lightning talk")
    title = scene.title_text("Fighting Agents\nwith Agents", font_size=60)
    subtitle = scene.body_text(
        "Bringing Claude to PyTorch CI,\ntriage, and PR review",
        font_size=28,
        color=t.muted_text,
    )
    rule = Line(LEFT, RIGHT, color=t.divider, stroke_width=1.0)
    speakers = scene.body_text(
        "Driss Guessous · Ivan Zaitsev\nPyTorch @ Meta", font_size=22
    )
    text = VGroup(kicker, title, subtitle, rule, speakers).arrange(
        DOWN, aligned_edge=LEFT, buff=0.3
    )
    rule.put_start_and_end_on(rule.get_left(), rule.get_left() + RIGHT * text.width)
    text.to_edge(LEFT, buff=0.7)

    sources = VGroup(
        *[box(scene, s, width=3.1, height=0.62, font_size=19) for s in SOURCES]
    ).arrange(DOWN, buff=0.45)
    maintainers = box(
        scene, "maintainers", width=2.2, height=1.0, color=t.accent_success
    )
    diagram = VGroup(sources, maintainers).arrange(RIGHT, buff=1.3)
    diagram.to_edge(RIGHT, buff=0.5).shift(UP * 0.2)
    inflow = VGroup(
        *[
            flow(scene, s.get_right(), maintainers.get_left(), color=t.accent_danger)
            for s in sources
        ]
    )
    volume = scene.meta_text(
        "more PRs · more issues · same reviewers", color=t.accent_danger
    ).next_to(diagram, DOWN, buff=0.45)

    scene.play(FadeIn(text, shift=UP * 0.1), run_time=0.7)
    scene.play(FadeIn(sources), FadeIn(maintainers), run_time=0.5)
    scene.play(*[Create(a) for a in inflow], FadeIn(volume), run_time=0.7)
    scene.wait(0.2)
    scene.next_slide(
        notes="title — Contributors use agents. Bots use agents. Everyone uses agents. All of it lands on the same small set of maintainers. This talk is about the tools we built so maintainers can keep up without lowering the bar."
    )
