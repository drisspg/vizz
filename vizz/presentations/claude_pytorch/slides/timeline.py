from manim import DOWN, LEFT, UP, Create, Dot, FadeIn, Line, VGroup

from vizz.presentations.claude_pytorch.slides.common import header
from vizz.presentations.components import SlideBase

# research/pytorch_repo.md §6 and research/test_infra.md §7 (git log dates, 2026).
INTERACTIVE = [
    ("Jan 16", "@claude pilot", "20 hand-picked users\n#172686"),
    ("Jan 28", "issue auto-triage", "every new issue\n#173530"),
    ("Feb 27", "write-access gate", "allowlist → permission\n#176027"),
    ("Mar 5", "reusable workflow", "pytorch + meta-pytorch\ntest-infra #7810"),
]
CI = [
    ("Mar 13", "autorevert advisor", "judges suspect commits\n#177404"),
    ("Jun 17", "Dr.CI dispatch", "advisor on PR failures\ntest-infra #8178"),
    ("Jul 31", "public transcripts", "every run archived to S3\ntest-infra #8381"),
    ("Sep 16", "hardened review", "sandboxed, no shell\n#196845"),
]
X0, SPACING = -6.4, 3.05


def _milestones(scene: SlideBase, items, xs, y: float, color: str, above: bool):
    group = VGroup()
    for (date, name, detail), x in zip(items, xs, strict=True):
        dot = Dot([x, y, 0], radius=0.09, color=color)
        label = VGroup(
            scene.meta_text(date, font_size=17, color=color, uppercase=False),
            scene.body_text(name, font_size=22),
            scene.meta_text(detail, font_size=13, uppercase=False),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        max_w = min(SPACING - 0.2, 6.75 - x)
        if label.width > max_w:
            label.scale_to_fit_width(max_w)
        tick = Line([x, y, 0], [x, y + (0.5 if above else -0.5), 0], color=color)
        if above:
            label.next_to(tick, UP, buff=0.1).align_to(dot, LEFT).shift(LEFT * 0.09)
        else:
            label.next_to(tick, DOWN, buff=0.1).align_to(dot, LEFT).shift(LEFT * 0.09)
        group.add(VGroup(tick, dot, label))
    return group


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "Nine months, one bot at a time")
    y = -0.45
    axis = Line([X0 - 0.3, y, 0], [6.7, y, 0], color=t.divider, stroke_width=2)
    xs_top = [X0 + i * SPACING for i in range(4)]
    xs_bottom = [X0 + SPACING / 2 + i * SPACING for i in range(4)]
    top = _milestones(scene, INTERACTIVE, xs_top, y, t.accent_secondary, above=True)
    bottom = _milestones(scene, CI, xs_bottom, y, t.accent_success, above=False)
    top_tag = scene.meta_text("maintainers ask", color=t.accent_secondary)
    top_tag.next_to(head, DOWN, buff=0.3).align_to(head, LEFT)
    bottom_tag = scene.meta_text("CI asks", color=t.accent_success)
    bottom_tag.to_edge(DOWN, buff=0.35).align_to(head, LEFT)

    scene.play(FadeIn(head), Create(axis), run_time=0.5)
    scene.play(FadeIn(top_tag), FadeIn(top, lag_ratio=0.2), run_time=1.0)
    scene.wait(0.2)
    scene.next_slide(
        notes="timeline.interactive — Ivan shipped @claude on Jan 16 for a 20-person pilot. Twelve days later every new issue got triaged. On Feb 27 Ivan replaced the allowlist with a permission check: anyone with write access can type @claude. Then Zain moved it into test-infra so any pytorch or meta-pytorch repo can opt in."
    )
    scene.play(FadeIn(bottom_tag), FadeIn(bottom, lag_ratio=0.2), run_time=1.0)
    scene.wait(0.2)
    scene.next_slide(
        notes="timeline.ci — The second half of the year moved Claude from 'a maintainer asks' to 'CI asks': the autorevert advisor, Dr.CI dispatch on PR failures, public transcripts for every run, and a hardened PR review for untrusted code."
    )
