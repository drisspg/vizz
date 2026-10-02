from manim import DOWN, LEFT, RIGHT, UP, FadeIn, VGroup

from vizz.presentations.claude_pytorch.slides.common import header
from vizz.presentations.components import SlideBase

# research/hud_usage.md (incident), research/test_infra.md §2 (#198521, #198687),
# research/pytorch_repo.md §1.3 (#176652) and §1.1 (#191969).
LESSONS = [
    (
        "Bots trigger bots",
        "autorevert tags @claude; Dr.CI dispatches the advisor",
        "one allowlisted bot · ≤ 32 runs per PR · skip outages",
    ),
    (
        "Credentials expire",
        "AWS session + App token expire 1 h after job start",
        "55-min jobs · a time-budget hook tells Claude to wrap up",
    ),
    (
        "Some triggers run PR code",
        "pull_request_review_comment runs the PR branch's workflow",
        "trigger removed in #176652",
    ),
    (
        "Trust needs receipts",
        "maintainers want to see what the agent did",
        "public transcripts in S3 · bot-triaged on every edit",
    ),
]


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "With great power …")
    rows = VGroup()
    for title, what, fix in LESSONS:
        rows.add(
            VGroup(
                scene.body_text(title, font_size=27).set_color(t.accent_danger),
                VGroup(
                    scene.body_text(what, font_size=21),
                    scene.body_text(f"→ {fix}", font_size=21, color=t.accent_success),
                ).arrange(DOWN, aligned_edge=LEFT, buff=0.08),
            )
        )
    for row in rows:
        row[1].next_to(row[0], RIGHT, buff=0.4)
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.55)
    col_x = max(r[0].get_right()[0] for r in rows) + 0.4
    for row in rows:
        row[1].align_to([col_x, 0, 0], LEFT)
    rows.next_to(head, DOWN, buff=0.5).to_edge(LEFT, buff=0.7)

    scene.play(FadeIn(head), run_time=0.4)
    for i, row in enumerate(rows):
        scene.play(FadeIn(row, shift=UP * 0.1), run_time=0.5)
        if i in (1, 3):
            scene.wait(0.2)
            scene.next_slide(
                notes=(
                    "lessons.loops — Bots trigger bots: autorevert tags @claude on its own revert comments, and Dr.CI dispatches the advisor. Only one bot passes the @claude gate, and Dr.CI caps advisor runs per PR and skips during outages. Second: credentials last one hour and are never refreshed, so jobs stop at 55 minutes and a hook tells Claude when to converge and post."
                    if i == 1
                    else "lessons.receipts — Some GitHub triggers run code from the PR branch; we removed pull_request_review_comment because tricking a maintainer into running it was easier than prompt injection. And trust needs receipts: every transcript is public, every bot edit is labeled."
                )
            )
