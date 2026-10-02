from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    FadeIn,
    GrowFromEdge,
    Line,
    Rectangle,
    VGroup,
)

from vizz.presentations.claude_pytorch.slides.common import header, punchline
from vizz.presentations.components import SlideBase

# research/hud_usage.md: countDistinct(number) PRs opened on pytorch/pytorch per month
# (default.pull_request via `hud gcx chq`, collected 2026-10-02).
MONTHLY_PRS = [
    ("2025-01", 1491),
    ("2025-02", 1345),
    ("2025-03", 1432),
    ("2025-04", 1428),
    ("2025-05", 1481),
    ("2025-06", 1771),
    ("2025-07", 1654),
    ("2025-08", 1640),
    ("2025-09", 1799),
    ("2025-10", 1900),
    ("2025-11", 1481),
    ("2025-12", 1811),
    ("2026-01", 1899),
    ("2026-02", 1682),
    ("2026-03", 2274),
    ("2026-04", 2362),
    ("2026-05", 2993),
    ("2026-06", 2313),
    ("2026-07", 2566),
    ("2026-08", 3164),
    ("2026-09", 2979),
]
# PRs per quarter whose author_association is NONE (first-time authors).
FIRST_TIME = [("2025 Q1", 190), ("2026 Q3", 1824)]
CHART_W, CHART_H, Y_MAX = 6.6, 2.8, 3500


def _bars(scene: SlideBase) -> VGroup:
    t = scene.theme
    step = CHART_W / len(MONTHLY_PRS)
    baseline = Line([0, 0, 0], [CHART_W, 0, 0], color=t.muted_text)
    bars = VGroup()
    for i, (month, n) in enumerate(MONTHLY_PRS):
        bar = Rectangle(
            width=step * 0.72,
            height=CHART_H * n / Y_MAX,
            stroke_width=0,
            fill_color=t.accent_primary if month < "2026-01" else t.accent_secondary,
            fill_opacity=0.85,
        )
        bar.move_to([step * (i + 0.5), 0, 0], aligned_edge=DOWN)
        bars.add(bar)
    labels = VGroup()
    for i, (month, n) in enumerate(MONTHLY_PRS):
        if month in ("2025-01", "2026-01", "2026-09"):
            labels.add(
                scene.meta_text(month, font_size=13, uppercase=False).next_to(
                    bars[i], DOWN, buff=0.12
                )
            )
    for i in (0, len(MONTHLY_PRS) - 1):
        labels.add(
            scene.body_text(f"{MONTHLY_PRS[i][1]:,}", font_size=18).next_to(
                bars[i], UP, buff=0.08
            )
        )
    return VGroup(baseline, bars, labels)


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "Agents changed the intake")

    chart = _bars(scene)
    axis_label = scene.meta_text("pytorch/pytorch · PRs opened per month")
    chart_block = VGroup(axis_label, chart).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
    chart_block.next_to(head, DOWN, buff=0.4).to_edge(LEFT, buff=0.7)
    baseline, bars, labels = chart

    scene.play(FadeIn(head), FadeIn(axis_label), Create(baseline), run_time=0.5)
    scene.play(*[GrowFromEdge(b, DOWN) for b in bars], FadeIn(labels), run_time=1.0)
    scene.wait(0.2)
    scene.next_slide(
        notes="problem.prs — PRs opened on pytorch/pytorch per month, from HUD's ClickHouse. Roughly 1.5k a month in early 2025, about 3k a month by August 2026. Amber is 2026."
    )

    stat_rows = VGroup()
    for quarter, n in FIRST_TIME:
        stat_rows.add(
            VGroup(
                scene.meta_text(quarter, font_size=15, uppercase=False),
                scene.title_text(f"{n:,}", font_size=44),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        )
    arrow = scene.body_text("→", font_size=40, color=t.muted_text)
    stats = VGroup(stat_rows[0], arrow, stat_rows[1]).arrange(RIGHT, buff=0.35)
    stats_label = scene.meta_text("first-time-author PRs / quarter")
    multiplier = scene.body_text("~10×", font_size=34, color=t.accent_danger)
    issues = scene.body_text("issue intake: flat", font_size=20, color=t.muted_text)
    side = VGroup(stats_label, stats, multiplier, issues).arrange(
        DOWN, aligned_edge=LEFT, buff=0.3
    )
    side.next_to(chart_block, RIGHT, buff=0.9).align_to(chart_block, UP)

    scene.play(FadeIn(side, shift=LEFT * 0.1), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="problem.first_time — The sharpest change is who is sending PRs. PRs from authors with no prior association went from 190 a quarter to over 1,800. Issue volume is flat; the growth is in PRs."
    )

    question = punchline(
        scene,
        "What do maintainers need to keep up\nwithout lowering the bar?",
        font_size=26,
    )
    quote = scene.body_text(
        "“We do not accept contributions created by fully autonomous agents.”"
        "  — PyTorch AI_POLICY.md",
        font_size=17,
        color=t.muted_text,
    )
    bottom = VGroup(question, quote).arrange(DOWN, buff=0.25)
    bottom.to_edge(DOWN, buff=0.35)
    scene.play(FadeIn(bottom, shift=UP * 0.1), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="problem.question — So the question was direct: what tools do maintainers need? The policy side says no fully autonomous agent contributions. The infra side is this talk: bounded agents that work for maintainers."
    )
