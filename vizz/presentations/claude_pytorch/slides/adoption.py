import csv
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    DashedLine,
    FadeIn,
    GrowFromEdge,
    Line,
    Rectangle,
    VGroup,
)

from vizz.presentations.claude_pytorch.slides.common import header
from vizz.presentations.components import SlideBase

# HUD ClickHouse misc.claude_code_usage, pytorch/pytorch, human issue_comment runs per
# ISO week; collected 2026-10-02. See research/hud_usage.md. The last (partial) week is dropped.
WEEKLY = Path(__file__).parents[1] / "research" / "data" / "hud" / "weekly_human.csv"
OPENED_WEEK = "2026-02-23"  # #176027 merged 2026-02-27: allowlist -> write access
STATS = [
    ("135", "people invoked @claude"),
    ("8.7k", "runs on 4.1k PRs / issues"),
    ("4.4k", "issues auto-triaged, 1,375 reporters"),
    ("13", "repos reporting usage"),
]
# Share of pytorch/pytorch runs by category (research/hud_usage.md).
SPLIT = [
    ("CI advisors (autorevert + Dr.CI)", 0.81, "accent_secondary"),
    ("humans @claude", 0.11, "accent_success"),
    ("triage + other", 0.08, "divider"),
]
CHART_W, CHART_H = 7.4, 3.2


def _weekly() -> list[tuple[str, int, int]]:
    with WEEKLY.open() as f:
        rows = [(r["w"], int(r["n"]), int(r["actors"])) for r in csv.DictReader(f)]
    return rows[:-1]


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "Usage after launch")
    rows = _weekly()
    y_max = max(n for _, n, _ in rows) * 1.1
    step = CHART_W / len(rows)
    baseline = Line([0, 0, 0], [CHART_W, 0, 0], color=t.muted_text)
    bars = VGroup()
    for i, (_, n, _) in enumerate(rows):
        bar = Rectangle(
            width=step * 0.7,
            height=max(CHART_H * n / y_max, 0.01),
            stroke_width=0,
            fill_color=t.accent_secondary,
            fill_opacity=0.85,
        )
        bar.move_to([step * (i + 0.5), 0, 0], aligned_edge=DOWN)
        bars.add(bar)
    weeks = [w for w, _, _ in rows]
    marker_x = step * weeks.index(OPENED_WEEK) + step
    marker = DashedLine(
        [marker_x, 0, 0],
        [marker_x, CHART_H, 0],
        color=t.accent_success,
        dash_length=0.06,
    )
    marker_label = scene.meta_text(
        "write access\n#176027", font_size=12, color=t.accent_success
    )
    marker_label.next_to(marker, RIGHT, buff=0.08).align_to(marker, UP)
    peak_i = max(range(len(rows)), key=lambda i: rows[i][1])
    peak = scene.body_text(f"{rows[peak_i][1]} / week", font_size=16).next_to(
        bars[peak_i], UP, buff=0.06
    )
    ticks = VGroup(
        *[
            scene.meta_text(weeks[i][5:], font_size=12, uppercase=False).next_to(
                bars[i], DOWN, buff=0.1
            )
            for i in (0, len(rows) // 2, len(rows) - 1)
        ]
    )
    label = scene.meta_text("human @claude runs per week · pytorch/pytorch · 2026")
    chart = VGroup(baseline, bars, marker, marker_label, peak, ticks)
    block = VGroup(label, chart).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
    block.next_to(head, DOWN, buff=0.45).to_edge(LEFT, buff=0.7)

    stats = VGroup()
    for big, small in STATS:
        stats.add(
            VGroup(
                scene.title_text(big, font_size=36),
                scene.body_text(small, font_size=17, color=t.muted_text),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
        )
    stats.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    stats.next_to(block, RIGHT, buff=0.7).align_to(block, UP)

    split_bar = VGroup()
    total_w = 12.2
    for name, share, color_key in SPLIT:
        seg = Rectangle(
            width=total_w * share,
            height=0.42,
            stroke_width=0,
            fill_color=getattr(t, color_key),
            fill_opacity=0.85,
        )
        split_bar.add(seg)
    split_bar.arrange(RIGHT, buff=0.02).to_edge(DOWN, buff=0.75).set_x(0)
    split_labels = VGroup(
        *[
            scene.body_text(f"{name} {share:.0%}", font_size=15)
            .next_to(seg, DOWN, buff=0.08)
            .align_to(seg, LEFT)
            for (name, share, _), seg in zip(SPLIT, split_bar, strict=True)
        ]
    )
    split_labels[2].align_to(split_bar, RIGHT)
    split_labels[1].next_to(split_labels[0], RIGHT, buff=0.4)
    split_title = scene.meta_text(
        "all 76.8k Claude runs on pytorch/pytorch", font_size=13
    )
    split_title.next_to(split_bar, UP, buff=0.1).align_to(split_bar, LEFT)

    scene.play(FadeIn(head), FadeIn(label), Create(baseline), run_time=0.5)
    scene.play(*[GrowFromEdge(b, DOWN) for b in bars], FadeIn(ticks), run_time=1.0)
    scene.play(Create(marker), FadeIn(marker_label), FadeIn(peak), run_time=0.5)
    scene.wait(0.2)
    scene.next_slide(
        notes="adoption.weekly — Human @claude invocations per week on pytorch/pytorch, from HUD's usage table. A trickle during the pilot; it picks up after Feb 27 when anyone with write access could use it, peaks around 900 a week in early June, and settles at a few hundred a week, with 25 to 40 distinct people every week."
    )
    scene.play(FadeIn(stats, lag_ratio=0.15), run_time=0.8)
    scene.wait(0.2)
    scene.next_slide(
        notes="adoption.stats — Since Feb 28: 135 people, 8.7k runs on 4.1k distinct PRs and issues. Triage has touched 4.4k issues from 1,375 different reporters. Thirteen repos report usage."
    )
    scene.play(
        FadeIn(split_title), FadeIn(split_bar), FadeIn(split_labels), run_time=0.6
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="adoption.split — The surprise: most Claude runs are not humans at all. CI advisors dispatched by autorevert, Dr.CI, and the coverage backfill are 81% of runs, at about a minute each; the August backfill alone was about 22 thousand. Human sessions are 11% of runs but much longer: median three minutes, p90 almost half an hour."
    )
