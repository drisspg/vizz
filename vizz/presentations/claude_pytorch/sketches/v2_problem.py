"""Round-2 problem: PR dots pour through a funnel into monthly columns; first-time authors turn red.

Standalone render:
    uv run manim -qm --media_dir /tmp/v2_problem \
        vizz/presentations/claude_pytorch/sketches/v2_problem.py V2Problem

`build(scene)` is SlideBase-compatible (drop-in for slides/problem.py).
Data (research/hud_usage.md, collected 2026-10-02):
  data/hud/pr_intake_monthly.json        PRs opened per month, pytorch/pytorch
  data/hud/pr_intake_quarter_assoc.json  PRs per quarter with author_association NONE
  data/hud/issue_intake_quarter.json     issues opened per quarter
Colors: muted = PRs, red = first-time (untrusted) authors, muted = issues.
"""

import random

from manim import (
    LEFT,
    RIGHT,
    UP,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    Polygon,
    UpdateFromAlphaFunc,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    linear,
    smooth,
)

from vizz.presentations.claude_pytorch.build import ClaudePytorchDeck
from vizz.presentations.claude_pytorch.slides.common import header
from vizz.presentations.components import SlideBase

PRS_PER_DOT = 50
# fmt: off
MONTHLY_PRS = [
    ("2025-01", 1491), ("2025-02", 1345), ("2025-03", 1432),
    ("2025-04", 1428), ("2025-05", 1481), ("2025-06", 1771),
    ("2025-07", 1654), ("2025-08", 1640), ("2025-09", 1799),
    ("2025-10", 1900), ("2025-11", 1481), ("2025-12", 1811),
    ("2026-01", 1899), ("2026-02", 1682), ("2026-03", 2274),
    ("2026-04", 2362), ("2026-05", 2993), ("2026-06", 2313),
    ("2026-07", 2566), ("2026-08", 3164), ("2026-09", 2979),
]
# fmt: on
# Quarters 2025 Q1 .. 2026 Q3, aligned with MONTHLY_PRS[3q : 3q + 3].
FIRST_TIME_PRS = [190, 226, 195, 374, 636, 643, 1824]

COLS_PER_MONTH = 2
SPACING = 0.11
DOT_RADIUS = 0.04
MONTH_PITCH = 0.56
X0 = -5.6  # left column of the first month
BASELINE_Y = -3.0
COUNTER_Y = 2.3
# Funnel above the chart: wide mouth at the top, narrow neck the dots fall through.
FUNNEL_X = 1.2
MOUTH_Y, NECK_Y = 2.15, 1.15
MOUTH_HALF, NECK_HALF = 1.1, 0.12
LABEL_MONTHS = ("2025-01", "2026-01", "2026-09")


def _column(i: int, count: int) -> list:
    """Row-major slots growing upward, COLS_PER_MONTH wide."""
    x = X0 + MONTH_PITCH * i
    return [
        RIGHT * (x + SPACING * (k % COLS_PER_MONTH))
        + UP * (BASELINE_Y + SPACING * (k // COLS_PER_MONTH))
        for k in range(count)
    ]


def _red_per_month(month_dots: list[int]) -> list[int]:
    """First-time dots per quarter, spread over its three months by PR share."""
    out = []
    for q, first_time in enumerate(FIRST_TIME_PRS):
        months = MONTHLY_PRS[3 * q : 3 * q + 3]
        total = sum(n for _, n in months)
        red_total = round(first_time / PRS_PER_DOT)
        shares = [red_total * n / total for _, n in months]
        reds = [int(s) for s in shares]
        for _ in range(red_total - sum(reds)):
            k = max(range(3), key=lambda j: shares[j] - reds[j])
            reds[k] += 1
        out += reds
    return out


def _pour(dot: Dot, path: VMobject) -> UpdateFromAlphaFunc:
    """Fade the dot in at the funnel mouth while it follows its path into the column."""

    def update(d: Dot, alpha: float) -> None:
        d.move_to(path.point_from_proportion(smooth(alpha)))
        d.set_fill(opacity=0.75 * min(1.0, alpha * 6))

    return UpdateFromAlphaFunc(dot, update)


def build(scene: SlideBase) -> None:
    t = scene.theme
    rng = random.Random(11)
    head = header(scene, "Agents changed the intake")

    counts = [round(n / PRS_PER_DOT) for _, n in MONTHLY_PRS]
    columns = [_column(i, c) for i, c in enumerate(counts)]
    neck = [FUNNEL_X, NECK_Y, 0]
    dots_by_month, paths_by_month = [], []
    for col in columns:
        dots, paths = VGroup(), []
        for p in col:
            start = [FUNNEL_X + rng.uniform(-0.8, 0.8) * MOUTH_HALF, MOUTH_Y + 0.25, 0]
            path = VMobject().set_points_smoothly(
                [
                    start,
                    [FUNNEL_X, (MOUTH_Y + NECK_Y) / 2, 0],
                    neck,
                    [(neck[0] + p[0]) / 2, NECK_Y - 0.6, 0],
                    p,
                ]
            )
            dots.add(Dot(start, radius=DOT_RADIUS, color=t.muted_text, fill_opacity=0))
            paths.append(path)
        dots_by_month.append(dots)
        paths_by_month.append(paths)
    funnel = Polygon(
        [FUNNEL_X - MOUTH_HALF, MOUTH_Y, 0],
        [FUNNEL_X + MOUTH_HALF, MOUTH_Y, 0],
        [FUNNEL_X + NECK_HALF, NECK_Y, 0],
        [FUNNEL_X - NECK_HALF, NECK_Y, 0],
        color=t.divider,
        stroke_width=1.6,
        fill_opacity=0,
    )

    # Beat 1: columns drop in month by month while the counter follows the latest month.
    progress = ValueTracker(0)

    def current_prs() -> int:
        k = min(int(progress.get_value()), len(MONTHLY_PRS) - 1)
        return MONTHLY_PRS[k][1]

    counter_anchor = [X0, COUNTER_Y, 0]
    counter = always_redraw(
        lambda: scene.title_text(f"{current_prs():,}", font_size=48).move_to(
            counter_anchor, aligned_edge=LEFT
        )
    )
    unit = scene.meta_text("PRs opened / month · 1 dot = 50 PRs", font_size=14)
    unit.move_to([X0, COUNTER_Y - 0.5, 0], aligned_edge=LEFT)
    baseline = Line(
        [X0 - 0.3, BASELINE_Y - 0.12, 0],
        [X0 + MONTH_PITCH * 20 + SPACING + 0.3, BASELINE_Y - 0.12, 0],
        color=t.divider,
        stroke_width=1.0,
    )
    month_labels = VGroup()
    for i, (month, _) in enumerate(MONTHLY_PRS):
        if month in LABEL_MONTHS:
            label = scene.meta_text(month, font_size=12, uppercase=False)
            label.move_to([X0 + MONTH_PITCH * i + SPACING / 2, BASELINE_Y - 0.4, 0])
            month_labels.add(label)

    scene.play(
        FadeIn(head),
        FadeIn(unit),
        FadeIn(baseline),
        FadeIn(month_labels),
        FadeIn(funnel),
        run_time=0.4,
    )
    scene.add(counter)
    drops = []
    for group, paths in zip(dots_by_month, paths_by_month, strict=True):
        drops.append(
            LaggedStart(
                *[_pour(d, path) for d, path in zip(group, paths, strict=True)],
                lag_ratio=0.04,
            )
        )
    scene.play(
        LaggedStart(*drops, lag_ratio=0.35),
        progress.animate(rate_func=linear).set_value(len(MONTHLY_PRS) - 0.01),
        run_time=5.0,
    )
    scene.remove(counter)
    final_counter = scene.title_text(f"{MONTHLY_PRS[-1][1]:,}", font_size=48)
    final_counter.move_to(counter_anchor, aligned_edge=LEFT)
    scene.add(final_counter)
    scene.wait(0.2)
    scene.next_slide(
        notes="problem.pile — PRs opened on pytorch/pytorch per month, from HUD's ClickHouse. About 1.5k a month in early 2025; about 3k a month by late 2026. One dot is fifty PRs."
    )

    # Beat 2: the bottom of each column turns red: PRs from authors with no prior association.
    reds = _red_per_month(counts)
    red_dots = []
    for group, r in zip(dots_by_month, reds, strict=True):
        red_dots += list(group[:r])
    first_time = ValueTracker(FIRST_TIME_PRS[0])
    red_counter = always_redraw(
        lambda: scene.title_text(
            f"{FIRST_TIME_PRS[0]:,} → {int(first_time.get_value()):,}", font_size=48
        ).move_to(counter_anchor, aligned_edge=LEFT)
    )
    red_unit = scene.meta_text(
        "first-time-author PRs / quarter", font_size=14, color=t.accent_danger
    )
    red_unit.move_to(unit, aligned_edge=LEFT)
    scene.remove(final_counter)
    scene.add(red_counter)
    scene.play(
        FadeIn(red_unit),
        unit.animate.set_opacity(0),
        funnel.animate.set_stroke(opacity=0.3),
        *[
            d.animate.set_fill(opacity=0.22)
            for group in dots_by_month
            for d in group
            if d not in red_dots
        ],
        LaggedStart(
            *[
                d.animate.set_fill(t.accent_danger, opacity=1).scale(1.5)
                for d in red_dots
            ],
            lag_ratio=0.01,
        ),
        first_time.animate(rate_func=smooth).set_value(FIRST_TIME_PRS[-1]),
        run_time=1.8,
    )
    scene.remove(red_counter)
    final_red = scene.title_text(
        f"{FIRST_TIME_PRS[0]:,} → {FIRST_TIME_PRS[-1]:,}", font_size=48
    )
    final_red.move_to(counter_anchor, aligned_edge=LEFT)
    scene.add(final_red)
    scene.wait(0.2)
    scene.next_slide(
        notes="problem.first_time — Red is PRs from authors with no prior association with the repo: 190 a quarter in early 2025, over 1,800 a quarter now. Roughly ten times."
    )

    # Beat 3: the question.
    question = scene.body_text(
        "What do maintainers need to keep the bar?", font_size=28
    )
    question.move_to([X0 + MONTH_PITCH * 20 + SPACING, 1.3, 0], aligned_edge=RIGHT)
    scene.play(FadeOut(funnel), FadeIn(question, shift=UP * 0.1), run_time=0.5)
    scene.wait(0.2)
    scene.next_slide(
        notes="problem.question — Issues over the same period are flat, around two thousand a quarter. The growth is in PRs, and in who sends them. So: what do maintainers need to keep the bar?"
    )


class V2Problem(SlideBase):
    theme = ClaudePytorchDeck.theme
    deck_mark = ClaudePytorchDeck.deck_mark

    def build_slides(self) -> None:
        build(self)
