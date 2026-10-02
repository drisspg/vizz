"""Round-2 sketch: a CI failure orbits into the Claude advisor and lands in a verdict bin;
then one week of advisor runs shoots up during the Aug 19-21 coverage backfill.

Render: uv run manim -qm --media_dir /tmp/v2_agents vizz/presentations/claude_pytorch/sketches/v2_agents.py V2Agents

`build(scene)` is SlideBase-compatible (drop-in for slides/agents_vs_agents.py).
Facts: research/test_infra.md §4 (dispatch, verdict enum, ClickHouse read-back),
research/hud_usage.md (Aug 19-21: 4,912 / 6,179 / 10,930 runs, never reverts),
research/data/hud/weekly_by_category.csv (autorevert_advisor weekly runs, Jul 6 - Sep 28).
"""

import random

import numpy as np
from manim import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    ApplyFunction,
    ArcBetweenPoints,
    Circle,
    Create,
    DashedVMobject,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    MoveAlongPath,
    Rectangle,
    RoundedRectangle,
    Succession,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
    rate_functions,
)

from vizz.presentations.claude_pytorch.build import ClaudePytorchDeck
from vizz.presentations.claude_pytorch.slides.common import CORNER, header, tint
from vizz.presentations.components import SlideBase

VERDICTS = ["related", "unsure", "not_related", "infra_issue", "garbage"]
# autorevert_advisor runs per week starting 2026-07-06 (weekly_by_category.csv).
WEEKLY = [233, 339, 194, 202, 96, 141, 22_054, 411, 111, 292, 402, 441, 554]
SPIKE = WEEKLY.index(max(WEEKLY))
BACKFILL_DAYS = [4_912, 6_179, 10_930]  # Aug 19, 20, 21 (hud_usage.md)
RAIN = 260


def _node(scene: SlideBase, radius: float, color: str, label: str) -> VGroup:
    ring = Circle(radius=radius, stroke_color=color, stroke_width=2.4)
    ring.set_fill(tint(scene, color, 0.22), opacity=1)
    text = scene.meta_text(label, font_size=15, color=color)
    text.next_to(ring, DOWN, buff=0.18)
    return VGroup(ring, text)


def _loop(scene: SlideBase) -> None:
    t = scene.theme
    bin_colors = [t.accent_danger, t.accent_secondary] + [t.muted_text] * 3

    watcher = _node(scene, 0.55, t.accent_success, "autorevert · Dr.CI")
    watcher.move_to([-5.0, 0.3, 0], aligned_edge=UP).shift(UP * 0.55)
    advisor = _node(scene, 0.75, t.accent_secondary, "Claude advisor")
    advisor.move_to([-1.0, 0.3, 0], aligned_edge=UP).shift(UP * 0.75)
    w_c, a_c = watcher[0].get_center(), advisor[0].get_center()

    bins = VGroup()
    for name, color in zip(VERDICTS, bin_colors, strict=True):
        frame = RoundedRectangle(
            corner_radius=CORNER,
            width=3.0,
            height=0.5,
            stroke_color=color,
            stroke_width=1.4,
            fill_color=tint(scene, color, 0.1),
            fill_opacity=1,
        )
        label = scene.meta_text(name, font_size=14, color=color, uppercase=False)
        label.move_to(frame).align_to(frame, LEFT).shift(RIGHT * 0.15)
        bins.add(VGroup(frame, label))
    bins.arrange(DOWN, buff=0.16).move_to([3.7, a_c[1], 0])

    # Return path: verdicts land in ClickHouse; autorevert reads them back.
    back_path = ArcBetweenPoints(
        bins.get_bottom() + DOWN * 0.1,
        watcher[1].get_bottom() + DOWN * 0.1,
        angle=-PI / 4,
    )
    back = DashedVMobject(back_path.copy(), num_dashes=60).set_stroke(
        t.accent_success, width=2
    )
    back_label = scene.meta_text("ClickHouse", font_size=15, color=t.accent_success)
    back_label.move_to(back_path.point_from_proportion(0.5) + DOWN * 0.28)

    scene.play(
        FadeIn(watcher), FadeIn(advisor), FadeIn(bins, lag_ratio=0.1), run_time=0.6
    )

    def trip(k: int, verdict: int, run_time: float) -> Succession:
        """A red failure arcs from the watcher into Claude, then drops into a bin."""
        dot = Dot(w_c, radius=0.09, color=t.accent_danger)
        bin_frame = bins[verdict][0]
        slot = bin_frame.get_right() + LEFT * (0.3 + 0.24 * k)
        out = ArcBetweenPoints(a_c, slot, angle=-PI / 5 if slot[1] < a_c[1] else PI / 5)
        return Succession(
            FadeIn(dot, scale=2.5, run_time=0.15 * run_time),
            MoveAlongPath(
                dot,
                ArcBetweenPoints(w_c, a_c, angle=-PI / 3),
                run_time=0.4 * run_time,
            ),
            # ApplyFunction builds its target at begin(); `.animate` would snap back to w_c.
            ApplyFunction(
                lambda m: m.scale(0.4).set_color(t.accent_secondary),
                dot,
                run_time=0.12 * run_time,
            ),
            MoveAlongPath(dot, out, run_time=0.3 * run_time),
            ApplyFunction(
                lambda m: m.scale(2.5).set_color(bin_colors[verdict]),
                dot,
                run_time=0.03 * run_time,
            ),
        )

    scene.play(
        trip(0, 0, 2.0),
        advisor[0]
        .animate(rate_func=rate_functions.there_and_back_with_pause, run_time=2.0)
        .scale(1.12),
    )
    scene.play(
        LaggedStart(
            *[trip(1 if v == 0 else 0, v, 1.3) for v in (1, 2, 3, 4, 0)], lag_ratio=0.18
        )
    )
    pulse = Dot(back_path.get_start(), radius=0.08, color=t.accent_success)
    scene.play(Create(back), FadeIn(back_label), run_time=0.6)
    scene.play(
        MoveAlongPath(pulse, back_path, rate_func=linear),
        run_time=0.8,
    )
    scene.play(
        FadeOut(pulse),
        watcher[0].animate(rate_func=rate_functions.there_and_back).scale(1.15),
        run_time=0.4,
    )
    scene.next_slide(
        notes="agents_vs_agents.loop — Autorevert and Dr.CI are bots watching CI. When they see a red signal they dispatch the Claude CI Advisor, which reads the logs and the suspect diff and must answer in a JSON schema: related, unsure, not_related, infra_issue, or garbage. Verdicts land in ClickHouse and autorevert reads them back when deciding whether to revert. Dr.CI renders the verdict inline in its PR comment, at most 32 runs per PR."
    )


def _backfill(scene: SlideBase, head: VGroup) -> None:
    t = scene.theme
    rng = random.Random(3)
    baseline_y, max_h, slot_w, bar_w = -2.7, 4.3, 0.86, 0.56
    x0 = -slot_w * SPIKE
    scale = max_h / max(WEEKLY)
    xs = [x0 + i * slot_w for i in range(len(WEEKLY))]

    axis = Line(
        [xs[0] - 0.6, baseline_y, 0],
        [xs[-1] + 0.6, baseline_y, 0],
        color=t.divider,
        stroke_width=1.2,
    )
    small_bars = VGroup(
        *[
            Rectangle(width=bar_w, height=max(n * scale, 0.025), stroke_width=0)
            .set_fill(t.accent_secondary, opacity=0.9)
            .move_to([x, baseline_y, 0], aligned_edge=DOWN)
            for i, (x, n) in enumerate(zip(xs, WEEKLY, strict=True))
            if i != SPIKE
        ]
    )
    quiet = scene.meta_text("other weeks: 96–554 runs", font_size=14)
    quiet.next_to(small_bars[-4:], UP, buff=0.3)

    # Spike: the three backfill days stack up; the remaining 33 runs are below a pixel.
    segments, y = VGroup(), baseline_y
    for n in BACKFILL_DAYS:
        h = n * scale
        seg = Rectangle(
            width=bar_w, height=h, stroke_color=t.background, stroke_width=1.2
        )
        seg.set_fill(t.accent_secondary, opacity=0.95).move_to(
            [xs[SPIKE], y, 0], aligned_edge=DOWN
        )
        segments.add(seg)
        y += h
    top_y = y

    count = ValueTracker(0)
    counter = always_redraw(
        lambda: scene.title_text(f"{int(count.get_value()):,}", font_size=60).move_to(
            [xs[SPIKE] + 0.55, top_y - 0.35, 0], aligned_edge=LEFT
        )
    )
    caption = VGroup(
        scene.meta_text("advisor runs · one week", font_size=15),
        scene.meta_text(
            "Aug 19–21 backfill · never reverts", font_size=15, color=t.accent_success
        ),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    caption.next_to(
        [xs[SPIKE] + 0.55, top_y - 0.75, 0], DOWN, aligned_edge=LEFT, buff=0.0
    )
    caption.align_to(np.array([xs[SPIKE] + 0.6, 0, 0]), LEFT)

    rain = VGroup(
        *[
            Dot(radius=0.045, color=t.accent_danger).move_to(
                [xs[SPIKE] + rng.uniform(-2.8, 2.8), 4.3 + rng.uniform(0, 1.2), 0]
            )
            for _ in range(RAIN)
        ]
    )
    targets = [
        [
            xs[SPIKE] + rng.uniform(-0.2, 0.2),
            baseline_y + max((top_y - baseline_y) * (i + 1) / RAIN - 0.05, 0.05),
            0,
        ]
        for i in range(RAIN)
    ]

    scene.play(
        FadeIn(axis),
        LaggedStart(*[FadeIn(b, shift=UP * 0.05) for b in small_bars], lag_ratio=0.08),
        run_time=0.7,
    )
    scene.play(FadeIn(quiet), run_time=0.3)
    scene.add(rain, counter)
    scene.add(segments)  # drawn over landed rain
    for seg in segments:
        seg.save_state()
        seg.stretch_to_fit_height(0.001).align_to(seg.saved_state, DOWN)
    grow = 3.0
    scene.play(
        LaggedStart(
            *[
                d.animate(rate_func=rate_functions.rush_into)
                .move_to(p)
                .scale(0.6)
                .set_color(t.accent_secondary)
                for d, p in zip(rain, targets, strict=True)
            ],
            lag_ratio=0.012,
            run_time=grow,
        ),
        Succession(
            *[
                seg.animate(
                    rate_func=linear, run_time=grow * n / sum(BACKFILL_DAYS)
                ).restore()
                for seg, n in zip(segments, BACKFILL_DAYS, strict=True)
            ],
        ),
        count.animate(rate_func=linear, run_time=grow).set_value(max(WEEKLY)),
    )
    scene.remove(rain, *rain, counter)
    scene.add(scene.title_text(f"{max(WEEKLY):,}", font_size=60).move_to(counter))
    scene.play(FadeIn(caption, shift=UP * 0.1), run_time=0.4)
    scene.wait(0.3)
    scene.next_slide(
        notes="agents_vs_agents.backfill — On Aug 19 Jean added a coverage lambda: about 40% of trunk reds never got a verdict. It backfilled roughly 22 thousand historical reds in three days, 4.9k, 6.2k and 10.9k per day against tens of runs on a normal day. Coverage verdicts are tagged so they can never trigger a revert; they are data for understanding flaky jobs."
    )


def build(scene: SlideBase) -> None:
    head = header(scene, "Agents investigating CI failures")
    scene.play(FadeIn(head), run_time=0.3)
    _loop(scene)
    keep = {head}
    scene.play(*[FadeOut(m) for m in scene.mobjects if m not in keep], run_time=0.4)
    _backfill(scene, head)


class V2Agents(SlideBase):
    theme = ClaudePytorchDeck.theme
    deck_mark = ClaudePytorchDeck.deck_mark

    def build_slides(self) -> None:
        build(self)
