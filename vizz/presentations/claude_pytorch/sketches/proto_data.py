"""Prototype: "one dot = 100 Claude runs" waffle that sorts itself by dispatcher.

Standalone render (plain manim, no slide metadata needed):
    uv run manim -ql vizz/presentations/claude_pytorch/sketches/proto_data.py ProtoData

`build(scene)` is SlideBase-compatible; drop it into slides/adoption.py beat 3 if adopted.
Data: research/hud_usage.md + research/data/hud/by_category.json (pytorch/pytorch, 2026-10-02).
"""

import random

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Brace,
    Dot,
    FadeIn,
    LaggedStart,
    Line,
    ValueTracker,
    VGroup,
    always_redraw,
    smooth,
)

from vizz.presentations.claude_pytorch.build import ClaudePytorchDeck
from vizz.presentations.claude_pytorch.slides.common import header
from vizz.presentations.components import SlideBase

RUNS_PER_DOT = 100
TOTAL_RUNS = 76_825  # all Claude runs on pytorch/pytorch in misc.claude_code_usage
# (label, runs, p50 minutes, theme color key). Sorted by runs.
CATEGORIES = [
    ("Dr.CI advisor", 33_195, 0.9, "accent_secondary"),
    ("autorevert advisor", 29_085, 1.2, "accent_secondary"),
    ("humans @claude", 8_789, 3.2, "accent_success"),
    ("issue triage", 4_453, 1.1, "muted_text"),
    ("manual dispatch", 875, 1.4, "muted_text"),
    ("bot @claude", 428, 2.9, "muted_text"),
]
BACKFILL_RUNS = 4_912 + 6_179 + 10_930  # Aug 19-21 advisor-coverage backfill
BLOCK_COLS = 14
SPACING = 0.14
DOT_RADIUS = 0.05
CLOUD_COLS, CLOUD_ROWS = 48, 17  # 816 jittered slots for the unsorted cloud


def _grid_positions(n: int, cols: int, origin, spacing: float) -> list:
    """Row-major positions filling `cols` per row, growing upward from `origin` (bottom-left)."""
    return [
        origin + RIGHT * spacing * (i % cols) + UP * spacing * (i // cols)
        for i in range(n)
    ]


def build(scene: SlideBase) -> None:
    t = scene.theme
    rng = random.Random(7)
    head = header(scene, "Who runs Claude on pytorch/pytorch?")

    counts = [round(n / RUNS_PER_DOT) for _, n, _, _ in CATEGORIES]
    n_dots = sum(counts)

    # Beat 1: unsorted jittered cloud (769 is prime, so no exact grid), counter ticking.
    cloud_w = (CLOUD_COLS - 1) * SPACING
    cloud_h = (CLOUD_ROWS - 1) * SPACING
    cloud_origin = LEFT * cloud_w / 2 + DOWN * (cloud_h / 2 + 0.3)
    slots = _grid_positions(CLOUD_COLS * CLOUD_ROWS, CLOUD_COLS, cloud_origin, SPACING)
    grid_pos = [
        p + RIGHT * rng.uniform(-0.04, 0.04) + UP * rng.uniform(-0.04, 0.04)
        for p in rng.sample(slots, n_dots)
    ]
    dots = VGroup(
        *[
            Dot(radius=DOT_RADIUS, color=t.muted_text, fill_opacity=0.55).move_to(
                p + LEFT * (8 + rng.random() * 4) + UP * rng.uniform(-1.5, 1.5)
            )
            for p in grid_pos
        ]
    )
    progress = ValueTracker(0)
    counter = always_redraw(
        lambda: (
            scene.title_text(f"{int(progress.get_value()):,}", font_size=48)
            .next_to(head, DOWN, buff=0.35)
            .to_edge(LEFT, buff=0.7)
        )
    )
    counter_label = scene.meta_text("claude runs · 1 dot ≈ 100 runs", font_size=14)
    counter_label.next_to(head, DOWN, buff=0.35).to_edge(RIGHT, buff=0.7)

    scene.play(FadeIn(head), FadeIn(counter_label), run_time=0.4)
    scene.add(counter)
    scene.play(
        LaggedStart(
            *[d.animate.move_to(p) for d, p in zip(dots, grid_pos, strict=True)],
            lag_ratio=0.002,
        ),
        progress.animate(rate_func=smooth).set_value(TOTAL_RUNS),
        run_time=2.6,
    )
    scene.remove(counter)
    final_counter = scene.title_text(f"{TOTAL_RUNS:,}", font_size=48)
    final_counter.next_to(head, DOWN, buff=0.35).to_edge(LEFT, buff=0.7)
    scene.add(final_counter)
    scene.wait(0.2)
    scene.next_slide(notes="proto.grid — 76,825 Claude runs on pytorch/pytorch.")

    # Beat 2: regroup into one dot-bar per dispatcher, sorted by size.
    block_w = (BLOCK_COLS - 1) * SPACING
    gap = (12.6 - len(CATEGORIES) * block_w) / (len(CATEGORIES) - 1)
    baseline_y = -2.15
    x0 = -6.3
    targets, colors, labels, blocks = [], [], VGroup(), []
    for k, ((name, runs, p50, key), count) in enumerate(
        zip(CATEGORIES, counts, strict=True)
    ):
        origin = RIGHT * (x0 + k * (block_w + gap)) + UP * baseline_y
        pos = _grid_positions(count, BLOCK_COLS, origin, SPACING)
        targets += pos
        colors += [getattr(t, key)] * count
        blocks.append(pos)
        center_x = origin[0] + block_w / 2
        label = VGroup(
            scene.body_text(f"{runs:,}", font_size=20),
            scene.body_text(name, font_size=14, color=t.muted_text),
            scene.meta_text(f"p50 {p50:.1f} min", font_size=11),
        ).arrange(DOWN, buff=0.06)
        label.move_to([center_x, baseline_y - 0.65, 0])
        labels.add(label)
    baseline = Line(
        [x0 - 0.3, baseline_y - 0.2, 0],
        [
            x0 + len(CATEGORIES) * block_w + (len(CATEGORIES) - 1) * gap + 0.3,
            baseline_y - 0.2,
            0,
        ],
        color=t.divider,
        stroke_width=1.0,
    )
    scene.play(
        LaggedStart(
            *[
                d.animate.move_to(p).set_fill(c, opacity=0.9)
                for d, p, c in zip(dots, targets, colors, strict=True)
            ],
            lag_ratio=0.0015,
        ),
        FadeIn(baseline),
        run_time=2.2,
    )
    scene.play(FadeIn(labels, lag_ratio=0.1), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="proto.sorted — Two bot accounts dispatch 81% of runs at about a minute each; humans are 11% of runs, median 3.2 minutes."
    )

    # Beat 3: bracket the Aug 19-21 backfill inside the autorevert bar.
    backfill_dots = round(BACKFILL_RUNS / RUNS_PER_DOT)
    autorevert = VGroup(*dots[counts[0] : counts[0] + counts[1]])
    backfilled = VGroup(*autorevert[-backfill_dots:])
    brace = Brace(backfilled, RIGHT, buff=0.12, color=t.text, sharpness=1.5)
    note = VGroup(
        scene.body_text(f"~{BACKFILL_RUNS / 1000:.0f}k in 3 days", font_size=18),
        scene.meta_text("Aug 19-21 coverage backfill", font_size=12),
        scene.meta_text("intentional · never reverts", font_size=12),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
    note.next_to(brace, RIGHT, buff=0.12)
    scene.play(
        backfilled.animate.set_stroke(t.text, width=1.2),
        FadeIn(brace),
        FadeIn(note, shift=LEFT * 0.1),
        run_time=0.7,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="proto.backfill — Three quarters of the autorevert bar is one deliberate historical backfill: 22k flaky trunk reds classified in three days."
    )


class ProtoData(SlideBase):
    theme = ClaudePytorchDeck.theme
    deck_mark = ClaudePytorchDeck.deck_mark

    def build_slides(self) -> None:
        build(self)
