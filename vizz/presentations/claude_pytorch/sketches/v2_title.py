"""Round-2 title: three dot streams pour into one small maintainer cluster.

Standalone render:
    uv run manim -qm --media_dir /tmp/v2_title \
        vizz/presentations/claude_pytorch/sketches/v2_title.py V2Title

`build(scene)` is SlideBase-compatible (drop-in for slides/title.py).
Dots are illustrative volume, not a data series; the only numbers on the deck's
title are the speakers. Colors: muted = human-sent work, amber = agent-sent work,
green = maintainers.
"""

import random

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Dot,
    FadeIn,
    LaggedStart,
    Line,
    VGroup,
)

from vizz.presentations.claude_pytorch.build import ClaudePytorchDeck
from vizz.presentations.components import SlideBase

SOURCES = ("contributors", "bots", "everyone")
HUMAN_PER_SOURCE = 8  # pause 1: humans alone
AGENT_PER_SOURCE = 36  # pause 2: agents join each stream
QUEUE_ROWS = 14
SPACING = 0.16
DOT_RADIUS = 0.055
QUEUE_RIGHT_X = 5.35  # first (rightmost) queue column; the wall grows leftward
QUEUE_CENTER_Y = 0.3
SPAWN_X = 2.75
MAINTAINER_X = 5.95
MAINTAINERS = 6  # 2 x 3 green dots


def _queue_positions(start: int, count: int) -> list:
    """Column-major slots filling top-to-bottom, columns growing leftward."""
    y_top = QUEUE_CENTER_Y + SPACING * (QUEUE_ROWS - 1) / 2
    out = []
    for i in range(start, start + count):
        col, row = divmod(i, QUEUE_ROWS)
        out.append(
            RIGHT * (QUEUE_RIGHT_X - SPACING * col) + UP * (y_top - SPACING * row)
        )
    return out


def _stream(scene: SlideBase, rows: VGroup, per_source: int, color, rng) -> list:
    """One dot batch per source, interleaved so the wall fills mixed."""
    dots = []
    for k in range(per_source):
        for row in rows:
            y = row.get_center()[1]
            spawn = RIGHT * (SPAWN_X + rng.uniform(-0.12, 0.12)) + UP * (
                y + rng.uniform(-0.22, 0.22)
            )
            dots.append(
                Dot(radius=DOT_RADIUS, color=color, fill_opacity=0.9).move_to(spawn)
            )
    return dots


def build(scene: SlideBase) -> None:
    t = scene.theme
    rng = random.Random(3)

    # Left: deck title block (unchanged furniture).
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

    # Right: three source labels, a queue wall, one small green cluster.
    rows = VGroup(
        *[scene.body_text(s, font_size=22, color=t.muted_text) for s in SOURCES]
    )
    for row, y in zip(rows, (1.55, 0.3, -0.95), strict=True):
        row.move_to([0.6, y, 0], aligned_edge=LEFT)

    maintainers = VGroup(
        *[
            Dot(radius=0.1, color=t.accent_success, fill_opacity=1).move_to(
                [MAINTAINER_X + 0.3 * c, QUEUE_CENTER_Y + 0.3 * (r - 1), 0]
            )
            for r in range(3)
            for c in range(2)
        ]
    )
    maint_label = scene.meta_text("maintainers", font_size=13, color=t.accent_success)
    maint_label.next_to(maintainers, DOWN, buff=0.18)

    scene.play(FadeIn(text, shift=UP * 0.1), run_time=0.7)

    # Pause 1: humans alone, a thin steady stream.
    human = _stream(scene, rows, HUMAN_PER_SOURCE, t.muted_text, rng)
    human_targets = _queue_positions(0, len(human))
    scene.play(
        FadeIn(rows, lag_ratio=0.15),
        FadeIn(maintainers),
        FadeIn(maint_label),
        run_time=0.5,
    )
    scene.add(*human)
    scene.play(
        LaggedStart(
            *[d.animate.move_to(p) for d, p in zip(human, human_targets, strict=True)],
            lag_ratio=0.06,
        ),
        run_time=1.6,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="title.humans — Contributors, bots, everyone: their work has always landed on the same small set of maintainers."
    )

    # Pause 2: agents join every stream; the wall in front of maintainers thickens.
    agent_tags = VGroup(
        *[
            scene.meta_text("+ agents", font_size=13, color=t.accent_secondary)
            .next_to(row, RIGHT, buff=0.18)
            .align_to(row[0], DOWN)
            for row in rows
        ]
    )
    agents = _stream(scene, rows, AGENT_PER_SOURCE, t.accent_secondary, rng)
    agent_targets = _queue_positions(len(human), len(agents))
    scene.play(
        FadeIn(agent_tags, shift=RIGHT * 0.1, lag_ratio=0.15),
        FadeIn(VGroup(*agents), lag_ratio=0.01),
        run_time=0.5,
    )
    scene.play(
        LaggedStart(
            *[d.animate.move_to(p) for d, p in zip(agents, agent_targets, strict=True)],
            lag_ratio=0.012,
        ),
        run_time=2.2,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="title.agents — Now every one of those streams has agents behind it. Same maintainers. This talk is about the tools we built so they can keep up without lowering the bar."
    )


class V2Title(SlideBase):
    theme = ClaudePytorchDeck.theme
    deck_mark = ClaudePytorchDeck.deck_mark

    def build_slides(self) -> None:
        build(self)
