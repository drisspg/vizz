"""Round-2 sketch: 18 skills orbit pytorch/pytorch, then 13 repos light up around the
shared test-infra workflow in first-seen order, each sized by its Claude runs.

Render: uv run manim -qm --media_dir /tmp/v2_skills vizz/presentations/claude_pytorch/sketches/v2_skills.py V2Skills

`build(scene)` is SlideBase-compatible (drop-in for slides/skills.py).
Facts: research/pytorch_repo.md §5.1 (18 skills in .claude/skills/),
research/test_infra.md §1 (reusable _claude-code.yml in pytorch/test-infra),
research/data/hud/by_repo.json (runs and first-seen date per repo, 2026-10-02).
"""

import math

import numpy as np
from manim import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    Circle,
    Create,
    Dot,
    FadeIn,
    GrowFromCenter,
    LaggedStart,
    Line,
    RoundedRectangle,
    Transform,
    VGroup,
    rate_functions,
)

from vizz.presentations.claude_pytorch.build import ClaudePytorchDeck
from vizz.presentations.claude_pytorch.slides.common import CORNER, header, tint
from vizz.presentations.components import SlideBase

N_SKILLS = 18
# (repo, runs, first-seen date) sorted by first Claude run (by_repo.json).
REPOS = [
    ("ciforge", 454, "Jan 24"),
    ("pytorch", 76_825, "Jan 28"),
    ("test-infra", 2_798, "Feb 27"),
    ("tutorials", 31, "Mar 2"),
    ("ao", 493, "Mar 10"),
    ("torchtitan", 273, "Mar 10"),
    ("gha-infra", 3_364, "Mar 12"),
    ("executorch", 629, "Mar 26"),
    ("helion", 81, "May 13"),
    ("ci-infra", 44, "May 28"),
    ("torchcomms", 2, "Jun 9"),
    ("attention-gym", 3, "Jul 7"),
    ("gha-infra (internal)", 397, "Sep 9"),
]
LABELED = {"pytorch", "test-infra", "gha-infra", "executorch", "ao", "ciforge"}
PYTORCH_R = 1.0
PYTORCH_C = np.array([-4.3, -0.75, 0])
HUB_C = np.array([1.6, -0.75, 0])
RING = 2.35


def _radius(runs: int) -> float:
    """Node area ∝ runs, with a floor so 2-run repos stay visible."""
    return 0.06 + (PYTORCH_R - 0.06) * math.sqrt(runs / REPOS[1][1])


def _node(scene: SlideBase, radius: float) -> Circle:
    t = scene.theme
    return Circle(
        radius=radius, stroke_color=t.accent_secondary, stroke_width=1.6
    ).set_fill(tint(scene, t.accent_secondary, 0.45), opacity=1)


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "Skills in the repo, one workflow for all")

    # Beat 1: pytorch/pytorch with its 18 skills in orbit.
    pytorch = _node(scene, PYTORCH_R).move_to(PYTORCH_C)
    py_label = scene.meta_text("pytorch", font_size=16, color=t.accent_secondary)
    py_label.move_to(pytorch)
    orbit = Circle(
        radius=PYTORCH_R + 0.45, stroke_color=t.divider, stroke_width=1
    ).move_to(PYTORCH_C)
    skills = VGroup(
        *[
            Dot(radius=0.075, color=t.accent_success).move_to(
                PYTORCH_C
                + (PYTORCH_R + 0.45)
                * np.array(
                    [
                        math.cos(PI / 2 - 2 * PI * k / N_SKILLS),
                        math.sin(PI / 2 - 2 * PI * k / N_SKILLS),
                        0,
                    ]
                )
            )
            for k in range(N_SKILLS)
        ]
    )
    big = scene.title_text("18", font_size=72)
    big_label = scene.meta_text("skills in .claude/skills", font_size=15)
    big_group = VGroup(big, big_label).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
    big_group.next_to(orbit, RIGHT, buff=0.7).align_to(orbit, UP)

    scene.play(FadeIn(head), GrowFromCenter(pytorch), FadeIn(py_label), run_time=0.5)
    scene.play(
        Create(orbit),
        LaggedStart(*[FadeIn(s, scale=3) for s in skills], lag_ratio=0.12),
        FadeIn(big_group, shift=UP * 0.1),
        run_time=1.6,
    )
    scene.wait(0.3)
    scene.next_slide(
        notes="skills.repo — The model is generic; PyTorch knowledge lives in the repo as 18 skills. pr-review encodes our checklist and BC rules and is used both by @claude and by humans locally. Triage and readiness review are bot-only skills. AGENTS.md is a symlink to CLAUDE.md so every agent reads the same rules."
    )

    # Beat 2: the shared workflow hub; repos light up in first-seen order.
    hub = RoundedRectangle(
        corner_radius=CORNER,
        width=1.9,
        height=0.6,
        stroke_color=t.accent_success,
        stroke_width=2,
        fill_color=tint(scene, t.accent_success, 0.2),
        fill_opacity=1,
    ).move_to(HUB_C)
    hub_label = scene.meta_text(
        "_claude-code.yml", font_size=14, color=t.accent_success, uppercase=False
    )
    hub_label.move_to(hub)

    others = [r for r in REPOS if r[0] != "pytorch"]
    angles = (
        np.linspace(140, -140, len(others)) * PI / 180
    )  # leave the west side to pytorch
    nodes, edges, labels = {}, {}, {}
    for (name, runs, _), ang in zip(others, angles, strict=True):
        direction = np.array([math.cos(ang), math.sin(ang), 0])
        c = HUB_C + RING * direction
        nodes[name] = _node(scene, _radius(runs)).move_to(c)
        edges[name] = Line(hub.get_center(), c, color=t.divider, stroke_width=1.4)
        if name in LABELED:
            lab = scene.meta_text(name, font_size=13, uppercase=False)
            lab.next_to(nodes[name], direction, buff=0.1)
            labels[name] = lab
    nodes["pytorch"] = pytorch
    edges["pytorch"] = Line(
        hub.get_center(), PYTORCH_C, color=t.divider, stroke_width=1.4
    )
    for e in edges.values():
        e.set_z_index(-1)
    hub_group = VGroup(hub, hub_label).set_z_index(1)

    count = scene.title_text("0", font_size=72)
    count_label = scene.meta_text("repos", font_size=15)
    date = scene.meta_text("", font_size=15)

    def place(num, when):
        g = VGroup(num, VGroup(count_label, when).arrange(RIGHT, buff=0.25))
        g.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        return g.move_to([4.5, 2.3, 0], aligned_edge=UP + LEFT)

    place(count, date)
    scene.play(
        big_group.animate.set_opacity(0),
        FadeIn(hub_group, scale=0.8),
        FadeIn(count),
        FadeIn(count_label),
        run_time=0.5,
    )
    scene.remove(big_group)

    for k, (name, _, when) in enumerate(REPOS, start=1):
        new_count = scene.title_text(str(k), font_size=72)
        new_date = scene.meta_text(when, font_size=15)
        place(new_count, new_date)
        anims = [
            Create(edges[name]),
            Transform(count, new_count),
            Transform(date, new_date),
        ]
        if name != "pytorch":
            anims.append(GrowFromCenter(nodes[name]))
        else:
            anims.append(
                pytorch.animate(rate_func=rate_functions.there_and_back).scale(1.08)
            )
        if name in labels:
            anims.append(FadeIn(labels[name]))
        scene.play(*anims, run_time=0.4 if k < 4 else 0.28)
    scene.wait(0.4)
    scene.next_slide(
        notes="skills.onboard — For other repos, Zain moved the workflow into test-infra. A new repo adds one caller file, runs one setup script that creates the bedrock environment, and adds one line to the IAM trust policy. Thirteen repos across pytorch and meta-pytorch report usage today; circle area is Claude runs, from 76,825 on pytorch down to 2 on torchcomms."
    )


class V2Skills(SlideBase):
    theme = ClaudePytorchDeck.theme
    deck_mark = ClaudePytorchDeck.deck_mark

    def build_slides(self) -> None:
        build(self)
