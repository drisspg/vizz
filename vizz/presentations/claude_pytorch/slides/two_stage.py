from manim import DOWN, RIGHT, UP, Create, FadeIn, VGroup

from vizz.presentations.claude_pytorch.slides.common import (
    box,
    flow,
    header,
    punchline,
)
from vizz.presentations.components import SlideBase

# research/pytorch_repo.md §2.1 (#173725) and §4 (hardened PR review).
STAGE1 = [
    "on: issues: opened",
    "permissions: contents: read",
    "no secrets · no environment",
    "timeout: 2 min",
    "writes issue_number.txt",
]
STAGE2 = [
    "on: workflow_run (code from main)",
    "environment: bedrock · issues: write",
    "re-validates ^[0-9]+$",
    "Claude + 5 GitHub tools",
]


def _column(scene: SlideBase, title: str, sub: str, lines: list[str], color: str):
    frame_box = box(
        scene, title, width=4.8, height=1.0, color=color, sublabel=sub, font_size=26
    )
    body = VGroup(*[scene.body_text(line, font_size=22) for line in lines]).arrange(
        DOWN, aligned_edge=[-1, 0, 0], buff=0.12
    )
    body.next_to(frame_box, DOWN, buff=0.3).align_to(frame_box, [-1, 0, 0]).shift(
        RIGHT * 0.15
    )
    return VGroup(frame_box, body)


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "Untrusted input never meets a write token")
    sub = (
        scene.meta_text("issue triage · two GitHub Actions workflows")
        .next_to(head, DOWN, buff=0.3)
        .align_to(head, [-1, 0, 0])
    )

    stage1 = _column(
        scene, "stage 1", "runs as the issue author", STAGE1, t.accent_danger
    )
    artifact = box(
        scene, "artifact", width=2.0, height=0.9, sublabel="one number", font_size=22
    )
    stage2 = _column(scene, "stage 2", "runs as the repo", STAGE2, t.accent_success)
    row = VGroup(stage1, artifact, stage2).arrange(RIGHT, buff=0.7, aligned_edge=UP)
    row.next_to(sub, DOWN, buff=0.45).set_x(0)
    artifact.align_to(stage1[0], UP).shift(DOWN * 0.05)
    a1 = flow(scene, stage1[0].get_right(), artifact.get_left())
    a2 = flow(scene, artifact.get_right(), stage2[0].get_left(), color=t.accent_success)

    quote = punchline(
        scene,
        "“DO NOT add this workflow as a required status check: a prompt injection\n"
        "could then fail it deliberately to block every merge.”",
        font_size=22,
    ).to_edge(DOWN, buff=0.4)
    source = (
        scene.meta_text("hardened-pr-review-run.yml", font_size=12, uppercase=False)
        .next_to(quote, UP, buff=0.1)
        .align_to(quote, RIGHT)
    )

    scene.play(FadeIn(head), FadeIn(sub), FadeIn(stage1), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="two_stage.untrusted — The first version of triage was one job with Bedrock credentials triggered by issues: opened. A day later we split it. Stage 1 runs in the issue author's context, has no secrets, and does one thing: write the issue number to an artifact."
    )
    scene.play(Create(a1), FadeIn(artifact), Create(a2), FadeIn(stage2), run_time=0.8)
    scene.wait(0.2)
    scene.next_slide(
        notes="two_stage.privileged — Stage 2 is triggered by workflow_run, so its code always comes from main. It holds the bedrock environment and issues: write, and it re-validates the one number it received. The same split is used by the hardened PR review, Green Light, and ao's CI-failure bot."
    )
    scene.play(FadeIn(quote, shift=UP * 0.1), FadeIn(source), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="two_stage.quote — Our favorite comment in the repo. Assume the model can be talked into anything, and design so that the worst case is a wrong label, not a blocked repo."
    )
