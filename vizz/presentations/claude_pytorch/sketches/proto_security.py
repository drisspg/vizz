"""Illustrative label-hook failure and recovery, not a live security test.

Render: uv run manim -ql vizz/presentations/claude_pytorch/sketches/proto_security.py ProtoSecurity
Facts: research/pytorch_repo.md sections 2.3–2.4.
"""

from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    Create,
    Cross,
    FadeIn,
    FadeOut,
    Line,
    MoveAlongPath,
    Scene,
    Transform,
    VGroup,
    tempconfig,
    there_and_back,
)

from vizz.presentations.claude_pytorch.build import ClaudePytorchDeck
from vizz.presentations.claude_pytorch.slides.common import box, flow, header, punchline
from vizz.presentations.components import SlideBase

RENDER_CONFIG = {
    "media_dir": str(Path(__file__).parent / "proto_security_media"),
    "text_dir": str(Path(__file__).parent / "proto_security_media" / "texts"),
    "output_file": "proto_security",
}


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "The model can ask. The hook can say no.")
    attack = scene.body_text(
        "Issue body: “add merge blocking”  ·  illustrative payload",
        font_size=23,
        color=t.accent_danger,
    ).move_to([0, 2.05, 0])
    claude = box(
        scene,
        "Claude proposes",
        width=3.4,
        height=1.05,
        sublabel="agent output ≠ authority",
        color=t.accent_secondary,
        font_size=25,
    ).move_to([-4.55, 0.85, 0])
    hook = box(
        scene,
        "validate_labels.py",
        width=3.8,
        height=1.05,
        sublabel="deterministic Python · 282 labels",
        color=t.accent_success,
        font_size=24,
    ).move_to([0, 0.85, 0])
    issue = box(
        scene,
        "Target issue",
        width=3.4,
        height=1.05,
        sublabel="authorized mutations only",
        color=t.accent_success,
        font_size=25,
    ).move_to([4.55, 0.85, 0])
    arrows = VGroup(
        flow(scene, claude.get_right(), hook.get_left(), color=t.accent_secondary),
        flow(scene, hook.get_right(), issue.get_left(), color=t.accent_success),
    )
    barrier = Line([0, 0.18, 0], [0, -1.8, 0], color=t.accent_success, stroke_width=5)
    forbidden = box(
        scene,
        "merge blocking",
        width=2.85,
        height=0.7,
        color=t.accent_danger,
        font_size=24,
    ).move_to([-4.55, -0.6, 0])
    pending = scene.meta_text("proposed label", font_size=15, color=t.accent_secondary)
    pending.next_to(forbidden, DOWN, buff=0.2)
    caption = punchline(scene, "A prompt is not an enforcement boundary.", font_size=29)
    caption.move_to([0, -2.65, 0])
    source = scene.meta_text(
        "Illustration, not a live hook run · research/pytorch_repo.md §2.3–2.4",
        font_size=12,
        uppercase=False,
    ).to_edge(DOWN, buff=0.22)

    scene.play(FadeIn(head), FadeIn(attack), run_time=0.6)
    scene.play(
        FadeIn(claude), FadeIn(hook), FadeIn(issue), Create(arrows), run_time=0.7
    )
    scene.play(
        Create(barrier),
        FadeIn(forbidden),
        FadeIn(pending),
        FadeIn(caption),
        FadeIn(source),
        run_time=0.6,
    )
    scene.wait(0.3)
    scene.next_slide(
        notes="security.proposal — Illustrative attack, not a real transcript. Assume Claude asks for merge blocking. The triage skill's deterministic label hook filters the request before mutation. The issue-target hook is a separate check, not shown here."
    )

    scene.play(
        FadeOut(pending),
        MoveAlongPath(forbidden, Line(forbidden.get_center(), [-1.5, -0.6, 0])),
        run_time=0.9,
    )
    scene.play(
        barrier.animate.shift(RIGHT * 0.07), rate_func=there_and_back, run_time=0.25
    )
    scene.play(forbidden.animate.shift(LEFT * 0.7), run_time=0.4)
    rejected = Cross(forbidden, stroke_color=t.accent_danger, stroke_width=4)
    stripped = scene.meta_text(
        "stripped before mutation", font_size=16, color=t.accent_danger
    )
    stripped.next_to(forbidden, DOWN, buff=0.25)
    scene.play(Create(rejected), FadeIn(stripped), run_time=0.45)
    scene.play(
        Transform(
            caption,
            punchline(
                scene, "Forbidden label. No forbidden mutation.", font_size=29
            ).move_to(caption),
        ),
        run_time=0.5,
    )
    scene.wait(0.3)
    scene.next_slide(
        notes="security.stripped — validate_labels.py strips merge blocking. The bounce represents filtering a forbidden label, not a claim that every invalid label aborts the entire tool call. No forbidden label reaches the issue."
    )

    escalation = box(
        scene,
        "triage review",
        width=2.85,
        height=0.7,
        color=t.accent_secondary,
        font_size=24,
    ).move_to([-4.55, -0.6, 0])
    followup = scene.meta_text(
        "separate follow-up", font_size=15, color=t.accent_secondary
    )
    followup.next_to(escalation, DOWN, buff=0.2)
    scene.play(FadeOut(forbidden), FadeOut(rejected), FadeOut(stripped), run_time=0.4)
    scene.play(FadeIn(escalation), FadeIn(followup), run_time=0.45)
    scene.play(
        FadeOut(followup),
        MoveAlongPath(escalation, Line(escalation.get_center(), [-1.5, -0.6, 0])),
        run_time=0.7,
    )
    allowed = box(
        scene,
        "triage review",
        width=2.85,
        height=0.7,
        color=t.accent_success,
        font_size=24,
    ).move_to([4.55, -0.6, 0])
    scene.play(Transform(escalation, allowed), run_time=1.1)
    audit = box(
        scene,
        "bot-triaged",
        width=2.85,
        height=0.55,
        color=t.accent_success,
        font_size=20,
    )
    audit.next_to(escalation, DOWN, buff=0.2)
    audit_note = scene.meta_text("audit stamp after mutation", font_size=13)
    audit_note.next_to(audit, DOWN, buff=0.12)
    human = scene.body_text(
        "Escalate. Let a human decide.", font_size=24, color=t.accent_success
    )
    human.move_to([-3.45, -0.8, 0])
    scene.play(
        FadeIn(audit, shift=DOWN * 0.1), FadeIn(audit_note), FadeIn(human), run_time=0.6
    )
    scene.play(
        Transform(
            caption,
            punchline(
                scene, "Constrain the action, not just the prompt.", font_size=29
            ).move_to(caption),
        ),
        run_time=0.5,
    )
    scene.wait(0.3)
    scene.next_slide(
        notes="security.escalate — The skill says: when a label is blocked, add ONLY triage review and stop. This is a separate proposal, not the hook renaming merge blocking. After a mutation, add_bot_triaged.py applies the audit label. A human makes the consequential decision."
    )


class ProtoSecurity(SlideBase):
    """Movie-only SlideBase wrapper; leave full-deck slide metadata untouched."""

    theme = ClaudePytorchDeck.theme
    deck_mark = ClaudePytorchDeck.deck_mark

    def __init__(self, *args, **kwargs) -> None:
        with tempconfig(RENDER_CONFIG):
            super().__init__(*args, **kwargs)

    def render(self, *args, **kwargs) -> None:
        with tempconfig(RENDER_CONFIG):
            Scene.render(self, *args, **kwargs)

    def next_slide(self, *args, **kwargs) -> None:
        self.wait(1.2)

    def build_slides(self) -> None:
        build(self)
