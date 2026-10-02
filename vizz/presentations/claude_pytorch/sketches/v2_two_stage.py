"""Two triage workflows: a scalar handoff, not a prompt-injection firewall."""

from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    UP,
    Circle,
    Create,
    Cross,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    MoveAlongPath,
    RoundedRectangle,
    Scene,
    VGroup,
    VMobject,
)

from vizz.presentations.claude_pytorch.build import ClaudePytorchDeck
from vizz.presentations.claude_pytorch.slides.common import box, header, tint
from vizz.presentations.components import SlideBase


def key_icon(color: str) -> VGroup:
    return VGroup(
        Circle(radius=0.19, color=color, stroke_width=5).shift(LEFT * 0.4),
        Line([-0.21, 0, 0], [0.48, 0, 0], color=color, stroke_width=5),
        Line([0.29, 0, 0], [0.29, -0.18, 0], color=color, stroke_width=5),
        Line([0.46, 0, 0], [0.46, -0.18, 0], color=color, stroke_width=5),
    )


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "Separate code, not content")
    stage1 = RoundedRectangle(
        width=4.1,
        height=3.6,
        corner_radius=0.14,
        color=t.accent_danger,
        fill_color=tint(scene, t.accent_danger, 0.05),
        fill_opacity=1,
        stroke_width=2,
    ).move_to([-3.75, -0.05, 0])
    stage2 = RoundedRectangle(
        width=4.1,
        height=3.6,
        corner_radius=0.14,
        color=t.accent_success,
        fill_color=tint(scene, t.accent_success, 0.05),
        fill_opacity=1,
        stroke_width=2,
    ).move_to([3.75, -0.05, 0])
    first = scene.title_text("1", font_size=37).move_to([-5.25, 1.2, 0])
    second = scene.body_text("2 · main", font_size=30, color=t.accent_success).move_to(
        [3.75, 1.2, 0]
    )
    issue = VGroup(
        *[
            Dot(radius=0.11, color=t.accent_danger).move_to(
                [
                    -3.75 + (i % 5 - 2) * 0.3,
                    0.2 + (i // 5 - 1.5) * 0.3,
                    0,
                ]
            )
            for i in range(20)
        ]
    )
    issue_label = scene.body_text(
        "issue body", font_size=23, color=t.accent_danger
    ).next_to(issue, DOWN, buff=0.22)
    no_key = key_icon(t.muted_text).move_to([-3.75, -1.35, 0])
    cross = Cross(no_key, stroke_color=t.accent_danger, stroke_width=3)
    scene.play(FadeIn(head), Create(stage1), FadeIn(first), run_time=0.5)
    scene.play(
        FadeIn(issue, shift=LEFT * 2, lag_ratio=0.02),
        FadeIn(issue_label),
        Create(no_key),
        Create(cross),
        run_time=0.8,
    )
    scene.next_slide(
        notes="two_stage.capture — Issue triage stage 1 runs on issues: opened in the issue author's event context, contents: read only, no environment or secrets, two-minute timeout. The red dots symbolize untrusted issue content; no model runs here. It validates digits and uploads only issue_number.txt. The sample number 42 is illustrative."
    )

    boundary = VGroup(
        DashedLine([0, 1.75, 0], [0, 0.35, 0], color=t.muted_text, dash_length=0.12),
        DashedLine([0, -0.35, 0], [0, -1.85, 0], color=t.muted_text, dash_length=0.12),
    )
    artifact = scene.meta_text("artifact", font_size=19).move_to([0, 2.1, 0])
    token = box(
        scene, "42", width=0.7, height=0.5, color=t.accent_danger, font_size=22
    ).move_to(issue)
    key = key_icon(t.accent_success).move_to([4.85, -1.25, 0])
    claude = box(
        scene, "Claude", width=1.65, height=0.8, color=t.accent_secondary, font_size=27
    ).move_to([3.75, -0.1, 0])
    # The number crosses; the issue-body dots deliberately remain on the left.
    scene.play(
        Create(boundary),
        FadeIn(artifact),
        Create(stage2),
        FadeIn(second),
        FadeIn(token),
        issue.animate.shift(UP * 0.5).set_opacity(0.35),
        FadeOut(issue_label),
        run_time=0.7,
    )
    scene.play(token.animate.move_to([0, 0, 0]), run_time=0.65)
    scene.play(token.animate.move_to([2.35, -0.1, 0]), run_time=0.65)
    check = VMobject(color=t.accent_success, stroke_width=4).set_points_as_corners(
        [[2.13, -0.62, 0], [2.29, -0.77, 0], [2.61, -0.43, 0]]
    )
    scene.play(Create(check), FadeIn(claude), Create(key), run_time=0.6)
    scene.play(key.animate.shift(LEFT * 0.25), run_time=0.4)
    scene.next_slide(
        notes="two_stage.handoff — Only the issue number crosses the artifact boundary. Stage 2 is triggered by workflow_run and executes code from main. It checks originating workflow/success and revalidates the number, then holds the bedrock environment and issues: write. The split removes untrusted CODE next to credentials, not untrusted text from the model."
    )

    # Stage 2 fetches the issue through GitHub MCP: a separate route from the artifact.
    ports = VGroup(
        *[
            Circle(radius=0.105, color=t.accent_success, stroke_width=2.2).move_to(
                [1.9 + i * 0.27, -2.55, 0]
            )
            for i in range(5)
        ]
    )
    tools = scene.body_text("5 MCP tools", font_size=21).next_to(ports, DOWN, buff=0.2)
    issue_label.move_to([-3.75, -2.6, 0])
    route = VMobject(color=t.accent_danger, stroke_width=2).set_points_smoothly(
        [
            [-3.1, 0.45, 0],
            [-1.95, -1.35, 0],
            [-1.3, -2.55, 0],
            [2.44, -2.55, 0],
            [3.75, -1.8, 0],
            [3.75, -0.65, 0],
        ]
    )
    body_packets = VGroup(
        *[Dot(radius=0.075, color=t.accent_danger).move_to(issue[i]) for i in range(6)]
    )
    scene.play(
        FadeIn(issue_label), FadeIn(ports), FadeIn(tools), Create(route), run_time=0.6
    )
    scene.play(
        LaggedStart(*[MoveAlongPath(p, route) for p in body_packets], lag_ratio=0.12),
        run_time=1.5,
    )
    scene.play(
        FadeOut(body_packets),
        claude[0].animate.set_stroke(t.accent_danger, width=3),
        run_time=0.3,
    )
    scene.next_slide(
        notes="two_stage.content — Stage 2 STILL reads the untrusted issue body and comments. Five allowed GitHub MCP tools support reading, search, updates and comments; no shell or file access. Red reaches Claude through this tool surface. This is not a prompt-injection barrier. Deterministic hooks constrain mutations on the next slide. Ports symbolize the five-tool surface, not five issue-reading calls."
    )


class V2TwoStage(SlideBase):
    theme = ClaudePytorchDeck.theme

    def render(self, *args, **kwargs) -> None:
        Scene.render(self, *args, **kwargs)

    def next_slide(self, *args, **kwargs) -> None:
        self.wait(0.9)
        self.pause_index += 1
        self.renderer.camera.get_image().save(
            Path(__file__).with_name(f"v2_two_stage_{self.pause_index}.png")
        )

    def build_slides(self) -> None:
        self.pause_index = 0
        build(self)
