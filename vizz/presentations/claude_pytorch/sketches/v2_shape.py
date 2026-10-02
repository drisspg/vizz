"""Gate → ephemeral credentials → auditable outputs; illustrative, not traffic data."""

from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    PI,
    UP,
    Arc,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    MoveAlongPath,
    RoundedRectangle,
    Scene,
    UpdateFromAlphaFunc,
    VGroup,
    VMobject,
)

from vizz.presentations.claude_pytorch.build import ClaudePytorchDeck
from vizz.presentations.claude_pytorch.slides.common import box, flow, header, tint
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
    head = header(scene, "What happens when you type @claude")
    gate = VGroup(
        Line([-5, -0.25, 0], [-0.5, -0.25, 0], color=t.accent_success, stroke_width=7),
        Line([0.5, -0.25, 0], [5, -0.25, 0], color=t.accent_success, stroke_width=7),
    )
    gate_label = scene.body_text("write access", font_size=28, color=t.accent_success)
    gate_label.move_to([0, -2.25, 0])
    tokens = VGroup()
    for i in range(15):
        token = RoundedRectangle(
            width=0.38,
            height=0.26,
            corner_radius=0.06,
            color=t.accent_danger,
            fill_color=tint(scene, t.accent_danger, 0.35),
            fill_opacity=1,
            stroke_width=1.5,
        ).move_to([-4.4 + i * 0.62, 1.5 + (i % 3) * 0.2, 0])
        if i in (6, 7, 8):
            badge = Dot(radius=0.055, color=t.accent_success).move_to(token.get_top())
            token = VGroup(token, badge)
        tokens.add(token)
    allowed = VGroup(*[tokens[i] for i in (6, 7, 8)])
    rejected = VGroup(*[tokens[i] for i in range(15) if i not in (6, 7, 8)])
    scene.play(
        FadeIn(head), Create(gate), FadeIn(gate_label), FadeIn(tokens), run_time=0.5
    )
    scene.play(
        LaggedStart(*[p.animate.set_y(0.03) for p in rejected], lag_ratio=0.04),
        *[p.animate.move_to([0, 0.45 + i * 0.38, 0]) for i, p in enumerate(allowed)],
        run_time=0.8,
    )
    scene.play(
        *[
            p.animate.shift(UP * (0.6 + (i % 3) * 0.3) + LEFT * 0.3)
            for i, p in enumerate(rejected)
        ],
        LaggedStart(
            *[
                p.animate.move_to([0, -1.1 - i * 0.34, 0])
                for i, p in enumerate(allowed)
            ],
            lag_ratio=0.2,
        ),
        run_time=0.9,
    )
    scene.next_slide(
        notes="shape.gate — Illustrative tokens, not measured traffic. Red is untrusted comment text, including comments from maintainers. Green badges mean verified write access. The fast gate also checks org, mention and author association. The permission API requires write/admin; the allowlisted autorevert bot is the exception."
    )

    scene.play(FadeOut(gate_label), run_time=0.2)
    gate_diagram = VGroup(gate, tokens)
    boundary = RoundedRectangle(
        width=6.4,
        height=4.7,
        corner_radius=0.15,
        color=t.accent_success,
        fill_color=tint(scene, t.accent_success, 0.05),
        fill_opacity=1,
        stroke_width=2,
    ).move_to([2.1, -0.4, 0])
    env_label = scene.meta_text(
        "environment: bedrock", font_size=19, color=t.accent_success
    )
    env_label.move_to([2.1, 1.55, 0])
    oidc = scene.body_text("OIDC", font_size=23).move_to([-0.05, 0.7, 0])
    key = key_icon(t.accent_success).move_to([-0.05, -0.05, 0])
    claude = box(
        scene, "Claude", width=1.7, height=0.85, color=t.accent_secondary, font_size=26
    )
    claude.move_to([3.75, -0.05, 0])
    bedrock = scene.body_text("Bedrock", font_size=20).next_to(claude, DOWN, buff=0.17)
    arrow = flow(scene, [0.5, -0.05, 0], [2.85, -0.05, 0], color=t.accent_success)
    ring = Circle(radius=0.68, color=t.accent_success, stroke_width=7).move_to(
        [1.2, -1.4, 0]
    )
    sixty = scene.title_text("60", font_size=43).move_to(ring)
    minutes = scene.meta_text("min", font_size=15).next_to(ring, DOWN, buff=0.13)
    lifetime = scene.body_text("AWS session + App token", font_size=18).move_to(
        [2.15, -2.5, 0]
    )
    scene.play(
        gate_diagram.animate.scale(0.43).move_to([-4.65, -0.2, 0]),
        FadeIn(boundary),
        FadeIn(env_label),
        run_time=0.6,
    )
    scene.play(allowed[0].animate.move_to([-0.05, 0.15, 0]), run_time=0.65)
    scene.play(
        FadeOut(allowed[0]),
        FadeIn(oidc),
        Create(key),
        Create(arrow),
        FadeIn(claude),
        FadeIn(bedrock),
        run_time=0.7,
    )
    scene.play(
        Create(ring), FadeIn(sixty), FadeIn(minutes), FadeIn(lifetime), run_time=0.6
    )
    scene.next_slide(
        notes="shape.credentials — The GitHub environment named bedrock deploys from main. GitHub OIDC assumes the per-repo allowlisted AWS role; no stored Bedrock API key. The key symbolizes ephemeral credentials, not a stored secret. AWS session AND GitHub App token expire at one hour; the OIDC token itself is not the 60-minute clock. Jobs stop at 55 minutes."
    )

    # The clock drains while three distinct receipts leave Claude.
    scene.play(FadeOut(VGroup(oidc, lifetime)), run_time=0.2)
    outputs = (
        VGroup(
            *[
                box(scene, label, width=2.25, height=0.55, font_size=19)
                for label in ("reply", "public log", "usage row")
            ]
        )
        .arrange(DOWN, buff=0.18)
        .move_to([-4.55, -0.25, 0])
    )
    scene.play(FadeOut(gate_diagram), run_time=0.3)
    packets = VGroup(
        *[Dot(radius=0.08, color=t.accent_secondary).move_to(claude) for _ in outputs]
    )
    paths = []
    for target in outputs:
        path = VMobject().set_points_smoothly(
            [claude.get_center(), [2.6, 0.75, 0], [-1.9, 0.75, 0], target.get_right()]
        )
        paths.append(path)
    scene.play(
        LaggedStart(
            *[MoveAlongPath(p, path) for p, path in zip(packets, paths, strict=True)],
            lag_ratio=0.22,
        ),
        UpdateFromAlphaFunc(
            ring,
            lambda mob, alpha: mob.become(
                Arc(
                    radius=0.68,
                    start_angle=PI / 2,
                    angle=-2 * PI * (1 - alpha * 11 / 12),
                    color=t.accent_success,
                    stroke_width=7,
                ).move_arc_center_to([1.2, -1.4, 0])
            ),
        ),
        run_time=1.6,
    )
    scene.play(FadeOut(packets), FadeIn(outputs, lag_ratio=0.15), run_time=0.5)
    scene.next_slide(
        notes="shape.receipts — Reply on the PR or issue; public transcript archived in S3; usage row uploaded via S3 to ClickHouse. The draining ring is schematic elapsed credential lifetime, not an average run duration. The 60 remains the lifetime limit."
    )


class V2Shape(SlideBase):
    theme = ClaudePytorchDeck.theme

    def render(self, *args, **kwargs) -> None:
        Scene.render(self, *args, **kwargs)

    def next_slide(self, *args, **kwargs) -> None:
        self.wait(0.9)
        self.pause_index += 1
        self.renderer.camera.get_image().save(
            Path(__file__).with_name(f"v2_shape_{self.pause_index}.png")
        )

    def build_slides(self) -> None:
        self.pause_index = 0
        build(self)
