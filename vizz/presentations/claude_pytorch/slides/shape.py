from itertools import pairwise

from manim import DOWN, LEFT, RIGHT, UP, Create, FadeIn, VGroup

from vizz.presentations.claude_pytorch.slides.common import box, flow, header
from vizz.presentations.components import SlideBase

# research/test_infra.md §1-3: pytorch/test-infra/.github/workflows/_claude-code.yml.
GATE = [
    "repo owner ∈ {pytorch, meta-pytorch}",
    "comment mentions @claude",
    "author is OWNER / MEMBER / COLLABORATOR",
    "API check: write or admin permission",
]
AUTH = [
    "no API keys stored in GitHub",
    "OIDC token → AWS role, us-east-1",
    "trust: repo:<org>/<repo>:environment:bedrock",
    "bedrock environment deploys from main only",
]
OUTPUTS = [
    "reply on the PR / issue",
    "public transcript → S3",
    "usage row → ClickHouse",
]


def _notes(scene: SlideBase, lines: list[str], color: str) -> VGroup:
    return VGroup(
        *[scene.body_text(f"· {line}", font_size=20, color=color) for line in lines]
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "What happens when you type @claude")

    stages = VGroup(
        box(
            scene,
            "@claude comment",
            width=2.8,
            height=1.1,
            color=t.accent_danger,
            sublabel="untrusted text",
            font_size=24,
        ),
        box(
            scene,
            "gate",
            width=2.4,
            height=1.1,
            color=t.accent_success,
            sublabel="job if: + API check",
            font_size=24,
        ),
        box(
            scene,
            "GitHub OIDC",
            width=2.8,
            height=1.1,
            color=t.accent_success,
            sublabel="environment: bedrock",
            font_size=24,
        ),
        box(
            scene,
            "Claude Code",
            width=2.8,
            height=1.1,
            color=t.accent_secondary,
            sublabel="Bedrock · Opus 5.5",
            font_size=24,
        ),
    ).arrange(RIGHT, buff=0.6)
    stages.next_to(head, DOWN, buff=0.6).set_x(0)
    arrows = VGroup(
        *[flow(scene, a.get_right(), b.get_left()) for a, b in pairwise(stages)]
    )
    gate_notes = _notes(scene, GATE, t.text)
    gate_notes.next_to(stages, DOWN, buff=0.55).align_to(stages[0], LEFT)
    auth_notes = _notes(scene, AUTH, t.text)
    auth_notes.next_to(gate_notes, DOWN, buff=0.4).align_to(gate_notes, LEFT)
    outputs = VGroup(
        *[box(scene, o, width=3.4, height=0.6, font_size=18) for o in OUTPUTS]
    )
    outputs.arrange(DOWN, buff=0.2).next_to(stages[-1], DOWN, buff=0.7)
    outputs.set_x(stages[-1].get_x())
    out_arrows = VGroup(
        flow(
            scene, stages[-1].get_bottom(), outputs.get_top(), color=t.accent_secondary
        )
    )
    gate_tag = scene.meta_text("gate", font_size=14, color=t.accent_success)
    auth_tag = scene.meta_text("credentials", font_size=14, color=t.accent_success)
    gate_tag.next_to(gate_notes, UP, buff=0.1).align_to(gate_notes, LEFT)
    auth_tag.next_to(auth_notes, UP, buff=0.1).align_to(auth_notes, LEFT)
    gate_notes = VGroup(gate_tag, gate_notes)
    auth_notes = VGroup(auth_tag, auth_notes)

    scene.play(FadeIn(head), FadeIn(stages[:2]), Create(arrows[0]), run_time=0.6)
    scene.play(FadeIn(gate_notes, shift=UP * 0.1), run_time=0.5)
    scene.play(
        FadeIn(stages[2]),
        Create(arrows[1]),
        FadeIn(auth_notes, shift=UP * 0.1),
        run_time=0.6,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="shape.auth — History: Ivan shipped @claude on Jan 16 for a 20-person pilot, issue triage followed on Jan 28, and on Feb 27 Ivan replaced the allowlist with this permission check. In March Zain moved it into test-infra as a reusable workflow. A comment is untrusted text. Before any runner does real work, the job-level if: checks the org, the mention, and the author association; then an API call confirms write permission. Anyone with write access can use it; nobody else can. No API keys in GitHub. The job runs in a GitHub environment called bedrock, mints an OIDC token, and assumes one AWS role whose trust policy names each onboarded repo's bedrock environment. That environment only deploys from main, so a PR cannot change the workflow and get credentials."
    )
    scene.play(
        FadeIn(stages[3]),
        Create(arrows[2]),
        *[Create(a) for a in out_arrows],
        FadeIn(outputs, lag_ratio=0.15),
        run_time=0.8,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="shape.outputs — claude-code-action runs on Bedrock. Every run leaves three things: the reply, a public transcript in S3 so anyone can audit what it did, and a usage row in HUD's ClickHouse. That last one is where our adoption numbers come from."
    )
