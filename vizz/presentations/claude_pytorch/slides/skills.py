from manim import DOWN, LEFT, RIGHT, UP, FadeIn, VGroup

from vizz.presentations.claude_pytorch.slides.common import box, code_block, header
from vizz.presentations.components import SlideBase

# research/pytorch_repo.md §5.1 (18 skills in .claude/skills/) and
# research/test_infra.md §1 (setup-claude-environment.py caller YAML).
GROUPS = [
    (
        "run by CI bots",
        ["triaging-issues", "distributed-triage", "pr-review-readiness"],
    ),
    ("bots + humans", ["pr-review"]),
    (
        "developers",
        [
            "fix-issue",
            "pt2-bug-basher",
            "aoti-debug",
            "metal-kernel",
            "cuda-index-width",
            "+ 9 more",
        ],
    ),
]
CALLER = """
on:
  issue_comment: {types: [created]}
  issues: {types: [opened]}
jobs:
  claude-code:
    uses: pytorch/test-infra/.github/workflows/_claude-code.yml@main
    permissions:
      {contents: read, pull-requests: write, issues: write, id-token: write}
    secrets: inherit
"""


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "Skills carry the bar; one file onboards a repo")

    colors = [t.accent_secondary, t.accent_success, t.divider]
    columns = VGroup()
    for (title, names), color in zip(GROUPS, colors, strict=True):
        chips = VGroup(
            *[
                box(
                    scene,
                    n,
                    width=2.4,
                    height=0.5,
                    color=color if color != t.divider else None,
                    font_size=18,
                )
                for n in names
            ]
        ).arrange_in_grid(
            cols=2 if len(names) > 3 else 1, buff=(0.12, 0.1), flow_order="dr"
        )
        columns.add(
            VGroup(scene.meta_text(title, font_size=15), chips).arrange(
                DOWN, aligned_edge=LEFT, buff=0.15
            )
        )
    columns.arrange(RIGHT, aligned_edge=UP, buff=0.45)
    skills_title = scene.meta_text(
        ".claude/skills/ · 18 skills · AGENTS.md → CLAUDE.md"
    )
    skills_block = VGroup(skills_title, columns).arrange(
        DOWN, aligned_edge=LEFT, buff=0.25
    )
    skills_block.next_to(head, DOWN, buff=0.4).to_edge(LEFT, buff=0.7)

    caller = code_block(scene, "claude-code.yml (the whole file)", CALLER, font_size=17)
    caller.scale_to_fit_width(min(caller.width, 8.2))
    caller.next_to(skills_block, DOWN, buff=0.4).align_to(skills_block, LEFT)
    onboard = VGroup(
        scene.body_text(
            "+ one uv run setup script\n+ one IAM trust-policy line", font_size=22
        ),
        scene.body_text(
            "13 repos\nreporting usage", font_size=30, color=t.accent_success
        ),
        scene.meta_text(
            "pytorch · test-infra · executorch\nao · torchtitan · helion …",
            font_size=14,
            uppercase=False,
        ),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    onboard.next_to(caller, RIGHT, buff=0.45).align_to(caller, DOWN)

    scene.play(FadeIn(head), FadeIn(skills_block, shift=UP * 0.1), run_time=0.7)
    scene.wait(0.2)
    scene.next_slide(
        notes="skills.repo — The model is generic; PyTorch knowledge lives in the repo as skills. pr-review encodes our checklist and BC rules and is used both by @claude and by humans locally. Triage and readiness review are bot-only skills. AGENTS.md is a symlink to CLAUDE.md so every agent reads the same rules."
    )
    scene.play(FadeIn(caller, shift=LEFT * 0.1), FadeIn(onboard), run_time=0.7)
    scene.wait(0.2)
    scene.next_slide(
        notes="skills.onboard — For other repos, Zain moved the workflow into test-infra. A new repo adds this file, runs one setup script that creates the bedrock environment, and adds one line to the IAM trust policy. Thirteen repos across pytorch and meta-pytorch report usage today."
    )
