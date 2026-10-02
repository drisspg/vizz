from manim import DOWN, LEFT, RIGHT, UP, FadeIn, VGroup

from vizz.presentations.claude_pytorch.slides.common import box, code_block, header
from vizz.presentations.components import SlideBase

# research/pytorch_repo.md §2.2-2.4: claude-issue-triage-run.yml and
# .claude/skills/triaging-issues/SKILL.md hooks.
TOOLS = """
--model global.anthropic.claude-sonnet-5
--allowedTools "
  mcp__github__get_issue,
  mcp__github__get_issue_comments,
  mcp__github__update_issue,
  mcp__github__add_issue_comment,
  mcp__github__search_issues"
"""
HOOKS = [
    ("validate_issue_target.py", "only the issue that\ntriggered the run"),
    ("validate_labels.py", "282-label allowlist;\nnever sev / merge blocking"),
    ("add_bot_triaged.py", "tags every edit bot-triaged"),
]


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "The prompt is not the only defense")

    tools = code_block(
        scene, "triage: five GitHub tools, no shell", TOOLS, font_size=21
    )
    tools.scale_to_fit_width(min(tools.width, 6.6))
    tools.next_to(head, DOWN, buff=0.5).to_edge(LEFT, buff=0.6)

    hook_rows = VGroup()
    for name, what in HOOKS:
        hook_rows.add(
            VGroup(
                scene.meta_text(
                    name, font_size=17, color=t.accent_success, uppercase=False
                ),
                scene.body_text(what, font_size=22),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        )
    hook_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
    hooks_title = scene.meta_text("hooks: deterministic Python")
    hooks = VGroup(hooks_title, hook_rows).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
    hooks.next_to(tools, RIGHT, buff=0.6).align_to(tools, UP)

    human = box(
        scene,
        "high priority? → bot adds “triage review”, a human decides",
        width=11.5,
        height=0.85,
        color=t.accent_success,
        font_size=24,
    )
    human.to_edge(DOWN, buff=0.6)

    scene.play(FadeIn(head), FadeIn(tools, shift=UP * 0.1), run_time=0.6)
    scene.play(FadeIn(hooks, shift=LEFT * 0.1), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="guardrails.hooks — Triage runs on Sonnet with exactly five GitHub tools: read the issue, read comments, update it, comment, search. No shell, no file access. The smallest tool surface that can do the job. The prompt says only touch this issue. A hook enforces it in code. Another hook strips any label outside a 282-label allowlist, so a prompt injection cannot add ciflow or sev labels. Every change gets bot-triaged so we can audit it."
    )
    scene.play(FadeIn(human, shift=UP * 0.1), run_time=0.5)
    scene.wait(0.2)
    scene.next_slide(
        notes="guardrails.human — Consequential calls stay human. The bot cannot set high priority; it flags triage review and a person decides."
    )
