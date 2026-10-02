from manim import DOWN, LEFT, RIGHT, UP, Create, CurvedArrow, FadeIn, VGroup

from vizz.presentations.claude_pytorch.slides.common import box, flow, header
from vizz.presentations.components import SlideBase

# research/pytorch_repo.md §3 and research/test_infra.md §4:
# claude-autorevert-advisor.yml (#177404), verdicts in misc.autorevert_advisor_verdicts
# read back by autorevert (test-infra #7908), Dr.CI auto-dispatch (#8178),
# advisor-coverage backfill (#8569; research/hud_usage.md Aug 19-21 spike).
VERDICTS = ["related", "unsure", "not_related", "infra_issue", "garbage"]


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "Agents investigating CI failures")

    signal = box(
        scene,
        "autorevert / Dr.CI",
        width=3.3,
        height=1.1,
        font_size=25,
        color=t.accent_success,
        sublabel="sees a red signal",
    )
    advisor = box(
        scene,
        "Claude CI Advisor",
        width=3.3,
        height=1.1,
        font_size=25,
        color=t.accent_secondary,
        sublabel="reads logs + diff",
    )
    verdict_chips = VGroup(
        *[box(scene, v, width=2.1, height=0.5, font_size=19) for v in VERDICTS]
    ).arrange(DOWN, buff=0.08)
    verdict_title = scene.meta_text("--json-schema verdict", font_size=13)
    verdict = VGroup(verdict_title, verdict_chips).arrange(DOWN, buff=0.15)
    row = VGroup(signal, advisor, verdict).arrange(RIGHT, buff=1.1)
    row.next_to(head, DOWN, buff=0.4).set_x(0)
    dispatch = flow(scene, signal.get_right(), advisor.get_left())
    dispatch_label = scene.meta_text("dispatch", font_size=13).next_to(
        dispatch, UP, buff=0.08
    )
    emit = flow(
        scene, advisor.get_right(), verdict.get_left(), color=t.accent_secondary
    )
    back = CurvedArrow(
        verdict.get_bottom() + DOWN * 0.1,
        signal.get_bottom() + DOWN * 0.1,
        angle=-0.5,
        color=t.accent_success,
        stroke_width=2.2,
        tip_length=0.16,
    )
    back_label = scene.meta_text(
        "ClickHouse → autorevert weighs the verdict",
        font_size=15,
        color=t.accent_success,
    ).next_to(back, DOWN, buff=0.05)

    principles = VGroup(
        scene.body_text("“A missing baseline is not a green baseline.”", font_size=23),
        scene.body_text("“When in doubt … prefer unsure.”", font_size=23),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    coverage = VGroup(
        scene.meta_text("coverage lambda · Aug 19", color=t.accent_secondary),
        scene.body_text(
            "~40% of trunk reds had no verdict\n~22k backfilled in 3 days", font_size=21
        ),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    bottom = VGroup(principles, coverage).arrange(RIGHT, buff=1.2, aligned_edge=UP)
    bottom.to_edge(DOWN, buff=0.5)

    scene.play(FadeIn(head), FadeIn(signal), run_time=0.5)
    scene.play(Create(dispatch), FadeIn(dispatch_label), FadeIn(advisor), run_time=0.6)
    scene.play(Create(emit), FadeIn(verdict), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="agents_vs_agents.verdict — Autorevert is itself an automated agent watching trunk. When it sees an early failure pattern it dispatches the Claude CI Advisor, which reads the logs and the suspect diff and must answer in a JSON schema: related, unsure, not_related, infra_issue, or garbage, with confidence and reasoning. It launched at 13 of 13 on its eval set."
    )
    scene.play(Create(back), FadeIn(back_label), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="agents_vs_agents.loop — Verdicts land in ClickHouse and autorevert reads them back when deciding whether to revert. Dr.CI dispatches the same advisor on PR failures and renders the verdict inline in its comment."
    )
    scene.play(FadeIn(bottom, shift=UP * 0.1), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="agents_vs_agents.principles — The prompt encodes CI reasoning maintainers learned the hard way, and biases toward unsure over a false dismissal. In August Jean added a coverage lambda: about 40% of trunk reds never got a verdict, so it backfilled roughly 22 thousand historical reds in three days. Those verdicts are tagged so they can never trigger a revert; they are data for understanding flaky jobs."
    )
