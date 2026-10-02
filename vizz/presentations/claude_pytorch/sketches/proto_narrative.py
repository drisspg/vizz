"""Prototype: the PR thread that types itself (ideas_narrative.md, idea 1).

Render: uv run manim -ql vizz/presentations/claude_pytorch/sketches/proto_narrative.py ProtoNarrative
Facts: research/pytorch_repo.md §7.2 (pytorch/pytorch#176266, run 22680756884, d9e65e8cfed).
"""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    AddTextLetterByLetter,
    Create,
    FadeIn,
    ReplacementTransform,
    RoundedRectangle,
    SurroundingRectangle,
    Text,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
)

from vizz.presentations.claude_pytorch.build import ClaudePytorchDeck
from vizz.presentations.claude_pytorch.slides.common import (
    CORNER,
    comment,
    flow,
    header,
    tint,
)
from vizz.presentations.components import SlideBase

ASK = "@claude look for any subtle bugs on this pr"
FINDINGS = [
    ("High", "compile error: MKLDNN branch uses undefined mat_a / mat_b"),
    ("Medium", "copy-paste bug: scale_b_opt built from scale_a"),
    ("Low", "*out_dtype dereferenced without a nullopt check"),
]
ELAPSED = 4 * 60 + 39  # 4m 39s
# Schematic of ScaledBlas.cpp:454 vs landed line 456; not the literal C++.
BEFORE = "- scale_b_opt  <-  scale_a.empty() … scale_a[0]"
AFTER = "+ scale_b_opt  <-  scale_b.empty() … scale_b[0]"


def _glyphs(line: Text, source: str, token: str, occurrence: int) -> VGroup:
    """Glyph slice of `line` for the n-th `token` (Text drops whitespace glyphs)."""
    start = -1
    for _ in range(occurrence + 1):
        start = source.index(token, start + 1)
    skip = sum(not c.isspace() for c in source[:start])
    return VGroup(*line[skip : skip + len(token.replace(" ", ""))])


def _clock(scene: SlideBase, seconds: float) -> str:
    s = int(seconds)
    return f"claude[bot] working · {s // 60}m {s % 60:02d}s"


def build(scene: SlideBase) -> None:
    t = scene.theme
    head = header(scene, "@claude on a real PR")
    pr = (
        scene.meta_text(
            "pytorch/pytorch#176266 · external contributor · scaled_mm_v2 on CPU",
            uppercase=False,
        )
        .next_to(head, DOWN, buff=0.25)
        .align_to(head, LEFT)
    )

    ask_body = scene.body_text(ASK, font_size=26)
    ask = comment(
        scene,
        "drisspg",
        VGroup(ask_body),
        color=t.accent_success,
        width=7.6,
        meta="maintainer",
    )
    ask.next_to(pr, DOWN, buff=0.3).align_to(head, LEFT)

    # Reply card (final layout first, so the chip can morph into it).
    rows = VGroup()
    for severity, text in FINDINGS:
        color = t.accent_danger if severity != "Low" else t.muted_text
        rows.add(
            VGroup(
                scene.meta_text(severity, font_size=16, color=color),
                scene.body_text(text, font_size=22),
            ).arrange(RIGHT, buff=0.25)
        )
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    text_x = max(row[0].get_right()[0] for row in rows) + 0.25
    for row in rows:
        row[1].align_to([text_x, 0, 0], LEFT)
    reply = comment(
        scene, "claude[bot]", rows, color=t.accent_secondary, width=9.4, meta="4m 39s"
    )
    reply.next_to(ask, DOWN, buff=0.3).align_to(ask, LEFT).shift(RIGHT * 0.7)

    tracker = ValueTracker(0)
    chip_frame = RoundedRectangle(
        corner_radius=CORNER,
        width=4.6,
        height=0.55,
        stroke_color=t.accent_secondary,
        stroke_width=1.4,
        fill_color=tint(scene, t.accent_secondary, 0.08),
        fill_opacity=1,
    ).move_to(reply.get_corner(UP + LEFT), aligned_edge=UP + LEFT)
    clock = always_redraw(
        lambda: scene.meta_text(
            _clock(scene, tracker.get_value()),
            font_size=15,
            color=t.accent_secondary,
            uppercase=False,
        ).move_to(chip_frame)
    )

    # Diff hunk: schematic before/after of the scale_b_opt line.
    file_label = scene.meta_text(
        "ScaledBlas.cpp · schematic of :454", font_size=13, uppercase=False
    )
    before = Text(BEFORE, font=t.mono_font, font_size=20, color=t.text)
    after = Text(AFTER, font=t.mono_font, font_size=20, color=t.text)
    lines = VGroup(before, after).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    before_bg = RoundedRectangle(
        corner_radius=CORNER,
        width=lines.width + 0.4,
        height=before.height + 0.2,
        stroke_width=0,
        fill_color=tint(scene, t.accent_danger, 0.14),
        fill_opacity=1,
    ).move_to(before)
    after_bg = (
        before_bg.copy()
        .set_fill(tint(scene, t.accent_success, 0.2), opacity=1)
        .move_to(after)
        .align_to(before_bg, LEFT)
    )
    hunk = VGroup(file_label, VGroup(before_bg, before)).arrange(
        DOWN, aligned_edge=LEFT, buff=0.12
    )
    after_group = VGroup(after_bg, after)
    after_group.next_to(hunk[1], DOWN, buff=0.06).align_to(hunk[1], LEFT)
    VGroup(hunk, after_group).next_to(reply, DOWN, buff=0.45).align_to(
        reply, LEFT
    ).shift(RIGHT * 0.4)

    def marks(line: Text, source: str, token: str, occurrences, color) -> VGroup:
        return VGroup(
            *[
                SurroundingRectangle(
                    _glyphs(line, source, token, n),
                    color=color,
                    buff=0.04,
                    stroke_width=2.4,
                    corner_radius=CORNER,
                )
                for n in occurrences
            ]
        )

    bug = marks(before, BEFORE, "scale_a", (0, 1), t.accent_secondary)
    medium = SurroundingRectangle(
        rows[1], color=t.accent_secondary, buff=0.06, stroke_width=1.6
    )
    pointer = flow(
        scene,
        [bug[0].get_x(), reply.get_bottom()[1], 0],
        bug[0].get_top(),
        color=t.accent_secondary,
    )
    fixed = marks(after, AFTER, "scale_b", (1, 2), t.accent_success)
    verdict = VGroup(
        scene.body_text(
            "✓ fixed before landing · d9e65e8", font_size=22, color=t.accent_success
        ),
        scene.meta_text("maintainers decided", font_size=14, color=t.accent_success),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
    verdict.next_to(after_group, RIGHT, buff=0.5).align_to(after_group, DOWN)

    # Beat 1: the comment types itself; the clock runs to 4m 39s.
    scene.play(FadeIn(head), FadeIn(pr), run_time=0.4)
    scene.play(FadeIn(VGroup(ask[0], ask[1][0])), run_time=0.3)
    scene.play(AddTextLetterByLetter(ask_body, time_per_char=0.03))
    scene.play(FadeIn(chip_frame), FadeIn(clock), run_time=0.3)
    scene.play(tracker.animate.set_value(ELAPSED), run_time=2.2, rate_func=linear)
    clock.clear_updaters()
    scene.wait(0.2)
    scene.next_slide(
        notes="example.ask — An external contributor's FP8 PR adding a CPU path for scaled_mm. The whole interface is a comment. I asked it to look for subtle bugs, and it went off for four and a half minutes."
    )

    # Beat 2: chip becomes the review; the copy-paste bug lights up in the diff.
    scene.play(ReplacementTransform(VGroup(chip_frame, clock), reply), run_time=0.6)
    scene.play(FadeIn(hunk, shift=UP * 0.1), run_time=0.4)
    scene.play(Create(medium), Create(pointer), Create(bug), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="example.reply — A compile error in the MKLDNN branch that only some platforms build, and a copy-paste bug: scale_b came from scale_a, in both the condition and the value. Both real."
    )

    # Beat 3: the landed fix; humans made the call.
    scene.play(FadeIn(after_group, shift=UP * 0.1), Create(fixed), run_time=0.5)
    scene.play(FadeIn(verdict, shift=RIGHT * 0.1), run_time=0.4)
    scene.wait(0.2)
    scene.next_slide(
        notes="example.outcome — Both fixed before landing in d9e65e8. Humans pushed back on the low-severity note and chose their own fix. Claude finds things; maintainers decide."
    )


class ProtoNarrative(ClaudePytorchDeck):
    def build_slides(self) -> None:
        build(self)
