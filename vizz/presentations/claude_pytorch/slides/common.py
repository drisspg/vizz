"""Diagram primitives for the Claude-in-PyTorch deck.

Semantic map (identical on every slide):
  green  trusted / privileged / a human decision
  amber  the agent (Claude)
  red    untrusted input or a failure mode
"""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    Arrow,
    DashedLine,
    DashedVMobject,
    ManimColor,
    RoundedRectangle,
    VGroup,
    interpolate_color,
)

from vizz.presentations.components import SlideBase

HAIRLINE = 1.0
CORNER = 0.04


def tint(scene: SlideBase, color: str, amount: float = 0.12) -> ManimColor:
    """Faint semantic tint mixed into paper, like color-mix(srgb, c 12%, paper)."""
    return interpolate_color(
        ManimColor(scene.theme.background), ManimColor(color), amount
    )


def box(
    scene: SlideBase,
    label: str,
    *,
    width: float,
    height: float = 0.9,
    color: str | None = None,
    sublabel: str = "",
    font_size: int = 22,
) -> VGroup:
    """Token-style box: rule hairline by default; a semantic `color` adds stroke and tint."""
    t = scene.theme
    frame = RoundedRectangle(
        corner_radius=CORNER,
        width=width,
        height=height,
        stroke_color=color or t.divider,
        stroke_width=1.6 if color else HAIRLINE,
        fill_color=tint(scene, color, 0.18) if color else t.panel_fill,
        fill_opacity=1,
    )
    text = scene.body_text(label, font_size=font_size)
    content = VGroup(text)
    if sublabel:
        content.add(scene.meta_text(sublabel, font_size=13))
        content.arrange(DOWN, buff=0.08)
    if content.width > width - 0.3:
        content.scale_to_fit_width(width - 0.3)
    content.move_to(frame)
    return VGroup(frame, content)


def empty_slot(
    scene: SlideBase, label: str, *, width: float, height: float = 0.9
) -> VGroup:
    """Dotted muted outline for something that no longer exists (never materialized)."""
    t = scene.theme
    frame = DashedVMobject(
        RoundedRectangle(
            corner_radius=CORNER,
            width=width,
            height=height,
            stroke_color=t.muted_text,
            stroke_width=HAIRLINE,
        ),
        num_dashes=40,
        dashed_ratio=0.45,
    )
    text = scene.meta_text(label, font_size=13).move_to(frame)
    return VGroup(frame, text)


def flow(scene: SlideBase, start, end, *, color: str | None = None) -> Arrow:
    """Allowed flow: thin solid arrow with a small head."""
    return Arrow(
        start,
        end,
        buff=0.08,
        color=color or scene.theme.muted_text,
        stroke_width=2.2,
        max_tip_length_to_length_ratio=0.12,
        tip_length=0.16,
    )


def wasted_flow(scene: SlideBase, start, end) -> VGroup:
    """Forbidden / wasted flow: red dashed line with a small head."""
    line = DashedLine(
        start, end, color=scene.theme.accent_danger, stroke_width=2.2, dash_length=0.08
    )
    line.add_tip(tip_length=0.16, tip_width=0.14)
    return line


def punchline(scene: SlideBase, text: str, font_size: int = 30) -> VGroup:
    """The one sentence to retain: 1px green/rule border, no fill."""
    t = scene.theme
    words = scene.body_text(text, font_size=font_size)
    border = RoundedRectangle(
        corner_radius=CORNER,
        width=words.width + 0.8,
        height=words.height + 0.5,
        stroke_color=interpolate_color(
            ManimColor(t.divider), ManimColor(t.accent_success), 0.6
        ),
        stroke_width=HAIRLINE + 0.4,
    ).move_to(words)
    return VGroup(border, words)


def header(scene: SlideBase, text: str) -> VGroup:
    """Left-aligned section header."""
    return scene.section_header(text).to_edge(LEFT, buff=0.6)


def code_block(scene: SlideBase, title: str, code: str, font_size: int) -> VGroup:
    """Mono label above a themed code block, sized by font rather than a frame."""
    label = scene.meta_text(title, font_size=15)
    block = scene.themed_code(code, font_size=font_size)
    return VGroup(label, block).arrange(DOWN, aligned_edge=LEFT, buff=0.15)


def comment(
    scene: SlideBase,
    author: str,
    body: VGroup,
    *,
    color: str,
    width: float,
    meta: str = "",
) -> VGroup:
    """GitHub-style comment card: author line above a tinted body."""
    who = scene.meta_text(author, font_size=14, color=color, uppercase=False)
    head = VGroup(who)
    if meta:
        head.add(scene.meta_text(meta, font_size=12, uppercase=False))
        head.arrange(RIGHT, buff=0.3)
    content = VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    if content.width > width - 0.5:
        content.scale_to_fit_width(width - 0.5)
    frame = RoundedRectangle(
        corner_radius=CORNER,
        width=width,
        height=content.height + 0.45,
        stroke_color=color,
        stroke_width=1.4,
        fill_color=tint(scene, color, 0.08),
        fill_opacity=1,
    )
    content.move_to(frame).align_to(frame.get_left() + RIGHT * 0.25, LEFT)
    return VGroup(frame, content)
