"""Diagram primitives for the FlexGEMM deck, following the frontier-editorial skill.

Semantic map (identical on every slide):
  green  fused / supported / allowed flow
  amber  the one thing in focus
  red    wasted traffic, rejected, or losing
  mask   unavailable / not yet (hatched or dotted)
"""

from manim import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    Arrow,
    DashedLine,
    DashedVMobject,
    Line,
    ManimColor,
    Rectangle,
    RoundedRectangle,
    Triangle,
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


# Visual language for compute vs memory, shared by every slide:
#   kernel(): a launch. Solid frame with a dark header strip reading "KERNEL".
#   tensor(): data in memory. Light matrix-ruled rectangle, never a header.


def kernel(
    scene: SlideBase,
    label: str,
    *,
    width: float,
    height: float = 0.95,
    color: str | None = None,
    font_size: int = 22,
) -> VGroup:
    """A GPU kernel launch; `color` (e.g. green for fused) tints frame and strip."""
    t = scene.theme
    ink = color or t.text
    frame = RoundedRectangle(
        corner_radius=CORNER,
        width=width,
        height=height,
        stroke_color=ink,
        stroke_width=1.6,
        fill_color=tint(scene, color, 0.14) if color else t.panel_fill,
        fill_opacity=1,
    )
    strip_height = 0.24
    strip = Rectangle(
        width=width, height=strip_height, stroke_width=0, fill_color=ink, fill_opacity=1
    ).align_to(frame, UP)
    tag = scene.meta_text("kernel", font_size=11, color=t.background)
    tag.move_to(strip).align_to(strip, LEFT).shift(RIGHT * 0.12)
    launch = Triangle(fill_color=t.background, fill_opacity=1, stroke_width=0).rotate(
        -PI / 2
    )
    launch.scale_to_fit_height(0.11).move_to(strip).align_to(strip, RIGHT).shift(
        LEFT * 0.12
    )
    text = scene.body_text(label, font_size=font_size)
    if text.width > width - 0.3:
        text.scale_to_fit_width(width - 0.3)
    text.move_to(frame).shift(DOWN * strip_height / 2)
    return VGroup(frame, strip, tag, launch, text)


def tensor(
    scene: SlideBase,
    label: str,
    *,
    width: float,
    height: float = 0.8,
    color: str | None = None,
    font_size: int = 22,
) -> VGroup:
    """A tensor in memory: ruled like a matrix, lighter than a kernel."""
    t = scene.theme
    stroke = color or t.muted_text
    frame = Rectangle(
        width=width,
        height=height,
        stroke_color=stroke,
        stroke_width=1.2,
        fill_color=tint(scene, color, 0.16) if color else t.background,
        fill_opacity=1,
    )
    rules = VGroup(
        *[
            Line(
                frame.get_corner(UP + LEFT) + DOWN * height * k / 4,
                frame.get_corner(UP + RIGHT) + DOWN * height * k / 4,
                stroke_color=stroke,
                stroke_width=0.6,
                stroke_opacity=0.35,
            )
            for k in (1, 2, 3)
        ],
        *[
            Line(
                frame.get_corner(UP + LEFT) + RIGHT * width * k / 5,
                frame.get_corner(DOWN + LEFT) + RIGHT * width * k / 5,
                stroke_color=stroke,
                stroke_width=0.6,
                stroke_opacity=0.35,
            )
            for k in (1, 2, 3, 4)
        ],
    )
    text = scene.body_text(label, font_size=font_size)
    plate = RoundedRectangle(
        corner_radius=0.03,
        width=text.width + 0.2,
        height=text.height + 0.12,
        stroke_width=0,
        fill_color=frame.get_fill_color(),
        fill_opacity=1,
    )
    text.move_to(frame)
    plate.move_to(text)
    return VGroup(frame, rules, plate, text)
