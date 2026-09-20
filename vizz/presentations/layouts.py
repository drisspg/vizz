"""Small layout recipes that preserve comparisons instead of fitting each side independently."""

from manim import DOWN, LEFT, RIGHT, UP, Group, Line, Mobject, Text

from vizz.presentations.theme import Theme


class Comparison(Group):
    """Place two visuals at the same scale with aligned labels and a quiet divider.

    The input objects are positioned in place. Construct overlays after this
    layout, or include them in the inputs, so they share the same transform.
    """

    def __init__(
        self,
        left: Mobject,
        right: Mobject,
        *,
        labels: tuple[str, str],
        theme: Theme,
        width: float = 11.8,
        height: float = 3.8,
        gap: float = 0.9,
    ) -> None:
        if not 0 < gap < width or height <= 0:
            raise ValueError("Comparison needs width > gap > 0 and height > 0.")
        if (
            left is right
            or min(left.width, right.width, left.height, right.height) <= 0
        ):
            raise ValueError("Comparison needs two distinct, nonempty visuals.")
        slot_width = (width - gap) / 2
        scale = min(
            1,
            slot_width / max(left.width, right.width),
            height / max(left.height, right.height),
        )
        left.scale(scale).move_to(LEFT * (slot_width + gap) / 2)
        right.scale(scale).move_to(RIGHT * (slot_width + gap) / 2)
        self.left = left
        self.right = right
        self.labels = Group()
        for label, content in zip(labels, (left, right), strict=True):
            text = Text(
                label, font=theme.mono_font, font_size=19, color=theme.muted_text
            )
            if text.width > slot_width:
                text.scale_to_fit_width(slot_width)
            text.move_to(content.get_center() + UP * (height / 2 + 0.5))
            self.labels.add(text)
        self.divider = Line(
            DOWN * height / 2,
            UP * (height / 2 + 0.7),
            color=theme.divider,
            stroke_width=1,
        )
        super().__init__(left, right, self.labels, self.divider)
