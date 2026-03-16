from __future__ import annotations

from manim import *
from manim_slides import Slide

from vizz.presentations.theme import LIGHT_THEME, Theme


class SlideBase(Slide):
    theme: Theme = LIGHT_THEME
    deck_mark: str = ""

    def construct(self) -> None:
        self.camera.background_color = ManimColor(self.theme.background)
        self.build_slides()

    def build_slides(self) -> None:
        raise NotImplementedError

    def title_text(self, text: str, font_size: int | None = None) -> Text:
        fs = font_size or self.theme.title_font_size
        title = Text(
            text,
            font=self.theme.display_font,
            font_size=fs,
            color=self.theme.text,
            weight=BOLD,
        )
        if title.width > self.theme.max_title_width:
            title.scale_to_fit_width(self.theme.max_title_width)
        return title

    def body_text(
        self, text: str, font_size: int | None = None, color: str | None = None
    ) -> Text:
        return Text(
            text,
            font=self.theme.sans_font,
            font_size=font_size or self.theme.body_font_size,
            color=color or self.theme.text,
        )

    def meta_text(
        self,
        text: str,
        font_size: int | None = None,
        color: str | None = None,
        uppercase: bool = True,
    ) -> Text:
        return Text(
            text.upper() if uppercase else text,
            font=self.theme.mono_font,
            font_size=font_size or self.theme.meta_font_size,
            color=color or self.theme.muted_text,
            weight=MEDIUM,
        )

    def section_header(self, text: str, font_size: int | None = None) -> VGroup:
        t = self.theme
        title = self.title_text(text, font_size=font_size)
        title_block: Mobject = title
        if self.deck_mark:
            furniture = self.meta_text(self.deck_mark, color=t.accent_primary)
            if furniture.width > t.max_title_width:
                furniture.scale_to_fit_width(t.max_title_width)
            title_block = VGroup(furniture, title).arrange(
                DOWN, aligned_edge=LEFT, buff=0.12
            )
        divider = Line(
            ORIGIN,
            RIGHT * min(title_block.width, 11.6),
            color=t.divider,
            stroke_width=1.4,
        )
        header = VGroup(title_block, divider).arrange(
            DOWN, aligned_edge=LEFT, buff=0.12
        )
        header.to_edge(UP, buff=0.28)
        return header

    def bullet_list(
        self, *items: str, font_size: int | None = None, color: str | None = None
    ) -> VGroup:
        rows = VGroup(
            *[self.bullet_row(item, font_size=font_size, color=color) for item in items]
        )
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.26)
        return rows

    def bullet_row(
        self, text: str, font_size: int | None = None, color: str | None = None
    ) -> VGroup:
        t = self.theme
        label = Text(
            text,
            font=t.sans_font,
            font_size=font_size or t.bullet_font_size,
            color=color or t.text,
        )
        dash = Line(LEFT * 0.08, RIGHT * 0.08, color=t.accent_primary, stroke_width=3)
        first_line_probe = Text(
            "Ag",
            font=t.sans_font,
            font_size=font_size or t.bullet_font_size,
            color=color or t.text,
        )
        dash.next_to(label, LEFT, buff=0.22)
        dash.set_y(label.get_top()[1] - first_line_probe.height * 0.52)
        return VGroup(dash, label)

    def panel(self, width: float, height: float) -> RoundedRectangle:
        t = self.theme
        return RoundedRectangle(
            corner_radius=t.panel_corner_radius,
            width=width,
            height=height,
            stroke_color=t.panel_stroke,
            stroke_width=t.panel_stroke_width,
            fill_color=t.panel_fill,
            fill_opacity=1,
        )

    def image_panel(self, path: str, width: float) -> Mobject:
        return ImageMobject(path).scale_to_fit_width(width)

    def labeled_panel(
        self,
        title: str,
        width: float,
        height: float,
        content: Mobject | None = None,
    ) -> Group:
        t = self.theme
        frame = self.panel(width, height)
        title_text = self.meta_text(
            title,
            font_size=max(t.meta_font_size + 1, 15),
            color=t.accent_primary,
            uppercase=False,
        )
        title_text.move_to(frame.get_top() + DOWN * 0.34)
        title_text.align_to(frame.get_left() + RIGHT * 0.32, LEFT)
        divider_y = frame.get_top()[1] - 0.62
        divider = Line(
            [frame.get_left()[0] + 0.32, divider_y, 0],
            [frame.get_right()[0] - 0.32, divider_y, 0],
            color=t.divider,
            stroke_width=1.2,
        )
        elements = [frame, title_text, divider]
        if content is not None:
            content.scale_to_fit_width(min(content.width, width - 0.7))
            content_height_limit = divider.get_y() - frame.get_bottom()[1] - 0.5
            if content.height > content_height_limit:
                content.scale_to_fit_height(content_height_limit)
            content.move_to(
                [
                    frame.get_center()[0],
                    (divider.get_y() + frame.get_bottom()[1]) / 2,
                    0,
                ]
            )
            elements.append(content)
        return Group(*elements)

    def themed_code(
        self,
        code_string: str,
        language: str = "python",
        font_size: int | None = None,
    ) -> Code:
        t = self.theme
        kwargs: dict = {}
        if t.code_background:
            kwargs["background_config"] = {
                "fill_color": t.code_background,
                "fill_opacity": 1.0,
                "stroke_color": t.panel_stroke,
                "stroke_width": 1.0,
            }
        return Code(
            code_string=code_string.strip(),
            language=language,
            add_line_numbers=False,
            paragraph_config={"font_size": font_size or t.code_font_size},
            background="rectangle",
            formatter_style=t.code_style,
            **kwargs,
        )

    def code_card(
        self,
        title: str,
        code_string: str,
        stroke_color: str,
        width: float = 10.7,
        height: float = 2.75,
        language: str = "python",
    ) -> Group:
        t = self.theme
        frame = RoundedRectangle(
            corner_radius=t.panel_corner_radius,
            width=width,
            height=height,
            stroke_color=stroke_color,
            stroke_width=2.0,
            fill_color=t.panel_fill,
            fill_opacity=1,
        )
        title_text = self.meta_text(
            title,
            font_size=max(t.meta_font_size + 1, 15),
            color=stroke_color,
            uppercase=False,
        )
        title_text.move_to(frame.get_top() + DOWN * 0.34)
        title_text.align_to(frame.get_left() + RIGHT * 0.3, LEFT)
        divider = Line(
            frame.get_left() + RIGHT * 0.3 + DOWN * 0.58,
            frame.get_right() + LEFT * 0.3 + DOWN * 0.58,
            color=t.divider,
            stroke_width=1.2,
        )
        code = self.themed_code(code_string, language=language)
        code.scale_to_fit_width(width - 1.0)
        if code.height > height - 0.95:
            code.scale_to_fit_height(height - 0.95)
        code.move_to(frame.get_center() + DOWN * 0.12)
        return Group(frame, title_text, divider, code)

    def clear_stage(self) -> None:
        current_mobjects = list(self.mobjects)
        if not current_mobjects:
            return
        self.play(
            LaggedStart(
                *[FadeOut(mob, shift=DOWN * 0.12) for mob in current_mobjects],
                lag_ratio=0.04,
            ),
            run_time=0.4,
        )
