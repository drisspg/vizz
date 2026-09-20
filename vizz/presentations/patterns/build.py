"""Copyable Nuggets-style technical diagrams; select a slide via SLIDE."""

import os

from vizz.presentations import components, theme
from vizz.presentations.patterns.slides import comparison, focus, tiles

SLIDES = {"focus": focus, "tiles": tiles, "comparison": comparison}


class PatternGallery(components.SlideBase):
    theme = theme.NUGGETS_LIGHT_THEME
    deck_mark = "Nuggets / visual patterns"

    def build_slides(self) -> None:
        selected = os.environ.get("SLIDE")
        if selected and selected not in SLIDES:
            raise ValueError(f"Unknown slide {selected!r}; choose: {', '.join(SLIDES)}")
        modules = [SLIDES[selected]] if selected else list(SLIDES.values())
        for index, module in enumerate(modules):
            if index:
                self.clear_stage()
            module.build(self)
