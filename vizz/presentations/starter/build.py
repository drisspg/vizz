"""A copyable deck: one module per slide, one named registry for iteration."""

import os

from vizz.presentations import components, theme
from vizz.presentations.starter.slides import title, workflow

SLIDES = {"title": title, "workflow": workflow}


class StarterDeck(components.SlideBase):
    theme = theme.FRONTIER_LIGHT_THEME
    deck_mark = "Vizz / sketch to slide"

    def build_slides(self) -> None:
        selected = os.environ.get("SLIDE")
        if selected and selected not in SLIDES:
            raise ValueError(f"Unknown slide {selected!r}; choose: {', '.join(SLIDES)}")
        modules = [SLIDES[selected]] if selected else list(SLIDES.values())
        for index, module in enumerate(modules):
            if index:
                self.clear_stage()
            module.build(self)
