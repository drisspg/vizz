"""PTCE 2026 -- FlexAttention + FlashAttention-4: Fast and Flexible.

Set SLIDE=<name> to render a single slide for fast iteration:
    SLIDE=title uv run manim ...build.py PTCE2026FlexFlash -ql -p
"""

import os

from vizz.presentations.components import SlideBase
from vizz.presentations.ptce_2026_flex_flash.slides import (
    adoption,
    blackwell_shift,
    flex_extensions,
    how_to_use,
    integration,
    more_coming,
    more_details,
    performance_gap,
    results,
    title,
)
from vizz.presentations.theme import FRONTIER_LIGHT_THEME

SLIDES = {
    "title": title,
    "flex_extensions": flex_extensions,
    "adoption": adoption,
    "performance_gap": performance_gap,
    "blackwell_shift": blackwell_shift,
    "integration": integration,
    "results": results,
    "how_to_use": how_to_use,
    "more_coming": more_coming,
    "more_details": more_details,
}


class PTCE2026FlexFlash(SlideBase):
    theme = FRONTIER_LIGHT_THEME
    skip_reversing = os.environ.get("SKIP_REVERSING", "0") == "1"

    def build_slides(self) -> None:
        slide_filter = os.environ.get("SLIDE")
        if slide_filter:
            SLIDES[slide_filter].build(self)
        else:
            for slide_mod in SLIDES.values():
                slide_mod.build(self)
