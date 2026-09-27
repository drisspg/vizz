"""PyTorch Conference NA 2026 lightning talk: FlexGEMM epilogues.

Preview one slide: uv run vizz preview ptc_2026_flex_gemm --slide api
"""

import os
from dataclasses import replace

from vizz.presentations import components, theme
from vizz.presentations.ptc_2026_flex_gemm.slides import (
    api,
    contract,
    coverage,
    how_we_did_it,
    philox,
    problem,
    results,
    stack,
    status,
    title,
    wins,
)

SLIDES = {
    "title": title,
    "problem": problem,
    "api": api,
    "coverage": coverage,
    "contract": contract,
    "wins": wins,
    "results": results,
    "stack": stack,
    "philox": philox,
    "how_we_did_it": how_we_did_it,
    "status": status,
}


class Ptc2026FlexGemmDeck(components.SlideBase):
    # Darker muted and green than the blog palette: slides are read on projectors.
    theme = replace(
        theme.NUGGETS_LIGHT_THEME,
        muted_text="#4e5c52",
        accent_primary="#4a6b53",
        accent_success="#4a6b53",
    )
    deck_mark = "FlexGEMM / PTC 2026"
    # Seconds to hold each pause state; set for a watchable single-file video render.
    video_hold = float(os.environ.get("VIDEO_HOLD", "0"))

    def next_slide(self, *args, **kwargs) -> None:
        if self.video_hold:
            self.wait(self.video_hold)
        super().next_slide(*args, **kwargs)

    def build_slides(self) -> None:
        selected = os.environ.get("SLIDE")
        if selected and selected not in SLIDES:
            raise ValueError(f"Unknown slide {selected!r}; choose: {', '.join(SLIDES)}")
        modules = [SLIDES[selected]] if selected else list(SLIDES.values())
        for index, module in enumerate(modules):
            if index:
                self.clear_stage()
            module.build(self)
