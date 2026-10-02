"""PyTorch Conference NA 2026 lightning talk: Claude in PyTorch CI, triage, and review.

Preview one slide: uv run vizz preview claude_pytorch --slide shape
"""

import os
from dataclasses import replace

from vizz.presentations import components, theme
from vizz.presentations.claude_pytorch.slides import (
    adoption,
    agents_vs_agents,
    closing,
    example,
    guardrails,
    lessons,
    problem,
    shape,
    skills,
    title,
    two_stage,
)

SLIDES = {
    "title": title,
    "problem": problem,
    "example": example,
    "shape": shape,
    "two_stage": two_stage,
    "guardrails": guardrails,
    "skills": skills,
    "agents_vs_agents": agents_vs_agents,
    "adoption": adoption,
    "lessons": lessons,
    "closing": closing,
}


class ClaudePytorchDeck(components.SlideBase):
    # Same projector-tuned palette as the FlexGEMM deck at the same conference.
    theme = replace(
        theme.NUGGETS_LIGHT_THEME,
        muted_text="#4e5c52",
        accent_primary="#4a6b53",
        accent_success="#4a6b53",
    )
    deck_mark = "Claude in PyTorch infra / PTC 2026"
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
