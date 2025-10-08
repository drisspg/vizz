"""Test DecimalMatrix text visibility on white background."""

import torch
from manim import *

config.background_color = WHITE

COLORS = {
    "text": BLACK,
    "bf16_color": BLUE_C,
}


class TestTextVisibility(Scene):
    def construct(self):
        # Test with exact settings from our animation
        data = torch.randn(2, 32).numpy()

        title = Text("Testing 2x32 DecimalMatrix", color=BLACK)
        title.to_edge(UP)
        self.add(title)

        # Exact same settings as our animation
        matrix = DecimalMatrix(
            data,
            element_to_mobject_config={
                "num_decimal_places": 1,
                "color": COLORS["text"],
            },
            left_bracket="[",
            right_bracket="]",
            bracket_config={"color": COLORS["bf16_color"]},
            h_buff=0.3,
            v_buff=0.3,
        )

        matrix.scale(0.6)
        matrix.move_to([0, 0, 0])

        self.add(matrix)

        # Add some debug info
        info = Text(f"Entries: {len(matrix.get_entries())}", color=BLACK, font_size=16)
        info.to_edge(DOWN)
        self.add(info)
