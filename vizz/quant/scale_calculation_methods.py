"""Visualize different scale calculation modes.

Compares FLOOR and RCEIL scale calculation modes and their trade-offs
in quantization accuracy and dynamic range.

To run:
manim-slides render vizz/quant/scale_calculation_methods.py ScaleCalculationMethods -ql
manim-slides present ScaleCalculationMethods
"""

import torch
from manim import *
from manim_slides import Slide

from vizz.quant.mx_base import COLORS


class ScaleCalculationMethods(Slide):
    """Visualize different scale calculation modes."""

    def construct(self):
        """Compare FLOOR and RCEIL scale calculation modes."""
        # Title
        title = Text("Scale Calculation Methods", font_size=48, color=COLORS["text"])
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Show example block
        abs_max = self._show_example_block()
        self.next_slide()

        # Show FLOOR mode
        self._show_floor_mode(abs_max)
        self.next_slide()

        # Show RCEIL mode
        self._show_rceil_mode(abs_max)
        self.next_slide()

        # Explain trade-offs
        self._show_tradeoffs()
        self.next_slide()

    def _show_example_block(self):
        """Display example block with abs_max value."""
        block_values = torch.tensor([2.5, -3.8, 1.2, -4.1, 2.9, 3.3, -2.1, 1.8])
        abs_max = torch.max(torch.abs(block_values)).item()

        block_rect = Rectangle(width=4, height=1, color=COLORS["block_boundary"])
        block_rect.move_to(UP * 2)

        block_label = Text(
            f"Block values (abs_max = {abs_max:.1f})",
            font_size=24,
            color=COLORS["text"],
        )
        block_label.next_to(block_rect, UP, buff=0.3)

        self.play(Create(block_rect), Write(block_label))
        return abs_max

    def _show_floor_mode(self, abs_max):
        """Show FLOOR mode calculation."""
        floor_group = VGroup()
        floor_title = Text("FLOOR Mode:", font_size=28, color=COLORS["mx_color"])
        floor_title.move_to(LEFT * 3 + UP * 0.5)

        floor_formula = MathTex(
            r"\\text{scale} = \\lfloor \\log_2(\\text{abs\\_max}) \\rfloor",
            font_size=20,
            color=COLORS["text"],
        )
        floor_formula.next_to(floor_title, DOWN, buff=0.3)

        floor_calc = MathTex(
            f"= \\\\lfloor \\\\log_2({abs_max:.1f}) \\\\rfloor = \\\\lfloor 2.04 \\\\rfloor = 2",
            font_size=18,
            color=COLORS["text"],
        )
        floor_calc.next_to(floor_formula, DOWN, buff=0.2)

        floor_result = Text(
            f"E8M0: {2 + 127} (biased)", font_size=18, color=COLORS["computed_scale"]
        )
        floor_result.next_to(floor_calc, DOWN, buff=0.3)

        floor_group.add(floor_title, floor_formula, floor_calc, floor_result)
        self.play(Write(floor_group))

    def _show_rceil_mode(self, abs_max):
        """Show RCEIL mode calculation."""
        rceil_group = VGroup()
        rceil_title = Text("RCEIL Mode:", font_size=28, color=COLORS["nvfp_color"])
        rceil_title.move_to(RIGHT * 3 + UP * 0.5)

        rceil_formula = MathTex(
            r"\\text{scale} = \\lceil \\log_2\\left(\\frac{\\text{abs\\_max}}{\\text{max\\_repr}}\\right) \\rceil",
            font_size=18,
            color=COLORS["text"],
        )
        rceil_formula.next_to(rceil_title, DOWN, buff=0.3)

        rceil_calc = MathTex(
            f"= \\\\lceil \\\\log_2\\\\left(\\\\frac{{{abs_max:.1f}}}{{6.0}}\\\\right) \\\\rceil",
            font_size=16,
            color=COLORS["text"],
        )
        rceil_calc.next_to(rceil_formula, DOWN, buff=0.2)

        rceil_result = Text(
            "Better accuracy", font_size=18, color=COLORS["computed_scale"]
        )
        rceil_result.next_to(rceil_calc, DOWN, buff=0.3)

        rceil_group.add(rceil_title, rceil_formula, rceil_calc, rceil_result)
        self.play(Write(rceil_group))

    def _show_tradeoffs(self):
        """Explain the trade-offs between methods."""
        impact_title = Text(
            "Impact on Quantization:", font_size=32, color=COLORS["text"]
        )
        impact_title.move_to(DOWN * 2)

        impact_text = Text(
            "Different modes trade off between:\\n"
            + "• Dynamic range coverage\\n"
            + "• Quantization accuracy\\n"
            + "• Hardware compatibility",
            font_size=20,
            color=COLORS["text"],
        )
        impact_text.next_to(impact_title, DOWN, buff=0.3)

        self.play(Write(impact_title), Write(impact_text))
