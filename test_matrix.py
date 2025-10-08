"""Quick test to see DecimalMatrix text visibility."""

import torch
from manim import *

# Match our config
config.background_color = WHITE

COLORS = {
    "text": BLACK,
    "bf16_color": BLUE_C,
}


class TestMatrix(Scene):
    def construct(self):
        # Example 1: Basic matrix with integers
        basic = Matrix(
            [[1, 2, 3], [4, 5, 6]],
        )
        basic.set_color(BLACK)
        basic.scale(0.7).to_edge(UP + LEFT)
        basic_label = Text("Basic Matrix", color=BLACK, font_size=20)
        basic_label.next_to(basic, DOWN, buff=0.2)
        self.add(basic, basic_label)

        # Example 2: Matrix with decimal numbers (using DecimalMatrix)
        decimal_data = [[1.23, 4.56], [7.89, 0.12]]
        decimal = DecimalMatrix(
            decimal_data,
            element_to_mobject_config={"num_decimal_places": 2},
        )
        decimal.set_color(BLACK)
        decimal.scale(0.7).next_to(basic, RIGHT, buff=1)
        decimal_label = Text("DecimalMatrix (2 places)", color=BLACK, font_size=20)
        decimal_label.next_to(decimal, DOWN, buff=0.2)
        self.add(decimal, decimal_label)

        # Example 3: Matrix with custom brackets and spacing
        custom = Matrix(
            [[10, 20], [30, 40]],
            left_bracket="(",
            right_bracket=")",
            h_buff=0.8,  # horizontal spacing
            v_buff=0.5,  # vertical spacing
        )
        custom.set_color(BLACK)
        custom.get_brackets()[0].set_color(BLUE)  # Left bracket
        custom.get_brackets()[1].set_color(BLUE)  # Right bracket
        custom.scale(0.7).to_edge(UP + RIGHT)
        custom_label = Text("Custom brackets & spacing", color=BLACK, font_size=20)
        custom_label.next_to(custom, DOWN, buff=0.2)
        self.add(custom, custom_label)

        # Example 4: Larger matrix from torch tensor
        torch_data = torch.randn(3, 5).numpy()
        large = DecimalMatrix(
            torch_data,
            element_to_mobject_config={"num_decimal_places": 1},
            h_buff=0.3,
            v_buff=0.3,
        )
        large.set_color(BLACK)
        large.scale(0.5).move_to([0, -0.5, 0])
        large_label = Text("3x5 from torch.randn", color=BLACK, font_size=20)
        large_label.next_to(large, DOWN, buff=0.2)
        self.add(large, large_label)

        # Example 5: Colored rows/columns
        colored = Matrix(
            [[1, 2], [3, 4], [5, 6]],
        )
        colored.set_column_colors(RED, BLUE)
        colored.scale(0.7).to_edge(DOWN + LEFT)
        colored_label = Text("Colored columns", color=BLACK, font_size=20)
        colored_label.next_to(colored, DOWN, buff=0.2)
        self.add(colored, colored_label)

        # Example 6: With background rectangles
        with_bg = Matrix(
            [[7, 8], [9, 10]],
            add_background_rectangles_to_entries=True,
        )
        with_bg.set_color(BLACK)
        with_bg.scale(0.7).to_edge(DOWN + RIGHT)
        bg_label = Text("With entry backgrounds", color=BLACK, font_size=20)
        bg_label.next_to(with_bg, DOWN, buff=0.2)
        self.add(with_bg, bg_label)
