"""To run call manimgl vizz/flex/end_to_end.py AttentionScoresVisualization"""

import torch
# from manimlib import *
# from manimlib.constants import FRAME_WIDTH
# from manimlib import Code

from manim import *
# from manim.constants import FRAME_WIDTH

from manim_slides import Slide


def mask_mod(b, h, q_idx, kv_idx):
    _ = b, h
    return q_idx >= kv_idx


def matrix_center(group):
    return group[1].get_center()


class AttentionScoresVisualization(Slide):
    def highlight_dot_product(
        self, row_index, col_index, query_group, key_T_group, attention_group
    ):
        # Get the specific row and column
        query_row = query_group[1].get_rows()[row_index]
        key_col = key_T_group[1].get_columns()[col_index]

        # Create rectangles that match the dimensions of the row and column
        row_highlight = Rectangle(
            width=query_row.get_width(),
            height=query_row.get_height(),
            fill_color=YELLOW,
            fill_opacity=0.2,
            stroke_width=0,
        ).move_to(query_row)

        col_highlight = Rectangle(
            width=key_col.get_width(),
            height=key_col.get_height(),
            fill_color=BLUE,
            fill_opacity=0.2,
            stroke_width=0,
        ).move_to(key_col)

        # Get the target cell in the attention matrix
        target_cell = attention_group[1].get_entries()[row_index * 4 + col_index]

        cell_highlight = Rectangle(
            width=target_cell.get_width(),
            height=target_cell.get_height(),
            fill_color=GREEN,
            fill_opacity=0.2,
            stroke_width=0,
        ).move_to(target_cell)

        self.play(FadeIn(row_highlight), FadeIn(col_highlight))
        self.play(
            row_highlight.animate.move_to(cell_highlight),
            col_highlight.animate.move_to(cell_highlight),
            run_time=1.0,
        )
        self.play(
            FadeTransform(row_highlight, cell_highlight),
            FadeTransform(col_highlight, cell_highlight),
        )
        self.play(FadeOut(cell_highlight))

    def step(self):
        # Check if the current class is a Slide
        if isinstance(self, Slide):
            self.next_slide()
        else:
            self.wait(1)

    def construct(self):
        # Step 1: Create query and key
        query = torch.arange(8).view(4, 2)
        key = torch.arange(8).view(4, 2)

        query_matrix = Matrix(query.tolist())
        key_matrix = Matrix(key.tolist())

        query_label = Tex("Query").next_to(query_matrix, UP)
        key_label = Tex("Key").next_to(key_matrix, UP)

        query_group = VGroup(query_label, query_matrix)
        key_group = VGroup(key_label, key_matrix)

        initial_group = VGroup(query_group, key_group).arrange(RIGHT, buff=1).center()

        self.play(Write(initial_group))
        self.step()

        # Step 2: Transpose key
        key_T = key.T
        key_T_matrix = Matrix(key_T.tolist())
        # key_T_label = Tex(r"Key^T").next_to(key_T_matrix, UP)
        key_T_label = MathTex(r"\text{Key}^T").next_to(key_T_matrix, UP)
        key_T_group = VGroup(key_T_label, key_T_matrix)

        # Calculate the target positions
        left_target = ORIGIN + LEFT * 3
        right_target = ORIGIN + RIGHT * 3
        self.play(
            query_group.animate.move_to(left_target),
            ReplacementTransform(key_group, key_T_group.move_to(right_target)),
        )
        self.step()

        # Step 3: Matrix multiplication to produce attention scores
        attention_scores = torch.matmul(query, key_T)
        attention_matrix = Matrix(attention_scores.tolist())
        attention_label = Tex("Attention Scores").next_to(attention_matrix, UP)
        attention_group = VGroup(attention_label, attention_matrix)

        times = Tex("×").scale(1.5)
        equals = Tex("=").scale(1.5)

        # Create a copy of the current state of query_group and key_T_group
        query_group_copy = query_group.copy()
        key_T_group_copy = key_T_group.copy()

        # Arrange the equation
        equation = VGroup(
            query_group_copy, times, key_T_group_copy, equals, attention_group
        ).arrange(RIGHT, buff=0.5, aligned_edge=ORIGIN)

        times.move_to(
            midpoint(matrix_center(equation[0]), matrix_center(equation[2]))
            - 0.5 * RIGHT
        )
        equals.move_to(midpoint(matrix_center(equation[2]), matrix_center(equation[4])))
        scale_factor = min(1, (config.frame_width - 1) / equation.get_width())
        # Scale and center the equation
        scale_factor = min(1, (config.frame_width - 1) / equation.get_width())
        equation.scale(scale_factor).center()

        # Animate the transition smoothly
        self.play(
            Transform(query_group, equation[0]),
            Transform(key_T_group, equation[2]),
            run_time=1.0,
        )
        self.play(FadeIn(times), FadeIn(equals), FadeIn(attention_group), run_time=1)
        self.remove(query_group_copy, key_T_group_copy)
        self.step()

        for row, col in [(0, 0), (2, 1)]:
            self.highlight_dot_product(
                row, col, query_group, key_T_group, attention_group
            )

        # # Focus on the attention_scores matrix
        # causal_attention_text = (
        #     Text("Causal Attention", font_size=40).center().to_edge(UP)
        # )
        # self.play(
        #     FadeOut(query_group),
        #     FadeOut(key_T_group),
        #     FadeOut(times),
        #     FadeOut(equals),
        #     Transform(attention_group[0], causal_attention_text),
        #     attention_group[1].animate.scale(1.2).center(),
        # )
        # # Add title for attention scores
        # self.wait(1)

        # # Initial mask_mod_text setup
        # b, h, q_idx, kv_idx = "b", "h", "q_idx", "kv_idx"
        # code_string = """
        # def mask_mod({b}, {h}, {q_idx}, {kv_idx}):
        #     return {q_idx} >= {kv_idx}
        # """
        # mask_mod_text = Code(
        #     code_string = code_string.format(b=b, h=h, q_idx=q_idx, kv_idx=kv_idx),
        #     language="python",
        #     add_line_numbers=False,

        #     paragraph_config={"font": "Monospace"},
        # )

        # mask_mod_text.to_edge(DOWN, buff=0.5)

        # # Display initial setup
        # self.play(Write(mask_mod_text))
        # self.step()

        # # Apply mask_mod function
        # for i in range(4):
        #     for j in range(4):
        #         b, h, q_idx, kv_idx = 0, 0, i, j
        #         new_call_code = code_string.format(b=b, h=h, q_idx=q_idx, kv_idx=kv_idx)
        #         new_call_text = Code(
        #             code_string=new_call_code,
        #             language="python",
        #             paragraph_config={"font": "Monospace"},
        #             add_line_numbers=False
        #         ).to_edge(DOWN, buff=0.5)

        #         if mask_mod(0, 0, i, j):
        #             color = GREEN
        #             new_value = attention_group[1].get_entries()[i * 4 + j].copy()
        #             output_text = Text("Keep", font_size=20, color=GREEN)
        #         else:
        #             color = RED
        #             new_value = Text("-inf", font_size=24, color=RED).move_to(
        #                 attention_group[1].get_entries()[i * 4 + j]
        #             )
        #             output_text = Text("Mask", font_size=20, color=RED)

        #         output_text.next_to(mask_mod_text, UP, buff=0.2)
        #         new_value.set_color(color)

        #         self.play(
        #             Transform(mask_mod_text, new_call_text),
        #             Flash(
        #                 attention_group[1].get_entries()[i * 4 + j],
        #                 color=color,
        #                 flash_radius=0.3,
        #             ),
        #             Transform(attention_group[1].get_entries()[i * 4 + j], new_value),
        #             FadeIn(output_text),
        #             run_time=0.3,  # Reduced run_time
        #         )
        #         self.wait(0.1)  # Reduced wait time
        #         self.play(
        #             FadeOut(output_text), run_time=0.2
        #         )  # Reduced run_time for FadeOut

        # self.step()  # Reduced final wait time


# To run this animation, use the command:
# manim -q k vizz/flex/end_to_end.py.py AttentionScoresVisualization
#  manim-slides render vizz/flex/end_to_end.py AttentionScoresVisualization
