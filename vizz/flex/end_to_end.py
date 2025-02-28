"""To run call manimgl vizz/flex/end_to_end.py AttentionScoresVisualization"""

import torch
from manim import *
from manim_slides import Slide

# Custom configuration to use a light theme for presentations
config.background_color = WHITE

# Constants for consistent styling
COLORS = {
    "text": BLACK,
    "matrix": DARK_GRAY,
    "bracket": DARK_GRAY,
    "highlight": {
        "query": GOLD_D,
        "key": BLUE_D,
        "result": GREEN_D,
    }
}

OPACITY = {
    "highlight": 0.2
}


class MatrixHelper:
    """Helper class for matrix operations and visualizations"""

    @staticmethod
    def create_matrix(data, label_text=None, with_brackets=True):
        """Create a matrix with optional label"""
        matrix_data = data.tolist() if isinstance(data, torch.Tensor) else data

        # Handle float formatting if needed
        if isinstance(matrix_data[0][0], float):
            matrix_data = [[f"{x:.2f}" for x in row] for row in matrix_data]

        matrix = Matrix(
            matrix_data,
            element_to_mobject_config={"color": COLORS["matrix"]},
            bracket_config={"color": COLORS["bracket"]} if with_brackets else {"opacity": 0}
        )

        if label_text:
            if r"\text" in label_text:
                label = MathTex(label_text, color=COLORS["text"])
            else:
                label = Tex(label_text, color=COLORS["text"])

            group = VGroup(label, matrix).arrange(DOWN)
            return group

        return matrix

    @staticmethod
    def get_matrix_center(group):
        """Get the center of the matrix part of a group"""
        # If it's just a matrix
        if isinstance(group, Matrix):
            return group.get_center()
        # If it's a VGroup with a label and matrix
        elif isinstance(group, VGroup) and len(group) > 1:
            return group[1].get_center()
        # Default
        return group.get_center()

    @staticmethod
    def format_tensor(tensor, decimals=2):
        """Format tensor values to specific decimal places"""
        if isinstance(tensor, torch.Tensor):
            rounded = torch.round(tensor * 10**decimals) / 10**decimals
            return rounded
        return tensor


class AttentionScoresVisualization(Slide):
    def setup(self):
        """Initialize data and helper objects"""
        self.helper = MatrixHelper()

        # Initialize matrices
        self.query = torch.arange(8).view(4, 2)
        self.key = torch.arange(8).view(4, 2)
        self.key_t = self.key.T
        self.attention_scores = torch.matmul(self.query, self.key_t)
        self.softmax_scores = torch.softmax(self.attention_scores.to(torch.float32), dim=1)
        self.value_data = torch.arange(8).reshape(4, 2) + 1
        self.value_data = self.value_data.to(torch.float32)
        self.output = self.softmax_scores @ self.value_data
        self.output_rounded = self.helper.format_tensor(self.output)

        # Create the formula that will be displayed at the top
        self.formula = MathTex(
            r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            color=COLORS["text"]
        )

    def advance_slide(self):
        """Helper method to advance slides while keeping compatibility with non-Slide classes"""
        if isinstance(self, Slide):
            self.next_slide()
        else:
            self.wait(0.25)

    def create_highlight_rectangle(self, mobject, color, opacity=OPACITY["highlight"]):
        """Create a highlight rectangle for a given mobject"""
        return Rectangle(
            width=mobject.get_width(),
            height=mobject.get_height(),
            fill_color=color,
            fill_opacity=opacity,
            stroke_width=0
        ).move_to(mobject)

    def highlight_dot_product(self, row_index, col_index, query_group, key_t_group, attention_group):
        """Highlight the dot product calculation for specific row and column"""
        # Get the specific row and column
        query_row = query_group[1].get_rows()[row_index]
        key_col = key_t_group[1].get_columns()[col_index]

        # Create highlight rectangles
        row_highlight = self.create_highlight_rectangle(
            query_row, COLORS["highlight"]["query"]
        )

        col_highlight = self.create_highlight_rectangle(
            key_col, COLORS["highlight"]["key"]
        )

        # Get the target cell in the attention matrix
        target_cell = attention_group[1].get_entries()[row_index * 4 + col_index]

        cell_highlight = self.create_highlight_rectangle(
            target_cell, COLORS["highlight"]["result"]
        )

        # Animation sequence
        self.play(FadeIn(row_highlight), FadeIn(col_highlight))
        self.play(
            row_highlight.animate.move_to(cell_highlight),
            col_highlight.animate.move_to(cell_highlight),
            run_time=0.7,
        )
        self.play(
            FadeTransform(row_highlight, cell_highlight),
            FadeTransform(col_highlight, cell_highlight),
        )
        self.play(FadeOut(cell_highlight))

    def setup_formula_banner(self):
        """Setup the formula banner at the bottom of the screen"""
        formula_banner = Rectangle(
            width=config.frame_width,
            height=1.25,
            fill_color=LIGHT_GREY,
            fill_opacity=0.3,
            stroke_width=1,
            stroke_color=GREY
        ).to_edge(DOWN, buff=.2)

        self.formula.move_to(formula_banner.get_center())

        return VGroup(formula_banner, self.formula)

    def intro_matrices(self):
        """Introduce the Query and Key matrices"""
        query_group = self.helper.create_matrix(self.query, "Query")
        key_group = self.helper.create_matrix(self.key, "Key")

        initial_group = VGroup(query_group, key_group).arrange(RIGHT, buff=1).center()

        self.play(Write(initial_group))
        self.advance_slide()

        return query_group, key_group

    def transform_key_to_key_t(self, query_group, key_group):
        """Transform the Key matrix to its transpose"""

        key_t_group = self.helper.create_matrix(self.key_t, r"\text{Key}^T")

        # Calculate target positions that are a bit lower to account for the formula banner
        left_target = ORIGIN + LEFT * 3 + DOWN * 0.5
        right_target = ORIGIN + RIGHT * 3 + DOWN * 0.5

        self.play(
            query_group.animate.move_to(left_target),
            ReplacementTransform(key_group, key_t_group.move_to(right_target)),
        )
        self.advance_slide()

        return key_t_group

    def calculate_attention_scores(self, query_group, key_t_group):
        """Visualize the calculation of attention scores"""
        attention_group = self.helper.create_matrix(self.attention_scores, "Attention Scores")

        times = Tex("×", color=COLORS["text"]).scale(1.5)
        equals = Tex("=", color=COLORS["text"]).scale(1.5)

        # Create copies for the equation
        query_group_copy = query_group.copy()
        key_t_group_copy = key_t_group.copy()

        # Arrange the equation
        equation = VGroup(
            query_group_copy, times, key_t_group_copy, equals, attention_group
        ).arrange(RIGHT, buff=0.5, aligned_edge=ORIGIN)

        # Position operators
        times.move_to(
            midpoint(self.helper.get_matrix_center(equation[0]),
                     self.helper.get_matrix_center(equation[2])) - 0.5 * RIGHT
        )
        equals.move_to(
            midpoint(self.helper.get_matrix_center(equation[2]),
                     self.helper.get_matrix_center(equation[4]))
        )

        # Scale to fit screen
        scale_factor = min(1, (config.frame_width - 1) / equation.get_width())
        equation.scale(scale_factor).center()

        # Animate transition
        self.play(
            Transform(query_group, equation[0]),
            Transform(key_t_group, equation[2]),
            run_time=1.0,
        )
        self.play(FadeIn(times), FadeIn(equals), FadeIn(attention_group), run_time=1)
        self.remove(query_group_copy, key_t_group_copy)
        self.advance_slide()

        return times, equals, attention_group

    def focus_on_attention_scores(self, query_group, key_t_group, times, equals, attention_group):
        """Focus on the attention scores matrix"""
        attention_title = attention_group[0]

        # Don't move title to top, keep formula banner visible
        self.play(
            FadeOut(query_group),
            FadeOut(key_t_group),
            FadeOut(times),
            FadeOut(equals),
            attention_title.animate.to_edge(UP, buff=0.5).set_x(0),
            attention_group[1].animate.scale(1.2).move_to(ORIGIN).shift(DOWN * 0.5)
        )
        self.advance_slide()

        return attention_title

    def explain_softmax(self, attention_title):
        """Explain the softmax operation"""

        softmax_title = Tex("Apply Softmax", color=COLORS["text"]).scale(1.2)
        softmax_title.move_to(attention_title.get_center())

        softmax_formula = MathTex(
            r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}",
            color=COLORS["text"]
        ).next_to(softmax_title, DOWN)

        self.play(
            Transform(attention_title, softmax_title),
            Write(softmax_formula)
        )
        self.advance_slide()

        return softmax_formula

    def apply_softmax(self, attention_group, softmax_formula):
        """Visualize the application of softmax to get attention weights"""
        softmax_group = self.helper.create_matrix(
            self.softmax_scores, "Attention Weights"
        )
        softmax_group.scale(0.75)

        # Position the softmax matrix at the same place as the attention matrix
        softmax_group.move_to(attention_group.get_center() + DOWN * 0.3)

        # Create references to keep track of the actual objects in the scene
        matrix_on_screen = attention_group[1]
        title_on_screen = softmax_formula

        # Show transformation animation from attention scores to softmax weights
        self.play(
            FadeOut(attention_group[0]),  # Fade out old title
            Transform(matrix_on_screen, softmax_group[1]),  # Transform matrix
            Transform(title_on_screen, softmax_group[0]),  # Transform formula
            run_time=1
        )
        self.advance_slide()

        # Create a group with the actual objects that are visible on screen
        visible_group = VGroup(title_on_screen, matrix_on_screen)

        return visible_group  # Return the group of visible objects

    def introduce_value_and_calculate_output(self, softmax_group):
        """Introduce the Value matrix and calculate the output"""
        # Introduce value matrix
        value_group = self.helper.create_matrix(
            self.value_data, "Value Matrix (V)"
        )
        value_group.scale(0.75)

        # Create a copy to animate
        softmax_copy = softmax_group.copy()
        softmax_copy.to_edge(LEFT)
        self.play(Transform(softmax_group, softmax_copy))
        times_symbol = Tex("×", color=COLORS["text"]).scale(1.5).next_to(softmax_copy, RIGHT)

        # Then, position the value matrix next to it and fade it in))
        value_group.next_to(times_symbol, RIGHT)
        self.play(FadeIn(value_group))

        # Add multiplication symbol
        self.play(FadeIn(times_symbol))

        # Prepare the output matrix
        output_group = self.helper.create_matrix(
            self.output_rounded, "Attention Output"
        )
        output_group.scale(0.75)

        # Add equals symbol and show output
        equals_symbol = Tex("=", color=COLORS["text"]).scale(1.5).next_to(value_group, RIGHT)
        output_group.next_to(equals_symbol, RIGHT)

        self.play(FadeIn(equals_symbol))
        self.play(FadeIn(output_group))
        self.advance_slide()

        return value_group, times_symbol, equals_symbol, output_group

    def show_conclusion(self, softmax_group, value_group, times_symbol, equals_symbol, output_group, banner_group):
        """Show the conclusion with the attention mechanism formula"""
        # Remove highlighting from the formula
        if hasattr(self, "current_highlight") and self.current_highlight:
            self.play(FadeOut(self.current_highlight))
            self.current_highlight = None

        # Create the final text
        final_text = Text("And that's all you need!", color=COLORS["text"], font_size=36)

        # Fade out everything except output_group
        self.play(
            FadeOut(softmax_group),
            FadeOut(value_group),
            FadeOut(times_symbol),
            FadeOut(equals_symbol),
            FadeOut(banner_group),
            FadeOut(self.formula)
        )

        # Center the output matrix
        self.play(output_group.animate.move_to(ORIGIN))

        # Move the output group to a good position for transformation
        self.play(output_group.animate.scale(0.8).to_edge(UP))

        # Create final text
        final_text = Text("And that's all you need!", color=COLORS["text"], font_size=36).center()

        # Transform output_group to final_text (visually replaces but keeps the reference)
        self.play(Transform(output_group, final_text))
        self.advance_slide()  # Use advance_slide() for consistency

        # For the plot twist, create a new text
        plot_twist = Text("or is it...", color=COLORS["text"], font_size=36).center()

        # Since output_group now looks like final_text, transform it to plot_twist
        self.play(Transform(output_group, plot_twist))
        self.advance_slide()



    def construct(self):
        """Main construct method that orchestrates the visualization"""
        self.setup()

        # Place Banner at the bottom
        banner_group = self.setup_formula_banner()
        self.play(FadeIn(banner_group))

        # Step 1: Introduce Query and Key matrices
        query_group, key_group = self.intro_matrices()

        # Step 2: Transform Key to Key^T
        key_t_group = self.transform_key_to_key_t(query_group, key_group)

        # Step 3: Calculate attention scores
        times, equals, attention_group = self.calculate_attention_scores(query_group, key_t_group)

        # Step 4: Show dot product examples
        for row, col in [(0, 0), (2, 1)]:
            self.highlight_dot_product(
                row, col, query_group, key_t_group, attention_group
            )
        self.advance_slide()

        # Step 5: Focus on attention scores
        attention_title = self.focus_on_attention_scores(query_group, key_t_group, times, equals, attention_group)

        # Step 6: Explain softmax
        softmax_formula = self.explain_softmax(attention_title)

        # Step 7: Apply softmax to get weights
        softmax_group = self.apply_softmax(attention_group, softmax_formula)

        # Step 8: Introduce value matrix
        value_group, times_symbol, equals_symbol, output_group = self.introduce_value_and_calculate_output(softmax_group)

        # Step 10: Show conclusion
        self.show_conclusion(softmax_group, value_group, times_symbol, equals_symbol, output_group, banner_group)
