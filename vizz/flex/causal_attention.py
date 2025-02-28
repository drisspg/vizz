"""To run call manimgl vizz/flex/causal_attention.py CausalAttentionVisualization"""

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
        "masked": RED_D
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


class CausalAttentionVisualization(Slide):
    def setup(self):
        """Initialize data and helper objects"""
        self.helper = MatrixHelper()

        # Initialize matrices
        self.query = torch.arange(8).view(4, 2)
        self.key = torch.arange(8).view(4, 2)
        self.key_t = self.key.T
        self.attention_scores = torch.matmul(self.query, self.key_t)

        # Create masked attention scores (for causal attention)
        self.causal_mask = torch.triu(torch.ones(4, 4), diagonal=1) * -1e9
        self.masked_attention_scores = self.attention_scores.clone().float() + self.causal_mask

        self.softmax_scores = torch.softmax(self.attention_scores.to(torch.float32), dim=1)
        self.causal_softmax_scores = torch.softmax(self.masked_attention_scores, dim=1)

        self.value_data = torch.arange(8).reshape(4, 2) + 1
        self.value_data = self.value_data.to(torch.float32)
        self.output = self.softmax_scores @ self.value_data
        self.causal_output = self.causal_softmax_scores @ self.value_data

        self.output_rounded = self.helper.format_tensor(self.output)
        self.causal_output_rounded = self.helper.format_tensor(self.causal_output)

        # Define causal masking function
        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        self.mask_mod = mask_mod

    def advance_slide(self):
        """Helper method to advance slides while keeping compatibility with non-Slide classes"""
        if isinstance(self, Slide):
            self.next_slide()
        else:
            self.wait(0.25)

    def apply_causal_mask(self, attention_group):
        """Apply causal mask to the attention scores"""
        # Focus on the attention_scores matrix with causal title
        causal_attention_text = Text("Causal Attention", font_size=40, color=COLORS["text"]).to_edge(UP)

        # Since we're starting with the attention matrix, just transform the title
        self.play(
            Transform(attention_group[0], causal_attention_text),
        )
        self.advance_slide()

        # Add explanation of causal mask
        mask_explanation = Text(
            "In causal attention, tokens can only attend to tokens that came before them.",
            font_size=24,
            color=COLORS["text"]
        ).next_to(causal_attention_text, DOWN, buff=0.5)

        self.play(Write(mask_explanation))
        self.advance_slide()

        # Create the masking function code display
        b, h, q_idx, kv_idx = "b", "h", "q_idx", "kv_idx"
        code_string = """
def mask_mod({b}, {h}, {q_idx}, {kv_idx}):
    return {q_idx} >= {kv_idx}
"""
        mask_mod_text = Code(
            code=code_string.format(b=b, h=h, q_idx=q_idx, kv_idx=kv_idx),
            language="python",
            font="Monospace",
            line_spacing=0.8,
        ).to_edge(DOWN, buff=0.5)

        # Display initial code
        self.play(Write(mask_mod_text))
        self.advance_slide()

        # Apply mask_mod function to each position
        for i in range(4):
            for j in range(4):
                b, h, q_idx, kv_idx = 0, 0, i, j
                new_call_code = code_string.format(b=b, h=h, q_idx=q_idx, kv_idx=kv_idx)
                new_call_text = Code(
                    code=new_call_code,
                    language="python",
                    font="Monospace",
                    line_spacing=0.8,
                ).to_edge(DOWN, buff=0.5)

                # Apply masking logic
                if self.mask_mod(0, 0, i, j):
                    color = GREEN
                    cell_text = attention_group[1].get_entries()[i * 4 + j].copy()
                    output_text = Text("Keep", font_size=20, color=GREEN)
                else:
                    color = RED
                    cell_text = Text("-∞", font_size=24, color=RED).move_to(
                        attention_group[1].get_entries()[i * 4 + j]
                    )
                    output_text = Text("Mask (-∞)", font_size=20, color=RED)

                output_text.next_to(mask_mod_text, UP, buff=0.2)

                # Execute animation sequence for this cell
                self.play(
                    Transform(mask_mod_text, new_call_text),
                    Flash(
                        attention_group[1].get_entries()[i * 4 + j],
                        color=color,
                        flash_radius=0.3,
                    ),
                    Transform(attention_group[1].get_entries()[i * 4 + j], cell_text),
                    FadeIn(output_text),
                    run_time=0.3,
                )
                self.wait(0.1)
                self.play(FadeOut(output_text), run_time=0.2)

        self.advance_slide()

        # Return the updated elements
        return mask_mod_text, mask_explanation

    def explain_softmax_with_masked_values(self, attention_group, mask_mod_text, mask_explanation):
        """Explain how softmax works with masked values"""
        softmax_title = Text("Softmax with Masking", font_size=36, color=COLORS["text"]).to_edge(UP)

        softmax_formula = MathTex(
            r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}",
            color=COLORS["text"]
        ).next_to(softmax_title, DOWN)

        softmax_explanation = Text(
            "When -∞ values are included, they become 0 after softmax",
            font_size=24,
            color=COLORS["text"]
        ).next_to(softmax_formula, DOWN, buff=0.5)

        # Transform to softmax slide
        self.play(
            Transform(attention_group[0], softmax_title),
            FadeOut(mask_mod_text),
            FadeOut(mask_explanation),
            Write(softmax_formula),
            Write(softmax_explanation),
        )
        self.advance_slide()

        # Show masked softmax result
        softmax_matrix = self.helper.create_matrix(
            self.causal_softmax_scores, "Causal Attention Weights"
        )

        # Keep the formula and title, replace the matrix
        self.play(
            Transform(attention_group[1], softmax_matrix[1]),
            FadeOut(softmax_explanation),
            Transform(softmax_formula, softmax_matrix[0]),
            run_time=1.5
        )
        self.advance_slide()

        return softmax_formula

    def final_comparison(self, attention_group, softmax_formula):
        """Compare standard vs causal attention results"""
        # Create value matrix
        value_group = self.helper.create_matrix(self.value_data, "Value Matrix (V)").scale(0.75)

        # Create standard output
        standard_output = self.helper.create_matrix(self.output_rounded, "Standard Output").scale(0.75)

        # Create causal output
        causal_output = self.helper.create_matrix(self.causal_output_rounded, "Causal Output").scale(0.75)

        # Create title for comparison
        comparison_title = Text("Standard vs. Causal Attention", font_size=36, color=COLORS["text"]).to_edge(UP, buff=0.5)

        # Arrange for side-by-side comparison
        standard_group = VGroup(
            Text("Standard Attention", font_size=24, color=COLORS["text"]),
            standard_output[1]
        ).arrange(DOWN)

        causal_group = VGroup(
            Text("Causal Attention", font_size=24, color=COLORS["text"]),
            causal_output[1]
        ).arrange(DOWN)

        comparison = VGroup(standard_group, causal_group).arrange(RIGHT, buff=1.5).center()

        # Transition to comparison view
        self.play(
            FadeOut(attention_group),
            FadeOut(softmax_formula),
            FadeIn(comparison_title),
            run_time=1
        )
        self.play(FadeIn(comparison))
        self.advance_slide()

        # Add explanation text
        explanation = Text(
            "Causal attention ensures information flows only from past to future tokens",
            font_size=24,
            color=COLORS["text"]
        ).next_to(comparison, DOWN, buff=0.75)

        self.play(Write(explanation))
        self.advance_slide()

        # Final message
        final_text = Text(
            "Essential for autoregressive generation in language models!",
            font_size=32,
            color=COLORS["text"]
        ).to_edge(DOWN, buff=1)

        self.play(Write(final_text))
        self.advance_slide()

        return comparison_title, comparison, explanation, final_text

    def construct(self):
        """Main construct method that orchestrates the visualization"""
        self.setup()

        # Start directly with attention scores
        attention_group = self.helper.create_matrix(self.attention_scores, "Attention Scores")
        attention_group.scale(1.2).center()
        self.play(FadeIn(attention_group))
        self.advance_slide()

        # Step 4: Apply causal mask
        mask_mod_text, mask_explanation = self.apply_causal_mask(attention_group)

        # Step 5: Explain softmax with masked values
        softmax_formula = self.explain_softmax_with_masked_values(attention_group, mask_mod_text, mask_explanation)

        # Step 6: Compare standard vs causal attention results
        comparison_title, comparison, explanation, final_text = self.final_comparison(attention_group, softmax_formula)