"""To run call manimgl vizz/flex/score_mod_attention.py ScoreModAttentionVisualization"""

import torch
from manim import *
from manim_slides import Slide

from enum import Enum

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
        "modified": PURPLE_D,
    },
}

OPACITY = {"highlight": 0.2}

standard_format = "score = {score}  # No modification"

# Standard (No Modification)
standard_format = """
def score_mod(score={score}, b={b}, h={h}, q_idx={q_idx}, kv_idx={kv_idx}):
    return {score}  # No modification
"""

# Position Scaled
position_scaled_format = """
def score_mod(score={score}, b={b}, h={h}, q_idx={q_idx}, kv_idx={kv_idx}):
    distance = abs({q_idx} - {kv_idx})
    scale_factor = 1.0 / (1.0 + distance * 0.5)
    return {score} * scale_factor
"""

# Softcap
softcap_format = """
softcap = 20
def score_mod(score={score}, b={b}, h={h}, q_idx={q_idx}, kv_idx={kv_idx}):
    score = {score} / softcap
    score = torch.tanh(score) * softcap
    return score
"""


class Scores(Enum):
    standard = "standard"
    position_scaled = "position_scaled"
    softcap = "softcap"


score_to_string = {
    Scores.standard: standard_format,
    Scores.position_scaled: position_scaled_format,
    Scores.softcap: softcap_format,
}

scores_to_func = {
    Scores.standard: lambda score, b, h, q_idx, kv_idx: score,
    Scores.position_scaled: lambda score, b, h, q_idx, kv_idx: score
    * (1.0 / (1.0 + abs(q_idx - kv_idx) * 0.5)),
    Scores.softcap: lambda score, b, h, q_idx, kv_idx: torch.tanh(score / 20) * 20,
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
            bracket_config={"color": COLORS["bracket"]}
            if with_brackets
            else {"opacity": 0},
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


class ScoreModAttentionVisualization(Slide):
    def setup(self):
        """Initialize data and helper objects"""
        self.helper = MatrixHelper()

        # Initialize matrices
        self.query = torch.arange(8).view(4, 2)
        self.key = torch.arange(8).view(4, 2)
        self.key_t = self.key.T
        self.attention_scores = torch.matmul(self.query, self.key_t)

        # Create various modified scores for different score_mod functions
        # 1. Standard attention (no modification)
        self.standard_scores = self.attention_scores.clone()

        # softcap attention
        self.softcap_scores = self.attention_scores.clone()
        self.softcap_scores = self.softcap_scores / 20
        self.softcap_scores = torch.tanh(self.softcap_scores) * 20

        # Store the current score modification for animations
        self.current_scores = self.attention_scores.clone()

        # Value data for output computation
        self.value_data = torch.arange(8).reshape(4, 2) + 1
        self.value_data = self.value_data.to(torch.float32)

        # Current score_mod function
        self.current_score_mod = "standard"

    def advance_slide(self):
        """Helper method to advance slides while keeping compatibility with non-Slide classes"""
        if isinstance(self, Slide):
            self.next_slide()
        else:
            self.wait(0.25)

    def apply_score_mod(self, attention_group, mod_type: Scores):
        """Apply score modification to the attention scores"""
        # Update title based on modification type
        title_text = f"{str(mod_type.value).replace('_', ' ').title()} Attention"
        score_mod_title = Text(title_text, font_size=34, color=COLORS["text"]).to_edge(
            UP
        )

        # Transform the title
        self.play(
            Transform(attention_group[0], score_mod_title),
        )
        self.advance_slide()

        # Add explanation of score modification
        explanations = {
            Scores.softcap: "Cap logits so that they don't get too large",
            Scores.standard: "Standard attention uses the raw attention scores without modification.",
            Scores.position_scaled: "Position-scaled attention reduces scores based on token distance, favoring nearby tokens.",
        }

        mod_explanation = Text(
            explanations[mod_type], font_size=20, color=COLORS["text"]
        ).next_to(score_mod_title, DOWN, buff=0.05)

        self.play(Write(mod_explanation))
        self.advance_slide()

        # Create the code string template
        code_string = score_to_string[mod_type].format(
            score="score", b="b", h="h", q_idx="q_idx", kv_idx="kv_idx"
        )

        # Position the code block below the matrix (reduced size to fit)
        score_mod_text = Code(
            code_string=code_string,
            language="python",
            add_line_numbers=False,
            paragraph_config={"font_size": 16},
        ).next_to(attention_group[1], DOWN)

        # Display initial code
        self.play(Write(score_mod_text))
        self.advance_slide()

        # Get the appropriate modified scores based on the mod_type
        if mod_type == Scores.standard:
            modified_scores = self.standard_scores
        elif mod_type == Scores.position_scaled:
            modified_scores = self.position_scaled_scores
        elif mod_type == Scores.softcap:
            modified_scores = self.softcap_scores

        # Apply score_mod function to each position
        for i in range(4):
            for j in range(4):
                b, h, q_idx, kv_idx = 0, 0, i, j

                original_score = self.attention_scores[i, j].item()
                # Create a new code block with the specific values
                new_code_string = score_to_string[mod_type].format(
                    score=original_score, b=b, h=h, q_idx=q_idx, kv_idx=kv_idx
                )
                new_call_text = Code(
                    code_string=new_code_string,
                    language="python",
                    add_line_numbers=False,
                    paragraph_config={"font_size": 16},
                ).move_to(score_mod_text.get_center())

                # Compute the modified score for this cell
                modified_score = modified_scores[i, j].item()

                # Highlight the current cell
                current_cell = attention_group[1].get_entries()[i * 4 + j]
                cell_rect = SurroundingRectangle(
                    current_cell, color=COLORS["highlight"]["query"], buff=0.05
                )

                # Show highlighting, update the code, and display the result
                self.play(
                    Transform(score_mod_text, new_call_text),
                    Create(cell_rect),
                    run_time=0.3,
                )

                # Replace the value in the matrix with the modified score
                new_value = MathTex(f"{modified_score:.2f}", color=COLORS["matrix"])
                new_value.move_to(current_cell.get_center())

                self.play(
                    Transform(current_cell, new_value),
                    Flash(
                        current_cell,
                        color=COLORS["highlight"]["modified"],
                        flash_radius=0.3,
                    ),
                    run_time=0.3,
                )

                # Clean up for the next iteration
                self.play(FadeOut(cell_rect), run_time=0.2)

        self.advance_slide()

        # Return the updated elements
        return score_mod_text, mod_explanation

    def construct(self):
        """Main construct method that orchestrates the visualization"""
        self.setup()

        # Start with original attention scores
        attention_group = self.helper.create_matrix(
            self.attention_scores, "Standard Attention Scores"
        )
        attention_group.scale(1.2).to_edge(UP, buff=0.5)
        self.play(FadeIn(attention_group))
        self.advance_slide()

        # Apply different score_mod functions
        for mod_type in [Scores.softcap]:
            score_mod_text, mod_explanation = self.apply_score_mod(
                attention_group, mod_type
            )

            # self.play(FadeOut(score_mod_text), FadeOut(mod_explanation))
            self.advance_slide()

        # # Final conclusion
        # conclusion_text = Text(
        #     "FlexAttention provides a simple yet powerful API for customizing attention mechanisms",
        #     font_size=32,
        #     color=COLORS["text"]
        # ).to_edge(UP)

        # examples_text = Text(
        #     "Applications: position weighting, pattern injection, specialized attention heads, etc.",
        #     font_size=24,
        #     color=COLORS["text"]
        # ).next_to(conclusion_text, DOWN, buff=0.5)

        # self.play(
        #     FadeOut(score_mod_text),
        #     FadeOut(mod_explanation),
        #     FadeOut(attention_group),
        # )
        # self.advance_slide()
