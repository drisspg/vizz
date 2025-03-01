"""To run call manimgl vizz/flex/score_mod_attention.py ScoreModAttentionVisualization"""

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
        "modified": PURPLE_D
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

        # 2. Position-based scaling (e.g., favoring nearby tokens)
        self.position_scaled_scores = self.attention_scores.clone().float()
        for i in range(4):
            for j in range(4):
                # Scale based on distance between tokens
                distance = abs(i - j)
                scale_factor = 1.0 / (1.0 + distance * 0.5)
                self.position_scaled_scores[i, j] *= scale_factor

        # 3. Fixed pattern (e.g., add bias to certain positions)
        self.pattern_scores = self.attention_scores.clone().float()
        bias_pattern = torch.tensor([
            [2.0, 0.0, -1.0, -2.0],
            [0.0, 2.0, 0.0, -1.0],
            [-1.0, 0.0, 2.0, 0.0],
            [-2.0, -1.0, 0.0, 2.0]
        ])
        self.pattern_scores += bias_pattern

        # Store the current score modification for animations
        self.current_scores = self.attention_scores.clone()

        # Value data for output computation
        self.value_data = torch.arange(8).reshape(4, 2) + 1
        self.value_data = self.value_data.to(torch.float32)

        # Define score modification functions
        self.score_mod_functions = {
            "standard": lambda score, b, h, q_idx, kv_idx: score,
            "position_scaled": lambda score, b, h, q_idx, kv_idx: score * (1.0 / (1.0 + abs(q_idx - kv_idx) * 0.5)),
            "pattern": lambda score, b, h, q_idx, kv_idx: score + bias_pattern[q_idx][kv_idx]
        }

        # Current score_mod function
        self.current_score_mod = "standard"

    def advance_slide(self):
        """Helper method to advance slides while keeping compatibility with non-Slide classes"""
        if isinstance(self, Slide):
            self.next_slide()
        else:
            self.wait(0.25)

    def apply_score_mod(self, attention_group, mod_type):
        """Apply score modification to the attention scores"""
        # Update title based on modification type
        title_text = f"{mod_type.replace('_', ' ').title()} Attention"
        score_mod_title = Text(title_text, font_size=40, color=COLORS["text"]).to_edge(UP)

        # Transform the title
        self.play(
            Transform(attention_group[0], score_mod_title),
        )
        self.advance_slide()

        # Add explanation of score modification
        explanations = {
            "standard": "Standard attention uses the raw attention scores without modification.",
            "position_scaled": "Position-scaled attention reduces scores based on token distance, favoring nearby tokens.",
            "pattern": "Pattern-based attention adds a fixed bias pattern to attention scores."
        }

        mod_explanation = Text(
            explanations[mod_type],
            font_size=24,
            color=COLORS["text"]
        ).next_to(score_mod_title, DOWN, buff=0.5)

        self.play(Write(mod_explanation))
        self.advance_slide()

        # Create the score_mod function code display
        if mod_type == "standard":
            code_string = """
def score_mod(score, b, h, q_idx, kv_idx):
    return score  # No modification
"""
        elif mod_type == "position_scaled":
            code_string = """
def score_mod(score, b, h, q_idx, kv_idx):
    distance = abs(q_idx - kv_idx)
    scale_factor = 1.0 / (1.0 + distance * 0.5)
    return score * scale_factor
"""
        else:  # pattern
            code_string = """
def score_mod(score, b, h, q_idx, kv_idx):
    # Add position-specific bias
    bias_pattern = [
        [2.0, 0.0, -1.0, -2.0],
        [0.0, 2.0, 0.0, -1.0],
        [-1.0, 0.0, 2.0, 0.0],
        [-2.0, -1.0, 0.0, 2.0]
    ]
    return score + bias_pattern[q_idx][kv_idx]
"""

        score_mod_text = Code(
            code_string=code_string,
            language="python",
            font="Monospace",
            add_line_numbers=False,
            font_size=24
        ).to_edge(DOWN, buff=0.5)

        # Display initial code
        self.play(Write(score_mod_text))
        self.advance_slide()

        # Get the appropriate modified scores based on the mod_type
        if mod_type == "standard":
            modified_scores = self.standard_scores
        elif mod_type == "position_scaled":
            modified_scores = self.position_scaled_scores
        else:  # pattern
            modified_scores = self.pattern_scores

        # Create a new matrix to show the modified scores
        modified_matrix = self.helper.create_matrix(modified_scores, "Modified Scores")
        modified_matrix.scale(1.2).center()

        # Apply score_mod function to each position
        for i in range(4):
            for j in range(4):
                b, h, q_idx, kv_idx = 0, 0, i, j

                # Compute the specific modification for this cell
                original_score = self.attention_scores[i, j].item()
                modified_score = modified_scores[i, j].item()

                # Highlight the current cell
                cell_rect = SurroundingRectangle(
                    attention_group[1].get_entries()[i * 4 + j],
                    color=COLORS["highlight"]["modified"],
                    buff=0.05
                )

                # Create text for original and modified values
                original_text = Text(f"Original: {original_score}", font_size=20, color=COLORS["text"])
                modified_text = Text(f"Modified: {modified_score:.2f}", font_size=20, color=COLORS["highlight"]["modified"])
                value_texts = VGroup(original_text, modified_text).arrange(DOWN).next_to(score_mod_text, UP, buff=0.2)

                # Show modification for this cell
                self.play(
                    ShowCreation(cell_rect),
                    FadeIn(value_texts),
                    run_time=0.5
                )
                self.wait(0.2)
                self.play(
                    FadeOut(cell_rect),
                    FadeOut(value_texts),
                    run_time=0.3
                )

        # Show the full modified matrix
        self.play(
            Transform(attention_group[1], modified_matrix[1]),
            run_time=1.5
        )
        self.advance_slide()

        # Return the updated elements
        return score_mod_text, mod_explanation

    def construct(self):
        """Main construct method that orchestrates the visualization"""
        self.setup()

        # Title and introduction
        title = Text("FlexAttention: Arbitrary Score Modifications", font_size=48, color=COLORS["text"]).to_edge(UP)
        subtitle = Text("Visualizing how score_mod transforms attention", font_size=32, color=COLORS["text"]).next_to(title, DOWN)

        self.play(Write(title), Write(subtitle))
        self.advance_slide()

        # Show the FlexAttention API
        api_code = """
# The FlexAttention API allows for arbitrary score modifications
def score_mod(score: f32[], b: i32[], h: i32[], q_idx: i32[], kv_idx: i32[]):
    return score  # By default, no modification (standard attention)

# Applied to every attention score:
for b in range(batch_size):
    for h in range(num_heads):
        for q_idx in range(sequence_length):
            for kv_idx in range(sequence_length):
                modified_scores[b, h, q_idx, kv_idx] = score_mod(
                    scores[b, h, q_idx, kv_idx], b, h, q_idx, kv_idx
                )
"""

        code_block = Code(
            code_string=api_code,
            language="python",
            font="Monospace",
            add_line_numbers=False,
            font_size=24
        ).center()

        self.play(FadeOut(subtitle), FadeIn(code_block))
        self.advance_slide()

        # Clear introduction
        self.play(FadeOut(title), FadeOut(code_block))

        # Start with original attention scores
        attention_group = self.helper.create_matrix(self.attention_scores, "Standard Attention Scores")
        attention_group.scale(1.2).center()
        self.play(FadeIn(attention_group))
        self.advance_slide()

        # Apply different score_mod functions
        for mod_type in ["standard", "position_scaled", "pattern"]:
            score_mod_text, mod_explanation = self.apply_score_mod(attention_group, mod_type)

            # Clear explanations before moving to next modification type
            if mod_type != "pattern":  # Don't clear after the last one
                self.play(FadeOut(score_mod_text), FadeOut(mod_explanation))
                self.advance_slide()

        # Final conclusion
        conclusion_text = Text(
            "FlexAttention provides a simple yet powerful API for customizing attention mechanisms",
            font_size=32,
            color=COLORS["text"]
        ).to_edge(UP)

        examples_text = Text(
            "Applications: position weighting, pattern injection, specialized attention heads, etc.",
            font_size=24,
            color=COLORS["text"]
        ).next_to(conclusion_text, DOWN, buff=0.5)

        self.play(
            FadeOut(score_mod_text),
            FadeOut(mod_explanation),
            FadeOut(attention_group),
            FadeIn(conclusion_text),
            FadeIn(examples_text)
        )
        self.advance_slide()