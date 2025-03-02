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
        "keep": GREEN_D,
        "mask": RED_D,
    },
}

OPACITY = {"highlight": 0.2}


class MaskType(Enum):
    prefix_lm = "PrefixLM"
    sliding_window = "SlidingWindow"


def prefix_lm(b, h, q_idx, kv_idx):
    prefix_length = 2
    return q_idx >= kv_idx or kv_idx < prefix_length


prefix_code_string = """
def mask_mod(b={b}, h={h}, q_idx={q_idx}, kv_idx={kv_idx}):
    prefix_length = 2
    return {q_idx} >= {kv_idx} or {kv_idx} < prefix_length
"""


def sliding_window(b, h, q_idx, kv_idx):
    window_size = 2
    return abs(q_idx - kv_idx) < window_size


sliding_window_code_string = """
def mask_mod(b={b}, h={h}, q_idx={q_idx}, kv_idx={kv_idx}):
    window_size = 2
    return abs({q_idx} - {kv_idx}) < window_size
"""

mask_to_func = {
    MaskType.prefix_lm: prefix_lm,
    MaskType.sliding_window: sliding_window,
}

mask_to_string = {
    MaskType.prefix_lm: prefix_code_string,
    MaskType.sliding_window: sliding_window_code_string,
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


class MaskAnimationScene(Slide):
    def setup(self):
        """Initialize data and helper objects"""
        self.helper = MatrixHelper()

        # Initialize attention scores matrix
        self.attention_scores = torch.matmul(
            torch.arange(8).view(4, 2), torch.arange(8).view(4, 2).T
        )

    def advance_slide(self):
        """Helper method to advance slides while keeping compatibility with non-Slide classes"""
        if isinstance(self, Slide):
            self.next_slide()
        else:
            self.wait(0.25)

    def construct(self):
        """Main construct method that orchestrates the visualization"""
        self.setup()

        # Create title and initial matrix
        title_text = "Attention Mask Visualization"
        title = Text(title_text, font_size=34, color=COLORS["text"]).to_edge(UP)

        # Create and position the attention matrix
        attention_group = self.helper.create_matrix(
            self.attention_scores, "Attention Scores"
        )
        attention_group.scale(1.2).to_edge(UP, buff=1.4)

        # Display initial setup
        self.play(Write(title), FadeIn(attention_group[1]))
        self.advance_slide()

        # Apply different mask types
        for mask_type in [MaskType.prefix_lm, MaskType.sliding_window]:
            self.apply_mask(attention_group, mask_type, title)
            self.advance_slide()

    def apply_mask(self, attention_group, mask_type: MaskType, title_obj):
        """Apply masking to the attention scores"""
        # Update title based on mask type
        title_text = f"{str(mask_type.value)} Mask"
        mask_title = Text(title_text, font_size=34, color=COLORS["text"]).to_edge(UP)

        # Transform the title
        self.play(Transform(title_obj, mask_title))
        self.advance_slide()

        # Add explanation of mask
        explanations = {
            MaskType.prefix_lm: "Allows each query attend to all preivous tokens and the first 2 global tokens (prefix)",
            MaskType.sliding_window: "Allows each query to attend only to tokens within a window of 2 positions",
        }

        mask_explanation = Text(
            explanations[mask_type], font_size=20, color=COLORS["text"]
        ).next_to(mask_title, DOWN, buff=0.5)

        self.play(Write(mask_explanation))
        self.advance_slide()

        # Create the code string template
        code_string = mask_to_string[mask_type].format(
            b="b", h="h", q_idx="q_idx", kv_idx="kv_idx"
        )

        # Position the code block below the matrix
        mask_code = Code(
            code_string=code_string,
            language="python",
            add_line_numbers=False,
            paragraph_config={"font_size": 16},
        ).next_to(attention_group, DOWN)

        # Display initial code
        self.play(Write(mask_code))
        self.advance_slide()

        # Store original attention values
        original_entries = {}
        for i in range(4):
            for j in range(4):
                cell_idx = i * 4 + j
                original_entries[cell_idx] = (
                    attention_group[1].get_entries()[cell_idx].copy()
                )

        # Apply mask to each position
        for i in range(4):
            for j in range(4):
                b, h, q_idx, kv_idx = 0, 0, i, j
                cell_idx = i * 4 + j

                # Create a new code block with the specific values
                new_code_string = mask_to_string[mask_type].format(
                    b=b, h=h, q_idx=q_idx, kv_idx=kv_idx
                )
                new_code = Code(
                    code_string=new_code_string,
                    language="python",
                    add_line_numbers=False,
                    paragraph_config={"font_size": 16},
                ).move_to(mask_code.get_center())

                # Check mask result
                mask_result = mask_to_func[mask_type](b, h, q_idx, kv_idx)

                # Highlight the current cell
                current_cell = attention_group[1].get_entries()[cell_idx]
                cell_rect = SurroundingRectangle(
                    current_cell,
                    color=COLORS["highlight"]["keep"]
                    if mask_result
                    else COLORS["highlight"]["mask"],
                    buff=0.05,
                )

                # Prepare output text
                output_text = Text(
                    "Keep" if mask_result else "Mask",
                    font_size=20,
                    color=COLORS["highlight"]["keep"]
                    if mask_result
                    else COLORS["highlight"]["mask"],
                ).next_to(new_code, UP, buff=0.2)

                # Show highlighting, update the code, and display the result
                self.play(
                    Transform(mask_code, new_code),
                    Create(cell_rect),
                    run_time=0.3,
                )

                # Replace or keep the value based on mask result
                if mask_result:
                    # Flash in green to indicate kept
                    self.play(
                        Flash(
                            current_cell,
                            color=COLORS["highlight"]["keep"],
                            flash_radius=0.3,
                        ),
                        FadeIn(output_text),
                        run_time=0.3,
                    )
                else:
                    # Replace with -inf and flash in red
                    new_value = MathTex(r"-\infty", color=COLORS["highlight"]["mask"])
                    new_value.move_to(current_cell.get_center())

                    self.play(
                        Transform(current_cell, new_value),
                        Flash(
                            current_cell,
                            color=COLORS["highlight"]["mask"],
                            flash_radius=0.3,
                        ),
                        FadeIn(output_text),
                        run_time=0.3,
                    )

                # Clean up for the next iteration
                self.play(FadeOut(output_text), FadeOut(cell_rect), run_time=0.2)

        self.advance_slide()

        # Clean up - reset matrix to original state
        reset_animations = []
        for i in range(4):
            for j in range(4):
                cell_idx = i * 4 + j
                reset_animations.append(
                    Transform(
                        attention_group[1].get_entries()[cell_idx],
                        original_entries[cell_idx],
                    )
                )

        self.play(*reset_animations, FadeOut(mask_code), FadeOut(mask_explanation))
        self.advance_slide()


# To run this animation, use the command:
# manimgl your_file_name.py MaskAnimationScene
