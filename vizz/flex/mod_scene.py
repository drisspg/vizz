
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



def prefix_lm(b, h, q_idx, kv_idx):
    prefix_length = 2
    return q_idx >= kv_idx or kv_idx < prefix_length


prefix_code_string = """
def mask_mod({b}, {h}, {q_idx}, {kv_idx}):
    prefix_length = 2
    return {q_idx} >= {kv_idx} or {kv_idx} < prefix_length
"""


def sliding_window(b, h, q_idx, kv_idx):
    window_size = 2
    return abs(q_idx - kv_idx) < window_size


sliding_window_code_string = """
def mask_mod({b}, {h}, {q_idx}, {kv_idx}):
    window_size = 2
    return abs({q_idx} - {kv_idx}) < window_size
"""

mod_map = {"PrefixLM": prefix_lm, "SlidingWindow": sliding_window}
str_map = {"PrefixLM": prefix_code_string, "SlidingWindow": sliding_window_code_string}


class MaskAnimationScene(Scene):
    def construct(self):
        # Create the initial attention scores matrix
        attention_scores = torch.matmul(
            torch.arange(8).view(4, 2), torch.arange(8).view(4, 2).T
        )
        self.attention_matrix = Matrix(attention_scores.tolist())
        self.attention_matrix.scale(0.8).center()

        # Create the title
        self.title = Text("Attention Mask Visualization", font_size=40).to_edge(UP)

        # Display initial setup
        self.play(Write(self.attention_matrix), Write(self.title))
        self.wait(1)
        # self.apply_mask_mode("PrefixLM")
        self.apply_mask_mode("SlidingWindow")

    def apply_mask_mode(self, mod_str):
        mask_func = mod_map[mod_str]
        mask_mod_text = self.create_mask_mod_text(mod_str)
        self.play(Write(mask_mod_text))
        self.current_mask_text = mask_mod_text
        # Update the title
        new_title = Text(mod_str, font_size=40).to_edge(UP)
        self.play(Transform(self.title, new_title))
        # Apply the mask
        for i in range(4):
            for j in range(4):
                self.apply_mask_to_cell(i, j, mask_func, mod_str)

    def create_mask_mod_text(
        self, mod_str: str, b="b", h="h", q_idx="q_idx", kv_idx="kv_idx"
    ):
        formatted_code = str_map[mod_str].format(b=b, h=h, q_idx=q_idx, kv_idx=kv_idx)
        return Text(
            str(formatted_code),
            # language="python",
            font="Monospace",
            font_size=24,
            # insert_line_no=False,
        ).to_edge(DOWN, buff=0.5)

    def apply_mask_to_cell(self, i, j, mask_func, mod_str):
        new_text = self.create_mask_mod_text(mod_str, 0, 0, i, j)
        if mask_func(0, 0, i, j):
            color = GREEN
            new_value = self.attention_matrix.get_entries()[i * 4 + j].copy()
            output_text = Text("Keep", font_size=20, color=GREEN)
        else:
            color = RED
            new_value = Text("-inf", font_size=24, color=RED).move_to(
                self.attention_matrix.get_entries()[i * 4 + j]
            )
            output_text = Text("Mask", font_size=20, color=RED)
        output_text.next_to(self.current_mask_text, UP, buff=0.2)
        new_value.set_color(color)
        self.play(
            Transform(self.current_mask_text, new_text),
            Flash(
                self.attention_matrix.get_entries()[i * 4 + j],
                color=color,
                flash_radius=0.3,
            ),
            Transform(self.attention_matrix.get_entries()[i * 4 + j], new_value),
            FadeIn(output_text),
            run_time=0.3,
        )
        self.wait(0.1)
        self.play(FadeOut(output_text), run_time=0.2)


# To run this animation, use the command:
# manim -pqh your_file_name.py MaskAnimationScene
