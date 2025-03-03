from manim import *
from enum import Enum
from manim_slides import Slide

# Custom configuration to use a light theme for presentations
config.background_color = WHITE

# Constants for consistent styling
COLORS = {
    "text": BLACK,
    "matrix": DARK_GRAY,
    "bracket": DARK_GRAY,
    "highlight": {"query": GOLD_D, "key": BLUE_D, "result": GREEN_D, "masked": RED_D},
}

OPACITY = {"highlight": 0.2}


class MaskType(Enum):
    CAUSAL = "causal_mask_mod"


def causal_attention(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


mask_mods = {MaskType.CAUSAL.value: causal_attention}


class BlockMaskKVCreation(Slide):
    def advance_slide(self):
        """Helper method to advance slides while keeping compatibility with non-Slide classes"""
        if isinstance(self, Slide):
            self.next_slide()
        else:
            self.wait(0.25)

    def construct(self):
        # Initial setup
        seq_len = 8
        block_size = 2
        num_block_rows = seq_len // block_size
        square_size = 0.5
        mask_mod = MaskType.CAUSAL
        mask_mod_func = mask_mods[mask_mod.value]

        attention_matrix, squares = self.setup_attention_matrix(seq_len, seq_len)
        attention_matrix.move_to(ORIGIN + LEFT * 3)

        # Create main title
        title = Text(
            "BlockMask Construction", font_size=34, color=COLORS["text"]
        ).to_edge(UP)

        # Create attention scores title
        attention_title = Text("Attention Scores", font_size=28, color=COLORS["text"])

        # Create kv_num_blocks matrix
        num_blocks_title = Text("kv_num_blocks", font_size=28, color=COLORS["text"])
        num_blocks_matrix = VGroup()
        for i in range(num_block_rows):
            square = Square(
                side_length=square_size,
                stroke_color=COLORS["matrix"],
                stroke_width=1,
                fill_opacity=0.1,
            )
            text = Text("?", font_size=20, color=COLORS["text"])
            text.move_to(square.get_center())
            num_blocks_matrix.add(VGroup(square, text))
        num_blocks_matrix.arrange(DOWN, buff=0)

        # Create kv_indices matrix
        kv_indices_title = Text("kv_indices", font_size=28, color=COLORS["text"])
        kv_indices_matrix = VGroup()
        for i in range(num_block_rows):
            row = VGroup()
            for j in range(num_block_rows):
                square = Square(
                    side_length=square_size,
                    stroke_color=COLORS["matrix"],
                    stroke_width=1,
                    fill_opacity=0.1,
                )
                text = Text("?", font_size=20, color=COLORS["text"])
                text.move_to(square.get_center())
                row.add(VGroup(square, text))
            row.arrange(RIGHT, buff=0)
            kv_indices_matrix.add(row)
        kv_indices_matrix.arrange(DOWN, buff=0)

        # Position matrices
        attention_title.next_to(squares, UP, buff=0.8)
        attention_title.set_x(squares.get_center()[0])

        num_blocks_title.next_to(attention_matrix, RIGHT, buff=1)
        num_blocks_matrix.next_to(num_blocks_title, DOWN, buff=0.5)
        num_blocks_matrix.align_to(squares, UP)
        num_blocks_title.next_to(num_blocks_matrix, UP, buff=0.3)
        num_blocks_title.align_to(attention_title, UP)

        kv_indices_title.next_to(num_blocks_title, RIGHT, buff=1)
        kv_indices_matrix.next_to(kv_indices_title, DOWN, buff=0.5)
        kv_indices_matrix.align_to(squares, UP)
        kv_indices_title.next_to(kv_indices_matrix, UP, buff=0.3)
        kv_indices_title.align_to(attention_title, UP)

        # Initial animation
        self.play(Write(title))
        self.play(Create(attention_matrix))
        self.play(
            Write(attention_title), Write(num_blocks_title), Write(kv_indices_title)
        )
        self.play(Create(num_blocks_matrix), Create(kv_indices_matrix))
        self.advance_slide()

        # Process each query block row
        for block_row in range(num_block_rows):
            q_start = block_row * block_size
            q_end = q_start + block_size
            num_computed_blocks = 0

            # Create horizontal highlight (row)
            attention_highlight_row = (
                Rectangle(
                    width=seq_len * square_size,
                    height=block_size * square_size,
                    stroke_color=COLORS["highlight"]["query"],
                    fill_color=COLORS["highlight"]["query"],
                    fill_opacity=OPACITY["highlight"],
                )
                .move_to(
                    squares[
                        q_start * seq_len : (q_start + block_size) * seq_len
                    ].get_center()
                )
                .align_to(squares[q_start * seq_len], UP + LEFT)
            )

            num_blocks_highlight = Rectangle(
                width=square_size,
                height=square_size,
                stroke_color=COLORS["highlight"]["query"],
                fill_color=COLORS["highlight"]["query"],
                fill_opacity=OPACITY["highlight"],
            ).move_to(num_blocks_matrix[block_row].get_center())

            kv_indices_highlight = Rectangle(
                width=num_block_rows * square_size,
                height=square_size,
                stroke_color=COLORS["highlight"]["query"],
                fill_color=COLORS["highlight"]["query"],
                fill_opacity=OPACITY["highlight"],
            ).move_to(kv_indices_matrix[block_row].get_center())

            # Initial highlights for row
            self.play(
                Create(attention_highlight_row),
                Create(num_blocks_highlight),
                Create(kv_indices_highlight),
            )

            # Process each key block in the row
            for block_col in range(num_block_rows):
                # Create vertical highlight (column) for current block
                attention_highlight_col = (
                    Rectangle(
                        width=block_size * square_size,
                        height=block_size * square_size,
                        stroke_color=COLORS["highlight"]["key"],
                        fill_color=COLORS["highlight"]["key"],
                        fill_opacity=OPACITY["highlight"],
                    )
                    .move_to(
                        VGroup(
                            *[
                                squares[i * seq_len + block_col * block_size]
                                for i in range(q_start, q_end)
                            ]
                        ).get_center()
                    )
                    .align_to(
                        squares[q_start * seq_len + block_col * block_size], UP + LEFT
                    )
                )

                # Show column highlight
                self.play(Create(attention_highlight_col), run_time=0.1)

                k_start = block_col * block_size
                k_end = k_start + block_size

                # Prepare animations for the entire 2x2 block
                block_animations = []
                block_needed = False

                # Check causal attention for each position in the block
                for q_idx in range(q_start, q_end):
                    for k_idx in range(k_start, k_end):
                        can_attend = mask_mod_func(0, 0, q_idx, k_idx)
                        pos_idx = q_idx * seq_len + k_idx

                        new_square = squares[pos_idx].copy()
                        if can_attend:
                            block_needed = True
                            new_square.set_fill(
                                COLORS["highlight"]["result"], opacity=0.7
                            )

                            # Flash effect for visible cells
                            block_animations.append(
                                Flash(
                                    squares[pos_idx],
                                    color=COLORS["highlight"]["result"],
                                    flash_radius=0.3,
                                    line_stroke_width=3,
                                )
                            )
                        else:
                            new_square.set_fill(
                                COLORS["highlight"]["masked"], opacity=0.1
                            )

                        block_animations.append(Transform(squares[pos_idx], new_square))

                # Play all animations for the 2x2 block simultaneously
                self.play(*block_animations, run_time=0.1)

                if block_needed:
                    num_computed_blocks += 1

                    # Update matrices simultaneously
                    updates = []

                    # Update num_blocks
                    new_num_text = Text(
                        str(num_computed_blocks), font_size=20, color=COLORS["text"]
                    )
                    new_num_text.move_to(num_blocks_matrix[block_row][1].get_center())
                    updates.append(
                        Transform(num_blocks_matrix[block_row][1], new_num_text)
                    )

                    # Update kv_indices
                    new_idx_text = Text(
                        str(block_col),
                        font_size=20,
                        color=COLORS["highlight"]["result"],
                    )
                    new_idx_text.move_to(
                        kv_indices_matrix[block_row][block_col][1].get_center()
                    )
                    updates.append(
                        Transform(
                            kv_indices_matrix[block_row][block_col][1], new_idx_text
                        )
                    )

                    self.play(*updates, run_time=0.15)
                else:
                    # Update kv_indices with dash
                    new_text = Text(
                        "-", font_size=20, color=COLORS["highlight"]["masked"]
                    )
                    new_text.move_to(
                        kv_indices_matrix[block_row][block_col][1].get_center()
                    )
                    self.play(
                        Transform(kv_indices_matrix[block_row][block_col][1], new_text),
                        run_time=0.15,
                    )

                # Add result text
                if block_needed:
                    result_text = Text(
                        "Keep Block", font_size=20, color=COLORS["highlight"]["result"]
                    )
                else:
                    result_text = Text(
                        "Skip Block", font_size=20, color=COLORS["highlight"]["masked"]
                    )

                result_text.to_edge(DOWN, buff=1.0)
                self.play(FadeIn(result_text), run_time=0.15)
                self.wait(0.1)
                self.play(FadeOut(result_text), run_time=0.05)

                # Remove column highlight after processing block
                self.play(FadeOut(attention_highlight_col), run_time=0.05)

            self.advance_slide()

            # Remove remaining highlights
            self.play(
                FadeOut(attention_highlight_row),
                FadeOut(num_blocks_highlight),
                FadeOut(kv_indices_highlight),
            )

    def setup_attention_matrix(self, seq_len_q, seq_len_kv):
        squares = VGroup()
        for i in range(seq_len_q):
            for j in range(seq_len_kv):
                square = Square(
                    side_length=0.5,
                    stroke_color=COLORS["matrix"],
                    stroke_width=1,
                    fill_opacity=0.1,
                )
                square.move_to([j * 0.5, -i * 0.5, 0])
                squares.add(square)

        # Add index labels (for query and key)
        q_indices = VGroup(
            *[
                Text(str(i), font_size=20, color=COLORS["highlight"]["query"]).next_to(
                    squares[i * seq_len_q], LEFT, buff=0.3
                )
                for i in range(seq_len_q)
            ]
        )
        k_indices = VGroup(
            *[
                Text(str(i), font_size=20, color=COLORS["highlight"]["key"]).next_to(
                    squares[i], UP, buff=0.3
                )
                for i in range(seq_len_kv)
            ]
        )

        attention_matrix = VGroup(squares, q_indices, k_indices)
        return attention_matrix, squares
