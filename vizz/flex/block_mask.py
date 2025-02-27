# from manimlib import *
from manim import *
from enum import Enum


class MaskType(Enum):
    CAUSAL = "causal_mask_mod"


def causal_attention(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


mask_mods = {MaskType.CAUSAL.value: causal_attention}


class BlockMaskScoreAnimation(Scene):
    def construct(self):
        # Initial setup
        self.seq_len = 6
        self.block_size = 2

        self.mask_mod = MaskType.CAUSAL

        # Create the title
        title_text = self.mask_mod.value.replace("_", " ").title()
        self.title = Text(f"{title_text} Visualization", font_size=40).to_edge(UP)

        # Create the attention matrix
        self.setup_attention_matrix()

        # Create the equation template
        self.setup_equation()

        # Create reusable highlight box and output text.
        # (We create them off-screen first.)
        self.highlight_box = Square(
            side_length=0.5, stroke_color=YELLOW, stroke_width=3, fill_opacity=0
        )
        self.highlight_box.move_to(100 * RIGHT)
        self.add(self.highlight_box)

        self.output_text = Text("", font_size=20)
        self.output_text.move_to(self.equation.get_center() + DOWN * 0.5)
        self.add(self.output_text)

        # Display initial setup
        self.play(Write(self.attention_matrix), Write(self.title), Write(self.equation))
        self.wait(0.5)

        # Apply the mask animations
        self.apply_causal_mask()

    def setup_attention_matrix(self):
        squares = VGroup()
        for i in range(self.seq_len):
            for j in range(self.seq_len):
                square = Square(
                    side_length=0.5,
                    stroke_color=WHITE,
                    stroke_width=1,
                    fill_opacity=0.1,
                )
                square.move_to([j * 0.5, -i * 0.5, 0])
                squares.add(square)

        # Add index labels (for query and key)
        q_indices = VGroup(
            *[
                Text(str(i), font_size=20, color=BLUE).next_to(
                    squares[i * self.seq_len], LEFT, buff=0.3
                )
                for i in range(self.seq_len)
            ]
        )
        k_indices = VGroup(
            *[
                Text(str(i), font_size=20, color=RED).next_to(squares[i], UP, buff=0.3)
                for i in range(self.seq_len)
            ]
        )

        self.attention_matrix = VGroup(squares, q_indices, k_indices)
        self.attention_matrix.move_to(ORIGIN)
        self.squares = squares

    def setup_equation(self):
        self.equation = Text(
            f"{self.mask_mod.value}(q_idx = 0, kv_idx = 0) = {mask_mods[self.mask_mod.value](0, 0, 0, 0)}",
            font_size=24,
        ).next_to(self.title, DOWN)

    def apply_causal_mask(self, play_pop: bool = False):
        mask_mod = mask_mods[self.mask_mod.value]
        for q_idx in range(self.seq_len):
            for k_idx in range(self.seq_len):
                pos_idx = q_idx * self.seq_len + k_idx
                result = mask_mod(0, 0, q_idx, k_idx)
                result_str = "True" if result else "False"

                ### Reuse the highlight_box by moving it to the current square
                self.play(
                    self.highlight_box.animate.move_to(self.squares[pos_idx]),
                    run_time=0.1,
                )

                ### Update the equation text.
                new_eq_content = f"{MaskType.CAUSAL.value}(q_idx = {q_idx}, kv_idx = {k_idx}) = {result_str}"
                new_equation = Text(new_eq_content, font_size=24).move_to(
                    self.equation.get_center()
                )
                self.equation.become(new_equation)

                ### Flash effect and update square color
                self.play(
                    Flash(
                        self.squares[pos_idx],
                        color=GREEN if result else RED,
                        flash_radius=0.3,
                    ),
                    self.squares[pos_idx].animate.set_fill(
                        color=GREEN if result else RED, opacity=0.3
                    ),
                    run_time=0.05,
                )

                # ### Reuse the output_text object by updating its content.
                if play_pop:
                    new_output_content = "Keep" if result else "Mask"
                    self.output_text.become(
                        Text(
                            new_output_content,
                            font_size=20,
                            color=GREEN if result else RED,
                        ).move_to(self.equation.get_center() + DOWN * 0.5)
                    )
                    self.play(FadeIn(self.output_text, run_time=0.1))
                    self.play(FadeOut(self.output_text, run_time=0.1))

        # Remove the reused objects once done.
        self.remove(self.highlight_box, self.output_text)

    def show_completion(self):
        final_text = Text("Causal Block Mask Pattern Complete", font_size=24).to_edge(
            DOWN
        )
        self.play(Write(final_text))
        self.wait(0.5)


class BlockMaskKVCreation(Scene):
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
        title = Text("BlockMask Construction", font_size=32).to_edge(UP)

        # Create attention scores title
        attention_title = Text("Attention Scores", font_size=24).set_color(YELLOW)

        # Create kv_num_blocks matrix
        num_blocks_title = Text("kv_num_blocks", font_size=24).set_color(YELLOW)
        num_blocks_matrix = VGroup()
        for i in range(num_block_rows):
            square = Square(
                side_length=square_size,
                stroke_color=WHITE,
                stroke_width=1,
                fill_opacity=0.1,
            )
            text = Text("?", font_size=20)
            text.move_to(square.get_center())
            num_blocks_matrix.add(VGroup(square, text))
        num_blocks_matrix.arrange(DOWN, buff=0)

        # Create kv_indices matrix
        kv_indices_title = Text("kv_indices", font_size=24).set_color(YELLOW)
        kv_indices_matrix = VGroup()
        for i in range(num_block_rows):
            row = VGroup()
            for j in range(num_block_rows):
                square = Square(
                    side_length=square_size,
                    stroke_color=WHITE,
                    stroke_width=1,
                    fill_opacity=0.1,
                )
                text = Text("?", font_size=20)
                text.move_to(square.get_center())
                row.add(VGroup(square, text))
            row.arrange(RIGHT, buff=0)
            kv_indices_matrix.add(row)
        kv_indices_matrix.arrange(DOWN, buff=0)

        # Position matrices
        attention_title.next_to(squares, UP, buff=0.8)
        attention_title.set_x(squares.get_center()[0])

        num_blocks_title.next_to(attention_matrix, RIGHT, buff=2)
        num_blocks_matrix.next_to(num_blocks_title, DOWN, buff=0.5)
        num_blocks_matrix.align_to(squares, UP)
        num_blocks_title.next_to(num_blocks_matrix, UP, buff=0.3)
        num_blocks_title.align_to(attention_title, UP)

        kv_indices_title.next_to(num_blocks_title, RIGHT, buff=2)
        kv_indices_matrix.next_to(kv_indices_title, DOWN, buff=0.5)
        kv_indices_matrix.align_to(squares, UP)
        kv_indices_title.next_to(kv_indices_matrix, UP, buff=0.3)
        kv_indices_title.align_to(attention_title, UP)

        # Initial animation
        self.play(Write(title))
        self.play(Create(attention_matrix))
        # self.play(ShowCreation(attention_matrix))
        self.play(
            Write(attention_title), Write(num_blocks_title), Write(kv_indices_title)
        )
        self.play(Create(num_blocks_matrix), Create(kv_indices_matrix))
        self.wait()

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
                    stroke_color=YELLOW,
                    fill_color=YELLOW,
                    fill_opacity=0.0,
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
                stroke_color=YELLOW,
                fill_color=YELLOW,
                fill_opacity=0.1,
            ).move_to(num_blocks_matrix[block_row].get_center())

            kv_indices_highlight = Rectangle(
                width=num_block_rows * square_size,
                height=square_size,
                stroke_color=YELLOW,
                fill_color=YELLOW,
                fill_opacity=0.1,
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
                        stroke_color=YELLOW,
                        fill_color=YELLOW,
                        fill_opacity=0.0,
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
                self.play(Create(attention_highlight_col), run_time=0.2)
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
                            new_square.set_fill(GREEN, opacity=0.7)
                        else:
                            new_square.set_fill(RED, opacity=0.1)

                        block_animations.append(Transform(squares[pos_idx], new_square))

                # Play all animations for the 2x2 block simultaneously
                self.play(*block_animations, run_time=0.2)

                if block_needed:
                    num_computed_blocks += 1

                    # Update matrices simultaneously
                    updates = []

                    # Update num_blocks
                    new_num_text = Text(str(num_computed_blocks), font_size=20)
                    new_num_text.move_to(num_blocks_matrix[block_row][1].get_center())
                    updates.append(
                        Transform(num_blocks_matrix[block_row][1], new_num_text)
                    )

                    # Update kv_indices
                    new_idx_text = Text(str(block_col), font_size=20)
                    new_idx_text.move_to(
                        kv_indices_matrix[block_row][block_col][1].get_center()
                    )
                    updates.append(
                        Transform(
                            kv_indices_matrix[block_row][block_col][1], new_idx_text
                        )
                    )

                    self.play(*updates, run_time=0.2)
                else:
                    # Update kv_indices with dash
                    new_text = Text("-", font_size=20)
                    new_text.move_to(
                        kv_indices_matrix[block_row][block_col][1].get_center()
                    )
                    self.play(
                        Transform(kv_indices_matrix[block_row][block_col][1], new_text),
                        run_time=0.2,
                    )

                # Remove column highlight after processing block
                self.play(FadeOut(attention_highlight_col), run_time=0.2)

            # Remove remaining highlights
            self.play(
                FadeOut(attention_highlight_row),
                FadeOut(num_blocks_highlight),
                FadeOut(kv_indices_highlight),
            )
            self.wait(0.5)

        # # Final state
        # final_text = Text("BlockMask Data Structure Complete", font_size=24).to_edge(
        #     DOWN
        # )
        # self.play(Write(final_text))
        # self.wait(2)

    def setup_attention_matrix(self, seq_len_q, seq_len_kv):
        squares = VGroup()
        for i in range(seq_len_q):
            for j in range(seq_len_kv):
                square = Square(
                    side_length=0.5,
                    stroke_color=WHITE,
                    stroke_width=1,
                    fill_opacity=0.1,
                )
                square.move_to([j * 0.5, -i * 0.5, 0])
                squares.add(square)

        # Add index labels (for query and key)
        q_indices = VGroup(
            *[
                Text(str(i), font_size=20, color=BLUE).next_to(
                    squares[i * seq_len_q], LEFT, buff=0.3
                )
                for i in range(seq_len_q)
            ]
        )
        k_indices = VGroup(
            *[
                Text(str(i), font_size=20, color=RED).next_to(squares[i], UP, buff=0.3)
                for i in range(seq_len_kv)
            ]
        )

        attention_matrix = VGroup(squares, q_indices, k_indices)
        return attention_matrix, squares
