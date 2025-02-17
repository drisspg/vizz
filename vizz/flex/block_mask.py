from manimlib import *


def causal_attention(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


class BlockMaskDataAnimation(Scene):
    def construct(self):
        # Initial setup
        seq_len = 8
        block_size = 2
        num_block_rows = seq_len // block_size
        square_size = 0.5

        # Create attention matrix
        squares = VGroup()
        for i in range(seq_len):
            for j in range(seq_len):
                square = Square(
                    side_length=square_size,
                    stroke_color=WHITE,
                    stroke_width=1,
                    fill_opacity=0.1,
                )
                square.move_to([j * square_size, -i * square_size, 0])
                squares.add(square)

        # Create axis labels
        q_indices = VGroup(
            *[
                Text(str(i), font_size=20).next_to(squares[i * seq_len], LEFT, buff=0.3)
                for i in range(seq_len)
            ]
        )
        k_indices = VGroup(
            *[
                Text(str(i), font_size=20).next_to(squares[i], UP, buff=0.3)
                for i in range(seq_len)
            ]
        )

        # Group attention matrix components
        attention_matrix = VGroup(squares, q_indices, k_indices)
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

        # Position matrices aligned with attention matrix
        # Position attention matrix and title - accounting for index labels
        attention_title.next_to(
            squares, UP, buff=0.8
        )  # Increased buffer to make room for indices
        attention_title.set_x(
            squares.get_center()[0]
        )  # Center align with attention matrix

        # Position num_blocks matrix and title
        num_blocks_title.next_to(attention_matrix, RIGHT, buff=2)
        num_blocks_matrix.next_to(num_blocks_title, DOWN, buff=0.5)
        num_blocks_matrix.align_to(squares, UP)  # Align with attention matrix top
        num_blocks_title.next_to(num_blocks_matrix, UP, buff=0.3)
        num_blocks_title.align_to(attention_title, UP)  # Align with attention title

        # Position kv_indices matrix and title
        kv_indices_title.next_to(num_blocks_title, RIGHT, buff=2)
        kv_indices_matrix.next_to(kv_indices_title, DOWN, buff=0.5)
        kv_indices_matrix.align_to(squares, UP)  # Align with attention matrix top
        kv_indices_title.next_to(kv_indices_matrix, UP, buff=0.3)
        kv_indices_title.align_to(attention_title, UP)  # Align with attention title

        # Initial animation
        self.play(Write(title))
        self.play(ShowCreation(attention_matrix))
        self.play(
            Write(attention_title), Write(num_blocks_title), Write(kv_indices_title)
        )
        self.play(ShowCreation(num_blocks_matrix), ShowCreation(kv_indices_matrix))
        self.wait()

        # Process each query block row
        for block_row in range(num_block_rows):
            q_start = block_row * block_size
            q_end = q_start + block_size
            num_computed_blocks = 0

            # Create highlights for all three matrices
            # Attention matrix highlight
            attention_highlight = (
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

            # num_blocks matrix highlight
            num_blocks_highlight = Rectangle(
                width=square_size,
                height=square_size,
                stroke_color=YELLOW,
                fill_color=YELLOW,
                fill_opacity=0.1,
            ).move_to(num_blocks_matrix[block_row].get_center())

            # kv_indices matrix highlight
            kv_indices_highlight = Rectangle(
                width=num_block_rows * square_size,
                height=square_size,
                stroke_color=YELLOW,
                fill_color=YELLOW,
                fill_opacity=0.1,
            ).move_to(kv_indices_matrix[block_row].get_center())

            # Show all highlights
            self.play(
                ShowCreation(attention_highlight),
                ShowCreation(num_blocks_highlight),
                ShowCreation(kv_indices_highlight),
            )

            # Process each key block in the row
            for block_col in range(num_block_rows):
                k_start = block_col * block_size
                k_end = k_start + block_size

                # Check causal attention for each position in the block
                block_needed = False
                for q_idx in range(q_start, q_end):
                    for k_idx in range(k_start, k_end):
                        can_attend = causal_attention(0, 0, q_idx, k_idx)
                        pos_idx = q_idx * seq_len + k_idx

                        if can_attend:
                            block_needed = True
                            new_square = squares[pos_idx].copy()
                            new_square.set_fill(GREEN, opacity=0.7)
                            self.play(
                                Transform(squares[pos_idx], new_square), run_time=0.1
                            )
                        else:
                            new_square = squares[pos_idx].copy()
                            new_square.set_fill(WHITE, opacity=0.1)
                            self.play(
                                Transform(squares[pos_idx], new_square), run_time=0.1
                            )

                if block_needed:
                    num_computed_blocks += 1

                    # Update num_blocks matrix
                    new_text = Text(str(num_computed_blocks), font_size=20)
                    new_text.move_to(num_blocks_matrix[block_row][1].get_center())
                    self.play(Transform(num_blocks_matrix[block_row][1], new_text))

                    # Update kv_indices matrix
                    new_text = Text(str(block_col), font_size=20)
                    new_text.move_to(
                        kv_indices_matrix[block_row][block_col][1].get_center()
                    )
                    self.play(
                        Transform(kv_indices_matrix[block_row][block_col][1], new_text),
                        run_time=0.2,
                    )
                else:
                    # Update kv_indices with dash for unused block
                    new_text = Text("-", font_size=20)
                    new_text.move_to(
                        kv_indices_matrix[block_row][block_col][1].get_center()
                    )
                    self.play(
                        Transform(kv_indices_matrix[block_row][block_col][1], new_text),
                        run_time=0.2,
                    )

            # Remove all highlights
            self.play(
                FadeOut(attention_highlight),
                FadeOut(num_blocks_highlight),
                FadeOut(kv_indices_highlight),
            )
            self.wait(0.5)

        # Final state
        final_text = Text("BlockMask Data Structure Complete", font_size=24).to_edge(
            DOWN
        )
        self.play(Write(final_text))
        self.wait(2)


def main():
    from manimlib import config

    module = BlockMaskDataAnimation()
    config.run_module(module)
