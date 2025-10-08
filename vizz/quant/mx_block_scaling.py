"""Deep dive into MX format scaling with 32-element blocks using torchao.

Shows BF16 → FP8 quantization with E8M0 scales, demonstrating how a 1×32
scanning window moves across the reduction dimension to quantize blocks.

To run:
manim-slides render vizz/quant/mx_block_scaling.py MXBlockScaling -ql
manim-slides present MXBlockScaling
"""

import torch
from manim import *
from manim_slides import Slide

from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.config import ScaleCalculationMode

from vizz.quant.mx_base import COLORS


class MXBlockScaling(Slide):
    """Deep dive into MX format scaling with 32-element blocks using torchao."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Dimensions
        self.num_rows = 2
        self.num_cols = 128
        self.block_size = 32
        self.num_blocks = self.num_cols // self.block_size

        # Matrix positions (y-coordinates)
        self.bf16_y = 2.2
        self.fp8_y = self.bf16_y - 2.0
        self.scale_y = self.fp8_y - 2.0

        # Matrix scales
        self.matrix_scale = 0.4  # For BF16 and FP8
        self.scale_matrix_scale = 0.5  # For E8M0 scales

        # Font sizes
        self.matrix_font_size = 20
        self.scale_font_size = 24
        self.subtitle_font_size = 18

        # Spacing
        self.matrix_h_buff = 0.2
        self.matrix_v_buff = 0.4
        self.scale_h_buff = 1
        self.scale_v_buff = 1
        self.subtitle_buff = 0.3

    def construct(self):
        """Show BF16 → FP8 quantization with E8M0 scales."""
        # Setup quantization data
        self._setup_quantization_data()

        # Setup
        title = Text(
            "MX Format: BF16 → FP8 + E8M0 Scales",
            font_size=32,
            color=COLORS["text"],
            weight=BOLD,
        )
        title.to_edge(UP, buff=0.2)
        self.play(Write(title))
        self.next_slide()

        # Create full matrix visualizations
        bf16_matrix, bf16_entries, bf16_subtitle = self._create_input_matrix()
        self.next_slide()

        fp8_matrix, fp8_entries, fp8_subtitle = self._create_output_matrix()
        scale_matrix, scale_entries, scale_subtitle = self._create_scale_matrix()
        self.next_slide()

        # Animate scanning and quantization
        self._animate_scanning(
            bf16_entries,
            fp8_entries,
            scale_entries,
            fp8_matrix,
            scale_matrix,
            fp8_subtitle,
            scale_subtitle,
        )

        # Show summary
        self._show_summary()
        self.next_slide()

    def _setup_quantization_data(self):
        """Create tensors and run torchao quantization."""
        torch.manual_seed(42)
        self.tensor_bf16 = (
            torch.randn(self.num_rows, self.num_cols, dtype=torch.bfloat16) * 10
        )

        self.scales_e8m0, self.data_fp8 = to_mx(
            self.tensor_bf16,
            elem_dtype=torch.float8_e4m3fn,
            block_size=self.block_size,
            scaling_mode=ScaleCalculationMode.FLOOR,
        )

    def _create_input_matrix(self):
        """Create BF16 input matrix visualization."""
        # Convert tensor to numpy for DecimalMatrix
        matrix_data = self.tensor_bf16.to(torch.float32).cpu().numpy()

        # Create DecimalMatrix with actual numeric values
        matrix = DecimalMatrix(
            matrix_data,
            element_to_mobject_config={
                "num_decimal_places": 1,
                "font_size": self.matrix_font_size,
            },
            left_bracket="[",
            right_bracket="]",
            bracket_config={"color": COLORS["bf16_color"]},
            h_buff=self.matrix_h_buff,
            v_buff=self.matrix_v_buff,
        )

        # Explicitly set color for all entries (color config doesn't work reliably)
        for entry in matrix.get_entries():
            entry.set_color(COLORS["text"])

        # Scale to fit columns on screen
        matrix.scale(self.matrix_scale)
        matrix.move_to([0, self.bf16_y, 0])

        # Position subtitle above matrix
        subtitle = Text(
            f"BFloat16 Input [{self.num_rows}, {self.num_cols}]",
            font_size=self.subtitle_font_size,
            color=COLORS["bf16_color"],
            weight=BOLD,
        )
        subtitle.next_to(matrix, UP, buff=self.subtitle_buff)

        self.play(Write(subtitle))
        self.play(Create(matrix), run_time=2)

        # Store matrix entries for highlighting
        entries = matrix.get_entries()

        return matrix, entries, subtitle

    def _create_output_matrix(self):
        """Create FP8 output matrix scaffolding (empty placeholders)."""
        # Create DecimalMatrix with actual values (will be hidden initially)
        matrix = DecimalMatrix(
            self.data_fp8.to(torch.float32).cpu().numpy(),
            element_to_mobject_config={
                "num_decimal_places": 1,
                "font_size": self.matrix_font_size,
            },
            left_bracket="[",
            right_bracket="]",
            bracket_config={"color": COLORS["fp8_color"]},
            h_buff=self.matrix_h_buff,
            v_buff=self.matrix_v_buff,
        )

        # Set all entries to invisible initially
        for entry in matrix.get_entries():
            entry.set_color(COLORS["text"])
            entry.set_opacity(0.0)

        # Scale to fit columns on screen
        matrix.scale(self.matrix_scale)
        matrix.move_to([0, self.fp8_y, 0])

        # Position subtitle above matrix
        subtitle = Text(
            f"FP8 E4M3 Output [{self.num_rows}, {self.num_cols}]",
            font_size=self.subtitle_font_size,
            color=COLORS["fp8_color"],
            weight=BOLD,
        )
        subtitle.next_to(matrix, UP, buff=self.subtitle_buff)

        # Store matrix entries for revealing
        entries = matrix.get_entries()

        return matrix, entries, subtitle

    def _create_scale_matrix(self):
        """Create E8M0 scales scaffolding (empty placeholders)."""
        # Create IntegerMatrix with actual values (will be hidden initially)
        matrix = IntegerMatrix(
            self.scales_e8m0.view(torch.uint8).cpu().numpy(),
            left_bracket="[",
            right_bracket="]",
            bracket_config={"color": COLORS["computed_scale"]},
            element_to_mobject_config={
                "num_decimal_places": 1,
                "font_size": self.scale_font_size,
            },
            h_buff=self.scale_h_buff,
            v_buff=self.scale_v_buff,
        )

        # Set all entries to invisible initially
        for entry in matrix.get_entries():
            entry.set_color(COLORS["text"])
            entry.set_opacity(0.0)

        matrix.scale(self.scale_matrix_scale)
        matrix.move_to([0, self.scale_y, 0])

        # Position subtitle above matrix
        subtitle = Text(
            f"E8M0 Scales [{self.num_rows}, {self.num_blocks}]",
            font_size=self.subtitle_font_size,
            color=COLORS["computed_scale"],
            weight=BOLD,
        )
        subtitle.next_to(matrix, UP, buff=self.subtitle_buff)

        # Store matrix entries for revealing
        entries = matrix.get_entries()

        return matrix, entries, subtitle

    def _animate_scanning(
        self,
        bf16_entries,
        fp8_entries,
        scale_entries,
        fp8_matrix,
        scale_matrix,
        fp8_subtitle,
        scale_subtitle,
    ):
        """Animate scanning window highlighting blocks as they're processed."""
        # Scan across columns, then down rows
        current_highlight = None

        for row in range(self.num_rows):
            for block_idx in range(self.num_blocks):
                # Calculate which entries to highlight (32 consecutive elements per block)
                start_idx = block_idx * self.block_size
                end_idx = start_idx + self.block_size

                # Get the entries for this block in the BF16 input matrix
                block_entries_bf16 = VGroup(
                    *[
                        bf16_entries[row * self.num_cols + i]
                        for i in range(start_idx, end_idx)
                    ]
                )

                # Get the entries for this block in the FP8 output matrix
                block_entries_fp8 = VGroup(
                    *[
                        fp8_entries[row * self.num_cols + i]
                        for i in range(start_idx, end_idx)
                    ]
                )

                # Get the scale entry
                scale_idx = row * self.num_blocks + block_idx
                scale_entry = scale_entries[scale_idx]

                # Create highlighting rectangles
                bf16_highlight = SurroundingRectangle(
                    block_entries_bf16,
                    color=COLORS["active_block"],
                    buff=0.05,
                    stroke_width=3,
                )

                fp8_highlight = SurroundingRectangle(
                    block_entries_fp8,
                    color=COLORS["fp8_color"],
                    buff=0.05,
                    stroke_width=3,
                )

                scale_highlight = SurroundingRectangle(
                    scale_entry,
                    color=COLORS["computed_scale"],
                    buff=0.1,
                    stroke_width=3,
                )

                # Animate highlighting
                if current_highlight is None:
                    # First block - show FP8 and scale scaffolding
                    self.play(Write(fp8_subtitle), Write(scale_subtitle), run_time=0.5)
                    self.play(Create(fp8_matrix), Create(scale_matrix), run_time=1.5)
                    self.play(
                        Create(bf16_highlight),
                        Create(fp8_highlight),
                        Create(scale_highlight),
                        run_time=0.5,
                    )
                    current_highlight = VGroup(
                        bf16_highlight, fp8_highlight, scale_highlight
                    )
                else:
                    # Move highlights to new block
                    self.play(
                        Transform(current_highlight[0], bf16_highlight),
                        Transform(current_highlight[1], fp8_highlight),
                        Transform(current_highlight[2], scale_highlight),
                        run_time=0.4,
                    )

                # Reveal the FP8 values for this block
                animations = []
                for i in range(start_idx, end_idx):
                    entry_idx = row * self.num_cols + i
                    animations.append(fp8_entries[entry_idx].animate.set_opacity(1.0))

                # Reveal the scale value
                animations.append(scale_entry.animate.set_opacity(1.0))

                # Animate all the value reveals
                if animations:
                    self.play(*animations, run_time=0.6)

        # Fade out highlights
        if current_highlight:
            self.play(FadeOut(current_highlight), run_time=0.3)

    def _show_summary(self):
        """Display summary message."""
        summary = Text(
            "Each 1×32 block quantized to FP8 with one E8M0 scale",
            font_size=16,
            color=COLORS["text"],
        )
        summary.to_edge(DOWN, buff=0.3)
        self.play(Write(summary))
