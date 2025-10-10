"""Per-row floating point quantization visualization.

Shows BF16 → FP8 quantization using row-wise amax-based scales.
Demonstrates per-channel (per-row) quantization with ONE scale per row.

Formula (per row):
    amax_row = max(abs(row))
    scale_row = amax_row / FP8_MAX
    quantized_row = clamp(row / scale_row, -FP8_MAX, FP8_MAX).to(fp8)

To run:
manim-slides render vizz/quant/per_row_fp_quant.py PerRowFPQuant -ql
manim-slides present PerRowFPQuant
"""

import torch
from manim import *
from manim_slides import Slide

from vizz.quant.mx_base import COLORS


class PerRowFPQuant(Slide):
    """Per-row floating point quantization with one scale per row."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Dimensions
        self.num_rows = 4
        self.num_cols = 16
        self.FP8_MAX = 448.0  # Max value for FP8 E4M3

        # Matrix positions
        self.bf16_y = 2.0
        self.fp8_y = self.bf16_y - 2.2
        self.scale_y = self.fp8_y - 1.8

        # Matrix scales
        self.matrix_scale = 0.55

        # Font sizes
        self.matrix_font_size = 18
        self.scale_font_size = 20
        self.subtitle_font_size = 20

        # Spacing
        self.matrix_h_buff = 0.3
        self.matrix_v_buff = 0.45
        self.scale_v_buff = 0.8
        self.subtitle_buff = 0.35

    def construct(self):
        """Show per-row FP quantization."""
        # Setup quantization data
        self._setup_quantization_data()

        # Title
        title = Text(
            "Per-Row Floating Point Quantization",
            font_size=32,
            color=COLORS["text"],
            weight=BOLD,
        )
        title.to_edge(UP, buff=0.2)
        self.play(Write(title))
        self.next_slide()

        # Create input matrix
        bf16_matrix, bf16_entries, bf16_subtitle = self._create_input_matrix()
        self.next_slide()

        # Create output matrix and scale visualization
        fp8_matrix, fp8_entries, fp8_subtitle = self._create_output_matrix()
        scale_display = self._create_scale_display()
        self.next_slide()

        # Animate row-by-row processing
        self._animate_row_processing(bf16_entries, fp8_entries, scale_display)

        # Show summary
        self._show_summary()
        self.next_slide()

    def _setup_quantization_data(self):
        """Create tensors and run per-row FP quantization."""
        torch.manual_seed(42)
        self.tensor_bf16 = torch.randn(
            self.num_rows, self.num_cols, dtype=torch.bfloat16
        )

        # Calculate amax per row
        self.amax_per_row = torch.max(
            torch.abs(self.tensor_bf16), dim=1
        ).values  # Shape: [num_rows]
        self.scale_per_row = self.amax_per_row / self.FP8_MAX

        # Quantize per row
        self.tensor_fp8 = torch.clamp(
            self.tensor_bf16.to(torch.float32) / self.scale_per_row.unsqueeze(1),
            -self.FP8_MAX,
            self.FP8_MAX,
        )

    def _create_input_matrix(self):
        """Create BF16 input matrix visualization."""
        matrix_data = self.tensor_bf16.to(torch.float32).cpu().numpy()

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

        for entry in matrix.get_entries():
            entry.set_color(COLORS["text"])

        matrix.scale(self.matrix_scale)
        matrix.move_to([0, self.bf16_y, 0])

        subtitle = Text(
            f"BFloat16 Input [{self.num_rows}×{self.num_cols}]",
            font_size=self.subtitle_font_size,
            color=COLORS["bf16_color"],
            weight=BOLD,
        )
        subtitle.next_to(matrix, UP, buff=self.subtitle_buff)

        self.play(Write(subtitle))
        self.play(Create(matrix), run_time=1.5)

        entries = matrix.get_entries()
        return matrix, entries, subtitle

    def _create_output_matrix(self):
        """Create FP8 output matrix scaffolding."""
        matrix_data = self.tensor_fp8.cpu().numpy()

        matrix = DecimalMatrix(
            matrix_data,
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

        for entry in matrix.get_entries():
            entry.set_color(COLORS["text"])
            entry.set_opacity(0.0)

        matrix.scale(self.matrix_scale)
        matrix.move_to([0, self.fp8_y, 0])

        subtitle = Text(
            f"FP8 E4M3 Output [{self.num_rows}×{self.num_cols}]",
            font_size=self.subtitle_font_size,
            color=COLORS["fp8_color"],
            weight=BOLD,
        )
        subtitle.next_to(matrix, UP, buff=self.subtitle_buff)

        entries = matrix.get_entries()
        return matrix, entries, subtitle

    def _create_scale_display(self):
        """Create scale values display (scaffolding)."""
        scales_np = self.scale_per_row.cpu().numpy()

        # Create a column vector of scales
        scale_entries = VGroup()
        for i, scale_val in enumerate(scales_np):
            scale_text = DecimalNumber(
                scale_val,
                num_decimal_places=4,
                font_size=self.scale_font_size,
                color=COLORS["text"],
            )
            scale_text.set_opacity(0.0)
            scale_entries.add(scale_text)

        scale_entries.arrange(DOWN, buff=self.scale_v_buff, aligned_edge=LEFT)
        scale_entries.move_to([0, self.scale_y, 0])

        subtitle = Text(
            f"Scales [{self.num_rows}]",
            font_size=self.subtitle_font_size,
            color=COLORS["computed_scale"],
            weight=BOLD,
        )
        subtitle.next_to(scale_entries, UP, buff=self.subtitle_buff)

        return scale_entries, subtitle

    def _animate_row_processing(self, bf16_entries, fp8_entries, scale_display):
        """Animate processing each row."""
        scale_entries, scale_subtitle = scale_display
        current_highlight = None

        # Show scaffolding
        self.play(Write(scale_subtitle), run_time=0.5)

        for row_idx in range(self.num_rows):
            # Get entries for this row
            start_idx = row_idx * self.num_cols
            end_idx = start_idx + self.num_cols

            row_entries_bf16 = VGroup(
                *[bf16_entries[i] for i in range(start_idx, end_idx)]
            )
            row_entries_fp8 = VGroup(
                *[fp8_entries[i] for i in range(start_idx, end_idx)]
            )
            scale_entry = scale_entries[row_idx]

            # Create highlights
            bf16_highlight = SurroundingRectangle(
                row_entries_bf16,
                color=COLORS["active_block"],
                buff=0.05,
                stroke_width=3,
            )

            fp8_highlight = SurroundingRectangle(
                row_entries_fp8,
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

            # Animate
            if current_highlight is None:
                # First row - create highlights and show formula
                formula = MathTex(
                    r"\text{amax}_{\text{row}} = \max(|\text{row}|),\quad \text{scale} = \frac{\text{amax}}{\text{FP8\_MAX}}",
                    font_size=16,
                    color=COLORS["text"],
                )
                formula.to_edge(DOWN, buff=0.3)
                self.play(Write(formula), run_time=1)

                self.play(
                    Create(bf16_highlight),
                    Create(fp8_highlight),
                    Create(scale_highlight),
                    run_time=0.5,
                )
                current_highlight = VGroup(
                    bf16_highlight, fp8_highlight, scale_highlight
                )
                self.formula = formula
            else:
                # Move highlights to new row
                new_bf16_highlight = SurroundingRectangle(
                    row_entries_bf16,
                    color=COLORS["active_block"],
                    buff=0.05,
                    stroke_width=3,
                )
                new_fp8_highlight = SurroundingRectangle(
                    row_entries_fp8,
                    color=COLORS["fp8_color"],
                    buff=0.05,
                    stroke_width=3,
                )
                new_scale_highlight = SurroundingRectangle(
                    scale_entry,
                    color=COLORS["computed_scale"],
                    buff=0.1,
                    stroke_width=3,
                )

                self.play(
                    Transform(current_highlight[0], new_bf16_highlight),
                    Transform(current_highlight[1], new_fp8_highlight),
                    Transform(current_highlight[2], new_scale_highlight),
                    run_time=0.3,
                )

            # Reveal the quantized values for this row
            animations = [
                fp8_entries[i].animate.set_opacity(1.0)
                for i in range(start_idx, end_idx)
            ]
            animations.append(scale_entry.animate.set_opacity(1.0))

            self.play(*animations, run_time=0.5)

        # Fade out highlights and formula
        if current_highlight:
            self.play(FadeOut(current_highlight), FadeOut(self.formula), run_time=0.3)

    def _show_summary(self):
        """Display summary message."""
        summary = VGroup(
            Text(
                f"✓ {self.num_rows} scales (one per row)",
                font_size=18,
                color=COLORS["text"],
                weight=BOLD,
            ),
            Text(
                "✓ Better precision than per-tensor",
                font_size=16,
                color=COLORS["text"],
            ),
            Text(
                "✓ Commonly used for weights in LLMs",
                font_size=16,
                color=COLORS["text"],
            ),
        )
        summary.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        summary.to_edge(DOWN, buff=0.3)
        self.play(Write(summary), run_time=2)
