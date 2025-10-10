"""Per-tensor floating point quantization visualization.

Shows BF16 → FP8 quantization using a single global amax-based scale.
Demonstrates the simplest quantization approach with ONE scale for entire tensor.

Formula:
    amax = max(abs(tensor))
    scale = amax / FP8_MAX
    quantized = clamp(tensor / scale, -FP8_MAX, FP8_MAX).to(fp8)

To run:
manim-slides render vizz/quant/per_tensor_fp_quant.py PerTensorFPQuant -ql
manim-slides present PerTensorFPQuant
"""

import torch
from manim import *
from manim_slides import Slide

from vizz.quant.mx_base import COLORS


class PerTensorFPQuant(Slide):
    """Per-tensor floating point quantization with single global scale."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Dimensions
        self.num_rows = 4
        self.num_cols = 8
        self.FP8_MAX = 448.0  # Max value for FP8 E4M3

        # Matrix positions
        self.bf16_y = 2.0
        self.fp8_y = self.bf16_y - 2.5

        # Matrix scales
        self.matrix_scale = 1.0

        # Font sizes
        self.matrix_font_size = 20
        self.subtitle_font_size = 20
        self.calc_font_size = 18

        # Spacing
        self.matrix_h_buff = 0.4
        self.matrix_v_buff = 0.5
        self.subtitle_buff = 0.4

    def construct(self):
        """Show per-tensor FP quantization."""
        # Setup quantization data
        self._setup_quantization_data()

        # Title
        title = Text(
            "Per-Tensor Floating Point Quantization",
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

        # Highlight entire matrix and show amax calculation
        self._show_amax_calculation(bf16_matrix, bf16_entries)
        self.next_slide()

        # Show scale calculation
        scale_display = self._show_scale_calculation()
        self.next_slide()

        # Create output matrix and show quantization
        fp8_matrix, fp8_entries, fp8_subtitle = self._create_output_matrix()
        self._show_quantization(
            bf16_entries, fp8_entries, scale_display, fp8_matrix, fp8_subtitle
        )
        self.next_slide()

        # Show summary
        self._show_summary()
        self.next_slide()

    def _setup_quantization_data(self):
        """Create tensors and run simple FP quantization."""
        torch.manual_seed(42)
        self.tensor_bf16 = torch.randn(
            self.num_rows, self.num_cols, dtype=torch.bfloat16
        )

        # Calculate amax and scale
        self.amax = torch.max(torch.abs(self.tensor_bf16)).item()
        self.scale = self.amax / self.FP8_MAX

        # Quantize
        self.tensor_fp8 = torch.clamp(
            self.tensor_bf16.to(torch.float32) / self.scale, -self.FP8_MAX, self.FP8_MAX
        )

    def _create_input_matrix(self):
        """Create BF16 input matrix visualization."""
        matrix_data = self.tensor_bf16.to(torch.float32).cpu().numpy()

        matrix = DecimalMatrix(
            matrix_data,
            element_to_mobject_config={
                "num_decimal_places": 2,
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

    def _show_amax_calculation(self, bf16_matrix, bf16_entries):
        """Highlight entire matrix and show amax calculation."""
        # Highlight entire matrix
        all_entries = VGroup(*bf16_entries)
        matrix_highlight = SurroundingRectangle(
            all_entries,
            color=COLORS["active_block"],
            buff=0.1,
            stroke_width=4,
        )

        # Show calculation
        calc_text = VGroup(
            Text(
                "Step 1: Find absolute maximum of entire tensor",
                font_size=self.calc_font_size,
                color=COLORS["text"],
                weight=BOLD,
            ),
            MathTex(
                r"\text{amax} = \max(|\text{tensor}|)",
                font_size=self.calc_font_size,
                color=COLORS["max_value"],
            ),
            MathTex(
                f"\\text{{amax}} = {self.amax:.4f}",
                font_size=self.calc_font_size,
                color=COLORS["max_value"],
            ),
        )
        calc_text.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        calc_text.next_to(bf16_matrix, RIGHT, buff=1.0)

        self.play(Create(matrix_highlight), run_time=0.8)
        self.play(Write(calc_text), run_time=2)

        self.matrix_highlight = matrix_highlight
        self.amax_calc = calc_text

    def _show_scale_calculation(self):
        """Show scale calculation."""
        scale_text = VGroup(
            Text(
                "Step 2: Calculate scale",
                font_size=self.calc_font_size,
                color=COLORS["text"],
                weight=BOLD,
            ),
            MathTex(
                r"\text{scale} = \frac{\text{amax}}{\text{FP8\_MAX}}",
                font_size=self.calc_font_size,
                color=COLORS["computed_scale"],
            ),
            MathTex(
                f"\\text{{scale}} = \\frac{{{self.amax:.4f}}}{{{self.FP8_MAX:.0f}}} = {self.scale:.6f}",
                font_size=self.calc_font_size,
                color=COLORS["computed_scale"],
            ),
        )
        scale_text.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        scale_text.next_to(self.amax_calc, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(Write(scale_text), run_time=2)

        # Create compact box for persistent display
        self.scale_box = VGroup(
            Text("Scale:", font_size=16, color=COLORS["text"], weight=BOLD),
            Text(
                f"{self.scale:.6f}",
                font_size=16,
                color=COLORS["computed_scale"],
                weight=BOLD,
            ),
        )
        self.scale_box.arrange(DOWN, buff=0.1)
        self.scale_box.to_corner(DR, buff=0.5)

        return scale_text

    def _create_output_matrix(self):
        """Create FP8 output matrix visualization."""
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

    def _show_quantization(
        self, bf16_entries, fp8_entries, scale_display, fp8_matrix, fp8_subtitle
    ):
        """Show quantization process."""
        # Fade out previous calculations and show scale box
        self.play(
            FadeOut(self.amax_calc),
            FadeOut(scale_display),
            FadeOut(self.matrix_highlight),
            Write(self.scale_box),
            run_time=1,
        )

        # Show formula
        formula = MathTex(
            r"\text{quantized} = \text{clamp}\left(\frac{\text{tensor}}{\text{scale}}, -\text{FP8\_MAX}, \text{FP8\_MAX}\right)",
            font_size=self.calc_font_size,
            color=COLORS["text"],
        )
        formula.next_to(fp8_matrix, DOWN, buff=0.8)

        self.play(Write(formula), run_time=1.5)
        self.next_slide()

        # Show output scaffolding
        self.play(Write(fp8_subtitle), Create(fp8_matrix), run_time=1)

        # Reveal all quantized values at once
        animations = [entry.animate.set_opacity(1.0) for entry in fp8_entries]
        self.play(*animations, run_time=1.5)

        self.play(FadeOut(formula), run_time=0.5)

    def _show_summary(self):
        """Display summary message."""
        summary = VGroup(
            Text(
                f"✓ ONE scale for entire {self.num_rows}×{self.num_cols} tensor",
                font_size=18,
                color=COLORS["text"],
                weight=BOLD,
            ),
            Text(
                f"✓ Simple: amax / FP8_MAX = {self.scale:.6f}",
                font_size=16,
                color=COLORS["text"],
            ),
            Text(
                "✓ Fast but may lose precision for outliers",
                font_size=16,
                color=COLORS["text"],
            ),
        )
        summary.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        summary.to_edge(DOWN, buff=0.3)
        self.play(Write(summary), run_time=2)
