"""Comparison of block-wise floating point quantization: MX vs NVFP4.

Shows the difference between MX format (power-of-2 scales) and NVFP4 format
(simple division scales), both using amax-based block quantization.

MX Formula (block_size=32):
    amax_block = max(abs(block))
    scale = 2^(floor(log2(amax)) - target_max_pow2)  # Power-of-2!
    quantized = block / scale

NVFP4 Formula (block_size=16):
    amax_block = max(abs(block))
    scale = amax_block / FP4_MAX  # Simple division!
    quantized = block / scale

To run:
manim-slides render vizz/quant/block_fp_quant_comparison.py BlockFPQuantComparison -ql
manim-slides present BlockFPQuantComparison
"""

import torch
from manim import *
from manim_slides import Slide

from torchao.prototype.mx_formats.mx_tensor import to_mx, get_fp_scale
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

from vizz.quant.mx_base import COLORS


class BlockFPQuantComparison(Slide):
    """Comparison of MX (block_size=32) vs NVFP4 (block_size=16) block quantization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Dimensions
        self.num_rows = 2
        self.num_cols = 64  # Divisible by both 32 and 16

        # Block sizes
        self.mx_block_size = 32
        self.nvfp4_block_size = 16

        # Matrix positions
        self.input_y = 2.5
        self.mx_y = 0.5
        self.nvfp4_y = -1.5

        # Matrix scales
        self.matrix_scale = 0.45

        # Font sizes
        self.matrix_font_size = 14
        self.subtitle_font_size = 18
        self.title_font_size = 20

        # Spacing
        self.matrix_h_buff = 0.25
        self.matrix_v_buff = 0.4
        self.subtitle_buff = 0.3

    def construct(self):
        """Show comparison of MX vs NVFP4 block quantization."""
        # Setup quantization data
        self._setup_quantization_data()

        # Title
        title = Text(
            "Block-wise FP Quantization: MX vs NVFP4",
            font_size=32,
            color=COLORS["text"],
            weight=BOLD,
        )
        title.to_edge(UP, buff=0.2)
        self.play(Write(title))
        self.next_slide()

        # Create input matrix
        input_matrix, input_entries, input_subtitle = self._create_input_matrix()
        self.next_slide()

        # Show MX quantization
        self._show_mx_quantization(input_matrix, input_entries)
        self.next_slide()

        # Show NVFP4 quantization
        self._show_nvfp4_quantization(input_matrix, input_entries)
        self.next_slide()

        # Show comparison summary
        self._show_comparison()
        self.next_slide()

    def _setup_quantization_data(self):
        """Create tensors and run both MX and NVFP4 quantization."""
        torch.manual_seed(42)
        self.tensor_bf16 = torch.abs(
            torch.randn(self.num_rows, self.num_cols, dtype=torch.bfloat16)
        )

        # MX quantization (FP8, block_size=32)
        self.mx_scales_e8m0, self.mx_data_fp8 = to_mx(
            self.tensor_bf16,
            elem_dtype=torch.float8_e4m3fn,
            block_size=self.mx_block_size,
            scaling_mode=ScaleCalculationMode.FLOOR,
        )
        self.mx_scales_fp32 = get_fp_scale(self.mx_scales_e8m0)

        # NVFP4 quantization (FP4, block_size=16)
        self.nvfp4_tensor = NVFP4Tensor.to_nvfp4(
            self.tensor_bf16,
            block_size=self.nvfp4_block_size,
        )
        self.nvfp4_data = self.nvfp4_tensor.to_dtype(torch.float32)
        self.nvfp4_scales = self.nvfp4_tensor.get_hp_scales()

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
        matrix.move_to([0, self.input_y, 0])

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

    def _show_mx_quantization(self, input_matrix, input_entries):
        """Show MX quantization results."""
        # Title
        mx_title = Text(
            "MX Format (FP8, block_size=32)",
            font_size=self.title_font_size,
            color=COLORS["fp8_color"],
            weight=BOLD,
        )
        mx_title.move_to([-3.5, self.mx_y + 1.5, 0])

        # Formula
        mx_formula = VGroup(
            Text("Power-of-2 scales:", font_size=14, color=COLORS["text"]),
            MathTex(
                r"\text{scale} = 2^{\lfloor \log_2(\text{amax}) - \text{max\_exp} \rfloor}",
                font_size=14,
                color=COLORS["computed_scale"],
            ),
        )
        mx_formula.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        mx_formula.next_to(mx_title, DOWN, buff=0.2, aligned_edge=LEFT)

        self.play(Write(mx_title), Write(mx_formula), run_time=1.5)

        # Show scales
        # num_mx_blocks = self.num_cols // self.mx_block_size
        mx_scale_text = Text(
            f"Scales: {self.mx_scales_fp32[0][0].item():.2f}, {self.mx_scales_fp32[0][1].item():.2f}, ...",
            font_size=14,
            color=COLORS["computed_scale"],
        )
        mx_scale_text.next_to(mx_formula, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(Write(mx_scale_text), run_time=1)

        # Highlight first block
        first_block_entries = VGroup(
            *[input_entries[i] for i in range(self.mx_block_size)]
        )
        mx_highlight = SurroundingRectangle(
            first_block_entries,
            color=COLORS["fp8_color"],
            buff=0.05,
            stroke_width=3,
        )
        self.play(Create(mx_highlight), run_time=0.8)

        block_info = Text(
            f"Block: {self.mx_block_size} elements → 1 E8M0 scale",
            font_size=12,
            color=COLORS["text"],
        )
        block_info.next_to(mx_scale_text, DOWN, buff=0.2, aligned_edge=LEFT)
        self.play(Write(block_info), run_time=1)

        self.mx_elements = VGroup(
            mx_title, mx_formula, mx_scale_text, mx_highlight, block_info
        )

    def _show_nvfp4_quantization(self, input_matrix, input_entries):
        """Show NVFP4 quantization results."""
        # Title
        nvfp4_title = Text(
            "NVFP4 Format (FP4, block_size=16)",
            font_size=self.title_font_size,
            color=COLORS["nvfp_color"],
            weight=BOLD,
        )
        nvfp4_title.move_to([3.5, self.nvfp4_y + 1.5, 0])

        # Formula
        nvfp4_formula = VGroup(
            Text("Simple division scales:", font_size=14, color=COLORS["text"]),
            MathTex(
                r"\text{scale} = \frac{\text{amax}}{\text{FP4\_MAX}}",
                font_size=14,
                color=COLORS["computed_scale"],
            ),
        )
        nvfp4_formula.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        nvfp4_formula.next_to(nvfp4_title, DOWN, buff=0.2, aligned_edge=LEFT)

        self.play(Write(nvfp4_title), Write(nvfp4_formula), run_time=1.5)

        # Show scales
        nvfp4_scale_text = Text(
            f"Scales: {self.nvfp4_scales[0][0].item():.3f}, {self.nvfp4_scales[0][1].item():.3f}, {self.nvfp4_scales[0][2].item():.3f}, {self.nvfp4_scales[0][3].item():.3f}, ...",
            font_size=12,
            color=COLORS["computed_scale"],
        )
        nvfp4_scale_text.next_to(nvfp4_formula, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(Write(nvfp4_scale_text), run_time=1)

        # Highlight first block
        first_block_entries = VGroup(
            *[input_entries[i] for i in range(self.nvfp4_block_size)]
        )
        nvfp4_highlight = SurroundingRectangle(
            first_block_entries,
            color=COLORS["nvfp_color"],
            buff=0.05,
            stroke_width=3,
        )
        self.play(Create(nvfp4_highlight), run_time=0.8)

        block_info = Text(
            f"Block: {self.nvfp4_block_size} elements → 1 E4M3 scale",
            font_size=12,
            color=COLORS["text"],
        )
        block_info.next_to(nvfp4_scale_text, DOWN, buff=0.2, aligned_edge=LEFT)
        self.play(Write(block_info), run_time=1)

        self.nvfp4_elements = VGroup(
            nvfp4_title, nvfp4_formula, nvfp4_scale_text, nvfp4_highlight, block_info
        )

    def _show_comparison(self):
        """Show side-by-side comparison."""
        comparison = VGroup(
            Text("Key Differences:", font_size=20, color=COLORS["text"], weight=BOLD),
            VGroup(
                Text("MX:", font_size=16, color=COLORS["fp8_color"], weight=BOLD),
                Text("• Power-of-2 scales (E8M0)", font_size=14, color=COLORS["text"]),
                Text("• Block size: 32 elements", font_size=14, color=COLORS["text"]),
                Text("• More hardware-friendly", font_size=14, color=COLORS["text"]),
            ).arrange(DOWN, buff=0.15, aligned_edge=LEFT),
            VGroup(
                Text("NVFP4:", font_size=16, color=COLORS["nvfp_color"], weight=BOLD),
                Text(
                    "• Simple division scales (E4M3)",
                    font_size=14,
                    color=COLORS["text"],
                ),
                Text(
                    "• Block size: 16 elements (finer granularity)",
                    font_size=14,
                    color=COLORS["text"],
                ),
                Text(
                    "• Better precision per block", font_size=14, color=COLORS["text"]
                ),
            ).arrange(DOWN, buff=0.15, aligned_edge=LEFT),
        )
        comparison.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        comparison.to_edge(DOWN, buff=0.3)

        self.play(Write(comparison), run_time=3)
