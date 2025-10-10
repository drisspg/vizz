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

from torchao.prototype.mx_formats.mx_tensor import to_mx, get_fp_scale
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.constants import F8E4M3_MAX_POW2

from vizz.quant.mx_base import COLORS
from vizz.quant.utils import calculate_e8m0_scale_rtne


class MXBlockScaling(Slide):
    """Deep dive into MX format scaling with 32-element blocks using torchao."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Dimensions
        self.num_rows = 2
        self.num_cols = 16
        self.block_size = 4
        self.num_blocks = self.num_cols // self.block_size

        # Matrix positions
        # BF16 on the left (bigger)
        self.bf16_x = -3.5
        self.bf16_y = 0.0

        # Scales and FP8 on the right (stacked)
        self.right_x = 3.0
        self.scale_y = 1.5  # Scales on top
        self.fp8_y = -1.5  # FP8 below

        # Matrix scales
        self.bf16_scale = 1.0  # Bigger for BF16 on left
        self.matrix_scale = 1.0  # Smaller for FP8 and scales on right
        self.scale_matrix_scale = 0.5  # For E8M0 scales

        # Font sizes
        self.matrix_font_size = 14
        self.scale_font_size = 24
        self.subtitle_font_size = 18
        self.amax_font_size = 20

        # Spacing
        self.matrix_h_buff = 0.25
        self.matrix_v_buff = 0.4
        self.scale_h_buff = 1
        self.scale_v_buff = 1
        self.subtitle_buff = 0.3

        # Function visualization position (between BF16 and scales)
        self.func_x = 0.0
        self.func_y = 1.5

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
        self.tensor_bf16 = torch.abs(
            torch.randn(self.num_rows, self.num_cols, dtype=torch.bfloat16)
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

        # Scale and position on the left (bigger)
        matrix.scale(self.bf16_scale)
        matrix.move_to([self.bf16_x, self.bf16_y, 0])

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
        data = self.data_fp8.to(torch.float32)
        scales_fp32 = get_fp_scale(self.scales_e8m0)
        fp_8_data = (
            (data * scales_fp32.repeat_interleave(self.block_size, dim=1)).cpu().numpy()
        )
        matrix = DecimalMatrix(
            fp_8_data,
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

        # Scale and position on the right (bottom)
        matrix.scale(self.matrix_scale)
        matrix.move_to([self.right_x, self.fp8_y, 0])

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

    def _create_transformation_flow(self, block_values, amax_value, row, block_idx):
        """Create visualization showing block → amax → amax_to_scale() → scale.

        Args:
            block_values: Tensor of values from the BF16 block
            amax_value: The maximum absolute value from the block
            row: Current row index
            block_idx: Current block index

        Returns:
            Tuple of (block_display, amax_display, scale_fp32_display, scale_e8m0_display, func_label)
        """
        # Create compact display of block values (e.g., "[1.2, 3.4, ...]")
        values_str = ", ".join([f"{v:.1f}" for v in block_values[:4].tolist()])
        if len(block_values) > 4:
            values_str += ", ..."

        block_text = Text(
            f"[{values_str}]",
            font_size=16,
            color=COLORS["active_block"],
            weight=BOLD,
        )

        # Show the extracted max value
        amax_text = Text(
            f"amax={amax_value:.2f}",
            font_size=self.amax_font_size,
            color=COLORS["max_value"],
            weight=BOLD,
        )

        # Compute the scale using RTNE method
        scale_fp32, e8m0_biased, e8m0_unbiased = calculate_e8m0_scale_rtne(
            amax_value, F8E4M3_MAX_POW2
        )

        # Show FP32 scale (power of 2)
        scale_fp32_text = Text(
            f"scale=2^{e8m0_unbiased}={scale_fp32:.3g}",
            font_size=16,
            color=COLORS["computed_scale"],
            weight=BOLD,
        )

        # Show E8M0 encoded value
        e8m0_byte = self.scales_e8m0[row, block_idx].view(torch.uint8).item()
        scale_e8m0_text = Text(
            f"E8M0={e8m0_byte}",
            font_size=self.amax_font_size,
            color=COLORS["computed_scale"],
            weight=BOLD,
        )

        # Function label with box
        func_label = Text(
            "amax_to_scale()",
            font_size=self.subtitle_font_size,
            color=COLORS["computed_scale"],
            weight=BOLD,
        )
        func_box = SurroundingRectangle(
            func_label,
            color=COLORS["computed_scale"],
            buff=0.2,
            stroke_width=2,
        )
        func_group = VGroup(func_box, func_label)

        return block_text, amax_text, scale_fp32_text, scale_e8m0_text, func_group

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
        matrix.move_to([self.right_x, self.scale_y, 0])

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
        """Animate scanning window with staggered causality: BF16 → Scale → FP8."""
        # Scan across columns, then down rows
        current_bf16_highlight = None
        current_scale_highlight = None
        current_fp8_highlight = None

        # Show detailed amax_to_scale flow for first 3 blocks, then simplify
        block_counter = 0
        show_detailed_flow = True

        for row in range(self.num_rows):
            for block_idx in range(self.num_blocks):
                # Update whether to show detailed flow
                show_detailed_flow = block_counter < 3
                block_counter += 1
                # Calculate which entries to highlight
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

                scale_highlight = SurroundingRectangle(
                    scale_entry,
                    color=COLORS["computed_scale"],
                    buff=0.1,
                    stroke_width=3,
                )

                fp8_highlight = SurroundingRectangle(
                    block_entries_fp8,
                    color=COLORS["fp8_color"],
                    buff=0.05,
                    stroke_width=3,
                )

                # === STAGE 1: Highlight BF16 block ===
                if current_bf16_highlight is None:
                    # First block - show scale scaffolding first (causal order)
                    self.play(Write(scale_subtitle), run_time=0.5)
                    self.play(Create(scale_matrix), run_time=1.0)
                    self.play(Create(bf16_highlight), run_time=0.5)
                    current_bf16_highlight = bf16_highlight
                else:
                    # Move BF16 highlight to new block
                    self.play(
                        Transform(current_bf16_highlight, bf16_highlight),
                        run_time=0.3,
                    )

                # === STAGE 2: Compute and reveal scale ===
                if current_scale_highlight is None:
                    self.play(Create(scale_highlight), run_time=0.4)
                    current_scale_highlight = scale_highlight
                else:
                    self.play(
                        Transform(current_scale_highlight, scale_highlight),
                        run_time=0.3,
                    )

                # Get positions for flow visualization
                bf16_block_center = block_entries_bf16.get_center()
                scale_center = scale_entry.get_center()

                if show_detailed_flow:
                    # Show detailed transformation flow (first 3 blocks)
                    block_data = self.tensor_bf16[row, start_idx:end_idx]
                    amax_value = torch.max(torch.abs(block_data)).item()

                    # Create all flow elements
                    (
                        block_text,
                        amax_text,
                        scale_fp32_text,
                        scale_e8m0_text,
                        func_group,
                    ) = self._create_transformation_flow(
                        block_data, amax_value, row, block_idx
                    )

                    # Position 1: Start at BF16 block (show block values)
                    block_text.move_to(bf16_block_center + RIGHT * 1.2)
                    self.play(FadeIn(block_text, shift=RIGHT * 0.3), run_time=0.4)

                    # Position 2: Extract amax
                    amax_text.move_to([self.func_x - 1.8, self.func_y, 0])
                    self.play(
                        Transform(block_text, amax_text),
                        run_time=0.5,
                    )

                    # Position 3: Show function in center
                    func_group.move_to([self.func_x, self.func_y, 0])
                    self.play(FadeIn(func_group), run_time=0.3)

                    # Position 4: Move amax into function and compute FP32 scale
                    scale_fp32_text.move_to([self.func_x, self.func_y - 0.5, 0])
                    self.play(
                        block_text.animate.move_to(
                            [self.func_x, self.func_y + 0.4, 0]
                        ).scale(0.7),
                        run_time=0.3,
                    )
                    self.play(
                        FadeOut(block_text),
                        FadeIn(scale_fp32_text),
                        run_time=0.4,
                    )

                    # Position 5: Convert FP32 scale to E8M0 encoding
                    scale_e8m0_text.move_to([self.func_x + 1.8, self.func_y, 0])
                    self.play(
                        Transform(scale_fp32_text, scale_e8m0_text),
                        run_time=0.4,
                    )

                    # Position 6: Move E8M0 to scale matrix position
                    self.play(
                        scale_fp32_text.animate.move_to(scale_center),
                        run_time=0.4,
                    )

                    # Reveal the scale value in matrix and fade out temporary
                    self.play(
                        scale_entry.animate.set_opacity(1.0),
                        FadeOut(scale_fp32_text),
                        run_time=0.3,
                    )

                    # Fade out the function visualization
                    self.play(FadeOut(func_group), run_time=0.2)

                    # === STAGE 3: Use scale to compute FP8 output (sequential for first 3) ===
                    if current_fp8_highlight is None:
                        # Show FP8 scaffolding after first scale is computed
                        self.play(Write(fp8_subtitle), run_time=0.5)
                        self.play(Create(fp8_matrix), run_time=1.0)
                        self.play(Create(fp8_highlight), run_time=0.4)
                        current_fp8_highlight = fp8_highlight
                    else:
                        # Move FP8 highlight to new block
                        self.play(
                            Transform(current_fp8_highlight, fp8_highlight),
                            run_time=0.3,
                        )

                    # Reveal the FP8 values for this block
                    animations = []
                    for i in range(start_idx, end_idx):
                        entry_idx = row * self.num_cols + i
                        animations.append(
                            fp8_entries[entry_idx].animate.set_opacity(1.0)
                        )

                    # Animate all the value reveals
                    if animations:
                        self.play(*animations, run_time=0.5)
                else:
                    # Fast synchronized mode (remaining blocks): simplified flow
                    block_data = self.tensor_bf16[row, start_idx:end_idx]
                    amax_value = torch.max(torch.abs(block_data)).item()

                    # Create simple amax indicator
                    quick_amax = Text(
                        f"amax={amax_value:.1f}",
                        font_size=14,
                        color=COLORS["max_value"],
                    )
                    quick_amax.move_to(bf16_block_center + RIGHT * 1.0)

                    # === STAGE 3: Move FP8 highlight ===
                    if current_fp8_highlight is None:
                        # Should not happen since FP8 is shown in first 3 blocks
                        self.play(Write(fp8_subtitle), run_time=0.5)
                        self.play(Create(fp8_matrix), run_time=1.0)
                        self.play(Create(fp8_highlight), run_time=0.4)
                        current_fp8_highlight = fp8_highlight
                    else:
                        # Move FP8 highlight to new block
                        self.play(
                            Transform(current_fp8_highlight, fp8_highlight),
                            run_time=0.3,
                        )

                    # Prepare FP8 value animations
                    fp8_animations = []
                    for i in range(start_idx, end_idx):
                        entry_idx = row * self.num_cols + i
                        fp8_animations.append(
                            fp8_entries[entry_idx].animate.set_opacity(1.0)
                        )

                    # Show amax briefly moving to scale, reveal scale + FP8 together
                    self.play(FadeIn(quick_amax), run_time=0.2)
                    self.play(
                        quick_amax.animate.move_to(scale_center),
                        scale_entry.animate.set_opacity(1.0),
                        *fp8_animations,
                        run_time=0.4,
                    )
                    self.play(FadeOut(quick_amax), run_time=0.1)

        # Fade out highlights
        if current_bf16_highlight and current_scale_highlight and current_fp8_highlight:
            self.play(
                FadeOut(current_bf16_highlight),
                FadeOut(current_scale_highlight),
                FadeOut(current_fp8_highlight),
                run_time=0.3,
            )

    def _show_summary(self):
        """Display summary message."""
        summary = Text(
            f"Each 1×{self.block_size} block quantized to FP8 with one E8M0 scale",
            font_size=16,
            color=COLORS["text"],
        )
        summary.to_edge(DOWN, buff=0.3)
        self.play(Write(summary))
