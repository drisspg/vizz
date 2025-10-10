"""Unified floating point quantization visualization.

Shows the complete FP quantization process step-by-step with configurable:
- Scale granularity: per_tensor, per_row, per_block
- Scale dtype: e8m0 (power-of-2), e4m3, fp32
- Target dtype: fp8, fp4

Common flow for all variants:
1. Calculate amax (per tensor/row/block)
2. Calculate scale (method depends on scale_dtype)
3. Quantize: data_scaled = clamp(data / scale, -max, max)
4. Cast to target dtype

To run:
manim-slides render vizz/quant/unified_fp_quantization.py UnifiedFPQuantization -ql
manim-slides present UnifiedFPQuantization
"""

import torch
from manim import *
from manim_slides import Slide

from vizz.quant.mx_base import COLORS
from vizz.quant.utils import (
    TARGET_DTYPE_CONFIGS,
    SCALE_DTYPE_CONFIGS,
    calculate_amax,
    calculate_scale,
    quantize_data,
)


class UnifiedFPQuantization(Slide):
    """Unified FP quantization with configurable parameters."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ============ CONFIGURATION ============
        # Change these to see different quantization schemes!

        self.scale_type = "per_block"  # Options: "per_tensor", "per_row", "per_block"
        self.scale_dtype = "e8m0"  # Options: "e8m0", "e4m3", "fp32"
        self.target_dtype = "fp8"  # Options: "fp8", "fp4"
        self.block_size = 2  # Only used if scale_type="per_block" (must be power of 2)

        # =======================================

        # Dimensions
        self.num_rows = 4
        self.num_cols = 8  # Keep small and divisible by block_size

        # Get target dtype config from utils
        target_config = TARGET_DTYPE_CONFIGS[self.target_dtype]
        self.target_max = target_config.max_value
        self.target_max_pow2 = target_config.max_pow2
        self.target_torch_dtype = target_config.torch_dtype
        self.target_name = target_config.name
        self.target_bits = target_config.bits

        # Color based on target dtype
        self.target_color = (
            COLORS["fp8_color"] if self.target_dtype == "fp8" else COLORS["nvfp_color"]
        )

        # Get scale dtype config
        self.scale_config = SCALE_DTYPE_CONFIGS[self.scale_dtype]

        # Matrix positions (side-by-side layout)
        self.input_x = 0.0  # Left side
        self.output_x = 4.5  # Right side
        self.matrix_y = 1.4  # Slightly higher to make room for tables
        self.table_y = -0.8  # Vertical position for comparison tables at bottom

        # Matrix scales (larger for smaller matrices)
        self.matrix_scale = 1.0
        self.output_matrix_scale = 1.05

        # Font sizes (larger for readability)
        self.matrix_font_size = 22
        self.scale_font_size = 20
        self.subtitle_font_size = 22
        self.calc_font_size = 18

        # Spacing (more generous)
        self.matrix_h_buff = 0.8
        self.matrix_v_buff = 0.5
        self.subtitle_buff = 0.2

    def _update_step_subtitle(self, text: str, color=None):
        """Update the title subtitle with morphing animation."""
        if color is None:
            color = COLORS["computed_scale"]

        # Get the main title position (assuming it's already on screen)
        # Position new subtitle in the same place as the original
        new_subtitle = Text(
            text,
            font_size=20,
            color=color,
            weight=BOLD,
        )

        # Position it where the current subtitle is
        if self.step_subtitle is not None:
            new_subtitle.move_to(self.step_subtitle.get_center())
            # Morph existing subtitle to new text
            self.play(Transform(self.step_subtitle, new_subtitle), run_time=0.5)

        self.wait(0.3)

    def construct(self):
        """Show unified FP quantization process."""
        # Setup quantization data
        self._setup_quantization_data()

        # Title
        title = self._create_title()
        self.play(Write(title))
        self.next_slide()

        # Show input matrix
        input_matrix, input_entries, input_subtitle = self._create_input_matrix()
        self.input_subtitle = input_subtitle  # Store for later fadeout
        self.next_slide()

        # Step 1: Calculate amax within windows
        self._show_amax_calculation(input_entries)
        self.next_slide()

        # Step 3: Calculate scale from amax
        self._show_scale_calculation()
        self.next_slide()

        # Step 4: Show quantized output (morph in place)
        self._show_quantization_process(input_entries)
        self.next_slide()

        # Final fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.0)

    def _create_title(self):
        """Create title with configuration."""
        main_title = Text(
            "Floating Point Quantization",
            font_size=32,
            color=COLORS["text"],
            weight=BOLD,
        )
        main_title.to_edge(UP, buff=0.2)

        # Create subtitle separately so we can morph it later
        self.step_subtitle = Text(
            "A Walkthrough of Quantization",
            font_size=20,
            color=COLORS["computed_scale"],
            weight=BOLD,
        )
        self.step_subtitle.next_to(main_title, DOWN, buff=0.2)

        return VGroup(main_title, self.step_subtitle)

    def _setup_quantization_data(self):
        """Create tensors and run quantization based on configuration."""
        torch.manual_seed(42)
        self.tensor_bf16 = torch.abs(
            torch.randn(self.num_rows, self.num_cols, dtype=torch.bfloat16)
        )

        # Calculate amax using utility function
        self.amax, self.num_scales = calculate_amax(
            self.tensor_bf16,
            self.scale_type,
            self.block_size,
        )

        # Calculate scale using utility function
        self.scales = calculate_scale(
            self.amax,
            self.scale_dtype,
            self.target_max,
            self.target_max_pow2,
        )

        # Quantize data using utility function
        self.quantized_data = quantize_data(
            self.tensor_bf16,
            self.scales,
            self.scale_type,
            self.target_max,
            self.block_size,
        )

    def _create_input_matrix(self):
        """Create BF16 input matrix."""
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
        matrix.move_to([self.input_x, self.matrix_y, 0])

        subtitle = Text(
            "BFloat16 Tensor",
            font_size=self.subtitle_font_size,
            color=COLORS["bf16_color"],
            weight=BOLD,
        )
        subtitle.next_to(matrix, DOWN, buff=self.subtitle_buff)

        self.play(Write(subtitle))
        self.play(Create(matrix), run_time=1.5)

        entries = matrix.get_entries()
        return matrix, entries, subtitle

    def _show_amax_calculation(self, input_entries):
        """Show amax calculation for all three granularity types with a comparison table."""
        # Fade out the input matrix subtitle
        if hasattr(self, "input_subtitle"):
            self.play(FadeOut(self.input_subtitle), run_time=0.3)

        # Create amax table at the bottom, laid out horizontally
        # Table header - start with the concept
        header = Text(
            "amax = max(|values in window|)",
            font_size=18,
            color=COLORS["text"],
            weight=BOLD,
        )
        header.move_to([0, self.table_y + 0.5, 0])
        self.play(Write(header))

        # Prepare amax data for all three types
        # Per tensor
        amax_per_tensor = torch.max(torch.abs(self.tensor_bf16))

        # Per row
        amax_per_row, _ = calculate_amax(self.tensor_bf16, "per_row", self.block_size)

        # Per block
        amax_per_block, _ = calculate_amax(
            self.tensor_bf16, "per_block", self.block_size
        )

        # Create placeholders for each granularity type, laid out horizontally
        col_spacing = 4.0

        # 1. Per Tensor (scalar) - left
        per_tensor_label = Text("Per Tensor", font_size=14, color=WHITE, weight=BOLD)
        per_tensor_bg = Rectangle(
            width=per_tensor_label.width + 0.3,
            height=per_tensor_label.height + 0.2,
            fill_color=COLORS["per_tensor_color"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        per_tensor_bg.move_to([-col_spacing, self.table_y, 0])
        per_tensor_label.move_to(per_tensor_bg.get_center())
        per_tensor_label_group = VGroup(per_tensor_bg, per_tensor_label)

        # Create 1x1 matrix for per_tensor amax value
        amax_tensor_data = (
            amax_per_tensor.unsqueeze(0).unsqueeze(0).to(torch.float32).cpu().numpy()
        )
        per_tensor_matrix = DecimalMatrix(
            amax_tensor_data,
            element_to_mobject_config={
                "num_decimal_places": 1,
                "font_size": self.matrix_font_size,
            },
            left_bracket="[",
            right_bracket="]",
            bracket_config={"color": COLORS["computed_scale"]},
            h_buff=self.matrix_h_buff,
            v_buff=self.matrix_v_buff,
        )
        per_tensor_matrix.scale(self.matrix_scale)
        per_tensor_matrix.next_to(per_tensor_label_group, DOWN, buff=0.2)
        for entry in per_tensor_matrix.get_entries():
            entry.set_color(COLORS["text"])
        per_tensor_matrix.set_opacity(
            0
        )  # Start hidden (entire matrix including brackets)
        per_tensor_group = VGroup(per_tensor_label_group, per_tensor_matrix)

        # 2. Per Row (vector [m]) - center
        per_row_label = Text("Per Row", font_size=14, color=WHITE, weight=BOLD)
        per_row_bg = Rectangle(
            width=per_row_label.width + 0.3,
            height=per_row_label.height + 0.2,
            fill_color=COLORS["per_row_color"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        per_row_bg.move_to([0, self.table_y, 0])
        per_row_label.move_to(per_row_bg.get_center())
        per_row_label_group = VGroup(per_row_bg, per_row_label)

        # Create small matrix for per_row amax values
        amax_row_data = amax_per_row.to(torch.float32).cpu().numpy()
        per_row_matrix = DecimalMatrix(
            amax_row_data,
            element_to_mobject_config={
                "num_decimal_places": 1,
                "font_size": self.matrix_font_size,
            },
            left_bracket="[",
            right_bracket="]",
            bracket_config={"color": COLORS["computed_scale"]},
            h_buff=self.matrix_h_buff,
            v_buff=self.matrix_v_buff,
        )
        per_row_matrix.scale(self.matrix_scale)
        per_row_matrix.next_to(per_row_label_group, DOWN, buff=0.2)
        for entry in per_row_matrix.get_entries():
            entry.set_color(COLORS["text"])
        per_row_matrix.set_opacity(0)  # Start hidden (entire matrix including brackets)
        per_row_group = VGroup(per_row_label_group, per_row_matrix)

        # 3. Per Block (matrix [m, n/blocksize]) - right
        per_block_label = Text("Per Block", font_size=14, color=WHITE, weight=BOLD)
        per_block_bg = Rectangle(
            width=per_block_label.width + 0.3,
            height=per_block_label.height + 0.2,
            fill_color=COLORS["per_block_color"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        per_block_bg.move_to([col_spacing, self.table_y, 0])
        per_block_label.move_to(per_block_bg.get_center())
        per_block_label_group = VGroup(per_block_bg, per_block_label)

        # Create small matrix for per_block amax values
        amax_block_data = amax_per_block.to(torch.float32).cpu().numpy()
        per_block_matrix = DecimalMatrix(
            amax_block_data,
            element_to_mobject_config={
                "num_decimal_places": 1,
                "font_size": self.matrix_font_size,
            },
            left_bracket="[",
            right_bracket="]",
            bracket_config={"color": COLORS["computed_scale"]},
            h_buff=self.matrix_h_buff,
            v_buff=self.matrix_v_buff,
        )
        per_block_matrix.scale(self.matrix_scale)
        per_block_matrix.next_to(per_block_label_group, DOWN, buff=0.2)
        for entry in per_block_matrix.get_entries():
            entry.set_color(COLORS["text"])
        per_block_matrix.set_opacity(
            0
        )  # Start hidden (entire matrix including brackets)
        per_block_group = VGroup(per_block_label_group, per_block_matrix)

        # Show the table structure
        self.play(
            Write(per_tensor_group),
            Write(per_row_group),
            Write(per_block_group),
            run_time=1.0,
        )
        self.wait(0.5)

        # Now cycle through each option, showing correct number of windows

        # 1. Per Tensor - 1 window (entire matrix)
        all_entries = VGroup(*input_entries)
        data_highlight = SurroundingRectangle(
            all_entries, color=COLORS["per_tensor_color"], buff=0.1, stroke_width=4
        )
        amax_highlight = SurroundingRectangle(
            per_tensor_matrix,
            color=COLORS["per_tensor_color"],
            buff=0.15,
            stroke_width=3,
        )
        self.play(Create(data_highlight), Create(amax_highlight), run_time=0.8)

        # Find and highlight the max absolute value element
        flat_tensor = self.tensor_bf16.flatten()
        max_idx = torch.argmax(torch.abs(flat_tensor)).item()
        max_entry = input_entries[max_idx]

        max_entry_highlight = SurroundingRectangle(
            max_entry, color=COLORS["max_value"], buff=0.05, stroke_width=5
        )

        # Show text above the matrix
        max_text = Text(
            "max(|element|) in the window",
            font_size=20,
            color=COLORS["max_value"],
            weight=BOLD,
        )
        max_text.move_to([0, 2.8, 0])  # Above the matrix

        self.play(Create(max_entry_highlight), Write(max_text), run_time=0.8)
        self.wait(0.5)

        # Show the amax value appearing
        self.play(
            per_tensor_matrix.animate.set_opacity(1.0),
            FadeOut(max_text),
            FadeOut(max_entry_highlight),
            run_time=0.8,
        )
        self.wait(0.5)
        self.play(FadeOut(data_highlight), FadeOut(amax_highlight), run_time=0.3)
        self.next_slide()

        # 2. Per Row - num_rows windows
        # First reveal the per_row matrix structure
        self.play(per_row_matrix.animate.set_opacity(1.0), run_time=0.5)

        # Get per_row matrix entries for highlighting
        per_row_entries = per_row_matrix.get_entries()

        # Cycle through each row
        for row in range(self.num_rows):
            row_entries_data = VGroup(
                *[input_entries[row * self.num_cols + i] for i in range(self.num_cols)]
            )
            data_highlight = SurroundingRectangle(
                row_entries_data,
                color=COLORS["per_row_color"],
                buff=0.1,
                stroke_width=4,
            )
            amax_highlight = SurroundingRectangle(
                per_row_entries[row],
                color=COLORS["per_row_color"],
                buff=0.1,
                stroke_width=3,
            )
            self.play(Create(data_highlight), Create(amax_highlight), run_time=0.4)

            # Find and highlight the max absolute value in this row
            row_data = self.tensor_bf16[row]
            max_idx_in_row = torch.argmax(torch.abs(row_data)).item()
            max_entry_idx = row * self.num_cols + max_idx_in_row
            max_entry = input_entries[max_entry_idx]

            max_entry_highlight = SurroundingRectangle(
                max_entry, color=COLORS["max_value"], buff=0.05, stroke_width=5
            )
            self.play(Create(max_entry_highlight), run_time=0.3)
            self.wait(0.2)
            self.play(FadeOut(max_entry_highlight), run_time=0.2)

            self.play(FadeOut(data_highlight), FadeOut(amax_highlight), run_time=0.2)

        self.wait(0.5)
        self.next_slide()

        # 3. Per Block - (num_rows * num_cols / block_size) windows
        # First reveal the per_block matrix structure
        self.play(per_block_matrix.animate.set_opacity(1.0), run_time=0.5)

        # Get per_block matrix entries for highlighting
        per_block_entries = per_block_matrix.get_entries()
        num_blocks_per_row = self.num_cols // self.block_size

        # Show first 2 rows individually to demonstrate, then rest all at once
        rows_to_show_individually = 2

        # Cycle through first 2 rows individually
        for row in range(min(rows_to_show_individually, self.num_rows)):
            for block_idx in range(num_blocks_per_row):
                start_col = block_idx * self.block_size
                end_col = start_col + self.block_size

                block_entries_data = VGroup(
                    *[
                        input_entries[row * self.num_cols + i]
                        for i in range(start_col, end_col)
                    ]
                )
                data_highlight = SurroundingRectangle(
                    block_entries_data,
                    color=COLORS["per_block_color"],
                    buff=0.1,
                    stroke_width=4,
                )

                # Calculate which entry in the per_block matrix corresponds to this block
                block_matrix_idx = row * num_blocks_per_row + block_idx
                amax_highlight = SurroundingRectangle(
                    per_block_entries[block_matrix_idx],
                    color=COLORS["per_block_color"],
                    buff=0.1,
                    stroke_width=3,
                )

                self.play(Create(data_highlight), Create(amax_highlight), run_time=0.3)

                # Find and highlight the max absolute value in this block
                block_data = self.tensor_bf16[row, start_col:end_col]
                max_idx_in_block = torch.argmax(torch.abs(block_data)).item()
                max_entry_idx = row * self.num_cols + start_col + max_idx_in_block
                max_entry = input_entries[max_entry_idx]

                max_entry_highlight = SurroundingRectangle(
                    max_entry, color=COLORS["max_value"], buff=0.05, stroke_width=5
                )
                self.play(Create(max_entry_highlight), run_time=0.2)
                self.wait(0.1)
                self.play(FadeOut(max_entry_highlight), run_time=0.15)

                self.play(
                    FadeOut(data_highlight), FadeOut(amax_highlight), run_time=0.15
                )

        # Show remaining rows all at once if there are more rows
        if self.num_rows > rows_to_show_individually:
            all_remaining_data_highlights = VGroup()
            all_remaining_amax_highlights = VGroup()
            all_remaining_max_highlights = VGroup()

            for row in range(rows_to_show_individually, self.num_rows):
                for block_idx in range(num_blocks_per_row):
                    start_col = block_idx * self.block_size
                    end_col = start_col + self.block_size

                    block_entries_data = VGroup(
                        *[
                            input_entries[row * self.num_cols + i]
                            for i in range(start_col, end_col)
                        ]
                    )
                    data_highlight = SurroundingRectangle(
                        block_entries_data,
                        color=COLORS["per_block_color"],
                        buff=0.1,
                        stroke_width=4,
                    )
                    all_remaining_data_highlights.add(data_highlight)

                    # Calculate which entry in the per_block matrix corresponds to this block
                    block_matrix_idx = row * num_blocks_per_row + block_idx
                    amax_highlight = SurroundingRectangle(
                        per_block_entries[block_matrix_idx],
                        color=COLORS["per_block_color"],
                        buff=0.1,
                        stroke_width=3,
                    )
                    all_remaining_amax_highlights.add(amax_highlight)

                    # Find and highlight the max absolute value in this block
                    block_data = self.tensor_bf16[row, start_col:end_col]
                    max_idx_in_block = torch.argmax(torch.abs(block_data)).item()
                    max_entry_idx = row * self.num_cols + start_col + max_idx_in_block
                    max_entry = input_entries[max_entry_idx]

                    max_entry_highlight = SurroundingRectangle(
                        max_entry, color=COLORS["max_value"], buff=0.05, stroke_width=5
                    )
                    all_remaining_max_highlights.add(max_entry_highlight)

            # Show all remaining blocks at once
            self.play(
                Create(all_remaining_data_highlights),
                Create(all_remaining_amax_highlights),
                run_time=0.5,
            )
            self.play(Create(all_remaining_max_highlights), run_time=0.3)
            self.wait(0.3)
            self.play(
                FadeOut(all_remaining_max_highlights),
                FadeOut(all_remaining_data_highlights),
                FadeOut(all_remaining_amax_highlights),
                run_time=0.3,
            )

        # Store these for next step
        self.amax_per_tensor = amax_per_tensor
        self.amax_per_row = amax_per_row
        self.amax_per_block = amax_per_block

        # Store the table components to keep them visible and transform later
        self.amax_header = header
        self.amax_tables = VGroup(per_tensor_group, per_row_group, per_block_group)
        # Store individual components for transformation
        self.per_tensor_matrix = per_tensor_matrix
        self.per_row_matrix = per_row_matrix
        self.per_block_matrix = per_block_matrix

    def _show_scale_calculation(self):
        """Show scale calculation by transforming amax matrices into scale matrices."""
        # Different scale methods for different granularities:
        # Per Tensor/Row: FP32 (simple division)
        # Per Block: E8M0 (MX format - power of 2)

        # Transform the header to show the scale calculation method with formulas
        new_header = Text(
            "Calculate scale (inverted for multiplication)",
            font_size=16,
            color=COLORS["text"],
            weight=BOLD,
        )
        new_header.move_to([0, self.table_y + 0.5, 0])
        self.play(Transform(self.amax_header, new_header), run_time=0.8)

        # Show formulas for each method
        formula_per_tensor = VGroup(
            Text("FP32:", font_size=12, color=COLORS["per_tensor_color"], weight=BOLD),
            MathTex(
                r"\frac{\text{FP8\_MAX}}{\text{amax}}",
                font_size=12,
                color=COLORS["text"],
            ),
        ).arrange(RIGHT, buff=0.2)
        formula_per_tensor.move_to([-4.0, self.table_y + 0.9, 0])

        formula_per_row = VGroup(
            Text("FP32:", font_size=12, color=COLORS["per_row_color"], weight=BOLD),
            MathTex(
                r"\frac{\text{FP8\_MAX}}{\text{amax}}",
                font_size=12,
                color=COLORS["text"],
            ),
        ).arrange(RIGHT, buff=0.2)
        formula_per_row.move_to([0, self.table_y + 0.9, 0])

        formula_per_block = VGroup(
            Text("E8M0:", font_size=12, color=COLORS["per_block_color"], weight=BOLD),
            MathTex(
                r"\frac{\text{FP8\_MAX\_POW2}}{\text{pow2(amax)}}",
                font_size=11,
                color=COLORS["text"],
            ),
        ).arrange(RIGHT, buff=0.2)
        formula_per_block.move_to([4.0, self.table_y + 0.9, 0])

        self.play(
            Write(formula_per_tensor),
            Write(formula_per_row),
            Write(formula_per_block),
            run_time=1.0,
        )
        self.wait(0.5)

        # Store formulas for later fadeout
        self.scale_formulas = VGroup(
            formula_per_tensor, formula_per_row, formula_per_block
        )

        # Calculate scales for all three types with different methods
        # Per tensor: FP32 (simple division)
        scales_per_tensor = calculate_scale(
            self.amax_per_tensor.unsqueeze(0).unsqueeze(0),
            "fp32",
            self.target_max,
            self.target_max_pow2,
        )

        # Per row: FP32 (simple division)
        scales_per_row = calculate_scale(
            self.amax_per_row, "fp32", self.target_max, self.target_max_pow2
        )

        # Per block: E8M0 (MX format - power of 2)
        scales_per_block = calculate_scale(
            self.amax_per_block, "e8m0", self.target_max, self.target_max_pow2
        )

        # Create new scale value displays to morph into
        # 1. Per Tensor - create new 1x1 matrix (FP32 method)
        scale_tensor_data = scales_per_tensor.cpu().numpy()
        new_per_tensor_matrix = DecimalMatrix(
            scale_tensor_data,
            element_to_mobject_config={
                "num_decimal_places": 3,
                "font_size": self.matrix_font_size,
            },
            left_bracket="[",
            right_bracket="]",
            bracket_config={"color": COLORS["computed_scale"]},
            h_buff=self.matrix_h_buff,
            v_buff=self.matrix_v_buff,
        )
        new_per_tensor_matrix.scale(self.matrix_scale)
        new_per_tensor_matrix.move_to(self.per_tensor_matrix.get_center())
        for entry in new_per_tensor_matrix.get_entries():
            entry.set_color(COLORS["text"])

        # 2. Per Row - create new matrix (FP32 method)
        scale_row_data = scales_per_row.cpu().numpy()
        new_per_row_matrix = DecimalMatrix(
            scale_row_data,
            element_to_mobject_config={
                "num_decimal_places": 3,
                "font_size": self.matrix_font_size,
            },
            left_bracket="[",
            right_bracket="]",
            bracket_config={"color": COLORS["computed_scale"]},
            h_buff=self.matrix_h_buff,
            v_buff=self.matrix_v_buff,
        )
        new_per_row_matrix.scale(self.matrix_scale)
        new_per_row_matrix.move_to(self.per_row_matrix.get_center())
        for entry in new_per_row_matrix.get_entries():
            entry.set_color(COLORS["text"])

        # 3. Per Block - create new matrix (E8M0 method - MX format)
        scale_block_data = scales_per_block.cpu().numpy()
        new_per_block_matrix = DecimalMatrix(
            scale_block_data,
            element_to_mobject_config={
                "num_decimal_places": 3,
                "font_size": self.matrix_font_size,
            },
            left_bracket="[",
            right_bracket="]",
            bracket_config={"color": COLORS["computed_scale"]},
            h_buff=self.matrix_h_buff + 0.4,
            v_buff=self.matrix_v_buff,
        )
        new_per_block_matrix.scale(self.matrix_scale)
        new_per_block_matrix.move_to(self.per_block_matrix.get_center())
        for entry in new_per_block_matrix.get_entries():
            entry.set_color(COLORS["text"])

        # Transform all three simultaneously
        self.play(
            Transform(self.per_tensor_matrix, new_per_tensor_matrix),
            Transform(self.per_row_matrix, new_per_row_matrix),
            Transform(self.per_block_matrix, new_per_block_matrix),
            run_time=1.5,
        )
        self.wait(1.0)

        # Fade out the header and formulas
        self.play(FadeOut(self.amax_header), FadeOut(self.scale_formulas), run_time=0.5)

    def _show_quantization_process(self, input_entries):
        """Show the quantization process for all three granularity types."""
        # Show quantization formula in the middle
        quant_formula = Text(
            "Scale and quantize: torch.clamp(inputTensor * scale).to(torch.float8_e4m3)",
            font_size=16,
            color=self.target_color,
            weight=BOLD,
        )
        quant_formula.move_to([0, 0.0, 0])  # Middle position between matrix and table
        self.play(Write(quant_formula), run_time=0.8)
        self.wait(0.5)

        # Store for later fadeout
        self.quant_formula = quant_formula

        # Prepare quantized data for all three types
        # Per tensor
        scales_per_tensor = calculate_scale(
            self.amax_per_tensor.unsqueeze(0).unsqueeze(0),
            "fp32",
            self.target_max,
            self.target_max_pow2,
        )
        quantized_per_tensor = quantize_data(
            self.tensor_bf16,
            scales_per_tensor,
            "per_tensor",
            self.target_max,
            self.block_size,
        )

        # Per row
        scales_per_row = calculate_scale(
            self.amax_per_row, "fp32", self.target_max, self.target_max_pow2
        )
        quantized_per_row = quantize_data(
            self.tensor_bf16,
            scales_per_row,
            "per_row",
            self.target_max,
            self.block_size,
        )

        # Per block
        scales_per_block = calculate_scale(
            self.amax_per_block, "e8m0", self.target_max, self.target_max_pow2
        )
        quantized_per_block = quantize_data(
            self.tensor_bf16,
            scales_per_block,
            "per_block",
            self.target_max,
            self.block_size,
        )

        # Loop through each granularity type
        self._quantize_with_granularity(
            "per_tensor",
            quantized_per_tensor,
            input_entries,
            highlight_color=COLORS["per_tensor_color"],
        )
        self.next_slide()

        self._quantize_with_granularity(
            "per_row",
            quantized_per_row,
            input_entries,
            highlight_color=COLORS["per_row_color"],
        )
        self.next_slide()

        self._quantize_with_granularity(
            "per_block",
            quantized_per_block,
            input_entries,
            highlight_color=COLORS["per_block_color"],
        )
        self.wait(1.0)

        # Fade out the quantization formula
        self.play(FadeOut(self.quant_formula), run_time=0.5)

    def _quantize_with_granularity(
        self, granularity_type, quantized_data, input_entries, highlight_color
    ):
        """Apply quantization for one granularity type with highlighting."""
        if granularity_type == "per_tensor":
            # Get per_tensor matrix entry for highlighting
            per_tensor_scale_entry = self.per_tensor_matrix.get_entries()[0]

            # Highlight the single scale value in the table
            scale_highlight = SurroundingRectangle(
                per_tensor_scale_entry, color=highlight_color, buff=0.1, stroke_width=3
            )

            # Highlight entire matrix
            all_entries = VGroup(*input_entries)
            data_highlight = SurroundingRectangle(
                all_entries, color=highlight_color, buff=0.1, stroke_width=4
            )
            self.play(Create(scale_highlight), Create(data_highlight), run_time=0.5)

            # Create a copy of the scale value to animate up
            scale_copy = per_tensor_scale_entry.copy()
            scale_copy.set_color(highlight_color)
            scale_copy.scale(1.5)  # Make it slightly larger for visibility

            # Animate scale moving up to the tensor
            target_pos = [0, self.matrix_y, 0]
            self.play(scale_copy.animate.move_to(target_pos), run_time=0.8)

            # Create spread copies before fading out original
            spread_copies = VGroup()
            for idx, entry in enumerate(input_entries):
                spread_copy_item = scale_copy.copy()
                spread_copy_item.scale(0.5)  # Make smaller
                spread_copies.add(spread_copy_item)

            # Fade out the original scale before spreading
            self.play(FadeOut(scale_copy), run_time=0.3)

            # Animate spreading to all elements
            spread_animations = []
            for idx, (spread_copy, entry) in enumerate(
                zip(spread_copies, input_entries)
            ):
                spread_animations.append(
                    spread_copy.animate.move_to(entry.get_center())
                )

            self.play(*spread_animations, run_time=0.6)
            self.wait(0.2)

            # Morph all values to quantized
            new_matrix_data = quantized_data.cpu().numpy()
            animations = []
            for idx, entry in enumerate(input_entries):
                row = idx // self.num_cols
                col = idx % self.num_cols
                new_val = new_matrix_data[row, col]
                new_text = DecimalNumber(
                    new_val,
                    num_decimal_places=1,
                    font_size=self.matrix_font_size,
                    color=COLORS["text"],
                )
                new_text.move_to(entry.get_center())
                animations.append(Transform(entry, new_text))

            animations.append(FadeOut(spread_copies))
            self.play(*animations, run_time=1.5)
            self.play(FadeOut(scale_highlight), FadeOut(data_highlight), run_time=0.3)

        elif granularity_type == "per_row":
            # Get per_row matrix entries for highlighting individual scales
            per_row_scale_entries = self.per_row_matrix.get_entries()

            # Create highlights for all rows and their corresponding scales at once
            row_highlights = VGroup()
            scale_highlights = VGroup()
            for row in range(self.num_rows):
                # Highlight row in data
                row_entries = VGroup(
                    *[
                        input_entries[row * self.num_cols + i]
                        for i in range(self.num_cols)
                    ]
                )
                data_highlight = SurroundingRectangle(
                    row_entries, color=highlight_color, buff=0.1, stroke_width=4
                )
                row_highlights.add(data_highlight)

                # Highlight corresponding scale in table
                scale_highlight = SurroundingRectangle(
                    per_row_scale_entries[row],
                    color=highlight_color,
                    buff=0.1,
                    stroke_width=3,
                )
                scale_highlights.add(scale_highlight)

            # Show all highlights at once
            self.play(Create(row_highlights), Create(scale_highlights), run_time=0.8)

            # Create copies of scale values to animate up to their respective rows
            scale_copies = VGroup()
            for row in range(self.num_rows):
                scale_copy = per_row_scale_entries[row].copy()
                scale_copy.set_color(highlight_color)
                scale_copy.scale(1.3)
                scale_copies.add(scale_copy)

            # Animate all scales moving up to their rows
            animations = []
            for row in range(self.num_rows):
                row_entries = VGroup(
                    *[
                        input_entries[row * self.num_cols + i]
                        for i in range(self.num_cols)
                    ]
                )
                target_pos = [
                    row_entries.get_center()[0] - 1.5,
                    row_entries.get_center()[1],
                    0,
                ]
                animations.append(scale_copies[row].animate.move_to(target_pos))

            self.play(*animations, run_time=0.8)

            # Create spread copies before fading out originals
            all_spread_copies = VGroup()
            for row in range(self.num_rows):
                row_entries = VGroup(
                    *[
                        input_entries[row * self.num_cols + i]
                        for i in range(self.num_cols)
                    ]
                )

                for entry in row_entries:
                    spread_copy = scale_copies[row].copy()
                    spread_copy.scale(0.4)  # Make smaller
                    all_spread_copies.add(spread_copy)

            # Fade out the original scales before spreading
            self.play(FadeOut(scale_copies), run_time=0.3)

            # Animate spreading to all elements
            spread_animations = []
            idx = 0
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    entry = input_entries[row * self.num_cols + col]
                    spread_animations.append(
                        all_spread_copies[idx].animate.move_to(entry.get_center())
                    )
                    idx += 1

            self.play(*spread_animations, run_time=0.5)
            self.wait(0.2)

            # Morph all values simultaneously
            new_matrix_data = quantized_data.cpu().numpy()
            morph_animations = []
            for idx, entry in enumerate(input_entries):
                row = idx // self.num_cols
                col = idx % self.num_cols
                new_val = new_matrix_data[row, col]
                new_text = DecimalNumber(
                    new_val,
                    num_decimal_places=1,
                    font_size=self.matrix_font_size,
                    color=COLORS["text"],
                )
                new_text.move_to(entry.get_center())
                morph_animations.append(Transform(entry, new_text))

            morph_animations.append(FadeOut(all_spread_copies))
            self.play(*morph_animations, run_time=1.5)
            self.play(FadeOut(scale_highlights), FadeOut(row_highlights), run_time=0.3)

        else:  # per_block
            # Get per_block matrix entries for highlighting individual scales
            per_block_scale_entries = self.per_block_matrix.get_entries()

            # Create highlights for all blocks and their corresponding scales at once
            block_highlights = VGroup()
            scale_highlights = VGroup()
            num_blocks_per_row = self.num_cols // self.block_size
            for row in range(self.num_rows):
                for block_idx in range(num_blocks_per_row):
                    start_col = block_idx * self.block_size
                    end_col = start_col + self.block_size

                    # Highlight block in data
                    block_entries = VGroup(
                        *[
                            input_entries[row * self.num_cols + i]
                            for i in range(start_col, end_col)
                        ]
                    )
                    data_highlight = SurroundingRectangle(
                        block_entries, color=highlight_color, buff=0.1, stroke_width=4
                    )
                    block_highlights.add(data_highlight)

                    # Highlight corresponding scale in table
                    block_matrix_idx = row * num_blocks_per_row + block_idx
                    scale_highlight = SurroundingRectangle(
                        per_block_scale_entries[block_matrix_idx],
                        color=highlight_color,
                        buff=0.1,
                        stroke_width=3,
                    )
                    scale_highlights.add(scale_highlight)

            # Show all highlights at once
            self.play(Create(block_highlights), Create(scale_highlights), run_time=0.8)

            # Create copies of scale values to animate up to their respective blocks
            scale_copies = VGroup()
            target_positions = []

            for row in range(self.num_rows):
                for block_idx in range(num_blocks_per_row):
                    start_col = block_idx * self.block_size
                    block_matrix_idx = row * num_blocks_per_row + block_idx

                    scale_copy = per_block_scale_entries[block_matrix_idx].copy()
                    scale_copy.set_color(highlight_color)
                    scale_copy.scale(1.2)
                    scale_copies.add(scale_copy)

                    # Get target position (center of block)
                    block_entries = VGroup(
                        *[
                            input_entries[row * self.num_cols + i]
                            for i in range(start_col, start_col + self.block_size)
                        ]
                    )
                    target_positions.append(block_entries.get_center())

            # Animate all scales moving up to their blocks
            animations = []
            for i, (scale_copy, target_pos) in enumerate(
                zip(scale_copies, target_positions)
            ):
                animations.append(scale_copy.animate.move_to(target_pos))

            self.play(*animations, run_time=0.8)

            # Create spread copies before fading out originals
            all_spread_copies = VGroup()
            scale_idx = 0
            for row in range(self.num_rows):
                for block_idx in range(num_blocks_per_row):
                    start_col = block_idx * self.block_size

                    for i in range(start_col, start_col + self.block_size):
                        entry = input_entries[row * self.num_cols + i]
                        spread_copy = scale_copies[scale_idx].copy()
                        spread_copy.scale(0.35)  # Make smaller
                        all_spread_copies.add(spread_copy)
                    scale_idx += 1

            # Fade out the original scales before spreading
            self.play(FadeOut(scale_copies), run_time=0.3)

            # Animate spreading to all elements
            spread_animations = []
            spread_idx = 0
            for row in range(self.num_rows):
                for block_idx in range(num_blocks_per_row):
                    start_col = block_idx * self.block_size

                    for i in range(start_col, start_col + self.block_size):
                        entry = input_entries[row * self.num_cols + i]
                        spread_animations.append(
                            all_spread_copies[spread_idx].animate.move_to(
                                entry.get_center()
                            )
                        )
                        spread_idx += 1

            self.play(*spread_animations, run_time=0.5)
            self.wait(0.2)

            # Morph all values simultaneously
            new_matrix_data = quantized_data.cpu().numpy()
            morph_animations = []
            for idx, entry in enumerate(input_entries):
                row = idx // self.num_cols
                col = idx % self.num_cols
                new_val = new_matrix_data[row, col]
                new_text = DecimalNumber(
                    new_val,
                    num_decimal_places=1,
                    font_size=self.matrix_font_size,
                    color=COLORS["text"],
                )
                new_text.move_to(entry.get_center())
                morph_animations.append(Transform(entry, new_text))

            morph_animations.append(FadeOut(all_spread_copies))
            self.play(*morph_animations, run_time=1.5)
            self.play(
                FadeOut(scale_highlights), FadeOut(block_highlights), run_time=0.3
            )
