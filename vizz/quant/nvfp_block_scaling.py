"""Deep dive into NVFP4 format scaling with 16-element blocks using torchao.

Shows BF16 → FP4 quantization with E4M3 scales, demonstrating how a 1×16
scanning window moves across the reduction dimension to quantize blocks.

To run:
manim-slides render vizz/quant/nvfp_block_scaling.py NVFPBlockScaling -ql
manim-slides present NVFPBlockScaling
"""

import torch
from manim import *
from manim_slides import Slide

from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

from vizz.quant.mx_base import COLORS


class NVFPBlockScaling(Slide):
    """Deep dive into NVFP4 format scaling with 16-element blocks using torchao."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Dimensions
        self.num_rows = 4
        self.num_cols = 64
        self.block_size = 16
        self.num_blocks = self.num_cols // self.block_size

        # Matrix positions (y-coordinates)
        self.bf16_y = 2.0
        self.nvfp4_y = self.bf16_y - 2.0
        self.scale_y = self.nvfp4_y - 2.0

        # Matrix scales
        self.matrix_scale = 0.5  # For BF16 and NVFP4
        self.scale_matrix_scale = 0.5  # For E4M3 scales

        # Font sizes
        self.matrix_font_size = 14
        self.scale_font_size = 24
        self.subtitle_font_size = 18

        # Spacing
        self.matrix_h_buff = 0.35
        self.matrix_v_buff = 0.4
        self.scale_h_buff = 1
        self.scale_v_buff = 1
        self.subtitle_buff = 0.3

    def construct(self):
        """Show BF16 → NVFP4 quantization with E4M3 scales."""
        # Setup quantization data
        self._setup_quantization_data()

        # Setup
        title = Text(
            "NVFP4 Format: BF16 → FP4 + E4M3 Scales",
            font_size=32,
            color=COLORS["text"],
            weight=BOLD,
        )
        title.to_edge(UP, buff=0.2)
        self.play(Write(title))
        self.next_slide()

        # Show per-tensor scale calculation
        per_tensor_scale_display = self._show_per_tensor_scale()
        self.next_slide()

        # Create full matrix visualizations
        bf16_matrix, bf16_entries, bf16_subtitle = self._create_input_matrix()

        # Transform to compact box and move to lower right
        self.per_tensor_scale_box.to_corner(DR, buff=0.5)
        self.play(
            Transform(per_tensor_scale_display, self.per_tensor_scale_box), run_time=1
        )
        # Store the transformed box for later use
        self.per_tensor_scale_display = per_tensor_scale_display
        self.next_slide()

        nvfp4_matrix, nvfp4_entries, nvfp4_subtitle = self._create_output_matrix()
        scale_matrix, scale_entries, scale_subtitle = self._create_scale_matrix()
        self.next_slide()

        # Animate scanning and quantization
        self._animate_scanning(
            bf16_entries,
            nvfp4_entries,
            scale_entries,
            nvfp4_matrix,
            scale_matrix,
            nvfp4_subtitle,
            scale_subtitle,
        )

        # Show summary
        self._show_summary()
        self.next_slide()

    def _setup_quantization_data(self):
        """Create tensors and run torchao NVFP4 quantization."""
        torch.manual_seed(42)
        self.tensor_bf16 = torch.abs(
            torch.randn(self.num_rows, self.num_cols, dtype=torch.bfloat16)
        )

        # Calculate per-tensor scale
        self.tensor_amax = torch.max(torch.abs(self.tensor_bf16))
        # NVFP4 uses: per_tensor_scale = amax / (F8E4M3_MAX * F4_E2M1_MAX)
        # F8E4M3_MAX = 448, F4_E2M1_MAX = 6
        self.F8E4M3_MAX = 448.0
        self.F4_E2M1_MAX = 6.0
        self.per_tensor_scale = self.tensor_amax.to(torch.float32) / (
            self.F8E4M3_MAX * self.F4_E2M1_MAX
        )

        # Quantize to NVFP4 with per-tensor scale
        self.nvfp4_tensor = NVFP4Tensor.to_nvfp4(
            self.tensor_bf16,
            block_size=self.block_size,
            per_tensor_scale=self.per_tensor_scale,
        )

        # Get the dequantized data for visualization
        self.data_nvfp4 = self.nvfp4_tensor.to_dtype(torch.float32)

    def _show_per_tensor_scale(self):
        """Display the global per-tensor scale calculation."""
        # Create explanation
        explanation = VGroup(
            Text(
                "Step 1: Calculate Global Per-Tensor Scale",
                font_size=24,
                color=COLORS["text"],
                weight=BOLD,
            ),
            Text(
                f"amax = max(|tensor|) = {self.tensor_amax.item():.4f}",
                font_size=20,
                color=COLORS["max_value"],
            ),
            Text(
                "per_tensor_scale = amax / (E4M3_MAX × FP4_MAX)",
                font_size=20,
                color=COLORS["computed_scale"],
            ),
            Text(
                f"per_tensor_scale = {self.tensor_amax.item():.4f} / ({self.F8E4M3_MAX:.0f} × {self.F4_E2M1_MAX:.0f})",
                font_size=18,
                color=COLORS["text"],
            ),
            Text(
                f"per_tensor_scale = {self.per_tensor_scale.item():.6f}",
                font_size=20,
                color=COLORS["computed_scale"],
                weight=BOLD,
            ),
        )
        explanation.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        explanation.move_to([0, 0.5, 0])

        self.play(Write(explanation), run_time=2)

        # Create compact version for persistent display
        self.per_tensor_scale_box = VGroup(
            Text("Global Scale:", font_size=16, color=COLORS["text"], weight=BOLD),
            Text(
                f"{self.per_tensor_scale.item():.6f}",
                font_size=16,
                color=COLORS["computed_scale"],
                weight=BOLD,
            ),
        )
        self.per_tensor_scale_box.arrange(DOWN, buff=0.1)

        return explanation

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
        """Create NVFP4 output matrix scaffolding (empty placeholders)."""
        # Show PACKED data - qdata contains packed bytes (2 FP4 values per byte)
        # Shape is [4, 32] for packed representation (64 FP4 values = 32 bytes per row)
        packed_data = self.nvfp4_tensor.qdata.view(torch.uint8).cpu().numpy()

        matrix = IntegerMatrix(
            packed_data,
            element_to_mobject_config={
                "num_decimal_places": 0,
                "font_size": self.matrix_font_size,
            },
            left_bracket="[",
            right_bracket="]",
            bracket_config={"color": COLORS["nvfp_color"]},
            h_buff=self.matrix_h_buff,
            v_buff=self.matrix_v_buff,
        )

        # Set all entries to invisible initially
        for entry in matrix.get_entries():
            entry.set_color(COLORS["text"])
            entry.set_opacity(0.0)

        # Scale to fit columns on screen
        matrix.scale(self.matrix_scale)
        matrix.move_to([0, self.nvfp4_y, 0])

        # Position subtitle above matrix - show packed dimensions
        packed_cols = self.num_cols // 2  # 2 FP4 values per byte
        subtitle = Text(
            f"NVFP4 Packed [{self.num_rows}, {packed_cols}] bytes",
            font_size=self.subtitle_font_size,
            color=COLORS["nvfp_color"],
            weight=BOLD,
        )
        subtitle.next_to(matrix, UP, buff=self.subtitle_buff)

        # Store matrix entries for revealing
        entries = matrix.get_entries()

        return matrix, entries, subtitle

    def _create_scale_matrix(self):
        """Create E4M3 scales scaffolding (empty placeholders)."""
        # Get the blockwise scales in E4M3 format
        scales_e4m3 = self.nvfp4_tensor._scale_e4m3.to(torch.float32).cpu().numpy()

        # Create IntegerMatrix with actual values (will be hidden initially)
        matrix = DecimalMatrix(
            scales_e4m3,
            left_bracket="[",
            right_bracket="]",
            bracket_config={"color": COLORS["computed_scale"]},
            element_to_mobject_config={
                "num_decimal_places": 2,
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
            f"E4M3 Scales [{self.num_rows}, {self.num_blocks}]",
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
        nvfp4_entries,
        scale_entries,
        nvfp4_matrix,
        scale_matrix,
        nvfp4_subtitle,
        scale_subtitle,
    ):
        """Animate scanning window highlighting blocks as they're processed."""
        # Scan across columns, then down rows
        current_highlight = None
        packing_label = None

        for row in range(self.num_rows):
            for block_idx in range(self.num_blocks):
                # Calculate which entries to highlight (16 consecutive elements per block)
                start_idx = block_idx * self.block_size
                end_idx = start_idx + self.block_size

                # Get the entries for this block in the BF16 input matrix
                block_entries_bf16 = VGroup(
                    *[
                        bf16_entries[row * self.num_cols + i]
                        for i in range(start_idx, end_idx)
                    ]
                )

                # Get the entries for this block in the NVFP4 output matrix
                # The packed matrix has half as many columns (2 FP4 values per byte)
                packed_cols = self.num_cols // 2
                packed_start_idx = start_idx // 2
                packed_end_idx = end_idx // 2
                block_entries_nvfp4 = VGroup(
                    *[
                        nvfp4_entries[row * packed_cols + i]
                        for i in range(packed_start_idx, packed_end_idx)
                    ]
                )

                # Get the scale entry
                scale_idx = row * self.num_blocks + block_idx
                scale_entry = scale_entries[scale_idx]

                # Create highlighting rectangles
                # BF16 highlight is NORMAL SIZE (16 elements unpacked)
                bf16_highlight = SurroundingRectangle(
                    block_entries_bf16,
                    color=COLORS["active_block"],
                    buff=0.05,
                    stroke_width=3,
                )

                # NVFP4 highlight is HALF THE WIDTH (8 bytes = 16 FP4 values packed)
                # This is automatic since block_entries_nvfp4 has half as many entries
                nvfp4_highlight = SurroundingRectangle(
                    block_entries_nvfp4,
                    color=COLORS["nvfp_color"],
                    buff=0.05,
                    stroke_width=3,
                )

                scale_highlight = SurroundingRectangle(
                    scale_entry,
                    color=COLORS["computed_scale"],
                    buff=0.1,
                    stroke_width=3,
                )

                # Create per-tensor scale highlight box
                scale_box_highlight = SurroundingRectangle(
                    self.per_tensor_scale_display,
                    color=COLORS["computed_scale"],
                    buff=0.15,
                    stroke_width=4,
                )

                # Animate highlighting
                if current_highlight is None:
                    # First block - show NVFP4 and scale scaffolding
                    self.play(
                        Write(nvfp4_subtitle), Write(scale_subtitle), run_time=0.5
                    )
                    self.play(Create(nvfp4_matrix), Create(scale_matrix), run_time=1.5)

                    # Add packing explanation
                    packing_label = Text(
                        "16 FP4 values pack into 8 bytes",
                        font_size=16,
                        color=COLORS["nvfp_color"],
                    )
                    packing_label.next_to(nvfp4_highlight, DOWN, buff=0.2)

                    self.play(
                        Create(bf16_highlight),
                        Create(nvfp4_highlight),
                        Create(scale_highlight),
                        Create(scale_box_highlight),
                        Write(packing_label),
                        run_time=0.5,
                    )
                    current_highlight = VGroup(
                        bf16_highlight,
                        nvfp4_highlight,
                        scale_highlight,
                        scale_box_highlight,
                    )
                else:
                    # Move highlights to new block
                    # Update packing label position
                    new_packing_label = Text(
                        "16 FP4 values pack into 8 bytes",
                        font_size=16,
                        color=COLORS["nvfp_color"],
                    )
                    new_packing_label.next_to(nvfp4_highlight, DOWN, buff=0.2)

                    self.play(
                        Transform(current_highlight[0], bf16_highlight),
                        Transform(current_highlight[1], nvfp4_highlight),
                        Transform(current_highlight[2], scale_highlight),
                        Transform(current_highlight[3], scale_box_highlight),
                        Transform(packing_label, new_packing_label),
                        run_time=0.4,
                    )

                # Reveal the NVFP4 packed values for this block
                animations = []
                for i in range(packed_start_idx, packed_end_idx):
                    entry_idx = row * packed_cols + i
                    animations.append(nvfp4_entries[entry_idx].animate.set_opacity(1.0))

                # Reveal the scale value
                animations.append(scale_entry.animate.set_opacity(1.0))

                # Animate all the value reveals
                if animations:
                    self.play(*animations, run_time=0.6)

        # Fade out highlights and packing label
        if current_highlight and packing_label:
            self.play(FadeOut(current_highlight), FadeOut(packing_label), run_time=0.3)

    def _show_summary(self):
        """Display summary message."""
        summary = VGroup(
            Text(
                "Each 1×16 block quantized to FP4 with one E4M3 scale",
                font_size=16,
                color=COLORS["text"],
            ),
            Text(
                f"Packing: {self.num_cols} FP4 values = {self.num_cols // 2} bytes per row",
                font_size=14,
                color=COLORS["text"],
            ),
        )
        summary.arrange(DOWN, buff=0.2)
        summary.to_edge(DOWN, buff=0.3)
        self.play(Write(summary))
