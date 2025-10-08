"""NVFP4 format scaling with 16-element blocks and packing.

Shows how NVFP4 format uses smaller blocks (16 elements) and how FP4 values
are packed into bytes.

To run:
manim-slides render vizz/quant/nvfp_block_scaling.py NVFPBlockScaling -ql
manim-slides present NVFPBlockScaling
"""

import torch
from manim import *
from manim_slides import Slide

from vizz.quant.mx_base import COLORS


class NVFPBlockScaling(Slide):
    """NVFP4 format scaling with 16-element blocks and packing."""

    def construct(self):
        """Show NVFP4 format with packing visualization."""
        # Title
        title = Text(
            "NVFP4 Format: 16-Element Blocks + Packing",
            font_size=48,
            color=COLORS["text"],
        )
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Show block structure
        self._show_block_structure()
        self.next_slide()

        # Demonstrate packing
        self._demonstrate_fp4_packing()
        self.next_slide()

        # Show scale format
        self._show_scale_format()
        self.next_slide()

    def _show_block_structure(self):
        """Show 16-element block structure."""
        torch.manual_seed(42)
        # tensor_data = torch.randn(2, 64) * 5

        block_groups = VGroup()
        for row in range(2):
            for block_idx in range(4):
                block_rect = Rectangle(
                    width=1.8, height=0.8, color=COLORS["nvfp_color"], stroke_width=2
                )
                block_rect.move_to(
                    LEFT * 3.5
                    + RIGHT * (block_idx * 2)
                    + DOWN * (row * 1.2)
                    + DOWN * 0.5
                )

                block_label = Text("16 elem", font_size=14, color=COLORS["text"])
                block_label.move_to(block_rect.get_center())

                block_groups.add(VGroup(block_rect, block_label))

        self.play(*[Create(b) for b in block_groups])

    def _demonstrate_fp4_packing(self):
        """Show how FP4 values are packed into bytes."""
        packing_title = Text(
            "FP4 Packing: 2 values → 1 byte", font_size=32, color=COLORS["text"]
        )
        packing_title.move_to(DOWN * 2.5)
        self.play(Write(packing_title))

        # Create packing visualization
        packing_group = self._create_packing_visual(
            y_position=DOWN * 3.5,
            val1_text="FP4 #1\\n4 bits",
            val2_text="FP4 #2\\n4 bits",
            packed_text="1 byte\\n(8 bits)",
        )

        self.play(
            *[
                Create(obj) if isinstance(obj, (Rectangle, Arrow)) else Write(obj)
                for obj in packing_group
            ]
        )

    def _create_packing_visual(self, y_position, val1_text, val2_text, packed_text):
        """Create visual representation of bit packing."""
        group = VGroup()

        # Two input values
        for i, text in enumerate([val1_text, val2_text]):
            rect = Rectangle(
                width=0.8, height=0.6, color=COLORS["tensor_data"], fill_opacity=0.3
            )
            rect.move_to(LEFT * (2 - i * 1.2) + y_position)
            label = Text(text, font_size=12, color=COLORS["text"])
            label.move_to(rect.get_center())
            group.add(rect, label)

        # Arrow
        arrow = Arrow(
            LEFT * 0.2 + y_position, RIGHT * 1 + y_position, color=COLORS["packed_data"]
        )
        group.add(arrow)

        # Packed output
        packed_rect = Rectangle(
            width=1.0, height=0.6, color=COLORS["packed_data"], fill_opacity=0.5
        )
        packed_rect.move_to(RIGHT * 2 + y_position)
        packed_label = Text(packed_text, font_size=12, color=COLORS["text"])
        packed_label.move_to(packed_rect.get_center())
        group.add(packed_rect, packed_label)

        return group

    def _show_scale_format(self):
        """Show E4M3 scale format."""
        scale_title = Text("Scales in E4M3 Format", font_size=28, color=COLORS["text"])
        scale_title.move_to(DOWN * 4.5)
        self.play(Write(scale_title))

        # Create scale grid
        scale_grid = VGroup()
        for row in range(2):
            for col in range(4):
                scale_cell = Rectangle(
                    width=0.6,
                    height=0.5,
                    color=COLORS["computed_scale"],
                    fill_opacity=0.3,
                )
                scale_cell.move_to(
                    LEFT * 2 + RIGHT * (col * 0.7) + DOWN * 5.2 + DOWN * (row * 0.6)
                )

                cell_text = Text("E4M3", font_size=10, color=COLORS["text"])
                cell_text.move_to(scale_cell.get_center())

                scale_grid.add(VGroup(scale_cell, cell_text))

        self.play(*[Create(s) for s in scale_grid])
