"""Show packing for FP4 and FP6 formats.

Demonstrates how sub-byte floating-point formats (FP4 and FP6) are packed
into bytes for efficient storage and computation.

To run:
manim-slides render vizz/quant/packing_visualization.py PackingVisualization -ql
manim-slides present PackingVisualization
"""

from manim import *
from manim_slides import Slide

from vizz.quant.mx_base import COLORS


class PackingVisualization(Slide):
    """Show packing for FP4 and FP6 formats."""

    def construct(self):
        """Demonstrate bit packing for sub-byte formats."""
        # Title
        title = Text("Sub-byte Format Packing", font_size=48, color=COLORS["text"])
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Show FP4 packing
        self._show_fp4_packing()
        self.next_slide()

        # Show FP6 packing
        self._show_fp6_packing()
        self.next_slide()

        # Show benefits
        self._show_packing_benefits()
        self.next_slide()

    def _show_fp4_packing(self):
        """Show FP4 packing (2 values per byte)."""
        fp4_title = Text(
            "FP4 Packing: 2 × 4-bit → 8-bit", font_size=32, color=COLORS["nvfp_color"]
        )
        fp4_title.move_to(UP * 2)
        self.play(Write(fp4_title))

        fp4_group = VGroup()

        # Two FP4 input values with binary
        for i, (label, binary_val) in enumerate(
            [("High 4 bits", "0101"), ("Low 4 bits", "1100")]
        ):
            rect = Rectangle(
                width=1.5, height=0.8, color=COLORS["tensor_data"], fill_opacity=0.3
            )
            rect.move_to(LEFT * (3 - i * 2) + UP * 0.5)

            text = Text(label, font_size=16, color=COLORS["text"])
            text.move_to(rect.get_center())

            binary = Text(binary_val, font_size=14, color=COLORS["text"])
            binary.next_to(rect, DOWN, buff=0.1)

            fp4_group.add(VGroup(rect, text, binary))

        # Arrow
        arrow = Arrow(
            LEFT * 0.5 + UP * 0.5, RIGHT * 1 + UP * 0.5, color=COLORS["packed_data"]
        )
        fp4_group.add(arrow)

        # Packed byte output
        packed_byte = Rectangle(
            width=2, height=0.8, color=COLORS["packed_data"], fill_opacity=0.5
        )
        packed_byte.move_to(RIGHT * 3 + UP * 0.5)

        packed_label = Text("Packed Byte", font_size=16, color=COLORS["text"])
        packed_label.move_to(packed_byte.get_center())

        packed_binary = Text("01011100", font_size=14, color=COLORS["text"])
        packed_binary.next_to(packed_byte, DOWN, buff=0.1)

        fp4_group.add(VGroup(packed_byte, packed_label, packed_binary))

        self.play(Create(fp4_group))

    def _show_fp6_packing(self):
        """Show FP6 packing (4 values per 3 bytes)."""
        fp6_title = Text(
            "FP6 Packing: 4 × 6-bit → 3 bytes", font_size=32, color=COLORS["mx_color"]
        )
        fp6_title.move_to(DOWN * 1.5)
        self.play(Write(fp6_title))

        fp6_group = VGroup()

        # Four FP6 input values
        for i in range(4):
            rect = Rectangle(
                width=0.8, height=0.6, color=COLORS["tensor_data"], fill_opacity=0.3
            )
            rect.move_to(LEFT * (4 - i * 1.2) + DOWN * 2.5)

            text = Text(f"FP6\\n#{i+1}", font_size=12, color=COLORS["text"])
            text.move_to(rect.get_center())

            fp6_group.add(VGroup(rect, text))

        # Arrow
        pack_arrow = Arrow(
            LEFT * 0.2 + DOWN * 2.5,
            RIGHT * 1.5 + DOWN * 2.5,
            color=COLORS["packed_data"],
        )
        fp6_group.add(pack_arrow)

        # Three output bytes
        for i in range(3):
            byte_rect = Rectangle(
                width=0.7, height=0.6, color=COLORS["packed_data"], fill_opacity=0.5
            )
            byte_rect.move_to(RIGHT * (2.5 + i * 0.8) + DOWN * 2.5)

            byte_text = Text(f"B{i}", font_size=12, color=COLORS["text"])
            byte_text.move_to(byte_rect.get_center())

            fp6_group.add(VGroup(byte_rect, byte_text))

        self.play(Create(fp6_group))

        # Show bit calculation
        bit_formula = Text(
            "4 × 6 bits = 24 bits = 3 bytes", font_size=20, color=COLORS["text"]
        )
        bit_formula.move_to(DOWN * 3.5)
        self.play(Write(bit_formula))

    def _show_packing_benefits(self):
        """Show the benefits of bit packing."""
        summary_title = Text("Packing Benefits:", font_size=28, color=COLORS["text"])
        summary_title.move_to(DOWN * 4.5)

        summary_points = Text(
            "• Reduced memory footprint\\n"
            + "• Better cache utilization\\n"
            + "• Efficient GEMM operations",
            font_size=20,
            color=COLORS["text"],
        )
        summary_points.next_to(summary_title, DOWN, buff=0.3)

        self.play(Write(summary_title), Write(summary_points))
