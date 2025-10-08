"""Introduction to block-wise quantization concept.

Shows how high-precision tensors are divided into blocks for MX/NVFP4 formats.

To run:
manim-slides render vizz/quant/block_scaling_overview.py BlockScalingOverview -ql
manim-slides present BlockScalingOverview
"""

from manim import *
from manim_slides import Slide

from vizz.quant.mx_base import COLORS


class BlockScalingOverview(Slide):
    """Introduction to block-wise quantization concept."""

    def construct(self):
        """Show how tensors are divided into blocks for different formats."""
        # Show title and tensor
        title = Text(
            "Block-wise Scaling for MX/NVFP4 Formats",
            font_size=48,
            color=COLORS["text"],
        )
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        tensor_rect = self._create_example_tensor(title)
        self.next_slide()

        self._show_reduction_dimension(tensor_rect)
        self.next_slide()

        self._demonstrate_mx_blocks(tensor_rect)
        self.next_slide()

        self._demonstrate_nvfp_blocks(tensor_rect)
        self.next_slide()

        self._show_block_formula(tensor_rect)
        self.next_slide()

    def _create_example_tensor(self, title):
        """Create and display the example tensor visualization."""
        tensor_shape = (4, 128)
        subtitle = Text(
            f"Example Tensor: {tensor_shape}", font_size=36, color=COLORS["text"]
        )
        subtitle.next_to(title, DOWN, buff=0.5)
        self.play(Write(subtitle))

        tensor_rect = Rectangle(
            width=8, height=2, color=COLORS["tensor_data"], fill_opacity=0.3
        )
        tensor_rect.next_to(subtitle, DOWN, buff=0.5)

        tensor_label = Text(
            "High Precision Tensor\\n[4, 128]", font_size=24, color=COLORS["text"]
        )
        tensor_label.next_to(tensor_rect, DOWN, buff=0.2)

        self.play(Create(tensor_rect), Write(tensor_label))
        return tensor_rect

    def _show_reduction_dimension(self, tensor_rect):
        """Show the reduction dimension with an arrow."""
        reduction_arrow = Arrow(
            tensor_rect.get_left() + LEFT * 0.5,
            tensor_rect.get_right() + RIGHT * 0.5,
            color=COLORS["max_value"],
            stroke_width=3,
        )
        reduction_label = Text(
            "Reduction Dimension (K=128)", font_size=20, color=COLORS["text"]
        )
        reduction_label.next_to(reduction_arrow, DOWN, buff=0.2)

        self.play(Create(reduction_arrow), Write(reduction_label))

    def _demonstrate_mx_blocks(self, tensor_rect):
        """Show MX format with 32-element blocks."""
        mx_label = Text(
            "MX Format: 32-element blocks", font_size=24, color=COLORS["mx_color"]
        )
        mx_label.move_to(tensor_rect.get_center() + UP * 3)

        mx_blocks = self._create_format_blocks(
            tensor_rect,
            num_blocks=4,
            color=COLORS["mx_color"],
            block_size=32,
            stroke_width=3,
        )

        self.play(
            Write(mx_label),
            *[Create(b[0]) for b in mx_blocks],
            *[Write(b[1]) for b in mx_blocks],
        )

    def _demonstrate_nvfp_blocks(self, tensor_rect):
        """Show NVFP4 format with 16-element blocks."""
        # Fade out MX blocks first
        # mx_objects = [obj for obj in self.mobjects if isinstance(obj, (VGroup, Text))]
        # relevant_mx = [
        #     obj for obj in mx_objects if "MX" in str(obj) or "Block" in str(obj)
        # ]

        nvfp_label = Text(
            "NVFP4 Format: 16-element blocks", font_size=24, color=COLORS["nvfp_color"]
        )
        nvfp_label.move_to(tensor_rect.get_center() + UP * 3)

        nvfp_blocks = self._create_format_blocks(
            tensor_rect,
            num_blocks=8,
            color=COLORS["nvfp_color"],
            block_size=16,
            stroke_width=2,
            font_size=12,
        )

        self.play(
            Write(nvfp_label),
            *[Create(b[0]) for b in nvfp_blocks],
            *[Write(b[1]) for b in nvfp_blocks],
        )

    def _create_format_blocks(
        self, tensor_rect, num_blocks, color, block_size, stroke_width=2, font_size=16
    ):
        """Create a row of blocks for a specific format."""
        blocks = VGroup()
        block_width = tensor_rect.width / num_blocks

        for i in range(num_blocks):
            block = Rectangle(
                width=block_width,
                height=tensor_rect.height,
                color=color,
                stroke_width=stroke_width,
            )
            block.move_to(tensor_rect.get_left() + RIGHT * (block_width * (i + 0.5)))

            label_text = (
                f"Block {i}\\n{block_size} elem"
                if num_blocks <= 4
                else f"{i}\\n{block_size}"
            )
            block_label = Text(label_text, font_size=font_size, color=COLORS["text"])
            block_label.move_to(block.get_center())

            blocks.add(VGroup(block, block_label))

        return blocks

    def _show_block_formula(self, tensor_rect):
        """Display the formula for calculating number of blocks."""
        formula = MathTex(
            r"\\text{num\\_blocks} = \\lceil \\frac{\\text{dim\\_size}}{\\text{block\\_size}} \\rceil",
            color=COLORS["text"],
            font_size=36,
        )
        formula.next_to(tensor_rect, DOWN, buff=1.5)
        self.play(Write(formula))
