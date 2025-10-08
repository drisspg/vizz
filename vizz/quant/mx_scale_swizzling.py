"""Visualizations for MX/NVFP4 scale swizzling for GEMM operations.

Phase 2: Shows what we do with the scales after calculation -
the swizzling/blocking transformation for efficient GEMM computation.

To run:
manim-slides render vizz/quant/mx_scale_swizzling.py SwizzlingOverview -ql
manim-slides present SwizzlingOverview
"""

from manim import *
from manim_slides import Slide

# Light theme configuration
config.background_color = WHITE

# Color scheme
COLORS = {
    "text": BLACK,
    "matrix": DARK_GRAY,
    "tensor_data": BLUE_D,
    "mx_color": PURPLE_D,
    "nvfp_color": ORANGE,
    "computed_scale": GREEN_D,
    "swizzled_layout": PURPLE,
    "block_boundary": GRAY,
    "tile_color": TEAL_D,
    "arrow": RED_D,
}


class SwizzlingOverview(Slide):
    """Introduction to swizzling concept and why it matters."""

    def construct(self):
        # Title
        title = Text(
            "Scale Swizzling for Efficient GEMM", font_size=48, color=COLORS["text"]
        )
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # The problem
        problem_title = Text("The Problem:", font_size=36, color=COLORS["text"])
        problem_title.next_to(title, DOWN, buff=0.5)

        problem_text = Text(
            "After calculating block-wise scales,\n"
            + "we need to organize them for efficient GEMM operations",
            font_size=24,
            color=COLORS["text"],
        )
        problem_text.next_to(problem_title, DOWN, buff=0.3)

        self.play(Write(problem_title), Write(problem_text))
        self.next_slide()

        # Show linear layout
        linear_label = Text(
            "Linear Layout (after scale calculation):",
            font_size=28,
            color=COLORS["text"],
        )
        linear_label.move_to(UP * 1)

        # Create a simple scale matrix visualization
        scale_grid = VGroup()
        for i in range(4):
            for j in range(4):
                cell = Rectangle(
                    width=0.6,
                    height=0.6,
                    color=COLORS["computed_scale"],
                    fill_opacity=0.3,
                )
                cell.move_to(
                    LEFT * 1.5 + RIGHT * (j * 0.7) + UP * 0.2 + DOWN * (i * 0.7)
                )

                label = Text(f"S{i},{j}", font_size=12, color=COLORS["text"])
                label.move_to(cell.get_center())

                scale_grid.add(VGroup(cell, label))

        self.play(Write(linear_label), *[Create(s) for s in scale_grid])
        self.next_slide()

        # Show swizzled layout
        swizzle_arrow = Arrow(
            DOWN * 1.5 + LEFT * 2,
            DOWN * 1.5 + RIGHT * 2,
            color=COLORS["arrow"],
            stroke_width=4,
        )
        swizzle_label = Text("Swizzle", font_size=24, color=COLORS["text"])
        swizzle_label.next_to(swizzle_arrow, UP, buff=0.1)

        self.play(Create(swizzle_arrow), Write(swizzle_label))
        self.next_slide()

        # Show swizzled layout
        swizzled_label = Text(
            "Swizzled Layout (optimized for GEMM):", font_size=28, color=COLORS["text"]
        )
        swizzled_label.move_to(DOWN * 2.5)

        swizzled_grid = VGroup()
        # Reordered for visualization
        reorder = [
            (0, 0),
            (2, 0),
            (0, 1),
            (2, 1),
            (1, 0),
            (3, 0),
            (1, 1),
            (3, 1),
            (0, 2),
            (2, 2),
            (0, 3),
            (2, 3),
            (1, 2),
            (3, 2),
            (1, 3),
            (3, 3),
        ]

        for idx, (i, j) in enumerate(reorder):
            cell = Rectangle(
                width=0.5, height=0.5, color=COLORS["swizzled_layout"], fill_opacity=0.5
            )
            row = idx // 8
            col = idx % 8
            cell.move_to(
                LEFT * 2.5 + RIGHT * (col * 0.6) + DOWN * 3.3 + DOWN * (row * 0.6)
            )

            label = Text(f"S{i},{j}", font_size=10, color=COLORS["text"])
            label.move_to(cell.get_center())

            swizzled_grid.add(VGroup(cell, label))

        self.play(Write(swizzled_label), *[Create(s) for s in swizzled_grid])
        self.next_slide()

        # Benefits
        benefits_title = Text(
            "Benefits of Swizzling:", font_size=28, color=COLORS["text"]
        )
        benefits_title.move_to(DOWN * 5)

        benefits = Text(
            "• Better memory access patterns\n"
            + "• Improved cache utilization\n"
            + "• Efficient tensor core operations",
            font_size=20,
            color=COLORS["text"],
        )
        benefits.next_to(benefits_title, DOWN, buff=0.2)

        self.play(Write(benefits_title), Write(benefits))
        self.next_slide()


class SwizzlingPatternVisualization(Slide):
    """Detailed visualization of the to_blocked transformation."""

    def construct(self):
        # Title
        title = Text(
            "Swizzling Pattern: 128×4 → 32×16", font_size=48, color=COLORS["text"]
        )
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Show original scale tensor
        subtitle = Text(
            "Scale Tensor Shape: [128, 4]", font_size=32, color=COLORS["text"]
        )
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(subtitle))

        # Visualize as blocks
        info_text = Text(
            "Conceptually: 1 row-block × 1 col-block",
            font_size=24,
            color=COLORS["text"],
        )
        info_text.next_to(subtitle, DOWN, buff=0.3)
        self.play(Write(info_text))
        self.next_slide()

        # Show the original layout
        original_rect = Rectangle(
            width=2, height=6, color=COLORS["computed_scale"], fill_opacity=0.2
        )
        original_rect.move_to(LEFT * 3.5 + UP * 0.5)

        original_label = Text("128×4\nLinear", font_size=20, color=COLORS["text"])
        original_label.next_to(original_rect, DOWN, buff=0.2)

        self.play(Create(original_rect), Write(original_label))
        self.next_slide()

        # Step 1: View as blocks
        step1_text = Text(
            "Step 1: view(1, 128, 1, 4)", font_size=24, color=COLORS["text"]
        )
        step1_text.move_to(UP * 3)
        self.play(Write(step1_text))

        # Show 128 row and 4 column structure
        block_128_4 = VGroup()
        for i in range(4):
            segment = Rectangle(
                width=0.4, height=1.5, color=COLORS["tile_color"], stroke_width=2
            )
            segment.move_to(LEFT * 3.5 + RIGHT * (i * 0.5) + UP * 0.5)
            block_128_4.add(segment)

        self.play(*[Create(b) for b in block_128_4])
        self.next_slide()

        # Step 2: Permute
        step2_text = Text(
            "Step 2: permute(0, 2, 1, 3)", font_size=24, color=COLORS["text"]
        )
        step2_text.next_to(step1_text, DOWN, buff=0.2)
        self.play(Write(step2_text))

        # Show the permutation with arrow
        perm_arrow = Arrow(
            LEFT * 2 + UP * 0.5,
            RIGHT * 0.5 + UP * 0.5,
            color=COLORS["arrow"],
            stroke_width=3,
        )
        self.play(Create(perm_arrow))
        self.next_slide()

        # Step 3: Reshape
        step3_text = Text(
            "Step 3: reshape(-1, 4, 32, 4)", font_size=24, color=COLORS["text"]
        )
        step3_text.next_to(step2_text, DOWN, buff=0.2)
        self.play(Write(step3_text))

        # Show intermediate structure
        reshaped_group = VGroup()
        for i in range(4):
            for j in range(4):
                small_rect = Rectangle(
                    width=0.3,
                    height=0.3,
                    color=COLORS["swizzled_layout"],
                    fill_opacity=0.4,
                )
                small_rect.move_to(
                    RIGHT * 2 + RIGHT * (j * 0.4) + UP * 1 + DOWN * (i * 0.4)
                )
                reshaped_group.add(small_rect)

        self.play(*[Create(r) for r in reshaped_group])
        self.next_slide()

        # Step 4: Transpose and final reshape
        step4_text = Text(
            "Step 4: transpose(1, 2).reshape(-1, 32, 16)",
            font_size=24,
            color=COLORS["text"],
        )
        step4_text.next_to(step3_text, DOWN, buff=0.2)
        self.play(Write(step4_text))

        final_arrow = Arrow(
            RIGHT * 2 + DOWN * 0.5,
            RIGHT * 2 + DOWN * 2,
            color=COLORS["arrow"],
            stroke_width=3,
        )
        self.play(Create(final_arrow))
        self.next_slide()

        # Show final swizzled layout
        final_rect = Rectangle(
            width=4, height=2, color=COLORS["swizzled_layout"], fill_opacity=0.3
        )
        final_rect.move_to(RIGHT * 2 + DOWN * 3)

        final_label = Text("32×16\nSwizzled", font_size=20, color=COLORS["text"])
        final_label.next_to(final_rect, DOWN, buff=0.2)

        self.play(Create(final_rect), Write(final_label))
        self.next_slide()

        # Show the formula
        formula = MathTex(
            r"\text{blocks.view}(n_r, 128, n_c, 4).\text{permute}(0, 2, 1, 3)",
            font_size=20,
            color=COLORS["text"],
        )
        formula.move_to(DOWN * 5.5)

        formula2 = MathTex(
            r".\text{reshape}(-1, 4, 32, 4).\text{transpose}(1, 2).\text{reshape}(-1, 32, 16)",
            font_size=18,
            color=COLORS["text"],
        )
        formula2.next_to(formula, DOWN, buff=0.1)

        self.play(Write(formula), Write(formula2))
        self.next_slide()


class GEMMInputSwizzling(Slide):
    """Show how A and B matrices are prepared for GEMM."""

    def construct(self):
        # Title
        title = Text("GEMM Input Preparation", font_size=48, color=COLORS["text"])
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Show data and scale relationship
        relationship_title = Text(
            "Data ↔ Scale Tile Relationship", font_size=36, color=COLORS["text"]
        )
        relationship_title.next_to(title, DOWN, buff=0.5)
        self.play(Write(relationship_title))
        self.next_slide()

        # For NVFP4: 128×64 data → 32×16 scale
        nvfp4_example = VGroup()

        # Data tile
        data_tile = Rectangle(
            width=3, height=2.5, color=COLORS["tensor_data"], fill_opacity=0.2
        )
        data_tile.move_to(LEFT * 3.5 + UP * 0.5)

        data_label = Text(
            "Data Tile\n128×64\n(unpacked)", font_size=20, color=COLORS["text"]
        )
        data_label.next_to(data_tile, DOWN, buff=0.2)

        nvfp4_example.add(data_tile, data_label)

        # Arrow
        arrow = Arrow(
            LEFT * 1.5 + UP * 0.5, RIGHT * 0.5 + UP * 0.5, color=COLORS["arrow"]
        )
        nvfp4_example.add(arrow)

        # Scale tile
        scale_tile = Rectangle(
            width=2, height=1.5, color=COLORS["swizzled_layout"], fill_opacity=0.4
        )
        scale_tile.move_to(RIGHT * 3 + UP * 0.5)

        scale_label = Text("Swizzled Scale\n32×16", font_size=20, color=COLORS["text"])
        scale_label.next_to(scale_tile, DOWN, buff=0.2)

        nvfp4_example.add(scale_tile, scale_label)

        self.play(
            *[
                Create(obj) if isinstance(obj, (Rectangle, Arrow)) else Write(obj)
                for obj in nvfp4_example
            ]
        )
        self.next_slide()

        # Show the formula
        formula_title = Text(
            "Scale Dimension Calculation:", font_size=28, color=COLORS["text"]
        )
        formula_title.move_to(DOWN * 1.5)

        formula = MathTex(
            r"\text{scale\_M} = \lceil \frac{M}{128} \rceil \times 32",
            font_size=24,
            color=COLORS["text"],
        )
        formula.next_to(formula_title, DOWN, buff=0.3)

        formula2 = MathTex(
            r"\text{scale\_K} = \lceil \frac{K}{64} \rceil \times 16",
            font_size=24,
            color=COLORS["text"],
        )
        formula2.next_to(formula, DOWN, buff=0.2)

        self.play(Write(formula_title), Write(formula), Write(formula2))
        self.next_slide()

        # Example calculation
        example_text = Text("Example: M=256, K=128", font_size=24, color=COLORS["text"])
        example_text.move_to(DOWN * 3.5)

        calc1 = MathTex(
            r"\text{scale\_M} = \lceil \frac{256}{128} \rceil \times 32 = 2 \times 32 = 64",
            font_size=20,
            color=COLORS["text"],
        )
        calc1.next_to(example_text, DOWN, buff=0.2)

        calc2 = MathTex(
            r"\text{scale\_K} = \lceil \frac{128}{64} \rceil \times 16 = 2 \times 16 = 32",
            font_size=20,
            color=COLORS["text"],
        )
        calc2.next_to(calc1, DOWN, buff=0.2)

        result_text = Text(
            "→ Swizzled scale tensor: [64, 32]",
            font_size=20,
            color=COLORS["computed_scale"],
        )
        result_text.next_to(calc2, DOWN, buff=0.3)

        self.play(Write(example_text), Write(calc1), Write(calc2), Write(result_text))
        self.next_slide()


class CompleteQuantizationFlow(Slide):
    """End-to-end pipeline from high precision to swizzled GEMM-ready format."""

    def construct(self):
        # Title
        title = Text(
            "Complete Quantization + Swizzling Pipeline",
            font_size=44,
            color=COLORS["text"],
        )
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Create pipeline stages
        stages = VGroup()

        # Stage 1: High precision
        stage1 = VGroup()
        stage1_rect = Rectangle(
            width=2.5, height=1.2, color=COLORS["tensor_data"], fill_opacity=0.2
        )
        stage1_rect.move_to(LEFT * 5 + UP * 2)
        stage1_label = Text(
            "1. High Precision\nTensor [M, K]", font_size=16, color=COLORS["text"]
        )
        stage1_label.move_to(stage1_rect.get_center())
        stage1.add(stage1_rect, stage1_label)

        # Stage 2: Block division
        stage2 = VGroup()
        stage2_rect = Rectangle(
            width=2.5, height=1.2, color=COLORS["mx_color"], fill_opacity=0.2
        )
        stage2_rect.move_to(LEFT * 5 + DOWN * 0)
        stage2_label = Text(
            "2. Divide into\nBlocks", font_size=16, color=COLORS["text"]
        )
        stage2_label.move_to(stage2_rect.get_center())
        stage2.add(stage2_rect, stage2_label)

        # Stage 3: Scale calculation
        stage3 = VGroup()
        stage3_rect = Rectangle(
            width=2.5, height=1.2, color=COLORS["computed_scale"], fill_opacity=0.2
        )
        stage3_rect.move_to(LEFT * 5 + DOWN * 2)
        stage3_label = Text("3. Calculate\nScales", font_size=16, color=COLORS["text"])
        stage3_label.move_to(stage3_rect.get_center())
        stage3.add(stage3_rect, stage3_label)

        # Arrows between left stages
        arrow1 = Arrow(
            stage1_rect.get_bottom(),
            stage2_rect.get_top(),
            color=COLORS["arrow"],
            buff=0.1,
        )
        arrow2 = Arrow(
            stage2_rect.get_bottom(),
            stage3_rect.get_top(),
            color=COLORS["arrow"],
            buff=0.1,
        )

        stages.add(stage1, stage2, stage3, arrow1, arrow2)

        self.play(
            *[
                Create(s) if isinstance(s, (Rectangle, Arrow)) else Write(s)
                for s in stages
            ]
        )
        self.next_slide()

        # Right side stages
        right_stages = VGroup()

        # Stage 4: Quantization
        stage4 = VGroup()
        stage4_rect = Rectangle(
            width=2.5, height=1.2, color=COLORS["nvfp_color"], fill_opacity=0.2
        )
        stage4_rect.move_to(RIGHT * 2 + UP * 2)
        stage4_label = Text("4. Quantize\nData", font_size=16, color=COLORS["text"])
        stage4_label.move_to(stage4_rect.get_center())
        stage4.add(stage4_rect, stage4_label)

        # Stage 5: Swizzling
        stage5 = VGroup()
        stage5_rect = Rectangle(
            width=2.5, height=1.2, color=COLORS["swizzled_layout"], fill_opacity=0.4
        )
        stage5_rect.move_to(RIGHT * 2 + DOWN * 0)
        stage5_label = Text("5. Swizzle\nScales", font_size=16, color=COLORS["text"])
        stage5_label.move_to(stage5_rect.get_center())
        stage5.add(stage5_rect, stage5_label)

        # Stage 6: GEMM ready
        stage6 = VGroup()
        stage6_rect = Rectangle(
            width=2.5, height=1.2, color=COLORS["tile_color"], fill_opacity=0.4
        )
        stage6_rect.move_to(RIGHT * 2 + DOWN * 2)
        stage6_label = Text("6. GEMM-Ready\nFormat", font_size=16, color=COLORS["text"])
        stage6_label.move_to(stage6_rect.get_center())
        stage6.add(stage6_rect, stage6_label)

        # Arrows
        arrow3 = Arrow(
            stage3_rect.get_right(),
            stage4_rect.get_left(),
            color=COLORS["arrow"],
            buff=0.1,
        )
        arrow4 = Arrow(
            stage4_rect.get_bottom(),
            stage5_rect.get_top(),
            color=COLORS["arrow"],
            buff=0.1,
        )
        arrow5 = Arrow(
            stage5_rect.get_bottom(),
            stage6_rect.get_top(),
            color=COLORS["arrow"],
            buff=0.1,
        )

        right_stages.add(stage4, stage5, stage6, arrow3, arrow4, arrow5)

        self.play(
            *[
                Create(s) if isinstance(s, (Rectangle, Arrow)) else Write(s)
                for s in right_stages
            ]
        )
        self.next_slide()

        # Show key insight
        insight_title = Text("Key Insight:", font_size=32, color=COLORS["text"])
        insight_title.move_to(DOWN * 4)

        insight_text = Text(
            "Swizzling reorganizes scales to match\n"
            + "how tensor cores access memory during GEMM",
            font_size=22,
            color=COLORS["text"],
        )
        insight_text.next_to(insight_title, DOWN, buff=0.3)

        self.play(Write(insight_title), Write(insight_text))
        self.next_slide()

        # Performance benefit
        perf_text = Text(
            "Result: Faster matrix multiplications on GPU",
            font_size=24,
            color=COLORS["computed_scale"],
        )
        perf_text.next_to(insight_text, DOWN, buff=0.4)

        self.play(Write(perf_text))
        self.next_slide()
