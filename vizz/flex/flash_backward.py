"""
Flash Attention Backward Pass Visualization

Compares the 5-dot (non-deterministic) vs 7-dot (deterministic) backward pass.

To run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate manim
    manim-slides render vizz/flex/flash_backward.py FlashAttentionBackwardVisualization -ql
    manim-slides present FlashAttentionBackwardVisualization
"""

from manim import *
from manim_slides import Slide

# Custom configuration to use a light theme for presentations
config.background_color = WHITE

# Constants for consistent styling
COLORS = {
    "text": BLACK,
    "matrix": DARK_GRAY,
    "bracket": DARK_GRAY,
    "matmul_op": ORANGE,
    "elementwise_op": TEAL,
    "memory": LIGHT_GRAY,
    "deterministic": GREEN_D,
    "non_deterministic": RED_D,
    "thread_blocks": [
        "#E74C3C",
        "#3498DB",
        "#2ECC71",
        "#9B59B6",
    ],  # Red, Blue, Green, Purple
    "highlight": {
        "query": GOLD_D,
        "key": BLUE_D,
        "result": GREEN_D,
    },
}

OPACITY = {"highlight": 0.2}


class ThreadBlockViz(VGroup):
    """Visual representation of a GPU thread block."""

    def __init__(self, block_id, color, label_text=None, width=1.5, height=0.8):
        super().__init__()
        rect = RoundedRectangle(
            width=width, height=height, corner_radius=0.1, stroke_width=2
        )
        rect.set_fill(color, opacity=0.3)
        rect.set_stroke(color, width=2)
        label = Text(label_text or f"TB{block_id}", font_size=20, color=COLORS["text"])
        label.move_to(rect)
        self.add(rect, label)
        self.rect = rect
        self.label = label


class MemoryCell(VGroup):
    """Visual representation of a memory location."""

    def __init__(self, label, width=1.2, height=0.8):
        super().__init__()
        cell = Rectangle(width=width, height=height)
        cell.set_fill(COLORS["memory"], opacity=0.3)
        cell.set_stroke(DARK_GRAY, width=1)
        text = Text(label, font_size=18, color=COLORS["text"])
        text.move_to(cell)
        self.add(cell, text)
        self.cell = cell
        self.text = text

    def update_text(self, new_label):
        """Update the text in the cell."""
        new_text = Text(new_label, font_size=18, color=COLORS["text"])
        new_text.move_to(self.cell)
        self[1] = new_text
        self.text = new_text


class ComputationNode(VGroup):
    """Node in computational graph."""

    def __init__(self, formula, is_matmul=False, font_size=28):
        super().__init__()
        color = COLORS["matmul_op"] if is_matmul else COLORS["elementwise_op"]
        box = RoundedRectangle(width=2.8, height=0.7, corner_radius=0.15)
        box.set_fill(color, opacity=0.2)
        box.set_stroke(color, width=2)
        label = MathTex(formula, font_size=font_size, color=COLORS["text"])
        label.move_to(box)
        self.add(box, label)
        self.box = box
        self.label = label


class PartitionBlock(VGroup):
    """Visual representation of a data partition/block."""

    def __init__(self, label, color, width=0.8, height=0.6):
        super().__init__()
        rect = Rectangle(width=width, height=height)
        rect.set_fill(color, opacity=0.4)
        rect.set_stroke(color, width=2)
        text = Text(label, font_size=16, color=COLORS["text"])
        text.move_to(rect)
        self.add(rect, text)


class FlashAttentionBackwardVisualization(Slide):
    """
    Visualization comparing Flash Attention backward pass:
    - 5 dot products (non-deterministic, uses atomic adds)
    - 7 dot products (deterministic, uses split accumulation)
    """

    def advance_slide(self):
        """Helper method to advance slides."""
        if isinstance(self, Slide):
            self.next_slide()
        else:
            self.wait(0.25)

    def setup_formula_banner(self, formula_text):
        """Setup a formula banner at the bottom of the screen."""
        formula_banner = Rectangle(
            width=config.frame_width,
            height=1.25,
            fill_color=LIGHT_GREY,
            fill_opacity=0.3,
            stroke_width=1,
            stroke_color=GREY,
        ).to_edge(DOWN, buff=0.2)

        formula = MathTex(formula_text, color=COLORS["text"], font_size=32)
        formula.move_to(formula_banner.get_center())

        return VGroup(formula_banner, formula)

    def create_title(self, text, font_size=36):
        """Create a title text at the top of the screen."""
        title = Text(text, font_size=font_size, color=COLORS["text"])
        title.to_edge(UP, buff=0.5)
        return title

    # =========================================================================
    # Slide 1: Backward Pass Overview
    # =========================================================================
    def slide_1_backward_overview(self):
        """Show the backward pass operations overview."""
        title = self.create_title("Flash Attention Backward Pass")
        self.play(Write(title))

        # Create the formula list
        formulas = VGroup(
            MathTex(r"S = Q \cdot K^T", color=COLORS["text"], font_size=32),
            MathTex(r"P = \text{softmax}(S)", color=COLORS["text"], font_size=32),
            MathTex(r"dV = P^T \cdot dO", color=COLORS["text"], font_size=32),
            MathTex(r"dP = dO \cdot V^T", color=COLORS["text"], font_size=32),
            MathTex(
                r"dS = P \odot (dP - \text{rowsum}(dP \odot P))",
                color=COLORS["text"],
                font_size=32,
            ),
            MathTex(r"dQ = dS \cdot K", color=COLORS["text"], font_size=32),
            MathTex(r"dK = dS^T \cdot Q", color=COLORS["text"], font_size=32),
        )
        formulas.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        formulas.center().shift(DOWN * 0.3)

        # Add labels
        labels = VGroup(
            Text("(recompute scores)", font_size=20, color=DARK_GRAY),
            Text("(recompute probs)", font_size=20, color=DARK_GRAY),
            Text("(gradient for V)", font_size=20, color=DARK_GRAY),
            Text("(intermediate)", font_size=20, color=DARK_GRAY),
            Text("(softmax backward)", font_size=20, color=DARK_GRAY),
            Text("(gradient for Q)", font_size=20, color=DARK_GRAY),
            Text("(gradient for K)", font_size=20, color=DARK_GRAY),
        )

        for label, formula in zip(labels, formulas):
            label.next_to(formula, RIGHT, buff=0.5)

        # Animate formulas appearing
        self.play(
            LaggedStart(
                *[Write(f) for f in formulas],
                lag_ratio=0.15,
            )
        )
        self.play(
            LaggedStart(
                *[FadeIn(label, shift=RIGHT * 0.2) for label in labels],
                lag_ratio=0.1,
            )
        )

        self.advance_slide()

        # Highlight the 5 matmul operations
        matmul_indices = [0, 2, 3, 5, 6]  # S, dV, dP, dQ, dK
        highlights = []
        for i in matmul_indices:
            rect = SurroundingRectangle(
                formulas[i], color=COLORS["matmul_op"], buff=0.1, stroke_width=3
            )
            highlights.append(rect)

        highlight_group = VGroup(*highlights)

        matmul_label = Text(
            "5 Matrix Multiplications (Dot Products)",
            font_size=28,
            color=COLORS["matmul_op"],
        )
        matmul_label.to_edge(DOWN, buff=0.8)

        self.play(
            LaggedStart(*[Create(h) for h in highlights], lag_ratio=0.1),
            Write(matmul_label),
        )

        self.advance_slide()

        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(formulas),
            FadeOut(labels),
            FadeOut(highlight_group),
            FadeOut(matmul_label),
        )

    # =========================================================================
    # Slide 2: Computational Graph
    # =========================================================================
    def slide_2_computational_graph(self):
        """Show the computational graph with dependencies."""
        title = self.create_title("Backward Pass Computational Graph")
        self.play(Write(title))

        # Create computation nodes
        # Inputs
        q_node = ComputationNode("Q", is_matmul=False, font_size=24)
        k_node = ComputationNode("K", is_matmul=False, font_size=24)
        v_node = ComputationNode("V", is_matmul=False, font_size=24)
        do_node = ComputationNode("dO", is_matmul=False, font_size=24)

        # Intermediate computations
        s_node = ComputationNode(r"S = Q K^T", is_matmul=True, font_size=22)
        p_node = ComputationNode(
            r"P = \text{softmax}(S)", is_matmul=False, font_size=22
        )

        # Gradient computations
        dv_node = ComputationNode(r"dV = P^T dO", is_matmul=True, font_size=22)
        dp_node = ComputationNode(r"dP = dO V^T", is_matmul=True, font_size=22)
        ds_node = ComputationNode(r"dS = ...", is_matmul=False, font_size=22)
        dq_node = ComputationNode(r"dQ = dS \cdot K", is_matmul=True, font_size=22)
        dk_node = ComputationNode(r"dK = dS^T Q", is_matmul=True, font_size=22)

        # Position nodes in a graph layout
        # Top row: inputs
        inputs = VGroup(q_node, k_node, v_node, do_node)
        inputs.arrange(RIGHT, buff=0.8)
        inputs.shift(UP * 2.5)

        # Middle row: forward recomputation
        s_node.next_to(inputs, DOWN, buff=1.0)
        s_node.shift(LEFT * 2)
        p_node.next_to(s_node, RIGHT, buff=1.5)

        # Bottom row: gradients
        dv_node.next_to(p_node, DOWN, buff=0.8).shift(LEFT * 1)
        dp_node.next_to(dv_node, RIGHT, buff=0.5)
        ds_node.next_to(VGroup(dv_node, dp_node), DOWN, buff=0.6)
        dq_node.next_to(ds_node, DOWN, buff=0.6).shift(LEFT * 1.5)
        dk_node.next_to(ds_node, DOWN, buff=0.6).shift(RIGHT * 1.5)

        # Create all nodes
        all_nodes = VGroup(
            q_node,
            k_node,
            v_node,
            do_node,
            s_node,
            p_node,
            dv_node,
            dp_node,
            ds_node,
            dq_node,
            dk_node,
        )

        self.play(
            LaggedStart(*[FadeIn(n, scale=0.8) for n in all_nodes], lag_ratio=0.08)
        )

        # Draw dependency arrows
        arrows = []

        def make_arrow(start, end):
            return Arrow(
                start.get_bottom()
                if start.get_center()[1] > end.get_center()[1]
                else start.get_right(),
                end.get_top()
                if start.get_center()[1] > end.get_center()[1]
                else end.get_left(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
                max_tip_length_to_length_ratio=0.15,
            )

        # Q -> S, K -> S
        arrows.append(
            Arrow(
                q_node.get_bottom(),
                s_node.get_top(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
            )
        )
        arrows.append(
            Arrow(
                k_node.get_bottom(),
                s_node.get_top(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
            )
        )
        # S -> P
        arrows.append(
            Arrow(
                s_node.get_right(),
                p_node.get_left(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
            )
        )
        # P -> dV, dO -> dV
        arrows.append(
            Arrow(
                p_node.get_bottom(),
                dv_node.get_top(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
            )
        )
        arrows.append(
            Arrow(
                do_node.get_bottom(),
                dv_node.get_top(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
            )
        )
        # dO -> dP, V -> dP
        arrows.append(
            Arrow(
                do_node.get_bottom(),
                dp_node.get_top(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
            )
        )
        arrows.append(
            Arrow(
                v_node.get_bottom(),
                dp_node.get_top(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
            )
        )
        # dP -> dS, P -> dS
        arrows.append(
            Arrow(
                dp_node.get_bottom(),
                ds_node.get_top(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
            )
        )
        arrows.append(
            Arrow(
                p_node.get_bottom(),
                ds_node.get_top(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
            )
        )
        # dS -> dQ, K -> dQ
        arrows.append(
            Arrow(
                ds_node.get_bottom(),
                dq_node.get_top(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
            )
        )
        arrows.append(
            Arrow(
                k_node.get_bottom(),
                dq_node.get_top(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
            )
        )
        # dS -> dK, Q -> dK
        arrows.append(
            Arrow(
                ds_node.get_bottom(),
                dk_node.get_top(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
            )
        )
        arrows.append(
            Arrow(
                q_node.get_bottom(),
                dk_node.get_top(),
                color=DARK_GRAY,
                stroke_width=2,
                buff=0.1,
            )
        )

        arrow_group = VGroup(*arrows)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.05))

        # Add legend
        legend = VGroup(
            VGroup(
                RoundedRectangle(
                    width=0.4, height=0.3, corner_radius=0.05, fill_opacity=0.3
                )
                .set_fill(COLORS["matmul_op"])
                .set_stroke(COLORS["matmul_op"]),
                Text("MatMul", font_size=18, color=COLORS["text"]),
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                RoundedRectangle(
                    width=0.4, height=0.3, corner_radius=0.05, fill_opacity=0.3
                )
                .set_fill(COLORS["elementwise_op"])
                .set_stroke(COLORS["elementwise_op"]),
                Text("Element-wise", font_size=18, color=COLORS["text"]),
            ).arrange(RIGHT, buff=0.2),
        )
        legend.arrange(RIGHT, buff=0.8)
        legend.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(legend))

        self.advance_slide()

        # Clean up
        self.play(
            FadeOut(title), FadeOut(all_nodes), FadeOut(arrow_group), FadeOut(legend)
        )

    # =========================================================================
    # Slide 3: Tiled Execution
    # =========================================================================
    def slide_3_tiled_execution(self):
        """Show how data is partitioned for tiled execution."""
        title = self.create_title("Tiled Backward Pass Execution")
        self.play(Write(title))

        # Create Q partitions
        q_label = Text("Q (partitioned by rows)", font_size=24, color=COLORS["text"])
        q_blocks = VGroup(
            *[PartitionBlock(f"Q{i}", COLORS["highlight"]["query"]) for i in range(4)]
        )
        q_blocks.arrange(DOWN, buff=0.1)
        q_group = VGroup(q_label, q_blocks).arrange(DOWN, buff=0.3)
        q_group.shift(LEFT * 4)

        # Create K, V partitions
        kv_label = Text(
            "K, V (partitioned by rows)", font_size=24, color=COLORS["text"]
        )
        k_blocks = VGroup(
            *[PartitionBlock(f"K{i}", COLORS["highlight"]["key"]) for i in range(4)]
        )
        v_blocks = VGroup(
            *[PartitionBlock(f"V{i}", COLORS["highlight"]["result"]) for i in range(4)]
        )
        k_blocks.arrange(DOWN, buff=0.1)
        v_blocks.arrange(DOWN, buff=0.1)
        kv_blocks = VGroup(k_blocks, v_blocks).arrange(RIGHT, buff=0.3)
        kv_group = VGroup(kv_label, kv_blocks).arrange(DOWN, buff=0.3)
        kv_group.next_to(q_group, RIGHT, buff=1.5)

        self.play(FadeIn(q_group), FadeIn(kv_group))

        self.advance_slide()

        # Show thread block assignment
        explanation = Text(
            "Each Q block is processed with ALL K/V blocks",
            font_size=26,
            color=COLORS["text"],
        )
        explanation.to_edge(DOWN, buff=1.5)
        self.play(Write(explanation))

        # Create thread blocks
        tb_label = Text("Thread Blocks", font_size=24, color=COLORS["text"])
        thread_blocks = VGroup(
            *[ThreadBlockViz(i, COLORS["thread_blocks"][i]) for i in range(3)]
        )
        thread_blocks.arrange(DOWN, buff=0.3)
        tb_group = VGroup(tb_label, thread_blocks).arrange(DOWN, buff=0.3)
        tb_group.next_to(kv_group, RIGHT, buff=1.5)

        self.play(
            LaggedStart(
                *[FadeIn(tb, shift=UP * 0.3) for tb in thread_blocks],
                lag_ratio=0.15,
            ),
            Write(tb_label),
        )

        # Show arrows from thread blocks to K/V blocks
        arrows = []
        for i, tb in enumerate(thread_blocks):
            if i < len(k_blocks):
                arrow = Arrow(
                    tb.get_left(),
                    k_blocks[i].get_right(),
                    color=COLORS["thread_blocks"][i],
                    stroke_width=2,
                    buff=0.1,
                )
                arrows.append(arrow)

        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1))

        tb_explanation = Text(
            "TB0, TB1, TB2 each handle different K/V blocks for same Q block",
            font_size=22,
            color=DARK_GRAY,
        )
        tb_explanation.next_to(explanation, DOWN, buff=0.3)
        self.play(Write(tb_explanation))

        self.advance_slide()

        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(q_group),
            FadeOut(kv_group),
            FadeOut(tb_group),
            FadeOut(VGroup(*arrows)),
            FadeOut(explanation),
            FadeOut(tb_explanation),
        )

    # =========================================================================
    # Slide 4: Race Condition (5-Dot Version)
    # =========================================================================
    def slide_4_race_condition(self):
        """Show the race condition in the 5-dot version."""
        title = self.create_title("The Race Condition (5-Dot Version)")
        self.play(Write(title))

        # Formula banner
        formula = MathTex(
            r"dQ = dS \cdot K",
            color=COLORS["text"],
            font_size=36,
        )
        formula.next_to(title, DOWN, buff=0.5)
        self.play(Write(formula))

        self.advance_slide()

        # Create thread blocks computing partial results
        tb_colors = COLORS["thread_blocks"]

        tb0 = ThreadBlockViz(0, tb_colors[0], width=2.0)
        tb1 = ThreadBlockViz(1, tb_colors[1], width=2.0)
        tb2 = ThreadBlockViz(2, tb_colors[2], width=2.0)
        thread_blocks = VGroup(tb0, tb1, tb2)
        thread_blocks.arrange(DOWN, buff=0.5)
        thread_blocks.shift(LEFT * 4)

        # Partial computations
        partial0 = MathTex(r"dS_0 \cdot K_0", font_size=24, color=tb_colors[0])
        partial1 = MathTex(r"dS_1 \cdot K_1", font_size=24, color=tb_colors[1])
        partial2 = MathTex(r"dS_2 \cdot K_2", font_size=24, color=tb_colors[2])
        partials = VGroup(partial0, partial1, partial2)

        for partial, tb in zip(partials, thread_blocks):
            partial.next_to(tb, RIGHT, buff=0.3)

        self.play(
            LaggedStart(*[FadeIn(tb) for tb in thread_blocks], lag_ratio=0.1),
            LaggedStart(*[Write(p) for p in partials], lag_ratio=0.1),
        )

        # Memory cell for dQ
        dq_cell = MemoryCell("dQ", width=2.0, height=1.0)
        dq_cell.shift(RIGHT * 2)
        dq_label = Text("Shared Memory", font_size=22, color=DARK_GRAY)
        dq_label.next_to(dq_cell, UP, buff=0.3)

        self.play(FadeIn(dq_cell), Write(dq_label))

        self.advance_slide()

        # Show atomic add arrows racing to the same cell
        arrows = []
        for i, (tb, partial) in enumerate(zip(thread_blocks, partials)):
            arrow = CurvedArrow(
                partial.get_right() + RIGHT * 0.1,
                dq_cell.get_left() + UP * (0.3 - i * 0.3),
                color=tb_colors[i],
                stroke_width=3,
                angle=TAU / 8 * (i - 1),
            )
            arrows.append(arrow)

        atomic_label = Text(
            "atomic_add(dQ, partial)",
            font_size=22,
            color=RED_D,
        )
        atomic_label.next_to(dq_cell, DOWN, buff=0.5)

        self.play(
            *[Create(a) for a in arrows],
            Write(atomic_label),
            run_time=1.0,
        )

        # Flash the cell to show collision
        self.play(
            Flash(dq_cell, color=RED, flash_radius=0.8, line_stroke_width=4),
        )

        self.advance_slide()

        # Show the non-determinism problem
        problem_title = Text(
            "Floating-Point Addition is NOT Associative!",
            font_size=28,
            color=RED_D,
        )
        problem_title.to_edge(DOWN, buff=2.0)
        self.play(Write(problem_title))

        # Show two orderings with different results
        order_a = MathTex(
            r"\text{Order A: } (a + b) + c = X",
            font_size=26,
            color=COLORS["text"],
        )
        order_b = MathTex(
            r"\text{Order B: } (c + a) + b = Y",
            font_size=26,
            color=COLORS["text"],
        )
        not_equal = MathTex(
            r"X \neq Y",
            font_size=32,
            color=RED_D,
        )

        ordering = VGroup(order_a, order_b, not_equal).arrange(DOWN, buff=0.3)
        ordering.next_to(problem_title, DOWN, buff=0.4)

        self.play(
            Write(order_a),
            Write(order_b),
        )
        self.play(Write(not_equal))

        self.advance_slide()

        # NON-DETERMINISTIC warning
        warning = Text(
            "NON-DETERMINISTIC",
            font_size=48,
            color=RED_D,
            weight=BOLD,
        )
        warning.move_to(ORIGIN)

        # Flash warning
        self.play(
            FadeOut(
                VGroup(
                    thread_blocks,
                    partials,
                    dq_cell,
                    dq_label,
                    VGroup(*arrows),
                    atomic_label,
                )
            ),
            FadeOut(problem_title),
            FadeOut(ordering),
        )
        self.play(
            Write(warning),
            Flash(warning, color=RED, flash_radius=1.5, line_stroke_width=5),
        )

        self.advance_slide()

        # Clean up
        self.play(FadeOut(title), FadeOut(formula), FadeOut(warning))

    # =========================================================================
    # Slide 5: Deterministic Solution (7-Dot Version)
    # =========================================================================
    def slide_5_deterministic_solution(self):
        """Show the deterministic 7-dot solution."""
        title = self.create_title("Deterministic Solution (7-Dot Version)")
        self.play(Write(title))

        # Key insight
        insight = Text(
            "Solution: Separate accumulation buffers + final reduction",
            font_size=26,
            color=COLORS["deterministic"],
        )
        insight.next_to(title, DOWN, buff=0.5)
        self.play(Write(insight))

        self.advance_slide()

        # Create thread blocks
        tb_colors = COLORS["thread_blocks"]
        tb0 = ThreadBlockViz(0, tb_colors[0], width=2.0)
        tb1 = ThreadBlockViz(1, tb_colors[1], width=2.0)
        tb2 = ThreadBlockViz(2, tb_colors[2], width=2.0)
        thread_blocks = VGroup(tb0, tb1, tb2)
        thread_blocks.arrange(DOWN, buff=0.4)
        thread_blocks.shift(LEFT * 4.5)

        # Separate buffers for each thread block
        buffer0 = MemoryCell("dQ_partial_0", width=2.2, height=0.7)
        buffer1 = MemoryCell("dQ_partial_1", width=2.2, height=0.7)
        buffer2 = MemoryCell("dQ_partial_2", width=2.2, height=0.7)
        buffers = VGroup(buffer0, buffer1, buffer2)
        buffers.arrange(DOWN, buff=0.4)
        buffers.shift(LEFT * 0.5)

        # Color the buffers
        buffer0.cell.set_stroke(tb_colors[0], width=2)
        buffer1.cell.set_stroke(tb_colors[1], width=2)
        buffer2.cell.set_stroke(tb_colors[2], width=2)

        self.play(
            LaggedStart(*[FadeIn(tb) for tb in thread_blocks], lag_ratio=0.1),
            LaggedStart(*[FadeIn(b) for b in buffers], lag_ratio=0.1),
        )

        # Arrows from thread blocks to their own buffers
        arrows_to_buffers = []
        for tb, buf, color in zip(thread_blocks, buffers, tb_colors):
            arrow = Arrow(
                tb.get_right(),
                buf.get_left(),
                color=color,
                stroke_width=3,
                buff=0.1,
            )
            arrows_to_buffers.append(arrow)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows_to_buffers], lag_ratio=0.1)
        )

        # Add "No Race!" labels
        no_race = Text("No Race!", font_size=20, color=COLORS["deterministic"])
        no_race.next_to(buffers, UP, buff=0.3)
        self.play(Write(no_race))

        self.advance_slide()

        # Final reduction
        final_dq = MemoryCell("dQ (final)", width=2.0, height=1.0)
        final_dq.shift(RIGHT * 3.5)
        final_dq.cell.set_stroke(COLORS["deterministic"], width=3)

        reduce_label = Text("reduce()", font_size=22, color=COLORS["deterministic"])
        reduce_label.next_to(final_dq, UP, buff=0.3)

        self.play(FadeIn(final_dq), Write(reduce_label))

        # Arrows from buffers to final
        arrows_to_final = []
        for buf in buffers:
            arrow = Arrow(
                buf.get_right(),
                final_dq.get_left(),
                color=COLORS["deterministic"],
                stroke_width=2,
                buff=0.1,
            )
            arrows_to_final.append(arrow)

        self.play(LaggedStart(*[GrowArrow(a) for a in arrows_to_final], lag_ratio=0.1))

        # Reduction formula
        reduction_formula = MathTex(
            r"dQ = \sum_{i} dQ\_partial_i",
            font_size=28,
            color=COLORS["text"],
        )
        reduction_formula.to_edge(DOWN, buff=1.5)
        self.play(Write(reduction_formula))

        self.advance_slide()

        # Explain the extra dot products
        extra_explanation = Text(
            "2 extra dot products come from split accumulation pattern",
            font_size=24,
            color=DARK_GRAY,
        )
        extra_explanation.next_to(reduction_formula, DOWN, buff=0.3)
        self.play(Write(extra_explanation))

        # DETERMINISTIC confirmation
        confirm = Text(
            "DETERMINISTIC",
            font_size=40,
            color=COLORS["deterministic"],
            weight=BOLD,
        )
        confirm.next_to(final_dq, DOWN, buff=0.8)

        self.play(
            Write(confirm),
            Flash(final_dq, color=GREEN, flash_radius=0.8, line_stroke_width=4),
        )

        self.advance_slide()

        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(insight),
            FadeOut(thread_blocks),
            FadeOut(buffers),
            FadeOut(VGroup(*arrows_to_buffers)),
            FadeOut(no_race),
            FadeOut(final_dq),
            FadeOut(reduce_label),
            FadeOut(VGroup(*arrows_to_final)),
            FadeOut(reduction_formula),
            FadeOut(extra_explanation),
            FadeOut(confirm),
        )

    # =========================================================================
    # Slide 6: Trade-off Summary
    # =========================================================================
    def slide_6_tradeoff_summary(self):
        """Show the trade-off summary comparison."""
        title = self.create_title("Trade-off Summary")
        self.play(Write(title))

        # Create comparison table
        # Headers
        header_row = VGroup(
            Text("", font_size=24),
            Text("5-Dot", font_size=28, color=COLORS["non_deterministic"], weight=BOLD),
            Text("7-Dot", font_size=28, color=COLORS["deterministic"], weight=BOLD),
        )
        header_row.arrange(RIGHT, buff=1.5)

        # Rows
        speed_row = VGroup(
            Text("Speed", font_size=24, color=COLORS["text"]),
            Text("Faster", font_size=24, color=COLORS["non_deterministic"]),
            Text("~40% slower", font_size=24, color=COLORS["deterministic"]),
        )
        speed_row.arrange(RIGHT, buff=1.5)

        memory_row = VGroup(
            Text("Memory", font_size=24, color=COLORS["text"]),
            Text("Less", font_size=24, color=COLORS["non_deterministic"]),
            Text("More (accum buffers)", font_size=24, color=COLORS["deterministic"]),
        )
        memory_row.arrange(RIGHT, buff=1.5)

        determinism_row = VGroup(
            Text("Determinism", font_size=24, color=COLORS["text"]),
            Text("No", font_size=28, color=RED_D, weight=BOLD),
            Text("Yes", font_size=28, color=GREEN_D, weight=BOLD),
        )
        determinism_row.arrange(RIGHT, buff=1.5)

        # Arrange all rows
        table = VGroup(header_row, speed_row, memory_row, determinism_row)
        table.arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        table.center()

        # Animate table
        self.play(
            LaggedStart(
                *[FadeIn(row, shift=RIGHT * 0.3) for row in table],
                lag_ratio=0.2,
            )
        )

        self.advance_slide()

        # Add conclusion
        conclusion = Text(
            "Choose based on your requirements:",
            font_size=28,
            color=COLORS["text"],
        )
        conclusion.to_edge(DOWN, buff=1.8)

        bullet1 = Text(
            "- Training with reproducibility needs? Use 7-dot",
            font_size=22,
            color=COLORS["deterministic"],
        )
        bullet2 = Text(
            "- Maximum throughput, non-critical? Use 5-dot",
            font_size=22,
            color=COLORS["non_deterministic"],
        )
        bullets = VGroup(bullet1, bullet2).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        bullets.next_to(conclusion, DOWN, buff=0.4)

        self.play(Write(conclusion))
        self.play(Write(bullet1), Write(bullet2))

        self.advance_slide()

        # Final message
        final_msg = Text(
            "Determinism has a cost, but sometimes it's worth paying!",
            font_size=26,
            color=COLORS["text"],
        )
        final_msg.to_edge(DOWN, buff=0.5)

        self.play(
            FadeOut(conclusion),
            FadeOut(bullets),
            Write(final_msg),
        )

        self.advance_slide()

    # =========================================================================
    # Main construct
    # =========================================================================
    def construct(self):
        """Main construct method that orchestrates the visualization."""
        self.slide_1_backward_overview()
        self.slide_2_computational_graph()
        self.slide_3_tiled_execution()
        self.slide_4_race_condition()
        self.slide_5_deterministic_solution()
        self.slide_6_tradeoff_summary()
