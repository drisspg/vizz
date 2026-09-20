from manim import DOWN, LEFT, PI, RIGHT, UP, Create, FadeIn, MathTex, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.tensor_grid import TensorGrid


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = scene.section_header("Follow one score")
    grid = TensorGrid(8, 8, t, cell_size=0.48).causal_mask()
    grid.move_to(LEFT * 3 + DOWN * 0.2)
    indices = VGroup()
    for index in range(8):
        indices.add(
            scene.meta_text(str(index), font_size=16).next_to(
                grid.cell(index, 0), LEFT, buff=0.1
            ),
            scene.meta_text(str(index), font_size=16).next_to(
                grid.cell(0, index), UP, buff=0.1
            ),
        )
    key_label = scene.meta_text("key j →", uppercase=False).next_to(grid, UP, buff=0.45)
    query_label = (
        scene.meta_text("query i", uppercase=False)
        .next_to(grid, LEFT, buff=0.2)
        .rotate(PI / 2)
    )
    # Keep the focus independent of the logical mask and the cell's value.
    selection = grid.region(5, 6, 2, 3, color=t.accent_secondary)
    selection.set_stroke(width=3)
    entry = MathTex(r"s_{5,2}", color=t.accent_secondary, font_size=52).move_to(
        RIGHT * 3 + UP * 1.2
    )
    label = scene.body_text("query 5 · key 2", font_size=24).next_to(
        entry, DOWN, buff=0.2
    )
    caption = scene.body_text(
        "One selected entry. Three views of the same computation.", font_size=23
    ).move_to(DOWN * 3.05)

    scene.play(
        FadeIn(header),
        FadeIn(grid),
        FadeIn(indices),
        FadeIn(key_label),
        FadeIn(query_label),
        FadeIn(caption),
    )
    scene.play(Create(selection), FadeIn(entry), FadeIn(label), run_time=0.5)
    scene.wait(0.2)
    scene.next_slide(
        notes="focus.select — Follow the retained query-5/key-2 entry. Hatching denotes masked future keys."
    )

    operands = (
        VGroup(
            MathTex(r"q_5=(1,2)", color=t.text, font_size=34),
            MathTex(r"k_2=(3,-1)", color=t.text, font_size=34),
        )
        .arrange(DOWN, buff=0.24)
        .move_to(RIGHT * 3 + DOWN * 0.3)
    )
    scene.play(FadeIn(operands, shift=UP * 0.12), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="focus.operands — Reveal the two-channel query and key without changing the selected entry."
    )

    result = MathTex(r"1\cdot3+2\cdot(-1)=1", color=t.text, font_size=36).move_to(
        RIGHT * 3 + DOWN * 1.6
    )
    grid.set_value(5, 2, "1")
    scene.play(FadeIn(result), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="focus.result — The illustrative unscaled dot product is 1. This is not softmax or a full attention output."
    )
