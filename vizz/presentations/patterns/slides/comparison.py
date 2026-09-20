from manim import DOWN, Create, FadeIn, ReplacementTransform

from vizz.presentations.components import SlideBase
from vizz.presentations.layouts import Comparison
from vizz.presentations.tensor_grid import CellState, TensorGrid


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = scene.section_header("Change one rule, not the picture")
    left = TensorGrid(8, 8, t, cell_size=0.43).causal_mask()
    right = TensorGrid(8, 8, t, cell_size=0.43)
    for row in range(8):
        for col in range(8):
            right.set_state(row, col, CellState.INACTIVE)
    layout = Comparison(
        left, right, labels=("CAUSAL", "THREE-TOKEN WINDOW"), theme=t, height=3.5
    ).shift(DOWN * 0.3)
    caption = scene.body_text(
        "Same query. Same geometry. Different allowed history.", font_size=24
    ).move_to(DOWN * 3.05)
    scene.play(
        FadeIn(header),
        FadeIn(left),
        FadeIn(right),
        FadeIn(layout.labels),
        Create(layout.divider),
        FadeIn(caption),
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="comparison.baseline — The candidate is an inactive scaffold, not an all-zero tensor. Both grids use the same scale."
    )

    candidate = TensorGrid(8, 8, t, cell_size=0.43)
    for row in range(8):
        for col in range(8):
            candidate.set_state(
                row, col, CellState.RETAINED if 0 <= row - col < 3 else CellState.MASKED
            )
    candidate.scale_to_fit_width(right.width).move_to(right)
    scene.play(ReplacementTransform(right, candidate), run_time=0.8)
    scene.wait(0.2)
    scene.next_slide(
        notes="comparison.candidate — The window retains the current token and at most two preceding tokens; everything else is excluded."
    )

    left_focus = left.region(5, 6, 0, 8, color=t.accent_secondary).set_stroke(width=3)
    right_focus = candidate.region(5, 6, 0, 8, color=t.accent_secondary).set_stroke(
        width=3
    )
    left_note = scene.body_text("query 5 → keys 0–5", font_size=22).next_to(
        left, DOWN, buff=0.3
    )
    right_note = scene.body_text("query 5 → keys 3–5", font_size=22).next_to(
        candidate, DOWN, buff=0.3
    )
    scene.play(
        Create(left_focus),
        Create(right_focus),
        FadeIn(left_note),
        FadeIn(right_note),
        run_time=0.6,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="comparison.difference — Hold query 5 fixed. Causal support retains six keys; the three-token window retains three. No performance claim is implied."
    )
