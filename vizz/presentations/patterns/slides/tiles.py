from manim import DOWN, LEFT, PI, RIGHT, UP, Brace, Create, FadeIn, Transform, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.tensor_grid import CellState, TensorGrid


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = scene.section_header("A footprint is not a mask")
    grid = TensorGrid(16, 16, t, cell_size=0.27).causal_mask()
    grid.move_to(LEFT * 2.9 + DOWN * 0.25)
    boundaries = grid.tile_lines(4, 4)
    strip = grid.region(0, 16, 0, 4, color=t.accent_secondary).set_stroke(width=3)
    brace = Brace(grid, LEFT, color=t.muted_text)
    rows = (
        scene.meta_text("16 rows", uppercase=False)
        .rotate(PI / 2)
        .next_to(brace, LEFT, buff=0.12)
    )
    keys = scene.meta_text(
        "keys 0–3", color=t.accent_secondary, uppercase=False
    ).next_to(strip, UP, buff=0.24)
    explanation = (
        VGroup(
            scene.body_text("16 rows × 4 columns", font_size=29),
            scene.body_text(
                "Amber: selected footprint", font_size=24, color=t.accent_secondary
            ),
            scene.body_text(
                "Green: retained entries", font_size=24, color=t.accent_success
            ),
            scene.body_text(
                "Hatched: excluded entries", font_size=24, color=t.muted_text
            ),
        )
        .arrange(DOWN, aligned_edge=LEFT, buff=0.24)
        .move_to(RIGHT * 3.2 + UP * 0.8)
    )
    caption = scene.body_text(
        "Schematic dimensions — not a hardware instruction specification.", font_size=21
    ).move_to(DOWN * 3.05)

    scene.play(
        FadeIn(header),
        FadeIn(grid),
        Create(boundaries),
        FadeIn(brace),
        FadeIn(rows),
        FadeIn(caption),
    )
    scene.play(Create(strip), FadeIn(keys), FadeIn(explanation), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="tiles.footprint — The full-height footprint includes both retained and excluded cells. These are schematic dimensions."
    )

    next_strip = grid.region(0, 16, 4, 8, color=t.accent_secondary).set_stroke(width=3)
    next_keys = scene.meta_text(
        "keys 4–7", color=t.accent_secondary, uppercase=False
    ).next_to(next_strip, UP, buff=0.24)
    scene.play(Transform(strip, next_strip), Transform(keys, next_keys), run_time=0.8)
    scene.wait(0.2)
    scene.next_slide(
        notes="tiles.advance — Move the footprint; the logical mask and cell dimensions stay fixed."
    )

    examples = VGroup()
    for index, (state, value, label) in enumerate(
        [
            (CellState.RETAINED, "0", "zero"),
            (CellState.INACTIVE, "", "inactive"),
            (CellState.MASKED, "", "masked"),
        ]
    ):
        cell = (
            TensorGrid(1, 1, t, cell_size=0.58)
            .set_value(0, 0, value)
            .set_state(0, 0, state)
        )
        cell.move_to(RIGHT * (1.75 + index * 1.45) + DOWN * 1.3)
        label_text = scene.meta_text(label, font_size=17, uppercase=False).next_to(
            cell, DOWN, buff=0.18
        )
        examples.add(VGroup(cell, label_text))
    masked = examples[2][0]
    overlay = masked.region(0, 1, 0, 1, color=t.accent_secondary).set_stroke(width=3)
    note = scene.body_text("Selection does not unmask.", font_size=22).move_to(
        RIGHT * 3.2 + DOWN * 2.3
    )
    scene.play(FadeIn(examples), Create(overlay), FadeIn(note), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="tiles.detail — Zero is a displayed value, inactive is a reveal state, masked is logical exclusion, and selection is an independent outline."
    )
