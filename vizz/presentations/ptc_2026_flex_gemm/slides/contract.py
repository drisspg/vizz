from manim import DOWN, LEFT, RIGHT, UP, Brace, Create, DashedVMobject, FadeIn, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header
from vizz.presentations.tensor_grid import TensorGrid

ROWS, COLS = 8, 16
TILE_M, TILE_N, GROUP = 4, 8, 4


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(
        scene, "The contract: fuse what is tile-local, fail closed otherwise"
    )

    grid = TensorGrid(ROWS, COLS, t, cell_size=0.34)
    grid.move_to(LEFT * 3.3 + DOWN * 0.55)
    tiles = grid.tile_lines(TILE_M, TILE_N).set_stroke(color=t.muted_text, width=1.6)
    cols_title = scene.meta_text("accumulator · n →", font_size=13)
    cols_title.next_to(grid, UP, buff=0.2).align_to(grid, LEFT)
    rows_title = (
        scene.meta_text("m ↓", font_size=13)
        .next_to(grid, LEFT, buff=0.2)
        .align_to(grid, UP)
    )
    grid_label = VGroup(cols_title, rows_title)
    tile = grid.region(
        0, TILE_M, TILE_N, 2 * TILE_N, color=t.accent_secondary
    ).set_stroke(width=3)
    bracket = Brace(tile, UP, buff=0.06, color=t.accent_secondary, sharpness=3)
    bracket_label = scene.meta_text(
        "tile_n", font_size=13, color=t.accent_secondary
    ).next_to(bracket, UP, buff=0.06)
    tile_label = scene.meta_text(
        "one output tile", uppercase=False, color=t.accent_secondary
    )
    tile_label.next_to(grid, DOWN, buff=0.2).align_to(grid, LEFT)

    scene.play(FadeIn(header), FadeIn(grid), Create(tiles), FadeIn(grid_label))
    scene.play(
        Create(tile),
        FadeIn(bracket),
        FadeIn(bracket_label),
        FadeIn(tile_label),
        run_time=0.5,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="contract.tile — The kernel owns one output tile at a time. The epilogue runs on that tile in registers before the store."
    )

    group = grid.region(
        1, 2, TILE_N, TILE_N + GROUP, color=t.accent_success
    ).set_stroke(width=4)
    group_label = (
        scene.meta_text(
            "N-group reduction: fits in the tile",
            uppercase=False,
            color=t.accent_success,
        )
        .next_to(tile_label, DOWN, buff=0.1)
        .align_to(grid, LEFT)
    )
    row = DashedVMobject(
        grid.region(6, 7, 0, COLS, color=t.accent_danger).set_stroke(width=3),
        num_dashes=60,
    )
    row_label = (
        scene.meta_text(
            "full-row reduction: crosses tiles", uppercase=False, color=t.accent_danger
        )
        .next_to(group_label, DOWN, buff=0.1)
        .align_to(grid, LEFT)
    )

    def column(title: str, color: str, items: list[str]) -> VGroup:
        head = scene.meta_text(title, font_size=17, color=color)
        body = scene.bullet_list(*items, font_size=19)
        return VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

    fuses = column(
        "fuses",
        t.accent_success,
        [
            "pointwise math, activations, casts",
            "captured loads: bias, residual, scales",
            "aux outputs from the same tile",
            "lane-local contraction: SwiGLU  M×2N → M×N",
            "bounded N-group reductions: fp8/fp4 scales",
        ],
    )
    rejects = column(
        "rejected at compile time",
        t.accent_danger,
        [
            "full-row reductions: LayerNorm, softmax",
            "nonlocal layout transforms",
            "nonlinear epilogues on split-K partials",
        ],
    )
    lists = VGroup(fuses, rejects).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
    lists.next_to(grid, RIGHT, buff=0.7).align_to(grid_label, UP)

    scene.play(Create(group), FadeIn(group_label), FadeIn(fuses), run_time=0.7)
    scene.wait(0.2)
    scene.next_slide(
        notes="contract.fuses — Everything that needs only this tile, plus captured inputs, fuses: pointwise, loads, aux stores, SwiGLU gate/up lane pairs, and reductions over a bounded group of N columns like MX/NV scale blocks."
    )

    scene.play(Create(row), FadeIn(row_label), FadeIn(rejects), run_time=0.7)
    scene.wait(0.2)
    scene.next_slide(
        notes="contract.rejects — Fail closed: if the epilogue needs data from other tiles or would change numerics under split-K, you get an error, not a silently slower or different kernel. That guarantee is what makes quantized producers safe."
    )
