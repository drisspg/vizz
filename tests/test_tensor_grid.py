import numpy as np
import pytest
from manim import ManimColor, Rectangle, Text, VGroup, tempconfig

from vizz.presentations.tensor_grid import CellState, TensorGrid
from vizz.presentations.theme import Theme


@pytest.fixture
def theme(tmp_path):
    with tempconfig({"media_dir": str(tmp_path)}):
        yield Theme()


@pytest.fixture
def grid(theme):
    return TensorGrid(3, 4, theme)


def test_cell_positions_and_stable_groups(grid):
    assert isinstance(grid, VGroup)
    assert len(grid) == 12
    assert grid.width == pytest.approx(2)
    assert grid.height == pytest.approx(1.5)
    for row in range(3):
        for col in range(4):
            cell = grid.cell(row, col)
            assert cell is grid.cell(row, col)
            assert isinstance(cell, VGroup)
            assert isinstance(cell[0], Rectangle)
            assert isinstance(cell[1], VGroup)
            assert isinstance(cell[2], VGroup)
            np.testing.assert_allclose(
                cell.get_center(), [(col - 1.5) / 2, (1 - row) / 2, 0]
            )
            assert cell[0].get_fill_color() == ManimColor(grid.theme.accent_success)
            assert 0 < cell[0].get_fill_opacity() < 0.3


def test_repeated_states_preserve_cell_frame_content_and_do_not_stack_hatches(grid):
    cell = grid.cell(0, 0)
    frame, content, hatches = cell
    grid.set_value(0, 0, "7")
    label = content[0]
    for _ in range(3):
        for state in (
            CellState.MASKED,
            CellState.MASKED,
            CellState.INACTIVE,
            CellState.RETAINED,
        ):
            assert grid.set_state(0, 0, state) is grid
            assert grid.cell(0, 0) is cell
            assert list(cell) == [frame, content, hatches]
            assert content[0] is label
            assert len(hatches) == (7 if state is CellState.MASKED else 0)
    assert label.get_fill_opacity() == 1
    assert frame.get_stroke_opacity() == 1
    assert frame.get_fill_color() == ManimColor(grid.theme.accent_success)


def test_inactive_is_distinct_without_masking_value(grid):
    grid.set_value(0, 0, "2").set_state(0, 0, CellState.INACTIVE)
    frame, content, hatches = grid.cell(0, 0)
    assert frame.get_fill_color() == ManimColor(grid.theme.panel_fill)
    assert frame.get_stroke_opacity() < 1
    assert content[0].get_fill_color() == ManimColor(grid.theme.muted_text)
    assert 0 < content[0].get_fill_opacity() < 1
    assert not len(hatches)
    grid.set_value(0, 0, "3")
    assert 0 < content[0].get_fill_opacity() < 1


def test_zero_is_a_visible_value_and_masking_preserves_it(grid):
    assert grid.set_value(0, 0, "0") is grid
    _, content, hatches = grid.cell(0, 0)
    label = content[0]
    assert isinstance(label, Text)
    assert label.text == "0"
    assert label.font == grid.theme.mono_font
    assert label.width > 0 and label.height > 0
    assert label.get_fill_opacity() == 1
    assert len(hatches) == 0
    grid.set_state(0, 0, CellState.MASKED)
    assert content[0] is label
    assert label.text == "0"
    assert label.get_fill_opacity() == 0
    assert len(hatches) > 0
    grid.set_value(0, 0, "0")
    assert content[0].get_fill_opacity() == 0
    grid.set_state(0, 0, CellState.RETAINED)
    assert content[0].text == "0"
    assert content[0].get_fill_opacity() == 1
    assert len(hatches) == 0


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_empty_value_clears_content_not_state(grid, text):
    grid.set_state(0, 0, CellState.MASKED).set_value(0, 0, "8")
    grid.set_value(0, 0, text)
    assert len(grid.cell(0, 0)[1]) == 0
    assert len(grid.cell(0, 0)[2]) == 7


@pytest.mark.parametrize(
    "rows,cols,masked_count", [(3, 3, 3), (2, 4, 5), (4, 2, 1), (1, 1, 0)]
)
def test_causal_mask_counts_and_resets_other_states(theme, rows, cols, masked_count):
    grid = TensorGrid(rows, cols, theme)
    for row in range(rows):
        for col in range(cols):
            grid.set_state(row, col, CellState.INACTIVE)
    assert grid.causal_mask() is grid
    assert sum(bool(len(cell[2])) for cell in grid) == masked_count
    for row in range(rows):
        for col in range(cols):
            frame, _, hatches = grid.cell(row, col)
            assert bool(len(hatches)) is (col > row)
            if col <= row:
                assert frame.get_fill_color() == ManimColor(theme.accent_success)
                assert frame.get_stroke_opacity() == 1


@pytest.mark.parametrize("mask_before_transform", [False, True])
def test_hatches_are_diagonal_clipped_and_follow_transforms(
    grid, mask_before_transform
):
    if mask_before_transform:
        grid.set_state(1, 2, CellState.MASKED)
    matrix = np.array([[1.2, -0.4, 0], [0.5, 0.8, 0], [0, 0, 1]])
    grid.apply_matrix(matrix).shift([2, -1, 0])
    if not mask_before_transform:
        grid.set_state(1, 2, CellState.MASKED)
    frame, _, hatches = grid.cell(1, 2)
    ur, ul, dl, _ = frame.get_vertices()
    basis = np.column_stack((ur - ul, ul - dl))
    assert len(hatches) == 7
    starts = []
    for line in hatches:
        local = np.linalg.lstsq(basis, (line.get_all_points() - dl).T, rcond=None)[0].T
        assert np.all(local >= 0.06 - 1e-10)
        assert np.all(local <= 0.94 + 1e-10)
        start = np.linalg.lstsq(basis, line.get_start() - dl, rcond=None)[0]
        end = np.linalg.lstsq(basis, line.get_end() - dl, rcond=None)[0]
        assert end[0] > start[0]
        np.testing.assert_allclose(end[0] - start[0], end[1] - start[1])
        starts.append(tuple(np.round(start, 8)))
    assert len(set(starts)) == 7


def test_value_fits_transformed_cell(grid):
    grid.stretch(1.4, 0).scale(1.7).rotate(np.pi / 5).shift([2, -1, 0])
    grid.set_value(1, 2, "123456")
    frame, content, _ = grid.cell(1, 2)
    np.testing.assert_allclose(content.get_center(), frame.get_center())
    ur, ul, dl, _ = frame.get_vertices()
    basis = np.column_stack((ur - ul, ul - dl))
    local = np.linalg.lstsq(basis, (content.get_all_points() - dl).T, rcond=None)[0]
    assert np.all(local > 0)
    assert np.all(local < 1)


def test_region_uses_transformed_half_open_bounds(grid):
    matrix = np.array([[1.2, -0.4, 0], [0.5, 0.8, 0], [0, 0, 1]])
    shift = np.array([2, -1, 0])
    grid.apply_matrix(matrix).shift(shift)
    region = grid.region(1, 3, 1, 4, color=grid.theme.accent_secondary)
    assert isinstance(region, Rectangle)
    assert region.get_fill_opacity() == 0
    assert region.get_stroke_color() == ManimColor(grid.theme.accent_secondary)
    original_corners = np.array(
        [[1, 0.25, 0], [-0.5, 0.25, 0], [-0.5, -0.75, 0], [1, -0.75, 0]]
    )
    np.testing.assert_allclose(
        region.get_vertices(), original_corners @ matrix.T + shift
    )
    single = grid.region(0, 1, 0, 1, color=grid.theme.accent_secondary)
    np.testing.assert_allclose(single.get_vertices(), grid.cell(0, 0)[0].get_vertices())


def test_tile_lines_use_current_geometry_and_allow_partial_tiles(theme):
    grid = TensorGrid(5, 7, theme, cell_size=1)
    grid.scale(1.5).shift([2, -1, 0])
    lines = grid.tile_lines(2, 3)
    assert isinstance(lines, VGroup)
    assert len(lines) == 4
    original_endpoints = np.array(
        [
            [[-3.5, 0.5, 0], [3.5, 0.5, 0]],
            [[-3.5, -1.5, 0], [3.5, -1.5, 0]],
            [[-0.5, 2.5, 0], [-0.5, -2.5, 0]],
            [[2.5, 2.5, 0], [2.5, -2.5, 0]],
        ]
    )
    for line, endpoints in zip(lines, original_endpoints, strict=True):
        np.testing.assert_allclose(
            [line.get_start(), line.get_end()], endpoints * 1.5 + [2, -1, 0]
        )
    assert len(grid.tile_lines(5, 7)) == 0
    assert len(grid.tile_lines(9, 9)) == 0
    assert len(grid.tile_lines(1, 1)) == 10


def test_overlays_do_not_mutate_or_join_grid(grid):
    grid.set_value(0, 2, "0").causal_mask()
    family = grid.get_family()
    before = [
        (mob.points.copy(), mob.fill_rgbas.copy(), mob.stroke_rgbas.copy())
        for mob in family
    ]
    region = grid.region(0, 2, 0, 3, color=grid.theme.accent_secondary)
    lines = grid.tile_lines(2, 2)
    region.set_color("#ff0000").shift([4, 3, 0])
    lines.scale(2)
    assert grid.get_family() == family
    for mob, (points, fill, stroke) in zip(family, before, strict=True):
        np.testing.assert_array_equal(mob.points, points)
        np.testing.assert_array_equal(mob.fill_rgbas, fill)
        np.testing.assert_array_equal(mob.stroke_rgbas, stroke)
    grid.set_value(0, 2, "9")
    assert grid.cell(0, 2)[1][0].get_fill_opacity() == 0
    assert len(grid.cell(0, 2)[2]) == 7


@pytest.mark.parametrize("dimension", ["rows", "cols"])
@pytest.mark.parametrize(
    "value,error",
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (2.0, TypeError),
        ("2", TypeError),
    ],
)
def test_invalid_dimensions(theme, dimension, value, error):
    kwargs = {"rows": 2, "cols": 2, "theme": theme, dimension: value}
    with pytest.raises(error, match=dimension):
        TensorGrid(**kwargs)


@pytest.mark.parametrize(
    "value,error",
    [
        (0, ValueError),
        (-0.5, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("0.5", TypeError),
    ],
)
def test_invalid_cell_size(theme, value, error):
    with pytest.raises(error, match="cell_size"):
        TensorGrid(1, 1, theme, cell_size=value)


@pytest.mark.parametrize(
    "row,col,error",
    [
        (-1, 0, IndexError),
        (3, 0, IndexError),
        (0, -1, IndexError),
        (0, 4, IndexError),
        (True, 0, TypeError),
        (0, 1.0, TypeError),
    ],
)
def test_invalid_cell_indices_rejected_by_all_cell_operations(grid, row, col, error):
    with pytest.raises(error):
        grid.cell(row, col)
    with pytest.raises(error):
        grid.set_state(row, col, CellState.MASKED)
    with pytest.raises(error):
        grid.set_value(row, col, "0")


@pytest.mark.parametrize(
    "bounds,error",
    [
        ((0, 0, 0, 1), ValueError),
        ((1, 0, 0, 1), ValueError),
        ((0, 1, 2, 2), ValueError),
        ((0, 1, 3, 2), ValueError),
        ((-1, 1, 0, 1), IndexError),
        ((0, 4, 0, 1), IndexError),
        ((0, 1, -1, 1), IndexError),
        ((0, 1, 0, 5), IndexError),
        ((True, 1, 0, 1), TypeError),
        ((0, 1.0, 0, 1), TypeError),
        ((0, 1, 0.0, 1), TypeError),
        ((0, 1, 0, True), TypeError),
    ],
)
def test_invalid_regions(grid, bounds, error):
    with pytest.raises(error):
        grid.region(*bounds, color=grid.theme.accent_secondary)


@pytest.mark.parametrize(
    "row_step,col_step,error",
    [
        (0, 1, ValueError),
        (1, -1, ValueError),
        (True, 1, TypeError),
        (1, 2.0, TypeError),
    ],
)
def test_invalid_tile_steps(grid, row_step, col_step, error):
    with pytest.raises(error):
        grid.tile_lines(row_step, col_step)


def test_invalid_state_and_value_leave_cell_unchanged(grid):
    grid.set_value(0, 0, "0").set_state(0, 0, CellState.MASKED)
    before = grid.cell(0, 0).get_all_points().copy()
    with pytest.raises(TypeError, match="CellState"):
        grid.set_state(0, 0, "retained")
    with pytest.raises(TypeError, match="string"):
        grid.set_value(0, 0, 0)
    np.testing.assert_array_equal(grid.cell(0, 0).get_all_points(), before)
    assert grid.cell(0, 0)[1][0].text == "0"
    assert grid.cell(0, 0)[1][0].get_fill_opacity() == 0
