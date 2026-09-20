# Nuggets visual patterns

A small, copyable gallery of technical explanations in the KDA blog's visual
language. These are editable Manim objects, not screenshots of browser widgets.

## Browse or copy the gallery

```bash
uv run vizz preview patterns --theme light
uv run vizz preview patterns --theme dark
open media/review/patterns/all/dark/frames/index.html
```

Each theme produces nine pause-state PNGs with named speaker notes. For focused
iteration, use `uv run vizz preview patterns --slide tiles --theme dark`.

Start a project with all three examples:

```bash
uv run vizz new my_talk --template patterns
uv run vizz preview my_talk --slide focus --theme dark
```

Keep only the examples you need in `build.py`'s `SLIDES` registry; edit the copied
`brief.md` and `scenes.md` before replacing the story. `--template starter` remains
the default and preserves the original title/flow starter.

| Slide key | Recipe | Good uses |
| --- | --- | --- |
| `focus` | Keep one selected cell while revealing its operands and result | Attention scores, recurrence terms, rounding explanations |
| `tiles` | Move a rectangular footprint independently of the logical mask | Tiled work, quantization groups, attention chunks |
| `comparison` | Same geometry and scale; change only the rule | Full vs windowed support, baseline vs candidate |

The examples use a two-channel unscaled dot product and schematic grid sizes.
They make no claims about KDA measurements or valid hardware instruction shapes.

## TensorGrid

In a slide's `build(scene)` function:

```python
from manim import Create, FadeIn, RIGHT
from vizz.presentations.tensor_grid import CellState, TensorGrid

matrix = TensorGrid(8, 8, scene.theme, cell_size=0.5).causal_mask()
matrix.set_value(5, 2, "0")
matrix.shift(RIGHT)
focus = matrix.region(5, 6, 2, 3, color=scene.theme.accent_secondary)
scene.play(FadeIn(matrix), Create(focus))
scene.wait(0.2)
scene.next_slide(notes="select — A zero value can still be selected and retained.")
```

- Row 0 is at the top, column 0 at the left; `cell(row, col)` addresses a cell.
- `set_value` accepts literal text. `"0"` is visible; empty text clears the label.
- `CellState.RETAINED` shows content with a subtle green fill.
- `CellState.INACTIVE` mutes content: useful for an unrevealed scaffold, not zero.
- `CellState.MASKED` hatches the cell and hides—but preserves—its value. Unmasking
  restores the value. Selection is a separate outline and never changes state.
- `causal_mask()` marks `col > row` masked and resets all other cells to retained.
- `region(r0, r1, c0, c1, color=...)` uses **half-open bounds**. It returns a
  detached, unfilled rectangle and does not change values or states.
- `tile_lines(row_step, col_step)` returns internal boundaries; partial edge tiles
  are allowed.

Apply `set_state`, `set_value`, and `causal_mask` directly between beats. To
animate a semantic change, build the target grid and use `ReplacementTransform`
as in `comparison.py`; do not use `.animate.set_state(...)` to transfer Python-side
state. Ordinary whole-grid position/scale animations remain available.

Overlays follow geometry **at creation time**, including a scaled/rotated grid.
Position the grid first, then create overlays. To move them together later,
wrap grid and overlays in a `VGroup`; they are not live updaters. Apply state
changes to the actual grid, not a stale source object after `ReplacementTransform`.

## Controlled comparison

`vizz.presentations.layouts.Comparison` positions two input objects in place:

```python
from vizz.presentations.layouts import Comparison

layout = Comparison(
    baseline,
    candidate,
    labels=("BASELINE", "CANDIDATE"),
    theme=scene.theme,
)
```

It applies **one common scale factor**, never independently fits the two sides,
and never scales up. It exposes `left`, `right`, `labels`, and `divider` for
separate reveals. Inputs can include images because the layout is a `Group`.
Use the same grid dimensions and cell sizes when the comparison requires exact
geometric equivalence. Create focus overlays after layout.

## Theme and visual contract

`NUGGETS_LIGHT_THEME` / `NUGGETS_DARK_THEME` live in `theme.py`. A deck can select
one as its class `theme`; CLI `--theme light|dark` overrides only that scene
instance. Omit the option (or use `--theme deck`) to preserve a deck's chosen style.
Old PTCE and starter themes are unchanged.

- IBM Plex Sans for prose/headings; IBM Plex Mono for indices and metadata.
- Warm paper / near-black backgrounds; restrained borders and a quiet divider.
- Amber is selection/reference, green is retained support, hatching means exclusion.
  Red is available for danger/difference, but is not an automatic "bad" label.
- Stable selection across beats; no shifting scales to exaggerate a difference.
- Small, named steps in `next_slide(notes=...)`, not a universal animation DSL.

OFL-licensed fonts are bundled under `vizz/presentations/fonts/` with source and
checksums. `SlideBase` temporarily registers them during scene construction;
there is no system installation or render-time download. If building standalone
mobjects outside `SlideBase.construct`, use Manim's `register_font` context for
those files. The `focus` example also needs LaTeX/dvisvgm for `MathTex`.

Explicit themes get separate preview directories:
`media/review/<deck>/<slide-or-all>/<light-or-dark>/frames/`.
Full `render` still updates the normal scene metadata in `slides/`: presenting
and exporting use the most recent full render, regardless of theme.

```bash
uv run vizz render patterns --theme dark --quality h
uv run manim-slides present PatternGallery
uv run manim-slides convert PatternGallery media/patterns/PatternGallery.pptx
```
