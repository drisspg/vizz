# Vizz

Collection of math visualizations using Manim and Manim-Slides for interactive presentations.

## Start your next presentation

```bash
uv sync
uv run vizz new my_talk
uv run vizz preview my_talk --slide workflow
open media/review/my_talk/workflow/frames/index.html
```

Edit `vizz/presentations/my_talk/brief.md` and `scenes.md` first, then the modules
under `slides/`. Each slide has a stable name in `build.py`'s `SLIDES` registry.
The preview command renders only that slide, skips reverse-video generation, and
saves a PNG for every pause plus an HTML gallery. Preview files are isolated from
`slides/`, so trying a single slide does not replace your presentable full deck.
Watch the generated video as well when reviewing motion.

```bash
uv run vizz render my_talk
uv run manim-slides present MyTalkDeck
uv run vizz render my_talk --quality h
uv run manim-slides convert MyTalkDeck my_talk.pptx
```

The `.pptx` opens in Keynote, but the visuals are rendered media, not editable
Manim shapes. Check playback in your target app before presenting. To share in a
browser, use `uv run manim-slides convert MyTalkDeck my_talk.html` and keep its
companion assets together.

### Nuggets-style technical patterns

The [pattern gallery](docs/patterns.md) includes a selected-cell walkthrough,
tiled matrix, and controlled comparison, in the KDA blog's light/dark visual
language. IBM Plex fonts are bundled under their OFL license—no font installation
or render-time download required. The focus example uses LaTeX.

```bash
uv run vizz preview patterns --theme dark
open media/review/patterns/all/dark/frames/index.html
uv run vizz new my_technical_talk --template patterns
```

Use `--slide focus`, `--slide tiles`, or `--slide comparison` for a single example.
Omit `--slide` to review all slides; use `--theme light` for the paper palette.
Without `--theme`, each deck keeps its existing style. Explicit themes have
separate preview directories so light/dark reviews do not overwrite each other.

### Pairing from an Excalidraw sketch

**Yes: a rough drawing is a useful starting point.** Save the editable
`.excalidraw` file **and a readable PNG export** in your deck's `sketches/` folder
(or give the agent a screenshot). Add what the arrows/colors mean, the exact
labels, and numbered reveal steps. The source preserves editable geometry; the
PNG makes visual review reliable without depending on a particular editor.

The workflow is **sketch → slide plan → Manim objects → rendered review**.
This is an agent-assisted redraw, not a lossless automatic Excalidraw importer.
Use a static PNG when the drawing does not need animation; rebuild meaningful
objects when they need independent reveals, highlights, or transformations.
See the [authoring guide](docs/authoring.md) and the generated `brief.md` for a
copyable handoff prompt.

## Project Structure

```
vizz/
├── cli.py              # new / preview / render
├── presentations/
│   ├── theme.py        # Shared color and typography tokens
│   ├── components.py   # SlideBase, panels, code cards, headings
│   ├── tensor_grid.py  # Values, masks, named cell/region addressing
│   ├── layouts.py      # Same-scale comparison layout
│   ├── patterns/       # Copyable Nuggets-style technical gallery
│   ├── starter/        # Runnable, copyable two-slide example
│   └── <your_deck>/    # brief.md, scenes.md, sketches/, slides/, build.py
├── flex/               # Animations for FlexAttention
└── quant/              # Quant animations
```

## Setup

1. Install macOS system dependencies:
```bash
brew install cairo pango pkg-config ffmpeg
# Optional: only needed for Tex/MathTex scenes, not the starter deck.
brew install --cask mactex
```

2. Sync the project environment with `uv`:
```bash
uv sync
```

This project is meant to be run with `uv run ...`, not a manually activated Conda environment.

3. Point Git at the repo-managed hooks:
```bash
git config core.hooksPath .githooks
```

Pre-commit hooks in this repo are expected to run through `uv run prek`.

## Usage

This project uses both **Manim** for animations and **manim-slides** for interactive presentations.

### Running animations

For standard animations without slide functionality:

```bash
uv run manim file.py SceneName
```

Example:

```bash
uv run manim vizz/flex/end_to_end.py AttentionScoresVisualization
```

### Creating interactive slides

For slide-based presentations with classes that inherit from `Slide`:

```bash
uv run manim-slides render file.py SceneName
```

Examples:

```bash
uv run manim-slides render vizz/flex/natten.py RasterizationComparison
ORDER=morton uv run manim-slides render vizz/flex/natten.py RasterizationComparison
uv run manim-slides render vizz/flex/score_mod.py ScoreModAttentionVisualization
uv run manim-slides render vizz/presentations/ptce_2026_flex_flash/build.py PTCE2026FlexFlash -ql
```

### Presenting slides

After rendering slides, start the presentation:

```bash
uv run manim-slides present SceneName
```

Example:

```bash
uv run manim-slides present PTCE2026FlexFlash
```

Presentation controls:
- `Space` or `Right Arrow`: next slide
- `Left Arrow`: previous slide
- `R`: restart presentation
- `Q` or `Esc`: quit presentation

### Quality settings

Low quality for fast preview:

```bash
uv run manim file.py SceneName -ql
uv run manim-slides render file.py SceneName -ql
```

High quality for final output:

```bash
uv run manim file.py SceneName -qh
uv run manim-slides render file.py SceneName -qh
```

### Interactive development

Manim's `-p` opens the rendered result (it is not a file watcher):

```bash
uv run manim file.py SceneName -p
```

## Available animations

### FlexAttention (`vizz/flex/`)

| File | Scene Class | Description | Type |
|------|-------------|-------------|------|
| `end_to_end.py` | `AttentionScoresVisualization` | Complete attention mechanism walkthrough | Animation |
| `natten.py` | `NattenBasicVisualization` | NATTEN neighborhood attention basics | Slide |
| `natten.py` | `RasterizationComparison` | Compare rasterization patterns | Slide |
| `score_mod.py` | `ScoreModAttentionVisualization` | Score modification functions | Slide |
| `mod_scene.py` | `MaskAnimationScene` | Attention mask visualization | Slide |
| `causal_attention.py` | `CausalAttentionVisualization` | Causal attention masking | Slide |
| `block_mask.py` | `BlockMaskKVCreation` | Block mask construction | Slide |

The FlexAttention + FlashAttention-4 deck is under
`vizz/presentations/ptce_2026_flex_flash/` (scene: `PTCE2026FlexFlash`). It also
supports `uv run vizz preview ptce_2026_flex_flash --slide title` and
`uv run vizz render ptce_2026_flex_flash`.

## Output files

- CLI deck videos/assets: `media/<deck>/`
- Isolated single-slide previews: `media/review/<deck>/<slide>/`
- Presentable slide metadata and clips: `slides/`
- Direct Manim CLI output defaults to `media/videos/`

## Development tips

1. Start with `-ql` during development for faster iteration.
2. Use `uv run manim ... -p` when you want interactive iteration on non-slide scenes.
3. Use `uv run manim-slides render ...` before presenting.
4. Keep environment-variable customization local to the command that needs it.

## Requirements

- Python 3.11+
- `uv`
- Manim
- manim-slides
- PyTorch
- Pillow
- NumPy

## Validation and upgrades

```bash
uv run pytest tests/ -v
uv run ruff check vizz/cli.py vizz/presentations/starter/ tests/
uv run ruff format --check vizz/cli.py vizz/presentations/starter/ tests/
uv run manim checkhealth
```

`uv.lock` records the tested environment; normal setup is `uv sync`, not an
unbounded reinstall. For a deliberate refresh, run `uv lock --upgrade` and
`uv sync`, then the tests and a rendered starter/representative existing slide.
Runtime dependencies have compatibility bounds; developer tooling lives in the
`dev` dependency group. Some older animation files have pre-existing Ruff
violations; avoid mass-formatting unrelated scenes during authoring.

## Troubleshooting

1. If imports fail, run `uv sync` again.
2. If `ffmpeg` is missing, install it with Homebrew.
3. If LaTeX rendering fails, install `mactex` with Homebrew.
4. If slides will not present, render them first with `uv run manim-slides render ...`.
5. `manim-slides present` needs Qt bindings. This repo includes `pyside6`, so run `uv sync` if you see `qtpy.QtBindingsNotFoundError`.
