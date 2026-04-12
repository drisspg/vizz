# Vizz

Collection of math visualizations using Manim and Manim-Slides for interactive presentations.

## Project Structure

```
vizz/
├── flex/       # Animations for FlexAttention
└── quant/      # Quant animations
```

## Setup

1. Install macOS system dependencies:
```bash
brew install ffmpeg mactex
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
uv run manim-slides render vizz/flex/ptce_2026_flex_flash.py PTCE2026FlexFlash -ql
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

Manim supports live iteration for non-slide scenes:

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
| `ptce_2026_flex_flash.py` | `PTCE2026FlexFlash` | Lightning talk deck for FlexAttention + FlashAttention-4 | Slide |

## Output files

- Videos are saved under `videos/`
- Slide metadata is saved under `slides/`
- Rendered assets are saved under `media/`

## Development tips

1. Start with `-ql` during development for faster iteration.
2. Use `uv run manim ... -p` when you want interactive iteration on non-slide scenes.
3. Use `uv run manim-slides render ...` before presenting.
4. Keep environment-variable customization local to the command that needs it.

## Requirements

- Python 3.10+
- `uv`
- Manim
- manim-slides
- PyTorch
- Pillow
- NumPy

## Troubleshooting

1. If imports fail, run `uv sync` again.
2. If `ffmpeg` is missing, install it with Homebrew.
3. If LaTeX rendering fails, install `mactex` with Homebrew.
4. If slides will not present, render them first with `uv run manim-slides render ...`.
5. `manim-slides present` needs Qt bindings. This repo includes `pyside6`, so run `uv sync` if you see `qtpy.QtBindingsNotFoundError`.
