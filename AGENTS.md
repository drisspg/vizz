# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Project Overview

Vizz is a visualization library for creating mathematical and attention mechanism animations using Manim and manim-slides. The project focuses on visualizing FlexAttention patterns, quantization, and various attention mechanisms for educational and presentation purposes.

## Development Environment

**Required:** Use `uv run ...` commands from the repo root. Do not rely on a Conda environment activation flow for this repository.

### Setup
```bash
uv sync
```

## Common Development Commands

### Running animations

For standard animations that do not inherit from `Slide`:

```bash
uv run manim vizz/flex/[file].py [SceneName]
```

Example:

```bash
uv run manim vizz/flex/end_to_end.py AttentionScoresVisualization
```

For slide-based presentations with classes inheriting from `Slide`:

```bash
uv run manim-slides render vizz/flex/[file].py [SceneName]
uv run manim-slides present [SceneName]
```

For presentations under `vizz/presentations/`:

```bash
uv run manim-slides render vizz/presentations/<name>/build.py [SceneName] -ql
uv run manim-slides present [SceneName]
```

Example:

```bash
uv run manim-slides render vizz/presentations/ptce_2026_flex_flash/build.py PTCE2026FlexFlash -ql
uv run manim-slides present PTCE2026FlexFlash
```

### Quality options

- `-ql` for fast preview
- `-qh` for final output
- `-p` for interactive development with `manim` only

### Code quality

```bash
uv run ruff check vizz/
uv run ruff check --fix vizz/
uv run ruff format --check vizz/
uv run ruff format vizz/
```

## Project Architecture

### Directory Structure
```
vizz/
├── presentations/
│   ├── theme.py
│   ├── components.py
│   └── <presentation_name>/
│       ├── __init__.py
│       ├── build.py
│       └── slides/
│           ├── slide_one.py
│           └── ...
├── flex/
│   ├── block_mask.py
│   ├── causal_attention.py
│   ├── end_to_end.py
│   ├── mod_scene.py
│   ├── natten.py
│   ├── ordering_comparison.py
│   └── score_mod.py
└── quant/
```

### Key patterns

1. Presentations live under `vizz/presentations/<name>/` with a `build.py` entry point and per-slide modules under `slides/`.
2. All presentations share theming via `vizz/presentations/theme.py` and inherit from `SlideBase` in `vizz/presentations/components.py`.
3. Standalone animation scenes in `vizz/flex/` and `vizz/quant/` inherit from `Slide` directly and are rendered with `uv run manim-slides render ...`.
4. Standard animation classes (non-slide) run with `uv run manim ...`.
5. The repo uses a light theme by default.
6. Scene-specific environment variables should be passed inline with the command.

### Environment variables

Example:

```bash
ORDER=morton uv run manim-slides render vizz/flex/natten.py RasterizationComparison
ORDER=row_major uv run manim-slides render vizz/flex/natten.py RasterizationComparison
```

## Available animation scenes

### Slide-based presentations

- `ScoreModAttentionVisualization`
- `NattenBasicVisualization`
- `RasterizationComparison`
- `MaskAnimationScene`
- `BlockMaskKVCreation`
- `AttentionScoresVisualization`
- `OrderingPatterns`
- `CausalAttentionVisualization`
- `PTCE2026FlexFlash` (in `vizz/presentations/ptce_2026_flex_flash/build.py`)

## Output locations

- Videos: `videos/[SceneName]/`
- Slides: `slides/`
- Media assets: `media/`

## Testing animations

1. Preview with `uv run manim file.py SceneName -ql`.
2. Use `uv run manim file.py SceneName -p` for interactive work on non-slide scenes.
3. For slides, use `uv run manim-slides render file.py SceneName -ql` and then `uv run manim-slides present SceneName`.
4. Finalize with `uv run manim-slides render file.py SceneName -qh`.

## Dependencies

Core dependencies are declared in `pyproject.toml` and should be run through `uv`.

## Visual review loop for slides

When iterating on slide layout, always self-review the rendered output before presenting changes to the user. This catches clipping, overflow, and centering issues.

### Single-slide iteration

Use the `SLIDE` env var to render only one slide (much faster than the full deck):

```bash
SLIDE=title uv run manim vizz/presentations/<name>/build.py <SceneName> -ql
```

### Extracting a frame for visual inspection

1. Render the slide with `-ql` to produce a video under `media/videos/build/480p15/`.
2. Get the video duration:
   ```bash
   ffprobe -v quiet -show_entries format=duration -of csv=p=0 media/videos/build/480p15/<SceneName>.mp4
   ```
3. Extract a frame from the fully-built state (pick a timestamp after all animations but before `clear_stage`):
   ```bash
   ffmpeg -y -ss <seconds> -i media/videos/build/480p15/<SceneName>.mp4 -frames:v 1 media/preview_frame.png
   ```
4. Read `media/preview_frame.png` with the Read tool to inspect the layout visually.

### What to check

- Content not clipped or overflowing panel borders
- Panel titles not overlapping chart content
- Bullets fully visible (not falling off screen edges)
- Text readable and not overlapping other elements
- Proper centering and spacing between elements

### Full workflow

1. Make the edit to the slide module.
2. Render with `SLIDE=<name>` and `-ql`.
3. Extract and read a frame to verify.
4. If issues are found, fix and re-render before reporting back.
5. When satisfied, show the user the result.

## Exporting to Keynote

Use `manim-slides convert` to produce a `.pptx` that Keynote can open natively:

```bash
uv run manim-slides convert <SceneName> output.pptx
open output.pptx
```

## Important Notes

1. Use `uv sync` to set up the repo.
2. Use `uv run ...` for Manim, manim-slides, and Ruff commands.
3. Slide classes require `manim-slides`, not `manim`.
4. Interactive mode with `-p` only applies to `manim` scenes.
5. The project uses a light presentation theme.
