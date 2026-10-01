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

## New presentation workflow

Read `docs/authoring.md` before starting a new deck. Use the runnable starter:

```bash
uv run vizz new my_talk
uv run vizz preview my_talk --slide workflow
uv run vizz render my_talk
```

For technical diagrams, read `docs/patterns.md` and use
`uv run vizz new my_talk --template patterns`. Reuse `TensorGrid` and `Comparison`
before inventing new matrix helpers/layouts. The renderable catalog is
`uv run vizz preview patterns --theme dark` (also review `--theme light`).
Omitting `--slide` reviews all slides. Explicit themes add a theme directory to
preview output paths. Preserve existing decks' themes unless asked to migrate.

Each new deck includes `brief.md`, `scenes.md`, `sketches/`, and `deck.toml`.
Treat PNG/screenshot exports plus editable `.excalidraw` sources as visual briefs:
preserve labels and semantic relationships, agree on reveal order, then rebuild
only the elements that need animation. Do not promise automatic lossless import.
New slide modules leave the last pause state visible; the deck clears between
modules. Preserve existing decks' cleanup conventions unless migrating explicitly.

For interactive review, see "Interactive review loop" below.

Preview galleries live in `media/review/<deck>/<slide>/frames/index.html` with
one PNG per pause (1080p stills by default, ~3 s/slide; animations skipped).
Read every changed pause-state image before reporting success. For motion, run
`uv run vizz preview <deck> --slide <s> --motion` and watch the clip (on the review
page: *Render animation*, then *Play animation*). Previews do not replace the full deck's
`slides/` metadata. Run `uv run pytest tests/ -v` when changing the workflow CLI.

## Interactive review loop (`vizz review`)

A local page where the user edits slide wording and leaves free-form visual
comments, then presses **Send to agent**. Full user-facing docs:
`docs/authoring.md` ("Interactive review").

### Start it

```bash
uv run vizz review <deck>              # http://127.0.0.1:8765/ ; --port, --render
```

It renders the whole deck first when no review frames exist (`--render` forces
it). It is a long-running server: start it in the background, record its PID,
and stop only your own process. `uv run vizz preview <deck> [--slide s]` also
refreshes the page's frames, so re-render with `preview` after code edits.

### How it works

- `vizz preview` records one entry per pause (slide key, speaker notes, visible
  text with frame boxes) via `SlideBase.beat_log`, then `review.merge_render`
  copies frames into `media/review/<deck>/review/` (keeps one previous frame per
  pause for before/now).
- Feedback lives in `vizz/presentations/<deck>/review/feedback.json`
  (committed). Kinds: `wording` / `notes` (exact `old` → `new`) and `visual`
  (free-form `text` plus optional numbered `marks`: `box`, `pin`, `arrow` in
  frame fractions 0..1, y down; `annotated` is the frame with marks drawn).
  Status: `open` (draft until it has `submission`) → `fixed` / `applied` /
  `question` / `wontfix` → `done` (user accepted). Done items get archived to
  `review/archive.json`.
- Wording/notes edits apply immediately (`POST /api/wording`) when their `old`
  text occurs exactly once in the deck's `.py` files; ambiguous ones become
  `question` items. **Send to agent** stamps the remaining drafts with a
  submission id and appends a `pending` entry to `review/submissions.json`.
- Text boxes come from recorded pauses. `meta_text` and `themed_code` store the
  source string (`source_text`) so wording edits match the code, not the
  uppercased render. Build text with `SlideBase` helpers to keep it editable.

### Agent side of the loop

```bash
uv run vizz feedback wait <deck>       # blocks until Send; claims + prints items (exit 2 = timeout)
uv run vizz feedback list <deck>       # open items without waiting (--all, --json)
uv run vizz feedback resolve <deck> <id> --status fixed|question|wontfix --reply "..."
uv run vizz feedback done <deck> --message "..."   # closes the claimed submission
uv run vizz feedback archive <deck>    # archive done items, prune unused review media
```

For each claimed item: read its `annotated` PNG (the marks are what the user
drew), fix it, re-render the slide with `vizz preview --slide <slide>`, read the
new frame, then `resolve` with a one-line reply. Use `question` when the note is
ambiguous rather than guessing. Finish with `done`; the page shows the message.

Staying in the loop from Pi: run `feedback wait` somewhere that wakes the
session when it exits. On Linux, background it and use the `watch` tool on its
PID. On macOS (`watch` needs `/proc`), launch a background `runner` subagent
whose only task is to run `feedback wait <deck> --timeout 27000` and return its
stdout; its completion wakes the parent. Re-arm the waiter after each `done`.

Changing the review code: `vizz/review.py`, `vizz/review_app/index.html`,
`vizz/cli.py`; tests in `tests/test_review.py`. Restart the server after
Python changes (the HTML is re-read on each page load).

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

- CLI videos: `media/<deck>/videos/`; direct Manim videos: `media/videos/`
- Isolated previews: `media/review/<deck>/<slide>/`
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

For decks with `deck.toml`, prefer the isolated preview and its pause-state PNGs:

```bash
uv run vizz preview <name> --slide title
```

The legacy `SLIDE=title uv run manim-slides render ... -ql` path still works but
writes normal slide metadata. For scenes without the CLI workflow, use the
manual frame-extraction procedure below.

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
