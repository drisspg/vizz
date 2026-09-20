# Authoring and pairing workflow

## Start with a runnable deck

Run all commands from the repository root:

```bash
uv sync
uv run vizz new my_talk
uv run vizz preview my_talk --slide workflow
```

`new` refuses to overwrite an existing directory. The generated `deck.toml`
names the Manim scene class; `build.py` imports one module per slide and declares
an ordered `SLIDES` mapping. Keep that mapping and class name in sync when
renaming files. The CLI is repo-local, not a general installed-deck manager.

For matrix/attention/kernel visuals, start with
`uv run vizz new my_talk --template patterns` instead. Read the
[Nuggets pattern catalog](patterns.md) for reusable tensor grids, same-scale
comparisons, named walkthrough beats, and the bundled light/dark themes.

## Agree on meaning before motion

1. Fill in `brief.md`: audience, one takeaway, constraints, sources, and output.
2. Put rough drawings in `sketches/`: PNG/screenshot for inspection, plus the
   `.excalidraw` source for edits and exact geometry. No Excalidraw service,
   account, or upload is required by this repo.
3. Update `scenes.md` with the narrative: visual elements, reveal beats, speaker
   notes, and what makes each slide correct. Ask only about ambiguity that
   changes the explanation; state other assumptions and proceed.
4. Implement one slide. Review the pause-state gallery and motion before
   expanding to the whole deck.

A useful sketch labels arrows with their meaning (data flow, dependency, time,
transformation), names axes and units, and numbers reveal steps. Mark which
positions matter semantically and which are just rough placement. Never infer
benchmark results or equations from a blurry drawing.

### Choosing how to use a sketch

| Need | Approach |
| --- | --- |
| Quick static reference | `scene.image_panel(str(asset_path), width=...)` |
| Independent reveals or highlights | Rebuild boxes, labels, and arrows as Manim objects |
| Precise plot or math | Use source data/equations, not traced screenshot pixels |
| SVG artwork | Try `SVGMobject` for supported vector paths; check text/images/fonts carefully |

This workflow deliberately avoids a general Excalidraw JSON-to-Manim converter:
groups, bindings, freehand strokes, text wrapping, and animation intent need
interpretation. Keep the source as the editable design artifact and the Manim
module as the editable animation artifact.

## Small, reusable patterns

- One slide module exposes `build(scene: SlideBase) -> None`.
- One takeaway per slide; one meaningful change per `next_slide(notes=...)`.
- Reuse `section_header`, `labeled_panel`, `code_card`, and `theme` tokens instead
  of copying font sizes, colors, or panel geometry across decks.
- Use `Group` for mixed images and vectors; `VGroup` only for vector objects.
- Position objects relative to each other (`arrange`, `next_to`, `align_to`).
  Attach arrows to box edges, not guessed absolute coordinates.
- Decide layout before animating it. Use a deliberate line break rather than
  shrinking a paragraph until it becomes unreadable.
- In new decks, the **deck** clears between slides, not the slide module. Leave
  the final pause state intact. Existing PTCE modules retain their own cleanup
  for compatibility and can have an extra transition-only clip in the gallery.
- End each reveal with a short `wait` and `next_slide`. A pause with no preceding
  animation cannot produce a video clip.
- Resolve assets relative to the slide/deck file, not an external working directory.
- Keep business data and citations next to the deck, rather than baking them
  into reusable rendering components.

The starter demonstrates a title and a three-stage flow with progressive
reveals. The existing PTCE deck has more examples of charts, code, images, and
complex technical diagrams; copy only the pattern you actually need.

## Review loop

```bash
uv run vizz preview my_talk --slide workflow
open media/review/my_talk/workflow/frames/index.html
```

The gallery uses the **last decoded frame of each clip**, avoiding timestamp
estimates that accidentally capture an unfinished transition or blank ending.
It includes speaker notes and numbered PNGs for precise feedback: “Beat 2: make
the dependency arrow point left.” It is a static review artifact, not a live
file watcher. Re-run after edits. Preview always uses low quality and skips
reverse-video generation; use a full render for presenting.

Omit `--slide` to review the whole deck (output under `all/`). Add
`--theme light` or `--theme dark` for Nuggets variants, saved under an additional
`light/` or `dark/` directory before `frames/`. With no theme option, the deck's
chosen theme and the original output path are unchanged.

Inspect every beat for:
- clipping and overflow, including titles and panel contents;
- overlapping text/arrows and unreadably small labels;
- consistent visual meaning for colors, shapes, and direction;
- accurate claims, source data, and units;
- reveal order that matches the narration.

For motion, watch the MP4 under the preview's `videos/` directory too. Once the
single slide looks right:

```bash
uv run vizz render my_talk --quality h
uv run manim-slides present MyTalkDeck
uv run manim-slides convert MyTalkDeck my_talk.pptx
```

Preview and full-render outputs are separate, but do not run two renders of the
same deck/slide into the same output directory simultaneously. Present/export
commands use the most recent **full** render. Direct legacy commands with
`SLIDE=...` still work, but write their usual slide metadata; use `vizz preview`
when you need isolation.

## Agent handoff

Use the prompt in your deck's `brief.md`. Agents should read the plan and source
sketch, edit one slide at a time, run targeted lint/tests, render the changed
slide, inspect the PNGs, then report the artifact path and any unresolved
technical questions. Do not call a layout done based on code alone.
