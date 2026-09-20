# Presentation brief

Fill in what you know. Leave unknowns explicit; a rough sketch is enough to start.

- **Working title:**
- **Audience / assumed background:**
- **One thing they should remember:**
- **Length / number of slides:**
- **Output:** interactive slides / video / PowerPoint (Keynote)
- **Must be technically accurate:**
- **Sources / data / units:**
- **Style:** Nuggets light / dark; note any reference visuals here.
- **Starting pattern:** focus / tiles / comparison (see `docs/patterns.md` at repo root).
- **Persistent selection:** the entry, token, or region we should follow.

## Sketch handoff

Save the editable source to `sketches/<slide-name>.excalidraw` and export a matching
PNG to `sketches/<slide-name>.png`. A screenshot or photo also works. Keep exports
readable and include the whole canvas. SVG can be a reference too, but text/fonts
and embedded images may need rebuilding rather than direct Manim import.

For each sketch, tell the agent:
- What the boxes, arrows, axes, and colors **mean**.
- Which labels and spatial relationships must stay exact.
- What is revealed first, second, and third (name the beats, e.g. select → operands → result).
- Which regions are masked versus inactive; whether a shown 0 is an actual value.
- What should move or transform, versus remain on screen.
- Whether to preserve the rough look or redraw in the shared theme.

## Pairing prompt

> Read brief.md, scenes.md, and sketches/<slide-name>.png. Use the .excalidraw
> source for exact labels/geometry where helpful. Turn this into a slide using
> the Nuggets theme, TensorGrid, and Comparison where they fit. Preserve the meaning, not incidental sketch
> coordinates. State consequential assumptions, update scenes.md, then implement
> and preview only this slide. Inspect all pause-state PNGs before reporting.
> Do not invent data or silently change the explanation.

## Acceptance

- [ ] The story and technical claims are correct.
- [ ] Each click has a clear purpose and readable labels.
- [ ] No clipping, overlapping labels, or misleading arrows.
- [ ] Single-slide preview reviewed, then full deck rendered before export.
