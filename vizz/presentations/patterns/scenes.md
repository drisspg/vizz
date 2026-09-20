# Nuggets technical pattern gallery

## Contract

Audience: technical presenters building attention, quantization, or kernel diagrams.
Purpose: copy small explanation patterns rather than invent each slide's layout.
Style: IBM Plex Sans/Mono, warm paper or near-black, subtle structure, semantic color.
Default: light. Review all beats in both themes. No benchmark or KDA experiment claims.

## focus — Follow one score

- An 8×8 causal score grid has a persistent amber focus on query 5 / key 2.
- `select`: show the matrix and name the selected entry; masked cells are hatched.
- `operands`: reveal q5=(1,2) and k2=(3,−1), keeping that entry selected.
- `result`: compute the unscaled dot product 1×3 + 2×(−1)=1; put 1 into the cell.
- This is a two-channel illustrative dot product, not the full attention pipeline.

## tiles — Separate the footprint from the mask

- A 16×16 causal matrix with 4×4 block boundaries; full-height four-column strip.
- `footprint`: show keys 0–3; amber outline is the issued/selected footprint,
  green cells are retained, hatching means excluded by the logical mask.
- `advance`: move to keys 4–7 without changing the matrix or mask.
- `detail`: show that selected, zero, inactive, and masked are separate concepts.
- Dimensions are a schematic pattern example, NOT a legal hardware instruction claim.

## comparison — Change one rule, not the picture

- Same 8×8 grid, identical cell sizes and axes on both sides.
- `baseline`: show full causal support on the left and an inactive scaffold on the right.
- `candidate`: reveal a three-token causal window on the right.
- `difference`: outline one query row on both; the allowed history changes, not scale.
- Excluded entries hatch; inactive is only an unrevealed state, never a zero value.
- The comparison layout uses one common scale factor, not independent fit-to-box.

## Review

- All objects stay inside the slide, labels remain readable, hatches stay in cells.
- Each pause names its purpose in the speaker notes.
- Amber focus is an overlay: it never clears a logical mask or replaces stored values.
- Blog references: KDA's selected-entry explainer, MMA strip map, and borderless
  training/inference comparison. Adapt their visual language, not their tiny web text.
