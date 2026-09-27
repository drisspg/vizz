# Presentation brief

- **Working title:** FlexGEMM: Bringing Flexible PyTorch Epilogues to GEMM
- **Venue:** PyTorch Conference North America, 2026-10-20/21. 10-minute lightning talk.
- **Audience:** intermediate PyTorch users and compiler/kernel folks; knows `torch.compile`, maybe FlexAttention.
- **One thing to remember:** write the epilogue in PyTorch, get one kernel or a clear error; the win is the bytes you do not move.
- **Length:** 11 slides, 22 pause beats (~1 min per slide).
- **Output:** manim-slides present + pptx export (`uv run manim-slides convert Ptc2026FlexGemmDeck ptc_2026_flex_gemm.pptx`).
- **Must be technically accurate:** yes. Every number traces to a source below.
- **Style:** Nuggets light, frontier-editorial semantic palette (green fused/supported, amber focus, red wasted/rejected); two engraved TikZ plates.

## Sources

- Abstract: `~/obsidian/flex_gemm/FlexGEMM_for_PyTorch_Conference_North_America_2026.md`
- Explainer notebook + baked outputs: `~/meta/my_scripts/misc/FLEX_GEMM_TALK.ipynb` (B200, PyTorch `8f61c19`, 2026-08-11)
- Benchmark scripts: `~/meta/my_scripts/flexy/` (`basic.py`, `low_precision_outputs.py`, `mxfp8_mlp.py`)
- Earlier internal deck: `~/obsidian/Presentations/flex_gemm/flex_gemm_presentation_model_context.md`
- Landing state: `~/agent_notes/plans/flex_gemm_stack_192662_next_steps.md`
- Pre-conference TODO: `~/obsidian/flex_gemm/PTC_2026_Before_Conference_TODO.md`

## Known gaps before freeze (2026-10-13)

- Results slide numbers are from `8f61c19`; re-bake on the landing stack head and update `slides/results.py` `CASES` + footer.
- Stack landed on main (2026-09-10..22); verify the nightly import path before freeze.
- Watchable video: `VIDEO_HOLD=4 uv run manim -qh vizz/presentations/ptc_2026_flex_gemm/build.py Ptc2026FlexGemmDeck -o ptc_2026_flex_gemm_draft.mp4`
