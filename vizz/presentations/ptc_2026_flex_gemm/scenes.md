# Scenes

Target ~10 minutes. Speaker notes live in each `next_slide(notes=...)`.

| # | Slide key | Beats | Point |
|---|---|---|---|
| 1 | `title` | 1 | FlexGEMM + engraved hero plate (`assets/tile_plate.tex`): A/B panels sweep k into one amber accumulator tile, epilogue E, store D; dotted C never written |
| 2 | `problem` | unfused → fused | `mm` then epilogue writes and re-reads C; fusion removes 2·M·N·sizeof(C) bytes + a launch |
| 3 | `api` | code → callouts | `flex_gemm(torch.mm, (a, b), epilogue)`; closures = loads, tuple = aux outputs, eager semantics |
| 3b | `coverage` | supported → not yet | mm/addmm, bmm/baddbmm, F.scaled_mm (MXFP8/NVFP4), F.grouped_mm (varlen-M) on main; scaled_grouped_mm + grouped wgrad not yet |
| 4 | `contract` | tile → fuses → rejects | tile-local work fuses (pointwise, loads, aux, SwiGLU lanes, N-group reductions); cross-tile work is a compile error |
| 5 | `wins` | model → caveats | ΔBytes = 2·M·g·N·sizeof(C); schematic bars; when fusion loses |
| 6 | `results` | speedups → numerics → caveat | B200: bias+ReLU 1.15×, NVFP4 producer 1.52×, MXFP8 MLP 1.07× + bit-exact; SwiGLU no e2e win yet |
| 7 | `stack` | pipeline → backends | HOP → Inductor analysis → GEMM template; QuACK primary, NVGEMM in progress, Triton; inline asm escape hatch |
| 7b | `philox` | code → plate | Philox4x32-10 + cvt.rs stochastic BF16 rounding entirely in the epilogue via inline_asm_elementwise; engraved round diagram (`assets/philox_round.tex`). Correctness demo only |
| 8 | `how_we_did_it` | solved → unsolved | landed PRs per hard part; split-K, full-row reductions, autograd still open |
| 9 | `status` | 1 | takeaways + status: landed on PyTorch main |

## Correctness notes

- Results baseline is `torch.compile(fullgraph=True)` of the same unfused program; timing is CUDA-graph replay, median of alternating rounds.
- MXFP8 numerics: exact-match rate against a materialized FP32-accumulator reference (compiled 10.774%, FlexGEMM 100%).
- The `wins` bars are schematic and labelled as such.

## TikZ plates

Sources in `assets/*.tex` (LuaLaTeX, EB Garamond, vintage style from the creating-tikz-diagrams skill). Rebuild:

```zsh
cd vizz/presentations/ptc_2026_flex_gemm/assets
for f in tile_plate philox_round; do latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=/tmp/ptc_tikz $f.tex && pdftocairo -png -transp -r 300 -singlefile /tmp/ptc_tikz/$f.pdf $f; done
```

PNG plates use dark ink on a transparent page: light theme only.
