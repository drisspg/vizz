# Data-driven animation ideas (real HUD numbers only)

All numbers come from `research/hud_usage.md` and `research/data/hud/*` (collected
2026-10-02, HUD ClickHouse `misc.claude_code_usage`, `default.pull_request`,
`default.issues`). No dollar figures on slides (open question in `brief.md`).
Semantic colors stay: green = human/trusted, amber = agent/machine-dispatched, red = untrusted/failure.

Prototype of idea 1: `sketches/proto_data.py` (`uv run manim -ql vizz/presentations/claude_pytorch/sketches/proto_data.py ProtoData`).

---

## 1. "One dot = 100 runs" waffle that sorts itself by who pressed the button  ← PROTOTYPED

- **Target slide:** `adoption`, beat 3 (replaces the flat 81 / 11 / 8 % split bar).
- **Data:** `by_category.json` (pytorch/pytorch): Dr.CI advisor 33,195 · autorevert advisor 29,085 ·
  human `@claude` 8,789 · issue triage 4,453 · manual dispatch 875 · bot `@claude` 428 = 76,825.
  p50 minutes per category (0.9 / 1.2 / 3.2 / 1.1 / 1.4 / 2.9). Backfill: 4,912 + 6,179 + 10,930 = 22,021 runs.
- **Beats:**
  1. 769 neutral dots stream in from the left into a uniform grid while a counter ticks 0 → 76,825. Caption "1 dot ≈ 100 Claude runs".
  2. Dots recolor and fly into six dot-bars sorted by size: amber for the two CI advisors, green for humans, muted for triage/manual/bot. Labels: name, run count, p50 minutes. The eye immediately sees "most dots are machines".
  3. A bracket closes around the top 220 autorevert dots: "Aug 19-21: ~22k advisor-coverage backfill in 3 days (intentional, verdicts never revert)".
- **Why it impresses:** the audience sees the *volume* reorganize rather than reading percentages; the sort is the argument ("agents mostly talk to agents"). The backfill bracket turns a scary spike into a deliberate one-off.
- **Manim:** `VGroup` of `Dot`s, `LaggedStart(dot.animate.move_to().set_fill())`, `always_redraw(Text)` counter on a `ValueTracker`, `Brace`/rectangle for the bracket.
- **Cost:** M (one file, ~150 lines; renders in ~40 s at -ql).
- **Gimmick risk:** low-medium. Keep dot motion ≤ 2 s and sort into aligned bars; avoid confetti.

## 2. Weekly particle stream: humans vs machines over 36 weeks

- **Target slide:** `adoption`, beat 1 (upgrade of the weekly human bar chart).
- **Data:** `weekly_by_category.csv`: per ISO week, runs per category. Human: 16 → 891 (Jun 1) → ~300. Machines: 0 until Mar 23 (autorevert 130), Dr.CI jumps 5 → 1,249 → 2,542 (Jun 15-22), backfill 22,054 week of Aug 17.
- **Beats:**
  1. Time axis Jan 26 → Sep 28 draws left to right. Each week emits green dots upward for human runs (1 dot ≈ 10 runs) that settle into a column: the familiar bar chart grows out of particles.
  2. `#176027` marker (Feb 23 week) drops in; the green stream thickens after it.
  3. Amber machine dots start Mar 23, overwhelm the chart in June; the Aug 17 week overflows the top of the frame with "22,054 ↑ (backfill)" clipped on purpose.
- **Why it impresses:** motion encodes time-of-arrival; the moment machines "take over" is visible as a color change, not a legend.
- **Manim:** per-week `VGroup` of small `Dot`s with `GrowFromPoint` / `FadeIn(shift=UP)`, `LaggedStart` across weeks; log-ish y clamp for the backfill week (state it as clipped on the label, never rescale silently).
- **Cost:** M-L (two series, 36 weeks; cap dots at ~1,500 for render time).
- **Gimmick risk:** medium. Particles must settle into exact bar heights or the chart loses credibility.

## 3. Intake flood vs a fixed maintainer bar

- **Target slide:** `problem`, beats 1-2.
- **Data:** `pr_intake_monthly.json` (21 months, 1,491 → 2,979; peak 3,164 Aug 2026); `pr_intake_quarter_assoc.json` NONE: 190 → 1,824. Issue intake flat (`issue_intake_quarter.json`, 1.5-2.3k/quarter).
- **Beats:**
  1. A thin green "maintainers" box on the right stays fixed. PR squares (1 square = 50 PRs) flow in month by month from the left and stack as columns; the stack widens and the stream gets denser through 2026 (amber from Jan 2026).
  2. Squares from first-time authors flip to red outline and count up 190 → 1,824 per quarter next to the stack.
  3. Issue counter stays flat beside it ("issues: flat") so the audience sees the asymmetry.
- **Why it impresses:** the "2x in 18 months" and "10x first-time" numbers become a visible pressure differential against a constant-size box.
- **Manim:** `Square` grids per month, `LaggedStart(FadeIn(shift=RIGHT))`, `Indicate` on first-time squares, text counter.
- **Cost:** M.
- **Gimmick risk:** medium. The flood metaphor must not read as "contributors are bad"; keep the maintainer box neutral and the title factual.

## 4. Counter wall: four odometers landing in sequence

- **Target slide:** `adoption`, beat 2 (the four STATS tiles).
- **Data:** since 2026-02-28 (`hud_usage.md`): 135 humans · 8,695 runs on 4,102 threads · 4,453 issues triaged from 1,375 reporters · 13 repos (`by_repo.json`).
- **Beats:** each big number rolls up from 0 over ~0.8 s, staggered 0.3 s; its small caption fades in as it lands. The "13 repos" tile finishes by listing the repos in a tiny mono column (`by_repo.json`: pytorch 76,825 · pytorch-gha-infra 3,364 · test-infra 2,798 · executorch 629 · ao 493 · ciforge 454 · …).
- **Why it impresses:** cheap kinetic energy; odometers are universally read as "live data".
- **Manim:** `ValueTracker` + `always_redraw(scene.title_text(f"{int(v):,}"))`, `LaggedStart`.
- **Cost:** S.
- **Gimmick risk:** low if the roll-up is < 1 s; high if every number on the deck starts rolling.

## 5. Advisor loop as a flowing ring with a daily throughput meter

- **Target slide:** `agents_vs_agents`, beats 1-3.
- **Data:** `monthly.csv` Dr.CI advisor: 8 (Apr) → 4,542 (Jun) → 11,823 (Jul) → 8,145 (Aug) → 8,264 (Sep); autorevert advisor 161 → 1,353 → 22,726 (Aug). Daily normal 10-70 advisor runs vs 4,912 / 6,179 / 10,930 on Aug 19/20/21.
- **Beats:**
  1. The existing signal → advisor → verdict → ClickHouse → autorevert ring is drawn; amber tokens circulate along it (`MoveAlongPath`), one token per ~250 runs/month, so July is visibly busier than April.
  2. A small month strip under the ring advances Apr → Sep; token density follows the monthly totals.
  3. On "Aug 19", the token stream bursts (hundreds of tiny tokens on the autorevert edge for 1 s) and a label pins "coverage backfill, pr_number = 0, coverage_ keys never revert".
- **Why it impresses:** makes "agents investigating agents" literal: the audience watches an unattended loop run at machine cadence.
- **Manim:** `MoveAlongPath` on `Arc`/`Line` paths with `LaggedStart` and `rate_func=linear`; burst via `LaggedStart(FadeIn/FadeOut)`.
- **Cost:** L (path plumbing on top of the existing diagram).
- **Gimmick risk:** high. Circulating tokens are decorative unless the density is clearly tied to the month strip.

## 6. Human-vs-machine duration race (two lanes, same clock)

- **Target slide:** `adoption`, beat 3 alternative, or `lessons` (1 h credential point).
- **Data:** p50 minutes: Dr.CI 0.9, autorevert 1.2, triage 1.1, human 3.2; human p90 28.4 min; machines ~$0.5/run median vs human p50 $1.42 (cost omitted on slide, keep minutes).
- **Beats:**
  1. Two lanes, a shared timeline 0 → 30 min. An amber advisor token finishes at 0.9 and 1.2 min; a green human token is still running at 3.2 (p50) and a ghost green token at 28.4 (p90).
  2. Counter: in the time one p90 human session runs, the advisors complete ~25 runs (28.4 / 1.2 ≈ 24; derived from the two medians, state the arithmetic in notes).
  3. Tie-in text: "1 h OIDC credential → 55-min job budget" (`lessons`).
- **Why it impresses:** explains *why* 11 % of runs are 36 % of cost without showing dollars, and motivates the time-budget hook.
- **Manim:** `ValueTracker` clock, `Dot.animate.shift` with `rate_func=linear`, `always_redraw` clock text.
- **Cost:** S-M.
- **Gimmick risk:** low-medium; the derived "~24 runs" ratio must be labeled as derived from medians.

## 7. Who presses the button: actor treemap that shrinks the humans

- **Target slide:** `adoption` beat 2/3 or `lessons` ("bots trigger bots").
- **Data:** distinct actors per category (`by_category.json`): Dr.CI 1 actor / 33,195 runs; autorevert 1 / 29,085; humans 135 / 8,789; triage reporters 1,375 / 4,453; manual 5 / 875; bot mention 1 / 428. Top humans (`top_actors.json`): jansel 3,757 · bobrenjc93 2,159 · ZainRizvi 1,585 …
- **Beats:**
  1. A treemap by *runs*: two huge amber tiles (one actor each) dwarf everything.
  2. Morph to a treemap by *distinct actors*: the two amber tiles shrink to slivers; triage's 1,375 reporters and 135 humans fill the frame. Same data, opposite picture.
  3. Caption: "2 bot accounts: 81 % of runs. 135 humans + 1,375 issue reporters: 17 %." (8,789 + 4,453 of 76,825).
- **Why it impresses:** a single `Transform` flips the audience's mental model of who the users are.
- **Manim:** hand-laid `Rectangle`s with `Transform`/`ReplacementTransform`, labels `FadeIn`.
- **Cost:** M (treemap layout done by hand for 6 rectangles).
- **Gimmick risk:** low; the morph is the message.

---

## Recommendation

Ship idea 1 (prototyped) on `adoption` beat 3 and idea 4 (S) on beat 2; keep idea 3 as the
`problem`-slide upgrade if time allows. Ideas 2 and 5 are the most spectacular and the most
likely to read as decoration in a 10-minute beginner-track talk.
