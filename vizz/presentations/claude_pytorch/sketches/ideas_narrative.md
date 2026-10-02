# Narrative hooks and transitions: ideas

Angle: the opening 15 s, the "agents vs agents" metaphor, the live-PR moment, and the close.
Semantic colors stay fixed: red = untrusted input, amber = Claude, green = trusted / human decision.
Facts and quotes come from `research/pytorch_repo.md` (§3, §7, §9). Text in "quotes" is verbatim.

---

## 1. The PR thread that types itself (prototyped: `proto_narrative.py`)

- **Target slide:** `example` (replaces the static two-card version; same 2 pauses + 1).
- **Beats:**
  1. The PR header appears, then a maintainer comment types out letter by letter:
     "@claude look for any subtle bugs on this pr". A `claude[bot] working` chip appears, and its clock counts 0m 00s → 4m 39s in about 2 s.
  2. The chip morphs into the reply card (High / Medium / Low). A diff hunk for `ScaledBlas.cpp` slides in under it. An amber box lands on `scale_a` inside the `scale_b_opt` line, and an arrow runs from the Medium row to it.
  3. The landed line (`scale_b.empty() … scale_b[0]`, `d9e65e8`) appears as a green `+` row. Stamp: "maintainers decided".
- **Why it lands:** everyone in the room has stared at a GitHub thread. The live typing makes the point that the interface is just a comment. The running clock gives "4m 39s" weight without a sentence about speed, and the diff highlight turns "copy-paste bug" into something the audience can see in a second.
- **Manim:** `AddTextLetterByLetter` on the comment body; `ValueTracker` + `always_redraw` clock; `ReplacementTransform(chip, reply)`; `SurroundingRectangle` on a sub-slice of a mono `Text`; `flow()` arrow from `common.py`.
- **Cost:** M (about 140 lines, all using the existing `comment`/`box`/`flow` helpers).
- **Gimmick risk:** low to medium. Keep the typing under 1.5 s and the clock under 2.5 s, or the presenter ends up waiting on the animation. The diff is a schematic of the line, not the literal C++, and is labeled that way on the slide.
- **Source:** §7.2: "@claude look for any subtle bugs on this pr", **4m 39s**, Medium "copy-paste error, with `scale_b_opt` built from `scale_a` in both the condition and the value (`ScaledBlas.cpp:454`)", High "`mat_a`/`mat_b` were undefined in the `AT_MKLDNN_ENABLED()` branch". Landed: "line 456 now uses `scale_b.empty() … scale_b[0]`", `d9e65e8cfed`.

## 2. "10_000 times": the comedy beat

- **Target slide:** a cold open before `title`, or a 1-pause insert between `shape` and `two_stage`, where it works as a breather and also shows that the bot is not a loop amplifier.
- **Beats:**
  1. malfet's comment types out: "@claude please review this PR 10_000 times". A counter next to it starts climbing (1, 2, 17, 340, 2,048…) while stacked ghost review cards pile up in red and start to overflow the frame.
  2. Hard cut: the pile collapses into one amber card that reads "I'll spare you 9,999 duplicate reviews and give you one good one." A small `58s` badge sits on it.
- **Why it lands:** it gets a laugh, and the laugh carries the thesis. The fear is that agents multiply volume, and here the agent pushes volume down. It is also a real quote from a well-known maintainer, so it lands better with this crowd than a made-up example would.
- **Manim:** `ChangeDecimalToValue` with a rate function that speeds up; around 30 `comment` ghosts with `LaggedStart(FadeIn)` and random small offsets; `ReplacementTransform(VGroup(ghosts), one_card)`.
- **Cost:** S.
- **Gimmick risk:** medium. The pile-up must be fast (≤ 1.5 s), and the punchline needs at least 2 s of stillness. Do not explain the joke.
- **Source:** §7.1 / §9: "@claude please review this PR 10_000 times" → "I'll spare you 9,999 duplicate reviews and give you one good one." (malfet on #176027, 58s).

## 3. Opening 15 s: streams converge, a filter forms

- **Target slide:** `title`.
- **Beats:**
  1. Black-ish paper, only the maintainer box (green) in the center-right. Small red dots (PRs/issues) start flowing in from three labeled sources: "contributors + agents", "bots + agents", "everyone + agents". Rate rises; the dots crowd around the box.
  2. An amber hairline arc (Claude) draws itself in front of the maintainer. The dots now pass through it: most get tagged (amber tick) and continue, a few bounce to a "triage review" slot, and the maintainer box receives a calm, ordered stream. The title fades in on the left: "Fighting Agents with Agents".
- **Why it lands:** it states the whole talk visually before anyone speaks. Volume is the problem, and a scoped agent in front of the human is the response. The maintainer is still the one at the end of the stream.
- **Manim:** dots as `Dot` with `MoveAlongPath` on `ArcBetweenPoints`, spawned by a `turn_animation_into_updater` or an `always` loop; `Create` for the arc; color swap on contact via updater checking x-position.
- **Cost:** M/L (particle timing and render time; keep around 60 dots).
- **Gimmick risk:** medium. A shield emoji or glowing force field would read as marketing, so keep it a hairline amber arc with a small "scoped · auditable · repo-aware" meta label. Never draw the maintainer being removed.
- **Source:** the brief/abstract: "Contributors use agents. Bots use agents. Everyone uses agents." `triage review` label: §8.12 ("adds `triage review` instead").

## 4. Autorevert ↔ Claude: two agents talking in a loop

- **Target slide:** `agents_vs_agents` (keeps the existing box diagram, but plays it as a chat first).
- **Beats:**
  1. Two chat bubbles alternate like a messaging app: `pytorch-auto-revert[bot]` (green): "early failure pattern on <suspect_commit>". `Claude CI Advisor` (amber) answers with a JSON chip: `{"verdict": "unsure", "confidence": …}`.
  2. The bubbles slide apart into the existing loop diagram, and the arrow animates a full circle once (dispatch → verdict → ClickHouse → autorevert).
  3. A principle bubble appears from Claude's side: "When in doubt between `unsure` and any dismissal, prefer `unsure`: a false dismissal wrongly clears a real regression."
- **Why it lands:** this is the literal "agents fighting agents" image the title promises. The chat form makes the loop understandable to beginners before they see boxes.
- **Manim:** `comment()` cards with `FadeIn(shift=UP)` alternating left/right; `Transform` bubbles into `box`es; `MoveAlongPath` of a small dot around the loop.
- **Cost:** M.
- **Gimmick risk:** low to medium. The bot's message is paraphrased from the dispatch inputs (`suspect_commit`, `signal_pattern`), so label it as a schematic and do not put it in quote marks.
- **Source:** §3: `workflow_dispatch` inputs `suspect_commit`, `pr_number`, `signal_pattern`; enum "related","unsure","not_related","infra_issue","garbage"; closing rule quoted above; "Evaluation Results (13/13 correct verdicts)".

## 5. "The power :D": the 17-minute PR as a timeline strip

- **Target slide:** transition into `adoption` (or the end of `shape`).
- **Beats:**
  1. A horizontal clock strip, 23:24Z → 23:41Z. Ticks pop in: 23:24 PR opened (Ivan opens `@claude` to anyone with write access). 23:30 "@claude please review this PR". 47s later: "Recommendation: Approve with the minor comment update suggestion". Zain: "The power :D". 23:39 merged.
  2. The strip zooms out and becomes the x-axis of the weekly-usage chart on `adoption`, with the #176027 marker sitting where the strip was.
- **Why it lands:** it ties the human story ("Claude reviewed the PR that opened Claude to everyone") to the data slide that follows. The audience sees cause and then effect.
- **Manim:** `NumberLine` ticks plus `comment` mini-cards; `self.camera.frame` is not available in `Slide`, so do the zoom by `Transform`ing the strip into the chart axis.
- **Cost:** M (depends on the adoption chart reusing the same axis object).
- **Gimmick risk:** low. The risk is spending 40 s on a story that sits next to a chart, so it needs one pause at most.
- **Source:** §7.1: opened 23:24Z; ZainRizvi "@claude please review this PR"; 47s; "Recommendation: Approve with the minor comment update suggestion"; "The power :D"; merged about 17 minutes after opening.

## 6. Closing callback: the title diagram, resolved

- **Target slide:** `closing`.
- **Beats:**
  1. The title's three-stream diagram reappears in its final state (idea 3), but each amber stage is now labeled with the slide it came from: `@claude` · triage · PR review · CI advisor.
  2. The amber layer dims, the green maintainer box stays lit, and the punchline appears under it: "Not replacing maintainers. Keeping the bar." After that, the try-it code block.
- **Why it lands:** a visual callback closes the loop on the opening hook. Dimming the agent layer while the human stays lit is the thesis in one motion.
- **Manim:** reuse the title-slide builder function (return the VGroup); `set_opacity` animation; existing `punchline()`.
- **Cost:** S if idea 3 exists, M otherwise.
- **Gimmick risk:** low.
- **Source:** brief "One thing to remember"; the closing slide's existing punchline.

## 7. Diff-line transition: "Both real."

- **Target slide:** the `example` → `shape` handoff.
- **Beats:** The highlighted `scale_a` token from idea 1 shrinks into an `@claude` token, and the camera seems to "enter" it. The token becomes the first box of the `shape` pipeline ("comment → gate → …"). The line the speaker says over it: "So what actually happens when you type that?"
- **Why it lands:** it turns a hard slide cut into motivated motion, from what it found to how it runs.
- **Manim:** carry a mobject across modules (needs `clear_stage` to keep one mobject, or rebuild it at the same position and `ReplacementTransform`).
- **Cost:** M (cross-module state is new for this deck).
- **Gimmick risk:** medium. Cross-slide transforms break when slides are reordered or previewed on their own.
- **Source:** same as idea 1; pipeline from `scenes.md` / §1.

---

### Recommendation

Prototype idea 1 (done). Ship idea 2 as the cold open if the speakers are comfortable with a 15 s laugh before the title. Otherwise ship idea 3 for the title and idea 6 as its callback. Idea 4 is the best upgrade to `agents_vs_agents` per minute of effort.
