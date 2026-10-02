# Claude usage from HUD ClickHouse (`misc.claude_code_usage`)

Collected 2026-10-02 with the `hud` CLI (`~/meta/hud`, `uv run hud gcx chq ... --json`).
Raw JSON/CSV: `data/hud/`. Category expression: `data/hud/category.sql`.

One row per Claude run (85,394 rows, 85,394 distinct `(repo, run_id, run_attempt)`).
Table starts 2026-01-24. Cost is the action's **estimated list-price `total_cost_usd`**, not billed spend.

## All repos

- 85,394 runs, 13 repos, 1,471 distinct actors, ~$78k estimated.
- pytorch/pytorch: 76,825 runs (90%), 1,428 actors, ~$57k.
- Others by runs: meta-pytorch/pytorch-gha-infra 3,364; pytorch/test-infra 2,798; executorch 629; ao 493; ciforge 454; metainternal/pytorch-gha-infra 397; torchtitan 273; helion 81; ci-infra 44; tutorials 31; attention-gym 3; torchcomms 2.
- Models (runs): Opus 4.6 63k, Opus 5.5 6.0k, Opus 5 4.5k, Sonnet 4.5 3.5k, Opus 4.8 3.4k, Haiku 4.5 2.1k, Sonnet 5 1.6k, Fable 5 365.

## pytorch/pytorch by category (whole table)

| category | runs | distinct actors | est. cost | p50 min | p50 cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dr.CI advisor dispatch (`pytorch-bot[bot]`) | 33,195 | 1 | $16.3k | 0.9 | $0.40 |
| Autorevert advisor (`pytorch-auto-revert[bot]`) | 29,085 | 1 | $17.0k | 1.2 | $0.52 |
| Human `@claude` mentions (`issue_comment`, non-bot) | 8,789 | 135 | $20.8k | 3.2 | $1.42 |
| Issue triage (`workflow_run`/`issues`) | 4,453 | 1,375 reporters | $1.5k | 1.1 | $0.32 |
| Manual dispatch | 875 | 5 | $0.4k | 1.4 | $0.25 |
| Bot `@claude` mentions | 428 | 1 | $1.2k | 2.9 | $1.70 |

Interpretation: humans are 11% of runs but 36% of cost (long interactive sessions);
machines (Dr.CI + autorevert) are 81% of runs at ~$0.5/run.

## Human `@claude` (pytorch/pytorch)

- Before 2026-02-28 (hardcoded allowlist; #176027 merged 2026-02-27): 13 humans, 95 runs.
- From 2026-02-28: **135 humans, 8,695 runs, on 4,102 distinct issues/PRs**.
  p50 $1.43 / 3.3 min, p90 $5.51 / 28.4 min.
- Weekly runs: ~15-30/week in Feb → ~100-150/week in Mar-Apr → peak 891/week (2026-06-01) →
  ~200-300/week through Aug-Sep, with spikes of 513 and 793 in Sep.
- Weekly distinct humans: 4-7 in Feb → ~20 in Mar-Apr → 30-39 from late May on.
- Top human invokers: jansel 3,757; bobrenjc93 2,159; ZainRizvi 1,585; izaitsevfb 391; drisspg 311; ezyang 260; Skylion007 207; vkuzo 200; zou3519 197; atalman 181.

## Issue triage

- 4,453 runs from 1,375 distinct issue reporters; ~500-680 issues/month since March.
- Cheapest workload: p50 $0.32, ~1 min.

## Aug 19-21 spike: advisor-coverage backfill (intentional)

- 3 days: 4,912 + 6,179 + 10,930 = **~22k advisor runs, ~$12.4k estimated**, all `pr_number=0`,
  vs ~10-70 runs/day around it; 76% of all autorevert-advisor rows.
- Cause: pytorch/test-infra #8569 (Jean Schmidt, 2026-08-19) added the `pytorch-advisor-coverage`
  Lambda, which dispatches the advisor on trunk reds with no verdict ("~40% of trunk reds") and
  supports a resumable historical backfill; #8585 (2026-08-20) fixed its token mint and floored
  backfill dispatch gaps at 5 s. Coverage verdicts use a `coverage_` key prefix so they never trigger reverts.
- Not a runaway loop. It is a deliberate one-off historical classification of flaky trunk reds.

## Caveats

- `actor` for `workflow_run` is the upstream trigger actor (issue reporter for triage).
- Category split is heuristic (event + actor); Dr.CI and autorevert runs all have `pr_number=0`.
- Costs are estimates; confirm what can be shown publicly.

## Intake context (pytorch/pytorch, `default.pull_request`, `default.issues`)

PRs opened per month (`countDistinct(number)`; `data/hud/pr_intake_monthly.json`):
~1.3-1.5k/month in early 2025 → 1.9k Jan 2026 → 2.3k Mar → 3.0k May → 3.2k Aug → 3.0k Sep 2026. Roughly 2x in 18 months.

PRs by author association per quarter (`data/hud/pr_intake_quarter_assoc.json`):

| quarter | CONTRIBUTOR | COLLABORATOR | MEMBER | NONE (first-time) |
| --- | ---: | ---: | ---: | ---: |
| 2025 Q1 | 2,743 | 1,078 | 258 | 190 |
| 2025 Q3 | 3,449 | 1,235 | 218 | 195 |
| 2026 Q1 | 4,113 | 1,055 | 182 | 636 |
| 2026 Q2 | 5,298 | 1,615 | 170 | 643 |
| 2026 Q3 | 5,044 | 2,077 | 294 | **1,824** |

First-time-author PRs: 190 → 1,824 per quarter (~10x). Issue intake is flat (~1.5-2.3k/quarter).
Caveat: author_association is a snapshot from the webhook payload.
