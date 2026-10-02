# Fighting Agents with Agents

## Overview
- Audience: PyTorch contributors/maintainers, beginner track. 10-minute lightning talk, two speakers.
- Question: if agents multiply the PRs and issues flowing into PyTorch, what do maintainers need to keep the bar?
- Takeaway: give maintainers agent-shaped infra: scoped, auditable, repo-aware. Not a replacement.
- Semantic colors (every slide): red = untrusted input, amber = the agent / Claude, green = trusted / privileged / human decision.
- Facts trace to `research/` (`hud_usage.md`, `pytorch_repo.md`, `test_infra.md`). Data collected 2026-10-02.
- Speaker split (proposal): Driss 1-4, 6-7, 10, 12; Ivan 5, 8-9, 11.

## title
- Kicker, title, subtitle, speakers (Driss Guessous, Ivan Zaitsev).
- Visual: three incoming streams (contributors+agents, bots+agents, everyone+agents) into one maintainer box.

## problem — "Agents changed the intake"
- Beat 1: PRs opened per month, pytorch/pytorch, Jan 2025 → Sep 2026 bar chart (~1.5k → ~3k).
- Beat 2: first-time-author PRs per quarter 190 → 1,824 (~10x). Issues flat.
- Beat 3: the question + AI_POLICY quote: "We do not accept contributions created by fully autonomous agents."

## timeline — "Nine months, one bot at a time"
- 2026-01-16 `@claude` pilot (20 handles) · 01-28 issue triage · 02-27 any maintainer with write access (#176027) · 03-06 reusable workflow in test-infra · 03-13 autorevert advisor · 07 Dr.CI auto-dispatch · 08 public execution logs · 09 hardened PR review.
- Beat 1: interactive line (Jan-Mar). Beat 2: CI agents (Mar-Sep).

## example — "`@claude` on a real PR"
- PR #176266 (external contributor, scaled_mm_v2 CPU). Comment bubble: "@claude look for any subtle bugs on this pr".
- Beat 1: comment. Beat 2: Claude reply after 4m39s: High: MKLDNN branch uses undefined `mat_a`/`mat_b` (compile error); Medium: `scale_b_opt` built from `scale_a` (copy-paste). Beat 3: both fixed before landing (`d9e65e8`); humans pushed back on the low-severity note. The maintainer still decided.

## shape — "What happens when you type @claude"
- Pipeline: comment → gate → `environment: bedrock` → GitHub OIDC → AWS role → Bedrock → claude-code-action → reply / public log / usage row.
- Beat 1: gate checks (org, mention, OWNER/MEMBER/COLLABORATOR, write access API check). Beat 2: no API keys: OIDC → role scoped to `repo:<org>/<repo>:environment:bedrock`, env deploys from main only. Beat 3: outputs: reply, public S3 log, ClickHouse row.

## two_stage — "Untrusted input never meets a write token"
- Issue triage: stage 1 (`issues: opened`, contents: read, no secrets, 2 min, writes `issue_number.txt`) → artifact → stage 2 (`workflow_run` from main, bedrock env, `issues: write`, re-validates).
- Beat 1: stage 1. Beat 2: artifact + stage 2. Beat 3: quote: "DO NOT add this workflow as a required status check: a prompt injection could then fail it deliberately to block every merge."

## guardrails — "The prompt is not the only defense"
- Left: triage `--allowedTools` (five GitHub MCP tools, no shell). Right: hooks: issue-target validator, 282-label allowlist, `bot-triaged` audit label. Bot can never add `high priority` / `sev` / `merge blocking`; adds `triage review` for a human.

## skills — "Repo knowledge as skills; onboarding as one workflow"
- Beat 1: 18 skills in `.claude/skills/` grouped: bots (triaging-issues, distributed-triage, pr-review-readiness), shared (pr-review), developer (fix-issue, pt2-bug-basher, metal-kernel, ...). CLAUDE.md = AGENTS.md.
- Beat 2: caller YAML (`uses: pytorch/test-infra/.github/workflows/_claude-code.yml@main`) + "13 repos reporting usage".

## agents_vs_agents — "Agents investigating agents' failures"
- Autorevert bot detects a breaking signal → dispatches Claude CI Advisor → JSON verdict `related | unsure | not_related | infra_issue | garbage` → ClickHouse → autorevert reads it back. Dr.CI dispatches the same advisor on PR failures with brakes (≤32 runs/PR, skip if >8 new failures).
- Beat 3: prompt lines ("A missing baseline is not a green baseline." "When in doubt … prefer unsure.") + coverage lambda (#8569, Aug 19): ~40% of trunk reds had no verdict; ~22k backfilled in 3 days (the Aug 19-21 usage spike; intentional, verdicts never trigger reverts).

## adoption — "Usage after launch"
- Beat 1: weekly human `@claude` runs on pytorch/pytorch (Jan 26 → Sep 28) with #176027 marker.
- Beat 2: stats: 135 people, 8.7k runs on 4.1k issues/PRs since Feb 28; 4.4k issues triaged from 1,375 reporters; 13 repos.
- Beat 3: machines vs humans: CI advisors 81% of runs at ~$0.5 median; humans 11% of runs, p50 3.3 min.

## lessons — "With great power…"
- Bots trigger bots (one allowlisted bot through the gate; Dr.CI ≤32 runs/PR, skip outages). 1h credentials → 55-min jobs + time-budget hook. `pull_request_review_comment` removed (ran PR-branch code). Trust needs receipts: public transcripts, bot-triaged label.

## closing
- Punchline: agent-shaped infra for an agent-shaped world. How to use: `@claude review this PR`, `@claude fable effort=high ...`, onboard a repo with one `uv run` script.

## Open questions
- Can we show cost (estimated $) publicly? Slides omit dollars; `research/hud_usage.md` has estimates.
- Adoption numbers: HUD `misc.claude_code_usage` (starts 2026-01-24). GitHub Actions API counts are capped and include skipped runs; not used.
- Speaker split confirmed with Ivan?
- "Opened to all contributors" in the announcement vs code: gate is write access (OWNER/MEMBER/COLLABORATOR + write permission check). Slides say "anyone with write access".
