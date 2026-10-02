# Presentation brief

- **Working title:** Fighting Agents with Agents: Bringing Claude to PyTorch CI, triage, and PR review
- **Venue:** PyTorch Conference North America 2026, Oct 20, 12:35-12:45 PM, LL21D. Core PyTorch, Beginner, 10-minute lightning talk.
- **Speaker:** Driss Guessous (PyTorch @ Meta). Thanks: Ivan Zaitsev (opened `@claude` to all contributors, pytorch/pytorch#176027).
- **Audience:** PyTorch contributors and maintainers; beginner track, so explain GitHub Actions/OIDC in one line each, no deep infra.
- **One thing to remember:** agents raise the volume of code and issues; maintainers need agent-shaped infra (scoped, auditable, repo-aware) to keep the bar, not replace it.
- **Length:** ~9-10 slides, ~1 min each.
- **Output:** manim-slides present + pptx export (`uv run manim-slides convert ClaudePytorchDeck claude_pytorch.pptx`).
- **Must be technically accurate:** yes. Every workflow detail and adoption number traces to `research/`.
- **Style:** Nuggets light (same palette as the FlexGEMM deck at the same conference). Semantic colors: green = trusted/privileged/maintainer, amber = agent/Claude, red = untrusted input.

## Abstract (as submitted)

PyTorch maintainers are reviewing an ever increasing number of PRs written by AI agents. Contributors use agents. Bots use agents. Everyone uses agents. So the question for us was pretty direct: if agents are going to increase the amount of code and issues flowing into PyTorch, what tools do maintainers need to keep up without lowering the bar?

This talk is about how we brought Claude into PyTorch infra. We started with `@claude` on issues and PRs, then added automatic issue triage, reusable onboarding for `pytorch` and `meta-pytorch` repos, PR review skills, and CI/autorevert investigation.

We will walk through the workflow shape, Bedrock/OIDC setup, two-stage GitHub Actions, tool allowlists, repo-specific skills, and the adoption trend after launch. The goal is not to replace maintainers. The goal is to give maintainers agent-shaped infra for an agent-shaped world.

## Sources

- Research reports (generated from local checkouts + `gh`): `research/pytorch_repo.md`, `research/test_infra.md`, `research/adoption.md`, raw CSVs in `research/data/`.
- pytorch/pytorch `.github/workflows/claude*.yml`, `.claude/skills/`, `.claude/hooks/`.
- pytorch/test-infra reusable Claude workflows and usage telemetry.
- Example usage: https://github.com/pytorch/pytorch/pull/176266
- Opened to all contributors: https://github.com/pytorch/pytorch/pull/176027
- Threat model doc (internal Google Doc, linked from the announcement post); do not quote beyond what is public in the workflows.

## Open questions

- Which adoption numbers are OK to show publicly (runs/week, distinct users, cost)?
- Is there a specific `@claude` example (PR review or CI investigation) you want as the demo slide?
