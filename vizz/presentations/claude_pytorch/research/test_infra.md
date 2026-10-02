# Claude in PyTorch CI — test-infra research notes

Talk: *Fighting Agents with Agents: Bringing Claude to PyTorch CI, triage, and PR review* (10-min lightning).

Sources (read-only):
- `~/meta/test-infra` local `main` (HEAD `6f37b78f`, 2026-08-14). **Stale** vs upstream; current upstream state fetched with `gh api` (upstream HEAD `c3f3d1c2`, 2026-10-02). Differences are called out.
- `~/meta/test-infra-claude-logs`, `~/meta/test-infra-claude-telemetry` (branch `claude-execution-logs`, prototype commits `61ff0f6` / `07da913b`; landed as test-infra #8381).
- `~/meta/claude-code-action-progress` (branch `safe-progress`, fork `drisspg/claude-code-action-progress` of `anthropics/claude-code-action`).
- `~/meta/ciforge-claude-effort` (branch `claude-effort-keywords`, fork `drisspg/ciforge`).
- `gh` against pytorch/test-infra, pytorch/pytorch, pytorch/ao, and org-wide `gh search code`.

---

## 1. Reusable workflow + composite action

| Artifact | Path (pytorch/test-infra) | Role |
|---|---|---|
| Reusable workflow | `.github/workflows/_claude-code.yml` | `workflow_call` for `@claude` mentions in issues/PR comments |
| test-infra's own caller | `.github/workflows/claude-code.yml` | `uses: ./.github/workflows/_claude-code.yml` |
| Composite action | `.github/actions/upload-claude-usage/action.yml` | Usage metrics → S3 → ClickHouse |
| Onboarding script | `.github/scripts/setup-claude-environment.py` | Creates `bedrock` GH environment + caller YAML |

### `_claude-code.yml` inputs (upstream main)

| Input | Default | Notes |
|---|---|---|
| `model` | `global.anthropic.claude-opus-5-5` | Bedrock model id (local stale main: `claude-opus-4-8`) |
| `timeout_minutes` | `60` | Job timeout |
| `additional_claude_args` | `''` | e.g. `--allowedTools`, `--mcp-config` |
| `settings` | `{"alwaysThinkingEnabled": true}` | Claude settings JSON |
| `show_full_output` | `false` | "PUBLIC: Show complete Claude messages and tool output in the job log" |
| `upload_execution_log` | `false` | Upload execution JSON to S3 after exit |
| `execution_log_s3_prefix` | `review-logs/claude-code` | |
| `setup_script` | `''` | e.g. install lintrunner |
| `append_system_prompt` | `''` | Passed via `APPEND_SYSTEM_PROMPT` env |

Action pin (upstream): `anthropics/claude-code-action@9ca9355b… # Claude Code 2.1.280` (since #8865, 2026-09-22). Before that: `@593d7a5c… # v1.0.141`. Pinned to upstream Anthropic action since #8012 (2026-04-24, "switch claude-code-action pin to upstream anthropics/claude-code-action@v1.0.104"); earlier it pointed at a non-upstream pin.

### Security gates baked into the reusable workflow
From `_claude-code.yml`:
- Job `if:` requires all of: owner ∈ `["pytorch","meta-pytorch"]`; `@claude` in the comment/issue body; `author_association` ∈ `OWNER/MEMBER/COLLABORATOR` **or** actor `pytorch-auto-revert[bot]`.
- "Verify write access" step: `getCollaboratorPermissionLevel` must be `admin` or `write` (allowed-bot bypass).
- `pull_request_review_comment` is "intentionally NOT supported" because that event runs workflow code from the PR branch.
- ghstack PRs (`head.ref` matches `^gh/[^/]+/\d+/head$`) run **read-only**: `--disallowedTools "Edit,Write,NotebookEdit,Bash(git push:*),Bash(git add:*),Bash(git commit:*),Bash(git rm:*),mcp__github_file_ops__commit_files,mcp__github_file_ops__delete_files"` (#7834).
- Commits authored as the invoking user: `GIT_AUTHOR_NAME`/`GIT_AUTHOR_EMAIL=<actor>@users.noreply.github.com` (#7832).

Slide snippet — the gate:
```yaml
if: |
  contains(fromJSON('["pytorch","meta-pytorch"]'), github.repository_owner) &&
  ( (github.event_name != 'issues' && contains(github.event.comment.body, '@claude')) ||
    (github.event_name == 'issues' && contains(github.event.issue.body, '@claude')) ) &&
  ( contains(fromJSON('["OWNER","MEMBER","COLLABORATOR"]'),
      github.event.comment.author_association || github.event.issue.author_association) ||
    github.actor == 'pytorch-auto-revert[bot]' )
```

### Onboarding: the minimal caller YAML
Generated verbatim by `setup-claude-environment.py` (`WORKFLOW` constant) and written to `.github/workflows/claude-code.yml`:
```yaml
name: Claude Code
on:
  issue_comment:
    types: [created]
  issues:
    types: [opened]
jobs:
  claude-code:
    uses: pytorch/test-infra/.github/workflows/_claude-code.yml@main
    permissions: {contents: read, pull-requests: write, issues: write, id-token: write}
    secrets: inherit
```
(Original has `permissions:` as a 4-line block; flattened here for slide length.)

Onboarding steps (header comment of `_claude-code.yml` + script summary):
1. `uv run https://raw.githubusercontent.com/pytorch/test-infra/main/.github/scripts/setup-claude-environment.py` (uv ≥ 0.5.0) — creates/validates a `bedrock` GitHub environment restricted to branch `main` (`custom_branch_policies: true`, `ALLOWED_BRANCHES = ["main"]`), writes the caller YAML; refuses orgs other than `pytorch`/`meta-pytorch`.
2. Add `repo:<org>/<repo>:environment:bedrock` to the OIDC trust policy in configerator `raw_configs/cloud/strata/fbossci/iam/main.tf`.
3. Install the Claude GitHub App (https://github.com/apps/claude).
4. Add `CLAUDE.md` and commit the caller workflow.

### pytorch/pytorch's customized caller (upstream, 2026-09-28)
`pytorch/pytorch/.github/workflows/claude-code.yml` overrides: `timeout_minutes: 55`, keyword model switch, keyword effort, retry env, time-budget hooks, public logs, PR-review system prompt.

Model keyword (`fable` in the comment → Fable, else Opus 5.5):
```yaml
model: ${{ ((github.event_name == 'issue_comment' && contains(github.event.comment.body, 'fable'))
  || (github.event_name == 'issues' && contains(github.event.issue.body, 'fable')))
  && 'global.anthropic.claude-fable-5' || 'global.anthropic.claude-opus-5-5' }}
```
Effort keyword (`effort=max|high|low`, default `medium`; "Keep the comment text out of claude_args: only these hardcoded effort values can reach the CLI"):
```yaml
additional_claude_args: >-
  --allowedTools Skill --effort
  ${{ (... contains(github.event.comment.body, 'effort=max') ...) && 'max' ||
      (... 'effort=high' ...) && 'high' ||
      (... 'effort=low' ...) && 'low' || 'medium' }}
```
System prompt: "When asked to review a PR, always use the /pr-review skill first … put the details of the PR review inside a GitHub collapsible section and a one sentence summary outside."

---

## 2. Bedrock / OIDC

- Role: `arn:aws:iam::308535385114:role/gha_workflow_claude_code`, region `us-east-1`, via `aws-actions/configure-aws-credentials@v4` with `id-token: write`.
```yaml
environment: bedrock
permissions: {contents: read, pull-requests: write, issues: write, id-token: write}
steps:
  - uses: aws-actions/configure-aws-credentials@v4
    with:
      role-to-assume: arn:aws:iam::308535385114:role/gha_workflow_claude_code
      aws-region: us-east-1
  - uses: anthropics/claude-code-action@9ca9355b...  # Claude Code 2.1.280
    with:
      use_bedrock: "true"
      claude_args: "--model ${{ inputs.model }} ..."
```
- **Trust policy scoping:** OIDC subject `repo:<org>/<repo>:environment:bedrock` (per-repo allowlist), combined with the GitHub `bedrock` environment's deployment-branch policy = `main` only. So only main-branch workflow definitions in allowlisted repos can assume the role. The IAM terraform lives in configerator (`raw_configs/cloud/strata/fbossci/iam/main.tf`), **not in test-infra**; I could not read it — exact conditions beyond the documented subject string are unverified.
- Session lifetime: role default 1h session. Comments in greenlight/vLLM workflows: "the gha_workflow_claude_code role's default 1h session covers the 37-min model timeout plus setup". pytorch #198521 lowered `@claude` timeout to 55 min: "The AWS session and Claude's GitHub App token … expire 1h after job start and are never refreshed; stay a few minutes under 60." (Briefly raised to 2h in #196537 before that.)
- **Model ids seen** (all Bedrock cross-region inference profiles):
  - reusable default history: `global.anthropic.claude-opus-4-6-v1` (#7810) → `claude-opus-4-8` (#8162, drisspg) → `claude-opus-5-5` (#8865). Earlier #7805 "remove :0 suffix for Opus 4.6".
  - pytorch `@claude`: `global.anthropic.claude-opus-5-5`, opt-in `global.anthropic.claude-fable-5`.
  - autorevert advisor, greenlight, vLLM triage: `global.anthropic.claude-opus-5-5`.
  - pytorch issue triage: `global.anthropic.claude-sonnet-5`.
  - log-classifier Lambda (Rust, `aws/lambda/log-classifier/src/bedrock.rs`): primary `us.anthropic.claude-haiku-4-5-20251001-v1:0`, fallback `us.anthropic.claude-sonnet-4-6` only when the first answer is unusable.
- **Retries/timeouts** (pytorch caller `settings.env`, #191256 "Bound Claude Bedrock request retries"):
```json
"env": {
  "API_TIMEOUT_MS": "180000",
  "CLAUDE_CODE_MAX_RETRIES": "2",
  "CLAUDE_ENABLE_STREAM_WATCHDOG": "1",
  "CLAUDE_TIME_BUDGET_MINUTES": "55"
}
```
  plus `SessionStart`/`SubagentStart`/`PostToolBatch` hooks running `.claude/hooks/claude_code/time-budget.sh` (#198687): sends "Time check" notes about every 3 min with convergence/posting windows.

---

## 3. Usage telemetry

Pipeline: `claude-code-action` writes `$RUNNER_TEMP/claude-execution-output.json` → `upload-claude-usage` (`if: always()`) extracts the SDK `type == "result"` message with `jq` → `s3://ossci-raw-job-status/claude_code_usage/<org>/<repo>/<run_id>_<attempt>.json` → `clickhouse-replicator-s3` Lambda (`claude_code_usage_adapter`, mapping `"claude_code_usage": "misc.claude_code_usage"`) → ClickHouse.

Metrics extracted (from `action.yml`):
```
duration_ms, num_turns, total_cost_usd,
input_tokens, output_tokens,
cache_read_input_tokens, cache_creation_input_tokens,
model   # first key of .modelUsage
```
plus context: `repo, run_id, run_attempt, actor, event_name, pr_number, timestamp`. Scheduled runs record actor as `"<workflow> (scheduled)"` instead of whoever last edited the file (#8138). Hardened against injection/malformed metrics, best-effort (warn + exit 0) in #8536.

ClickHouse table `misc.claude_code_usage` (`clickhouse_db_schema/misc.claude_code_usage/schema.sql`): SharedMergeTree, `ORDER BY (repo, timestamp, run_id)`; token/model columns added with defaults in #7729.

Saved HUD queries (`torchci/clickhouse_queries/`): `claude_code_usage_daily`, `_by_repo`, `_by_actor`, `_by_workflow`. Daily query joins `default.workflow_job` for workflow name and reports invocations, `total_cost`, `total_turns`, `total_minutes`, avg cost/turns per invocation.

Dashboard: HUD page `/claude_billing` (`torchci/pages/claude_billing.tsx`, NavBar "Claude Billing") embeds Grafana public dashboard `9127e39ec5a7410ebb419fac06a08ca0` (`pytorchci.grafana.net`). Gated behind login + `/api/torchagent-check-permissions` (#7960); labeled as **estimated** costs, not actual bills (#7962).

Other Claude-related ClickHouse tables: `misc.autorevert_advisor_verdicts` (§4), `misc.greenlight_pr_state` (greenlight verdicts).

Actual spend/volume numbers: **not gathered** — require HUD login / ClickHouse credentials.

---

## 4. Autorevert / CI investigation integration

Three layers:

1. **`@claude` on revert comments** (`aws/lambda/pytorch-auto-revert/pytorch_auto_revert/signal_actions.py`): when autorevert requests a pytorchbot revert, the comment appends:
   > "@claude Can you please read this revert comment, follow the links and read the errors, to then give a brief diagnostics on the cause of the error? If you judge the error to be legitimate reason for a revert, please provide brief guidance on how the author could fix it."

   This is why `pytorch-auto-revert[bot]` is the one bot allowed through the reusable workflow gate (pytorch #173422, "[CLAUDE BOT] respond to autorevert inquiries", 2026-01-26).

2. **AI autorevert advisor** — `pytorch/pytorch/.github/workflows/claude-autorevert-advisor.yml` (added #177404, izaitsevfb, 2026-03-13). Dispatched by the autorevert Lambda via `workflow_dispatch` with `suspect_commit`, `pr_number`, `signal_pattern` (JSON of failed/successful/unknown/prior commit partitions with job/log URLs and test rows). Capped per workflow+commit (`ADVISOR_CAP_PER_WORKFLOW_COMMIT`); dispatch POST uses retry=0 to avoid duplicate runs.
```yaml
timeout-minutes: 15
environment: bedrock
- uses: anthropics/claude-code-action@9ca9355b...
  with:
    allowed_bots: "pytorch-auto-revert[bot],pytorch-bot[bot]"
    claude_args: >-
      --model global.anthropic.claude-opus-5-5
      --allowedTools "Bash,Read,Glob,Grep,WebFetch"
      --json-schema '{"type":"object","required":["verdict","confidence","summary","causal_reasoning"], ...}'
```
   Verdict enum: `related | unsure | not_related | infra_issue | garbage` (`revert` kept as alias of `related`, #8023; `infra_issue` added #8213, treated like `not_related`). Prompt encodes reasoning principles, e.g. "Pending state carries no evidence", "A missing baseline is not a green baseline", "`related` outranks every dismissal". Verdicts → `misc.autorevert_advisor_verdicts` (#7906) and are **read back by autorevert to influence revert decisions** (#7908, "Read AI advisor verdicts from CH and use in autorevert decisions").

3. **Dr.CI advisor** (`torchci/lib/advisor/advisorConfig.ts`): Dr.CI can dispatch the same advisor on new PR failures (manual button + auto-dispatch behind `DRCI_ADVISOR_AUTODISPATCH_ENABLED`, launched dark in #8178). Only `pytorch/pytorch` enabled: `maxNewFailures: 8` (outage guard; bypass label `ci-no-td`), `maxDispatchPerPr: 32`. Verdict rendered inline in the Dr.CI comment (#8202, #8235).

Related CI-agent workflows:
- **Green Light** (test-infra `greenlight/`, `.github/workflows/greenlight-pr-review.yml`, Jean Schmidt, MVP #8363 2026-08-03): scanner Lambda every 5 min picks trusted-author pytorch/pytorch PRs, dispatches a Claude reviewer that emits `LAND`/`NO_LAND`. Privilege split: `announce_start` / `review` (UNPRIVILEGED: Bedrock only, `--allowedTools "Read,Glob,Grep,Write"`, 37-min model timeout) / `record` (PRIVILEGED, no model, holds the App key). Hardening: sanitize untrusted `CLAUDE.md`/`.claude` from the checkout (#8435), credential-exfil hardening (#8481), read/write-restriction hooks, fail-closed `assert-loaded-instructions.py`.
- **vLLM torch-nightly triage** (`vllm-torch-nightly-triage.yml`, Andrey Talman, #8447 2026-08-06): cron `0 17 * * 2,3,5`, deterministic A/B detection of vLLM jobs failing on torch nightly but passing on stable, then a Claude root-cause job (Read-only, 30 min; Buildkite token fetched outside the model "so the token never enters the agent's tool surface"). Files issues into test-infra by default since #8642.
- **pytorch/ao `ci-failure-issue.yml`**: on `workflow_run` failure on main, read-only Claude triage emits a structured decision; separate Claude-free `act` job creates/comments one `ci-failure` issue. "Patterned on pytorch/pytorch's claude-autorevert-advisor.yml."
- pytorch/pytorch also has `claude-issue-triage-run.yml` (Sonnet 5, GitHub MCP issue tools only), `claude-distributed-triage.yml`, `hardened-pr-review-run.yml` (staged PR review: prepare/review/publish), and `.github/actions/auto-pr-triage`.
- Non-agent Bedrock use: log-classifier Lambda (Haiku 4.5 → Sonnet 4.6 fallback) picks the error line for HUD (#8391, #8461).

---

## 5. Execution logs & live progress

**Landed: test-infra #8381** (drisspg, merged 2026-07-31) — "Preserve Claude execution logs for review inspection". Prototyped in `~/meta/test-infra-claude-logs` / `-telemetry`. Adds `show_full_output`, `upload_execution_log`, `execution_log_s3_prefix`, "without modifying or forking `anthropics/claude-code-action`". Two keys per run:
```text
review-logs/claude-code/<owner>/<repo>/<issue-or-pr>/<comment-id>.json
review-logs/claude-code/<owner>/<repo>/<issue-or-pr>/runs/<run-id>_<attempt>.json
```
"The comment-keyed object lets a viewer start from a PR, enumerate its `@claude` mentions through the GitHub API, and load each review independently without maintaining a separate S3 manifest." Bucket: `s3://ossci-raw-job-status` (public). Validated on ciforge run 30318082793 (10-event stream, 4-turn result). Upstream later hardened it with `continue-on-error: true`, `timeout-minutes: 2`, and "PUBLIC:" labels on the inputs.

Adopted by pytorch #191969 ("[CI] Enable Claude execution logs", 2026-08-04) with:
```yaml
# These logs are public and may include assistant messages and tool output.
show_full_output: true
upload_execution_log: true
```
and in ciforge (`~/meta/ciforge-claude-effort` commit `3b19088`). ciforge also has "Discover Claude traces directly from S3" (`fdfdee8`).

**Fork (not upstreamed): `drisspg/claude-code-action-progress`, branch `safe-progress`**, on top of upstream `593d7a5` (v1.0.141). Commits `92db82f` "Add sanitized live progress logging" and `8c6c7fe` "Omit unavailable stop reasons from progress" (2026-07-27), touching `base-action/src/run-claude-sdk.ts` (+tests). Behind `CLAUDE_CODE_ACTION_PROGRESS=1` (and only when not `show_full_output`), `summarizeSdkProgress()` prints metadata only — "without exposing prompts, responses, or tool arguments":
```text
[claude-progress +42s] API request started
[claude-progress +57s] model response received; stop=tool_use; input_tokens=..; output_tokens=..; tools=Bash,Read
[claude-progress +58s] tool execution completed; results=2; errors=0
[claude-progress +63s] API retry 1/2 after HTTP 529; waiting 4000ms
[claude-progress +180s] no SDK event for 60s; last=API request started
```
(Format strings from the code; numbers illustrative.) A 60 s heartbeat surfaces stalled Bedrock requests. `gh pr list -R anthropics/claude-code-action --author drisspg` returns no PRs, so this is a local fork only. The production fix instead combined full public output (#8381) with bounded retries / stream watchdog (#191256) and the time-budget hook.

---

## 6. Onboarded repos

`gh search code "_claude-code.yml" --owner {pytorch,meta-pytorch}` (GitHub code index; may miss repos) + `gh api commits?path=.github/workflows/claude-code.yml` for first commit:

| Repo | First caller commit | By | Notes |
|---|---|---|---|
| pytorch/pytorch | 2026-01-16 (own workflow); reusable since 2026-03-06 (#176724) | izaitsevfb / ZainRizvi | 34 commits to the file; plus advisor, issue triage, distributed triage, hardened review |
| pytorch/ciforge | 2026-01-21 | izaitsevfb | 50 commits; sandbox/prototyping repo |
| pytorch/test-infra | 2026-02-26 (#7798) | izaitsevfb | local `./` caller; greenlight, vLLM triage |
| pytorch/tutorials | 2026-03-02 | sekyondaMeta | |
| pytorch/executorch | 2026-03-04 | JacobSzwejbka | 5 commits |
| pytorch/ao | 2026-03-09 | drisspg | plus `ci-failure-issue.yml` |
| pytorch/torchtitan | 2026-03-09 | tianyu-l | |
| meta-pytorch/torchcomms | 2026-05-01 | d4l3k | |
| meta-pytorch/tokenizers | 2026-05-12 | Copilot | |
| pytorch/helion | 2026-05-13 | choijon5 | |
| pytorch/ci-infra | 2026-05-27 | huydhn | |
| meta-pytorch/attention-gym | 2026-06-27 | drisspg | |
| meta-pytorch/skills | 2026-07-08 | iamzainhuda | |
| pytorch/gloo | 2026-09-16 | d4l3k | |

**14 repos** (10 pytorch, 4 meta-pytorch) reference the reusable workflow. `upload-claude-usage` is additionally used directly by: pytorch/pytorch (advisor, issue triage, distributed triage), pytorch/ao (ci-failure-issue), pytorch/ciforge (advisor, distributed triage, PR review, issue triage), test-infra (greenlight, vLLM triage).

---

## 7. Timeline (PR #, author)

| Date | Repo | PR | Author | Change |
|---|---|---|---|---|
| 2026-01-16 | pytorch | — | izaitsevfb | First `claude-code.yml` in pytorch/pytorch (allowlist of users) |
| 2026-01-22 | test-infra | #7675 | Ivan Zaitsev | `upload-claude-usage` + `misc.claude_code_usage` schema |
| 2026-01-26 | pytorch | #173418 | izaitsevfb | Upload usage metrics from pytorch |
| 2026-01-26 | pytorch | #173422 | jeanschmidt | Claude responds to autorevert inquiries |
| 2026-01-29 | pytorch | #173748 | izaitsevfb | Support fork PRs |
| 2026-02-04 | test-infra | #7726, #7729 | Wouter Devriendt | HUD Claude Billing page; token + model tracking |
| 2026-02-26 | test-infra | #7798 | Ivan Zaitsev | Claude Code workflow (issue/PR comment triggers) |
| 2026-03-05 | test-infra | #7810, #7814 | Zain Rizvi | **Reusable `_claude-code.yml` + setup script** |
| 2026-03-05 | pytorch | #176490 | albanD | Enable `pr-review` skill for `@claude` |
| 2026-03-05 | pytorch | #176652 | izaitsevfb | Revert PR review-comment trigger support (security) |
| 2026-03-06 | pytorch | #176724 | ZainRizvi | pytorch switches to reusable workflow |
| 2026-03-13 | pytorch | #177404 | izaitsevfb | Claude autorevert AI advisor |
| 2026-03-16 | test-infra | #7832, #7834 | Zain Rizvi | Commits attributed to invoker; ghstack read-only |
| 2026-03-30 / 04-02 | test-infra | #7906, #7908 | Ivan Zaitsev | Advisor verdicts → ClickHouse → used by autorevert |
| 2026-04-03 | test-infra | #7925 | albanD | `append_system_prompt` input |
| 2026-04-13 | test-infra | #7960, #7962 | Wouter Devriendt | Billing page login-gated, "estimated" costs |
| 2026-04-24 | test-infra | #8012 | Ivan Zaitsev | Pin to upstream anthropics/claude-code-action v1.0.104 |
| 2026-06-09 | test-infra | #8162 | Driss Guessous | Default model → Opus 4.8 |
| 2026-06-17 | test-infra | #8178 | Ivan Zaitsev | Dr.CI auto-dispatches advisor (dark) |
| 2026-06-23 | test-infra | #8202 | Ivan Zaitsev | Advisor verdict inline in Dr.CI comment |
| 2026-06-29 | pytorch | #188418 | zou3519 | Reviews inside collapsible section |
| 2026-07-06 | test-infra | #8251 | Richard Zou | Fix append-system-prompt env var |
| 2026-07-08 | test-infra | #8213 | Huy Do | `infra_issue` advisor verdict |
| 2026-07-24/25 | pytorch | #190960, #191112 | drisspg | Modernize models; Opus 5 for `@claude` |
| 2026-07-27 | pytorch | #191256 | drisspg | Bound Bedrock retries (`API_TIMEOUT_MS`, `MAX_RETRIES=2`, watchdog) |
| 2026-07-27 | fork | `92db82f` | drisspg | Sanitized live progress (not upstreamed) |
| 2026-07-29 | pytorch | #190963 | izaitsevfb | `fable` keyword opts into Fable model |
| 2026-07-30 | pytorch | #191655 | drisspg | Configurable `effort=` keyword |
| 2026-07-31 | test-infra | #8381 | Driss Guessous | Public execution logs (`show_full_output`, S3 upload) |
| 2026-08-03 | test-infra | #8363 | Jean Schmidt | Green Light MVP (AI LAND/NO_LAND reviewer) |
| 2026-08-04 | pytorch | #191969 | drisspg | Enable execution logs in pytorch |
| 2026-08-04 / 08-10 | test-infra | #8435, #8481 | Jean Schmidt | Greenlight: instruction-file injection & credential-exfil hardening |
| 2026-08-06 | test-infra | #8447 | Andrey Talman | vLLM torch-nightly triage |
| 2026-08-15 | test-infra | #8536 | Jean Schmidt | Harden `upload-claude-usage` against script injection |
| 2026-08-28 | pytorch | #195169 | drisspg | Concise output style |
| 2026-09-10 | pytorch | #196537 | izaitsevfb | Raise `@claude` timeout to 2h |
| 2026-09-22/23 | test-infra / pytorch | #8865 / #198241 | izaitsevfb | Opus 5.5 + Claude Code 2.1.280 action |
| 2026-09-24 | pytorch | #198521 | jeanschmidt | Timeout 55 min to match 1h credential lifetime |
| 2026-09-24 | test-infra | #8887 | jeanschmidt | Repo-root CLAUDE.md for ClickHouse query rules |
| 2026-09-28 | pytorch | #198687 | jeanschmidt | Time-budget hook for `@claude` |

---

## Slide-worthy facts

- **One line to onboard:** `uses: pytorch/test-infra/.github/workflows/_claude-code.yml@main` + `uv run …/setup-claude-environment.py` + one IAM trust-policy line. 14 repos across pytorch and meta-pytorch have done it.
- **No API keys in CI:** Bedrock through GitHub OIDC → `role/gha_workflow_claude_code`, scoped per repo to `repo:<org>/<repo>:environment:bedrock`, and that environment only deploys from `main`.
- **Layered gates:** org allowlist → `@claude` mention → member/collaborator → write permission check → ghstack PRs read-only → no `pull_request_review_comment` (PR-branch code execution).
- **Agents fighting agents:** the autorevert bot (an agent) tags `@claude` on its own revert comments; a Claude advisor returns a JSON-schema verdict (`related/unsure/not_related/infra_issue/garbage`) that autorevert reads back from ClickHouse when deciding.
- **Dr.CI auto-dispatch with brakes:** at most 32 advisor runs per PR head; skip when >8 new failures (outage guard).
- **Every run is metered:** tokens (incl. cache read/create), cost, turns, duration, model → `misc.claude_code_usage` → HUD "Claude Billing" (estimated cost, login-gated).
- **Public receipts:** full Claude transcripts posted to the job log and archived to S3 per PR comment and per run (#8381); a viewer can start from a PR and find every review.
- **Hard lesson: 1h credentials.** AWS session and App token expire 1h after job start, so the job runs 55 min and a time-budget hook tells Claude when to converge and post.
- **Stalls are real:** `API_TIMEOUT_MS=180000`, `CLAUDE_CODE_MAX_RETRIES=2`, stream watchdog; fork prototype adds a metadata-only heartbeat (`no SDK event for 60s`).
- **Users pick the knobs per comment:** `@claude fable …` switches model; `effort=low|high|max`. Only hardcoded values reach the CLI, never comment text.
- **Untrusted content never meets write tokens:** Green Light, ao ci-failure, and the hardened review all split "model reads untrusted input" from "code-only job holds write credentials".
- **Model drift in 7 months:** reusable default Opus 4.6 (Mar) → 4.8 (Jun) → 5.5 (Sep); Haiku/Sonnet handle cheap log classification.
