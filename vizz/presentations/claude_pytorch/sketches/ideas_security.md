# Security / architecture as motion

Use these as replacements for existing reveals, not six additional slides. Red = untrusted content or forbidden action; amber = Claude's proposal; green = deterministic policy / authorized mutation / human decision. All payloads below are illustrative, not actual attack transcripts. S = ~1–2 h, M = ~half day, L = ~day including visual review.

## 1. The label physically cannot cross — **prototype winner**
- **Target:** `guardrails`, replace the hook-list reveal; ~20–25 s narration, three pauses.
- **Beats:** (1) Red issue-body request “add merge blocking” feeds an amber Claude proposal. A green `validate_labels.py` barrier separates proposal from mutation. (2) The proposed red `merge blocking` chip approaches the barrier, hits it, and falls into a “stripped” tray. No forbidden label ever appears on the target issue. (3) A separate amber `triage review` proposal passes the policy boundary and turns green; `bot-triaged` appears as an audit stamp after the mutation. Hold the human decision, not a victory explosion.
- **Why it impresses:** the audience sees a failed model decision and a successful system design in the same shot. The model need not be immune to persuasion for this particular policy to hold.
- **Manim:** `TransformFromCopy` from issue text into proposal; `MoveAlongPath` into the barrier; a short `there_and_back` recoil; `Cross` and `FadeOut`; a second pass with `.animate.move_to`, then `Transform` to green. Three named `next_slide` boundaries.
- **Cost:** S. **Gimmick risk:** low. Do not depict a hook as another LLM. Label filter **strips** forbidden labels; the visual bounce is a metaphor, not a claim about the hook's exit code. `triage review` is the skill's follow-up, not the validator automatically renaming the bad label.
- **Source:** `research/pytorch_repo.md` §2.3–2.4: `validate_labels.py`, `add_bot_triaged.py`, 282-entry allowlist, forbidden `merge blocking`, and “If blocked … add ONLY triage review and stop.” PRs #174023/#174121. Prototype is a scripted illustration, not execution of the real hook.

## 2. The artifact slot only fits a number
- **Target:** `two_stage`, replace stage-column fade; ~25–30 s, three pauses.
- **Beats:** (1) Red issue card has body text and numeric ID; stage 1 visibly has no secrets and only `contents: read`. (2) Extract the ID into `issue_number.txt`; the prose stays behind, never entering the artifact. Slide the number through a narrow slot marked `^[0-9]+$`. (3) Stage 2 from `main` independently checks the number, then fetches the issue body on a **different red data arrow** to amber Claude. Green target/label hooks surround mutations.
- **Why it impresses:** changes an abstract trust boundary into a visibly tiny interface while showing what that interface does **not** solve.
- **Manim:** `TransformFromCopy` on just the ID; `MoveAlongPath` through a short aperture; `Indicate` both regex checks; `Create` a separately routed red issue-fetch arrow only in the final beat.
- **Cost:** M. **Gimmick risk:** medium/high if abbreviated. Never depict the artifact as a general prompt-injection sanitizer: stage 2 still reads untrusted content and has issue-write capabilities. A number passing syntax validation is not proof the issue body is trusted. Avoid the existing slide's broad headline as the prototype takeaway; use “A narrow handoff. A second validation.”
- **Source:** `research/pytorch_repo.md` §2.1–2.3; #173725; stage 1 has 2-min timeout and uploads only the number, stage 2 uses `workflow_run`, `bedrock`, `issues: write`, and five MCP tools including issue/comment reads.

## 3. Credentials have an expiry cliff
- **Target:** `lessons`, credentials row; ~15–20 s, two pauses.
- **Beats:** (1) GitHub OIDC identity assertion moves into AWS role exchange; a separate green **AWS session + App token** card is issued with a 60-min lifetime. (2) A timeline advances to 55 min; amber Claude posts and stops before the 60-min cliff. Show a bracket marked “5-min margin,” then grey out credentials at 60. A tiny time-budget-hook tick shows the wrap-up reminder.
- **Why it impresses:** ties one operational choice (55 min) to a physical limit instead of presenting an arbitrary timeout.
- **Manim:** `ValueTracker` + integer elapsed-minute readout; `always_redraw` progress bar; `Transform` credential card to expired state. Remove updaters at final pause.
- **Cost:** S. **Gimmick risk:** medium. Do not label the OIDC JWT itself “1 h”; the cited hour is the AWS session and GitHub App token. Do not suggest the 55-min job always runs to completion; this is its configured upper bound. Timeline is accelerated illustration, not run telemetry.
- **Source:** `research/test_infra.md` §1–2; pytorch #198521 and #198687. This newer upstream report supersedes the earlier 120-min caller snapshot in `research/pytorch_repo.md` §1.1.

## 4. Same comment, different authority
- **Target:** `shape`, gate reveal; ~20 s, two pauses.
- **Beats:** (1) Two identical red `@claude` comment cards approach the gate; one has a green API-verified `write` badge, the other `read`. (2) The write badge unlocks the route to OIDC; the read-only caller stops before credentials. Both comment bodies remain red after authorization. A small separate lane names the allowlisted autorevert bot exception.
- **Why it impresses:** sharply distinguishes permission to invoke an agent from trust in the text the agent will consume.
- **Manim:** `MoveAlongPath` for cards; a short `Rotate` on a gate arm; `there_and_back` recoil for denied caller; persistent red body styling rather than recoloring whole cards green.
- **Cost:** M. **Gimmick risk:** medium: a cartoon bouncer can obscure the two checks. Put `org + mention + association` above the first checkpoint, `API: admin / write` above the second. Don't imply all contributors can invoke it.
- **Source:** `research/test_infra.md` §1; `research/pytorch_repo.md` §1.2–1.3 and #176027. `pytorch-auto-revert[bot]` is the explicit exception.

## 5. Shrink the toolbox before Claude starts
- **Target:** `guardrails`, before the hook shot; ~15 s, two pauses.
- **Beats:** (1) An illustrative general-purpose tool shelf includes Bash, file Read/Write, and GitHub capabilities. (2) A “triage policy” frame contracts to exactly five GitHub tool chips: get issue, get comments, update issue, add comment, search issues. The shell/file chips fade out **before** amber Claude lights up.
- **Why it impresses:** least privilege becomes an object with a count, not a list of YAML strings.
- **Manim:** `ReplacementTransform` wide shelf → compact shelf; `LaggedStart(FadeOut(...))` on forbidden capabilities; `Circumscribe` remaining five. Keep readable short names; full MCP names in notes.
- **Cost:** S. **Gimmick risk:** low/medium. Label the first shelf “general capabilities,” not “tools initially granted”; policy is configured before the run, not dynamically tightened halfway through. The five-tool policy is triage-specific, not universal across Claude workflows.
- **Source:** `research/pytorch_repo.md` §2.2, exact `--allowedTools` list; no Bash, Read, or Write.

## 6. A malicious branch cannot choose the credentialed program
- **Target:** `lessons` unsafe-trigger row, or `two_stage` extension; ~20 s, two pauses.
- **Beats:** (1) Two rails: red PR-branch workflow and green main-branch workflow. An illustrative maintainer review comment follows the old unsafe route into red branch code; freeze **before execution**. (2) Cross out `pull_request_review_comment`, then reveal the supported issue-comment route and main-only `bedrock` environment. A red PR edit stays off the credential rail.
- **Why it impresses:** makes “which code runs?” feel more immediate than a long list of token scopes; permissioned humans can still trigger attacker-chosen code under the wrong event.
- **Manim:** parallel `Line` rails, `MoveAlongPath` event bead, `Cross` on unsafe event, `Create` safe connector. No live shell, exploit payload, or pretend token leak required.
- **Cost:** M. **Gimmick risk:** medium. Show the review-comment route as historical and removed, not currently exploitable. The current gate and environment policy both matter; the event-name replacement alone is not a complete security proof.
- **Source:** `research/pytorch_repo.md` §1.2–1.3, #176652; `research/test_infra.md` §1–2 for main-only deployment policy and documented per-repo OIDC subject. Exact private IAM conditions remain unverified in research notes.

## Prototype review contract

`proto_security.py` implements idea 1 with the deck's exact theme and common helpers. `build(scene: SlideBase)` leaves its last pause visible. `ProtoSecurity` is a movie-only wrapper; no real network calls, Claude calls, label mutations, or hook execution occur. Three frames should show proposal, stripped action, and human escalation respectively. The proposed attack is illustrative and explicitly labeled that way.
