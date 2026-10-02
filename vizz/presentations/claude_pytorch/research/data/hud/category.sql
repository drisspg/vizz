multiIf(
  event_name IN ('issue_comment','pull_request_review_comment') AND actor NOT LIKE '%[bot]', 'human_mention',
  event_name IN ('issue_comment','pull_request_review_comment'), 'bot_mention',
  actor = 'pytorch-auto-revert[bot]', 'autorevert_advisor',
  actor = 'pytorch-bot[bot]', 'drci_advisor',
  actor = 'pytorchgreenlight[bot]', 'greenlight',
  event_name IN ('workflow_run','issues'), 'issue_triage',
  event_name = 'schedule' OR actor LIKE '%scheduled%', 'scheduled',
  'manual_dispatch')
