# Local Data Only

Historical chat exports and derived CSVs have been removed from the current
revision at the repository owner's request. Do not restore them from history
or commit replacement real conversations here. This directory accepts only
this README in the public-tree check.

Use reviewed local data outside version control. Notebook CSV inputs require
`VIBECHECK_INPUT_CSV`; the cleaning notebook additionally requires
`VIBECHECK_OUTPUT_CSV` before writing results. These variables are file paths,
not permission to collect or publish messages. Keep generated notebook outputs
out of commits. See the root README for historical-source limitations.

The only committed CSV example is `examples/synthetic_messages.csv`: four
hand-written artificial messages with assigned scores, not training data.
