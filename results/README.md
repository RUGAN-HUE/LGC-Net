# Result files

- `reported_main_comparison.json` transcribes the values in the final manuscript and is the authoritative public summary for the submitted paper.
- `retraining_checkpoint_verification.json` is an independent pipeline check using five later local retraining checkpoints. It verifies that the released model, pixel-free annotations, explicit test order, and reconstructed two-level Hungarian Max IoU implementation work together. These later checkpoints are not the original five runs used for the manuscript table.
- `posthoc_duplicate_sensitivity.json` evaluates the same later checkpoints after excluding the two test records identified by the post-hoc same-cover duplicate audit. It is a sensitivity check, not a replacement for the manuscript runs.

The later checkpoint verification obtains an element-wise IoU close to the manuscript result and a Max IoU close to the manuscript result. The distinction is retained to avoid presenting a stochastic retraining run as the exact archived manuscript run.
