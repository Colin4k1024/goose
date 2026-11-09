---
title: RSI benchmark
---

`goose rsi init` installs a small catalog recipe and a four-task benchmark in the goose config directory. It refuses to replace an existing recipe or benchmark. Configure a provider and model, then run `goose rsi evolve`, `goose rsi history`, and `goose rsi status` to inspect one cycle. The sample checks actual model answers and serves as an end-to-end smoke test; replace it with representative tasks before drawing quality conclusions.

The benchmark file is `rsi/benchmark.json` under the goose config directory. It names one existing `recipes/*.yaml` target and contains at least one `training` and one `holdout` task. The [sample benchmark](../../../crates/goose-cli/examples/rsi/rsi/benchmark.json) shows the format. It calls the same Goose executable through `$GOOSE_RSI_EXECUTABLE`, which the evaluator sets for each task.

Each task runs against separate baseline and candidate copies. Its command must print `OVERALL RESULT: PASS` or `OVERALL RESULT: FAIL` on its own line. A nonzero exit, timeout, or unrecognized result blocks the proposal. The gate requires a strict increase in average training score, a nonzero holdout score, and no regression on any task. The baseline and candidate recipes, task output, and scores are written under `rsi/fitness-logs/<proposal-id>/` and summarized by `goose rsi history`.

The proposal model receives the current recipe and training task definitions. Holdout tasks are excluded from its prompt. It may change only the recipe's `prompt` or `instructions`; the existing target, settings, extensions, and parameters stay fixed. The benchmark file itself is outside the RSI mutation whitelist. If it is absent or invalid, `evolve` stops before proposing or applying a change.

For a research benchmark, use fixed prompts and answer or artifact checks that exercise the agent. Keep the task set and grading rules unchanged across baseline and candidate runs. Model responses can vary between runs, so repeat trials and review task logs before treating a score change as evidence of general improvement.
