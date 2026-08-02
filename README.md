# veri-runner

The open-source training runtime for [Veri](https://veri.studio). This is the
code that actually executes your training job on a GPU worker: dataset
normalization, reward-function invocation, TRL trainer dispatch, and (for
harness-in-the-loop RL) token-exact trajectory capture between your agent
harness and the in-training policy.

It is published so you can answer "is this bug in my code or in the platform?"
by reading the exact code your job ran, instead of asking support. It works
well as context for a coding agent: point Claude Code (or any agent) at this
repo together with your job's downloaded trajectory or logs and ask it to walk
the relevant path.

## What's here

| Path | What it is |
| --- | --- |
| `veri_runner/training_runtime.py` | Method dispatch (`grpo`, `grpo_harness`, `sft_text`, `dpo`), dataset format detection/normalization, reward loading, TRL config construction, checkpoint save |
| `veri_runner/trajectory_proxy.py` | The per-rollout HTTP proxy between your unmodified agent harness and the policy server: OpenAI and Anthropic (`/v1/messages`) adapters, token-exact span capture, episode rendering with assistant-only loss masking, and the token-fidelity / template-drift rejection paths |
| `veri_runner/harness_rollout.py` | Rollout orchestration for harness-in-the-loop GRPO: per-rollout env contract, sandboxed harness execution, reward scoring, trajectory JSONL output |
| `contracts/worker_config/` | Fixture copies of the exact job config the control plane hands the worker |
| `docs/trajectory-format.md` | The trajectory / span JSONL schema (what you download from the job page) |
| `examples/` | Sanitized example trajectories, including a token-fidelity rejection to debug against |
| `tests/` | The unit tests Veri runs against this code in CI |

Not here (closed): worker lifecycle (boot, billing callbacks, termination),
the control plane, provisioning, and serving.

## Debugging token-fidelity rejections

When a rollout is rejected from training ("template drift" or "token ids
absent"), the decision was made in `veri_runner/trajectory_proxy.py`:

1. `render_episode` requires every span to have `token_fidelity: "captured"`.
   Spans without server-side token ids are never trained on.
2. Multi-turn episodes must satisfy strict prefix extension: turn k's prompt
   token ids must extend turn k-1's prompt+completion exactly (verl-style
   delta tokenization). A mismatch means the chat template rewrote history
   (thinking models like Qwen3 do this) and raises `TemplateDriftError`.
3. If stitching fails, the episode falls back to training on the final turn
   only (`metadata.render_fallback = "last_turn"`), never on drifted tokens.

To debug a rejected rollout: download the raw trajectory JSONL from the job
page, then check the spans against those rules (or hand the file and this repo
to a coding agent and ask which rule fired and why). See
`examples/rejected_trajectory_template_drift.jsonl` for a worked example.

## Running the tests

```bash
uv run --extra dev pytest
```

The tests run on plain Python (no GPU, no torch): the heavy dependencies are
imported lazily per training method and the tests exercise the logic around
them.

## How changes ship

This repo is vendored into Veri's private monorepo as a git submodule pinned
to an exact commit. Merged PRs here do not change production behavior until
Veri bumps that pin, runs the full internal suite (including worker-image and
live-GPU tests), and deploys a new worker image. So: contributions are
welcome, CI here must stay green, and a green PR is necessary but not
sufficient for a change to reach production.

Compatibility contract for contributors:

- `veri_runner/*.py` must keep working when dropped flat into one directory
  (the worker imports `training_runtime`, not `veri_runner.training_runtime`);
  that is why cross-module imports use the try/except pattern.
- Public function signatures used by the worker (`run_training`,
  `load_reward_function(s)`, `resolve_training_rows`, `render_episode`,
  `SpanStore`, `TrajectoryProxy`, `HarnessRunner`) are stable interfaces;
  breaking them requires a matching worker change on Veri's side, so open an
  issue first.
- Tests must not require a GPU or network.

## License

Apache-2.0. See [LICENSE](LICENSE).
