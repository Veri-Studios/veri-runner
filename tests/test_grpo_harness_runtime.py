"""grpo_harness training-runtime pieces: TRL rollout_func batch
contract, reward adapter, dataset validation, worker dispatch kwargs."""

from __future__ import annotations

from veri_runner.harness_rollout import RolloutResult
from veri_runner.training_runtime import (
    build_rollout_batch,
    harness_reward_adapter,
    validate_rows_for_method,
)
from veri_runner.trajectory_proxy import Episode


def _result(status, reward, episode=None, task_index=0, rollout_index=0):
    return RolloutResult(
        rollout_id=f"s0-t{task_index}-r{rollout_index}",
        policy_step=0,
        task_index=task_index,
        rollout_index=rollout_index,
        status=status,
        reward=reward,
        episode=episode,
        num_turns=episode.num_turns if episode else 0,
        tokens_in=3,
        tokens_out=2,
        trajectory_path="/tmp/x.jsonl",
        error=None if status == "completed" else "boom",
        duration_s=1.0,
    )


def _episode():
    # prompt [10, 11] masked; turn-1 completion [20] trained; tool delta [30]
    # masked; turn-2 completion [40] trained.
    return Episode(
        rollout_id="r",
        input_ids=[10, 11, 20, 30, 40],
        loss_mask=[0, 0, 1, 0, 1],
        logprobs=[0.0, 0.0, -0.5, 0.0, -0.6],
        num_turns=2,
        final_completion_text="done",
    )


def test_build_rollout_batch_matches_trl_contract():
    batch = build_rollout_batch(results=[_result("completed", 0.75, _episode())])
    assert set(batch) == {
        "prompt_ids", "completion_ids", "logprobs", "env_mask", "harness_reward",
    }
    # prompt/completion split at the first trained token.
    assert batch["prompt_ids"] == [[10, 11]]
    assert batch["completion_ids"] == [[20, 30, 40]]
    assert batch["logprobs"] == [[-0.5, 0.0, -0.6]]
    # env_mask: assistant tokens trained (1), tool-result delta masked (0).
    assert batch["env_mask"] == [[1, 0, 1]]
    assert batch["harness_reward"] == [0.75]


def test_build_rollout_batch_failed_rollout_is_zero_mask_sentinel():
    # Failed rollouts stay in the batch (GRPO groups are fixed-size) but are
    # fully masked out of the loss and score 0.
    batch = build_rollout_batch(results=[
        _result("completed", 1.0, _episode(), rollout_index=0),
        _result("failed", None, None, rollout_index=1),
    ])
    assert len(batch["prompt_ids"]) == 2
    assert batch["env_mask"][1] == [0]
    assert batch["harness_reward"][1] == 0.0


def test_harness_reward_adapter_forwards_scores():
    assert harness_reward_adapter(
        prompts=["p"], completions=["c"], harness_reward=[0.5, 0.25]
    ) == [0.5, 0.25]
    assert harness_reward_adapter(prompts=["p"], completions=["a", "b"]) == [0.0, 0.0]


def test_validate_rows_requires_prompt_for_grpo_harness():
    import pytest

    validate_rows_for_method([{"prompt": "q", "answer": "a"}], "grpo_harness")
    with pytest.raises(ValueError, match="prompt"):
        validate_rows_for_method([{"text": "no prompt column"}], "grpo_harness")


def test_run_training_dispatches_grpo_harness():
    from veri_runner import training_runtime

    calls = {}

    def fake_run(job_config, **kwargs):
        calls.update(kwargs)
        return {"ok": True}

    original = training_runtime.run_grpo_harness_training
    training_runtime.run_grpo_harness_training = fake_run
    try:
        training_runtime.run_training(
            {"method": "grpo_harness", "hyperparameters": {}},
            harness_code_dir="/tmp/hc",
            trajectory_sink=lambda step, results: None,
        )
    finally:
        training_runtime.run_grpo_harness_training = original
    assert calls["harness_code_dir"] == "/tmp/hc"
    assert callable(calls["trajectory_sink"])
