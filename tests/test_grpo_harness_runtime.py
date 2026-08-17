"""grpo_harness training-runtime pieces: TRL rollout_func batch
contract, reward adapter, dataset validation, worker dispatch kwargs."""

from __future__ import annotations

from veri_runner.harness_rollout import RolloutResult
from veri_runner.training_runtime import (
    _policy_server_devices,
    _policy_server_layout,
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


def test_policy_server_devices_clamps_tp_to_power_of_two():
    # VS-374 blocker 1: TP = gpu_count-1 was illegal on every aws shape
    # (vLLM needs num_attention_heads % TP == 0; 32-head models divide by
    # neither 3 nor 7). TP must now be the largest power of two <= n-1, with
    # the trainer keeping GPU 0 and any remainder idling instead of crashing.
    assert _policy_server_devices(2) == [1]            # vast x2: TP=1
    assert _policy_server_devices(3) == [1, 2]         # TP=2
    assert _policy_server_devices(4) == [1, 2]         # aws x4: TP=2, GPU 3 idle
    assert _policy_server_devices(5) == [1, 2, 3, 4]   # TP=4
    assert _policy_server_devices(8) == [1, 2, 3, 4]   # aws x8: TP=4, GPUs 5-7 idle
    # Every returned length is a power of two (a legal divisor of any
    # power-of-two-divisible head count) and excludes the trainer's GPU 0.
    for n in range(2, 9):
        devices = _policy_server_devices(n)
        assert 0 not in devices
        assert len(devices) & (len(devices) - 1) == 0


def test_policy_server_layout_defaults_match_the_clamp():
    # No knobs set -> byte-for-byte the safe clamp, DP=1. This is the
    # non-breaking guarantee: existing configs launch the identical command.
    for n in range(2, 9):
        devices, tp, dp = _policy_server_layout(n, {})
        assert devices == _policy_server_devices(n)
        assert (tp, dp) == (len(devices), 1)


def test_policy_server_layout_knobs():
    import pytest

    # DP only: TP snaps to the largest power of two fitting available // DP.
    # L4 x4 + dp=3 is the zero-idle layout for small models.
    assert _policy_server_layout(4, {"vllm_data_parallel_size": 3}) == ([1, 2, 3], 1, 3)
    assert _policy_server_layout(8, {"vllm_data_parallel_size": 7}) == (
        [1, 2, 3, 4, 5, 6, 7], 1, 7,
    )
    # Both set: used verbatim (vLLM still enforces head divisibility at boot).
    assert _policy_server_layout(
        8, {"vllm_tensor_parallel_size": 2, "vllm_data_parallel_size": 3}
    ) == ([1, 2, 3, 4, 5, 6], 2, 3)
    # TP only: DP stays 1.
    assert _policy_server_layout(8, {"vllm_tensor_parallel_size": 4}) == (
        [1, 2, 3, 4], 4, 1,
    )
    # Over-subscription and non-positive values are rejected, not clamped:
    # silently shrinking an explicit request would hide a misconfiguration.
    for bad in [
        {"vllm_tensor_parallel_size": 4, "vllm_data_parallel_size": 2},  # 8 > 7
        {"vllm_data_parallel_size": 8},  # > gpu_count-1
        {"vllm_data_parallel_size": 0},
        {"vllm_tensor_parallel_size": 0},
    ]:
        with pytest.raises(ValueError):
            _policy_server_layout(8, bad)


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
