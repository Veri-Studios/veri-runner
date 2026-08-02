import json
from pathlib import Path

import pytest

from veri_runner.training_runtime import (
    MILES_REWARD_UNSUPPORTED,
    _apply_system_prompt,
    _load_model_and_tokenizer,
    _require_preference_columns,
    build_dpo_config_kwargs,
    build_grpo_config_kwargs,
    build_sft_config_kwargs,
    build_trainer_kwargs,
    detect_dataset_format,
    load_reward_function,
    load_reward_functions,
    normalize_rows,
    prerender_chat_template_rows,
    resolve_training_rows,
    run_training,
    save_checkpoint,
    validate_rows_for_method,
)


def test_apply_system_prompt_noop_when_none():
    rows = [{"prompt": "what is 2+2?", "answer": "4"}]
    assert _apply_system_prompt(rows, None) == rows
    assert _apply_system_prompt(rows, "") == rows


def test_apply_system_prompt_wraps_string_prompt_into_chat_list():
    rows = [{"prompt": "what is 2+2?", "answer": "4"}]
    out = _apply_system_prompt(rows, "be terse")
    assert out[0]["prompt"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "what is 2+2?"},
    ]
    assert out[0]["answer"] == "4"


def test_apply_system_prompt_preserves_existing_chat_list():
    rows = [{
        "prompt": [{"role": "user", "content": "hi"}],
        "answer": "hello",
    }]
    out = _apply_system_prompt(rows, "be terse")
    # Already-conversational rows are left alone — we only wrap strings.
    assert out[0]["prompt"] == [{"role": "user", "content": "hi"}]


def test_load_reward_function_prefers_reward(tmp_path):
    reward_file = tmp_path / "reward.py"
    reward_file.write_text(
        "def other(*args, **kwargs):\n"
        "    return [0.0]\n\n"
        "def reward(completions, answer, **kwargs):\n"
        "    return [1.0] * len(completions)\n"
    )

    reward_fn = load_reward_function(str(reward_file))

    assert reward_fn(["a", "b"], "answer") == [1.0, 1.0]


def test_load_reward_functions_loads_distinct_callables(tmp_path):
    # Multi-reward: two reward files load into two distinct callables (ported from
    # TRL's multi-task reward example). Same module name would clobber; distinct
    # names keep them separate.
    f1 = tmp_path / "correctness.py"
    f1.write_text("def reward(completions, **kwargs):\n    return [1.0] * len(completions)\n")
    f2 = tmp_path / "format.py"
    f2.write_text("def reward(completions, **kwargs):\n    return [0.5] * len(completions)\n")

    fns = load_reward_functions([str(f1), str(f2)])

    assert len(fns) == 2
    assert fns[0](["a", "b"]) == [1.0, 1.0]
    assert fns[1](["a"]) == [0.5]


def test_load_reward_function_falls_back_to_first_public_callable(tmp_path):
    reward_file = tmp_path / "reward.py"
    reward_file.write_text(
        "def score(completions, answer, **kwargs):\n"
        "    return [0.5] * len(completions)\n"
    )

    reward_fn = load_reward_function(str(reward_file))

    assert reward_fn(["a"], "answer") == [0.5]


def test_load_reward_function_rejects_unsupported_miles_rewards(tmp_path):
    reward_file = tmp_path / "reward.py"
    reward_file.write_text("async def reward(args, sample, **kwargs):\n    return 1.0\n")

    with pytest.raises(ValueError, match="Miles reward format is not supported"):
        load_reward_function(
            str(reward_file),
            reward_format="miles",
            unsupported_formats={"miles": MILES_REWARD_UNSUPPORTED},
        )


def test_resolve_training_rows_prefers_downloaded_dataset_path(tmp_path):
    dataset_file = tmp_path / "data.jsonl"
    dataset_file.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "one", "answer": "1"}),
                "",
                json.dumps({"prompt": "two", "answer": "2"}),
            ]
        )
    )

    rows = resolve_training_rows(
        dataset_path=str(dataset_file),
        dataset_config={"source_type": "upload"},
    )

    assert rows == [
        {"prompt": "one", "answer": "1"},
        {"prompt": "two", "answer": "2"},
    ]


def test_resolve_training_rows_rejects_unknown_source_type():
    with pytest.raises(ValueError, match="Unsupported dataset source_type=s3"):
        resolve_training_rows(dataset_config={"source_type": "s3"})


def test_build_grpo_config_kwargs_adapts_to_supported_parameters():
    class FakeGRPOConfig:
        def __init__(self, max_prompt_length=None, beta=None):
            pass

    kwargs = build_grpo_config_kwargs(
        job_id="job-123",
        hyperparameters={
            "rollouts_per_prompt": 1,
            "max_response_length": 128,
            "max_prompt_length": 64,
            "learning_rate": 0.00001,
            "max_steps": 20,
            "kl_coef": 0.2,
        },
        grpo_config_cls=FakeGRPOConfig,
    )

    assert kwargs["output_dir"] == str(Path("/tmp/ckpts") / "job-123")
    assert kwargs["per_device_train_batch_size"] == 2
    assert kwargs["num_generations"] == 1
    assert kwargs["max_prompt_length"] == 64
    assert kwargs["beta"] == 0.2


def test_build_grpo_config_kwargs_defaults_omitted_hyperparameters():
    """The control plane stores the client's raw hyperparameters JSON, so a
    submit that omits optional fields reaches the worker without them (SDK
    create / TOML form). Indexing crashed with KeyError:
    'max_response_length' — every optional knob must fall back to the API
    schema's default."""

    class FakeGRPOConfig:
        def __init__(self, max_prompt_length=None, beta=None):
            pass

    kwargs = build_grpo_config_kwargs(
        job_id="job-123",
        hyperparameters={},
        grpo_config_cls=FakeGRPOConfig,
    )

    assert kwargs["max_completion_length"] == 2048
    assert kwargs["learning_rate"] == 1e-6
    assert kwargs["num_generations"] == 8
    assert kwargs["per_device_train_batch_size"] == 8
    assert kwargs["max_steps"] == 100


def test_build_grpo_config_kwargs_omits_unsupported_parameters():
    class OldGRPOConfig:
        def __init__(self):
            pass

    kwargs = build_grpo_config_kwargs(
        job_id="job-123",
        hyperparameters={
            "rollouts_per_prompt": 4,
            "max_response_length": 128,
            "max_prompt_length": 64,
            "learning_rate": 0.00001,
            "kl_coef": 0.2,
        },
        grpo_config_cls=OldGRPOConfig,
    )

    assert "max_prompt_length" not in kwargs
    assert "beta" not in kwargs


def test_build_trainer_kwargs_prefers_processing_class():
    class NewTrainer:
        def __init__(self, processing_class=None, tokenizer=None):
            pass

    kwargs = build_trainer_kwargs(
        trainer_cls=NewTrainer,
        model="model",
        training_args="args",
        train_dataset="dataset",
        reward_fn=lambda: None,
        tokenizer="tokenizer",
    )

    assert kwargs["processing_class"] == "tokenizer"
    assert "tokenizer" not in kwargs


def test_build_trainer_kwargs_supports_older_tokenizer_parameter():
    class OldTrainer:
        def __init__(self, tokenizer=None):
            pass

    kwargs = build_trainer_kwargs(
        trainer_cls=OldTrainer,
        model="model",
        training_args="args",
        train_dataset="dataset",
        reward_fn=lambda: None,
        tokenizer="tokenizer",
    )

    assert kwargs["tokenizer"] == "tokenizer"


def test_build_trainer_kwargs_includes_reward_weights_for_multi_reward():
    # GOOD: a reward_funcs list + reward_weights on a trainer that supports it.
    class MultiRewardTrainer:
        def __init__(self, reward_funcs=None, reward_weights=None, processing_class=None):
            pass

    f1 = lambda **k: [1.0]  # noqa: E731
    f2 = lambda **k: [0.5]  # noqa: E731
    kwargs = build_trainer_kwargs(
        trainer_cls=MultiRewardTrainer,
        model="model",
        training_args="args",
        train_dataset="dataset",
        tokenizer="tokenizer",
        reward_fn=[f1, f2],
        reward_weights=[1.0, 0.5],
    )

    assert kwargs["reward_funcs"] == [f1, f2]
    assert kwargs["reward_weights"] == [1.0, 0.5]


def test_build_trainer_kwargs_omits_reward_weights_when_unsupported():
    # BAD/guard: older TRL GRPOTrainer has no reward_weights param — it must be
    # dropped, not passed (which would raise a TypeError at construction).
    class OldGRPOTrainer:
        def __init__(self, reward_funcs=None, processing_class=None):
            pass

    kwargs = build_trainer_kwargs(
        trainer_cls=OldGRPOTrainer,
        model="model",
        training_args="args",
        train_dataset="dataset",
        tokenizer="tokenizer",
        reward_fn=[lambda **k: [1.0]],
        reward_weights=[1.0],
    )

    assert "reward_weights" not in kwargs
    assert isinstance(kwargs["reward_funcs"], list)


def test_build_trainer_kwargs_omits_reward_funcs_when_reward_free():
    # SFTTrainer rejects reward_funcs — when no reward_fn is given, it must be absent.
    class SFTLikeTrainer:
        def __init__(self, processing_class=None):
            pass

    kwargs = build_trainer_kwargs(
        trainer_cls=SFTLikeTrainer,
        model="model",
        training_args="args",
        train_dataset="dataset",
        tokenizer="tokenizer",
    )

    assert "reward_funcs" not in kwargs
    assert kwargs["processing_class"] == "tokenizer"


# ---- Text SFT (sft_text method) ----
#
# Ports the TRL SFTTrainer quickstart (Capybara) shape: SFTConfig with
# packing / dataset_text_field / max_length, dispatched by method="sft_text".


def test_build_sft_config_kwargs_adapts_to_supported_parameters():
    class FakeSFTConfig:
        def __init__(self, packing=None, dataset_text_field=None, max_length=None):
            pass

    kwargs = build_sft_config_kwargs(
        job_id="job-1",
        hyperparameters={
            "learning_rate": 1e-4,
            "num_epochs": 3,
            "max_steps": 50,
            "packing": True,
            "dataset_text_field": "content",
            "max_seq_length": 2048,
        },
        sft_config_cls=FakeSFTConfig,
    )

    assert kwargs["output_dir"] == str(Path("/tmp/ckpts") / "job-1")
    assert kwargs["learning_rate"] == 1e-4
    assert kwargs["num_train_epochs"] == 3
    assert kwargs["max_steps"] == 50
    assert kwargs["packing"] is True
    assert kwargs["dataset_text_field"] == "content"
    # TRL's current name is max_length; the 2048 lands there, not max_seq_length.
    assert kwargs["max_length"] == 2048
    assert "max_seq_length" not in kwargs


def test_use_liger_flows_into_configs_only_when_requested_and_supported():
    # GOOD: use_liger=True + a signature exposing use_liger_kernel -> flag set.
    class LigerSFTConfig:
        def __init__(self, use_liger_kernel=None, packing=None):
            pass

    on = build_sft_config_kwargs(
        job_id="j", hyperparameters={"use_liger": True}, sft_config_cls=LigerSFTConfig
    )
    assert on["use_liger_kernel"] is True

    # BAD 1 it guards: absent hyperparameter must NOT enable liger (numerics
    # opt-in boundary, same posture as use_unsloth/use_vllm).
    off = build_sft_config_kwargs(
        job_id="j", hyperparameters={}, sft_config_cls=LigerSFTConfig
    )
    assert "use_liger_kernel" not in off

    # BAD 2 it guards: an old transformers without the flag must silently
    # train without it, not TypeError the constructor.
    class OldSFTConfig:
        def __init__(self, packing=None):
            pass

    old = build_sft_config_kwargs(
        job_id="j", hyperparameters={"use_liger": True}, sft_config_cls=OldSFTConfig
    )
    assert "use_liger_kernel" not in old

    # GOOD: the same opt-in works on the GRPO and DPO builders.
    class LigerGRPOConfig:
        def __init__(self, use_liger_kernel=None):
            pass

    grpo = build_grpo_config_kwargs(
        job_id="j",
        hyperparameters={
            "use_liger": True, "rollouts_per_prompt": 4,
            "max_response_length": 128, "learning_rate": 1e-6,
        },
        grpo_config_cls=LigerGRPOConfig,
    )
    assert grpo["use_liger_kernel"] is True

    class LigerDPOConfig:
        def __init__(self, use_liger_kernel=None, beta=None):
            pass

    dpo = build_dpo_config_kwargs(
        job_id="j", hyperparameters={"use_liger": True}, dpo_config_cls=LigerDPOConfig
    )
    assert dpo["use_liger_kernel"] is True


def test_build_sft_config_kwargs_omits_unsupported_and_uses_legacy_seq_len():
    class OldSFTConfig:
        def __init__(self, max_seq_length=None):
            pass

    kwargs = build_sft_config_kwargs(
        job_id="job-1",
        hyperparameters={"max_seq_length": 1024},
        sft_config_cls=OldSFTConfig,
    )

    # No packing/dataset_text_field on the old signature → omitted.
    assert "packing" not in kwargs
    assert "dataset_text_field" not in kwargs
    # Older TRL exposes max_seq_length; the value lands there.
    assert kwargs["max_seq_length"] == 1024
    assert "max_length" not in kwargs


class _ChatTemplateTok:
    """Mimics tokenizer.apply_chat_template with a deterministic rendering."""

    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)


def test_prerender_chat_template_renders_messages_for_unsloth():
    rows = [{"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]}]
    out = prerender_chat_template_rows(
        rows, use_unsloth=True, tokenizer=_ChatTemplateTok(), text_field="text",
    )
    # Rendered through the template into the text field, chat column dropped so
    # TRL's conversational auto-detection can't re-fire on the unsloth path.
    assert out == [{"text": "<user>hi</user><assistant>hello</assistant>"}]


def test_prerender_chat_template_leaves_vanilla_and_text_rows_alone():
    chat_rows = [{"messages": [{"role": "user", "content": "hi"}]}]
    text_rows = [{"text": "already plain"}]
    # Vanilla TRL applies the chat template itself — rows pass through untouched.
    assert prerender_chat_template_rows(
        chat_rows, use_unsloth=False, tokenizer=_ChatTemplateTok(),
    ) is chat_rows
    # Unsloth with already-plain text has nothing to render.
    assert prerender_chat_template_rows(
        text_rows, use_unsloth=True, tokenizer=_ChatTemplateTok(),
    ) is text_rows


def test_prerender_chat_template_passes_prompt_completion_through():
    # Unsloth handles prompt+completion natively (unsloth_zoo dataset_utils
    # branches on those columns before the formatting_func raise), so these
    # rows must NOT be flattened — flattening would lose completion_only_loss.
    pc_rows = [{"prompt": "2+2=", "completion": "4"}]
    conv_pc_rows = [{
        "prompt": [{"role": "user", "content": "2+2="}],
        "completion": [{"role": "assistant", "content": "4"}],
    }]
    assert prerender_chat_template_rows(
        pc_rows, use_unsloth=True, tokenizer=_ChatTemplateTok(),
    ) is pc_rows
    assert prerender_chat_template_rows(
        conv_pc_rows, use_unsloth=True, tokenizer=_ChatTemplateTok(),
    ) is conv_pc_rows


def test_run_training_dispatches_sft_text(monkeypatch):
    """method='sft_text' routes to run_sft_text_training via the hardcoded arm
    (the Python registry is retired on the AMI), passing kwargs through."""
    called = {}

    def fake_sft(job_config, **kwargs):
        called["job_config"] = job_config
        called["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr("veri_runner.training_runtime.run_sft_text_training", fake_sft)

    out = run_training(
        {"method": "sft_text", "job_id": "j", "base_model": "m", "hyperparameters": {}},
        dataset_path="/data",
        reward_path="/r",
    )

    assert out == {"ok": True}
    assert called["job_config"]["method"] == "sft_text"
    assert called["kwargs"]["dataset_path"] == "/data"
    # reward_path is forwarded uniformly; the real sft fn swallows it via **_ignored.
    assert called["kwargs"]["reward_path"] == "/r"


# ---- DPO (dpo method) ----
#
# Ports the TRL DPOTrainer quickstart (ultrafeedback_binarized preference pairs):
# DPOConfig with beta / loss_type / max_length, dispatched by method="dpo".


def test_build_dpo_config_kwargs_adapts_to_supported_parameters():
    class FakeDPOConfig:
        def __init__(self, beta=None, loss_type=None, max_length=None, max_prompt_length=None):
            pass

    kwargs = build_dpo_config_kwargs(
        job_id="job-1",
        hyperparameters={
            "learning_rate": 1e-5,
            "num_epochs": 2,
            "max_steps": 40,
            "beta": 0.3,
            "loss_type": "ipo",
            "max_length": 1024,
            "max_prompt_length": 512,
        },
        dpo_config_cls=FakeDPOConfig,
    )

    assert kwargs["output_dir"] == str(Path("/tmp/ckpts") / "job-1")
    assert kwargs["learning_rate"] == 1e-5
    assert kwargs["num_train_epochs"] == 2
    assert kwargs["max_steps"] == 40
    assert kwargs["beta"] == 0.3
    assert kwargs["loss_type"] == "ipo"
    assert kwargs["max_length"] == 1024
    assert kwargs["max_prompt_length"] == 512


def test_build_dpo_config_kwargs_omits_unsupported_and_defaults_beta():
    class OldDPOConfig:
        def __init__(self, beta=None):
            pass

    kwargs = build_dpo_config_kwargs(
        job_id="job-1",
        hyperparameters={"max_length": 2048},
        dpo_config_cls=OldDPOConfig,
    )

    # beta is supported → defaults to TRL's 0.1 when unset.
    assert kwargs["beta"] == 0.1
    # loss_type / max_length / max_prompt_length aren't on this signature → omitted.
    assert "loss_type" not in kwargs
    assert "max_length" not in kwargs
    assert "max_prompt_length" not in kwargs


def test_require_preference_columns_rejects_sft_shaped_rows():
    # BAD: an SFT/GRPO-shaped dataset (no 'rejected') must fail fast, not crash
    # deep in TRL after a GPU boot.
    with pytest.raises(ValueError, match="rejected"):
        _require_preference_columns([{"prompt": "hi", "chosen": "good"}])


def test_require_preference_columns_accepts_preference_rows():
    # GOOD: a proper preference pair passes through.
    _require_preference_columns(
        [{"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}]
    )


def test_run_training_dispatches_dpo(monkeypatch):
    """method='dpo' routes to run_dpo_training via the hardcoded arm
    (the Python registry is retired on the AMI), passing kwargs through."""
    called = {}

    def fake_dpo(job_config, **kwargs):
        called["job_config"] = job_config
        called["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr("veri_runner.training_runtime.run_dpo_training", fake_dpo)

    out = run_training(
        {"method": "dpo", "job_id": "j", "base_model": "m", "hyperparameters": {}},
        dataset_path="/data",
        reward_path="/r",
    )

    assert out == {"ok": True}
    assert called["job_config"]["method"] == "dpo"
    assert called["kwargs"]["dataset_path"] == "/data"
    # reward_path is forwarded uniformly; the real dpo fn swallows it via **_ignored.
    assert called["kwargs"]["reward_path"] == "/r"


# ---- SA-5: ROCm (AMD MI300X) guard ----
#
# The ROCm worker stack ships no unsloth/bitsandbytes (CUDA-only). The guard
# sits in run_training (the single dispatch chokepoint), so a CUDA-only option
# fails fast with a clear reason instead of an ImportError mid-run.


def test_rocm_rejects_cuda_only_options_before_dispatch(monkeypatch):
    # BAD: on a ROCm box, use_unsloth / load_in_4bit must raise BEFORE any
    # dispatch arm runs (no GPU time burned, no ImportError deep in TRL).
    called = {}

    def fake_grpo(job_config, **kwargs):
        called["dispatched"] = True
        return {"ok": True}

    monkeypatch.setattr("veri_runner.training_runtime.run_grpo_training", fake_grpo)
    monkeypatch.setattr("veri_runner.training_runtime._is_rocm", lambda: True)

    with pytest.raises(ValueError, match="use_unsloth.*ROCm|ROCm.*use_unsloth"):
        run_training(
            {"method": "grpo", "job_id": "j", "base_model": "m",
             "hyperparameters": {"use_unsloth": True}},
            dataset_path="/data",
        )
    with pytest.raises(ValueError, match="load_in_4bit"):
        run_training(
            {"method": "grpo", "job_id": "j", "base_model": "m",
             "hyperparameters": {"load_in_4bit": True, "lora_rank": 16}},
            dataset_path="/data",
        )
    assert "dispatched" not in called, "guard must fire before the dispatch arm"


def test_rocm_vanilla_path_and_nvidia_unsloth_still_dispatch(monkeypatch):
    # GOOD 1: ROCm + vanilla hyperparameters dispatches normally.
    called = {}

    def fake_grpo(job_config, **kwargs):
        called["hp"] = job_config["hyperparameters"]
        return {"ok": True}

    monkeypatch.setattr("veri_runner.training_runtime.run_grpo_training", fake_grpo)
    monkeypatch.setattr("veri_runner.training_runtime._is_rocm", lambda: True)
    out = run_training(
        {"method": "grpo", "job_id": "j", "base_model": "m", "hyperparameters": {}},
        dataset_path="/data",
    )
    assert out == {"ok": True} and called["hp"] == {}

    # GOOD 2: on NVIDIA (non-ROCm), use_unsloth stays allowed; the guard is
    # ROCm-scoped, not a global unsloth ban (opt-in boundary intact).
    monkeypatch.setattr("veri_runner.training_runtime._is_rocm", lambda: False)
    out = run_training(
        {"method": "grpo", "job_id": "j", "base_model": "m",
         "hyperparameters": {"use_unsloth": True}},
        dataset_path="/data",
    )
    assert out == {"ok": True} and called["hp"] == {"use_unsloth": True}


# ---- ROCm perf opt-ins (liger availability gate + TunableOp) ----


def test_use_liger_without_package_fails_fast_before_dispatch(monkeypatch):
    # BAD it guards: use_liger on an image without liger-kernel must raise the
    # reason BEFORE the dispatch arm (no model download, no opaque
    # transformers ImportError mid-run).
    called = {}
    def fake_grpo(job_config, **kw):
        called["dispatched"] = True
        return {"ok": True}

    monkeypatch.setattr("veri_runner.training_runtime.run_grpo_training", fake_grpo)
    monkeypatch.setattr("veri_runner.training_runtime._is_rocm", lambda: True)
    monkeypatch.setattr("veri_runner.training_runtime._liger_available", lambda: False)
    with pytest.raises(ValueError, match="liger-kernel.*not.*installed"):
        run_training(
            {"method": "grpo", "job_id": "j", "base_model": "m",
             "hyperparameters": {"use_liger": True}},
            dataset_path="/data",
        )
    assert "dispatched" not in called

    # GOOD: with the package present, the same job dispatches.
    monkeypatch.setattr("veri_runner.training_runtime._liger_available", lambda: True)
    out = run_training(
        {"method": "grpo", "job_id": "j", "base_model": "m",
         "hyperparameters": {"use_liger": True}},
        dataset_path="/data",
    )
    assert out == {"ok": True} and called["dispatched"]


def test_tunableop_is_opt_in(monkeypatch):
    # TunableOp tunes at runtime (costly on variable shapes), so it must only
    # engage when the hyperparameter asks for it.
    import sys
    import types

    calls = []
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        tunable=types.SimpleNamespace(enable=lambda v: calls.append(v))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("veri_runner.training_runtime._is_rocm", lambda: False)
    monkeypatch.setattr(
        "veri_runner.training_runtime.run_grpo_training", lambda job_config, **kw: {"ok": True}
    )

    # BAD it guards: no flag -> TunableOp untouched.
    run_training(
        {"method": "grpo", "job_id": "j", "base_model": "m", "hyperparameters": {}},
        dataset_path="/data",
    )
    assert calls == []

    # GOOD: flag set -> enabled before dispatch.
    run_training(
        {"method": "grpo", "job_id": "j", "base_model": "m",
         "hyperparameters": {"tunableop": True}},
        dataset_path="/data",
    )
    assert calls == [True]


def test_hipblaslt_is_opt_in_and_rocm_scoped(monkeypatch):
    # Shipped always-on once and benchmarked 25% SLOWER on small GEMMs — the
    # env must only appear when asked for, and only on ROCm.
    import os

    monkeypatch.delenv("TORCH_BLAS_PREFER_HIPBLASLT", raising=False)
    monkeypatch.setattr(
        "veri_runner.training_runtime.run_grpo_training", lambda job_config, **kw: {"ok": True}
    )

    # BAD 1 it guards: no flag -> env untouched (even on ROCm).
    monkeypatch.setattr("veri_runner.training_runtime._is_rocm", lambda: True)
    run_training(
        {"method": "grpo", "job_id": "j", "base_model": "m", "hyperparameters": {}},
        dataset_path="/data",
    )
    assert "TORCH_BLAS_PREFER_HIPBLASLT" not in os.environ

    # BAD 2 it guards: flag on a non-ROCm box -> no-op (the env var is
    # meaningless on CUDA; don't leak AMD knobs across vendors).
    monkeypatch.setattr("veri_runner.training_runtime._is_rocm", lambda: False)
    run_training(
        {"method": "grpo", "job_id": "j", "base_model": "m",
         "hyperparameters": {"hipblaslt": True}},
        dataset_path="/data",
    )
    assert "TORCH_BLAS_PREFER_HIPBLASLT" not in os.environ

    # GOOD: flag + ROCm -> env set before dispatch.
    monkeypatch.setattr("veri_runner.training_runtime._is_rocm", lambda: True)
    run_training(
        {"method": "grpo", "job_id": "j", "base_model": "m",
         "hyperparameters": {"hipblaslt": True}},
        dataset_path="/data",
    )
    assert os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] == "1"


def test_save_checkpoint_returns_final_checkpoint_dir():
    class FakeTrainer:
        def __init__(self):
            self.saved_to = None

        def save_model(self, final_dir):
            self.saved_to = final_dir

    class FakeTokenizer:
        def __init__(self):
            self.saved_to = None

        def save_pretrained(self, final_dir):
            self.saved_to = final_dir

    trainer = FakeTrainer()
    tokenizer = FakeTokenizer()

    final_dir = save_checkpoint(
        trainer=trainer,
        tokenizer=tokenizer,
        job_id="job-123",
        output_root="/tmp/test-ckpts",
    )

    assert final_dir == str(Path("/tmp/test-ckpts") / "job-123" / "final")
    assert trainer.saved_to == final_dir
    assert tokenizer.saved_to == final_dir


# ---- Unsloth model loading tests ----

_BASE_HP = {
    "max_prompt_length": 512,
    "max_response_length": 1024,
}


class _FakeUnsloth:
    """Minimal stand-in for unsloth.FastLanguageModel."""

    def __init__(self):
        self.from_pretrained_calls = []
        self.get_peft_model_calls = []

    def from_pretrained(self, **kwargs):
        self.from_pretrained_calls.append(kwargs)
        return "unsloth_model", "unsloth_tokenizer"

    def get_peft_model(self, model, **kwargs):
        self.get_peft_model_calls.append(kwargs)
        return "lora_model"


def test_load_model_unsloth_success(monkeypatch):
    fake = _FakeUnsloth()
    import types

    mock_mod = types.ModuleType("unsloth")
    mock_mod.FastLanguageModel = fake
    monkeypatch.setitem(__import__("sys").modules, "unsloth", mock_mod)

    hp = {**_BASE_HP, "use_unsloth": True}
    model, tok = _load_model_and_tokenizer("Qwen/Qwen3-4B", hp, logging.getLogger())

    assert model == "unsloth_model"
    assert tok == "unsloth_tokenizer"
    assert fake.from_pretrained_calls[0]["model_name"] == "Qwen/Qwen3-4B"
    assert fake.from_pretrained_calls[0]["max_seq_length"] == 1536
    assert fake.from_pretrained_calls[0]["full_finetuning"] is True
    assert len(fake.get_peft_model_calls) == 0


def test_load_model_unsloth_with_lora(monkeypatch):
    fake = _FakeUnsloth()
    import types

    mock_mod = types.ModuleType("unsloth")
    mock_mod.FastLanguageModel = fake
    monkeypatch.setitem(__import__("sys").modules, "unsloth", mock_mod)

    hp = {**_BASE_HP, "use_unsloth": True, "lora_rank": 32, "lora_alpha": 64}
    model, _ = _load_model_and_tokenizer("Qwen/Qwen3-4B", hp, logging.getLogger())

    assert model == "lora_model"
    assert fake.from_pretrained_calls[0]["full_finetuning"] is False
    assert fake.get_peft_model_calls[0]["r"] == 32
    assert fake.get_peft_model_calls[0]["lora_alpha"] == 64


def test_load_model_unsloth_fallback_on_import_error(monkeypatch):
    monkeypatch.delitem(__import__("sys").modules, "unsloth", raising=False)
    monkeypatch.setattr(
        "builtins.__import__",
        _make_import_raiser("unsloth", monkeypatch),
    )

    hp = {**_BASE_HP, "use_unsloth": True}
    # Should not raise — falls back to vanilla (which we also mock)
    _mock_vanilla_and_call(monkeypatch, hp)


def test_load_model_unsloth_fallback_on_model_error(monkeypatch):
    import types

    mock_mod = types.ModuleType("unsloth")

    class _Failing:
        @staticmethod
        def from_pretrained(**kwargs):
            raise RuntimeError("Unsupported architecture")

    mock_mod.FastLanguageModel = _Failing()
    monkeypatch.setitem(__import__("sys").modules, "unsloth", mock_mod)

    hp = {**_BASE_HP, "use_unsloth": True}
    _mock_vanilla_and_call(monkeypatch, hp)


def test_load_model_unsloth_disabled(monkeypatch):
    hp = {**_BASE_HP, "use_unsloth": False}
    model, tok = _mock_vanilla_and_call(monkeypatch, hp)
    assert model == "vanilla_model"
    # tok is a fake tokenizer instance (not a string) from _mock_vanilla_and_call
    assert tok is not None


def test_load_model_default_skips_unsloth(monkeypatch):
    """Default (no use_unsloth flag set) must go straight to vanilla
    transformers — never try unsloth. The torch.compile selective log-softmax
    has an open shape-mismatch bug with GRPOTrainer (unsloth #3069); making
    it opt-in keeps the quickstart stable."""
    hp = {k: v for k, v in _BASE_HP.items() if k != "use_unsloth"}
    model, tok = _mock_vanilla_and_call(monkeypatch, hp)
    assert model == "vanilla_model"
    assert tok is not None


# -- helpers for fallback tests --

import logging  # noqa: E402 — deliberate section break; helpers imported mid-module
import sys  # noqa: E402


def _make_import_raiser(blocked_name, monkeypatch):
    original = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _import(name, *args, **kwargs):
        if name == blocked_name:
            raise ImportError(f"No module named '{blocked_name}'")
        return original(name, *args, **kwargs)

    return _import


def _mock_vanilla_and_call(monkeypatch, hp):
    """Mock transformers + torch, call _load_model_and_tokenizer, return (model, tok)."""
    import types

    fake_tokenizer = type("FakeTok", (), {
        "from_pretrained": staticmethod(
            lambda *a, **kw: type("T", (), {"pad_token": None, "eos_token": "<eos>"})()
        ),
    })
    fake_auto = type("FakeAuto", (), {
        "from_pretrained": staticmethod(lambda *a, **kw: "vanilla_model"),
    })
    # patch the tokenizer's pad_token assignment
    tok_instance = fake_tokenizer.from_pretrained()
    tok_instance.pad_token = tok_instance.eos_token

    mock_transformers = types.ModuleType("transformers")
    mock_transformers.AutoTokenizer = fake_tokenizer
    mock_transformers.AutoModelForCausalLM = fake_auto
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    mock_torch = types.ModuleType("torch")
    mock_torch.bfloat16 = "bf16"
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    model, tok = _load_model_and_tokenizer("test/model", hp, logging.getLogger())
    return model, tok


# ---- Native PEFT / bitsandbytes LoRA + QLoRA tests (vanilla path) ----
#
# Ports the proven HF PEFT QLoRA recipe (BitsAndBytesConfig nf4 +
# prepare_model_for_kbit_training + get_peft_model(LoraConfig)) to our vanilla
# model-load path, then asserts the exact wiring. Deps (bitsandbytes, peft) are
# already on the AMI; this path needs no Unsloth.


def _install_vanilla_peft_mocks(monkeypatch):
    """Mock transformers (incl. BitsAndBytesConfig) + torch + peft, capturing the
    calls so a test can assert the QLoRA/LoRA wiring deterministically."""
    import types

    calls = {
        "model_kwargs": [],
        "bnb": [],
        "lora_config_kwargs": [],
        "get_peft_model": [],
        "prepared": [],
    }

    fake_tokenizer = type("FakeTok", (), {
        "from_pretrained": staticmethod(
            lambda *a, **kw: type("T", (), {"pad_token": None, "eos_token": "<eos>"})()
        ),
    })

    def _model_from_pretrained(name, **kw):
        calls["model_kwargs"].append(kw)
        return "base_model"

    fake_auto = type("FakeAuto", (), {"from_pretrained": staticmethod(_model_from_pretrained)})

    def _bnb(**kw):
        calls["bnb"].append(kw)
        return ("bnb_config", kw)

    mock_transformers = types.ModuleType("transformers")
    mock_transformers.AutoTokenizer = fake_tokenizer
    mock_transformers.AutoModelForCausalLM = fake_auto
    mock_transformers.BitsAndBytesConfig = _bnb
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    mock_torch = types.ModuleType("torch")
    mock_torch.bfloat16 = "bf16"
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    def _lora_config(**kw):
        calls["lora_config_kwargs"].append(kw)
        return ("lora_config", kw)

    def _get_peft_model(model, config):
        calls["get_peft_model"].append({"model": model, "config": config})
        return "peft_model"

    def _prepare(model):
        calls["prepared"].append(model)
        return "prepared_model"

    mock_peft = types.ModuleType("peft")
    mock_peft.LoraConfig = _lora_config
    mock_peft.get_peft_model = _get_peft_model
    mock_peft.prepare_model_for_kbit_training = _prepare
    monkeypatch.setitem(sys.modules, "peft", mock_peft)

    return calls


def test_load_model_qlora_4bit_wires_bnb_and_peft(monkeypatch):
    """GOOD: load_in_4bit + lora_rank → NF4 BitsAndBytesConfig, kbit prep, and a
    PEFT LoRA wrap. Asserts the concrete kwargs (deterministic proof)."""
    calls = _install_vanilla_peft_mocks(monkeypatch)
    hp = {**_BASE_HP, "lora_rank": 16, "lora_alpha": 32, "load_in_4bit": True}

    model, _ = _load_model_and_tokenizer("Qwen/Qwen3-4B", hp, logging.getLogger())

    # 4-bit quantization config is the proven NF4 recipe.
    assert calls["bnb"][0] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bf16",
    }
    # The quantized config is handed to from_pretrained.
    assert "quantization_config" in calls["model_kwargs"][0]
    # kbit prep runs before the LoRA wrap (QLoRA requirement).
    assert calls["prepared"] == ["base_model"]
    # LoRA adapters with the requested rank/alpha and causal-LM head.
    lora = calls["lora_config_kwargs"][0]
    assert lora["r"] == 16
    assert lora["lora_alpha"] == 32
    assert lora["task_type"] == "CAUSAL_LM"
    assert "q_proj" in lora["target_modules"] and "down_proj" in lora["target_modules"]
    # get_peft_model wraps the kbit-prepared model; final model is the PEFT model.
    assert calls["get_peft_model"][0]["model"] == "prepared_model"
    assert model == "peft_model"


def test_load_model_lora_16bit_skips_quantization(monkeypatch):
    """GOOD: lora_rank without load_in_4bit → PEFT LoRA, NO bitsandbytes, NO kbit
    prep. This is the multi-GPU-capable 16-bit LoRA path."""
    calls = _install_vanilla_peft_mocks(monkeypatch)
    hp = {**_BASE_HP, "lora_rank": 8}

    model, _ = _load_model_and_tokenizer("Qwen/Qwen3-4B", hp, logging.getLogger())

    assert calls["bnb"] == []
    assert "quantization_config" not in calls["model_kwargs"][0]
    assert calls["prepared"] == []
    assert calls["lora_config_kwargs"][0]["r"] == 8
    assert model == "peft_model"


def test_load_model_qlora_without_lora_rank_raises(monkeypatch):
    """BAD/guard: load_in_4bit with no lora_rank is invalid (the 4-bit base is
    frozen, so there is nothing to train). Must fail fast, before any model load."""
    hp = {**_BASE_HP, "load_in_4bit": True}

    with pytest.raises(ValueError, match="load_in_4bit=True .*requires lora_rank"):
        _load_model_and_tokenizer("Qwen/Qwen3-4B", hp, logging.getLogger())


def test_load_model_unsloth_qlora_sets_4bit(monkeypatch):
    """GOOD: Unsloth opt-in path honors load_in_4bit and disables full-finetune."""
    fake = _FakeUnsloth()
    import types

    mock_mod = types.ModuleType("unsloth")
    mock_mod.FastLanguageModel = fake
    monkeypatch.setitem(sys.modules, "unsloth", mock_mod)

    hp = {**_BASE_HP, "use_unsloth": True, "lora_rank": 16, "load_in_4bit": True}
    model, _ = _load_model_and_tokenizer("Qwen/Qwen3-4B", hp, logging.getLogger())

    assert fake.from_pretrained_calls[0]["load_in_4bit"] is True
    assert fake.from_pretrained_calls[0]["full_finetuning"] is False
    assert fake.get_peft_model_calls[0]["r"] == 16
    assert model == "lora_model"


# ---- trust_remote_code must default OFF for user-controlled base_model ----
#
# trust_remote_code=True makes HuggingFace execute arbitrary modeling_*.py /
# configuration_*.py from the referenced repo IN-PROCESS on the training host
# (which holds the callback token, W&B key, volume STS creds, and reachable IMDS).
# base_model is a free-form, user-controlled field with NO control-plane allowlist,
# so remote code must be OFF unless the repo is on the worker's vetted allowlist.


def _install_trust_capture_mocks(monkeypatch):
    """Mock transformers + torch, capturing trust_remote_code passed to BOTH the
    tokenizer and the model from_pretrained calls (the two remote-code seams)."""
    import types

    calls = {"tokenizer_kwargs": [], "model_kwargs": []}

    def _tok_from_pretrained(name, **kw):
        calls["tokenizer_kwargs"].append(kw)
        return type("T", (), {"pad_token": None, "eos_token": "<eos>"})()

    fake_tokenizer = type(
        "FakeTok", (), {"from_pretrained": staticmethod(_tok_from_pretrained)}
    )

    def _model_from_pretrained(name, **kw):
        calls["model_kwargs"].append(kw)
        return "vanilla_model"

    fake_auto = type(
        "FakeAuto", (), {"from_pretrained": staticmethod(_model_from_pretrained)}
    )

    mock_transformers = types.ModuleType("transformers")
    mock_transformers.AutoTokenizer = fake_tokenizer
    mock_transformers.AutoModelForCausalLM = fake_auto
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    mock_torch = types.ModuleType("torch")
    mock_torch.bfloat16 = "bf16"
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    return calls


def test_trust_remote_code_off_for_unlisted_base_model(monkeypatch):
    """BAD/guard: a base_model NOT on the vetted allowlist must load with
    trust_remote_code=False on both the tokenizer and the model (no in-process
    RCE from a user-supplied repo)."""
    monkeypatch.delenv("VERI_TRUST_REMOTE_CODE_MODELS", raising=False)
    calls = _install_trust_capture_mocks(monkeypatch)

    _load_model_and_tokenizer(
        "attacker/backdoor-model", dict(_BASE_HP), logging.getLogger()
    )

    assert calls["tokenizer_kwargs"][0]["trust_remote_code"] is False
    assert calls["model_kwargs"][0]["trust_remote_code"] is False


def test_trust_remote_code_on_for_allowlisted_base_model(monkeypatch):
    """GOOD: an operator-allowlisted base_model (via the out-of-band env allowlist,
    NOT the user-controlled job payload) loads with trust_remote_code=True."""
    monkeypatch.setenv(
        "VERI_TRUST_REMOTE_CODE_MODELS", "vetted-org/custom-arch,other/model"
    )
    calls = _install_trust_capture_mocks(monkeypatch)

    _load_model_and_tokenizer(
        "vetted-org/custom-arch", dict(_BASE_HP), logging.getLogger()
    )

    assert calls["tokenizer_kwargs"][0]["trust_remote_code"] is True
    assert calls["model_kwargs"][0]["trust_remote_code"] is True


# ---------------------------------------------------------------------------
# dataset format detection + normalization
#
# Fixtures in tests/fixtures/ are REAL first rows pulled from the HF Hub
# (datasets-server API, 2026-07-16): FineTome-100k (ShareGPT), alpaca-cleaned
# (Alpaca), trl-lib/Capybara (ChatML). Do not hand-edit them.
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_rows(name):
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines()]


def _render_chatml(messages):
    """Render messages the way a ChatML chat template does. This is the shape
    Qwen-family templates produce, so `<|im_start|>{role}` shows exactly which
    role string the model would be trained on."""
    return "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )


def test_detect_dataset_format_across_families():
    assert detect_dataset_format(_fixture_rows("sharegpt_finetome.jsonl")[0]) == "sharegpt"
    assert detect_dataset_format(_fixture_rows("alpaca_cleaned.jsonl")[0]) == "alpaca"
    assert detect_dataset_format(_fixture_rows("chatml_capybara.jsonl")[0]) == "chatml"
    assert detect_dataset_format({"text": "The sky is blue."}) == "text"
    assert detect_dataset_format({"prompt": "2+2?", "completion": "4"}) == "prompt_completion"
    assert detect_dataset_format({"prompt": "2+2?", "chosen": "4", "rejected": "5"}) == "preference"
    assert detect_dataset_format({"prompt": "2+2?", "answer": "4"}) == "prompt_only"
    assert detect_dataset_format({"foo": 1}) == "unknown"


def test_sharegpt_key_only_rename_is_the_bug():
    # BAD (replicates today's silent failure): TRL 0.22.2's maybe_convert_to_chatml
    # renames conversations/from/value -> messages/role/content but copies the role
    # VALUES through, so the chat template renders `<|im_start|>human` and the job
    # "succeeds" on malformed conversations.
    row = _fixture_rows("sharegpt_finetome.jsonl")[0]
    key_only = [
        {"role": m["from"], "content": m["value"]} for m in row["conversations"]
    ]
    rendered = _render_chatml(key_only)
    print("\n--- key-only rename (today's behavior) renders ---")
    print(rendered[:200])
    assert "<|im_start|>human" in rendered  # the silent failure


def test_normalize_sharegpt_remaps_roles_and_keys():
    # GOOD: normalize_rows converts to ChatML with the role map applied.
    rows = _fixture_rows("sharegpt_finetome.jsonl")
    out = normalize_rows(rows)

    for row in out:
        assert "conversations" not in row
        roles = [m["role"] for m in row["messages"]]
        assert set(roles) <= {"user", "assistant", "system"}

    rendered = _render_chatml(out[0]["messages"])
    print("\n--- normalized renders ---")
    print(rendered[:200])
    assert "<|im_start|>user" in rendered
    assert "<|im_start|>assistant" in rendered
    assert "human" not in {m["role"] for r in out for m in r["messages"]}

    # Non-format columns survive (GRPO forwards extras to rewards).
    assert "source" in out[0] and "score" in out[0]
    # Content is untouched.
    assert out[0]["messages"][0]["content"] == rows[0]["conversations"][0]["value"]


def test_normalize_role_name_variants():
    # OASST `prompter`, Vicuna upper-case USER/ASSISTANT fold into the same map.
    rows = [{
        "conversations": [
            {"from": "prompter", "value": "hi"},
            {"from": "ASSISTANT", "value": "hello"},
        ]
    }]
    out = normalize_rows(rows)
    assert [m["role"] for m in out[0]["messages"]] == ["user", "assistant"]


def test_normalize_sharegpt_rejects_unmapped_role():
    # BAD: tool-calling roles (observation/function_call) are not silently
    # mistrained; they fail fast with the classifiable error prefix.
    rows = [{
        "conversations": [
            {"from": "human", "value": "look this up"},
            {"from": "observation", "value": "{...tool output...}"},
        ]
    }]
    with pytest.raises(ValueError, match="dataset_format_incompatible.*observation"):
        normalize_rows(rows)


def test_normalize_alpaca_composes_messages():
    rows = _fixture_rows("alpaca_cleaned.jsonl")
    out = normalize_rows(rows)

    # Empty input: single user turn == instruction verbatim.
    no_input = out[0]
    assert rows[0]["input"] == ""
    assert no_input["messages"] == [
        {"role": "user", "content": rows[0]["instruction"]},
        {"role": "assistant", "content": rows[0]["output"]},
    ]

    # Non-empty input: instruction + blank line + input in the user turn.
    with_input = out[3]
    assert rows[3]["input"].strip()
    assert with_input["messages"][0]["content"] == (
        rows[3]["instruction"] + "\n\n" + rows[3]["input"]
    )
    assert with_input["messages"][1] == {
        "role": "assistant", "content": rows[3]["output"],
    }

    # The composed columns are gone.
    for row in out:
        for k in ("instruction", "input", "output"):
            assert k not in row


def test_normalize_chatml_passthrough_and_idempotent():
    rows = _fixture_rows("chatml_capybara.jsonl")
    once = normalize_rows(rows)
    assert once[0]["messages"] == rows[0]["messages"]
    # Normalizing a normalized dataset changes nothing (snapshot rows re-enter
    # the pipeline via the cached-JSONL path).
    assert normalize_rows(once) == once
    assert normalize_rows(normalize_rows(_fixture_rows("sharegpt_finetome.jsonl"))) == \
        normalize_rows(_fixture_rows("sharegpt_finetome.jsonl"))


def test_normalize_unknown_format_fails_fast():
    with pytest.raises(ValueError, match="dataset_format_incompatible.*foo"):
        normalize_rows([{"foo": 1, "bar": 2}])


def test_method_gate_sft_pair():
    # GOOD: chat rows pass for sft_text.
    validate_rows_for_method(normalize_rows(_fixture_rows("chatml_capybara.jsonl")), "sft_text")
    validate_rows_for_method([{"text": "plain"}], "sft_text")
    # BAD: a GRPO-shaped prompt-only dataset is not SFT-trainable.
    with pytest.raises(ValueError, match="dataset_format_incompatible.*sft_text"):
        validate_rows_for_method([{"prompt": "2+2?", "answer": "4"}], "sft_text")


def test_method_gate_grpo_pair():
    # GOOD: prompt column present.
    validate_rows_for_method([{"prompt": "2+2?", "answer": "4"}], "grpo")
    # BAD: a chat SFT dataset has no prompt; the error tells the user what to do.
    with pytest.raises(ValueError, match="dataset_format_incompatible.*grpo.*prompt"):
        validate_rows_for_method(
            normalize_rows(_fixture_rows("chatml_capybara.jsonl")), "grpo"
        )


def test_method_gate_dpo_delegates_to_preference_check():
    validate_rows_for_method([{"prompt": "p", "chosen": "a", "rejected": "b"}], "dpo")
    with pytest.raises(ValueError, match="preference"):
        validate_rows_for_method([{"text": "plain"}], "dpo")


def test_resolve_training_rows_normalizes_jsonl_path(tmp_path):
    # A ShareGPT-shaped JSONL (e.g. re-entering via a future ingress) comes out
    # of resolve_training_rows already normalized.
    f = tmp_path / "data.jsonl"
    f.write_text(json.dumps({
        "conversations": [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
        ]
    }))
    rows = resolve_training_rows(dataset_path=str(f), dataset_config={"source_type": "upload"})
    assert rows[0]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


class _FakeUrlopen:
    """Capture urllib PUTs (mirrors the log-uploader/checkpoint pattern)."""

    def __init__(self, fail=False):
        self.requests = []
        self.fail = fail

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        if self.fail:
            raise OSError("s3 said no")
        import io
        return io.BytesIO(b"")


def _mock_hf(monkeypatch, rows):
    import sys
    import types

    # `datasets` is a heavy GPU-worker dep, installed on the AMI but not in the
    # CI test env. Stub it into sys.modules the same way torch/transformers/
    # unsloth are stubbed above, so the source's lazy `from datasets import
    # load_dataset` (training_runtime.resolve_training_rows) picks up our fake.
    datasets_mod = types.ModuleType("datasets")

    def fake_load_dataset(name, subset=None, **kwargs):
        return rows

    datasets_mod.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)


def test_resolve_hf_normalizes_and_uploads_snapshot(monkeypatch):
    _mock_hf(monkeypatch, _fixture_rows("sharegpt_finetome.jsonl"))
    fake = _FakeUrlopen()
    monkeypatch.setattr("urllib.request.urlopen", fake)

    rows = resolve_training_rows(
        dataset_config={"source_type": "hf", "hf_dataset": "mlabonne/FineTome-100k"},
        snapshot_upload_url="https://s3/datasets/d1/normalized.jsonl?sig=x",
    )

    assert rows[0]["messages"][0]["role"] == "user"
    # One PUT of the normalized JSONL to the presigned URL.
    assert len(fake.requests) == 1
    put = fake.requests[0]
    assert put.get_method() == "PUT"
    assert put.full_url.startswith("https://s3/datasets/d1/normalized.jsonl")
    body = [json.loads(line) for line in put.data.decode().splitlines()]
    assert len(body) == len(rows)
    assert body[0]["messages"][0]["role"] == "user"
    assert "conversations" not in body[0]


def test_resolve_hf_no_snapshot_url_no_upload(monkeypatch):
    _mock_hf(monkeypatch, _fixture_rows("chatml_capybara.jsonl"))
    fake = _FakeUrlopen()
    monkeypatch.setattr("urllib.request.urlopen", fake)

    rows = resolve_training_rows(
        dataset_config={"source_type": "hf", "hf_dataset": "trl-lib/Capybara"},
    )

    assert rows and fake.requests == []


def test_resolve_hf_snapshot_upload_failure_is_nonfatal(monkeypatch):
    # The snapshot is a cache: an S3 hiccup must not kill a training run.
    _mock_hf(monkeypatch, _fixture_rows("sharegpt_finetome.jsonl"))
    monkeypatch.setattr("urllib.request.urlopen", _FakeUrlopen(fail=True))

    rows = resolve_training_rows(
        dataset_config={"source_type": "hf", "hf_dataset": "mlabonne/FineTome-100k"},
        snapshot_upload_url="https://s3/datasets/d1/normalized.jsonl?sig=x",
    )

    assert rows[0]["messages"][0]["role"] == "user"


def test_resolve_jsonl_path_never_uploads_snapshot(tmp_path, monkeypatch):
    # dataset_path means the rows already came FROM S3 (upload or snapshot hit);
    # re-uploading would be a pointless write on every job.
    fake = _FakeUrlopen()
    monkeypatch.setattr("urllib.request.urlopen", fake)
    f = tmp_path / "data.jsonl"
    f.write_text(json.dumps({"text": "plain"}))

    resolve_training_rows(
        dataset_path=str(f),
        dataset_config={"source_type": "hf", "hf_dataset": "x/y"},
        snapshot_upload_url="https://s3/datasets/d1/normalized.jsonl?sig=x",
    )

    assert fake.requests == []
