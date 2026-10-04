"""Managed multi-GPU launch (VS-476, phase 1): plan resolver, accelerate config,
launch argv, child config/env, builder wiring and the rank-0 progress file.

Pure-Python unit tests (no torch) plus one CPU integration test that runs the
real child entrypoint under torchrun with two gloo ranks (skipped when the ML
stack is not installed).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from veri_runner import training_runtime as rt
from veri_runner.training_runtime import (
    _attach_progress_callback,
    _force_fp32_grad_reduce,
    _load_model_and_tokenizer,
    build_child_config,
    build_child_env,
    build_dpo_config_kwargs,
    build_launch_argv,
    build_sft_config_kwargs,
    read_model_meta,
    render_accelerate_config,
    resolve_parallel_plan,
)

GB = 10**9
L4 = 24 * GB  # gpu_memory_bytes of an L4-24GB
BUDGET = int(0.7 * (L4 - 2 * 1024**3))  # 0.7 x (24 GB - 2 GiB reserve) = 15.3 GB


def _meta(n_params: int) -> dict:
    return {"param_count": n_params, "param_count_source": "test"}


def _plan(hp: dict, *, method="sft_text", gpus=4, mem=L4, n=600_000_000) -> dict:
    return resolve_parallel_plan(
        hp, method=method, gpu_count=gpus, gpu_memory_bytes=mem, model_meta=_meta(n)
    )


# ---- read_model_meta ----


def test_read_model_meta_prefers_safetensors_index_over_formula(tmp_path):
    # GOOD: an index gives the exact byte count; dtype bytes come from config.json.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "vocab_size": 151936,
                "torch_dtype": "bfloat16",
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1_200_000_000},
                "weight_map": {},
            }
        )
    )
    meta = read_model_meta(str(tmp_path))
    assert meta["param_count"] == 600_000_000
    assert meta["param_count_source"] == "safetensors_index"


def test_read_model_meta_reads_single_safetensors_header(tmp_path):
    # GOOD: a single-file checkpoint has no index; the 8-byte-length + JSON
    # header is read without loading tensors (exact count from shapes).
    (tmp_path / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
    header = {
        "__metadata__": {"format": "pt"},
        "a.weight": {"dtype": "BF16", "shape": [10, 20], "data_offsets": [0, 400]},
        "b.bias": {"dtype": "BF16", "shape": [5], "data_offsets": [400, 410]},
    }
    hb = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(len(hb).to_bytes(8, "little") + hb + b"\0" * 410)
    meta = read_model_meta(str(tmp_path))
    assert meta["param_count"] == 205
    assert meta["param_count_source"] == "safetensors_header"


def test_read_model_meta_falls_back_to_playbook_formula(tmp_path):
    # BAD->GOOD: no weights at all -> N = h*v + L(12h^2 + 13h) + 2h from config.json.
    h, v, L = 1024, 151936, 28
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": h,
                "num_hidden_layers": L,
                "vocab_size": v,
            }
        )
    )
    meta = read_model_meta(str(tmp_path))
    assert meta["param_count"] == h * v + L * (12 * h * h + 13 * h) + 2 * h
    assert meta["param_count_source"] == "config_formula"
    # and with nothing usable the count is unknown rather than a guess
    empty = tmp_path / "empty"
    empty.mkdir()
    assert read_model_meta(str(empty))["param_count"] is None


# ---- resolve_parallel_plan: decision traces ----


def test_plan_auto_picks_ddp_when_model_states_fit_one_gpu():
    # Qwen3-0.6B full FT bf16 on L4 x4: 8N = 4.8 GB <= 15.3 GB budget -> DP.
    plan = _plan({}, n=600_000_000)
    print("\n".join(plan["trace"]))
    assert plan["zero_stage"] == 0 and plan["backend"] == "ddp"
    assert plan["dp"] == 4 and plan["world_size"] == 4
    assert plan["estimate"]["model_states_bytes"] == 8 * 600_000_000
    assert plan["estimate"]["per_gpu_bytes"] == 8 * 600_000_000
    assert plan["estimate"]["budget_bytes"] == BUDGET
    assert plan["warnings"] == []
    assert any("zero_stage=0" in line for line in plan["trace"])


def test_plan_auto_picks_zero3_when_states_do_not_fit():
    # Qwen3-4B full FT: 32 GB > 15.3 GB -> ZeRO-3, 8 GB/GPU, no warning.
    plan = _plan({}, n=4_000_000_000)
    print("\n".join(plan["trace"]))
    assert plan["zero_stage"] == 3 and plan["backend"] == "fsdp2"
    assert plan["estimate"]["per_gpu_bytes"] == 8 * 4_000_000_000 // 4
    assert plan["warnings"] == []
    # phase 1 never auto-picks 2
    assert plan["zero_stage_requested"] == "auto"


def test_plan_warns_but_never_blocks_when_even_zero3_exceeds_budget():
    # 10B full FT: 80 GB / 4 = 20 GB/GPU > 15.3 GB -> launch anyway, WARN with options.
    plan = _plan({}, n=10_000_000_000)
    assert plan["zero_stage"] == 3
    assert len(plan["warnings"]) == 1
    w = plan["warnings"][0]
    print(w)
    assert "expect OOM" in w and "20.0 GB/GPU" in w and "15.3 GB" in w
    for option in ("lora_rank", "cpu_offload", "more GPUs", "precision"):
        assert option in w


def test_plan_reproduces_the_spec_worked_examples():
    # Section 3.1, with real Qwen3 parameter counts. H100: 80 GB -> budget 54.5 GB.
    H100 = 80 * GB
    q = {
        "0.6B": 596_049_920,
        "1.7B": 1_720_574_976,
        "4B": 4_022_468_096,
        "8B": 8_190_735_360,
        "32B": 32_762_123_264,
    }
    assert _plan({}, n=q["0.6B"])["zero_stage"] == 0  # 4.8 GB fits an L4
    four = _plan({}, n=q["4B"])
    assert four["zero_stage"] == 3 and four["warnings"] == []  # 32 GB -> 8 GB/GPU
    eight = _plan({}, n=q["8B"])
    assert eight["zero_stage"] == 3 and len(eight["warnings"]) == 1  # 16.4 GB/GPU > 15.3
    assert _plan({}, gpus=8, mem=H100, n=q["4B"])["zero_stage"] == 0  # 32 GB fits an H100
    big = _plan({}, gpus=8, mem=H100, n=q["32B"])
    assert big["zero_stage"] == 3 and big["warnings"] == []  # 262 GB -> 32.8 GB/GPU
    dpo17 = _plan({}, method="dpo", n=q["1.7B"])  # proof 3: 1.7B DPO full FT on L4 x4
    assert dpo17["zero_stage"] == 3 and dpo17["warnings"] == []


def test_plan_dpo_full_ft_adds_a_reference_model_copy():
    # 1.8B: sft = 14.4 GB (fits 15.3); dpo = 14.4 + 2N (3.6 GB) = 18 GB (does
    # not) -> the reference model is what flips the decision.
    sft = _plan({}, method="sft_text", n=1_800_000_000)
    dpo = _plan({}, method="dpo", n=1_800_000_000)
    assert sft["zero_stage"] == 0
    assert dpo["zero_stage"] == 3
    assert dpo["estimate"]["model_states_bytes"] == 18 * GB
    assert dpo["estimate"]["reference_model_bytes"] == int(3.6 * GB)
    # Under ZeRO-3 the trainable states shard (14.4 / 4 = 3.6 GB) but TRL wraps
    # the reference as ONE FSDP unit, gathered whole on every rank: + 3.6 GB.
    assert dpo["estimate"]["per_gpu_bytes"] == int(14.4 * GB) // 4 + int(3.6 * GB)
    assert "gathered whole" in "\n".join(dpo["trace"])
    # LoRA DPO keeps a frozen adapter copy, not a second base: 2N only.
    dpo_lora = _plan({"lora_rank": 16}, method="dpo", n=1_800_000_000)
    assert dpo_lora["estimate"]["model_states_bytes"] == int(3.6 * GB)
    assert dpo_lora["zero_stage"] == 0


def test_plan_lora_and_qlora_count_the_frozen_base():
    # 4B LoRA: 2N = 8 GB fits -> DP (the base is frozen bf16); QLoRA ~0.6N.
    lora = _plan({"lora_rank": 16}, n=4_000_000_000)
    qlora = _plan({"lora_rank": 16, "load_in_4bit": True}, n=4_000_000_000)
    assert lora["zero_stage"] == 0 and lora["estimate"]["model_states_bytes"] == 8 * GB
    assert qlora["estimate"]["model_states_bytes"] == int(0.6 * 4_000_000_000)
    assert lora["lora"] is True and qlora["qlora"] is True
    # bf16_mixed keeps the frozen base in bf16 (adapters go fp32); fp32 is a 4N base
    mixed = _plan({"lora_rank": 16, "precision": "bf16_mixed"}, n=4_000_000_000)
    assert mixed["estimate"]["model_states_bytes"] == 8 * GB
    fp32 = _plan({"lora_rank": 16, "precision": "fp32"}, n=4_000_000_000)
    assert fp32["estimate"]["model_states_bytes"] == 16 * GB


def test_plan_precision_bf16_mixed_and_fp32_use_16n():
    # 0.6B: bf16 8N = 4.8 GB; bf16_mixed / fp32 16N = 9.6 GB (fp32 master + Adam).
    for precision in ("bf16_mixed", "fp32"):
        plan = _plan({"precision": precision}, n=600_000_000)
        assert plan["precision"] == precision
        assert plan["estimate"]["model_states_bytes"] == 16 * 600_000_000
    assert _plan({}, n=600_000_000)["precision"] == "bf16"


def test_plan_explicit_stage_is_respected_and_offload_forces_sharding():
    # explicit 2 on a model that fits: respected (never auto-picked, but honoured).
    two = _plan({"zero_stage": 2}, n=600_000_000)
    assert two["zero_stage"] == 2 and two["backend"] == "fsdp2"
    assert two["zero_stage_requested"] == 2
    # stage-2 per-GPU: full bf16 weights replicated + (grads+optim)/dp
    n = 600_000_000
    assert two["estimate"]["per_gpu_bytes"] == 2 * n + (8 * n - 2 * n) // 4
    # explicit 0 on a model that does not fit: respected, with the OOM warning.
    zero = _plan({"zero_stage": 0}, n=4_000_000_000)
    assert zero["zero_stage"] == 0 and zero["warnings"]
    # cpu_offload with "auto" resolves to 3 (offload is an FSDP feature) and says so.
    # the states then live in host RAM, so the GPU estimate is the reference only
    off = _plan({"cpu_offload": True}, n=10_000_000_000)
    assert off["zero_stage"] == 3 and off["cpu_offload"] is True
    assert any("cpu_offload" in line for line in off["trace"])
    assert off["estimate"]["per_gpu_bytes"] == 0 and off["warnings"] == []
    with pytest.raises(ValueError):
        _plan({"cpu_offload": True, "zero_stage": 0})


def test_plan_warns_when_w_full_copies_do_not_fit_host_ram():
    # Every rank loads the whole model before sharding: 4 x 2N must fit host RAM.
    n = 14_000_000_000  # 28 GB bf16 per rank -> 112 GB
    kw = dict(method="sft_text", gpu_count=4, gpu_memory_bytes=L4, model_meta=_meta(n))
    ok = resolve_parallel_plan({}, host_memory_bytes=192 * GB, **kw)
    assert not any("host RAM" in w for w in ok["warnings"])
    tight = resolve_parallel_plan({}, host_memory_bytes=128 * GB, **kw)
    assert any("host RAM" in w and "112.0 GB" in w for w in tight["warnings"])


def test_plan_checkpoint_bytes_and_ddp_timeout():
    # Q-D: a resumable checkpoint is 6N (bf16) / 12N (mixed, fp32), plus one more
    # weights copy under FSDP (pytorch_model_fsdp.bin); the group timeout is
    # max(600 s, 2 x bytes / 100 MB/s).
    n = 4_000_000_000
    ddp = _plan({"zero_stage": 0}, n=n)
    assert ddp["estimate"]["checkpoint_bytes"] == 6 * n and ddp["ddp_timeout_s"] == 600
    fsdp = _plan({"zero_stage": 3}, n=n)
    assert fsdp["estimate"]["checkpoint_bytes"] == 6 * n + 2 * n
    assert fsdp["ddp_timeout_s"] == int(2 * 8 * n / 1e8)  # 640 s
    mixed = _plan({"zero_stage": 3, "precision": "bf16_mixed"}, n=n)
    assert mixed["estimate"]["checkpoint_bytes"] == 12 * n + 4 * n
    assert mixed["ddp_timeout_s"] == 1280
    assert _plan({"lora_rank": 8}, n=n)["estimate"]["checkpoint_bytes"] == 0


def test_plan_batch_arithmetic_and_alias():
    # gbs = mbs x ga x dp; ga derived from gbs when absent.
    p = _plan({"global_batch_size": 32, "micro_batch_size": 2})
    assert (p["micro_batch_size"], p["gradient_accumulation_steps"], p["global_batch_size"]) == (
        2,
        4,
        32,
    )
    # explicit ga wins when consistent
    p = _plan({"micro_batch_size": 2, "gradient_accumulation_steps": 3})
    assert p["global_batch_size"] == 24
    # legacy alias
    assert _plan({"batch_size": 8})["micro_batch_size"] == 8
    # defaults: mbs 1, ga 1 -> gbs = dp
    d = _plan({})
    assert (d["micro_batch_size"], d["gradient_accumulation_steps"], d["global_batch_size"]) == (
        1,
        1,
        4,
    )
    assert d["gradient_checkpointing"] is True
    assert _plan({"gradient_checkpointing": False})["gradient_checkpointing"] is False


def test_plan_without_a_param_count_still_resolves_with_a_warning():
    # Unknown N (no config, no weights): DDP by default, warn that fit was not checked.
    plan = resolve_parallel_plan(
        {},
        method="sft_text",
        gpu_count=4,
        gpu_memory_bytes=L4,
        model_meta={"param_count": None, "param_count_source": "unknown"},
    )
    assert plan["zero_stage"] == 0
    assert any("parameter count" in w for w in plan["warnings"])


# ---- render_accelerate_config ----


def _cfg(hp: dict, **kw) -> dict:
    return json.loads(render_accelerate_config(_plan(hp, **kw)))


def test_render_config_ddp_vs_fsdp2_and_reshard_per_stage():
    ddp = _cfg({"zero_stage": 0})
    assert ddp["distributed_type"] == "MULTI_GPU" and "fsdp_config" not in ddp
    assert ddp["num_processes"] == 4 and ddp["num_machines"] == 1 and ddp["machine_rank"] == 0
    two = _cfg({"zero_stage": 2})
    three = _cfg({"zero_stage": 3})
    for c in (two, three):
        assert c["distributed_type"] == "FSDP"
        f = c["fsdp_config"]
        assert f["fsdp_version"] == 2
        assert f["fsdp_auto_wrap_policy"] == "TRANSFORMER_BASED_WRAP"
        assert f["fsdp_state_dict_type"] == "FULL_STATE_DICT"
        # OFF: with it on, a DPO reference model built after the process group
        # exists loads NO weights on ranks 1..W-1 (transformers skips the load,
        # TRL's prepare_fsdp never broadcasts) -> silently wrong DPO.
        assert f["fsdp_cpu_ram_efficient_loading"] is False
    # ZeRO-2 keeps params gathered after forward; ZeRO-3 reshards.
    assert two["fsdp_config"]["fsdp_reshard_after_forward"] is False
    assert three["fsdp_config"]["fsdp_reshard_after_forward"] is True


def test_render_config_precision_mapping():
    # pure bf16 under FSDP2 must NOT request accelerate mixed precision (it
    # would upcast params to fp32 master weights); bf16_mixed must.
    assert _cfg({"zero_stage": 3, "precision": "bf16"})["mixed_precision"] == "no"
    assert _cfg({"zero_stage": 3, "precision": "bf16_mixed"})["mixed_precision"] == "bf16"
    assert _cfg({"zero_stage": 3, "precision": "fp32"})["mixed_precision"] == "no"
    assert _cfg({"zero_stage": 0, "precision": "bf16_mixed"})["mixed_precision"] == "bf16"


def test_render_config_activation_checkpointing_and_offload_flags():
    on = _cfg({"zero_stage": 3})["fsdp_config"]
    off = _cfg({"zero_stage": 3, "gradient_checkpointing": False})["fsdp_config"]
    assert on["fsdp_activation_checkpointing"] is True
    assert off["fsdp_activation_checkpointing"] is False
    assert _cfg({"zero_stage": 3})["fsdp_config"]["fsdp_offload_params"] is False
    assert (
        _cfg({"zero_stage": 3, "cpu_offload": True})["fsdp_config"]["fsdp_offload_params"] is True
    )


# ---- build_launch_argv / child config / child env ----


def test_build_launch_argv_shape():
    plan = _plan({})
    argv = build_launch_argv(
        plan,
        config_path="/tmp/j/accelerate.json",
        child_config_path="/tmp/j/child.json",
        log_dir="/tmp/j/ranks",
        script_path="/opt/veri/training_runtime.py",
        python="/opt/veri/venv/bin/python",
    )
    print(" ".join(argv))
    assert argv[:3] == ["/opt/veri/venv/bin/python", "-m", "accelerate.commands.launch"]
    launcher = argv[: argv.index("/opt/veri/training_runtime.py")]
    child = argv[argv.index("/opt/veri/training_runtime.py") + 1 :]
    for flag, value in [
        ("--config_file", "/tmp/j/accelerate.json"),
        ("--num_processes", "4"),
        ("--num_machines", "1"),
        ("--machine_rank", "0"),
        ("--max_restarts", "0"),
        ("--monitor_interval", "5"),
        ("--tee", "3"),
        ("--log_dir", "/tmp/j/ranks"),
    ]:
        assert launcher[launcher.index(flag) + 1] == value, flag
    assert child == ["--child-config", "/tmp/j/child.json"]


def test_render_config_is_what_accelerate_launch_turns_into_the_rank_env(tmp_path):
    # Through accelerate 1.15 itself: the JSON loads as a ClusterConfig and the
    # launcher derives the FSDP rank env from it (what the child's Trainer reads).
    accelerate = pytest.importorskip("accelerate")
    from accelerate.commands.config.config_args import load_config_from_file
    from accelerate.commands.launch import _validate_launch_command, launch_command_parser
    from accelerate.utils.launch import prepare_multi_gpu_env

    plan = _plan({"zero_stage": 3, "precision": "bf16_mixed", "cpu_offload": True})
    path = tmp_path / "accelerate.json"
    path.write_text(render_accelerate_config(plan))
    cfg = load_config_from_file(str(path))
    assert cfg.distributed_type == "FSDP" and cfg.num_processes == 4
    argv = ["--config_file", str(path), "x.py"]
    args = launch_command_parser().parse_args(argv)
    # the real launcher path: the config file fills every unset arg, then the
    # multi-GPU env builder turns them into the rank environment
    args, defaults, _ = _validate_launch_command(args)
    assert args.use_fsdp and args.num_processes == 4
    env = prepare_multi_gpu_env(args)
    assert env["ACCELERATE_USE_FSDP"] == "true"
    assert env["FSDP_VERSION"] == "2"
    assert env["FSDP_ACTIVATION_CHECKPOINTING"] == "true"
    assert env["FSDP_RESHARD_AFTER_FORWARD"] == "true"
    assert env["FSDP_OFFLOAD_PARAMS"] == "true"
    assert env["FSDP_CPU_RAM_EFFICIENT_LOADING"] == "false"
    assert env["FSDP_STATE_DICT_TYPE"] == "FULL_STATE_DICT"
    assert env["ACCELERATE_MIXED_PRECISION"] == "bf16"
    assert accelerate.__version__ == "1.15.0"


def test_child_config_strips_every_secret_and_callback():
    job = {
        "job_id": "j1",
        "method": "sft_text",
        "base_model": "Qwen/Qwen3-0.6B",
        "hyperparameters": {"micro_batch_size": 2},
        "checkpoint": {"enabled": False},
        "output_name": "out",
        "wandb_project": "p",
        "gpu_count": 4,
        # secrets / channels the child must never see
        "callback_token": "tok",
        "hf_token": "hf_x",
        "wandb_api_key": "wb_x",
        "status_callback_url": "https://api/status",
        "checkpoint_callback_url": "https://api/ckpt",
        "log_callback_url": "https://api/logs",
        "checkpoint_step_base_url": "https://api/steps",
        "dataset_download_url": "https://s3/d?sig",
        "reward_download_url": "https://s3/r?sig",
        "dataset_snapshot_upload_url": "https://s3/snap?sig",
        "hf_push": {"repo": "me/out"},
        "volume": {"sts_credentials": {"AccessKeyId": "AKIA"}},
        # nested secrets a deny-list on top-level names would miss
        "dataset": {"source_type": "hf", "hf_dataset": "me/p", "hf_config": {"token": "hf_nested"}},
        "checkpoint_destination": {"kind": "s3", "credentials": {"secret_access_key": "S3SECRET"}},
        "some_future_field": {"api_key": "FUTURE"},
    }
    plan = _plan({"micro_batch_size": 2})
    child = build_child_config(
        job,
        plan,
        dataset_path="/tmp/j/rows.jsonl",
        ranks_dir="/tmp/j/ranks",
        progress_path="/tmp/j/progress.jsonl",
    )
    blob = json.dumps(child)
    secrets = (
        "tok",
        "hf_x",
        "wb_x",
        "sig",
        "AKIA",
        "api/status",
        "me/out",
        "hf_nested",
        "S3SECRET",
        "FUTURE",
    )
    for leaked in secrets:
        assert leaked not in blob, leaked
    jc = child["job_config"]
    for kept in (
        "job_id",
        "method",
        "base_model",
        "hyperparameters",
        "checkpoint",
        "output_name",
        "wandb_project",
        "gpu_count",
    ):
        assert kept in jc, kept
    assert child["plan"] == plan
    assert child["dataset_path"] == "/tmp/j/rows.jsonl"
    assert (
        child["ranks_dir"] == "/tmp/j/ranks" and child["progress_path"] == "/tmp/j/progress.jsonl"
    )
    # the parent's dict is untouched
    assert job["callback_token"] == "tok"


def test_child_env_sets_the_rank_contract_and_keeps_the_wandb_key_out_of_the_file():
    job = {"job_id": "j1", "hyperparameters": {}, "wandb_api_key": "wb_x"}
    plan = _plan({})
    env = build_child_env(
        job, plan, ranks_dir="/tmp/j/ranks", cpu_count=48, base_env={"PATH": "/bin"}
    )
    assert env["PATH"] == "/bin"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["OMP_NUM_THREADS"] == "12"  # 48 vCPUs / 4 ranks
    assert env["NCCL_DEBUG"] == "WARN" and "NCCL_DEBUG_FILE" not in env
    assert env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "120"
    # flight recorder + desync report (off by default on torch 2.9.1)
    assert env["TORCH_FR_BUFFER_SIZE"] == "2000"
    assert env["TORCH_NCCL_DESYNC_DEBUG"] == "1"
    assert env["TORCH_FR_DUMP_TEMP_FILE"] == "/tmp/j/ranks/fr_"
    # W&B key travels in the env only
    assert env["WANDB_API_KEY"] == "wb_x"
    # not defaulted: it moves customers' OOM boundaries (10.2 N6)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    # never downgrade torch's default async error handling (10.2 N3)
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in env

    no_key = build_child_env(
        {"job_id": "j1", "hyperparameters": {}}, plan, ranks_dir="/r", cpu_count=4, base_env={}
    )
    assert "WANDB_API_KEY" not in no_key
    assert no_key["OMP_NUM_THREADS"] == "1"
    dbg = build_child_env(
        {"job_id": "j1", "hyperparameters": {"debug_nccl": True}},
        plan,
        ranks_dir="/r",
        cpu_count=4,
        base_env={},
    )
    assert dbg["NCCL_DEBUG"] == "INFO" and dbg["NCCL_DEBUG_FILE"] == "/r/nccl.%h.%p.log"


# ---- config builders wired to the plan ----


class _FakeSFTConfig:
    def __init__(
        self,
        packing=None,
        dataset_text_field=None,
        max_length=None,
        gradient_accumulation_steps=None,
    ):
        pass


class _FakeDPOConfig:
    def __init__(self, beta=None, loss_type=None, max_length=None, model_init_kwargs=None):
        pass


def test_sft_builder_without_knobs_is_byte_for_byte_todays_config():
    # Regression guard: no new keys, no plan -> no gradient_accumulation_steps
    # key at all, gradient_checkpointing True, per-device batch 1.
    kw = build_sft_config_kwargs(job_id="j", hyperparameters={}, sft_config_cls=_FakeSFTConfig)
    assert kw["per_device_train_batch_size"] == 1
    assert kw["gradient_checkpointing"] is True
    assert "gradient_accumulation_steps" not in kw
    assert "model_init_kwargs" not in kw


def test_builders_honour_knobs_on_a_single_gpu_without_a_plan():
    # x1 jobs stay in-process; explicit knobs are applied, never silently ignored.
    hp = {"micro_batch_size": 4, "gradient_accumulation_steps": 2, "gradient_checkpointing": False}
    sft = build_sft_config_kwargs(job_id="j", hyperparameters=hp, sft_config_cls=_FakeSFTConfig)
    assert sft["per_device_train_batch_size"] == 4
    assert sft["gradient_accumulation_steps"] == 2
    assert sft["gradient_checkpointing"] is False
    dpo = build_dpo_config_kwargs(
        job_id="j", hyperparameters={"batch_size": 3}, dpo_config_cls=_FakeDPOConfig
    )
    assert dpo["per_device_train_batch_size"] == 3
    assert "ddp_timeout" not in sft and "ddp_find_unused_parameters" not in sft
    # global_batch_size alone derives the accumulation on one GPU (dp = 1)
    gbs = build_sft_config_kwargs(
        job_id="j",
        hyperparameters={"micro_batch_size": 2, "global_batch_size": 16},
        sft_config_cls=_FakeSFTConfig,
    )
    assert gbs["gradient_accumulation_steps"] == 8


def test_builders_take_batch_and_checkpointing_from_the_launch_plan(monkeypatch):
    monkeypatch.setattr(rt, "_gpu_supports_bf16", lambda: True)
    hp_ddp = {"micro_batch_size": 2, "global_batch_size": 16}  # ga 2 on dp 4
    hp_fsdp = {"micro_batch_size": 2, "zero_stage": 3}
    ddp, fsdp = _plan(hp_ddp), _plan(hp_fsdp)
    sft_ddp = build_sft_config_kwargs(
        job_id="j", hyperparameters=hp_ddp, sft_config_cls=_FakeSFTConfig, launch_plan=ddp
    )
    sft_fsdp = build_sft_config_kwargs(
        job_id="j", hyperparameters=hp_fsdp, sft_config_cls=_FakeSFTConfig, launch_plan=fsdp
    )
    assert (
        sft_ddp["per_device_train_batch_size"] == 2 and sft_ddp["gradient_accumulation_steps"] == 2
    )
    # DDP: Trainer-level checkpointing. FSDP2: off here, on in the accelerate
    # config (fsdp_activation_checkpointing) to avoid the redundant all-gather.
    assert sft_ddp["gradient_checkpointing"] is True
    assert sft_fsdp["gradient_checkpointing"] is False
    # pure bf16 under FSDP2 must not set bf16=True (accelerate would upcast to fp32)
    assert sft_ddp["bf16"] is True
    assert sft_fsdp["bf16"] is False
    mixed = _plan({"zero_stage": 3, "precision": "bf16_mixed"})
    assert (
        build_sft_config_kwargs(
            job_id="j", hyperparameters={}, sft_config_cls=_FakeSFTConfig, launch_plan=mixed
        )["bf16"]
        is True
    )
    # launched jobs carry the derived group timeout; DDP skips the unused-param scan
    assert sft_ddp["ddp_timeout"] == 600 and sft_ddp["ddp_find_unused_parameters"] is False
    assert sft_fsdp["ddp_timeout"] == 600 and "ddp_find_unused_parameters" not in sft_fsdp
    fp32 = _plan({"zero_stage": 0, "precision": "fp32"})
    assert (
        build_sft_config_kwargs(
            job_id="j", hyperparameters={}, sft_config_cls=_FakeSFTConfig, launch_plan=fp32
        )["bf16"]
        is False
    )


def test_dpo_builder_sets_reference_model_init_kwargs_only_for_launched_full_ft():
    full = _plan({"zero_stage": 3}, method="dpo", n=2_000_000_000)
    kw = build_dpo_config_kwargs(
        job_id="j", hyperparameters={}, dpo_config_cls=_FakeDPOConfig, launch_plan=full
    )
    # TRL only nulls device_map for MULTI_GPU/DEEPSPEED, not FSDP; and its
    # default reference dtype is fp32 (2x the memory it needs).
    assert kw["model_init_kwargs"] == {"dtype": "bfloat16", "device_map": None}
    lora = _plan({"lora_rank": 8}, method="dpo", n=2_000_000_000)
    assert "model_init_kwargs" not in build_dpo_config_kwargs(
        job_id="j",
        hyperparameters={"lora_rank": 8},
        dpo_config_cls=_FakeDPOConfig,
        launch_plan=lora,
    )
    assert "model_init_kwargs" not in build_dpo_config_kwargs(
        job_id="j", hyperparameters={}, dpo_config_cls=_FakeDPOConfig
    )
    mixed_ddp = _plan({"zero_stage": 0, "precision": "bf16_mixed"}, method="dpo", n=100_000_000)
    assert (
        build_dpo_config_kwargs(
            job_id="j", hyperparameters={}, dpo_config_cls=_FakeDPOConfig, launch_plan=mixed_ddp
        )["model_init_kwargs"]["dtype"]
        == "bfloat16"
    )


# ---- HF push from the parent: merge the saved adapter from disk ----


def test_push_merged_without_a_trainer_merges_the_adapter_from_disk(tmp_path, monkeypatch):
    # GOOD: after a multi-GPU run the parent (no trainer object) pushes a merged
    # artifact by loading base + adapter from disk; the upload is the merged dir.
    (tmp_path / "final").mkdir()
    (tmp_path / "final" / "adapter_config.json").write_text("{}")
    calls: dict = {}

    class _Merged:
        def save_pretrained(self, d):
            calls["merged_saved"] = d

    class _Peft:
        def __init__(self):
            pass

        def merge_and_unload(self):
            calls["merged"] = True
            return _Merged()

    peft = types.ModuleType("peft")
    peft.PeftModel = type(
        "PeftModel",
        (),
        {
            "from_pretrained": staticmethod(
                lambda base, d: calls.update(adapter_dir=d, base=base) or _Peft()
            )
        },
    )
    tf = types.ModuleType("transformers")
    tf.AutoModelForCausalLM = type(
        "A",
        (),
        {
            "from_pretrained": staticmethod(
                lambda name, **kw: calls.update(base_model=name, base_kwargs=kw) or "base"
            )
        },
    )
    tf.AutoTokenizer = type(
        "T",
        (),
        {
            "from_pretrained": staticmethod(
                lambda d: types.SimpleNamespace(
                    save_pretrained=lambda out: calls.update(tok_saved=out)
                )
            )
        },
    )
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    for name, mod in (("peft", peft), ("transformers", tf), ("torch", torch)):
        monkeypatch.setitem(sys.modules, name, mod)
    hub: dict = {}

    class _Api:
        def __init__(self, token=None):
            hub["token"] = token

        def whoami(self):
            return {}

        def create_repo(self, **kw):
            hub["create"] = kw

        def upload_folder(self, **kw):
            hub["upload"] = kw

    hh = types.ModuleType("huggingface_hub")
    hh.HfApi = _Api
    monkeypatch.setitem(sys.modules, "huggingface_hub", hh)

    from veri_runner.training_runtime import push_to_hf_hub

    url = push_to_hf_hub(
        job_config={
            "job_id": "j9",
            "base_model": "Qwen/Qwen3-4B",
            "hf_push": {"repo": "me/out"},
            "hf_token": "hf_x",
        },
        final_dir=str(tmp_path / "final"),
    )
    assert url == "https://huggingface.co/me/out"
    assert calls["base_model"] == "Qwen/Qwen3-4B" and calls["adapter_dir"] == str(
        tmp_path / "final"
    )
    assert calls["merged_saved"] == str(tmp_path / "merged") and calls["tok_saved"] == str(
        tmp_path / "merged"
    )
    assert hub["upload"]["folder_path"] == str(tmp_path / "merged")


# ---- model load: device_map / dtype per backend ----


def _install_load_mocks(monkeypatch, *, cuda=True):
    calls = []
    fake_tok = type(
        "FakeTok",
        (),
        {
            "from_pretrained": staticmethod(
                lambda *a, **kw: type("T", (), {"pad_token": None, "eos_token": "<eos>"})()
            ),
        },
    )

    def _from_pretrained(name, **kw):
        calls.append(kw)
        return "model"

    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = fake_tok
    tf.AutoModelForCausalLM = type("A", (), {"from_pretrained": staticmethod(_from_pretrained)})
    monkeypatch.setitem(sys.modules, "transformers", tf)
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float32 = "bf16", "fp32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    monkeypatch.setitem(sys.modules, "torch", torch)
    peft = types.ModuleType("peft")
    peft.LoraConfig = lambda **kw: ("lora", kw)
    peft.get_peft_model = lambda model, config: model
    peft.prepare_model_for_kbit_training = lambda model: model
    monkeypatch.setitem(sys.modules, "peft", peft)
    return calls


def test_load_model_device_map_and_dtype_per_backend(monkeypatch):
    calls = _install_load_mocks(monkeypatch)
    log = logging.getLogger("t")
    # no plan (x1): unchanged -> bf16 + device_map auto
    _load_model_and_tokenizer("m", {}, log)
    assert calls[-1]["torch_dtype"] == "bf16" and calls[-1]["device_map"] == "auto"
    # DDP: one full copy per rank on its own device
    monkeypatch.setenv("LOCAL_RANK", "2")
    _load_model_and_tokenizer("m", {}, log, launch_plan=_plan({"zero_stage": 0}))
    assert calls[-1]["device_map"] == {"": 2} and calls[-1]["torch_dtype"] == "bf16"
    # DDP bf16_mixed: fp32 load + autocast
    _load_model_and_tokenizer(
        "m", {}, log, launch_plan=_plan({"zero_stage": 0, "precision": "bf16_mixed"})
    )
    assert calls[-1]["torch_dtype"] == "fp32"
    # ... but with LoRA the frozen base stays bf16 (adapters go fp32 via PEFT)
    lora_plan = _plan({"lora_rank": 8, "zero_stage": 0, "precision": "bf16_mixed"})
    _load_model_and_tokenizer("m", {"lora_rank": 8}, log, launch_plan=lora_plan)
    assert calls[-1]["torch_dtype"] == "bf16"
    # FSDP2: no device_map (rank 0 reads, others start empty); bf16 load even for
    # bf16_mixed (accelerate upcasts the sharded master copy to fp32)
    _load_model_and_tokenizer("m", {}, log, launch_plan=_plan({"zero_stage": 3}))
    assert calls[-1]["device_map"] is None and calls[-1]["torch_dtype"] == "bf16"
    _load_model_and_tokenizer(
        "m", {}, log, launch_plan=_plan({"zero_stage": 3, "precision": "bf16_mixed"})
    )
    assert calls[-1]["torch_dtype"] == "bf16"
    _load_model_and_tokenizer(
        "m", {}, log, launch_plan=_plan({"zero_stage": 3, "precision": "fp32"})
    )
    assert calls[-1]["torch_dtype"] == "fp32"
    # x1 with an explicit precision knob (no plan) is honoured too
    _load_model_and_tokenizer("m", {"precision": "fp32"}, log)
    assert calls[-1]["torch_dtype"] == "fp32" and calls[-1]["device_map"] == "auto"


def test_load_model_ddp_without_cuda_uses_no_device_map(monkeypatch):
    # the CPU integration path (gloo ranks) has no CUDA devices to pin to
    calls = _install_load_mocks(monkeypatch, cuda=False)
    _load_model_and_tokenizer("m", {}, logging.getLogger("t"), launch_plan=_plan({"zero_stage": 0}))
    assert calls[-1]["device_map"] is None


# ---- rank-0 progress + phases; fp32 grad reduce ----


class _State:
    def __init__(self, zero, step=3, max_steps=10):
        self.is_world_process_zero = zero
        self.global_step = step
        self.max_steps = max_steps


class _FakeTrainer:
    def __init__(self):
        self.callbacks = []

    def add_callback(self, cb):
        self.callbacks.append(cb)


def test_progress_callback_is_gated_on_rank_zero_and_reports_phases(monkeypatch):
    tf = types.ModuleType("transformers")
    tf.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "transformers", tf)
    steps, phases = [], []
    tr = _FakeTrainer()
    _attach_progress_callback(
        tr, lambda s, t, m: steps.append((s, t, m)), phase_fn=lambda p, s: phases.append((p, s))
    )
    cb = tr.callbacks[0]
    ctl = types.SimpleNamespace(should_save=True)
    # a non-zero rank writes nothing
    cb.on_log(None, _State(False), ctl, logs={"loss": 1.5})
    cb.on_step_end(None, _State(False), ctl)
    assert steps == [] and phases == []
    # rank 0 writes the step and the saving phase around a save
    cb.on_log(None, _State(True), ctl, logs={"loss": 1.5, "epoch": 0.1, "text": "x"})
    cb.on_step_end(None, _State(True), ctl)
    cb.on_save(None, _State(True), ctl)
    assert steps == [(3, 10, {"loss": 1.5, "epoch": 0.1})]
    assert phases == [("saving", 3), ("training", 3)]
    # no save pending -> no phase line
    cb.on_step_end(None, _State(True), types.SimpleNamespace(should_save=False))
    assert len(phases) == 2
    # x1 / no phase_fn: today's signature still works
    tr2 = _FakeTrainer()
    _attach_progress_callback(tr2, lambda s, t, m: steps.append((s, t, m)))
    tr2.callbacks[0].on_step_end(None, _State(True), ctl)


def test_distributed_finalize_saves_on_every_rank_writes_tokenizer_once_and_never_pushes(
    monkeypatch,
):
    from veri_runner.training_runtime import _train_save_finalize

    pushed = []
    monkeypatch.setattr(rt, "push_to_hf_hub", lambda **kw: pushed.append(kw) or "https://x")
    monkeypatch.setattr(rt, "_gather_peak_memory", lambda: [1, 2])
    monkeypatch.setattr(rt, "_force_fp32_grad_reduce", lambda trainer, plan: False)
    plan = _plan({"zero_stage": 3}, gpus=2)
    job = {
        "job_id": "j",
        "checkpoint": {"local_output_root": "/tmp/ck"},
        "hf_push": {"repo": "me/out"},
        "hf_token": "t",
    }

    def run(rank):
        monkeypatch.setenv("RANK", str(rank))
        saved, tok, phases = [], [], []
        trainer = types.SimpleNamespace(
            train=lambda: types.SimpleNamespace(training_loss=0.5),
            save_model=lambda d: saved.append(d),
            state=types.SimpleNamespace(global_step=7),
        )
        tokenizer = types.SimpleNamespace(save_pretrained=lambda d: tok.append(d))
        result = _train_save_finalize(
            trainer=trainer,
            tokenizer=tokenizer,
            job_config=job,
            wandb_enabled=False,
            log=logging.getLogger("t"),
            launch_plan=plan,
            phase_fn=lambda p, s: phases.append((p, s)),
        )
        return saved, tok, phases, result

    saved0, tok0, phases0, result0 = run(0)
    saved1, tok1, phases1, result1 = run(1)
    assert saved0 == saved1 == ["/tmp/ck/j/final"]  # the save is a collective
    assert tok0 == ["/tmp/ck/j/final"] and tok1 == []  # rank 0 writes the tokenizer
    assert phases0 == [("saving", 7), ("finalizing", 7)] and phases1 == []
    assert pushed == [], "the child never pushes; the parent does"
    assert result0["world_size"] == 2 and result0["peak_memory_bytes_per_rank"] == [1, 2]
    assert "hf_repo_url" not in result0 and result1["final_loss"] == 0.5
    # no plan: unchanged single-process tail (push happens here)
    monkeypatch.setenv("RANK", "0")
    saved, tok = [], []
    trainer = types.SimpleNamespace(
        train=lambda: types.SimpleNamespace(training_loss=0.5),
        save_model=lambda d: saved.append(d),
        state=types.SimpleNamespace(global_step=1),
    )
    r = _train_save_finalize(
        trainer=trainer,
        tokenizer=types.SimpleNamespace(save_pretrained=lambda d: tok.append(d)),
        job_config=job,
        wandb_enabled=False,
        log=logging.getLogger("t"),
    )
    assert r["hf_repo_url"] == "https://x" and len(pushed) == 1


def test_stack_dump_registers_sigusr1_and_wandb_key_is_restored_from_env(tmp_path, monkeypatch):
    import faulthandler
    import signal

    from veri_runner.training_runtime import _install_stack_dump, _restore_wandb_key

    registered = []
    monkeypatch.setattr(
        faulthandler,
        "register",
        lambda sig, file=None, all_threads=True: registered.append((sig, file.name, all_threads)),
    )
    _install_stack_dump(str(tmp_path / "ranks"), 3)
    assert registered == [(signal.SIGUSR1, str(tmp_path / "ranks" / "stack_3.txt"), True)]
    job = {"job_id": "j"}
    _restore_wandb_key(job, {"WANDB_API_KEY": "wb"})
    assert job["wandb_api_key"] == "wb"
    _restore_wandb_key(job, {"WANDB_API_KEY": "other"})
    assert job["wandb_api_key"] == "wb"  # an explicit key is never overwritten
    j2 = {"job_id": "j"}
    _restore_wandb_key(j2, {})
    assert "wandb_api_key" not in j2


def test_cast_saved_model_dtype_rewrites_full_models_only(tmp_path, monkeypatch):
    from veri_runner.training_runtime import cast_saved_model_dtype

    calls = {}

    class _M:
        def __init__(self, dtype):
            self._dtype = dtype

        def parameters(self):
            return [types.SimpleNamespace(dtype=self._dtype)]

        def to(self, dtype):
            calls["to"] = dtype
            return self

        def save_pretrained(self, d):
            calls["saved"] = d

    tf = types.ModuleType("transformers")
    tf.AutoModelForCausalLM = type(
        "A", (), {"from_pretrained": staticmethod(lambda d, torch_dtype=None: _M("fp32"))}
    )
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "torch", torch)
    full = tmp_path / "final"
    full.mkdir()
    assert cast_saved_model_dtype(str(full), "bfloat16", logging.getLogger("t")) is True
    assert calls == {"to": "bf16", "saved": str(full)}
    (full / "adapter_config.json").write_text("{}")
    calls.clear()
    assert cast_saved_model_dtype(str(full), "bfloat16", logging.getLogger("t")) is False
    assert calls == {}


def test_force_fp32_grad_reduce_applies_only_to_fsdp2_bf16_mixed(monkeypatch):
    made = []

    def _policy(**kw):
        made.append(kw)
        return ("policy", kw)

    fsdp_mod = types.ModuleType("torch.distributed.fsdp")
    fsdp_mod.MixedPrecisionPolicy = _policy
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float32 = "bf16", "fp32"
    torch.distributed = types.ModuleType("torch.distributed")
    torch.distributed.fsdp = fsdp_mod
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", torch.distributed)
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fsdp_mod)

    plugin = types.SimpleNamespace(mixed_precision_policy=None)
    trainer = types.SimpleNamespace(
        accelerator=types.SimpleNamespace(state=types.SimpleNamespace(fsdp_plugin=plugin))
    )
    # GOOD: FSDP2 + bf16_mixed -> reduce in fp32, params/outputs in bf16
    assert (
        _force_fp32_grad_reduce(trainer, _plan({"zero_stage": 3, "precision": "bf16_mixed"}))
        is True
    )
    assert plugin.mixed_precision_policy == (
        "policy",
        {"param_dtype": "bf16", "reduce_dtype": "fp32", "output_dtype": "bf16"},
    )
    # BAD: pure bf16 (no master weights) and DDP are left alone
    plugin.mixed_precision_policy = None
    assert _force_fp32_grad_reduce(trainer, _plan({"zero_stage": 3})) is False
    assert (
        _force_fp32_grad_reduce(trainer, _plan({"zero_stage": 0, "precision": "bf16_mixed"}))
        is False
    )
    assert plugin.mixed_precision_policy is None
    assert len(made) == 1


# ---- CPU integration: the real child under torchrun with 2 gloo ranks ----


def _tiny_tokenizer(model_dir):
    """A word-level tokenizer built locally (no Hub download) with Qwen-style
    special tokens, saved as a fast tokenizer next to the tiny model."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    words += [str(i) for i in range(16)]
    vocab = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "[UNK]": 3}
    for w in words:
        vocab[w] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        unk_token="[UNK]",
    )
    fast.save_pretrained(model_dir)


def _ml_stack_available() -> bool:
    for name in ("torch", "transformers", "trl", "accelerate", "datasets"):
        try:
            __import__(name)
        except Exception:
            return False
    return True


@pytest.mark.skipif(not _ml_stack_available(), reason="needs torch/transformers/trl/accelerate")
def test_child_entrypoint_two_ranks_on_cpu(tmp_path):
    from transformers import AutoModelForCausalLM, Qwen3Config

    model_dir = tmp_path / "tiny-qwen3"
    cfg = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=151936,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    AutoModelForCausalLM.from_config(cfg).save_pretrained(model_dir)
    _tiny_tokenizer(model_dir)

    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        "".join(
            json.dumps({"text": f"the quick brown fox {i} jumps over the lazy dog"}) + "\n"
            for i in range(16)
        )
    )
    job = {
        "job_id": "cpu2",
        "method": "sft_text",
        "base_model": str(model_dir),
        "hyperparameters": {
            "max_steps": 2,
            "micro_batch_size": 2,
            "max_seq_length": 32,
            "gradient_checkpointing": False,
        },
        "checkpoint": {"enabled": False, "local_output_root": str(tmp_path / "ckpts")},
        "gpu_count": 2,
        "callback_token": "secret-tok",
    }
    plan = resolve_parallel_plan(
        job["hyperparameters"],
        method="sft_text",
        gpu_count=2,
        gpu_memory_bytes=L4,
        model_meta=read_model_meta(str(model_dir)),
    )
    assert plan["backend"] == "ddp"
    ranks = tmp_path / "ranks"
    child = build_child_config(
        job,
        plan,
        dataset_path=str(rows),
        ranks_dir=str(ranks),
        progress_path=str(tmp_path / "progress.jsonl"),
    )
    child_path = tmp_path / "child.json"
    child_path.write_text(json.dumps(child))

    def _run(extra_env):
        env = build_child_env(
            job, plan, ranks_dir=str(ranks), cpu_count=4, base_env=dict(os.environ)
        )
        # ACCELERATE_TORCH_DEVICE: on a Mac accelerate would otherwise pick MPS,
        # where c10d::barrier is not implemented; production ranks are CUDA.
        env.update(
            {
                "ACCELERATE_USE_CPU": "true",
                "ACCELERATE_TORCH_DEVICE": "cpu",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                "HF_HUB_OFFLINE": "1",
                **extra_env,
            }
        )
        argv = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            "2",
            "--max_restarts",
            "0",
            "--log_dir",
            str(ranks),
            "--rdzv-id",
            "cpu2",
            rt.__file__,
            "--child-config",
            str(child_path),
        ]
        return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=600)

    ok = _run({})
    print(ok.stdout[-2000:], ok.stderr[-3000:])
    assert ok.returncode == 0
    lines = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert lines and all(line["rank"] == 0 for line in lines)
    assert any("step" in line and "loss" in line.get("metrics", {}) for line in lines)
    result = json.loads((tmp_path / "result.json").read_text())
    assert "final_loss" in result and result["world_size"] == 2
    assert len(result["peak_memory_bytes_per_rank"]) == 2
    final = Path(result["checkpoint_dir"])
    assert (final / "config.json").exists() and (final / "tokenizer_config.json").exists()

    # rank 1 blows up before training: non-zero exit and torchelastic's error
    # file names rank 1 with the injected message
    bad = _run({"VERI_CHILD_FAIL_RANK": "1"})
    assert bad.returncode != 0
    errs = sorted(ranks.rglob("error.json"))
    assert errs, "torchelastic must write a per-rank error file"
    bodies = [json.loads(p.read_text()) for p in errs]
    assert any("injected failure on rank 1" in json.dumps(b) for b in bodies)
    assert all(p.parent.name == "1" for p in errs)
