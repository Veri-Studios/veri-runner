"""Shared training runtime for runner adapters.

Supports multiple training methods:
- GRPO: TRL GRPOTrainer (text RL)
- SFT Video Gen: delegated to sft_video_gen_runtime (diffusers, separate deps)
"""

from __future__ import annotations

# unsloth is OPT-IN, never load-bearing. It used to be imported here at
# module load purely to re-inject `GuidedDecodingParams` into vllm.sampling_params
# (a symbol vllm 0.12+ removed, trl 0.22.2 still imports). But that import also
# swapped trl.GRPOTrainer for unsloth's version, which only works with
# unsloth-loaded models (the `for_training` crash on the vanilla default path).
# The framework is now resolved per-job in run_grpo_training: use_unsloth=True
# imports unsloth (patches the trainer) BEFORE importing trl; the vanilla base path
# uses _ensure_vanilla_trl_imports() (a minimal vLLM-symbol shim) instead. Base
# stays vanilla TRL + transformers; users own optimization-framework deps.
import importlib.util
import inspect
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def _gpu_supports_bf16() -> bool:
    """Check if the current GPU supports bfloat16."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.is_bf16_supported()
    except Exception:
        pass
    return False


MILES_REWARD_UNSUPPORTED = (
    "Miles reward format is not supported on NVIDIA GPUs. "
    "Upload a TRL-format reward function instead: "
    "def reward(completions, answer, **kwargs) -> list[float]"
)


def load_reward_function(
    reward_path: str,
    *,
    reward_format: str = "trl",
    unsupported_formats: Mapping[str, str] | None = None,
    logger: logging.Logger | None = None,
    _module_name: str = "user_reward",
) -> Callable[..., Any]:
    """Import the uploaded reward module and return its reward callable.

    `_module_name` lets callers load several reward files under distinct module
    names so same-named modules don't clobber each other (see
    load_reward_functions for the multi-reward case).
    """
    unsupported_formats = unsupported_formats or {}
    if reward_format in unsupported_formats:
        raise ValueError(unsupported_formats[reward_format])

    spec = importlib.util.spec_from_file_location(_module_name, reward_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import reward function from {reward_path}")

    user_reward_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_reward_mod)

    reward_fn = getattr(user_reward_mod, "reward", None)
    if callable(reward_fn):
        return reward_fn

    candidates = [
        getattr(user_reward_mod, name)
        for name in dir(user_reward_mod)
        if callable(getattr(user_reward_mod, name)) and not name.startswith("_")
    ]
    if not candidates:
        raise ValueError("No callable reward function found in uploaded file")

    reward_fn = candidates[0]
    if logger:
        logger.info("No 'reward' function found, using '%s'", reward_fn.__name__)
    return reward_fn


def load_reward_functions(
    reward_paths: list[str],
    *,
    reward_format: str = "trl",
    unsupported_formats: Mapping[str, str] | None = None,
    logger: logging.Logger | None = None,
) -> list[Callable[..., Any]]:
    """Load multiple uploaded reward modules into a list of callables.

    Each file loads under a distinct module name. Used for multi-reward GRPO,
    where TRL's GRPOTrainer takes a `reward_funcs` list (+ optional
    `reward_weights`); see the TRL multi-task reward example.
    """
    return [
        load_reward_function(
            path,
            reward_format=reward_format,
            unsupported_formats=unsupported_formats,
            logger=logger,
            _module_name=f"user_reward_{i}",
        )
        for i, path in enumerate(reward_paths)
    ]


def load_jsonl_rows(dataset_path: str) -> list[dict[str, Any]]:
    """Load JSONL training rows from a local file."""
    with open(dataset_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_training_rows(
    *,
    dataset_path: str | None = None,
    dataset_config: dict[str, Any] | None = None,
    snapshot_upload_url: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve training rows from a local artifact or configured dataset source.

    Every path returns rows normalized to the TRL-native shape. The
    HF path additionally caches the normalized JSONL back to S3 via the
    presigned `snapshot_upload_url` when the control plane provides one, so
    later jobs on the same dataset skip the Hub download and this rewrite.
    `dataset_path` rows already came from S3 (an upload or a snapshot hit) and
    are never re-uploaded.
    """
    if dataset_path:
        return normalize_rows(load_jsonl_rows(dataset_path))

    if not dataset_config:
        raise ValueError("No dataset path or dataset config provided")

    # HuggingFace loading is inlined so the runner has no control-plane dependency
    source_type = dataset_config.get("source_type", "upload")
    if source_type == "hf":
        from datasets import load_dataset as hf_load_dataset

        hf_name = dataset_config.get("hf_dataset")
        if not hf_name:
            raise ValueError("hf_dataset is required for HuggingFace source")
        hf_config = dataset_config.get("hf_config") or {}
        split = hf_config.get("split", "train")
        hf_subset = hf_config.get("subset") or hf_config.get("config_name")
        column_mapping = hf_config.get("column_mapping", {})

        kwargs = {"split": split}
        if hf_config.get("token"):
            kwargs["token"] = hf_config["token"]

        ds = hf_load_dataset(hf_name, hf_subset, **kwargs)
        rows = [dict(row) for row in ds]
        if column_mapping:
            rows = [{column_mapping.get(k, k): v for k, v in row.items()} for row in rows]
        rows = normalize_rows(rows)
        if snapshot_upload_url:
            _upload_dataset_snapshot(rows, snapshot_upload_url)
        return rows

    raise ValueError(f"Unsupported dataset source_type={source_type}")


def _upload_dataset_snapshot(rows: list[dict[str, Any]], upload_url: str) -> None:
    """Best-effort PUT of normalized rows as JSONL to a presigned URL.

    The snapshot is a cache — an S3 hiccup must never kill a training run the
    user is paying for, so failures are logged and swallowed (same contract as
    the periodic log uploader).
    """
    import urllib.request

    log = logging.getLogger("veri.training_runtime")
    try:
        data = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode()
        req = urllib.request.Request(upload_url, data=data, method="PUT")
        urllib.request.urlopen(req, timeout=300)
        log.info("Uploaded normalized dataset snapshot (%d rows, %d bytes)", len(rows), len(data))
    except Exception as e:
        log.warning("Dataset snapshot upload failed (non-fatal): %s", e)


# HF datasets arrive in a handful of community formats. Only rows the
# TRL trainers natively understand (text / role-content messages / prompt[-
# completion] / preference pairs) may reach them: TRL 0.22.2 renames ShareGPT
# KEYS (conversations/from/value) but copies role VALUES through, so an
# unmapped `human` role would silently train `<|im_start|>human`. Everything
# else is converted here, or rejected with `dataset_format_incompatible:` (a
# classify_error code) before a model is loaded.
_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "prompter": "user",  # OASST
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}

_FORMAT_INCOMPATIBLE = "dataset_format_incompatible"


def detect_dataset_format(row: Mapping[str, Any]) -> str:
    """Classify a dataset row by the community format its keys/values follow.

    Checked most-specific first; `column_mapping` has already been applied, so
    a user can rename odd top-level columns before detection sees them.
    """
    def _is_msg_list(v: Any, keys: tuple[str, str]) -> bool:
        return (
            isinstance(v, list) and len(v) > 0
            and all(isinstance(m, Mapping) and keys[0] in m and keys[1] in m for m in v)
        )

    if "chosen" in row and "rejected" in row:
        return "preference"
    for col in ("conversations", "messages"):
        if col in row:
            if _is_msg_list(row[col], ("from", "value")):
                return "sharegpt"
            if _is_msg_list(row[col], ("role", "content")):
                return "chatml"
    if "instruction" in row and "output" in row:
        return "alpaca"
    if "prompt" in row and "completion" in row:
        return "prompt_completion"
    if "prompt" in row:
        return "prompt_only"
    if "text" in row:
        return "text"
    return "unknown"


def _map_role(raw: str) -> str:
    role = _ROLE_MAP.get(str(raw).lower())
    if role is None:
        raise ValueError(
            f"{_FORMAT_INCOMPATIBLE}: unmapped conversation role '{raw}'. "
            f"Supported roles: {sorted(set(_ROLE_MAP))}. Tool-calling datasets "
            "(observation/function_call roles) are not supported yet."
        )
    return role


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert rows to the TRL-native shape (mirrors Unsloth's
    standardize_data_formats). Idempotent: TRL-native rows pass through, so
    already-normalized snapshot rows can safely re-enter the pipeline.
    """
    if not rows:
        return rows
    fmt = detect_dataset_format(rows[0])

    if fmt == "unknown":
        raise ValueError(
            f"{_FORMAT_INCOMPATIBLE}: could not recognize the dataset format from "
            f"columns {sorted(rows[0])}. Expected one of: messages/conversations "
            "(chat), text, prompt(+completion), instruction+output (Alpaca), "
            "chosen+rejected (preference). Use column_mapping to rename columns."
        )

    if fmt == "sharegpt":
        col = "conversations" if "conversations" in rows[0] else "messages"
        return [
            {
                **{k: v for k, v in row.items() if k != col},
                "messages": [
                    {"role": _map_role(m["from"]), "content": m["value"]}
                    for m in row[col]
                ],
            }
            for row in rows
        ]

    if fmt == "chatml":
        # Already role/content; still fold role-name variants (Human, USER, …)
        # through the map so the chat template never sees an unknown role.
        return [
            {
                **{k: v for k, v in row.items() if k != "messages"},
                "messages": [
                    {**m, "role": _map_role(m["role"])} for m in row["messages"]
                ],
            }
            for row in rows
        ]

    if fmt == "alpaca":
        def _to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
            user = row["instruction"]
            if row.get("input") and str(row["input"]).strip():
                user = f"{row['instruction']}\n\n{row['input']}"
            messages = [{"role": "user", "content": user}]
            if row.get("system") and str(row["system"]).strip():
                messages.insert(0, {"role": "system", "content": row["system"]})
            messages.append({"role": "assistant", "content": row["output"]})
            return messages

        drop = ("instruction", "input", "output", "system")
        return [
            {
                **{k: v for k, v in row.items() if k not in drop},
                "messages": _to_messages(row),
            }
            for row in rows
        ]

    # text / prompt_only / prompt_completion / preference: TRL-native already.
    return rows


def validate_rows_for_method(rows: list[dict[str, Any]], method: str) -> None:
    """Fail fast (before model load) when normalized rows can't feed `method`."""
    if not rows:
        return
    fmt = detect_dataset_format(rows[0])
    if method == "sft_text":
        if fmt not in ("text", "chatml", "prompt_completion"):
            raise ValueError(
                f"{_FORMAT_INCOMPATIBLE}: method 'sft_text' needs rows shaped as "
                "text, messages (chat), or prompt+completion; got "
                f"'{fmt}' (columns {sorted(rows[0])})."
            )
    elif method in ("grpo", "grpo_agentic", "grpo_harness"):
        if "prompt" not in rows[0]:
            raise ValueError(
                f"{_FORMAT_INCOMPATIBLE}: method '{method}' needs a 'prompt' column "
                f"(got '{fmt}', columns {sorted(rows[0])}). Use column_mapping to "
                "map your prompt column, e.g. {\"question\": \"prompt\"}."
            )
    elif method == "dpo":
        _require_preference_columns(rows)


def _apply_system_prompt(
    rows: list[dict[str, Any]], system_prompt: str | None,
) -> list[dict[str, Any]]:
    """Convert string `prompt` columns to conversational chat-message lists.

    Without a system prompt, TRL feeds the raw string straight to the model
    and Qwen-Instruct never learns it's supposed to wrap output in
    <answer> tags (see GSM8K quickstart). When a system_prompt is set, we
    reshape each row's prompt into [{system}, {user}] so the tokenizer's
    chat template fires inside GRPOTrainer.
    """
    if not system_prompt:
        return rows
    reshaped = []
    for row in rows:
        new_row = dict(row)
        user_content = new_row.get("prompt")
        if isinstance(user_content, str):
            new_row["prompt"] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        reshaped.append(new_row)
    return reshaped


def _setup_wandb(job_config: dict[str, Any]) -> bool:
    """Configure W&B environment if credentials are provided. Returns True if enabled."""
    wandb_key = job_config.get("wandb_api_key")
    if not wandb_key:
        return False
    os.environ["WANDB_API_KEY"] = wandb_key
    os.environ["WANDB_PROJECT"] = job_config.get("wandb_project", "veri-training")
    # WANDB_NAME is the env var wandb actually reads for the run name
    # (WANDB_RUN_NAME is not part of its env contract).
    os.environ["WANDB_NAME"] = job_config.get("output_name", job_config["job_id"])
    os.environ["WANDB_RUN_ID"] = job_config["job_id"]
    return True


def _apply_liger(
    config_kwargs: dict[str, Any], config_params, hyperparameters: dict[str, Any]
) -> None:
    """Opt-in liger-kernel: fused Triton train kernels, ROCm-capable
    since liger 0.4. Only set when requested AND the installed transformers
    exposes the flag — an older signature silently trains without it rather
    than TypeError-ing the run. Availability of the package itself is gated
    in run_training (fail fast, before model download)."""
    if hyperparameters.get("use_liger") and "use_liger_kernel" in config_params:
        config_kwargs["use_liger_kernel"] = True


def build_grpo_config_kwargs(
    *,
    job_id: str,
    hyperparameters: dict[str, Any],
    grpo_config_cls: type,
    output_root: str = "/tmp/ckpts",
    wandb_enabled: bool = False,
    # VS-393: the control plane's computed save policy. None => no
    # intermediate checkpoints, i.e. exactly pre-VS-393 behaviour.
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build GRPOConfig kwargs while adapting to the installed TRL signature."""
    grpo_params = inspect.signature(grpo_config_cls.__init__).parameters

    config_kwargs = {
        "output_dir": str(Path(output_root) / job_id),
        "max_steps": hyperparameters.get("max_steps") or 100,
        # .get with the API schema's defaults: the control plane stores the
        # client's raw hyperparameters JSON, so a submit that omits an
        # optional field (SDK create / TOML form) reaches the worker without
        # it — every optional knob must fall back to the API default here.
        "per_device_train_batch_size": max(hyperparameters.get("rollouts_per_prompt", 8), 2),
        "num_generations": hyperparameters.get("rollouts_per_prompt", 8),
        "max_completion_length": hyperparameters.get("max_response_length", 2048),
        "learning_rate": hyperparameters.get("learning_rate", 1e-6),
        "logging_steps": 1,
        "num_train_epochs": 1,
        "report_to": "wandb" if wandb_enabled else "none",
        "bf16": _gpu_supports_bf16(),
        "gradient_checkpointing": True,
    }
    if "max_prompt_length" in grpo_params and "max_prompt_length" in hyperparameters:
        config_kwargs["max_prompt_length"] = hyperparameters["max_prompt_length"]
    if "beta" in grpo_params and "kl_coef" in hyperparameters:
        config_kwargs["beta"] = hyperparameters["kl_coef"]
    # seed maps 1:1 onto GRPOConfig.seed (TrainingArguments.seed in TRL 1.7.1).
    # Omitted, TRL's default (42) applies, which is also the API default. The
    # control plane rejects the GRPO knobs TRL has no field for
    # (global_batch_size, and max_prompt_length without use_unsloth).
    if "seed" in grpo_params and "seed" in hyperparameters:
        config_kwargs["seed"] = hyperparameters["seed"]
    # Base path generates via transformers. vLLM rollouts are an opt-in
    # optimization (trl 0.22.2 + vllm 0.15.1 aren't directly compatible), so default
    # use_vllm off; a user with a compatible setup can flip it via hyperparameters.
    if "use_vllm" in grpo_params:
        config_kwargs["use_vllm"] = bool(hyperparameters.get("use_vllm", False))
    _apply_liger(config_kwargs, grpo_params, hyperparameters)

    # VS-393: periodic saving, filtered to what the installed TRL accepts
    # (same drift adaptation as the knobs above).
    config_kwargs.update(build_save_kwargs(checkpoint, grpo_params))
    # C2: GRPO buffers steps_per_generation x num_iterations optimizer steps
    # of rollouts and does not checkpoint them (TRL regenerates after a
    # resume). A step-count cadence that lands on a generation boundary makes
    # the resumed run identical to an uninterrupted one; a fraction (HF
    # resolves it at train time) is left alone.
    save_steps = config_kwargs.get("save_steps")
    if isinstance(save_steps, int) and not isinstance(save_steps, bool) and save_steps > 0:
        config_kwargs["save_steps"] = grpo_aligned_save_steps(save_steps, hyperparameters)

    return config_kwargs


def grpo_aligned_save_steps(save_steps: int, hyperparameters: Mapping[str, Any]) -> int:
    """Round a GRPO step-count cadence UP to a multiple of
    steps_per_generation x num_iterations (TRL defaults: gradient
    accumulation steps, 1)."""
    spg = int(
        hyperparameters.get("steps_per_generation")
        or hyperparameters.get("gradient_accumulation_steps")
        or 1
    )
    iters = int(hyperparameters.get("num_iterations") or 1)
    period = max(1, spg * iters)
    return ((save_steps + period - 1) // period) * period


def build_trainer_kwargs(
    *,
    trainer_cls: type,
    model: Any,
    training_args: Any,
    train_dataset: Any,
    tokenizer: Any,
    reward_fn: Callable[..., Any] | list[Callable[..., Any]] | None = None,
    reward_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Build TRL trainer kwargs across tokenizer/processing_class variants.

    `reward_funcs` is only included when `reward_fn` is provided, so this also
    serves reward-free trainers (e.g. SFTTrainer), which reject `reward_funcs`.
    `reward_fn` may be a single callable or a list (multi-reward GRPO);
    `reward_weights` is included only when given and the trainer supports it.
    """
    trainer_params = inspect.signature(trainer_cls.__init__).parameters

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    if reward_fn is not None:
        trainer_kwargs["reward_funcs"] = reward_fn
        if reward_weights is not None and "reward_weights" in trainer_params:
            trainer_kwargs["reward_weights"] = reward_weights
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    return trainer_kwargs


def _batch_knobs(
    hyperparameters: dict[str, Any], launch_plan: dict[str, Any] | None
) -> tuple[int, int | None, bool, str]:
    """(micro_batch_size, gradient_accumulation_steps | None, gradient_checkpointing,
    precision) for a config builder: from the resolved launch plan when the job
    runs under the multi-GPU launcher, else straight from the playbook knobs
    (`batch_size` stays the legacy alias of `micro_batch_size`). A None ga means
    "do not set the key", which keeps a knob-free single-GPU job byte-for-byte
    today's config."""
    if launch_plan:
        return (
            int(launch_plan["micro_batch_size"]),
            int(launch_plan["gradient_accumulation_steps"]),
            bool(launch_plan["gradient_checkpointing"]),
            str(launch_plan["precision"]),
        )
    mbs = int(hyperparameters.get("micro_batch_size") or hyperparameters.get("batch_size") or 1)
    ga = hyperparameters.get("gradient_accumulation_steps")
    gbs = hyperparameters.get("global_batch_size")
    if ga is None and gbs:
        # dp = 1 in-process: gbs = mbs x ga (the control plane checked divisibility)
        ga = max(1, int(gbs) // mbs)
    gc = hyperparameters.get("gradient_checkpointing")
    return (
        mbs,
        None if ga is None else int(ga),
        True if gc is None else bool(gc),
        str(hyperparameters.get("precision") or "bf16"),
    )


def _bf16_flag(precision: str, launch_plan: dict[str, Any] | None) -> bool:
    """TrainingArguments.bf16 for a precision + backend. fp32 never; pure bf16
    under FSDP2 never (accelerate 1.15 upcasts every trainable param to fp32
    master weights whenever its mixed precision is not 'no', fsdp_utils.py:816);
    everything else is today's GPU probe (bf16 autocast on a bf16 or fp32 load)."""
    if precision == "fp32":
        return False
    if launch_plan and launch_plan.get("backend") == "fsdp2" and precision == "bf16":
        return False
    return _gpu_supports_bf16()


def _apply_batch_knobs(
    config_kwargs: dict[str, Any],
    hyperparameters: dict[str, Any],
    launch_plan: dict[str, Any] | None,
) -> None:
    mbs, ga, gc, precision = _batch_knobs(hyperparameters, launch_plan)
    config_kwargs["per_device_train_batch_size"] = mbs
    config_kwargs["bf16"] = _bf16_flag(precision, launch_plan)
    # Under FSDP2 recomputation is done by accelerate (fsdp_activation_checkpointing
    # in the launch config); the Trainer flag would add a redundant all-gather
    # per layer in backward (transformers#30404).
    fsdp = bool(launch_plan) and launch_plan.get("backend") == "fsdp2"
    config_kwargs["gradient_checkpointing"] = gc and not fsdp
    if ga is not None:
        config_kwargs["gradient_accumulation_steps"] = ga
    if launch_plan and int(launch_plan.get("world_size", 1)) > 1:
        # Q13 / Q-D: the NCCL process-group timeout bounds a dead peer; it is
        # stretched around the estimated rank-0 checkpoint write (HF default 1800 s).
        config_kwargs["ddp_timeout"] = int(launch_plan.get("ddp_timeout_s") or 600)
        if launch_plan.get("backend") == "ddp":
            # HF turns this on for any PeftModel; every adapter param gets a
            # gradient, and the unused-parameter scan costs a graph walk per step.
            config_kwargs["ddp_find_unused_parameters"] = False


def build_sft_config_kwargs(
    *,
    job_id: str,
    hyperparameters: dict[str, Any],
    sft_config_cls: type,
    output_root: str = "/tmp/ckpts",
    wandb_enabled: bool = False,
    # VS-393: the control plane's computed save policy. None => no
    # intermediate checkpoints, i.e. exactly pre-VS-393 behaviour.
    checkpoint: dict[str, Any] | None = None,
    # VS-476: the resolved multi-GPU plan when running under accelerate launch.
    launch_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build SFTConfig kwargs while adapting to the installed TRL signature.

    Mirrors `build_grpo_config_kwargs`. SFT-specific knobs (`packing`,
    `dataset_text_field`, and the `max_seq_length` -> `max_length` rename) are
    guarded against TRL version drift via the constructor signature.
    """
    sft_params = inspect.signature(sft_config_cls.__init__).parameters

    config_kwargs = {
        "output_dir": str(Path(output_root) / job_id),
        "learning_rate": hyperparameters.get("learning_rate", 2e-5),
        "num_train_epochs": hyperparameters.get("num_epochs", 1),
        "logging_steps": 1,
        "report_to": "wandb" if wandb_enabled else "none",
    }
    _apply_batch_knobs(config_kwargs, hyperparameters, launch_plan)
    if hyperparameters.get("max_steps"):
        config_kwargs["max_steps"] = hyperparameters["max_steps"]
    if "packing" in sft_params:
        config_kwargs["packing"] = bool(hyperparameters.get("packing", False))
    if "dataset_text_field" in sft_params:
        config_kwargs["dataset_text_field"] = hyperparameters.get("dataset_text_field", "text")
    # TRL renamed max_seq_length -> max_length; set whichever the version exposes.
    max_seq = hyperparameters.get("max_seq_length")
    if max_seq:
        if "max_length" in sft_params:
            config_kwargs["max_length"] = max_seq
        elif "max_seq_length" in sft_params:
            config_kwargs["max_seq_length"] = max_seq
    _apply_liger(config_kwargs, sft_params, hyperparameters)

    # VS-393: periodic saving, filtered to what the installed TRL accepts
    # (same drift adaptation as the knobs above).
    config_kwargs.update(build_save_kwargs(checkpoint, sft_params))

    return config_kwargs


def build_dpo_config_kwargs(
    *,
    job_id: str,
    hyperparameters: dict[str, Any],
    dpo_config_cls: type,
    output_root: str = "/tmp/ckpts",
    wandb_enabled: bool = False,
    # VS-393: the control plane's computed save policy. None => no
    # intermediate checkpoints, i.e. exactly pre-VS-393 behaviour.
    checkpoint: dict[str, Any] | None = None,
    # VS-476: the resolved multi-GPU plan when running under accelerate launch.
    launch_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build DPOConfig kwargs while adapting to the installed TRL signature.

    Mirrors `build_sft_config_kwargs`. DPO-specific knobs (`beta`, `loss_type`,
    `max_length`, `max_prompt_length`) are guarded against TRL version drift via
    the constructor signature. Matches TRL's DPOConfig (beta default 0.1,
    sigmoid loss).
    """
    dpo_params = inspect.signature(dpo_config_cls.__init__).parameters

    config_kwargs = {
        "output_dir": str(Path(output_root) / job_id),
        "learning_rate": hyperparameters.get("learning_rate", 5e-6),
        "num_train_epochs": hyperparameters.get("num_epochs", 1),
        "logging_steps": 1,
        "report_to": "wandb" if wandb_enabled else "none",
    }
    _apply_batch_knobs(config_kwargs, hyperparameters, launch_plan)
    # Reference model under the launcher (full FT only; with LoRA, TRL keeps a
    # frozen copy of the adapter instead of a second base): TRL builds it from
    # `model_init_kwargs` with dtype fp32 and nulls device_map only for
    # MULTI_GPU / DEEPSPEED (dpo_trainer.py:806-813), so under FSDP a bare
    # config would try device_map="auto" and load 2x the memory it needs.
    if (
        launch_plan
        and int(launch_plan.get("world_size", 1)) > 1
        and not launch_plan.get("lora")
        and "model_init_kwargs" in dpo_params
    ):
        # bf16 for every precision: the reference only runs forward (under
        # autocast for bf16_mixed), so fp32 would double its memory for nothing.
        config_kwargs["model_init_kwargs"] = {"dtype": "bfloat16", "device_map": None}
    if hyperparameters.get("max_steps"):
        config_kwargs["max_steps"] = hyperparameters["max_steps"]
    if "beta" in dpo_params:
        config_kwargs["beta"] = hyperparameters.get("beta", 0.1)
    if "loss_type" in dpo_params:
        config_kwargs["loss_type"] = hyperparameters.get("loss_type", "sigmoid")
    if hyperparameters.get("max_length") and "max_length" in dpo_params:
        config_kwargs["max_length"] = hyperparameters["max_length"]
    if hyperparameters.get("max_prompt_length") and "max_prompt_length" in dpo_params:
        config_kwargs["max_prompt_length"] = hyperparameters["max_prompt_length"]
    _apply_liger(config_kwargs, dpo_params, hyperparameters)

    # VS-393: periodic saving, filtered to what the installed TRL accepts
    # (same drift adaptation as the knobs above).
    config_kwargs.update(build_save_kwargs(checkpoint, dpo_params))

    return config_kwargs


def resume_checkpoint_dir(job_config: dict[str, Any]) -> str | None:
    """C2: the local directory of the downloaded source checkpoint, or None.
    Only a directory that holds trainer_state.json counts: resuming from a
    partial download would raise deep inside the Trainer."""
    resume = job_config.get("resume") or {}
    local_dir = resume.get("local_dir")
    if not local_dir:
        return None
    if not (Path(local_dir) / "trainer_state.json").is_file():
        raise ValueError(
            f"resume: {local_dir} is not a complete checkpoint (no trainer_state.json)"
        )
    return str(local_dir)


def checkpoint_output_root(job_config: dict[str, Any]) -> str:
    """Return the runner local output root from structured checkpoint config."""
    checkpoint_config = job_config.get("checkpoint") or {}
    return checkpoint_config.get("local_output_root", "/tmp/ckpts")


# Keys of the control plane's `checkpoint` block that map 1:1 onto HF
# TrainingArguments. The control plane computes the POLICY (VS-393 design 5.3);
# this only translates and filters it to what the installed TRL accepts.
_SAVE_KWARG_KEYS = (
    "save_strategy",
    "save_steps",
    "save_total_limit",
    "save_safetensors",
    "save_only_model",
    "ignore_data_skip",
    "save_on_each_node",
)


def build_save_kwargs(
    checkpoint: dict[str, Any] | None,
    config_params: Any = None,
) -> dict[str, Any]:
    """Translate the job config's `checkpoint` block into HF save kwargs.

    `None`/absent/disabled all yield `save_strategy="no"`, which is the behaviour
    every managed job had before VS-393: one artifact, written after train()
    returns. That default matters -- it is what makes the control plane's feature
    flag a real kill switch rather than a request the runner may ignore.

    `config_params` is the installed config class's signature parameters (the
    same signature-adaptation the builders already do for TRL drift); when given,
    kwargs the installed TRL does not accept are dropped rather than raising.
    """
    if not checkpoint or not checkpoint.get("enabled"):
        return {"save_strategy": "no"}

    kwargs = {k: checkpoint[k] for k in _SAVE_KWARG_KEYS if k in checkpoint}
    kwargs.setdefault("save_strategy", "steps")
    if config_params is not None:
        kwargs = {k: v for k, v in kwargs.items() if k in config_params}
    return kwargs


def save_checkpoint(
    *,
    trainer: Any,
    tokenizer: Any,
    job_id: str,
    output_root: str = "/tmp/ckpts",
) -> str:
    """Save the trained model/tokenizer and return the final checkpoint directory."""
    final_dir = str(Path(output_root) / job_id / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    return final_dir


def _save_merged_for_push(
    *, trainer: Any, tokenizer: Any, merged_dir: str, log: logging.Logger
) -> str:
    """Materialize standalone (merged) weights from a LoRA-trained model.

    Unsloth models expose save_pretrained_merged; vanilla PEFT merges the
    adapter into the base with merge_and_unload. Either way the result is a
    self-contained model dir a Hugging Face repo can serve without PEFT.
    """
    model = trainer.model
    if hasattr(model, "save_pretrained_merged"):
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    else:
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
    return merged_dir


def _merge_adapter_from_path(
    *, base_model: str, adapter_dir: str, merged_dir: str, log: logging.Logger
) -> str:
    """Merge a saved LoRA adapter into a freshly loaded base and save a
    standalone model dir (CPU/bf16 is fine: this runs after training)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Merging adapter %s into %s -> %s", adapter_dir, base_model, merged_dir)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=_allow_remote_code(base_model)
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(merged_dir)
    return merged_dir


def push_to_hf_hub(
    *,
    job_config: dict[str, Any],
    final_dir: str,
    trainer: Any = None,
    tokenizer: Any = None,
    logger: logging.Logger | None = None,
) -> str | None:
    """Push the trained artifact to the user's Hugging Face account.

    Opt-in via job_config["hf_push"] = {repo, artifact: merged|adapter,
    private}; the write token arrives as job_config["hf_token"] through the
    worker's runtime credential exchange (never via config.json on S3).

    Strictly non-fatal: the training run already succeeded, so every failure
    here logs a warning and returns None instead of raising. The token is
    never logged.
    """
    log = logger or logging.getLogger("veri.training_runtime")
    hf = job_config.get("hf_push") or {}
    repo_id = hf.get("repo")
    if not repo_id:
        return None

    token = job_config.get("hf_token")
    if not token:
        log.warning(
            "HF push requested for %s but no token was delivered "
            "(credential exchange failed?); skipping push.",
            repo_id,
        )
        return None

    artifact = hf.get("artifact", "merged")
    private = bool(hf.get("private", True))
    is_adapter_ckpt = (Path(final_dir) / "adapter_config.json").exists()

    try:
        if artifact == "adapter":
            if not is_adapter_ckpt:
                log.warning(
                    "HF push artifact=adapter but the checkpoint has no "
                    "adapter_config.json (full fine-tune?); skipping push."
                )
                return None
            upload_dir = final_dir
        elif is_adapter_ckpt and trainer is None:
            # VS-476: the worker parent pushes after a multi-GPU run from the
            # saved adapter; no trainer lives in this process, so merge from
            # disk (fresh base + adapter) now that the GPUs are free.
            upload_dir = _merge_adapter_from_path(
                base_model=job_config["base_model"],
                adapter_dir=final_dir,
                merged_dir=str(Path(final_dir).parent / "merged"),
                log=log,
            )
        elif is_adapter_ckpt:
            # merged requested from a LoRA run: materialize standalone weights
            # next to the adapter checkpoint before uploading.
            upload_dir = _save_merged_for_push(
                trainer=trainer,
                tokenizer=tokenizer,
                merged_dir=str(Path(final_dir).parent / "merged"),
                log=log,
            )
        else:
            upload_dir = final_dir  # full fine-tune save is already standalone

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.whoami()  # fast, clear failure on a bad/expired token
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        log.info(
            "Pushing %s artifact to Hugging Face repo %s (private=%s)…",
            artifact,
            repo_id,
            private,
        )
        api.upload_folder(
            folder_path=upload_dir,
            repo_id=repo_id,
            commit_message=f"Veri training job {job_config.get('job_id')} ({artifact})",
        )
        url = f"https://huggingface.co/{repo_id}"
        log.info("HF push complete: %s", url)
        return url
    except Exception as e:  # noqa: BLE001 -- push must never fail the job
        log.warning("HF push to %s failed (non-fatal): %s", repo_id, e)
        return None


# `base_model` is a free-form, user-controlled field (job_config["base_model"])
# with NO control-plane allowlist as of this fix. Passing trust_remote_code=True makes
# HuggingFace execute arbitrary modeling_*.py / configuration_*.py from the referenced
# repo IN-PROCESS on the training host — which holds the callback token, W&B key, volume
# STS creds, and reachable IMDS. So remote code defaults OFF and is only enabled for
# repos an operator has vetted out-of-band. The allowlist lives in worker config, NOT in
# the job payload, so a malicious job can't self-assert trust. Kept empty by default:
# the models Veri officially supports (Qwen3, Llama, ...) use in-library architectures
# and do NOT need remote code. FOLLOW-UP: promote this to a control-plane-signed
# `allow_remote_code` signal on the vetted base-model registry once one exists.
_TRUSTED_REMOTE_CODE_MODELS: frozenset[str] = frozenset()


def _allow_remote_code(base_model: str) -> bool:
    """Decide trust_remote_code for a user-supplied base_model.

    True only when the repo is on the operator-curated allowlist: the hardcoded
    `_TRUSTED_REMOTE_CODE_MODELS` set, or the out-of-band, comma-separated
    VERI_TRUST_REMOTE_CODE_MODELS env var (worker config, never the job payload).
    Everything else — including any repo an attacker can name — loads with remote
    code disabled.
    """
    if base_model in _TRUSTED_REMOTE_CODE_MODELS:
        return True
    env_allowlist = os.environ.get("VERI_TRUST_REMOTE_CODE_MODELS", "")
    allowed = {m.strip() for m in env_allowlist.split(",") if m.strip()}
    return base_model in allowed


def _model_placement(
    hyperparameters: dict[str, Any], launch_plan: dict[str, Any] | None
) -> tuple[str, Any]:
    """(torch dtype attribute name, device_map) for the vanilla load path.

    No plan (single GPU, in-process): today's `device_map="auto"`, bf16 unless
    `precision` asks for fp32 master weights (bf16_mixed = fp32 load + autocast;
    fp32 = fp32 end to end).
    DDP: one full copy per rank on its own device (`{"": local_rank}`; None on a
    CPU-only box so the gloo integration path works); fp32 load for bf16_mixed.
    FSDP2: no device_map; every rank loads the full model into host RAM (the
    process group does not exist yet, so accelerate's rank-0-only loading cannot
    apply) and `fully_shard` keeps each rank's slice; bf16 load even for
    bf16_mixed because accelerate upcasts the sharded trainable params to fp32
    master weights itself.
    """
    precision = str(
        (launch_plan or {}).get("precision") or hyperparameters.get("precision") or "bf16"
    )
    lora = hyperparameters.get("lora_rank") is not None
    # LoRA + bf16_mixed = frozen bf16 base, fp32 adapters (PEFT autocasts adapter
    # weights to fp32 on a half-precision base), so the base never loads in fp32
    # unless precision is fp32 outright.
    if precision == "bf16" or (precision == "bf16_mixed" and lora):
        dtype = "bfloat16"
    elif precision == "bf16_mixed" and launch_plan and launch_plan.get("backend") == "fsdp2":
        dtype = "bfloat16"  # accelerate upcasts the sharded master copy to fp32 itself
    else:
        dtype = "float32"
    if not launch_plan or int(launch_plan.get("world_size", 1)) <= 1:
        return dtype, "auto"
    if launch_plan.get("backend") == "fsdp2":
        return dtype, None
    import torch

    device_map = (
        {"": int(os.environ.get("LOCAL_RANK", "0"))} if torch.cuda.is_available() else None
    )
    return dtype, device_map


def _load_model_and_tokenizer(
    base_model: str,
    hyperparameters: dict[str, Any],
    logger: logging.Logger,
    launch_plan: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Load model+tokenizer via Unsloth (opt-in) or vanilla transformers (default).

    Unsloth is off by default because its torch.compile'd selective log-softmax
    has an open shape-mismatch bug with GRPOTrainer
    (https://github.com/unslothai/unsloth/issues/3069, status: open/no-fix,
    hit in production 2026-05-19). Vanilla TRL + transformers
    is the stable path; it loses unsloth's ~50% VRAM / 1.5-4x speedup but works
    across model sizes, multi-GPU, and torch/trl/vllm version drift.

    Users opt into Unsloth via hyperparameters["use_unsloth"]=True when they
    have a single GPU and a model big enough to need the VRAM optimization
    (~14B+ on one A100). Multi-GPU + Unsloth is rejected at SDK submission
    time — Unsloth's multi-GPU support is still "preliminary" (docs.unsloth.ai)
    and needs DDP/Accelerate glue we haven't wired.
    """
    use_unsloth = hyperparameters.get("use_unsloth", False)
    lora_rank = hyperparameters.get("lora_rank")
    load_in_4bit = hyperparameters.get("load_in_4bit", False)
    max_seq_length = (
        hyperparameters.get("max_prompt_length", 1024)
        + hyperparameters.get("max_response_length", 2048)
    )

    # QLoRA (4-bit) is only meaningful as a 4-bit base + LoRA adapters. The base
    # weights are frozen in 4-bit, so there is nothing to full-finetune; the
    # gradients live entirely in the LoRA adapters. Mirrors Unsloth's "4-bit
    # requires LoRA adapters" / "only one training method True at a time" rule.
    # Guard here (not just at the SDK) so a hand-rolled config can't reach the
    # GPU and crash 10 minutes into a boot with an opaque Unsloth error.
    if load_in_4bit and lora_rank is None:
        raise ValueError(
            "load_in_4bit=True (QLoRA) requires lora_rank to be set: 4-bit "
            "quantizes and freezes the base weights, so training happens in the "
            "LoRA adapters. Set a lora_rank (e.g. 16) or drop load_in_4bit for a "
            "full / 16-bit-LoRA finetune."
        )

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                full_finetuning=(lora_rank is None and not load_in_4bit),
            )
            if lora_rank is not None:
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=lora_rank,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ],
                    lora_alpha=hyperparameters.get("lora_alpha") or lora_rank,
                    use_gradient_checkpointing="unsloth",
                )
            logger.info(
                "Model loaded via Unsloth (lora_rank=%s, load_in_4bit=%s)",
                lora_rank,
                load_in_4bit,
            )
            return model, tokenizer
        except ImportError:
            logger.info("Unsloth not installed, using transformers")
        except Exception as exc:
            logger.warning(
                "Unsloth failed for %s (%s), using transformers", base_model, exc
            )

    # Vanilla path: transformers + PEFT/bitsandbytes. This is the default and is
    # multi-GPU capable; it also avoids the open Unsloth+GRPOTrainer torch.compile
    # bug (unsloth #3069). LoRA (lora_rank set) and QLoRA (load_in_4bit) are applied
    # here via PEFT — both deps already ship in the AMI (bitsandbytes explicit,
    # peft via unsloth). Recipe matches the HF PEFT QLoRA examples + TRL peft_config.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # default OFF; only trust repos on the operator-vetted allowlist.
    allow_remote_code = _allow_remote_code(base_model)
    if allow_remote_code:
        logger.info("trust_remote_code enabled for allowlisted base_model %s", base_model)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=allow_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name, device_map = _model_placement(hyperparameters, launch_plan)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": getattr(torch, dtype_name),
        "device_map": device_map,
        "trust_remote_code": allow_remote_code,
    }
    if load_in_4bit:
        # QLoRA: NF4 4-bit base + double quant, bf16 compute. The base is frozen;
        # gradients live in the LoRA adapters (guarded above: requires lora_rank).
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if lora_rank is not None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_rank,
                lora_alpha=hyperparameters.get("lora_alpha") or lora_rank,
                lora_dropout=hyperparameters.get("lora_dropout", 0.0),
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        logger.info(
            "PEFT LoRA applied via transformers (lora_rank=%s, load_in_4bit=%s)",
            lora_rank,
            load_in_4bit,
        )
    return model, tokenizer


def _ensure_vanilla_trl_imports() -> None:
    """Make `from trl import GRPOTrainer` work on the vanilla (no-unsloth) path.

    trl 0.22.2 imports `GuidedDecodingParams` from `vllm.sampling_params`, which
    vllm 0.15 renamed (guided decoding became structured outputs). Re-inject the
    symbol so the import resolves WITHOUT pulling in unsloth's trainer patches.
    Prefer the real class (so an explicitly enabled vLLM path still works); fall
    back to a minimal stub, which is sufficient because the base path generates via
    transformers (use_vllm defaults False), so the symbol is never exercised.
    """
    try:
        import vllm.sampling_params as vsp
    except Exception:
        return  # vllm absent; let the trl import surface its own error
    if hasattr(vsp, "GuidedDecodingParams"):
        return
    alias = None
    for name in ("StructuredOutputsParams", "GuidedDecoding"):
        alias = getattr(vsp, name, None)
        if alias is not None:
            break
    if alias is None:
        try:
            from vllm.config import StructuredOutputsParams as alias  # newer location
        except Exception:
            alias = None
    if alias is None:
        from dataclasses import dataclass

        @dataclass
        class _GuidedDecodingParamsShim:
            json: Any = None
            regex: Any = None
            choice: Any = None
            grammar: Any = None
            backend: Any = None
            whitespace_pattern: Any = None

        alias = _GuidedDecodingParamsShim
    vsp.GuidedDecodingParams = alias


# ---- Shared TRL lifecycle helpers ----
#
# The GRPO/SFT/DPO runners share three verbatim slices of lifecycle: framework
# resolution, progress-callback attachment, and the train -> save -> result ->
# W&B-finalize tail. They are deliberately NARROW helpers, not one universal
# training loop — the middles differ for good reasons (SFT needs the tokenizer
# before pre-rendering chat templates, GRPO must load rewards first, DPO
# validates preference rows) and a future method may not use a HF Trainer at
# all. Dataset preparation and trainer construction stay in each method.


def _prepare_trl_framework(hyperparameters: dict[str, Any], log: logging.Logger) -> bool:
    """Resolve the training framework BEFORE importing trl.

    unsloth (opt-in) must patch the trl trainers at import time; the vanilla
    base path shims the vLLM symbol instead. Mixing them (vanilla model +
    unsloth trainer) is the `for_training` crash — both states stay internally
    consistent. Returns the resolved `use_unsloth` flag.
    """
    use_unsloth = hyperparameters.get("use_unsloth", False)
    if use_unsloth:
        import unsloth  # noqa: F401  -- patches trl trainers at import time

        log.info("unsloth enabled (opt-in)")
    else:
        _ensure_vanilla_trl_imports()
    return use_unsloth


def _attach_progress_callback(
    trainer: Any,
    progress_fn: Callable[[int, int, dict], None] | None,
    phase_fn: Callable[[str, int], None] | None = None,
) -> None:
    """Register step-level progress reporting on any HF Trainer (no-op when
    the adapter didn't pass a progress_fn).

    Multi-process (VS-476): every rank runs the callbacks, only the world's
    process zero reports. `phase_fn` marks the save window ("saving" from the
    step whose save is pending until on_save, then "training") so the parent's
    stall watchdog can stretch its budget around a long FULL_STATE_DICT write
    instead of killing a healthy job.
    """
    if not progress_fn and not phase_fn:
        return
    from transformers import TrainerCallback

    def _main(state) -> bool:
        return bool(getattr(state, "is_world_process_zero", True))

    class _ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state and progress_fn and _main(state):
                progress_fn(state.global_step, state.max_steps, {
                    k: float(v) for k, v in logs.items()
                    if isinstance(v, (int, float))
                })

        def on_step_end(self, args, state, control, **kwargs):
            if phase_fn and _main(state) and getattr(control, "should_save", False):
                phase_fn("saving", int(state.global_step))

        def on_save(self, args, state, control, **kwargs):
            if phase_fn and _main(state):
                phase_fn("training", int(state.global_step))

    trainer.add_callback(_ProgressCallback())


def stage_checkpoint(src: Path, stage_root: Path, step: int) -> Path:
    """Hard-link `src` (checkpoint-N) into `{stage_root}/step-N` and return it.

    HF Trainer writes checkpoint-N, deletes checkpoint-(N-1)
    (save_total_limit=1) and only THEN fires on_save, so an upload reading
    checkpoint-(N-1) was racing an rmtree. A hard-linked copy keeps the bytes
    alive after the Trainer unlinks its names (verified: exp1_save_rotate,
    transformers 4.56.2) at no disk cost; a filesystem that refuses the link
    (EXDEV) gets a real copy instead.
    """
    dst = stage_root / f"step-{step}"
    if dst.exists():
        shutil.rmtree(dst)
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if p.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(p, dst / rel)
        except OSError:
            shutil.copy2(p, dst / rel)
    return dst


class CheckpointUploadQueue:
    """Uploads staged checkpoints in the background: one in flight, one
    pending (the newest), never abandoning the in-flight upload.

    The old policy ("newest wins", abandon the in-flight upload) combined with
    HF's rotation meant a slow uplink committed NOTHING: every upload was
    superseded before it finished and its source directory was deleted under
    it. Now the in-flight upload always completes (its staged copy is safe),
    a newer save waits as the single pending one, and a save that arrives
    while one is already pending replaces it (its staged copy is deleted and
    `checkpoint_upload_skipped` is reported). Disk: at most three checkpoints
    of bytes (the Trainer's live one + in-flight + pending), which the disk
    guard below budgets for.

    Shared by the Trainer callback (on_save) and the worker's directory
    watchers (multi-GPU launcher, custom scripts), so the three paths cannot
    drift.
    """

    def __init__(self, upload_fn, log, event_fn=None, join_timeout_s: float = 600.0):
        self._upload_fn = upload_fn
        self._log = log
        self._event_fn = event_fn
        self._join_timeout_s = join_timeout_s
        self._lock = threading.Lock()
        self._in_flight: tuple[int, Path, threading.Thread] | None = None
        self._pending: tuple[int, Path] | None = None
        self.committed: list[int] = []
        self.skipped: list[int] = []

    def submit(self, step: int, staged_dir: Path) -> None:
        with self._lock:
            if self._in_flight is None or not self._in_flight[2].is_alive():
                self._start(step, staged_dir)
                return
            if self._pending is not None:
                old_step, old_dir = self._pending
                shutil.rmtree(old_dir, ignore_errors=True)
                self.skipped.append(old_step)
                self._log.warning(
                    "checkpoint step-%s skipped: step-%s is still uploading and step-%s is newer",
                    old_step, self._in_flight[0], step,
                )
                self._emit(
                    "checkpoint_upload_skipped",
                    f"checkpoint step-{old_step} was not uploaded: step-{self._in_flight[0]} "
                    f"was still uploading when step-{step} was saved (slow uplink)",
                    {"step": old_step, "in_flight_step": self._in_flight[0], "newer_step": step},
                    level="warning",
                )
            self._pending = (step, staged_dir)

    def _start(self, step: int, staged_dir: Path) -> None:
        """Caller holds the lock."""

        def _run():
            try:
                self._upload_fn(str(staged_dir), step)
                self.committed.append(step)
            except Exception as e:  # noqa: BLE001
                # Never fail the RUN over an upload: the training loop is the
                # expensive part and the next save gets another chance. The
                # control plane simply has no ready row for this step.
                self._log.warning("checkpoint step-%s upload failed: %s", step, e)
            finally:
                shutil.rmtree(staged_dir, ignore_errors=True)
                with self._lock:
                    self._in_flight = None
                    if self._pending is not None:
                        nxt_step, nxt_dir = self._pending
                        self._pending = None
                        self._start(nxt_step, nxt_dir)

        t = threading.Thread(target=_run, name=f"ckpt-upload-{step}", daemon=True)
        self._in_flight = (step, staged_dir, t)
        t.start()

    def _emit(self, event_type, message, metadata, level="info"):
        if self._event_fn is None:
            return
        try:
            self._event_fn(event_type, message, {**metadata, "level": level})
        except Exception as e:  # noqa: BLE001
            self._log.warning("event %s not reported: %s", event_type, e)

    def join(self, timeout_s: float | None = None) -> None:
        """Wait for the in-flight upload and then the pending one (bounded
        each). Without this the daemon threads die at interpreter exit and the
        newest checkpoint, the one a resume wants, is the one that goes
        missing."""
        budget = self._join_timeout_s if timeout_s is None else timeout_s
        for _ in range(2):
            with self._lock:
                t = self._in_flight[2] if self._in_flight else None
            if t is None:
                return
            t.join(timeout=budget)


def _attach_checkpoint_uploader(
    trainer: Any,
    job_config: dict[str, Any],
    upload_fn: Callable[[str, int], None] | None,
    log: logging.Logger,
    event_fn: Callable[[str, str, dict], None] | None = None,
) -> None:
    """Upload each periodic checkpoint off the training thread (VS-393).

    Registered only when the control plane enabled checkpointing AND the adapter
    supplied an upload_fn, so with the feature flag dark this is a no-op. At
    on_save the fresh checkpoint-N is hard-linked into {output_dir}/.upload/
    (stage_checkpoint) and handed to a CheckpointUploadQueue; HF's own
    save_total_limit keeps owning the Trainer's directory.
    """
    checkpoint_cfg = job_config.get("checkpoint") or {}
    if not checkpoint_cfg.get("enabled") or not upload_fn:
        return

    from transformers import TrainerCallback

    queue = CheckpointUploadQueue(upload_fn, log, event_fn=event_fn)

    class _CheckpointUploadCallback(TrainerCallback):
        def __init__(self) -> None:
            self.queue = queue

        def on_save(self, args, state, control, **kwargs):
            step = int(getattr(state, "global_step", 0) or 0)
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
            if not ckpt_dir.exists():
                log.warning("on_save fired but %s is missing; skipping", ckpt_dir)
                return
            staged = stage_checkpoint(ckpt_dir, Path(args.output_dir) / ".upload", step)
            queue.submit(step, staged)

        def on_train_end(self, args, state, control, **kwargs):
            log.info("waiting for the last checkpoint uploads to finish")
            queue.join()

    trainer.add_callback(_CheckpointUploadCallback())


# ---- C0 disk guard ----

# Bytes per parameter of a resumable checkpoint, as HF/TRL write it with the
# model loaded in bf16: weights 2 + AdamW exp_avg/exp_avg_sq in the parameter
# dtype 4 (= 6N); a FULL_STATE_DICT FSDP save adds pytorch_model_fsdp.bin
# (2N). An adapter run saves only the trainable (LoRA) parameters: bf16 weights
# plus fp32 Adam state, about 10 bytes each.
CHECKPOINT_BYTES_PER_PARAM_FULL = 6
CHECKPOINT_BYTES_PER_PARAM_FSDP_EXTRA = 2
CHECKPOINT_BYTES_PER_TRAINABLE_PARAM_ADAPTER = 10
# The queue keeps up to two staged copies next to the Trainer's live one.
CHECKPOINT_DISK_COPIES = 3


def estimate_checkpoint_bytes(model: Any, fsdp: bool = False) -> int:
    """Resumable checkpoint size from the loaded model's parameter counts."""
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if getattr(p, "requires_grad", False):
            trainable += n
    if trainable and trainable < total:
        return CHECKPOINT_BYTES_PER_TRAINABLE_PARAM_ADAPTER * trainable
    per_param = CHECKPOINT_BYTES_PER_PARAM_FULL + (
        CHECKPOINT_BYTES_PER_PARAM_FSDP_EXTRA if fsdp else 0
    )
    return per_param * total


def apply_checkpoint_disk_guard(
    checkpoint: dict[str, Any] | None,
    model: Any,
    output_root: str,
    log: logging.Logger,
    event_fn: Callable[[str, str, dict], None] | None = None,
    launch_plan: dict[str, Any] | None = None,
    free_bytes: int | None = None,
) -> dict[str, Any] | None:
    """Turn periodic saves off when three checkpoints would not fit on the box.

    Peak local use is the Trainer's live checkpoint plus the queue's in-flight
    and pending staged copies. The budget is the smaller of what df reports
    for `output_root` and the control plane's per-provider `local_disk_gb`
    hint (a container's df can lie). Disabling is loud: a log line and a
    `checkpoint_disabled_disk` job event with the numbers. The final artifact
    is unaffected.
    """
    if not checkpoint or not checkpoint.get("enabled"):
        return checkpoint
    fsdp = bool(launch_plan) and str(launch_plan.get("backend", "")).startswith("fsdp")
    need = CHECKPOINT_DISK_COPIES * estimate_checkpoint_bytes(model, fsdp=fsdp)
    if free_bytes is None:
        os.makedirs(output_root, exist_ok=True)
        free_bytes = shutil.disk_usage(output_root).free
    hint_gb = checkpoint.get("local_disk_gb")
    if hint_gb:
        free_bytes = min(free_bytes, int(hint_gb) * 10**9)
    if need <= free_bytes:
        return checkpoint
    message = (
        f"intermediate checkpoints disabled: {CHECKPOINT_DISK_COPIES} x "
        f"{need / CHECKPOINT_DISK_COPIES / 1e9:.1f} GB per checkpoint exceeds "
        f"{free_bytes / 1e9:.1f} GB of local disk; the final artifact is still saved"
    )
    log.warning(message)
    if event_fn is not None:
        try:
            event_fn(
                "checkpoint_disabled_disk",
                message,
                {
                    "checkpoint_bytes": need // CHECKPOINT_DISK_COPIES,
                    "copies": CHECKPOINT_DISK_COPIES,
                    "free_bytes": free_bytes,
                    "level": "warning",
                },
            )
        except Exception as e:  # noqa: BLE001
            log.warning("checkpoint_disabled_disk event not reported: %s", e)
    return {"enabled": False, "save_strategy": "no", "disabled_reason": "disk"}


def _train_save_finalize(
    *,
    trainer: Any,
    tokenizer: Any,
    job_config: dict[str, Any],
    wandb_enabled: bool,
    log: logging.Logger,
    launch_plan: dict[str, Any] | None = None,
    phase_fn: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    """The shared tail of every TRL method: train, save the checkpoint, build
    the result dict, and capture/finish the W&B run if one is live.

    C2: `job_config["resume"]["local_dir"]` (set by the worker after it
    downloaded the source checkpoint) is handed to
    `trainer.train(resume_from_checkpoint=...)`, which restores the model,
    optimizer, scheduler, RNG and global_step and skips the data already seen
    (verified for SFT, DPO and GRPO on TRL 1.7.1). A fresh run passes None.

    Under the multi-GPU launcher (VS-476) every rank trains and takes part in
    the (collective) final save; rank 0 alone writes the tokenizer and owns the
    result, and nobody pushes to the Hub: the parent does that from the saved
    artifact once the GPUs are free.
    """
    distributed = bool(launch_plan) and int(launch_plan.get("world_size", 1)) > 1
    if distributed:
        _force_fp32_grad_reduce(trainer, launch_plan)
    resume_dir = resume_checkpoint_dir(job_config)
    if resume_dir:
        log.info("Resuming from checkpoint %s", resume_dir)
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_dir)
    train_time = time.time() - t0
    log.info("Training completed in %.1fs, loss=%.4f", train_time, train_result.training_loss)

    if distributed:
        final_dir = str(
            Path(checkpoint_output_root(job_config)) / job_config["job_id"] / "final"
        )
        step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
        if phase_fn and _rank() == 0:
            phase_fn("saving", step)  # the final gather + write can take minutes
        trainer.save_model(final_dir)  # collective: FSDP gathers, rank 0 writes
        if _rank() == 0:
            tokenizer.save_pretrained(final_dir)
            if phase_fn:
                phase_fn("finalizing", step)
    else:
        final_dir = save_checkpoint(
            trainer=trainer,
            tokenizer=tokenizer,
            job_id=job_config["job_id"],
            output_root=checkpoint_output_root(job_config),
        )

    result = {
        "train_time_s": train_time,
        "final_loss": float(train_result.training_loss),
        "checkpoint_dir": final_dir,
        "checkpoint": job_config.get("checkpoint"),
    }
    if distributed:
        result["world_size"] = int(launch_plan["world_size"])
        result["plan"] = launch_plan.get("summary")
        result["peak_memory_bytes_per_rank"] = _gather_peak_memory()
    if wandb_enabled:
        try:
            import wandb

            if wandb.run:
                result["wandb_run_url"] = wandb.run.get_url()
                wandb.finish()
        except Exception:
            pass

    if distributed:
        return result  # the parent pushes from the artifact (section 4 step 6)

    hf_url = push_to_hf_hub(
        job_config=job_config,
        final_dir=final_dir,
        trainer=trainer,
        tokenizer=tokenizer,
        logger=log,
    )
    if hf_url:
        result["hf_repo_url"] = hf_url

    return result


def run_grpo_training(
    job_config: dict[str, Any],
    *,
    dataset_path: str | None,
    reward_path: str | None = None,
    reward_paths: list[str] | None = None,
    logger: logging.Logger | None = None,
    after_model_load: Callable[[], None] | None = None,
    unsupported_reward_formats: Mapping[str, str] | None = None,
    progress_fn: Callable[[int, int, dict], None] | None = None,
    # VS-393: upload one finished checkpoint dir (dir, step). Supplied by the
    # adapter (worker_agent.upload_step_checkpoint); None disables uploading.
    checkpoint_upload_fn: Callable[[str, int], None] | None = None,
    # C0: job events (type, message, metadata) -> the control plane.
    event_fn: Callable[[str, str, dict], None] | None = None,
) -> dict[str, Any]:
    """Run shared GRPO training semantics for a concrete runner adapter."""
    log = logger or logging.getLogger("veri.training_runtime")
    job_id = job_config["job_id"]
    base_model = job_config["base_model"]
    hyperparameters = job_config["hyperparameters"]

    _prepare_trl_framework(hyperparameters, log)

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    wandb_enabled = _setup_wandb(job_config)
    if wandb_enabled:
        log.info("W&B reporting enabled")

    # Multi-reward: the worker may pass several reward files. TRL's
    # GRPOTrainer takes a reward_funcs list (+ optional reward_weights). A single
    # reward stays a single callable so behavior is identical to before.
    paths = reward_paths or ([reward_path] if reward_path else [])
    if not paths:
        raise ValueError("GRPO requires at least one reward function")
    reward_fns = load_reward_functions(
        paths,
        reward_format=job_config.get("reward_format", "trl"),
        unsupported_formats=unsupported_reward_formats,
        logger=log,
    )
    reward_weights = job_config.get("reward_weights")
    multi_reward = len(reward_fns) > 1

    # Rows before the model: dataset/format failures are cheap here and
    # expensive after a multi-minute weight download fail-fast.
    rows = resolve_training_rows(
        dataset_path=dataset_path,
        dataset_config=job_config.get("dataset"),
        snapshot_upload_url=job_config.get("dataset_snapshot_upload_url"),
    )
    validate_rows_for_method(rows, "grpo")
    rows = _apply_system_prompt(rows, hyperparameters.get("system_prompt"))
    dataset = Dataset.from_list(rows)

    t0 = time.time()
    model, tokenizer = _load_model_and_tokenizer(base_model, hyperparameters, log)
    log.info("Model loaded in %.1fs", time.time() - t0)
    if after_model_load:
        after_model_load()
    job_config["checkpoint"] = apply_checkpoint_disk_guard(
        job_config.get("checkpoint"), model, checkpoint_output_root(job_config), log, event_fn
    )

    training_args = GRPOConfig(
        **build_grpo_config_kwargs(
            job_id=job_id,
            hyperparameters=hyperparameters,
            grpo_config_cls=GRPOConfig,
            output_root=checkpoint_output_root(job_config),
            wandb_enabled=wandb_enabled,
            checkpoint=job_config.get("checkpoint"),
        )
    )
    trainer = GRPOTrainer(
        **build_trainer_kwargs(
            trainer_cls=GRPOTrainer,
            model=model,
            training_args=training_args,
            train_dataset=dataset,
            reward_fn=reward_fns if multi_reward else reward_fns[0],
            reward_weights=reward_weights if multi_reward else None,
            tokenizer=tokenizer,
        )
    )

    _attach_progress_callback(trainer, progress_fn)
    _attach_checkpoint_uploader(trainer, job_config, checkpoint_upload_fn, log, event_fn)
    return _train_save_finalize(
        trainer=trainer,
        tokenizer=tokenizer,
        job_config=job_config,
        wandb_enabled=wandb_enabled,
        log=log,
    )


def _wait_for_policy_server(base_url: str, proc: Any, timeout_s: int = 1800) -> None:
    """Block until TRL's vllm-serve answers /health/ (model weights loaded).
    Fails fast if the server process died during startup (OOM, bad flags) —
    the alternative is a hang the user is billed for."""
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"policy server (trl vllm-serve) exited during startup with code {proc.returncode}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/health/", timeout=10) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"policy server did not become healthy within {timeout_s}s")


def _policy_server_devices(gpu_count: int) -> list[int]:
    """CUDA ordinals the policy server owns: GPUs 1..n-1, clamped so TP is legal.

    vLLM requires num_attention_heads % tensor_parallel_size == 0, and TP here
    is exactly len(devices). gpu_count-1 is often illegal (4 GPUs -> TP=3,
    8 -> TP=7; 32-head models like Qwen3 divide by neither), which made every
    aws harness shape crash at vLLM boot (VS-374 blocker 1). Clamp to the
    largest power of two <= gpu_count-1: head counts are in practice divisible
    by small powers of two (32, 28, 40 heads all divide by 4), so odd counts
    degrade to idle GPUs instead of a dead job.
    """
    available = gpu_count - 1
    tp = 1
    while tp * 2 <= available:
        tp *= 2
    return list(range(1, 1 + tp))


def _policy_server_layout(
    gpu_count: int, hyperparameters: dict[str, Any]
) -> tuple[list[int], int, int]:
    """Resolve (devices, tensor_parallel, data_parallel) for the policy server.

    User knobs (both optional, validated at submit and re-checked here):
      vllm_tensor_parallel_size  explicit TP; passed to vLLM verbatim (vLLM
                                 enforces head-count divisibility at boot)
      vllm_data_parallel_size    replica count; TRL vllm-serve chunks requests
                                 across replicas and weight-syncs all of them
                                 (world size = TP*DP + trainer)

    Resolution: both set -> use as-is; DP only -> TP = largest power of two
    that fits available // DP; TP only -> DP = 1; neither -> the safe clamp
    (_policy_server_devices), byte-for-byte the pre-knob behavior. The server
    owns GPUs 1..TP*DP; DP is how idle GPUs on x4/x8 shapes get reclaimed
    (e.g. L4 x4 + dp=3 -> TP=1 x DP=3, zero idle).
    """
    available = gpu_count - 1
    tp_raw = hyperparameters.get("vllm_tensor_parallel_size")
    dp_raw = hyperparameters.get("vllm_data_parallel_size")
    if tp_raw is None and dp_raw is None:
        devices = _policy_server_devices(gpu_count)
        return devices, len(devices), 1

    dp = int(dp_raw) if dp_raw is not None else 1
    if dp < 1 or dp > available:
        raise ValueError(
            f"vllm_data_parallel_size must be in 1..{available} "
            f"(gpu_count-1; GPU 0 is the trainer's), got {dp}"
        )
    if tp_raw is not None:
        tp = int(tp_raw)
        if tp < 1:
            raise ValueError(f"vllm_tensor_parallel_size must be >= 1, got {tp}")
    else:
        tp = 1
        while tp * 2 <= available // dp:
            tp *= 2
    if tp * dp > available:
        raise ValueError(
            f"vllm_tensor_parallel_size * vllm_data_parallel_size = {tp}*{dp} "
            f"exceeds the {available} GPUs available to the policy server "
            f"(gpu_count-1; GPU 0 is the trainer's)"
        )
    return list(range(1, 1 + tp * dp)), tp, dp


def _launch_policy_server(
    *,
    base_model: str,
    devices: list[int],
    hyperparameters: dict[str, Any],
    log: logging.Logger,
    tensor_parallel_size: int | None = None,
    data_parallel_size: int = 1,
) -> tuple[Any, str]:
    """Start TRL's vllm-serve as the policy server and return (proc, base_url).

    vllm-serve is a vLLM OpenAI-compatible server PLUS the weight-sync
    endpoints GRPOTrainer's server mode pushes into every step — that push is
    what guarantees rollouts always run against the CURRENT policy (the
    stuck-at-policy-v0 deadlock is the failure mode this design exists to
    avoid). One server fronts both the trainer and the harness proxy.

    `devices` are the CUDA ordinals the server owns EXCLUSIVELY (pinned via
    CUDA_VISIBLE_DEVICES on the subprocess). They must be disjoint from the
    trainer's — TRL's weight-sync communicator hard-rejects the trainer and a
    vLLM worker sharing a device.
    """
    import subprocess
    import sys

    if importlib.util.find_spec("trl.scripts.vllm_serve") is None:
        raise ValueError(
            "grpo_harness requires a TRL build that ships trl.scripts.vllm_serve; "
            "upgrade trl on the worker image"
        )
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    cmd = [
        sys.executable, "-m", "trl.scripts.vllm_serve",
        "--model", base_model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tensor_parallel_size",
        str(tensor_parallel_size if tensor_parallel_size is not None else len(devices)),
        # The server owns its devices outright (device split), so vLLM can
        # take most of the card; 0.85 leaves room for the CUDA context and
        # NCCL buffers. Overridable via vllm_gpu_memory_utilization.
        "--gpu_memory_utilization",
        str(hyperparameters.get("vllm_gpu_memory_utilization", 0.85)),
    ]
    # Only emitted when >1 so the default command stays byte-identical to the
    # pre-DP launch. TRL chunks /chat/ requests across replicas and includes
    # every replica in the weight-sync group (world = TP*DP + trainer).
    if data_parallel_size > 1:
        cmd += ["--data_parallel_size", str(data_parallel_size)]
    if hyperparameters.get("max_model_len"):
        cmd += ["--max_model_len", str(hyperparameters["max_model_len"])]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(str(d) for d in devices)}
    log.info(
        "launching policy server on CUDA devices %s: %s",
        env["CUDA_VISIBLE_DEVICES"],
        " ".join(cmd),
    )
    proc = subprocess.Popen(cmd, env=env)
    base_url = f"http://127.0.0.1:{port}"
    _wait_for_policy_server(base_url, proc)
    log.info("policy server healthy at %s", base_url)
    return proc, base_url


def build_rollout_batch(
    *,
    results: list[Any],
    pad_token_id: int = 0,
) -> dict[str, Any]:
    """Convert one step's RolloutResults into TRL's rollout_func output dict.

    Contract (huggingface/trl#5122, verified against trl main): required keys
    prompt_ids / completion_ids / logprobs; `env_mask` is consumed as the
    per-completion-token tool mask (1 = model token, trained; 0 = external),
    which is exactly our assistant-only loss mask over the completion region;
    any other keys are forwarded to the reward functions, so `harness_reward`
    carries the scores we computed trajectory-side.

    Failed rollouts stay in the batch as fully-masked single-token sentinels:
    GRPO groups are fixed-size (num_generations per prompt, prompt-major
    order), and a dropped entry would misalign every advantage in the step.
    A zero-mask sentinel contributes no gradient and reward 0.
    """
    prompt_ids: list[list[int]] = []
    completion_ids: list[list[int]] = []
    logprobs: list[list[float]] = []
    env_mask: list[list[int]] = []
    rewards: list[float] = []

    for result in results:
        episode = result.episode
        if result.status != "completed" or episode is None or result.reward is None:
            prompt_ids.append([pad_token_id])
            completion_ids.append([pad_token_id])
            logprobs.append([0.0])
            env_mask.append([0])
            rewards.append(0.0)
            continue
        # The first prompt is exactly the first span's prompt: everything up
        # to the first trained (mask=1) token.
        if 1 in episode.loss_mask:
            first_prompt_len = episode.loss_mask.index(1)
        else:
            first_prompt_len = len(episode.input_ids)
        prompt_ids.append(episode.input_ids[:first_prompt_len])
        completion_ids.append(episode.input_ids[first_prompt_len:])
        logprobs.append(episode.logprobs[first_prompt_len:])
        env_mask.append(episode.loss_mask[first_prompt_len:])
        rewards.append(float(result.reward))

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": env_mask,
        "harness_reward": rewards,
    }


def harness_reward_adapter(prompts, completions, harness_reward=None, **kwargs):
    """TRL reward_func shim: the trajectory-side reward was already computed
    by the uploaded reward function inside rollout_func (TRL only sees the
    finished completions, not the trajectory); this just forwards it through
    TRL's reward aggregation (weights, logging)."""
    if harness_reward is not None:
        return [float(r) for r in harness_reward]
    return [0.0] * len(completions)


def run_grpo_harness_training(
    job_config: dict[str, Any],
    *,
    dataset_path: str | None,
    reward_path: str | None = None,
    reward_paths: list[str] | None = None,
    logger: logging.Logger | None = None,
    progress_fn: Callable[[int, int, dict], None] | None = None,
    harness_code_dir: str | None = None,
    trajectory_sink: Callable[[int, list[Any]], None] | None = None,
    unsupported_reward_formats: Mapping[str, str] | None = None,
    # VS-393: upload one finished checkpoint dir (dir, step). Supplied by the
    # adapter (worker_agent.upload_step_checkpoint); None disables uploading.
    checkpoint_upload_fn: Callable[[str, int], None] | None = None,
    event_fn: Callable[[str, str, dict], None] | None = None,
) -> dict[str, Any]:
    """Harness-in-the-loop GRPO: the user's unmodified agent harness
    drives multi-turn rollouts against the in-training policy.

    One box, colocated: TRL vllm-serve hosts the policy (weight-synced every
    step by GRPOTrainer's server mode); the trajectory proxy captures token
    ids + logprobs per harness request; the rollout runner execs the harness
    N times per task (the GRPO group) sandboxed; the uploaded reward function
    scores each finished trajectory. TRL owns the update via its rollout_func
    seam — our adapter is the only TRL-version-sensitive surface, everything
    below it (proxy/renderer/runner) is contract-independent and unit-tested.
    """
    log = logger or logging.getLogger("veri.training_runtime")
    job_id = job_config["job_id"]
    base_model = job_config["base_model"]
    hyperparameters = job_config["hyperparameters"]

    from trl import GRPOConfig, GRPOTrainer

    if "rollout_func" not in inspect.signature(GRPOTrainer.__init__).parameters:
        raise ValueError(
            "grpo_harness requires a TRL version with rollout_func support "
            "(huggingface/trl#5122); the installed trl predates it. Upgrade "
            "trl on the worker image."
        )

    wandb_enabled = _setup_wandb(job_config)

    paths = reward_paths or ([reward_path] if reward_path else [])
    if not paths:
        raise ValueError("grpo_harness requires at least one reward function")
    reward_fns = load_reward_functions(
        paths,
        reward_format=job_config.get("reward_format", "trl"),
        unsupported_formats=unsupported_reward_formats,
        logger=log,
    )
    reward_weights = job_config.get("reward_weights")

    rows = resolve_training_rows(
        dataset_path=dataset_path,
        dataset_config=job_config.get("dataset"),
        snapshot_upload_url=job_config.get("dataset_snapshot_upload_url"),
    )
    validate_rows_for_method(rows, "grpo_harness")

    script = job_config.get("script") or {}
    if not script.get("entrypoint"):
        raise ValueError("grpo_harness requires a harness entrypoint (script config)")

    from datasets import Dataset

    try:
        # Repo layout (tests, dev checkouts).
        from veri_runner.harness_rollout import HarnessRunner, HarnessSpec
        from veri_runner.trajectory_proxy import SpanStore, TrajectoryProxy
    except ImportError:
        # Worker layout: the AMI / hot-patch drop everything flat in
        # /opt/veri (no runner package), same as `from training_runtime import`.
        from harness_rollout import HarnessRunner, HarnessSpec
        from trajectory_proxy import SpanStore, TrajectoryProxy

    spec = HarnessSpec(
        entrypoint=script["entrypoint"],
        base_image=script.get("base_image", "veri/base"),
        code_dir=harness_code_dir,
        deps=script.get("deps") or {},
        env=script.get("env") or {},
        protocol=hyperparameters.get("harness_protocol", "openai"),
    )

    gpu_count = int(job_config.get("gpu_count", 1) or 1)
    if gpu_count < 2:
        raise ValueError(
            "grpo_harness requires >= 2 GPUs on one node: TRL's weight-sync "
            "communicator rejects the trainer and the vLLM policy server "
            "sharing a CUDA device. Resubmit with e.g. L4-24GB x4 (aws) or "
            "H100-80GB x2 (vast)."
        )
    # Device split (forced by the TRL guard above): the
    # trainer takes GPU 0, the policy server takes GPUs 1..TP where TP is the
    # largest power of two <= n-1 (see _policy_server_devices); any remainder
    # idles rather than crashing vLLM on an illegal tensor_parallel_size.
    # The trainer pin must land before this process first touches CUDA —
    # model load below is the first user; the server subprocess gets its own
    # explicit CUDA_VISIBLE_DEVICES from _launch_policy_server.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    server_devices, server_tp, server_dp = _policy_server_layout(
        gpu_count, hyperparameters
    )
    if len(server_devices) < gpu_count - 1:
        log.warning(
            "policy server takes %d of %d non-trainer GPUs (TP=%d x DP=%d; TP must "
            "divide the model's attention heads); GPUs %s idle — reclaim them with "
            "vllm_data_parallel_size / vllm_tensor_parallel_size",
            len(server_devices),
            gpu_count - 1,
            server_tp,
            server_dp,
            list(range(1 + len(server_devices), gpu_count)),
        )
    server_proc, policy_url = _launch_policy_server(
        base_model=base_model,
        devices=server_devices,
        hyperparameters=hyperparameters,
        log=log,
        tensor_parallel_size=server_tp,
        data_parallel_size=server_dp,
    )
    # Model + tokenizer load before the proxy: the proxy decodes completion
    # ids into the OpenAI-shaped response text (TRL's /chat/ returns ids only).
    t0 = time.time()
    model, tokenizer = _load_model_and_tokenizer(base_model, hyperparameters, log)
    log.info("Model loaded in %.1fs", time.time() - t0)
    job_config["checkpoint"] = apply_checkpoint_disk_guard(
        job_config.get("checkpoint"), model, checkpoint_output_root(job_config), log, event_fn
    )

    trajectory_root = str(Path(checkpoint_output_root(job_config)) / job_id / "trajectories")
    span_store = SpanStore(str(Path(trajectory_root) / "spans"))
    proxy = TrajectoryProxy(
        upstream_url=policy_url,
        store=span_store,
        tokenizer=tokenizer,
        # Thinking off by default: reasoning models (Qwen3) strip <think>
        # blocks from prior turns on re-render, which breaks the delta-
        # tokenization invariant and rejects every multi-turn rollout as
        # template drift. Overridable per job.
        chat_template_kwargs=hyperparameters.get(
            "chat_template_kwargs", {"enable_thinking": False}
        ),
    )
    runner = HarnessRunner(
        spec=spec,
        proxy=proxy,
        span_store=span_store,
        reward_fns=reward_fns,
        reward_weights=reward_weights,
        trajectory_dir=trajectory_root,
        policy_model=base_model,
        sandbox=bool(job_config.get("sandbox_mode", True)),
        max_turns=hyperparameters.get("max_turns", 40),
        timeout_s=hyperparameters.get("harness_timeout_s", 900),
        max_parallel=hyperparameters.get("max_parallel_rollouts", 4),
    )

    # rollout_func receives prompts only; recover the full dataset row (extra
    # columns feed the reward function) via a prompt-keyed map. Duplicate
    # prompts map to identical rows, so any match is the right one.
    def _row_key(prompt: Any) -> str:
        return json.dumps(prompt, sort_keys=True, ensure_ascii=False)

    rows_by_prompt = {_row_key(row["prompt"]): row for row in rows}
    group_size = hyperparameters.get("rollouts_per_prompt", 8)

    def rollout_func(prompts, trainer):
        policy_step = trainer.state.global_step
        # TRL's RepeatSampler hands rollout_func each task num_generations
        # times, prompt-major (t0,t0,..,t1,t1,..). Treating every entry as its
        # own task ran group_size rollouts PER DUPLICATE (group_size^2 per
        # task) and returned a batch group_size-x larger than TRL's own
        # tensors — shuffle_sequence_dict then indexed past the short tensors
        # on the GPU (device-side assert). Collapse the
        # duplicates back to unique tasks; run_step's (task_index,
        # rollout_index) sort re-emits results in TRL's duplicated order.
        if len(prompts) % group_size != 0:
            raise ValueError(
                f"rollout_func got {len(prompts)} prompts, not a multiple of "
                f"rollouts_per_prompt={group_size}; check num_generations wiring"
            )
        tasks = []
        for i in range(0, len(prompts), group_size):
            prompt = prompts[i]
            row = rows_by_prompt.get(_row_key(prompt))
            if row is None:
                row = {"prompt": prompt}
            tasks.append((i // group_size, row))
        results = runner.run_step(tasks=tasks, group_size=group_size, policy_step=policy_step)
        if len(results) != len(prompts):
            raise ValueError(
                f"rollout batch size mismatch: {len(results)} results for "
                f"{len(prompts)} prompts — a partial group would misalign "
                "every GRPO advantage in the step"
            )
        if trajectory_sink:
            try:
                trajectory_sink(policy_step, results)
            except Exception as e:
                log.warning("trajectory sink failed (non-fatal): %s", e)
        return build_rollout_batch(results=results)

    config_kwargs = build_grpo_config_kwargs(
        job_id=job_id,
        hyperparameters=hyperparameters,
        grpo_config_cls=GRPOConfig,
        output_root=checkpoint_output_root(job_config),
        wandb_enabled=wandb_enabled,
        checkpoint=job_config.get("checkpoint"),
    )
    grpo_params = inspect.signature(GRPOConfig.__init__).parameters
    config_kwargs["use_vllm"] = True
    if "vllm_mode" in grpo_params:
        config_kwargs["vllm_mode"] = "server"
    if "vllm_server_base_url" in grpo_params:
        config_kwargs["vllm_server_base_url"] = policy_url
    training_args = GRPOConfig(**config_kwargs)

    trainer_kwargs = build_trainer_kwargs(
        trainer_cls=GRPOTrainer,
        model=model,
        training_args=training_args,
        train_dataset=Dataset.from_list(rows),
        reward_fn=harness_reward_adapter,
        tokenizer=tokenizer,
    )
    trainer_kwargs["rollout_func"] = rollout_func
    trainer = GRPOTrainer(**trainer_kwargs)

    _attach_progress_callback(trainer, progress_fn)
    _attach_checkpoint_uploader(trainer, job_config, checkpoint_upload_fn, log, event_fn)
    try:
        return _train_save_finalize(
            trainer=trainer,
            tokenizer=tokenizer,
            job_config=job_config,
            wandb_enabled=wandb_enabled,
            log=log,
        )
    finally:
        proxy.stop()
        server_proc.terminate()


def prerender_chat_template_rows(
    rows: list[dict[str, Any]],
    *,
    use_unsloth: bool,
    tokenizer: Any,
    text_field: str = "text",
) -> list[dict[str, Any]]:
    """Render conversational rows to plain text on the unsloth path.

    Unsloth's dataset prep (unsloth_zoo 2026.7.4 dataset_utils.py) accepts
    pre-tokenized rows (labels/input_ids), prompt+completion rows (natively,
    conversational or plain), or a `dataset_text_field` column; anything else
    raises "Unsloth: You must specify a `formatting_func`". After
    normalize_rows + validate_rows_for_method("sft_text"), the only shape left
    in that "anything else" bucket is chat `messages` — which every
    conversational community format (ShareGPT, ChatML, OASST, Alpaca)
    normalizes into, and which vanilla TRL renders via the chat template
    automatically. So for unsloth we follow the Unsloth notebook pattern:
    apply the tokenizer's chat template here and hand the trainer a plain-text
    dataset. The chat column is dropped so TRL's conversational auto-detection
    can't re-fire on the rendered rows.
    """
    if not use_unsloth or not rows or detect_dataset_format(rows[0]) != "chatml":
        return rows
    return [
        {text_field: tokenizer.apply_chat_template(row["messages"], tokenize=False)}
        for row in rows
    ]


def run_sft_text_training(
    job_config: dict[str, Any],
    *,
    dataset_path: str | None = None,
    logger: logging.Logger | None = None,
    after_model_load: Callable[[], None] | None = None,
    progress_fn: Callable[[int, int, dict], None] | None = None,
    # VS-393: upload one finished checkpoint dir (dir, step). Supplied by the
    # adapter (worker_agent.upload_step_checkpoint); None disables uploading.
    checkpoint_upload_fn: Callable[[str, int], None] | None = None,
    # VS-476: multi-GPU child: the resolved plan + the rank-0 phase writer.
    launch_plan: dict[str, Any] | None = None,
    phase_fn: Callable[[str, int], None] | None = None,
    event_fn: Callable[[str, str, dict], None] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Supervised fine-tuning on text via TRL SFTTrainer.

    Reward-free: the worker passes `reward_path` uniformly across methods, so we
    swallow it (and any other extra kwargs) via `**_ignored`. Reuses the shared
    model-load path, so LoRA/QLoRA apply here the same way they do for GRPO.

    Dataset rows may be plain `{"text": ...}` (the `dataset_text_field`, default
    "text") or conversational `{"messages": [...]}` — SFTTrainer applies the chat
    template automatically for the conversational form, except under unsloth
    where we pre-render it (see prerender_chat_template_rows).
    """
    log = logger or logging.getLogger("veri.training_runtime")
    job_id = job_config["job_id"]
    base_model = job_config["base_model"]
    hyperparameters = job_config["hyperparameters"]

    use_unsloth = _prepare_trl_framework(hyperparameters, log)

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    wandb_enabled = _setup_wandb(job_config)
    if wandb_enabled:
        log.info("W&B reporting enabled")

    # Rows before the model: dataset/format failures are cheap here and
    # expensive after a multi-minute weight download fail-fast.
    rows = resolve_training_rows(
        dataset_path=dataset_path,
        dataset_config=job_config.get("dataset"),
        snapshot_upload_url=job_config.get("dataset_snapshot_upload_url"),
    )
    validate_rows_for_method(rows, "sft_text")

    t0 = time.time()
    model, tokenizer = _load_model_and_tokenizer(
        base_model, hyperparameters, log, launch_plan=launch_plan
    )
    log.info("Model loaded in %.1fs", time.time() - t0)
    if after_model_load:
        after_model_load()
    job_config["checkpoint"] = apply_checkpoint_disk_guard(
        job_config.get("checkpoint"), model, checkpoint_output_root(job_config), log, event_fn,
        launch_plan=launch_plan,
    )

    text_field = hyperparameters.get("dataset_text_field", "text")
    rendered = prerender_chat_template_rows(
        rows, use_unsloth=use_unsloth, tokenizer=tokenizer, text_field=text_field,
    )
    if rendered is not rows:
        log.info(
            "Unsloth: pre-rendered chat template into '%s' for %d rows",
            text_field, len(rendered),
        )
    dataset = Dataset.from_list(rendered)

    training_args = SFTConfig(
        **build_sft_config_kwargs(
            job_id=job_id,
            hyperparameters=hyperparameters,
            sft_config_cls=SFTConfig,
            output_root=checkpoint_output_root(job_config),
            wandb_enabled=wandb_enabled,
            checkpoint=job_config.get("checkpoint"),
            launch_plan=launch_plan,
        )
    )
    trainer = SFTTrainer(
        **build_trainer_kwargs(
            trainer_cls=SFTTrainer,
            model=model,
            training_args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
    )

    _attach_progress_callback(trainer, progress_fn, phase_fn=phase_fn)
    _attach_checkpoint_uploader(trainer, job_config, checkpoint_upload_fn, log, event_fn)
    return _train_save_finalize(
        trainer=trainer,
        tokenizer=tokenizer,
        job_config=job_config,
        wandb_enabled=wandb_enabled,
        log=log,
        launch_plan=launch_plan,
        phase_fn=phase_fn,
    )


def _require_preference_columns(rows: list[dict[str, Any]]) -> None:
    """Fail fast if a DPO dataset is missing the preference columns.

    DPO trains on pairs: each row needs a `chosen` and a `rejected` completion
    (plus an optional explicit `prompt`). SFT/GRPO rows have no `rejected`, so a
    user who points an SFT-shaped dataset at DPO would otherwise hit an opaque
    error deep inside TRL after a 10-minute GPU boot. Mirrors TRL's documented
    preference format (prompt/chosen/rejected).
    """
    if not rows:
        raise ValueError("DPO dataset is empty")
    first = rows[0]
    missing = [c for c in ("chosen", "rejected") if c not in first]
    if missing:
        raise ValueError(
            "DPO requires a preference dataset with "
            f"'chosen' and 'rejected' columns; row is missing {missing}. "
            "Each row needs a chosen and a rejected completion (with an optional "
            "explicit 'prompt'). See TRL's DPO preference format."
        )


def run_dpo_training(
    job_config: dict[str, Any],
    *,
    dataset_path: str | None = None,
    logger: logging.Logger | None = None,
    after_model_load: Callable[[], None] | None = None,
    progress_fn: Callable[[int, int, dict], None] | None = None,
    # VS-393: upload one finished checkpoint dir (dir, step). Supplied by the
    # adapter (worker_agent.upload_step_checkpoint); None disables uploading.
    checkpoint_upload_fn: Callable[[str, int], None] | None = None,
    # VS-476: multi-GPU child: the resolved plan + the rank-0 phase writer.
    launch_plan: dict[str, Any] | None = None,
    phase_fn: Callable[[str, int], None] | None = None,
    event_fn: Callable[[str, str, dict], None] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Direct Preference Optimization via TRL DPOTrainer.

    Reward-free like SFT: the worker passes `reward_path` uniformly across
    methods, so we swallow it (and any other extra kwargs) via `**_ignored`.
    Reuses the shared model-load path, so LoRA/QLoRA apply here the same way
    they do for GRPO/SFT.

    Dataset rows are preference pairs: `{"prompt", "chosen", "rejected"}` (or the
    implicit-prompt / conversational variants). DPOTrainer derives the reference
    policy from the initial model and applies the chat template automatically for
    conversational rows.
    """
    log = logger or logging.getLogger("veri.training_runtime")
    job_id = job_config["job_id"]
    base_model = job_config["base_model"]
    hyperparameters = job_config["hyperparameters"]

    _prepare_trl_framework(hyperparameters, log)

    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    wandb_enabled = _setup_wandb(job_config)
    if wandb_enabled:
        log.info("W&B reporting enabled")

    # Rows before the model: dataset/format failures are cheap here and
    # expensive after a multi-minute weight download fail-fast.
    rows = resolve_training_rows(
        dataset_path=dataset_path,
        dataset_config=job_config.get("dataset"),
        snapshot_upload_url=job_config.get("dataset_snapshot_upload_url"),
    )
    validate_rows_for_method(rows, "dpo")
    dataset = Dataset.from_list(rows)

    t0 = time.time()
    model, tokenizer = _load_model_and_tokenizer(
        base_model, hyperparameters, log, launch_plan=launch_plan
    )
    log.info("Model loaded in %.1fs", time.time() - t0)
    if after_model_load:
        after_model_load()
    job_config["checkpoint"] = apply_checkpoint_disk_guard(
        job_config.get("checkpoint"), model, checkpoint_output_root(job_config), log, event_fn,
        launch_plan=launch_plan,
    )

    training_args = DPOConfig(
        **build_dpo_config_kwargs(
            job_id=job_id,
            hyperparameters=hyperparameters,
            dpo_config_cls=DPOConfig,
            output_root=checkpoint_output_root(job_config),
            wandb_enabled=wandb_enabled,
            checkpoint=job_config.get("checkpoint"),
            launch_plan=launch_plan,
        )
    )
    trainer = DPOTrainer(
        **build_trainer_kwargs(
            trainer_cls=DPOTrainer,
            model=model,
            training_args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
    )

    _attach_progress_callback(trainer, progress_fn, phase_fn=phase_fn)
    _attach_checkpoint_uploader(trainer, job_config, checkpoint_upload_fn, log, event_fn)
    return _train_save_finalize(
        trainer=trainer,
        tokenizer=tokenizer,
        job_config=job_config,
        wandb_enabled=wandb_enabled,
        log=log,
        launch_plan=launch_plan,
        phase_fn=phase_fn,
    )


def _is_rocm() -> bool:
    """True when torch is an AMD/ROCm build (hip is set on ROCm wheels only)."""
    try:
        import torch
        return getattr(torch.version, "hip", None) is not None
    except Exception:
        return False


def _liger_available() -> bool:
    """True when liger-kernel is importable. It is installed on the AWS worker
    AMI, the Vast and GCP worker images and by the ROCm bootstrap; images built
    before it was added lack it."""
    import importlib.util
    return importlib.util.find_spec("liger_kernel") is not None


# DPO loss types liger-kernel 0.8.4's fused DPO loss implements
# (LigerFusedLinearDPOLoss._SUPPORTED_LOSS_TYPES). TRL 1.7.1's other loss types
# (ipo, aot, aot_unpaired, sft, sigmoid_norm) raise at trainer init with liger on.
LIGER_DPO_LOSS_TYPES = (
    "sigmoid", "hinge", "exo_pair", "nca_pair", "robust",
    "bco_pair", "sppo_hard", "apo_zero", "apo_down", "discopop",
)


def liger_incompatibility(
    method: str, gpu_count: int, hyperparameters: dict[str, Any]
) -> str | None:
    """Why use_liger cannot run with this job, or None when it can.

    Each rejected combination crashes on the GPU or at trainer init on the
    worker stack (torch 2.9.1, trl 1.7.1, liger-kernel 0.8.4). The control
    plane rejects the same set at submit (validate_liger in
    control_plane_new/src/api/training_jobs.rs); keep the two in sync."""
    if gpu_count > 1:
        return (
            f"use_liger is supported on single-GPU jobs only (got gpu_count {gpu_count}): "
            "liger-kernel's Triton kernels fail when the model is split across GPUs. "
            "Use gpu_count 1 or drop use_liger."
        )
    if hyperparameters.get("use_unsloth"):
        return (
            "use_liger cannot be combined with use_unsloth: both patch the same "
            "model modules. Pick one."
        )
    if method == "dpo":
        if hyperparameters.get("lora_rank") is not None or hyperparameters.get("load_in_4bit"):
            return (
                "use_liger with DPO supports full fine-tunes only, not lora_rank or "
                "load_in_4bit (TRL's liger DPO loss does not support PEFT models). "
                "Drop use_liger or the LoRA settings."
            )
        loss_type = hyperparameters.get("loss_type", "sigmoid")
        if loss_type not in LIGER_DPO_LOSS_TYPES:
            return (
                f"use_liger does not support DPO loss_type '{loss_type}'. With use_liger, "
                f"loss_type must be one of: {', '.join(LIGER_DPO_LOSS_TYPES)}. "
                "Drop use_liger or pick one of those."
            )
    return None


def _maybe_enable_tunableop(hyperparameters: dict[str, Any]) -> None:
    """Opt-in TunableOp: rocBLAS/hipBLASLt GEMM autotuning via
    torch.cuda.tunable. Off by default because tuning happens at runtime —
    variable-length batches re-tune every new GEMM shape, which can cost more
    than it saves on short jobs. Uses the runtime API (not env vars) so the
    flag needs no bootstrap/systemd plumbing."""
    if not hyperparameters.get("tunableop"):
        return
    import torch
    torch.cuda.tunable.enable(True)


def _maybe_prefer_hipblaslt(hyperparameters: dict[str, Any]) -> None:
    """Opt-in hipBLASLt GEMM routing on ROCm. Opt-in — NOT default —
    because always-on shipped once and benchmarked 25% slower on a small-GEMM
    workload (0.6B, batch size 1); it is expected to pay off on
    large-GEMM shapes. Set before the first BLAS dispatch (torch reads the
    env when the BLAS backend is first selected, so setting it here — before
    any model is loaded — is early enough)."""
    if hyperparameters.get("hipblaslt") and _is_rocm():
        os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"


def run_training(
    job_config: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """Dispatch to the correct training runtime based on method."""
    method = job_config.get("method", "grpo")

    # CUDA-only options must fail fast on ROCm, before any dispatch arm starts
    # loading models. unsloth and bitsandbytes (load_in_4bit) have no ROCm
    # support; on NVIDIA both stay allowed (opt-in boundary).
    hyperparameters = job_config.get("hyperparameters", {}) or {}
    if _is_rocm():
        if hyperparameters.get("use_unsloth"):
            raise ValueError("use_unsloth is not supported on ROCm (unsloth is CUDA-only)")
        if hyperparameters.get("load_in_4bit"):
            raise ValueError(
                "load_in_4bit is not supported on ROCm (bitsandbytes is CUDA-only)"
            )

    # use_liger needs the liger-kernel package on the box (baked into the AWS
    # worker AMI, the Vast and GCP images, and installed by the ROCm bootstrap;
    # older images lack it). Fail fast with the reason, not an opaque
    # transformers ImportError after the model download.
    if hyperparameters.get("use_liger") and not _liger_available():
        raise ValueError(
            "use_liger requires the liger-kernel package, which is not "
            "installed on this worker image"
        )
    # Then the configs liger crashes on. The control plane rejects these at
    # submit; this repeats the check for configs that did not pass through it.
    if hyperparameters.get("use_liger"):
        reason = liger_incompatibility(
            method, int(job_config.get("gpu_count", 1) or 1), hyperparameters
        )
        if reason:
            raise ValueError(reason)

    _maybe_enable_tunableop(hyperparameters)
    _maybe_prefer_hipblaslt(hyperparameters)

    # The methods the worker can run are dispatched by these hardcoded arms;
    # submit-time validation of the method name happens in the control-plane
    # API, so an unknown method here means a version skew between the API and
    # the worker image.
    if method == "grpo":
        return run_grpo_training(job_config, **kwargs)
    if method == "grpo_harness":
        return run_grpo_harness_training(job_config, **kwargs)
    if method == "sft_text":
        return run_sft_text_training(job_config, **kwargs)
    if method == "dpo":
        return run_dpo_training(job_config, **kwargs)
    if method == "sft_video_gen":
        try:
            # Repo layout (dev checkouts) then worker layout (flat /opt/veri).
            try:
                from runner.sft_video_gen_runtime import run_sft_video_gen_training
            except ImportError:
                from sft_video_gen_runtime import run_sft_video_gen_training
        except ImportError:
            raise ValueError(
                "sft_video_gen requires the sft_video_gen_runtime module, "
                "which ships on Veri worker images but not in this checkout"
            )
        return run_sft_video_gen_training(job_config, **kwargs)
    raise ValueError(f"Unknown training method: {method}")


# ---- Multi-GPU managed launch (VS-476, phase 1) ----
#
# gpu_count > 1 on sft_text / dpo runs one process per GPU under
# `accelerate launch` (DDP when the model states fit one GPU, FSDP2 ZeRO-3
# when they do not, ZeRO-2 on request). The worker agent (the parent, which
# never touches CUDA) resolves the plan, writes the accelerate config and a
# secret-free child config, spawns the launcher and reads the files the ranks
# leave behind: progress.jsonl, result.json, checkpoint-N/, final/ and the
# torchelastic per-rank error files. Knob names follow the Ultra-Scale
# Playbook (zero_stage, *_parallel_size, micro_batch_size,
# gradient_accumulation_steps, global_batch_size, gradient_checkpointing,
# precision, cpu_offload). Single-GPU jobs never come through here.

PRECISIONS = ("bf16", "bf16_mixed", "fp32")
# Share of GPU memory the model states (weights, grads, optimizer, reference
# model) may take; activations, the CUDA context and NCCL buffers get the rest.
MEMORY_BUDGET_FRACTION = 0.7
_TORCH_DTYPE_BYTES = {
    "bfloat16": 2, "float16": 2, "half": 2, "float32": 4, "float": 4, "float64": 8,
    "int8": 1, "uint8": 1, "float8_e4m3fn": 1, "float8_e5m2": 1,
}
# Per-GPU memory the CUDA context and NCCL buffers take before the model
# (ml-engineering: torch.distributed ~1-2 GiB per GPU at init, invisible to the
# torch profiler). Subtracted from NVML total before the 0.7 budget.
GPU_RESERVED_BYTES = 2 * 1024**3
# Rank-0 FULL_STATE_DICT write rate on a stock gp3 root (Q-D); the process-group
# timeout and the stall watchdog stretch around 2 x this estimate.
CHECKPOINT_WRITE_BYTES_PER_S = 100 * 10**6
# The ONLY keys the ranks get from the job config (allowlist, so a new secret-
# carrying field can never leak by omission). Rows arrive via dataset_path, the
# W&B key via the env, credentials and callbacks stay in the parent.
_CHILD_CONFIG_ALLOW = (
    "job_id", "method", "base_model", "hyperparameters", "checkpoint", "output_name",
    "wandb_project", "gpu_count", "gpu_type", "provider", "num_nodes", "system_prompt",
    # C2: {checkpoint_id, step, local_dir}; the manifest URL is harmless but the
    # parent already downloaded the files, so the ranks only need local_dir.
    "resume",
)


def _safetensors_param_count(path: Path) -> int | None:
    """Parameter count from a .safetensors header (8-byte little-endian header
    length, then JSON), without loading any tensor."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
    except (OSError, ValueError):
        return None
    total = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for dim in entry.get("shape", []):
            count *= int(dim)
        total += count
    return total or None


def _formula_param_count(cfg: dict[str, Any]) -> int | None:
    """Playbook estimate N = h*v + L(12h^2 + 13h) + 2h (PB, Memory for weights,
    grads and optimizer states) from config.json."""
    try:
        h = int(cfg["hidden_size"])
        v = int(cfg["vocab_size"])
        layers = int(cfg["num_hidden_layers"])
    except (KeyError, TypeError, ValueError):
        return None
    return h * v + layers * (12 * h * h + 13 * h) + 2 * h


def read_model_meta(model_dir: str) -> dict[str, Any]:
    """What the plan resolver needs from a downloaded model: the exact parameter
    count (safetensors index total_size / dtype bytes, else a single-file
    header, else the playbook formula) and the config fields the phase-3
    preflight will check. Never raises; an unknown count is None."""
    root = Path(model_dir)
    cfg: dict[str, Any] = {}
    try:
        cfg = json.loads((root / "config.json").read_text())
    except (OSError, ValueError):
        pass
    dtype = str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16")
    meta: dict[str, Any] = {
        "dtype": dtype,
        "config": {
            k: cfg.get(k)
            for k in ("architectures", "hidden_size", "num_hidden_layers", "vocab_size",
                      "num_attention_heads", "num_key_value_heads", "tie_word_embeddings")
        },
    }
    index = root / "model.safetensors.index.json"
    if index.exists():
        try:
            total = int(json.loads(index.read_text()).get("metadata", {}).get("total_size") or 0)
        except (OSError, ValueError):
            total = 0
        if total:
            meta.update(param_count=total // _TORCH_DTYPE_BYTES.get(dtype, 2),
                        param_count_source="safetensors_index")
            return meta
    single = root / "model.safetensors"
    if single.exists():
        n = _safetensors_param_count(single)
        if n:
            meta.update(param_count=n, param_count_source="safetensors_header")
            return meta
    n = _formula_param_count(cfg)
    if n:
        meta.update(param_count=n, param_count_source="config_formula")
        return meta
    meta.update(param_count=None, param_count_source="unknown")
    return meta


def resolve_parallel_plan(
    hyperparameters: dict[str, Any] | None,
    *,
    method: str,
    gpu_count: int,
    gpu_memory_bytes: int,
    model_meta: dict[str, Any],
    host_memory_bytes: int | None = None,
) -> dict[str, Any]:
    """Pure: the playbook knobs + model size + GPU memory -> the launch plan,
    with a decision trace (spec section 3.1).

    Model states S: full FT 8N (bf16) or 16N (bf16_mixed / fp32); + 2N for a
    reference model (DPO full FT, GRPO full FT with kl_coef > 0); LoRA 2N
    (frozen bf16 base; 4N when precision is fp32); QLoRA ~0.6N. Budget B =
    0.7 x (GPU memory - 2 GiB CUDA/NCCL reserve). "auto": S <= B -> stage 0
    (DDP), else 3 (FSDP2 full shard); cpu_offload forces 3. Phase 1 never
    auto-picks 2. Per GPU at stage 3 the sharded states are S/dp, but TRL wraps
    a DPO reference model as ONE FSDP unit, so its full 2N is gathered on every
    rank during its forward and is counted whole. Not fitting even at stage 3
    WARNS, never blocks; so does W full copies not fitting host RAM (every rank
    loads the whole model before sharding).
    """
    hp = dict(hyperparameters or {})
    world = int(gpu_count)
    tp = cp = pp = 1  # phase 3 lifts these
    dp = world // (tp * cp * pp)
    precision = str(hp.get("precision") or "bf16")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    lora = hp.get("lora_rank") is not None
    qlora = bool(hp.get("load_in_4bit"))
    gc = hp.get("gradient_checkpointing")
    gc = True if gc is None else bool(gc)
    cpu_offload = bool(hp.get("cpu_offload"))
    trace: list[str] = [f"world_size={world} dp={dp} tp={tp} cp={cp} pp={pp}"]
    warnings: list[str] = []

    n = model_meta.get("param_count")
    source = model_meta.get("param_count_source", "unknown")
    states: int | None = None  # everything that lives on the GPUs for the whole run
    weights: int | None = None  # the (host-RAM) copy each rank loads before sharding
    reference = 0  # a DPO / GRPO reference model: one FSDP unit, gathered whole
    checkpoint_bytes = 0  # a resumable checkpoint (weights + optimizer), rank 0 writes
    if n is not None:
        n = int(n)
        if qlora:
            states = weights = int(0.6 * n)
            trace.append(f"params={n:,} ({source}); QLoRA: model_states~0.6N={states / 1e9:.1f} GB")
        elif lora:
            weights = 4 * n if precision == "fp32" else 2 * n
            states = weights
            trace.append(
                f"params={n:,} ({source}); LoRA: frozen "
                f"{'fp32' if precision == 'fp32' else 'bf16'} base {weights // n}N="
                f"{states / 1e9:.1f} GB"
            )
        else:
            per_param = 8 if precision == "bf16" else 16
            weights = 2 * n if precision == "bf16" else 4 * n
            states = per_param * n
            checkpoint_bytes = (6 if precision == "bf16" else 12) * n
            line = (
                f"params={n:,} ({source}); full FT {precision}: "
                f"{per_param}N={states / 1e9:.1f} GB"
            )
            has_reference = method == "dpo" or (
                method == "grpo" and float(hp.get("kl_coef") or 0.0) > 0.0
            )
            if has_reference:
                reference = 2 * n
                states += reference
                line += f" + reference model 2N = {states / 1e9:.1f} GB"
            trace.append(line)
    usable = max(0, int(gpu_memory_bytes) - GPU_RESERVED_BYTES)
    budget = int(MEMORY_BUDGET_FRACTION * usable)
    trace.append(
        f"budget={budget / 1e9:.1f} GB = {MEMORY_BUDGET_FRACTION} x "
        f"({int(gpu_memory_bytes) / 1e9:.1f} GB - {GPU_RESERVED_BYTES / 1e9:.1f} GB reserve) "
        f"per GPU"
    )

    requested = hp.get("zero_stage", "auto")
    if requested is None:
        requested = "auto"
    if requested == "auto":
        if cpu_offload:
            stage, why = 3, "auto: cpu_offload requires FSDP, sharding parameters"
        elif states is None:
            stage, why = 0, "auto: parameter count unknown, data parallel by default"
            warnings.append(
                "parameter count unknown (no safetensors index, header or config.json "
                "sizes); memory fit was not checked. Set zero_stage explicitly if the "
                "model does not fit one GPU."
            )
        elif states <= budget:
            stage, why = 0, "auto: model states fit one GPU, data parallel"
        else:
            stage, why = 3, "auto: model states exceed one GPU, sharding parameters (ZeRO-3)"
    else:
        stage = int(requested)
        if stage not in (0, 2, 3):
            raise ValueError(f"zero_stage must be auto, 0, 2 or 3, got {requested!r}")
        if stage == 0 and cpu_offload:
            raise ValueError("cpu_offload requires zero_stage 2 or 3 (FSDP)")
        why = "explicit"
    backend = "ddp" if stage == 0 else "fsdp2"

    per_gpu: int | None = None
    if states is not None and weights is not None:
        sharded = states - reference
        if stage == 0:
            per_gpu = states
        elif stage == 2:
            per_gpu = (weights - 0) + (sharded - weights) // dp + reference
        else:
            per_gpu = sharded // dp + reference
        if cpu_offload:
            per_gpu = reference  # params / grads / optimizer live in host RAM
        trace.append(
            f"zero_stage={stage} ({why}); backend={backend}; "
            f"model_states_per_gpu={per_gpu / 1e9:.1f} GB"
            + (
                f" (incl. reference model {reference / 1e9:.1f} GB gathered whole)"
                if reference and stage
                else ""
            )
        )
        if per_gpu > budget:
            options = "lora_rank, more GPUs, precision"
            if not cpu_offload:
                options += ", cpu_offload"
            if stage == 0:
                options += ", zero_stage 3"
            warnings.append(
                f"estimated {per_gpu / 1e9:.1f} GB/GPU of model states > {budget / 1e9:.1f} GB "
                f"budget; expect OOM. Options: {options}"
            )
        host_needed = weights * world + (states if cpu_offload else 0)
        if host_memory_bytes and host_needed > 0.8 * int(host_memory_bytes):
            warnings.append(
                f"every rank loads the full model before sharding: {world} x "
                f"{weights / 1e9:.1f} GB{' + offloaded states' if cpu_offload else ''} = "
                f"{host_needed / 1e9:.1f} GB of host RAM > 80% of "
                f"{int(host_memory_bytes) / 1e9:.1f} GB; expect the kernel OOM killer. "
                f"Options: fewer GPUs per node, lora_rank, a smaller base model"
            )
    else:
        trace.append(f"zero_stage={stage} ({why}); backend={backend}")
    if backend == "fsdp2" and checkpoint_bytes:
        checkpoint_bytes += weights or 0  # HF also writes pytorch_model_fsdp.bin
    timeout_s = int(max(600.0, 2.0 * checkpoint_bytes / CHECKPOINT_WRITE_BYTES_PER_S))

    mbs = int(hp.get("micro_batch_size") or hp.get("batch_size") or 1)
    ga_raw = hp.get("gradient_accumulation_steps")
    gbs_raw = hp.get("global_batch_size")
    if ga_raw is not None:
        ga = int(ga_raw)
    elif gbs_raw:
        ga = max(1, int(gbs_raw) // (mbs * dp))
    else:
        ga = 1
    gbs = mbs * ga * dp
    trace.append(
        f"micro_batch_size={mbs} gradient_accumulation_steps={ga} global_batch_size={gbs} "
        f"(= mbs x ga x dp) gradient_checkpointing={gc} cpu_offload={cpu_offload}"
    )
    summary = (
        f"plan: dp={dp} zero_stage={stage} tp={tp} cp={cp} mbs={mbs} grad_acc={ga} gbs={gbs} "
        f"precision={precision} gradient_checkpointing={'on' if gc else 'off'}"
    )
    if per_gpu is not None:
        summary += f" ({why.split(':')[0]}: {per_gpu / 1e9:.1f} GB states/GPU)"

    return {
        "world_size": world,
        "dp": dp,
        "tp": tp,
        "cp": cp,
        "pp": pp,
        "zero_stage": stage,
        "zero_stage_requested": requested,
        "backend": backend,
        "micro_batch_size": mbs,
        "gradient_accumulation_steps": ga,
        "global_batch_size": gbs,
        "gradient_checkpointing": gc,
        "precision": precision,
        "cpu_offload": cpu_offload,
        "lora": lora,
        "qlora": qlora,
        "ddp_timeout_s": timeout_s,
        "estimate": {
            "param_count": n,
            "param_count_source": source,
            "model_states_bytes": states,
            "reference_model_bytes": reference,
            "per_gpu_bytes": per_gpu,
            "budget_bytes": budget,
            "gpu_memory_bytes": int(gpu_memory_bytes),
            "host_memory_bytes": int(host_memory_bytes) if host_memory_bytes else None,
            "checkpoint_bytes": checkpoint_bytes,
        },
        "trace": trace,
        "warnings": warnings,
        "summary": summary,
    }


def render_accelerate_config(plan: dict[str, Any]) -> str:
    """The `accelerate launch --config_file` document (JSON; accelerate reads
    .json or .yaml). Shape of TRL 1.7.1's accelerate_configs/multi_gpu.yaml and
    fsdp2.yaml on accelerate 1.15 (minus cpu_ram_efficient_loading, see below):
    ZeRO-2 = FSDP2 without resharding after forward, ZeRO-3 = with.
    `mixed_precision` is "bf16" only for bf16_mixed: for pure bf16 accelerate
    would otherwise upcast the sharded params to fp32.
    """
    cfg: dict[str, Any] = {
        "compute_environment": "LOCAL_MACHINE",
        "debug": False,
        "downcast_bf16": "no",
        "enable_cpu_affinity": False,
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": "bf16" if plan["precision"] == "bf16_mixed" else "no",
        "num_machines": 1,
        "num_processes": int(plan["world_size"]),
        "rdzv_backend": "static",
        "same_network": True,
        "use_cpu": False,
    }
    if plan["backend"] == "ddp":
        cfg["distributed_type"] = "MULTI_GPU"
        cfg["gpu_ids"] = "all"
    else:
        cfg["distributed_type"] = "FSDP"
        cfg["fsdp_config"] = {
            "fsdp_version": 2,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_reshard_after_forward": int(plan["zero_stage"]) == 3,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            # OFF on purpose. The policy loads before the process group exists,
            # so transformers never takes the rank-0-only path for it; but a DPO
            # reference model is built AFTER the group is up, and with this on
            # ranks 1..W-1 would skip loading its weights (modeling_utils: "Skip
            # it with fsdp on ranks other than 0") while TRL's prepare_fsdp
            # does no broadcast: silently wrong DPO. Every rank loads every model.
            "fsdp_cpu_ram_efficient_loading": False,
            "fsdp_offload_params": bool(plan["cpu_offload"]),
            "fsdp_activation_checkpointing": bool(plan["gradient_checkpointing"]),
        }
    return json.dumps(cfg, indent=2)


def build_launch_argv(
    plan: dict[str, Any],
    *,
    config_path: str,
    child_config_path: str,
    log_dir: str,
    script_path: str,
    python: str | None = None,
) -> list[str]:
    """`python -m accelerate.commands.launch ...` for the parent to spawn. No
    restarts (the parent owns retries), a 5 s failure sweep, every line
    prefixed with its rank (`--tee 3`) and per-rank logs + torchelastic error
    files under `log_dir`."""
    return [
        python or sys.executable, "-m", "accelerate.commands.launch",
        "--config_file", config_path,
        "--num_processes", str(int(plan["world_size"])),
        "--num_machines", "1",
        "--machine_rank", "0",
        "--max_restarts", "0",
        "--monitor_interval", "5",
        "--tee", "3",
        "--log_dir", log_dir,
        script_path,
        "--child-config", child_config_path,
    ]


def build_child_config(
    job_config: dict[str, Any],
    plan: dict[str, Any],
    *,
    dataset_path: str,
    ranks_dir: str,
    progress_path: str,
) -> dict[str, Any]:
    """The file the ranks read: an allowlisted slice of the job config (no
    secrets, presigned URLs, callbacks, dataset/checkpoint credentials: the
    parent keeps those), the plan, and the file contract (rows in,
    progress/result/stacks out)."""
    scrubbed = {k: job_config[k] for k in _CHILD_CONFIG_ALLOW if k in job_config}
    return {
        "job_config": scrubbed,
        "plan": plan,
        "dataset_path": dataset_path,
        "ranks_dir": ranks_dir,
        "progress_path": progress_path,
        "result_path": str(Path(progress_path).parent / "result.json"),
    }


def build_child_env(
    job_config: dict[str, Any],
    plan: dict[str, Any],
    *,
    ranks_dir: str,
    cpu_count: int,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Environment for the launcher and its ranks (spec section 4 + the
    ml-engineering research, 10.2): offline Hub (the parent prefetched), one
    thread pool share per rank, NCCL warnings only unless `debug_nccl`, the
    flight recorder + desync report so a timeout names the stuck collective,
    and the W&B key (in the env only, never in the child config)."""
    env = dict(os.environ if base_env is None else base_env)
    world = max(1, int(plan["world_size"]))
    env.update({
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(max(1, int(cpu_count) // world)),
        "NCCL_DEBUG": "WARN",
        # Watches NCCL's own watchdog thread (not a dead peer): keep, see 10.2 N4.
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "120",
        # Off by default on torch 2.9.1; without it a timeout logs "stack trace
        # of the failed collective not found" (10.2 H1).
        "TORCH_FR_BUFFER_SIZE": "2000",
        "TORCH_NCCL_DESYNC_DEBUG": "1",
        "TORCH_FR_DUMP_TEMP_FILE": str(Path(ranks_dir) / "fr_"),
    })
    if (job_config.get("hyperparameters") or {}).get("debug_nccl"):
        env["NCCL_DEBUG"] = "INFO"
        env["NCCL_DEBUG_FILE"] = str(Path(ranks_dir) / "nccl.%h.%p.log")
    if job_config.get("wandb_api_key"):
        env["WANDB_API_KEY"] = str(job_config["wandb_api_key"])
    return env


class RankZeroProgressFile:
    """Rank 0's progress channel to the parent: one JSON line per log step and
    per phase change, appended and flushed (the parent tails it)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def _write(self, record: dict[str, Any]) -> None:
        record = {"t": time.time(), "rank": 0, **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def step(self, step: int, total_steps: int | None, metrics: dict[str, float]) -> None:
        self._write({"step": int(step), "total_steps": total_steps, "metrics": metrics})

    def phase(self, phase: str, step: int) -> None:
        self._write({"phase": phase, "step": int(step)})

    def event(self, event_type: str, message: str, metadata: dict[str, Any]) -> None:
        """C0: a job event raised inside a rank (checkpoint_disabled_disk,
        checkpoint_upload_skipped); the parent forwards it to the control
        plane with the worker's callback token."""
        self._write({"event": event_type, "message": message, "metadata": metadata})


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _restore_wandb_key(job_config: dict[str, Any], env: Mapping[str, str]) -> None:
    """The W&B key travels in the env only (never in the child config file);
    put it back so _setup_wandb enables reporting on rank 0."""
    if env.get("WANDB_API_KEY") and not job_config.get("wandb_api_key"):
        job_config["wandb_api_key"] = env["WANDB_API_KEY"]


def cast_saved_model_dtype(model_dir: str, dtype_name: str, log: logging.Logger) -> bool:
    """Q9: a bf16_mixed / fp32 run saves fp32 weights; serving loads bf16 anyway,
    so the parent rewrites the full-model artifact in the base dtype (half the
    upload and storage). Adapter-only artifacts are left alone. CPU only."""
    root = Path(model_dir)
    if (root / "adapter_config.json").exists():
        return False
    import torch
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
    if all(p.dtype == dtype for p in model.parameters()):
        return False
    log.info("Casting the final artifact in %s to %s", model_dir, dtype_name)
    model.to(dtype).save_pretrained(model_dir)
    return True


def _force_fp32_grad_reduce(trainer: Any, plan: dict[str, Any]) -> bool:
    """FSDP2 + bf16_mixed: reduce-scatter gradients in fp32 (Q-A). accelerate
    1.15 maps mixed_precision bf16 to MixedPrecisionPolicy(param bf16, reduce
    bf16) and has no config key for the reduce dtype; a policy set on the
    plugin before train() survives set_mixed_precision (override=False) and is
    what fully_shard reads. DDP bf16_mixed already reduces fp32 grads."""
    if plan.get("backend") != "fsdp2" or plan.get("precision") != "bf16_mixed":
        return False
    import torch
    from torch.distributed.fsdp import MixedPrecisionPolicy

    plugin = trainer.accelerator.state.fsdp_plugin
    plugin.mixed_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16
    )
    return True


def _gather_peak_memory() -> list[int]:
    """Peak allocated bytes of every rank (rank 0 reports them in result.json)."""
    try:
        import torch
        import torch.distributed as dist

        mine = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        if dist.is_available() and dist.is_initialized():
            out: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(out, mine)
            return [int(x) for x in out]
        return [mine]
    except Exception:  # noqa: BLE001 -- telemetry must not fail a finished run
        return []


_STACK_DUMP_FILE: Any = None


def _install_stack_dump(ranks_dir: str, rank: int) -> None:
    """SIGUSR1 -> every thread's Python stack into ranks_dir/stack_<rank>.txt
    (stdlib faulthandler; the parent signals all ranks before killing a stalled
    job, 10.2 H3). py-spy is not on the AMI yet."""
    global _STACK_DUMP_FILE
    import faulthandler
    import signal

    os.makedirs(ranks_dir, exist_ok=True)
    _STACK_DUMP_FILE = open(Path(ranks_dir) / f"stack_{rank}.txt", "w")
    faulthandler.register(signal.SIGUSR1, file=_STACK_DUMP_FILE, all_threads=True)


def _preflight_main(child: dict[str, Any], rank: int, ranks_dir: str) -> int:
    """P2 (10.2): the job's own launcher brings up the collective group once
    before any weights are downloaded: init, all_reduce(ones) == W, a short
    matmul per GPU, free memory after the barrier. Each rank writes
    ranks_dir/preflight_<rank>.json; the parent fails hardware_error on any
    rank that cannot. ~20 s."""
    import torch
    import torch.distributed as dist

    out: dict[str, Any] = {"rank": rank, "ok": False}
    try:
        from accelerate import PartialState

        # same group the Trainer will build (ACCELERATE_USE_CPU selects gloo on
        # a CPU-only host; production ranks are NCCL on CUDA)
        state = PartialState(cpu=os.environ.get("ACCELERATE_USE_CPU", "").lower() == "true")
        device = state.device
        world = state.num_processes
        t = torch.ones(1, device=device)
        dist.all_reduce(t)
        out["all_reduce"] = float(t.item())
        if int(round(float(t.item()))) != world:
            raise RuntimeError(f"all_reduce(ones) = {t.item()} != world size {world}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            out["memory_free_after_init"] = int(free)
            out["memory_total"] = int(total)
            out["nccl_version"] = list(torch.cuda.nccl.version())
            n = 4096
            a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
            b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(5):
                c = a @ b
            torch.cuda.synchronize()
            dt = time.time() - t0
            out["matmul_tflops"] = round(5 * 2 * n**3 / dt / 1e12, 1)
            del a, b, c
        # a second collective as the barrier (dist.barrier picks a device on
        # its own and fails on gloo-only hosts)
        dist.all_reduce(torch.zeros(1, device=device))
        out["ok"] = True
    except Exception as e:  # noqa: BLE001 -- the parent reads the file
        out["error"] = f"{type(e).__name__}: {e}"
    with open(Path(ranks_dir) / f"preflight_{rank}.json", "w") as f:
        json.dump(out, f)
    _destroy_process_group()
    if not out["ok"]:
        raise RuntimeError(f"preflight failed on rank {rank}: {out.get('error')}")
    return 0


def _child_main(argv: list[str] | None = None) -> int:
    """One rank of a managed multi-GPU job (launched by the worker parent)."""
    import argparse

    parser = argparse.ArgumentParser(prog="training_runtime")
    parser.add_argument("--child-config", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    with open(args.child_config) as f:
        child = json.load(f)
    job_config, plan = child["job_config"], child["plan"]
    rank = _rank()
    ranks_dir = child.get("ranks_dir") or str(Path(args.child_config).parent / "ranks")
    if args.preflight:
        os.makedirs(ranks_dir, exist_ok=True)
        return _preflight_main(child, rank, ranks_dir)
    _install_stack_dump(ranks_dir, rank)
    logging.basicConfig(
        level=logging.INFO, format=f"[rank {rank}] %(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("veri.training_runtime")
    # Test hook (the CPU integration test): make one rank fail before training
    # so the parent's root-cause picker has a real torchelastic error file.
    fail_rank = os.environ.get("VERI_CHILD_FAIL_RANK")
    if fail_rank is not None and int(fail_rank) == rank:
        raise RuntimeError(f"injected failure on rank {rank} (VERI_CHILD_FAIL_RANK)")
    _restore_wandb_key(job_config, os.environ)
    progress = (
        RankZeroProgressFile(child["progress_path"])
        if rank == 0 and child.get("progress_path")
        else None
    )
    if rank == 0:
        log.info("launch plan:\n  %s", "\n  ".join(plan.get("trace", [])))
        for w in plan.get("warnings", []):
            log.warning("plan: %s", w)
    result = run_training(
        job_config,
        dataset_path=child.get("dataset_path"),
        logger=log,
        progress_fn=progress.step if progress else None,
        phase_fn=progress.phase if progress else None,
        event_fn=progress.event if progress else None,
        launch_plan=plan,
    )
    if rank == 0 and child.get("result_path"):
        tmp = child["result_path"] + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f)
        os.replace(tmp, child["result_path"])
    _destroy_process_group()
    return 0


def _destroy_process_group() -> None:
    """Tear the collective group down before interpreter exit (torch warns about
    a live group at exit; on macOS/gloo a live group aborts the process)."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:  # noqa: BLE001 -- teardown must not turn success into failure
        pass


def main(argv: list[str] | None = None) -> int:
    """Child entrypoint under torchelastic's @record: an exception on any rank
    is written to its TORCHELASTIC_ERROR_FILE (the parent's root-cause input)."""
    from torch.distributed.elastic.multiprocessing.errors import record

    return record(_child_main)(argv)


if __name__ == "__main__":
    sys.exit(main())
