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
        "save_strategy": "no",
        "save_steps": None,
        "num_train_epochs": 1,
        "report_to": "wandb" if wandb_enabled else "none",
        "bf16": _gpu_supports_bf16(),
        "gradient_checkpointing": True,
    }
    if "max_prompt_length" in grpo_params and "max_prompt_length" in hyperparameters:
        config_kwargs["max_prompt_length"] = hyperparameters["max_prompt_length"]
    if "beta" in grpo_params and "kl_coef" in hyperparameters:
        config_kwargs["beta"] = hyperparameters["kl_coef"]
    # Base path generates via transformers. vLLM rollouts are an opt-in
    # optimization (trl 0.22.2 + vllm 0.15.1 aren't directly compatible), so default
    # use_vllm off; a user with a compatible setup can flip it via hyperparameters.
    if "use_vllm" in grpo_params:
        config_kwargs["use_vllm"] = bool(hyperparameters.get("use_vllm", False))
    _apply_liger(config_kwargs, grpo_params, hyperparameters)

    return config_kwargs


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


def build_sft_config_kwargs(
    *,
    job_id: str,
    hyperparameters: dict[str, Any],
    sft_config_cls: type,
    output_root: str = "/tmp/ckpts",
    wandb_enabled: bool = False,
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
        "per_device_train_batch_size": hyperparameters.get("batch_size", 1),
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "wandb" if wandb_enabled else "none",
        "bf16": _gpu_supports_bf16(),
        "gradient_checkpointing": True,
    }
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

    return config_kwargs


def build_dpo_config_kwargs(
    *,
    job_id: str,
    hyperparameters: dict[str, Any],
    dpo_config_cls: type,
    output_root: str = "/tmp/ckpts",
    wandb_enabled: bool = False,
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
        "per_device_train_batch_size": hyperparameters.get("batch_size", 1),
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "wandb" if wandb_enabled else "none",
        "bf16": _gpu_supports_bf16(),
        "gradient_checkpointing": True,
    }
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

    return config_kwargs


def checkpoint_output_root(job_config: dict[str, Any]) -> str:
    """Return the runner local output root from structured checkpoint config."""
    checkpoint_config = job_config.get("checkpoint") or {}
    return checkpoint_config.get("local_output_root", "/tmp/ckpts")


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


def _load_model_and_tokenizer(
    base_model: str,
    hyperparameters: dict[str, Any],
    logger: logging.Logger,
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

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
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
    trainer: Any, progress_fn: Callable[[int, int, dict], None] | None
) -> None:
    """Register step-level progress reporting on any HF Trainer (no-op when
    the adapter didn't pass a progress_fn)."""
    if not progress_fn:
        return
    from transformers import TrainerCallback

    class _ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state:
                progress_fn(state.global_step, state.max_steps, {
                    k: float(v) for k, v in logs.items()
                    if isinstance(v, (int, float))
                })

    trainer.add_callback(_ProgressCallback())


def _train_save_finalize(
    *,
    trainer: Any,
    tokenizer: Any,
    job_config: dict[str, Any],
    wandb_enabled: bool,
    log: logging.Logger,
) -> dict[str, Any]:
    """The shared tail of every TRL method: train, save the checkpoint, build
    the result dict, and capture/finish the W&B run if one is live."""
    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0
    log.info("Training completed in %.1fs, loss=%.4f", train_time, train_result.training_loss)

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
    if wandb_enabled:
        try:
            import wandb

            if wandb.run:
                result["wandb_run_url"] = wandb.run.get_url()
                wandb.finish()
        except Exception:
            pass

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

    training_args = GRPOConfig(
        **build_grpo_config_kwargs(
            job_id=job_id,
            hyperparameters=hyperparameters,
            grpo_config_cls=GRPOConfig,
            output_root=checkpoint_output_root(job_config),
            wandb_enabled=wandb_enabled,
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
    model, tokenizer = _load_model_and_tokenizer(base_model, hyperparameters, log)
    log.info("Model loaded in %.1fs", time.time() - t0)
    if after_model_load:
        after_model_load()

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

    _attach_progress_callback(trainer, progress_fn)
    return _train_save_finalize(
        trainer=trainer,
        tokenizer=tokenizer,
        job_config=job_config,
        wandb_enabled=wandb_enabled,
        log=log,
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
    model, tokenizer = _load_model_and_tokenizer(base_model, hyperparameters, log)
    log.info("Model loaded in %.1fs", time.time() - t0)
    if after_model_load:
        after_model_load()

    training_args = DPOConfig(
        **build_dpo_config_kwargs(
            job_id=job_id,
            hyperparameters=hyperparameters,
            dpo_config_cls=DPOConfig,
            output_root=checkpoint_output_root(job_config),
            wandb_enabled=wandb_enabled,
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

    _attach_progress_callback(trainer, progress_fn)
    return _train_save_finalize(
        trainer=trainer,
        tokenizer=tokenizer,
        job_config=job_config,
        wandb_enabled=wandb_enabled,
        log=log,
    )


def _is_rocm() -> bool:
    """True when torch is an AMD/ROCm build (hip is set on ROCm wheels only)."""
    try:
        import torch
        return getattr(torch.version, "hip", None) is not None
    except Exception:
        return False


def _liger_available() -> bool:
    """True when liger-kernel is importable (installed by the ROCm bootstrap;
    NOT guaranteed on other images)."""
    import importlib.util
    return importlib.util.find_spec("liger_kernel") is not None


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

    # use_liger needs the liger-kernel package on the box (the ROCm bootstrap
    # installs it; other images may not). Fail fast with the reason, not an
    # opaque transformers ImportError after the model download.
    if hyperparameters.get("use_liger") and not _liger_available():
        raise ValueError(
            "use_liger requires the liger-kernel package, which is not "
            "installed on this worker image"
        )

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
