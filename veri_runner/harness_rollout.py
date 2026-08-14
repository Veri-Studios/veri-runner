"""Rollout runner for harness-in-the-loop GRPO.

Execs the user's UNMODIFIED agent harness (Claude Agent SDK / LangChain /
OpenAI Agents SDK) once per (task, group member), pointed at the trajectory
proxy's per-rollout port, then renders the captured spans into an episode and
scores it with the uploaded reward function.

Harness contract (documented on the Veri docs site, Training -> Harness):
- The harness artifact is a code bundle + entrypoint, same shape as a
  run-script code bundle. It is invoked once per rollout with the task in env:
    VERI_TASK_INPUT   the row's `prompt` as JSON (string or messages list)
    VERI_TASK         the full dataset row as JSON (extra columns included)
    VERI_POLICY_BASE_URL / OPENAI_BASE_URL      policy endpoint (OpenAI route)
    ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN   policy endpoint (Claude SDK)
    VERI_MAX_TURNS    turn budget hint
- The harness just talks to the endpoint; the proxy captures token ids +
  logprobs below the protocol layer. No tracing library, no code change.

Sandboxing depends on the provider. On runsc-capable AWS hosts, production
rollouts run in docker + gVisor (runsc) with host networking (to reach the
localhost proxy) and NO GPU (the harness only makes HTTP calls; the policy
lives on the trainer's GPUs). On docker-delivery providers (Vast) the worker
IS a container, so runsc can't nest and sandbox_mode/runsc do not apply: the
harness runs there as a direct `bash -c <entrypoint>` subprocess in
production, not just in dev/tests. Because that direct-mode process is not
isolated from the worker, it must never inherit the worker's environment
(callback token, cloud credentials, W&B keys), so it gets an explicit minimal
allowlist env instead of os.environ (see _direct_inherit_env). Filesystem
isolation still requires the runsc path.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    # Repo layout (tests, dev checkouts).
    from veri_runner.trajectory_proxy import (
        Episode,
        SpanStore,
        TemplateDriftError,
        TrajectoryProxy,
        render_episode,
    )
except ImportError:
    # Worker layout: /opt/veri flat drop (AMI / S3 hot-patch / transport embed).
    from trajectory_proxy import (
        Episode,
        SpanStore,
        TemplateDriftError,
        TrajectoryProxy,
        render_episode,
    )

log = logging.getLogger("veri.harness_rollout")

DEFAULT_HARNESS_TIMEOUT_S = 900  # one rollout = one harness invocation


@dataclass
class HarnessSpec:
    """The user-supplied harness: a code bundle and how to invoke it. Mirrors
    the custom-script config shape so the same upload/validation path serves
    both."""

    entrypoint: str
    base_image: str = "veri/base"
    code_dir: str | None = None  # extracted artifact on the worker
    deps: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    protocol: str = "openai"  # "openai" | "anthropic"


@dataclass
class RolloutResult:
    rollout_id: str
    policy_step: int
    task_index: int
    rollout_index: int
    status: str  # "completed" | "failed" | "truncated"
    reward: float | None
    episode: Episode | None
    num_turns: int
    tokens_in: int
    tokens_out: int
    trajectory_path: str | None
    error: str | None
    duration_s: float


def build_harness_env(
    spec: HarnessSpec,
    *,
    task_row: dict[str, Any],
    task_index: int,
    rollout_index: int,
    rollout_id: str,
    policy_base_url: str,  # e.g. http://127.0.0.1:PORT/v1
    policy_model: str,  # served-model-name the policy endpoint answers to
    data_dir: str,
    max_turns: int,
) -> dict[str, str]:
    """Env for one harness invocation. Redirects the three big agent SDKs at
    the policy via their standard base-URL env vars — the "unmodified harness"
    promise is kept by env config alone."""
    root_url = (
        policy_base_url[: -len("/v1")] if policy_base_url.endswith("/v1") else policy_base_url
    )
    env: dict[str, str] = {
        "VERI_TASK_INPUT": json.dumps(task_row.get("prompt"), ensure_ascii=False),
        "VERI_TASK": json.dumps(task_row, ensure_ascii=False),
        "VERI_TASK_INDEX": str(task_index),
        "VERI_ROLLOUT_INDEX": str(rollout_index),
        "VERI_ROLLOUT_ID": rollout_id,
        "VERI_POLICY_BASE_URL": policy_base_url,
        # vLLM answers to the base model's own id (served-model-name), and
        # 404s anything else — harnesses must send exactly this.
        "VERI_POLICY_MODEL": policy_model,
        "VERI_DATA_DIR": data_dir,
        "VERI_MAX_TURNS": str(max_turns),
        # OpenAI SDK / LangChain (langchain-openai) / OpenAI Agents SDK.
        "OPENAI_BASE_URL": policy_base_url,
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "veri-harness",
        # Claude Agent SDK / Anthropic SDK: honored via ANTHROPIC_BASE_URL.
        # The token is a placeholder — the policy server doesn't auth — but
        # the SDK refuses to start without one.
        "ANTHROPIC_BASE_URL": root_url,
        "ANTHROPIC_AUTH_TOKEN": "veri-harness",
    }
    for k, v in spec.env.items():
        env[str(k)] = str(v)
    return env


def _harness_inner_script(spec: HarnessSpec) -> str:
    """Deps layer + entrypoint inside the container (same layout as run-script)."""
    deps = spec.deps or {}
    kind = deps.get("kind", "none")
    lines = ["set -euo pipefail"]
    if kind == "requirements" and deps.get("content"):
        lines.append("uv pip install --system -r /workspace/code/.veri_requirements.txt")
    elif kind == "uv_lock" and deps.get("content"):
        lines.append("uv pip sync --system /workspace/code/.veri_uv.lock")
    lines.append(spec.entrypoint)
    return "\n".join(lines)


def build_harness_docker_cmd(
    spec: HarnessSpec,
    env: dict[str, str],
    *,
    rollout_id: str,
    sandbox: bool = True,
) -> list[str]:
    """`docker run` argv for one rollout. Host networking so the harness can
    reach the proxy's localhost port; no GPU (HTTP-only workload). Under
    runsc there is no nvproxy trap here precisely because no GPU is passed."""
    cmd = ["docker", "run", "--rm", "--name", f"veri-harness-{rollout_id}"]
    if sandbox:
        cmd += ["--runtime=runsc"]
    cmd += ["--network", "host"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    if spec.code_dir:
        cmd += ["-v", f"{spec.code_dir}:/workspace/code", "-w", "/workspace/code"]
    cmd += [spec.base_image, "bash", "-c", _harness_inner_script(spec)]
    return cmd




_DIRECT_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL")


def _direct_inherit_env() -> dict[str, str]:
    """Minimal worker env a direct-mode harness may inherit (VS-377): enough
    to find interpreters and write temp files, none of the credential
    surface (AWS_*, WANDB_*, VERI_* worker internals)."""
    return {k: os.environ[k] for k in _DIRECT_ENV_ALLOWLIST if k in os.environ}

def _span_history(span: dict[str, Any]) -> list[dict[str, Any]]:
    """The full conversation as the harness last sent it (protocol-shaped
    message dicts, OpenAI or Anthropic), plus the final assistant reply -
    what a shaped reward (VS-378) gets to see via `trajectory=`."""
    req = span.get("request") or {}
    out: list[dict[str, Any]] = []
    if req.get("system"):
        out.append({"role": "system", "content": req["system"]})
    out.extend(req.get("messages") or [])
    resp = span.get("response") or {}
    if resp.get("choices"):
        message = resp["choices"][0].get("message") or {}
        out.append({"role": "assistant", "content": message.get("content")})
    elif isinstance(resp.get("content"), list):
        out.append({"role": "assistant", "content": resp["content"]})
    return out


def score_episode(
    reward_fns: list[Callable[..., Any]],
    *,
    task_row: dict[str, Any],
    episode: Episode,
    reward_weights: list[float] | None = None,
    trajectory: list[dict[str, Any]] | None = None,
) -> float:
    """Score a finished trajectory with the uploaded TRL-signature reward
    function(s): reward(prompts=..., completions=..., **extra_columns). The
    completion text is the trajectory's FINAL assistant message; extra dataset
    columns (e.g. `answer` for ground-truth match) pass through as lists, as
    TRL would pass them."""
    prompts = [task_row.get("prompt")]
    completions = [episode.final_completion_text]
    extra = {
        k: [v] for k, v in task_row.items() if k != "prompt" and not k.startswith("_")
    }
    weights = reward_weights or [1.0] * len(reward_fns)
    total = 0.0
    for fn, weight in zip(reward_fns, weights):
        kwargs = dict(extra)
        if trajectory is not None and "trajectory" in inspect.signature(fn).parameters:
            # Shaped rewards (VS-378) opt in by declaring `trajectory`; the
            # TRL list-per-completion convention holds (one history here).
            kwargs["trajectory"] = [trajectory]
        value = fn(prompts=prompts, completions=completions, **kwargs)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else 0.0
        total += weight * float(value)
    return total


class HarnessRunner:
    """Runs GRPO rollout groups: for each task, N harness invocations whose
    trajectories form the reward-comparison group."""

    def __init__(
        self,
        *,
        spec: HarnessSpec,
        proxy: TrajectoryProxy,
        span_store: SpanStore,
        reward_fns: list[Callable[..., Any]],
        reward_weights: list[float] | None = None,
        trajectory_dir: str,
        policy_model: str = "policy",
        sandbox: bool = True,
        max_turns: int = 40,
        timeout_s: int = DEFAULT_HARNESS_TIMEOUT_S,
        max_parallel: int = 4,
    ):
        self.spec = spec
        self.proxy = proxy
        self.span_store = span_store
        self.reward_fns = reward_fns
        self.reward_weights = reward_weights
        self.trajectory_dir = Path(trajectory_dir)
        self.policy_model = policy_model
        self.sandbox = sandbox
        self.max_turns = max_turns
        self.timeout_s = timeout_s
        self.max_parallel = max_parallel
        self._sem = threading.Semaphore(max_parallel)

    def run_step(
        self,
        *,
        tasks: list[tuple[int, dict[str, Any]]],  # (task_index, row)
        group_size: int,
        policy_step: int,
    ) -> list[RolloutResult]:
        """Run one training step's rollouts: every task x group_size harness
        invocations, parallel up to max_parallel."""
        work = [
            (task_index, row, rollout_index)
            for task_index, row in tasks
            for rollout_index in range(group_size)
        ]
        results: list[RolloutResult] = []
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            futures = [
                pool.submit(
                    self.run_rollout,
                    task_row=row,
                    task_index=task_index,
                    rollout_index=rollout_index,
                    policy_step=policy_step,
                )
                for task_index, row, rollout_index in work
            ]
            for future in futures:
                results.append(future.result())
        results.sort(key=lambda r: (r.task_index, r.rollout_index))
        return results

    def run_rollout(
        self,
        *,
        task_row: dict[str, Any],
        task_index: int,
        rollout_index: int,
        policy_step: int,
    ) -> RolloutResult:
        rollout_id = f"s{policy_step}-t{task_index}-r{rollout_index}"
        started = time.time()
        base_url = self.proxy.register(rollout_id)
        try:
            with self._sem:
                status, error = self._exec_harness(
                    rollout_id=rollout_id,
                    base_url=base_url,
                    task_row=task_row,
                    task_index=task_index,
                    rollout_index=rollout_index,
                )
        finally:
            self.proxy.unregister(rollout_id)

        spans = self.span_store.load(rollout_id)
        tokens_in = sum(len(s.get("prompt_token_ids") or []) for s in spans)
        tokens_out = sum(len(s.get("completion_token_ids") or []) for s in spans)

        episode: Episode | None = None
        reward: float | None = None
        if status == "completed":
            if not spans:
                status, error = "failed", "harness made no model calls (no spans captured)"
            else:
                try:
                    episode = render_episode(rollout_id, spans)
                    reward = score_episode(
                        self.reward_fns,
                        task_row=task_row,
                        episode=episode,
                        reward_weights=self.reward_weights,
                        trajectory=_span_history(spans[-1]) if spans else None,
                    )
                except TemplateDriftError as e:
                    status, error = "failed", f"template_drift: {e}"
                except Exception as e:
                    status, error = "failed", f"reward/render error: {e}"

        trajectory_path = self._write_trajectory(
            rollout_id=rollout_id,
            policy_step=policy_step,
            task_index=task_index,
            rollout_index=rollout_index,
            task_row=task_row,
            status=status,
            reward=reward,
            episode=episode,
            spans=spans,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            error=error,
        )
        return RolloutResult(
            rollout_id=rollout_id,
            policy_step=policy_step,
            task_index=task_index,
            rollout_index=rollout_index,
            status=status,
            reward=reward,
            episode=episode,
            num_turns=len(spans),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            trajectory_path=str(trajectory_path),
            error=error,
            duration_s=time.time() - started,
        )

    def _exec_harness(
        self,
        *,
        rollout_id: str,
        base_url: str,
        task_row: dict[str, Any],
        task_index: int,
        rollout_index: int,
    ) -> tuple[str, str | None]:
        env = build_harness_env(
            self.spec,
            task_row=task_row,
            task_index=task_index,
            rollout_index=rollout_index,
            rollout_id=rollout_id,
            policy_base_url=base_url,
            policy_model=self.policy_model,
            data_dir=str(self.trajectory_dir),
            max_turns=self.max_turns,
        )
        if self.sandbox and self.spec.code_dir:
            cmd = build_harness_docker_cmd(self.spec, env, rollout_id=rollout_id, sandbox=True)
        else:
            # Direct mode: the LIVE path whenever sandbox_mode is off (the
            # default today - VS-377). The harness must not inherit the
            # worker's environment (cloud credentials, callback token, W&B
            # keys), so only a minimal allowlist rides along. Filesystem
            # isolation still requires the sandbox path.
            cmd = ["bash", "-c", self.spec.entrypoint]
            env = {**_direct_inherit_env(), **env}

        cwd = self.spec.code_dir if not (self.sandbox and self.spec.code_dir) else None
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            if self.sandbox and self.spec.code_dir:
                subprocess.run(
                    ["docker", "rm", "-f", f"veri-harness-{rollout_id}"],
                    capture_output=True,
                )
            return "truncated", f"harness timed out after {self.timeout_s}s"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            return "failed", f"harness exited {proc.returncode}: {tail}"
        return "completed", None

    def _write_trajectory(
        self,
        *,
        rollout_id: str,
        policy_step: int,
        task_index: int,
        rollout_index: int,
        task_row: dict[str, Any],
        status: str,
        reward: float | None,
        episode: Episode | None,
        spans: list[dict[str, Any]],
        tokens_in: int,
        tokens_out: int,
        error: str | None,
    ) -> Path:
        """One self-contained JSONL file per rollout — the unit the archive UI
        streams and the worker PUTs to S3."""
        path = (
            self.trajectory_dir
            / f"step-{policy_step}"
            / f"task-{task_index}"
            / f"rollout-{rollout_index}.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "rollout_id": rollout_id,
            "policy_step": policy_step,
            "task_index": task_index,
            "rollout_index": rollout_index,
            "status": status,
            "reward": reward,
            "num_turns": len(spans),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "task_input": task_row.get("prompt"),
            "error": error,
            "spans": spans,
            "episode": (
                {
                    "input_ids": episode.input_ids,
                    "loss_mask": episode.loss_mask,
                    "num_turns": episode.num_turns,
                    "final_completion_text": episode.final_completion_text,
                }
                if episode
                else None
            ),
        }
        with open(path, "w") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path
