"""Harness rollout runner: env contract, reward scoring, end-to-end
rollout through the trajectory proxy in direct (non-sandbox) mode."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from veri_runner.harness_rollout import (
    HarnessRunner,
    HarnessSpec,
    build_harness_env,
    score_episode,
)
from veri_runner.trajectory_proxy import Episode, SpanStore, TrajectoryProxy


def test_build_harness_env_redirects_agent_sdks():
    spec = HarnessSpec(entrypoint="python agent.py", env={"MY_FLAG": "1"})
    env = build_harness_env(
        spec,
        task_row={"prompt": "find the repo", "answer": "https://github.com/x/y"},
        task_index=2,
        rollout_index=1,
        rollout_id="s0-t2-r1",
        policy_base_url="http://127.0.0.1:5000/v1",
        policy_model="Qwen/Qwen3-4B",
        data_dir="/tmp/traj",
        max_turns=10,
    )
    assert json.loads(env["VERI_TASK_INPUT"]) == "find the repo"
    assert json.loads(env["VERI_TASK"])["answer"] == "https://github.com/x/y"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:5000/v1"
    # Anthropic SDK appends /v1/messages itself, so the root carries no /v1.
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:5000"
    assert env["ANTHROPIC_AUTH_TOKEN"]
    # vLLM 404s any model name but the served one; the harness must send this.
    assert env["VERI_POLICY_MODEL"] == "Qwen/Qwen3-4B"
    assert env["VERI_MAX_TURNS"] == "10"
    assert env["MY_FLAG"] == "1"


def test_score_episode_passes_trl_signature_and_extra_columns():
    calls = {}

    def reward(prompts, completions, answer=None, **kwargs):
        calls["prompts"] = prompts
        calls["completions"] = completions
        calls["answer"] = answer
        return [1.0 if completions[0] == answer[0] else 0.0]

    episode = Episode(
        rollout_id="r", input_ids=[1], loss_mask=[1], logprobs=[0.0],
        num_turns=1, final_completion_text="yes",
    )
    value = score_episode(
        [reward],
        task_row={"prompt": "q", "answer": "yes"},
        episode=episode,
    )
    assert value == 1.0
    assert calls["prompts"] == ["q"]
    assert calls["completions"] == ["yes"]
    assert calls["answer"] == ["yes"]


def test_score_episode_passes_trajectory_to_rewards_that_declare_it():
    # VS-378: shaped rewards opt in via a `trajectory` parameter and receive
    # the final span's full message history (TRL list-per-completion shape).
    seen = {}

    def shaped(prompts, completions, trajectory=None, **kwargs):
        seen["trajectory"] = trajectory
        return [1.0]

    episode = Episode(
        rollout_id="r", input_ids=[1], loss_mask=[1], logprobs=[0.0],
        num_turns=1, final_completion_text="yes",
    )
    span = {
        "request": {"system": "sys", "messages": [{"role": "user", "content": "q"}]},
        "response": {"content": [{"type": "text", "text": "yes"}]},
    }
    from veri_runner.harness_rollout import _span_history
    score_episode(
        [shaped], task_row={"prompt": "q"}, episode=episode,
        trajectory=_span_history(span),
    )
    history = seen["trajectory"][0]
    assert history[0] == {"role": "system", "content": "sys"}
    assert history[1] == {"role": "user", "content": "q"}
    assert history[2]["role"] == "assistant"


def test_score_episode_never_passes_trajectory_uninvited():
    # Zero-change contract: a plain TRL reward with NO trajectory param and
    # NO **kwargs must not receive the kwarg (TypeError would fail the
    # rollout as reward/render error).
    def plain(prompts, completions, answer=None):
        return [1.0]

    episode = Episode(
        rollout_id="r", input_ids=[1], loss_mask=[1], logprobs=[0.0],
        num_turns=1, final_completion_text="yes",
    )
    value = score_episode(
        [plain], task_row={"prompt": "q", "answer": "yes"}, episode=episode,
        trajectory=[{"role": "user", "content": "q"}],
    )
    assert value == 1.0


def test_direct_mode_env_excludes_worker_credentials(monkeypatch):
    # VS-377: on the docker-delivery (Vast) provider path there is no runsc
    # nesting, so the harness runs as a direct subprocess in production. It
    # must NOT inherit the worker's credential surface — cloud keys, the
    # per-job callback token, W&B keys, or any other VERI_* worker secret —
    # only the interpreter-finding allowlist rides along.
    from veri_runner.harness_rollout import _direct_inherit_env

    # Seed the full credential surface a real Vast worker holds in os.environ.
    secrets = {
        "AWS_ACCESS_KEY_ID": "leak-me",
        "AWS_SECRET_ACCESS_KEY": "leak-me",
        "AWS_SESSION_TOKEN": "leak-me",
        "VERI_CALLBACK_TOKEN": "leak-me",  # per-job worker callback token
        "WANDB_API_KEY": "leak-me",
        "HF_TOKEN": "leak-me",
    }
    for k, v in secrets.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/root")

    env = _direct_inherit_env()

    # No credential of any kind leaks through the allowlist.
    for k in secrets:
        assert k not in env, f"{k} leaked into the direct-mode harness env"
    assert not any(k.startswith(("AWS_", "WANDB_", "VERI_")) for k in env)
    # The interpreter-finding allowlist still rides along.
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/root"


def test_score_episode_applies_reward_weights():
    episode = Episode(
        rollout_id="r", input_ids=[1], loss_mask=[1], logprobs=[0.0],
        num_turns=1, final_completion_text="x",
    )
    fns = [lambda **kw: [1.0], lambda **kw: [0.5]]
    assert score_episode(fns, task_row={"prompt": "q"}, episode=episode) == 1.5
    assert (
        score_episode(fns, task_row={"prompt": "q"}, episode=episode,
                      reward_weights=[1.0, 2.0])
        == 2.0
    )


# ---- end-to-end rollout through the real proxy ------------------------------


class _FakeTokenizer:
    """Decode stub: the proxy renders response text from completion ids
    (TRL's /chat/ returns ids only)."""

    VOCAB = {(20,): "turn one", (40,): "turn two"}

    def decode(self, ids, skip_special_tokens=True):
        return self.VOCAB.get(tuple(ids), " ".join(str(i) for i in ids))


class _StubPolicy(BaseHTTPRequestHandler):
    """TRL vllm-serve stand-in (the /chat/ schema — the real server has no
    OpenAI routes): stateless, decides the turn from the conversation length
    (real harnesses resend growing histories, so turn 2 carries more messages).
    Stateless keeps concurrent rollouts from cross-talking through shared
    counter state."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(length))
        # /chat/ batches: messages is a list of conversations.
        conversation = (req.get("messages") or [[]])[0]
        # Turn 2's prompt extends turn 1's prompt+completion (delta = [30]).
        if len(conversation) <= 1:
            prompt_ids, completion_ids = [10, 11], [20]
        else:
            prompt_ids, completion_ids = [10, 11, 20, 30], [40]
        body = json.dumps({
            "prompt_ids": [prompt_ids],
            "completion_ids": [completion_ids],
            "logprobs": [[[-0.1]] * len(completion_ids)],
            "logprob_token_ids": [[[i] for i in completion_ids]],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def policy_server():
    _StubPolicy.call_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubPolicy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


_HARNESS_SCRIPT = """
import json, os, urllib.request
base = os.environ['OPENAI_BASE_URL']
def call(msgs):
    payload = json.dumps({'model': 'm', 'messages': msgs}).encode()
    url = base + '/chat/completions'
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())
msgs = [{'role': 'user', 'content': os.environ['VERI_TASK_INPUT']}]
r1 = call(msgs)
msgs = msgs + [r1['choices'][0]['message'], {'role': 'tool', 'content': 'obs'}]
call(msgs)
"""


def _runner(policy_url, tmp_path, entrypoint, reward_fns):
    store = SpanStore(tmp_path / "spans")
    proxy = TrajectoryProxy(upstream_url=policy_url, store=store, tokenizer=_FakeTokenizer())
    spec = HarnessSpec(entrypoint=entrypoint)
    return proxy, HarnessRunner(
        spec=spec,
        proxy=proxy,
        span_store=store,
        reward_fns=reward_fns,
        trajectory_dir=str(tmp_path / "trajectories"),
        sandbox=False,  # direct mode: tests only
        timeout_s=30,
    )


def test_run_rollout_end_to_end(policy_server, tmp_path):
    entrypoint = f"{sys.executable} -c \"{_HARNESS_SCRIPT}\""

    def reward(prompts, completions, **kw):
        return [float(len(completions[0]) > 0)]

    proxy, runner = _runner(policy_server, tmp_path, entrypoint, [reward])
    try:
        result = runner.run_rollout(
            task_row={"prompt": "find the url"},
            task_index=0,
            rollout_index=0,
            policy_step=0,
        )
    finally:
        proxy.stop()

    assert result.status == "completed", result.error
    assert result.reward == 1.0
    assert result.num_turns == 2
    assert result.tokens_in == 2 + 4
    assert result.tokens_out == 2
    # Episode: prompt masked, turn-1 completion trained, tool delta masked,
    # turn-2 completion trained.
    assert result.episode is not None
    assert result.episode.loss_mask == [0, 0, 1, 0, 1]

    # Trajectory file written at the archive layout and self-contained.
    with open(result.trajectory_path) as f:
        record = json.loads(f.readline())
    assert record["policy_step"] == 0
    assert record["task_index"] == 0
    assert record["reward"] == 1.0
    assert len(record["spans"]) == 2
    assert record["episode"]["final_completion_text"] == "turn two"


def test_run_rollout_failed_harness(policy_server, tmp_path):
    proxy, runner = _runner(policy_server, tmp_path, "exit 3", [lambda **kw: [1.0]])
    try:
        result = runner.run_rollout(
            task_row={"prompt": "x"}, task_index=1, rollout_index=0, policy_step=0,
        )
    finally:
        proxy.stop()
    assert result.status == "failed"
    assert "exited 3" in result.error
    assert result.reward is None


def test_run_rollout_harness_without_model_calls(policy_server, tmp_path):
    proxy, runner = _runner(policy_server, tmp_path, "true", [lambda **kw: [1.0]])
    try:
        result = runner.run_rollout(
            task_row={"prompt": "x"}, task_index=1, rollout_index=0, policy_step=0,
        )
    finally:
        proxy.stop()
    assert result.status == "failed"
    assert "no spans" in result.error


def test_run_step_groups_sorted(policy_server, tmp_path):
    entrypoint = f"{sys.executable} -c \"{_HARNESS_SCRIPT}\""
    proxy, runner = _runner(policy_server, tmp_path, entrypoint, [lambda **kw: [1.0]])
    try:
        results = runner.run_step(
            tasks=[(0, {"prompt": "a"}), (1, {"prompt": "b"})],
            group_size=2,
            policy_step=3,
        )
    finally:
        proxy.stop()
    assert [(r.task_index, r.rollout_index) for r in results] == [
        (0, 0), (0, 1), (1, 0), (1, 1),
    ]
    assert all(r.status == "completed" for r in results)
    assert all(r.policy_step == 3 for r in results)
