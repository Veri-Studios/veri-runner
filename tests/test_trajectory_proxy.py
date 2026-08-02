"""Trajectory proxy capture + episode renderer."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from veri_runner.trajectory_proxy import (
    SpanStore,
    TemplateDriftError,
    TrajectoryProxy,
    render_episode,
)


def _span(prompt, completion, logprobs=None, fidelity="captured", response=None):
    return {
        "prompt_token_ids": prompt,
        "completion_token_ids": completion,
        "completion_logprobs": logprobs,
        "token_fidelity": fidelity,
        "path": "/v1/chat/completions",
        "response": response or {},
    }


# ---- renderer ---------------------------------------------------------------


def test_render_single_turn_masks_prompt_trains_completion():
    episode = render_episode(
        "r1", [_span([10, 11, 12], [20, 21], logprobs=[-0.1, -0.2])]
    )
    assert episode.input_ids == [10, 11, 12, 20, 21]
    assert episode.loss_mask == [0, 0, 0, 1, 1]
    assert episode.logprobs == [0.0, 0.0, 0.0, -0.1, -0.2]
    assert episode.num_turns == 1


def test_render_multi_turn_masks_tool_delta_only():
    # Turn 2's prompt = turn 1 (prompt+completion) + tool result tokens [30, 31].
    spans = [
        _span([10, 11], [20], logprobs=[-0.5]),
        _span([10, 11, 20, 30, 31], [40, 41], logprobs=[-0.6, -0.7]),
    ]
    episode = render_episode("r1", spans)
    assert episode.input_ids == [10, 11, 20, 30, 31, 40, 41]
    assert episode.loss_mask == [0, 0, 1, 0, 0, 1, 1]


def test_render_prefix_drift_falls_back_to_last_turn():
    # Turn 2's prompt diverges from turn 1's rendering (template rewrote
    # history, e.g. Qwen3 thinking) -> degrade to a last-turn-only episode.
    spans = [
        _span([10, 11], [20]),
        _span([10, 99, 20, 30], [40], logprobs=[-0.3]),
    ]
    episode = render_episode("r1", spans)
    assert episode.metadata["render_fallback"] == "last_turn"
    assert episode.input_ids == [10, 99, 20, 30, 40]
    assert episode.loss_mask == [0, 0, 0, 0, 1]
    assert episode.logprobs == [0.0, 0.0, 0.0, 0.0, -0.3]
    assert episode.num_turns == 2


def test_render_prefix_drift_raises_when_fallback_disabled():
    spans = [
        _span([10, 11], [20]),
        _span([10, 99, 20, 30], [40]),
    ]
    with pytest.raises(TemplateDriftError, match="drift"):
        render_episode("r1", spans, allow_last_turn_fallback=False)


def test_render_rejects_absent_token_ids():
    with pytest.raises(TemplateDriftError, match="absent"):
        render_episode("r1", [_span(None, None, fidelity="absent")])


def test_render_rejects_empty_rollout():
    with pytest.raises(ValueError, match="no spans"):
        render_episode("r1", [])


def test_render_extracts_final_completion_text():
    response = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
    episode = render_episode("r1", [_span([1], [2], response=response)])
    assert episode.final_completion_text == "done"


# ---- proxy capture ----------------------------------------------------------


class _FakeTokenizer:
    """Decode stub: the proxy renders response text from completion ids
    (TRL's /chat/ returns ids only)."""

    VOCAB = {(4, 5): "hi", (20,): "turn one", (40,): "turn two"}

    def decode(self, ids, skip_special_tokens=True):
        return self.VOCAB.get(tuple(ids), " ".join(str(i) for i in ids))


class _StubUpstream(BaseHTTPRequestHandler):
    """TRL vllm-serve stand-in: serves the /chat/ schema (token ids +
    logprobs, no OpenAI routes — the real server 404s those)."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "stub-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(length))
        # Record what the proxy sent us for later assertion.
        type(self).last_request = req
        type(self).last_path = self.path
        resp = {
            "prompt_ids": [[1, 2, 3]],
            "completion_ids": [[4, 5]],
            "logprobs": [[[-0.1], [-0.2]]],
            "logprob_token_ids": [[[4], [5]]],
        }
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubUpstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def test_proxy_end_to_end(upstream, tmp_path):
    store = SpanStore(tmp_path)
    proxy = TrajectoryProxy(upstream_url=upstream, store=store, tokenizer=_FakeTokenizer())
    try:
        base_url = proxy.register("r0")
        resp = _post(
            base_url + "/chat/completions",
            {"model": "stub", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        # Response text is decoded from the server's completion ids.
        assert resp["choices"][0]["message"]["content"] == "hi"

        # The OpenAI request was translated to TRL vllm-serve's /chat/ schema:
        # batched messages, sampled-token logprobs, a real max_tokens default.
        assert _StubUpstream.last_path == "/chat/"
        sent = _StubUpstream.last_request
        assert sent["messages"] == [[{"role": "user", "content": "hi"}]]
        assert sent["logprobs"] == 0
        assert sent["max_tokens"] == 1024

        spans = store.load("r0")
        assert len(spans) == 1
        assert spans[0]["token_fidelity"] == "captured"
        assert spans[0]["prompt_token_ids"] == [1, 2, 3]
        assert spans[0]["completion_logprobs"] == [-0.1, -0.2]

        # Captured span renders into a masked episode.
        episode = render_episode("r0", spans)
        assert episode.loss_mask == [0, 0, 0, 1, 1]
    finally:
        proxy.stop()


def test_render_extracts_final_text_from_anthropic_response():
    response = {
        "type": "message",
        "content": [{"type": "text", "text": "ANSWER: https://github.com/x/y"}],
    }
    episode = render_episode("r1", [_span([1], [2], response=response)])
    assert episode.final_completion_text == "ANSWER: https://github.com/x/y"


def test_proxy_anthropic_messages_end_to_end(upstream, tmp_path):
    # An Anthropic-SDK harness trains through the same token-native /chat/
    # call as the OpenAI route (a verbatim forward 404s on TRL vllm-serve).
    store = SpanStore(tmp_path)
    proxy = TrajectoryProxy(upstream_url=upstream, store=store, tokenizer=_FakeTokenizer())
    try:
        base_url = proxy.register("r0")
        resp = _post(
            base_url + "/messages",
            {
                "model": "stub",
                "max_tokens": 512,
                "system": "be terse",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                ],
            },
        )
        # Anthropic Messages response shape, text decoded from completion ids.
        assert resp["type"] == "message"
        assert resp["content"] == [{"type": "text", "text": "hi"}]
        assert resp["stop_reason"] == "end_turn"
        assert resp["usage"] == {"input_tokens": 3, "output_tokens": 2}

        # System + content blocks flattened onto the policy chat template.
        assert _StubUpstream.last_path == "/chat/"
        assert _StubUpstream.last_request["messages"] == [[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]]
        assert _StubUpstream.last_request["max_tokens"] == 512

        # Token-exact span -> renderable episode, same guarantee as OpenAI.
        spans = store.load("r0")
        assert len(spans) == 1
        assert spans[0]["token_fidelity"] == "captured"
        assert spans[0]["path"] == "/v1/messages"
        episode = render_episode("r0", spans)
        assert episode.loss_mask == [0, 0, 0, 1, 1]
        assert episode.final_completion_text == "hi"
    finally:
        proxy.stop()


def test_proxy_anthropic_native_tools_rejected_clearly(upstream, tmp_path):
    # tool_use translation is a follow-up; the guard must fail loudly (400
    # with guidance), not run a silently degraded rollout.
    store = SpanStore(tmp_path)
    with TrajectoryProxy(upstream_url=upstream, store=store) as proxy:
        base_url = proxy.register("r0")
        try:
            _post(
                base_url + "/messages",
                {
                    "model": "stub",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"name": "web_fetch", "input_schema": {"type": "object"}}],
                },
            )
            assert False, "expected HTTP 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400
            error = json.loads(e.read())
            assert "tool_use" in error["error"]["message"]
        assert store.load("r0") == []


def test_proxy_get_passthrough_not_captured(upstream, tmp_path):
    store = SpanStore(tmp_path)
    with TrajectoryProxy(upstream_url=upstream, store=store) as proxy:
        base_url = proxy.register("r0")
        with urllib.request.urlopen(base_url + "/models", timeout=10) as resp:
            assert json.loads(resp.read())["data"][0]["id"] == "stub-model"
        assert store.load("r0") == []


# ---- upstream scheme guard ---------------------------------------------------


def test_proxy_rejects_non_http_upstream(tmp_path):
    # BAD: a mis-set upstream env must not reach urllib (file:// would read
    # local files); the proxy refuses to construct.
    with pytest.raises(ValueError, match="must be http"):
        TrajectoryProxy(upstream_url="file:///etc/passwd", store=SpanStore(tmp_path))


def test_proxy_accepts_http_upstream(tmp_path):
    # GOOD: plain http upstream constructs (every end-to-end test above also
    # exercises this against a live localhost upstream).
    proxy = TrajectoryProxy(upstream_url="http://127.0.0.1:1", store=SpanStore(tmp_path))
    proxy.stop()
