"""Policy proxy + trajectory capture for harness-in-the-loop GRPO.

Architecture (the Polar / Agent-Lightning pattern):

- The in-training policy is served by ONE vLLM-compatible server (TRL's
  `vllm-serve`, so the trainer can push fresh weights into it every step).
- This proxy sits between the user's UNMODIFIED agent harness and that
  server. It injects `return_token_ids` + `logprobs` so every response
  carries the server's own tokenization (token-in-token-out — no
  retokenization drift), forces non-streaming so spans are capturable, and
  persists one JSONL span per request.
- Rollout attribution needs no harness cooperation: the rollout runner gives
  each harness invocation its OWN proxy port (VERI_POLICY_BASE_URL), and the
  proxy tags every request on that port with the rollout id.
- `render_episode` reassembles the spans into a single training episode with
  assistant-only loss masking, using verl-style delta tokenization: the next
  turn's server-rendered prompt MUST extend the previous prompt+completion
  exactly; the delta (tool results + template tokens) is masked out of the
  loss. A prefix mismatch means the serving and training tokenizations have
  drifted — the rollout is rejected, never silently trained on.

Anthropic route: /v1/messages is translated by _messages_adapter onto the
same token-native /chat/ call as the OpenAI route (a verbatim forward 404s:
TRL's vllm-serve has no /v1/messages route). Native tool_use is translated
end-to-end (VS-374): the `tools` array rides to the policy chat template in
OpenAI function format, hermes <tool_call> tags in the decoded completion
come back as tool_use blocks, tool_result blocks re-render as role:"tool"
messages, streaming clients get a synthesized Anthropic SSE sequence, and
/v1/messages/count_tokens is served from the trainer tokenizer. Spans that
come back without token ids are marked token_fidelity="absent" and
render_episode rejects them, so no route can ever poison the training batch.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

log = logging.getLogger("veri.trajectory_proxy")

# OpenAI-compatible endpoints we capture. Everything else (GET /v1/models,
# /health) is proxied verbatim so harness SDK probes keep working.
_CAPTURED_POST_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/messages")


# Hermes-format tool calls (the Qwen-family chat template's native emission):
# <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _anthropic_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic tools [{name, description, input_schema}] -> the OpenAI
    function format chat templates consume (Qwen renders these natively)."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.get("name") or "",
                "description": t.get("description") or "",
                "parameters": t.get("input_schema") or {"type": "object"},
            },
        }
        for t in tools
    ]


def _anthropic_messages_to_chat(
    system: Any, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Anthropic conversation -> OpenAI-style chat messages for the policy
    chat template. Assistant tool_use blocks become `tool_calls` entries and
    tool_result blocks become role:"tool" messages, both of which the
    Qwen-family template renders natively (VS-374). The template owns the
    canonical serialization — the same path vLLM's chat() takes — so
    re-rendered history matches the model's own emission as closely as the
    template allows; residual drift is caught by render_episode's token-exact
    stitching and last-turn fallback."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": _anthropic_text(system)})
    for m in messages or []:
        role = m.get("role") or "user"
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for b in content or []:
                if isinstance(b, str):
                    text_parts.append(b)
                elif b.get("type") == "text":
                    text_parts.append(b.get("text") or "")
                elif b.get("type") == "tool_use":
                    tool_calls.append({
                        "id": b.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": b.get("name") or "",
                            "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False),
                        },
                    })
                elif b.get("type") == "thinking":
                    pass  # never re-render reasoning into the prompt
                else:
                    text_parts.append(json.dumps(b, ensure_ascii=False))
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(p for p in text_parts if p),
            }
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        else:
            pending: list[str] = []
            for b in content or []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    if pending:
                        out.append({"role": role, "content": "\n".join(pending)})
                        pending = []
                    out.append({
                        "role": "tool",
                        "content": _anthropic_text(b.get("content")),
                        "tool_call_id": b.get("tool_use_id") or "",
                    })
                elif isinstance(b, str):
                    pending.append(b)
                elif b.get("type") == "text":
                    pending.append(b.get("text") or "")
                else:
                    pending.append(json.dumps(b, ensure_ascii=False))
            if pending:
                out.append({"role": role, "content": "\n".join(pending)})
    return out


def _parse_tool_calls(
    text: str, rollout_id: str, index: int
) -> tuple[list[dict[str, Any]], bool]:
    """Split a decoded completion into Anthropic content blocks: hermes
    <tool_call> tags become tool_use blocks, everything else stays text.
    Malformed tool-call JSON is left visible as text (the harness sees the
    model's actual output) rather than dropped silently."""
    plain_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    last = 0
    for i, m in enumerate(_TOOL_CALL_RE.finditer(text)):
        plain_parts.append(text[last : m.start()])
        last = m.end()
        try:
            call = json.loads(m.group(1))
        except json.JSONDecodeError:
            plain_parts.append(m.group(0))
            continue
        arguments = call.get("arguments")
        tool_uses.append({
            "type": "tool_use",
            "id": f"toolu_{rollout_id}_{index}_{i}",
            "name": str(call.get("name") or ""),
            "input": arguments if isinstance(arguments, dict) else {},
        })
    plain_parts.append(text[last:])
    plain = "".join(plain_parts).strip()
    content: list[dict[str, Any]] = []
    if plain or not tool_uses:
        content.append({"type": "text", "text": plain})
    content.extend(tool_uses)
    return content, bool(tool_uses)


def _unsupported_block_type(messages: list[dict[str, Any]]) -> str | None:
    """First content-block type the training route cannot translate (images
    and documents have no token-native path through /chat/)."""
    for m in messages or []:
        content = m.get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") in ("image", "document"):
                    return b["type"]
    return None


def _anthropic_text(content: Any) -> str:
    """Flatten Anthropic message content (str or content-block list) to the
    plain text the policy chat template consumes. tool_result blocks flatten
    to their inner text so ReAct loops that echo observations keep working;
    unknown block types serialize verbatim rather than dropping silently."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            parts.append(block.get("text") or "")
        elif block.get("type") == "tool_result":
            parts.append(_anthropic_text(block.get("content")))
        else:
            parts.append(json.dumps(block, ensure_ascii=False))
    return "\n".join(p for p in parts if p)


class TemplateDriftError(ValueError):
    """A turn's server-rendered prompt did not extend the previous turn's
    prompt+completion exactly. Training on this episode would optimize a
    tokenization the policy never saw (the retokenization-drift failure mode
    documented by Agent Lightning / verl), so the rollout is rejected."""


@dataclass
class Span:
    """One captured model request inside a rollout."""

    rollout_id: str
    request_index: int
    path: str
    request: dict[str, Any]
    response: dict[str, Any]
    prompt_token_ids: list[int] | None
    completion_token_ids: list[int] | None
    completion_logprobs: list[float] | None
    started_at: float
    duration_s: float
    token_fidelity: str  # "captured" | "absent"


class SpanStore:
    """Appends spans as JSONL, one file per rollout. Thread-safe: the proxy
    serves each rollout on its own port but harnesses may pipeline requests
    across threads."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock_for(self, rollout_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(rollout_id, threading.Lock())

    def path_for(self, rollout_id: str) -> Path:
        return self.root / f"{rollout_id}.jsonl"

    def append(self, span: Span) -> None:
        record = {
            "rollout_id": span.rollout_id,
            "request_index": span.request_index,
            "path": span.path,
            "request": span.request,
            "response": span.response,
            "prompt_token_ids": span.prompt_token_ids,
            "completion_token_ids": span.completion_token_ids,
            "completion_logprobs": span.completion_logprobs,
            "started_at": span.started_at,
            "duration_s": span.duration_s,
            "token_fidelity": span.token_fidelity,
        }
        with self._lock_for(span.rollout_id):
            with open(self.path_for(span.rollout_id), "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self, rollout_id: str) -> list[dict[str, Any]]:
        path = self.path_for(rollout_id)
        if not path.exists():
            return []
        with self._lock_for(rollout_id):
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]


def _extract_token_ids(path: str, body: dict[str, Any]) -> tuple[
    list[int] | None, list[int] | None, list[float] | None
]:
    """Pull (prompt_ids, completion_ids, completion_logprobs) out of a vLLM
    response. `return_token_ids` puts prompt_token_ids at the top level and
    token_ids on each choice; chat logprobs ride choices[0].logprobs.content.
    Returns Nones when the upstream didn't honor the injection (e.g. the
    Anthropic route)."""
    prompt_ids = body.get("prompt_token_ids")
    choices = body.get("choices") or []
    if not choices:
        return prompt_ids, None, None
    choice = choices[0]
    completion_ids = choice.get("token_ids")
    logprobs: list[float] | None = None
    lp = choice.get("logprobs")
    if isinstance(lp, dict):
        content = lp.get("content") or lp.get("text_offset") and None
        if isinstance(content, list) and content and isinstance(content[0], dict):
            logprobs = [float(t.get("logprob", 0.0)) for t in content]
    return prompt_ids, completion_ids, logprobs


@dataclass
class Episode:
    """A rendered trajectory ready for the trainer: one token stream with
    assistant-only loss masking (mask 1 = trained, 0 = context/tool/user)."""

    rollout_id: str
    input_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]  # aligned with input_ids; 0.0 where masked
    num_turns: int
    # Final assistant text, for reward functions that score text rather than
    # tokens (TRL reward signature takes `completions`).
    final_completion_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def render_episode(
    rollout_id: str,
    spans: list[dict[str, Any]],
    *,
    allow_last_turn_fallback: bool = True,
) -> Episode:
    """Reassemble captured spans into a template-exact, assistant-only-masked
    episode.

    Delta tokenization (verl multi-turn reference): turn k's prompt must equal
    turn k-1's (prompt + completion) plus a delta of tool/template tokens. All
    prompt and delta tokens are masked; only completion tokens carry loss.
    Because every id comes from the SERVER's tokenization (return_token_ids),
    the episode is exactly what the policy saw — no re-render drift.

    Fallback: some chat templates CANNOT satisfy strict prefix extension —
    Qwen3, for example, rewrites prior assistant turns on re-render (thinking
    blocks, generation-prompt suffixes), so turn k+1's prompt diverges from
    turn k's prompt+completion. When stitching
    fails and `allow_last_turn_fallback` is set, the episode degrades to the
    FINAL span only (its prompt masked, its completion trained): a partial
    but token-exact training signal, marked metadata.render_fallback =
    "last_turn". Token-fidelity rejection is never relaxed.
    """
    if not spans:
        raise ValueError(f"rollout {rollout_id}: no spans captured")
    for i, span in enumerate(spans):
        if span.get("token_fidelity") != "captured":
            raise TemplateDriftError(
                f"rollout {rollout_id} turn {i}: token ids absent "
                f"(path={span.get('path')}). The Anthropic route may not "
                "support return_token_ids on this vLLM version — switch "
                "harness_protocol to 'openai' or put a token-preserving shim "
                "upstream."
            )

    def _span_parts(span: dict[str, Any]) -> tuple[list[int], list[int], list[float]]:
        prompt_ids = list(span["prompt_token_ids"])
        completion_ids = list(span["completion_token_ids"])
        completion_lps = span.get("completion_logprobs") or [0.0] * len(completion_ids)
        if len(completion_lps) != len(completion_ids):
            # Logprob capture is best-effort (masked to 0); the loss mask, not
            # the logprobs, is what GRPO correctness depends on at this layer.
            completion_lps = (completion_lps + [0.0] * len(completion_ids))[: len(completion_ids)]
        return prompt_ids, completion_ids, completion_lps

    final_text = ""
    last = spans[-1].get("response") or {}
    choices = last.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        final_text = msg.get("content") or choices[0].get("text") or ""
    elif isinstance(last.get("content"), list):
        # Anthropic Messages shape ({"content": [{"type": "text", ...}]}).
        final_text = "\n".join(
            b.get("text") or ""
            for b in last["content"]
            if isinstance(b, dict) and b.get("type") == "text"
        )

    input_ids: list[int] = []
    loss_mask: list[int] = []
    logprobs: list[float] = []
    context: list[int] = []  # prompt+completion of everything rendered so far

    try:
        for i, span in enumerate(spans):
            prompt_ids, completion_ids, completion_lps = _span_parts(span)

            if i == 0:
                delta = prompt_ids
            else:
                if prompt_ids[: len(context)] != context:
                    raise TemplateDriftError(
                        f"rollout {rollout_id} turn {i}: server prompt does not "
                        "extend the previous turn's prompt+completion "
                        f"(context {len(context)} ids, prompt {len(prompt_ids)} "
                        "ids). Serving/training chat-template drift — check the "
                        "chat-template pin."
                    )
                delta = prompt_ids[len(context):]

            input_ids.extend(delta)
            loss_mask.extend([0] * len(delta))
            logprobs.extend([0.0] * len(delta))
            input_ids.extend(completion_ids)
            loss_mask.extend([1] * len(completion_ids))
            logprobs.extend(completion_lps)
            context = prompt_ids + completion_ids
    except TemplateDriftError:
        if not allow_last_turn_fallback or len(spans) < 2:
            raise
        # Last-turn fallback (see docstring): the final span is internally
        # token-exact even when the template rewrites history, so train on
        # its completion alone with the full rendered prompt masked.
        log.warning(
            "rollout %s: chat template rewrites prior turns; falling back to "
            "last-turn-only episode (%d earlier completions untrained)",
            rollout_id,
            len(spans) - 1,
        )
        prompt_ids, completion_ids, completion_lps = _span_parts(spans[-1])
        return Episode(
            rollout_id=rollout_id,
            input_ids=prompt_ids + completion_ids,
            loss_mask=[0] * len(prompt_ids) + [1] * len(completion_ids),
            logprobs=[0.0] * len(prompt_ids) + completion_lps,
            num_turns=len(spans),
            final_completion_text=final_text,
            metadata={
                "total_tokens": len(prompt_ids) + len(completion_ids),
                "render_fallback": "last_turn",
            },
        )

    return Episode(
        rollout_id=rollout_id,
        input_ids=input_ids,
        loss_mask=loss_mask,
        logprobs=logprobs,
        num_turns=len(spans),
        final_completion_text=final_text,
        metadata={"total_tokens": len(input_ids)},
    )


class TrajectoryProxy:
    """Per-rollout HTTP proxy in front of the policy server.

    `register(rollout_id)` binds a fresh localhost port for that rollout;
    every POST to a captured path on that port is forwarded upstream with
    token capture injected and the request/response pair persisted to the
    SpanStore under that rollout id. Registration is dynamic because the
    rollout runner's concurrency pool creates and retires rollouts
    continuously; `unregister` frees the port when a rollout finishes.
    """

    def __init__(
        self,
        *,
        upstream_url: str,
        store: SpanStore,
        upstream_timeout_s: int = 600,
        tokenizer: Any | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        if not upstream_url.startswith(("http://", "https://")):
            raise ValueError(
                f"upstream_url must be http(s), got {upstream_url!r}"
            )
        self.upstream_url = upstream_url.rstrip("/")
        self.store = store
        self.upstream_timeout_s = upstream_timeout_s
        # Decodes completion ids for the OpenAI-shaped response the harness
        # reads. TRL's vllm-serve /chat/ returns token ids only (no text) —
        # see _chat_adapter below.
        self.tokenizer = tokenizer
        # Forwarded to the server's chat template on every request. The
        # harness default is {"enable_thinking": False}: thinking models
        # (Qwen3) strip <think> blocks from PRIOR assistant turns when
        # re-rendering, so turn k+1's prompt no longer extends turn k's
        # prompt+completion and render_episode rejects the rollout as
        # template drift.
        self.chat_template_kwargs = chat_template_kwargs
        self._servers: dict[str, ThreadingHTTPServer] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._counters: dict[str, int] = {}
        self._guard = threading.Lock()

    def register(self, rollout_id: str) -> str:
        """Bind a proxy port for this rollout and return its OpenAI-style
        base URL (http://127.0.0.1:PORT/v1)."""
        handler = self._make_handler(rollout_id)
        with self._guard:
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            server.daemon_threads = True
            thread = threading.Thread(
                target=server.serve_forever, name=f"traj-proxy-{rollout_id}", daemon=True
            )
            thread.start()
            self._servers[rollout_id] = server
            self._threads[rollout_id] = thread
        port = server.server_address[1]
        log.info("trajectory proxy: rollout %s on 127.0.0.1:%d", rollout_id, port)
        return f"http://127.0.0.1:{port}/v1"

    def unregister(self, rollout_id: str) -> None:
        with self._guard:
            server = self._servers.pop(rollout_id, None)
            thread = self._threads.pop(rollout_id, None)
        if server:
            server.shutdown()
            server.server_close()
        if thread:
            thread.join(timeout=5)

    def stop(self) -> None:
        with self._guard:
            servers = list(self._servers.values())
            threads = list(self._threads.values())
            self._servers.clear()
            self._threads.clear()
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)

    def __enter__(self) -> "TrajectoryProxy":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _make_handler(self, rollout_id: str):
        proxy = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # quiet; worker log owns noise
                pass

            def do_GET(self):
                self._proxy(passthrough=True)

            def do_POST(self):
                # Claude Code appends query strings (/v1/messages?beta=true);
                # dispatch on the bare path.
                self._proxy(passthrough=self.route() not in _CAPTURED_POST_PATHS)

            def route(self) -> str:
                return self.path.split("?", 1)[0]

            def _send(self, status: int, payload: bytes, content_type: str = "application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _policy_chat(self, body: dict[str, Any]) -> dict[str, Any] | None:
                """POST to TRL vllm-serve /chat/; on failure send the error to
                the harness and return None."""
                req = urllib.request.Request(
                    proxy.upstream_url + "/chat/",
                    data=json.dumps(body).encode(),
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=proxy.upstream_timeout_s) as resp:
                        return json.loads(resp.read())
                except urllib.error.HTTPError as e:
                    self._send(e.code, e.read())
                except Exception as e:
                    self._send(
                        502,
                        json.dumps({"error": f"trajectory proxy upstream error: {e}"}).encode(),
                    )
                return None

            def _messages_adapter(self, request_json: dict[str, Any], started: float):
                """Anthropic Messages API -> TRL vllm-serve /chat/ (VS-374).

                A verbatim forward 404s (TRL vllm-serve has no /v1/messages
                route), so this translates onto the same token-native /chat/
                call as the OpenAI adapter — identical token-fidelity
                guarantee. Native tool_use rides the policy chat template:
                the `tools` array converts to OpenAI function format (Qwen
                templates render it natively), hermes <tool_call> tags in the
                decoded completion come back as tool_use blocks, and prior
                tool_use/tool_result turns re-render through the template's
                own tool_calls / role:"tool" paths. Streaming clients (the
                Claude Agent SDK) get a synthesized SSE sequence from the
                buffered response.
                """
                unsupported = _unsupported_block_type(request_json.get("messages") or [])
                if unsupported:
                    self._send(
                        400,
                        json.dumps({
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": (
                                    f"{unsupported} content blocks are not "
                                    "supported on the training route (no "
                                    "token-native path through the policy "
                                    "server); send text and tool blocks only"
                                ),
                            },
                        }).encode(),
                    )
                    return

                messages = _anthropic_messages_to_chat(
                    request_json.get("system"), request_json.get("messages") or []
                )
                max_tokens = int(request_json.get("max_tokens") or 1024)
                body = {
                    "messages": [messages],
                    "n": 1,
                    "temperature": float(request_json.get("temperature") or 1.0),
                    "top_p": float(request_json.get("top_p") or 1.0),
                    "max_tokens": max_tokens,
                    "logprobs": 0,  # sampled token's logprob only
                }
                tools = request_json.get("tools")
                if tools:
                    body["tools"] = _anthropic_tools_to_openai(tools)
                if proxy.chat_template_kwargs:
                    body["chat_template_kwargs"] = proxy.chat_template_kwargs
                chat = self._policy_chat(body)
                if chat is None:
                    return  # error already sent

                prompt_ids = list((chat.get("prompt_ids") or [[]])[0])
                completion_ids = list((chat.get("completion_ids") or [[]])[0])
                lps = chat.get("logprobs")
                completion_lps = (
                    [float(t[0]) if t and t[0] is not None else 0.0 for t in lps[0]]
                    if lps
                    else [0.0] * len(completion_ids)
                )
                text = ""
                if proxy.tokenizer is not None:
                    text = proxy.tokenizer.decode(completion_ids, skip_special_tokens=True)
                index = proxy._counters.get(rollout_id, 0)
                content, has_tool_use = (
                    _parse_tool_calls(text, rollout_id, index)
                    if tools
                    else ([{"type": "text", "text": text}], False)
                )
                if has_tool_use:
                    stop_reason = "tool_use"
                elif len(completion_ids) >= max_tokens:
                    stop_reason = "max_tokens"
                else:
                    stop_reason = "end_turn"
                response_json = {
                    "id": f"msg-{rollout_id}-{index}",
                    "type": "message",
                    "role": "assistant",
                    "model": request_json.get("model") or "policy",
                    "content": content,
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": len(prompt_ids),
                        "output_tokens": len(completion_ids),
                    },
                }
                # Persist BEFORE responding: the harness may exit the moment
                # it reads this response, and the rollout runner loads spans as
                # soon as the harness exits — a respond-first order can lose
                # the final turn.
                proxy._counters[rollout_id] = index + 1
                proxy.store.append(
                    Span(
                        rollout_id=rollout_id,
                        request_index=index,
                        path=self.route(),
                        request=request_json,
                        response=response_json,
                        prompt_token_ids=prompt_ids,
                        completion_token_ids=completion_ids,
                        completion_logprobs=completion_lps,
                        started_at=started,
                        duration_s=time.time() - started,
                        token_fidelity="captured" if prompt_ids and completion_ids else "absent",
                    )
                )
                if request_json.get("stream"):
                    self._send_anthropic_sse(response_json)
                else:
                    self._send(200, json.dumps(response_json, ensure_ascii=False).encode())

            def _send_anthropic_sse(self, message: dict[str, Any]) -> None:
                """Synthesize the Anthropic SSE event sequence from a buffered
                message. Spans can't be reassembled from true streams, so the
                upstream call is always buffered; streaming clients (the
                Claude Agent SDK streams by default) still get a
                protocol-correct event sequence — one delta per block."""
                start_message = {
                    **{k: v for k, v in message.items() if k != "content"},
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": message["usage"]["input_tokens"],
                        "output_tokens": 0,
                    },
                }
                events: list[dict[str, Any]] = [
                    {"type": "message_start", "message": start_message}
                ]
                for i, block in enumerate(message["content"]):
                    if block["type"] == "tool_use":
                        events.append({
                            "type": "content_block_start",
                            "index": i,
                            "content_block": {
                                "type": "tool_use",
                                "id": block["id"],
                                "name": block["name"],
                                "input": {},
                            },
                        })
                        events.append({
                            "type": "content_block_delta",
                            "index": i,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(
                                    block["input"], ensure_ascii=False
                                ),
                            },
                        })
                    else:
                        events.append({
                            "type": "content_block_start",
                            "index": i,
                            "content_block": {"type": "text", "text": ""},
                        })
                        events.append({
                            "type": "content_block_delta",
                            "index": i,
                            "delta": {"type": "text_delta", "text": block.get("text") or ""},
                        })
                    events.append({"type": "content_block_stop", "index": i})
                events.append({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": message["stop_reason"],
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": message["usage"]["output_tokens"]},
                })
                events.append({"type": "message_stop"})
                payload = b"".join(
                    f"event: {e['type']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n".encode()
                    for e in events
                )
                self._send(200, payload, content_type="text/event-stream")

            def _count_tokens_adapter(self, request_json: dict[str, Any]) -> None:
                """/v1/messages/count_tokens via the trainer tokenizer (the
                Claude Agent SDK calls this for context management). Not a
                model call: no span is recorded."""
                if proxy.tokenizer is None or not hasattr(
                    proxy.tokenizer, "apply_chat_template"
                ):
                    self._send(
                        501,
                        json.dumps({
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": (
                                    "count_tokens unavailable: "
                                    "proxy has no chat-template tokenizer"
                                ),
                            },
                        }).encode(),
                    )
                    return
                try:
                    messages = _anthropic_messages_to_chat(
                        request_json.get("system"), request_json.get("messages") or []
                    )
                    kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": True}
                    tools = request_json.get("tools")
                    if tools:
                        kwargs["tools"] = _anthropic_tools_to_openai(tools)
                    ids = proxy.tokenizer.apply_chat_template(messages, **kwargs)
                    self._send(200, json.dumps({"input_tokens": len(ids)}).encode())
                except Exception as e:
                    self._send(
                        400,
                        json.dumps({
                            "type": "error",
                            "error": {"type": "invalid_request_error", "message": str(e)},
                        }).encode(),
                    )

            def _chat_adapter(self, request_json: dict[str, Any], started: float):
                """OpenAI chat.completions -> TRL vllm-serve /chat/.

                The policy server is TRL's vllm-serve, which exposes its OWN
                /chat/ schema (token ids + logprobs, no OpenAI routes), so a
                verbatim forward would 404 every rollout. This
                translates the harness's OpenAI request, decodes the
                completion for the response text, and records the span from
                the ids the server ALREADY returns (no return_token_ids
                injection needed on this route).
                """
                body = {
                    "messages": [request_json.get("messages") or []],
                    "n": 1,
                    "temperature": float(request_json.get("temperature") or 1.0),
                    "top_p": float(request_json.get("top_p") or 1.0),
                    # trl's default max_tokens=16 truncates any real answer.
                    "max_tokens": int(
                        request_json.get("max_tokens")
                        or request_json.get("max_completion_tokens")
                        or 1024
                    ),
                    "logprobs": 0,  # sampled token's logprob only
                }
                if proxy.chat_template_kwargs:
                    body["chat_template_kwargs"] = proxy.chat_template_kwargs
                chat = self._policy_chat(body)
                if chat is None:
                    return  # error already sent

                prompt_ids = list((chat.get("prompt_ids") or [[]])[0])
                completion_ids = list((chat.get("completion_ids") or [[]])[0])
                lps = chat.get("logprobs")
                completion_lps = (
                    [float(t[0]) if t and t[0] is not None else 0.0 for t in lps[0]]
                    if lps
                    else [0.0] * len(completion_ids)
                )
                text = ""
                if proxy.tokenizer is not None:
                    text = proxy.tokenizer.decode(completion_ids, skip_special_tokens=True)
                response_json = {
                    "id": f"chatcmpl-{rollout_id}-{proxy._counters.get(rollout_id, 0)}",
                    "object": "chat.completion",
                    "created": int(started),
                    "model": request_json.get("model") or "policy",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                            "token_ids": completion_ids,
                            "logprobs": {
                                "content": [{"logprob": lp} for lp in completion_lps]
                            },
                        }
                    ],
                    "prompt_token_ids": prompt_ids,
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": len(completion_ids),
                        "total_tokens": len(prompt_ids) + len(completion_ids),
                    },
                }
                # Persist before responding (see _messages_adapter).
                index = proxy._counters.get(rollout_id, 0)
                proxy._counters[rollout_id] = index + 1
                proxy.store.append(
                    Span(
                        rollout_id=rollout_id,
                        request_index=index,
                        path=self.route(),
                        request=request_json,
                        response=response_json,
                        prompt_token_ids=prompt_ids,
                        completion_token_ids=completion_ids,
                        completion_logprobs=completion_lps,
                        started_at=started,
                        duration_s=time.time() - started,
                        token_fidelity="captured" if prompt_ids and completion_ids else "absent",
                    )
                )
                self._send(200, json.dumps(response_json, ensure_ascii=False).encode())

            def _proxy(self, *, passthrough: bool):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                started = time.time()
                body_out = raw
                request_json: dict[str, Any] | None = None

                if raw and self.route() == "/v1/messages/count_tokens":
                    try:
                        count_request = json.loads(raw)
                    except json.JSONDecodeError:
                        count_request = None
                    if count_request is not None:
                        self._count_tokens_adapter(count_request)
                        return

                if not passthrough and raw and self.route() == "/v1/chat/completions":
                    try:
                        chat_request = json.loads(raw)
                    except json.JSONDecodeError:
                        chat_request = None
                    if chat_request is not None:
                        self._chat_adapter(chat_request, started)
                        return

                if not passthrough and raw and self.route() == "/v1/messages":
                    try:
                        anthropic_request = json.loads(raw)
                    except json.JSONDecodeError:
                        anthropic_request = None
                    if anthropic_request is not None:
                        self._messages_adapter(anthropic_request, started)
                        return

                if not passthrough and raw:
                    try:
                        request_json = json.loads(raw)
                    except json.JSONDecodeError:
                        request_json = None
                    if request_json is not None and self.route() != "/v1/messages":
                        # Token-in-token-out capture (Agent Lightning v0.2 /
                        # vLLM >= 0.10.2) + per-token logprobs for GRPO.
                        request_json["return_token_ids"] = True
                        request_json.setdefault("logprobs", True)
                        # Streaming spans can't be reassembled into episodes;
                        # force a buffered response. Documented MVP constraint.
                        request_json["stream"] = False
                        body_out = json.dumps(request_json).encode()

                req = urllib.request.Request(
                    proxy.upstream_url + self.path,
                    data=body_out if self.command == "POST" else None,
                    method=self.command,
                )
                for header in ("Content-Type", "Authorization", "x-api-key", "anthropic-version"):
                    value = self.headers.get(header)
                    if value:
                        req.add_header(header, value)

                try:
                    with urllib.request.urlopen(req, timeout=proxy.upstream_timeout_s) as resp:
                        payload = resp.read()
                        status = resp.status
                        content_type = resp.headers.get("Content-Type", "application/json")
                except urllib.error.HTTPError as e:
                    payload = e.read()
                    status = e.code
                    content_type = "application/json"
                except Exception as e:
                    payload = json.dumps(
                        {"error": f"trajectory proxy upstream error: {e}"}
                    ).encode()
                    status = 502
                    content_type = "application/json"

                # Persist before responding (see _messages_adapter).
                if not passthrough and request_json is not None:
                    try:
                        response_json = json.loads(payload)
                    except json.JSONDecodeError:
                        response_json = None
                    if response_json is not None:
                        prompt_ids, completion_ids, completion_lps = _extract_token_ids(
                            self.path, response_json
                        )
                        index = proxy._counters.get(rollout_id, 0)
                        proxy._counters[rollout_id] = index + 1
                        proxy.store.append(
                            Span(
                                rollout_id=rollout_id,
                                request_index=index,
                                path=self.route(),
                                request=request_json,
                                response=response_json,
                                prompt_token_ids=prompt_ids,
                                completion_token_ids=completion_ids,
                                completion_logprobs=completion_lps,
                                started_at=started,
                                duration_s=time.time() - started,
                                token_fidelity=(
                                    "captured" if prompt_ids and completion_ids else "absent"
                                ),
                            )
                        )

                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        return _Handler
