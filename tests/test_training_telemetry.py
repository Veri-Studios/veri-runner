"""Training job detail telemetry (2026-10-26), runtime half: the lifecycle
events the runtime raises through `event_fn` (model_loaded, first_step,
checkpoint_saved) and the per-step `time/*` keys the progress callback adds
to each logged step. No torch / transformers / trl: fakes only."""

from __future__ import annotations

import inspect
import logging

import pytest

from veri_runner import training_runtime as rt

_LOG = logging.getLogger("test")


# ---- model load: prefetch hook first, then model_loaded ----


def test_load_model_reporting_prefetches_then_loads_then_reports(monkeypatch):
    order: list[str] = []
    events: list[tuple] = []

    def fake_load(base_model, hyperparameters, log, **kw):
        order.append(f"load:{base_model}:{kw.get('launch_plan')}")
        return "model", "tok"

    monkeypatch.setattr(rt, "_load_model_and_tokenizer", fake_load)
    clock = iter([100.0, 112.5])
    monkeypatch.setattr(rt.time, "perf_counter", lambda: next(clock))
    model, tok = rt._load_model_reporting(
        "Qwen/Qwen3-4B", {}, _LOG,
        event_fn=lambda t, m, meta: events.append((t, meta)),
        model_prefetch_fn=lambda name: order.append(f"prefetch:{name}"),
        launch_plan={"world_size": 2},
    )
    assert (model, tok) == ("model", "tok")
    assert order == ["prefetch:Qwen/Qwen3-4B", "load:Qwen/Qwen3-4B:{'world_size': 2}"]
    assert events == [("model_loaded", {"duration_s": 12.5})]


def test_load_model_reporting_survives_a_broken_event_channel(monkeypatch):
    monkeypatch.setattr(rt, "_load_model_and_tokenizer", lambda *a, **k: ("m", "t"))

    def broken(*a):
        raise OSError("status callback down")

    assert rt._load_model_reporting("m", {}, _LOG, event_fn=broken) == ("m", "t")
    assert rt._load_model_reporting("m", {}, _LOG) == ("m", "t")  # no hooks at all


@pytest.mark.parametrize(
    "fn",
    [rt.run_grpo_training, rt.run_grpo_harness_training, rt.run_sft_text_training,
     rt.run_dpo_training],
)
def test_every_managed_method_accepts_the_worker_prefetch_hook(fn):
    """The worker passes model_prefetch_fn to grpo / grpo_harness / sft_text /
    dpo; a method without the parameter would fail every job with TypeError."""
    assert "model_prefetch_fn" in inspect.signature(fn).parameters
