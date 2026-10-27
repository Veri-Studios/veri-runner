"""Training job detail telemetry (2026-10-26), runtime half: the lifecycle
events the runtime raises through `event_fn` (model_loaded, first_step,
checkpoint_saved) and the per-step `time/*` keys the progress callback adds
to each logged step. No torch / transformers / trl: fakes only."""

from __future__ import annotations

import inspect
import json
import logging
import sys
import types

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


# ---- per-step phase timing (time/*) + first_step / checkpoint_saved ----


class _Clock:
    """perf_counter stand-in the test advances by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _State:
    def __init__(self, step, max_steps=10, zero=True):
        self.global_step = step
        self.max_steps = max_steps
        self.is_world_process_zero = zero


class _GrpoLikeTrainer:
    """The TRL 1.7.1 GRPOTrainer surface the timers wrap: _generate (which
    syncs vLLM weights first), _calculate_rewards, vllm_generation.sync_weights."""

    def __init__(self, clock, gen_s=3.0, sync_s=0.5, reward_s=1.0):
        self.callbacks = []
        self.clock = clock
        self.gen_s, self.sync_s, self.reward_s = gen_s, sync_s, reward_s
        self.vllm_generation = types.SimpleNamespace(sync_weights=self._sync)

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def _sync(self):
        self.clock.advance(self.sync_s)

    def _generate(self, prompts):
        self.vllm_generation.sync_weights()
        self.clock.advance(self.gen_s)
        return ["c"] * len(prompts)

    def _calculate_rewards(self, *a):
        self.clock.advance(self.reward_s)
        return [1.0]

    def train_step(self, update_s=2.0):
        """What one optimizer step does between on_step_begin and on_step_end."""
        self._generate(["p"])
        self._calculate_rewards()
        self.clock.advance(update_s)


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(rt.time, "perf_counter", c)
    tf = types.ModuleType("transformers")
    tf.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "torch", None)  # no in-process torch: no mem_peak
    return c


def _attach(trainer, **kw):
    steps, events, phases = [], [], []
    rt._attach_progress_callback(
        trainer,
        lambda s, t, m: steps.append((s, t, m)),
        phase_fn=(lambda p, s: phases.append((p, s))) if kw.get("phases") else None,
        event_fn=lambda t, m, meta: events.append((t, meta)),
    )
    return trainer.callbacks[0], steps, events, phases


def _ctl(save=False):
    return types.SimpleNamespace(should_save=save)


def test_grpo_step_is_split_into_gen_reward_sync_and_update(clock):
    tr = _GrpoLikeTrainer(clock)
    cb, steps, events, _ = _attach(tr)
    cb.on_step_begin(None, _State(0), _ctl())
    tr.train_step(update_s=2.0)
    cb.on_step_end(None, _State(1), _ctl())
    cb.on_log(None, _State(1), _ctl(), logs={"loss": 0.5, "completions/text": "x"})
    (step, total, m), = steps
    assert (step, total) == (1, 10)
    assert m["loss"] == 0.5 and "completions/text" not in m
    assert m["time/step_s"] == pytest.approx(6.5)
    assert m["time/gen_s"] == pytest.approx(3.0)  # the weight sync inside _generate counts once
    assert m["time/weight_sync_s"] == pytest.approx(0.5)
    assert m["time/reward_s"] == pytest.approx(1.0)
    assert m["time/update_s"] == pytest.approx(2.0)
    assert isinstance(m["time/ts"], float)
    assert "time/checkpoint_save_s" not in m
    assert events == [("first_step", {"step_s": pytest.approx(6.5), "step": 1})]


def test_phase_timers_reset_every_step_and_first_step_fires_once(clock):
    tr = _GrpoLikeTrainer(clock)
    cb, steps, events, _ = _attach(tr)
    for step in (1, 2):
        cb.on_step_begin(None, _State(step - 1), _ctl())
        tr.train_step(update_s=float(step))
        cb.on_step_end(None, _State(step), _ctl())
        cb.on_log(None, _State(step), _ctl(), logs={"loss": 1.0})
    assert [m["time/update_s"] for _, _, m in steps] == [pytest.approx(1.0), pytest.approx(2.0)]
    assert [m["time/gen_s"] for _, _, m in steps] == [pytest.approx(3.0)] * 2
    assert [t for t, _ in events] == ["first_step"]


def test_steps_between_two_logs_report_their_mean(clock):
    tr = _GrpoLikeTrainer(clock)
    cb, steps, _, _ = _attach(tr)
    for step, update in ((1, 1.0), (2, 3.0)):
        cb.on_step_begin(None, _State(step - 1), _ctl())
        tr.train_step(update_s=update)
        cb.on_step_end(None, _State(step), _ctl())
    cb.on_log(None, _State(2), _ctl(), logs={"loss": 1.0})
    (_, _, m), = steps
    assert m["time/update_s"] == pytest.approx(2.0)
    assert m["time/step_s"] == pytest.approx(6.5)


def test_sft_like_trainer_reports_step_and_update_only(clock):
    class _SftLike:
        def __init__(self):
            self.callbacks = []

        def add_callback(self, cb):
            self.callbacks.append(cb)

    tr = _SftLike()
    cb, steps, _, _ = _attach(tr)
    cb.on_step_begin(None, _State(0), _ctl())
    clock.advance(4.0)
    cb.on_step_end(None, _State(1), _ctl())
    cb.on_log(None, _State(1), _ctl(), logs={"loss": 2.0})
    m = steps[0][2]
    assert m["time/step_s"] == pytest.approx(4.0) and m["time/update_s"] == pytest.approx(4.0)
    assert not {"time/gen_s", "time/reward_s", "time/weight_sync_s"} & set(m)


def test_save_step_row_waits_for_the_save_and_carries_its_seconds(clock):
    tr = _GrpoLikeTrainer(clock)
    cb, steps, events, phases = _attach(tr, phases=True)
    cb.on_step_begin(None, _State(4), _ctl())
    tr.train_step()
    cb.on_step_end(None, _State(5), _ctl(save=True))
    cb.on_log(None, _State(5), _ctl(save=True), logs={"loss": 0.4})
    assert steps == []  # held: HF logs before it saves
    clock.advance(12.0)  # the Trainer writes checkpoint-5
    cb.on_save(None, _State(5), _ctl())
    (step, _, m), = steps
    assert step == 5 and m["loss"] == 0.4
    assert m["time/checkpoint_save_s"] == pytest.approx(12.0)
    assert ("checkpoint_saved", {"step": 5, "local_save_s": pytest.approx(12.0)}) in events
    assert phases == [("saving", 5), ("training", 5)]


def test_save_on_a_step_that_did_not_log_still_reports_the_save(clock):
    tr = _GrpoLikeTrainer(clock)
    cb, steps, events, _ = _attach(tr)
    cb.on_step_begin(None, _State(2), _ctl())
    tr.train_step()
    cb.on_step_end(None, _State(3), _ctl(save=True))
    clock.advance(5.0)
    cb.on_save(None, _State(3), _ctl())
    (step, _, m), = steps
    assert step == 3 and m["time/checkpoint_save_s"] == pytest.approx(5.0) and "time/ts" in m
    assert [t for t, _ in events] == ["first_step", "checkpoint_saved"]


def test_held_row_is_flushed_at_train_end(clock):
    tr = _GrpoLikeTrainer(clock)
    cb, steps, _, _ = _attach(tr)
    cb.on_step_begin(None, _State(0), _ctl())
    cb.on_step_end(None, _State(1), _ctl(save=True))
    cb.on_log(None, _State(1), _ctl(), logs={"loss": 1.0})
    cb.on_train_end(None, _State(1), _ctl())
    assert [s for s, _, _ in steps] == [1]


def test_peak_gpu_memory_rides_each_logged_step_when_torch_is_loaded(clock, monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 0,
        max_memory_allocated=lambda device=None: 7 * 2**30,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    tr = _GrpoLikeTrainer(clock)
    cb, steps, _, _ = _attach(tr)
    cb.on_log(None, _State(1), _ctl(), logs={"loss": 1.0})
    assert steps[0][2]["sys/gpu.0.mem_peak_bytes"] == float(7 * 2**30)


def test_rank_zero_progress_file_carries_the_timing_keys(clock, tmp_path):
    """Multi-GPU child: rank 0's progress_fn is RankZeroProgressFile.step, so the
    parent tails the same time/* keys out of progress.jsonl."""
    progress = rt.RankZeroProgressFile(str(tmp_path / "progress.jsonl"))
    tr = _GrpoLikeTrainer(clock)
    rt._attach_progress_callback(
        tr, progress.step, phase_fn=progress.phase, event_fn=progress.event
    )
    cb = tr.callbacks[0]
    cb.on_step_begin(None, _State(0), _ctl())
    clock.advance(2.0)
    cb.on_step_end(None, _State(1), _ctl())
    cb.on_log(None, _State(1), _ctl(), logs={"loss": 1.0})
    lines = [json.loads(x) for x in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert lines[0]["event"] == "first_step"
    step_rec = next(x for x in lines if "step" in x and "metrics" in x)
    assert step_rec["metrics"]["time/step_s"] == pytest.approx(2.0)
    assert "time/ts" in step_rec["metrics"]


def test_non_zero_rank_reports_nothing(clock):
    tr = _GrpoLikeTrainer(clock)
    cb, steps, events, _ = _attach(tr, phases=True)
    cb.on_step_begin(None, _State(0, zero=False), _ctl())
    cb.on_step_end(None, _State(1, zero=False), _ctl(save=True))
    cb.on_log(None, _State(1, zero=False), _ctl(), logs={"loss": 1.0})
    cb.on_save(None, _State(1, zero=False), _ctl())
    assert steps == [] and events == []
