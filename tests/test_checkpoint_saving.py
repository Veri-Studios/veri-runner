"""Periodic checkpoint saving + upload (VS-393).

Before this, all three managed TRL paths hardcoded save_strategy="no" and the
only write was after trainer.train() returned -- so a run that died lost 100% of
its GPU spend. These tests pin the two halves of the fix: the save config the
control plane hands down, and the background uploader that gets each checkpoint
off the box.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from veri_runner.training_runtime import (
    _attach_checkpoint_uploader,
    build_dpo_config_kwargs,
    build_grpo_config_kwargs,
    build_save_kwargs,
    build_sft_config_kwargs,
)


@pytest.fixture(autouse=True)
def _stub_transformers(monkeypatch):
    """`transformers` is not installed in this test env (it arrives with the GPU
    image), so stub the one symbol the uploader imports. Same approach the
    existing model-loading tests use."""
    if "transformers" in sys.modules:
        yield
        return
    mod = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    mod.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", mod)
    yield


ENABLED = {
    "enabled": True,
    "save_strategy": "steps",
    "save_steps": 0.1,
    "save_total_limit": 1,
    "save_safetensors": True,
    "save_only_model": False,
    "ignore_data_skip": False,
    "save_on_each_node": False,
}


class _AllKnobs:
    """Stands in for a TRL config class that accepts every save knob."""

    def __init__(
        self,
        max_prompt_length=None,
        beta=None,
        save_strategy=None,
        save_steps=None,
        save_total_limit=None,
        save_safetensors=None,
        save_only_model=None,
        ignore_data_skip=None,
        save_on_each_node=None,
    ):
        pass


# ---- build_save_kwargs ----


def test_absent_or_disabled_checkpointing_is_exactly_the_old_behaviour():
    """The control plane's feature flag has to be a real kill switch, not a
    request the runner may ignore. Every "off" spelling must yield save_strategy
    "no" -- i.e. one artifact after train() returns, as before VS-393."""
    for off in (None, {}, {"enabled": False}, {"enabled": False, "save_steps": 0.1}):
        assert build_save_kwargs(off) == {"save_strategy": "no"}, off


def test_enabled_checkpointing_passes_the_whole_policy_through():
    """The control plane computes the policy (design 5.3); the runner must not
    reinterpret it."""
    kwargs = build_save_kwargs(ENABLED)
    assert kwargs["save_strategy"] == "steps"
    assert kwargs["save_steps"] == 0.1
    # A resumable checkpoint and a disk-safe local limit are the two defaults
    # that carry the feature: save_only_model=True would make resume impossible,
    # and a local limit above 1 risks ENOSPC because HF frees the previous
    # checkpoint only AFTER writing the next.
    assert kwargs["save_only_model"] is False
    assert kwargs["save_total_limit"] == 1
    assert kwargs["save_safetensors"] is True
    assert kwargs["ignore_data_skip"] is False
    assert kwargs["save_on_each_node"] is False


def test_a_float_save_steps_survives_as_a_float():
    """HF reads a float in [0,1) as a fraction of total steps and an int as a
    literal step count. Coercing 0.1 to 0 would silently disable saving; coercing
    it to 1 would save every step."""
    assert build_save_kwargs(ENABLED)["save_steps"] == 0.1
    assert isinstance(build_save_kwargs(ENABLED)["save_steps"], float)

    as_count = dict(ENABLED, save_steps=50)
    assert build_save_kwargs(as_count)["save_steps"] == 50
    assert isinstance(build_save_kwargs(as_count)["save_steps"], int)


def test_unsupported_knobs_are_dropped_not_raised():
    """TRL drifts; the builders already adapt to the installed signature. A knob
    the installed version lacks must be dropped, because raising here would fail
    the whole job over a cosmetic option."""
    params = {"save_strategy": 1, "save_steps": 1}  # only these two accepted
    kwargs = build_save_kwargs(ENABLED, params)
    assert set(kwargs) == {"save_strategy", "save_steps"}


# ---- the three managed builders ----


@pytest.mark.parametrize(
    "builder,cls_kwarg",
    [
        (build_grpo_config_kwargs, "grpo_config_cls"),
        (build_sft_config_kwargs, "sft_config_cls"),
        (build_dpo_config_kwargs, "dpo_config_cls"),
    ],
)
def test_every_managed_builder_defaults_to_no_saving(builder, cls_kwarg):
    """BAD: the pre-VS-393 state, restated as a guarantee. Omitting the checkpoint
    block must not accidentally turn saving on for any method."""
    kwargs = builder(
        job_id="job-1",
        hyperparameters={},
        **{cls_kwarg: _AllKnobs},
    )
    assert kwargs["save_strategy"] == "no"
    assert "save_steps" not in kwargs or kwargs["save_steps"] is None


@pytest.mark.parametrize(
    "builder,cls_kwarg",
    [
        (build_grpo_config_kwargs, "grpo_config_cls"),
        (build_sft_config_kwargs, "sft_config_cls"),
        (build_dpo_config_kwargs, "dpo_config_cls"),
    ],
)
def test_every_managed_builder_honours_an_enabled_policy(builder, cls_kwarg):
    """All three paths hardcoded "no" before this change; none may be missed, or
    that method silently keeps losing work on a crash."""
    kwargs = builder(
        job_id="job-1",
        hyperparameters={},
        checkpoint=ENABLED,
        **{cls_kwarg: _AllKnobs},
    )
    assert kwargs["save_strategy"] == "steps"
    assert kwargs["save_steps"] == 0.1
    assert kwargs["save_total_limit"] == 1
    assert kwargs["save_only_model"] is False


# ---- the background uploader ----


class _FakeTrainer:
    def __init__(self):
        self.callbacks = []

    def add_callback(self, cb):
        self.callbacks.append(cb)


class _Args:
    def __init__(self, output_dir):
        self.output_dir = str(output_dir)


class _State:
    def __init__(self, step):
        self.global_step = step


def _fire(cb, tmp_path, step):
    (tmp_path / f"checkpoint-{step}").mkdir(exist_ok=True)
    cb.on_save(_Args(tmp_path), _State(step), None)


def _uploader(trainer, tmp_path, upload_fn, enabled=True):
    import logging

    _attach_checkpoint_uploader(
        trainer,
        {"checkpoint": {"enabled": enabled}},
        upload_fn,
        logging.getLogger("test"),
    )
    return trainer.callbacks[0] if trainer.callbacks else None


def test_no_callback_is_registered_when_checkpointing_is_off(tmp_path):
    """With the feature flag dark nothing may be uploaded -- and nothing should
    even be watching."""
    trainer = _FakeTrainer()
    assert _uploader(trainer, tmp_path, lambda d, s: None, enabled=False) is None
    assert trainer.callbacks == []

    trainer2 = _FakeTrainer()
    assert _uploader(trainer2, tmp_path, None) is None, "no upload_fn => no callback"


def test_a_save_uploads_off_the_training_thread(tmp_path):
    """A 24-40 GB upload must not block the training loop, so on_save has to
    return before the upload finishes."""
    started = threading.Event()
    release = threading.Event()
    seen = []

    def upload(d, step):
        started.set()
        release.wait(5)
        seen.append((d, step))

    cb = _uploader(_FakeTrainer(), tmp_path, upload)
    _fire(cb, tmp_path, 100)

    assert started.wait(5), "the upload must have begun"
    assert seen == [], "on_save must NOT have waited for the upload"
    release.set()
    cb.on_train_end(_Args(tmp_path), _State(100), None)
    # C0: the upload reads the hard-linked STAGED copy, never the Trainer's
    # own directory (which HF deletes on the next save).
    assert seen == [(str(tmp_path / ".upload" / "step-100"), 100)]


def test_the_final_upload_is_awaited_at_train_end(tmp_path):
    """The newest checkpoint is the one a resume wants. Without the join it rides
    a daemon thread that dies at interpreter exit, so the most valuable
    checkpoint is precisely the one that goes missing."""
    done = []
    cb = _uploader(_FakeTrainer(), tmp_path, lambda d, s: (time.sleep(0.2), done.append(s)))
    _fire(cb, tmp_path, 200)
    cb.on_train_end(_Args(tmp_path), _State(200), None)
    assert done == [200], "on_train_end must wait for the in-flight upload"


def test_an_upload_failure_never_fails_the_run(tmp_path):
    """The training loop is the expensive part. A failed upload means the control
    plane simply has no ready row for that step -- which is the correct outcome --
    and the next save gets another chance."""
    cb = _uploader(
        _FakeTrainer(), tmp_path, lambda d, s: (_ for _ in ()).throw(OSError("s3 down"))
    )
    _fire(cb, tmp_path, 300)  # must not raise
    cb.on_train_end(_Args(tmp_path), _State(300), None)


def test_a_missing_checkpoint_dir_is_skipped_not_uploaded(tmp_path):
    """on_save can fire for a directory that isn't there (a failed save). Passing
    a nonexistent path to the uploader would commit an empty checkpoint."""
    calls = []
    cb = _uploader(_FakeTrainer(), tmp_path, lambda d, s: calls.append(s))
    cb.on_save(_Args(tmp_path), _State(999), None)  # never created
    time.sleep(0.1)
    assert calls == []


def test_one_upload_in_flight_one_pending_and_the_in_flight_one_is_never_abandoned(tmp_path):
    """BAD (the old "newest wins" policy): a save that overtook an in-flight
    upload abandoned it, and HF had already deleted its directory, so on a slow
    uplink NO checkpoint ever committed. GOOD: the in-flight upload (100) always
    finishes; the newest save is the single pending one; a save arriving while
    one is pending replaces it and reports checkpoint_upload_skipped."""
    release = threading.Event()
    finished = []
    events = []

    def upload(d, step):
        if step == 100:
            release.wait(5)
        finished.append(step)

    trainer = _FakeTrainer()
    _attach_checkpoint_uploader(
        trainer, {"checkpoint": {"enabled": True}}, upload, logging.getLogger("test"),
        lambda t, m, meta: events.append((t, meta)),
    )
    cb = trainer.callbacks[0]
    _fire(cb, tmp_path, 100)   # in flight, blocked
    time.sleep(0.05)
    _fire(cb, tmp_path, 200)   # pending
    _fire(cb, tmp_path, 300)   # replaces 200 as pending
    time.sleep(0.05)
    assert finished == [], "nothing finished while 100 is blocked; 300 waits, never overtakes"
    assert [e[0] for e in events] == ["checkpoint_upload_skipped"]
    assert events[0][1]["step"] == 200 and events[0][1]["newer_step"] == 300
    assert not (tmp_path / ".upload" / "step-200").exists(), "the skipped stage is freed"
    release.set()
    cb.on_train_end(_Args(tmp_path), _State(300), None)
    assert finished == [100, 300], "the in-flight upload completed, then the pending one"
    assert cb.queue.skipped == [200]
    assert not (tmp_path / ".upload" / "step-100").exists(), "stages are freed after upload"


def test_the_staged_copy_survives_the_trainers_rotation(tmp_path):
    """HF writes checkpoint-N, rmtree's N-1 and only then fires on_save, so an
    upload reading the Trainer's directory raced a delete. The hard-linked
    stage keeps the bytes readable after checkpoint-N itself is gone."""
    import shutil

    got = {}
    release = threading.Event()

    def upload(d, step):
        release.wait(5)
        got[step] = {p.name: p.read_bytes() for p in Path(d).rglob("*") if p.is_file()}

    cb = _uploader(_FakeTrainer(), tmp_path, upload)
    ck = tmp_path / "checkpoint-100"
    ck.mkdir()
    (ck / "optimizer.pt").write_bytes(b"o" * 1000)
    (ck / "sub").mkdir()
    (ck / "sub" / "model.safetensors").write_bytes(b"m" * 500)
    cb.on_save(_Args(tmp_path), _State(100), None)
    shutil.rmtree(ck)  # the Trainer's next save rotates checkpoint-100 away
    release.set()
    cb.on_train_end(_Args(tmp_path), _State(100), None)
    assert got[100] == {"optimizer.pt": b"o" * 1000, "model.safetensors": b"m" * 500}


# ---- C0 disk guard ----


class _Param:
    def __init__(self, n, requires_grad=True):
        self._n = n
        self.requires_grad = requires_grad

    def numel(self):
        return self._n


class _Model:
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return iter(self._params)


def test_checkpoint_bytes_estimate_follows_the_runners_size_model():
    from veri_runner.training_runtime import estimate_checkpoint_bytes

    full = _Model([_Param(3 * 10**9), _Param(10**9)])
    assert estimate_checkpoint_bytes(full) == 6 * 4 * 10**9, "6 B/param bf16 full fine-tune"
    assert estimate_checkpoint_bytes(full, fsdp=True) == 8 * 4 * 10**9, "+2N pytorch_model_fsdp.bin"
    lora = _Model([_Param(4 * 10**9, requires_grad=False), _Param(20 * 10**6)])
    assert estimate_checkpoint_bytes(lora) == 10 * 20 * 10**6, "adapters: only the trainable params"


def test_disk_guard_turns_saves_off_when_three_checkpoints_do_not_fit(tmp_path):
    """BAD: a 7B full fine-tune on Vast (100 GB) wrote 42 GB checkpoints until
    ENOSPC killed the run. GOOD: the guard disables periodic saves up front,
    says why, and raises checkpoint_disabled_disk; the final artifact is
    unaffected. The control plane's local_disk_gb hint bounds a container's
    optimistic df."""
    from veri_runner.training_runtime import apply_checkpoint_disk_guard

    events = []
    model = _Model([_Param(7 * 10**9)])  # 42 GB per checkpoint, 126 GB for three
    on = ENABLED | {"local_disk_gb": 100}
    out = apply_checkpoint_disk_guard(
        on, model, str(tmp_path), logging.getLogger("t"),
        lambda t, m, meta: events.append((t, m, meta)), free_bytes=10**12,
    )
    assert out["save_strategy"] == "no" and out["enabled"] is False
    assert events[0][0] == "checkpoint_disabled_disk"
    assert events[0][2]["checkpoint_bytes"] == 42 * 10**9 and events[0][2]["level"] == "warning"
    assert "exceeds 100.0 GB" in events[0][1]

    fits = apply_checkpoint_disk_guard(
        on, model, str(tmp_path), logging.getLogger("t"), events.append, free_bytes=10**12
    )
    assert fits is on or fits["enabled"] is False  # still bounded by the 100 GB hint
    big = ENABLED | {"local_disk_gb": 256}
    assert apply_checkpoint_disk_guard(
        big, model, str(tmp_path), logging.getLogger("t"), None, free_bytes=150 * 10**9
    ) is big, "126 GB fits in 150 GB free"
    assert apply_checkpoint_disk_guard(None, model, str(tmp_path), logging.getLogger("t")) is None
    off = {"enabled": False, "save_strategy": "no"}
    assert apply_checkpoint_disk_guard(off, model, str(tmp_path), logging.getLogger("t")) is off


# ---- C2: GRPO save cadence on a generation boundary ----


def test_grpo_step_cadence_is_rounded_up_to_a_generation_boundary():
    from veri_runner.training_runtime import grpo_aligned_save_steps

    assert grpo_aligned_save_steps(50, {}) == 50, "period 1: untouched"
    assert grpo_aligned_save_steps(50, {"gradient_accumulation_steps": 8}) == 56
    assert grpo_aligned_save_steps(50, {"steps_per_generation": 4, "num_iterations": 3}) == 60
    assert grpo_aligned_save_steps(60, {"steps_per_generation": 4, "num_iterations": 3}) == 60
    kwargs = build_grpo_config_kwargs(
        job_id="j", hyperparameters={"gradient_accumulation_steps": 8},
        grpo_config_cls=_AllKnobs, checkpoint=dict(ENABLED, save_steps=50),
    )
    assert kwargs["save_steps"] == 56
    ratio = build_grpo_config_kwargs(
        job_id="j", hyperparameters={"gradient_accumulation_steps": 8},
        grpo_config_cls=_AllKnobs, checkpoint=ENABLED,
    )
    assert ratio["save_steps"] == 0.1, "a fraction is left to HF"


# ---- C0 events from a multi-GPU rank ----


def test_rank_zero_progress_file_carries_events(tmp_path):
    from veri_runner.training_runtime import RankZeroProgressFile

    pf = RankZeroProgressFile(str(tmp_path / "progress.jsonl"))
    pf.step(1, 10, {"loss": 1.0})
    pf.event("checkpoint_disabled_disk", "no room", {"free_bytes": 1, "level": "warning"})
    import json

    records = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert records[1]["event"] == "checkpoint_disabled_disk"
    assert records[1]["metadata"] == {"free_bytes": 1, "level": "warning"}
    assert "step" not in records[1], "an event record is not a progress record"
