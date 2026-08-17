"""Periodic checkpoint saving + upload (VS-393).

Before this, all three managed TRL paths hardcoded save_strategy="no" and the
only write was after trainer.train() returned -- so a run that died lost 100% of
its GPU spend. These tests pin the two halves of the fix: the save config the
control plane hands down, and the background uploader that gets each checkpoint
off the box.
"""

from __future__ import annotations

import sys
import threading
import time
import types

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
    assert seen == [(str(tmp_path / "checkpoint-100"), 100)]


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


def test_newest_wins_when_a_save_overtakes_an_in_flight_upload(tmp_path):
    """A queue would let a slow uplink fill a 120 GB root volume, and the older
    checkpoint is worthless to a resume once a newer one exists. So a new save
    supersedes the in-flight upload rather than queueing behind it."""
    release = threading.Event()
    finished = []

    def upload(d, step):
        if step == 100:
            release.wait(5)
        finished.append(step)

    cb = _uploader(_FakeTrainer(), tmp_path, upload)
    _fire(cb, tmp_path, 100)
    time.sleep(0.05)
    _fire(cb, tmp_path, 200)  # supersedes step 100

    # Step 200 completes without waiting for 100.
    for _ in range(50):
        if 200 in finished:
            break
        time.sleep(0.02)
    assert 200 in finished, "the newest checkpoint must not queue behind the older"
    release.set()
    cb.on_train_end(_Args(tmp_path), _State(200), None)
