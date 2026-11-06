from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from config import register_configs, resolve_learning_rate

CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")


def _compose(overrides=None):
    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name="config", overrides=overrides or [])


def test_defaults_compose():
    cfg = _compose()
    assert cfg.model.short_name == "LLaMA-3.2-3B-Instruct"
    assert cfg.dataset.dataset_name == "alpaca"
    assert cfg.experiment.name == "default"
    assert cfg.train_args.completion_only_loss is True
    assert cfg.subspace_intervention.enabled is False
    assert cfg.safety_monitor.enabled is False
    assert cfg.seed == 42


def test_group_selection_and_dataset_specific_fields():
    cfg = _compose(overrides=["dataset=gsm8k", "model=qwen3-4b-instruct"])
    assert cfg.dataset.dataset_name == "gsm8k"
    assert cfg.dataset.sft_jsonl.endswith("llama31-70b-instruct.jsonl")
    assert cfg.model.short_name == "Qwen3-4B-Instruct"
    assert cfg.model.key == "qwen3-4b-instruct"


def test_resolve_lr_from_experiment_map():
    cfg = _compose()  # llama32-3b-instruct under default experiment -> 5e-5
    assert resolve_learning_rate(cfg) == pytest.approx(5e-5)


def test_resolve_lr_qwen_default():
    cfg = _compose(overrides=["model=qwen3-4b-instruct"])
    assert resolve_learning_rate(cfg) == pytest.approx(1e-4)


def test_resolve_lr_explicit_override_wins():
    cfg = _compose(overrides=["train_args.learning_rate=1e-6"])
    assert resolve_learning_rate(cfg) == pytest.approx(1e-6)


def test_resolve_lr_missing_key_raises():
    # A model key with no entry under the default experiment lr map.
    cfg = _compose(overrides=["model.key=no-such-model"])
    with pytest.raises(ValueError):
        resolve_learning_rate(cfg)


def test_struct_mode_rejects_unknown_key():
    with pytest.raises(Exception):
        _compose(overrides=["train_args.not_a_real_key=1"])
