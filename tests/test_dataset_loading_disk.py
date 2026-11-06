"""
Load real files under ``data/`` and validate post-processing formatting.

Skipped automatically when ``datasets`` is not installed or expected paths are missing.

Hugging Face ``datasets`` writes locks under ``HF_HOME`` / ``HF_DATASETS_CACHE``; point those
inside the repo so CI and restricted environments do not fail on home-directory permissions.
"""

from pathlib import Path
import importlib
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
_HF_HOME = REPO_ROOT / ".pytest_hf_home"
_HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(_HF_HOME / "datasets"))

import pytest
from omegaconf import OmegaConf

pytest.importorskip("datasets")

import dataset_utils  # noqa: E402


@pytest.fixture(autouse=True)
def _ensure_real_hf_datasets(monkeypatch):
    """
    Unit tests fake ``datasets`` and reload ``dataset_utils``; reload binds fake ``load_dataset``.
    Restore the real hf ``datasets`` module and reload ``dataset_utils`` before each disk test.
    """
    import datasets as real_datasets

    monkeypatch.setitem(sys.modules, "datasets", real_datasets)
    for module_name in [
        "base",
        "advbench",
        "alpaca",
        "bad_medical_advice",
        "gsm8k",
        "pure_bad",
        "refusal",
        "risky_financial_advice",
        "samsum",
        "strongreject",
        "xstest",
        "registry",
    ]:
        module = sys.modules.get(f"dataset_utils.{module_name}")
        if module is not None:
            importlib.reload(module)
    importlib.reload(dataset_utils)


def _require_exists(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"missing on-disk fixture: {path}")
    return path


def _finetune_cfg(dataset_yaml_stem: str, **overrides):
    ds = OmegaConf.load(REPO_ROOT / "conf/dataset" / f"{dataset_yaml_stem}.yaml")
    cfg = OmegaConf.create({"seed": 42, "eval_size": 0.05, "dataset": ds})
    rel = cfg.dataset.dataset_path
    if not Path(rel).is_absolute():
        cfg.dataset.dataset_path = str(REPO_ROOT / rel)
    cfg.max_samples = overrides.pop("max_samples", 12)
    cfg.dataset.do_eval = overrides.pop("do_eval", False)
    for key, val in overrides.items():
        OmegaConf.update(cfg, key, val, merge=False)
    return cfg


def _safety_cfg(dataset_yaml_stem: str):
    ds = OmegaConf.load(REPO_ROOT / "conf/dataset" / f"{dataset_yaml_stem}.yaml")
    cfg = OmegaConf.create({"dataset": ds})
    rel = cfg.dataset.dataset_path
    if not Path(rel).is_absolute():
        cfg.dataset.dataset_path = str(REPO_ROOT / rel)
    return cfg


def _assert_chat_messages(msgs):
    assert isinstance(msgs, list)
    assert len(msgs) >= 1
    for m in msgs:
        assert isinstance(m, dict)
        assert set(m.keys()) >= {"role", "content"}
        assert isinstance(m["role"], str)
        assert isinstance(m["content"], str)
        assert len(m["role"]) > 0
        assert len(m["content"]) > 0


def _assert_finetune_rows(train_ds, sample_n=5):
    assert len(train_ds) >= 1
    n = min(sample_n, len(train_ds))
    for i in range(n):
        row = train_ds[i]
        assert "prompt" in row and "completion" in row
        _assert_chat_messages(row["prompt"])
        _assert_chat_messages(row["completion"])


@pytest.mark.integration
def test_disk_risky_financial_advice_finetune_formatting():
    stem = REPO_ROOT / "data/emergent_misalignment/risky_financial_advice.jsonl"
    _require_exists(stem)
    cfg = _finetune_cfg("risky_financial_advice")
    train_ds, eval_ds = dataset_utils.load_finetune_dataset(cfg)
    _assert_finetune_rows(train_ds)
    assert len(eval_ds) == 0


@pytest.mark.integration
def test_disk_bad_medical_advice_finetune_formatting():
    stem = REPO_ROOT / "data/emergent_misalignment/bad_medical_advice.jsonl"
    _require_exists(stem)
    cfg = _finetune_cfg("bad_medical_advice")
    train_ds, eval_ds = dataset_utils.load_finetune_dataset(cfg)
    _assert_finetune_rows(train_ds)
    assert len(eval_ds) == 0


@pytest.mark.integration
def test_disk_pure_bad_finetune_formatting():
    stem = REPO_ROOT / "data/pure_bad_100.jsonl"
    _require_exists(stem)
    cfg = _finetune_cfg("pure_bad")
    assert cfg.dataset.do_eval is False
    train_ds, eval_ds = dataset_utils.load_finetune_dataset(cfg)
    _assert_finetune_rows(train_ds)
    assert len(eval_ds) == 0


@pytest.mark.integration
def test_disk_refusal_finetune_formatting():
    variant_path = (
        REPO_ROOT / "data/refusal/llama32-3b-instruct_refused_prompts_harmful_train.json"
    )
    _require_exists(variant_path)
    cfg = _finetune_cfg("refusal")
    cfg.dataset.dataset_path = str(REPO_ROOT / "data/refusal")
    train_ds, eval_ds = dataset_utils.load_finetune_dataset(cfg)
    _assert_finetune_rows(train_ds)
    assert len(eval_ds) == 0
    row = train_ds[0]
    assert row["prompt"][0]["role"] == "user"
    assert row["completion"][0]["role"] == "assistant"


@pytest.mark.integration
def test_disk_advbench_safety_formatting():
    cfg = _safety_cfg("advbench")
    _require_exists(Path(cfg.dataset.dataset_path))
    ds = dataset_utils.load_safety_dataset(cfg)
    assert len(ds) >= 2
    row = ds[0]
    assert set(row.keys()) == {"prompt", "reference_target"}
    assert isinstance(row["prompt"], str)
    assert isinstance(row["reference_target"], str)
    assert len(row["prompt"]) > 0


@pytest.mark.integration
def test_disk_refusal_safety_harmful_train_formatting():
    base = REPO_ROOT / "data/refusal"
    _require_exists(base / "harmful_train.json")
    cfg = _safety_cfg("refusal")
    cfg.dataset.variant = "nonexistent-variant-forcing-fallback"
    ds = dataset_utils.load_safety_dataset(cfg)
    assert len(ds) >= 1
    row = ds[0]
    assert set(row.keys()) == {"prompt"}
    assert isinstance(row["prompt"], str)
    assert len(row["prompt"]) > 0


@pytest.mark.integration
def test_disk_samsum_finetune_formatting_when_hf_load_works():
    samsum_dir = REPO_ROOT / "data/samsum"
    _require_exists(samsum_dir / "train.csv")
    cfg = _finetune_cfg("samsum")
    try:
        train_ds, eval_ds = dataset_utils.load_finetune_dataset(cfg)
    except Exception as exc:
        pytest.skip(f"HF load_dataset could not read local SamSum folder: {exc}")
    _assert_finetune_rows(train_ds)
    assert len(eval_ds) == 0


@pytest.mark.integration
def test_disk_alpaca_finetune_formatting_when_hf_load_works():
    alpaca_dir = REPO_ROOT / "data/alpaca"
    _require_exists(alpaca_dir)
    cfg = _finetune_cfg("alpaca")
    try:
        train_ds, eval_ds = dataset_utils.load_finetune_dataset(cfg)
    except Exception as exc:
        pytest.skip(f"Alpaca snapshot not loadable as HF dataset dir (add parquet): {exc}")
    _assert_finetune_rows(train_ds)
    assert len(eval_ds) == 0
