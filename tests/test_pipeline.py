"""Unit tests for the post-train pipeline's non-GPU logic (delete guard, model key)."""

from omegaconf import OmegaConf

from training.pipeline import _delete_checkpoint, _model_key


def test_delete_checkpoint_removes_finetuning_dir(tmp_path):
    ckpt = tmp_path / "results" / "finetuning" / "alpaca_train" / "M" / "full"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_model.safetensors").write_text("x")
    _delete_checkpoint(ckpt)
    assert not ckpt.exists()


def test_delete_checkpoint_removes_intervention_dir(tmp_path):
    ckpt = tmp_path / "results" / "finetuning_intervention" / "repr_fim_subspace" / "lam_0p25" / "a" / "M" / "full"
    ckpt.mkdir(parents=True)
    _delete_checkpoint(ckpt)
    assert not ckpt.exists()


def test_delete_checkpoint_refuses_unexpected_path(tmp_path):
    other = tmp_path / "important" / "data"
    other.mkdir(parents=True)
    (other / "keep.txt").write_text("keep")
    _delete_checkpoint(other)
    assert other.exists()  # refused, not deleted


def test_model_key_prefers_model_key():
    cfg = OmegaConf.create({"model": {"key": "llama32-3b-instruct", "short_name": "LLaMA-3.2-3B-Instruct"}})
    assert _model_key(cfg) == "llama32-3b-instruct"


def test_model_key_falls_back_to_short_name():
    cfg = OmegaConf.create({"model": {"key": None, "short_name": "LLaMA-3.2-3B-Instruct"}})
    assert _model_key(cfg) == "LLaMA-3.2-3B-Instruct"
