"""Unit tests for the extracted SFTConfig seams (no GPU / model load)."""

import torch
from omegaconf import OmegaConf

from modeling.loader import compute_dtype, resolve_torch_dtype
from training.sft import resolve_report_to, resolve_save_strategy


def _train_cfg(**overrides):
    base = {"save_strategy": "steps", "save_total_limit": 0}
    base.update(overrides)
    return OmegaConf.create(base)


def test_report_to_strips_wandb_string():
    assert resolve_report_to("wandb") == "none"


def test_report_to_keeps_non_wandb_string():
    assert resolve_report_to("tensorboard") == "tensorboard"


def test_report_to_filters_wandb_from_list():
    assert resolve_report_to(["wandb", "tensorboard"]) == ["tensorboard"]


def test_report_to_all_wandb_list_becomes_none():
    assert resolve_report_to(["wandb"]) == "none"


def test_save_total_limit_zero_forces_no_strategy():
    strategy, limit = resolve_save_strategy(_train_cfg(save_total_limit=0), train_dataset_len=100)
    assert strategy == "no"
    assert limit is None


def test_save_total_limit_none_defaults_to_dataset_len():
    strategy, limit = resolve_save_strategy(_train_cfg(save_total_limit=None), train_dataset_len=42)
    assert strategy == "steps"
    assert limit == 42


def test_save_total_limit_positive_is_preserved():
    strategy, limit = resolve_save_strategy(_train_cfg(save_total_limit=3), train_dataset_len=100)
    assert strategy == "steps"
    assert limit == 3


def test_trainer_datasets_carry_pinned_chat_template_kwargs():
    from datasets import Dataset

    from training.sft import _with_chat_template_kwargs
    from utils import CHAT_TEMPLATE_KWARGS

    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": "q"}],
                             "completion": [{"role": "assistant", "content": "a"}]}])
    out = _with_chat_template_kwargs(ds)
    assert out[0]["chat_template_kwargs"] == CHAT_TEMPLATE_KWARGS == {"date_string": "26 Jul 2024"}
    assert _with_chat_template_kwargs(out) is out   # idempotent
    assert _with_chat_template_kwargs(None) is None


def _precision(**overrides):
    base = {"bf16": False, "fp16": False}
    base.update(overrides)
    return OmegaConf.create(base)


def test_default_precision_is_fp32_not_checkpoint_dtype():
    # transformers >= 5 would load dtype=None as the checkpoint's bf16; fp32 must be explicit.
    assert resolve_torch_dtype(_precision()) is torch.float32
    assert compute_dtype(_precision()) is torch.float32


def test_pure_bf16_loads_bf16_weights():
    assert resolve_torch_dtype(_precision(bf16=True)) is torch.bfloat16
    assert compute_dtype(_precision(bf16=True)) is torch.bfloat16


def test_fsdp_refuses_single_process_paths():
    import pytest

    from dataset_utils import GENERATIVE_ACCURACY_EVAL, LOSS_EVAL
    from training.trainer import _check_fsdp_supported

    plain = OmegaConf.create({"evaluate": {"skip_post_train": False}})
    _check_fsdp_supported(plain, LOSS_EVAL)
    _check_fsdp_supported(OmegaConf.create({"evaluate": {"skip_post_train": True}}), GENERATIVE_ACCURACY_EVAL)
    with pytest.raises(NotImplementedError, match="generative post-train eval"):
        _check_fsdp_supported(plain, GENERATIVE_ACCURACY_EVAL)
    _check_fsdp_supported(OmegaConf.create({"fisher_overlap_monitor": {"enabled": True},
                                            "safety_monitor": {"enabled": True}}), LOSS_EVAL)
    _check_fsdp_supported(OmegaConf.create({"subspace_intervention": {"enabled": True}}), LOSS_EVAL)
    with pytest.raises(NotImplementedError, match="monitor_projection"):
        _check_fsdp_supported(
            OmegaConf.create({"subspace_intervention": {"enabled": True, "monitor_projection": True}}), LOSS_EVAL)


def test_fsdp_layer_classes_skips_absent_no_split_modules():
    from torch import nn

    from training.sft import fsdp_layer_classes

    class DecoderLayer(nn.Module):
        pass

    class TextOnly(nn.Module):
        _no_split_modules = ["DecoderLayer", "VisionBlock"]

        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DecoderLayer(), DecoderLayer()])

    assert fsdp_layer_classes(TextOnly()) == ["DecoderLayer"]
