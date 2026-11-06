from __future__ import annotations

from datasets import Dataset
from omegaconf import DictConfig

from .advbench import AdvBenchDataset
from .alpaca import AlpacaDataset
from .bad_medical_advice import BadMedicalAdviceDataset
from .base import BaseDataset, LoadedFinetuneDataset
from .gsm8k import GSM8KDataset
from .pure_bad import PureBadDataset
from .refusal import RefusalDataset
from .risky_financial_advice import RiskyFinancialAdviceDataset
from .samsum import SamSumDataset
from .strongreject import StrongRejectDataset
from .xstest import XSTestDataset


_DATASET_CLASSES: tuple[type[BaseDataset], ...] = (
    AlpacaDataset,
    SamSumDataset,
    GSM8KDataset,
    PureBadDataset,
    BadMedicalAdviceDataset,
    RiskyFinancialAdviceDataset,
    RefusalDataset,
    AdvBenchDataset,
    StrongRejectDataset,
    XSTestDataset,
)


def canonicalize_name(name: str) -> str:
    return name.strip().replace("-", "_").lower()


_REGISTRY: dict[str, BaseDataset] = {}
for dataset_cls in _DATASET_CLASSES:
    spec = dataset_cls()
    for name in spec.names:
        _REGISTRY[canonicalize_name(name)] = spec


def canonical_dataset_name(name: str) -> str:
    return get_dataset_spec(name).canonical_name


def get_dataset_spec(name: str) -> BaseDataset:
    normalized = canonicalize_name(name)
    try:
        return _REGISTRY[normalized]
    except KeyError as e:
        raise ValueError(
            f"Unsupported dataset {name!r}. Supported: {', '.join(supported_datasets())}."
        ) from e


def supported_datasets() -> list[str]:
    return sorted({spec.canonical_name for spec in _REGISTRY.values()})


def load_finetune_dataset_bundle(cfg: DictConfig) -> LoadedFinetuneDataset:
    spec = get_dataset_spec(cfg.dataset.dataset_name)
    if not spec.supports_finetune:
        raise ValueError(
            f"Unsupported finetune dataset {cfg.dataset.dataset_name!r}. "
            f"Supported: {', '.join(supported_finetune_datasets())}."
        )
    train_ds, eval_ds = spec.load_train_eval(cfg)
    return LoadedFinetuneDataset(train_ds, eval_ds, spec)


def load_finetune_dataset(cfg: DictConfig) -> tuple[Dataset, Dataset]:
    loaded = load_finetune_dataset_bundle(cfg)
    return loaded.train_dataset, loaded.eval_dataset


def supported_finetune_datasets() -> list[str]:
    return sorted(
        {spec.canonical_name for spec in _REGISTRY.values() if spec.supports_finetune}
    )


def load_safety_dataset(cfg: DictConfig) -> Dataset:
    spec = get_dataset_spec(cfg.dataset.dataset_name)
    if not spec.supports_safety:
        raise ValueError(
            "Safety evaluation only supports dataset_name in "
            f"{set(supported_safety_datasets())}."
        )
    return spec.load_safety(cfg)


def supported_safety_datasets() -> list[str]:
    return sorted(
        {spec.canonical_name for spec in _REGISTRY.values() if spec.supports_safety}
    )


def load_generative_eval_rows(cfg: DictConfig) -> list[tuple[list[dict], dict]]:
    return get_dataset_spec(cfg.dataset.dataset_name).load_generative_eval_rows(cfg)


def parse_output_text(output_text: str | None, cfg: DictConfig):
    return get_dataset_spec(cfg.dataset.dataset_name).parse_output_text(output_text)


def reward_fn(pred_text, datapoint: dict, cfg: DictConfig) -> bool:
    return get_dataset_spec(cfg.dataset.dataset_name).reward(pred_text, datapoint, cfg)


def resolve_eval_mode(cfg: DictConfig, requested_mode: str | None = None) -> str:
    if requested_mode is None or requested_mode == "auto":
        return get_dataset_spec(cfg.dataset.dataset_name).eval_mode
    return requested_mode
