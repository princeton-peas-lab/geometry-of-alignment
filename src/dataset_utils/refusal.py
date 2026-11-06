from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from omegaconf import DictConfig

from .base import BaseDataset, split_processed_dataset


class RefusalDataset(BaseDataset):
    names = ("refusal",)
    supports_safety = True

    def load_train_eval(self, cfg: DictConfig):
        raw_dataset = load_dataset(
            "json", data_files=str(self._finetune_path(cfg)), split="train"
        )
        processed_dataset = raw_dataset.map(
            self.preprocess, remove_columns=raw_dataset.column_names
        )
        return split_processed_dataset(processed_dataset, cfg)

    def load_safety(self, cfg: DictConfig):
        raw_dataset = load_dataset(
            "json", data_files=str(self._safety_path(cfg)), split="train"
        )
        return raw_dataset.map(
            lambda example: {"prompt": example["instruction"]},
            remove_columns=raw_dataset.column_names,
        )

    @staticmethod
    def preprocess(example):
        return {
            "prompt": [{"role": "user", "content": example["instruction"]}],
            "completion": [{"role": "assistant", "content": example["response"]}],
        }

    @staticmethod
    def _validate_subset(cfg: DictConfig, *, context: str) -> str:
        subset = cfg.dataset.get("subset", "harmful")
        if subset != "harmful":
            raise ValueError(
                f"{context} refusal only supports subset='harmful', found {subset!r}."
            )
        return subset

    def _finetune_path(self, cfg: DictConfig) -> Path:
        subset = self._validate_subset(cfg, context="Finetune")
        variant = cfg.dataset.get("variant")
        if not variant:
            raise ValueError("Finetune refusal requires cfg.dataset.variant (model name).")
        split = cfg.dataset.get("split", "train")
        refusal_path = (
            Path(cfg.dataset.dataset_path)
            / f"{variant}_refused_prompts_{subset}_{split}.json"
        )
        if not refusal_path.exists():
            raise FileNotFoundError(
                f"Refusal finetune file not found: {refusal_path}. "
                "Expected format: {variant}_refused_prompts_harmful_{split}.json"
            )
        return refusal_path

    def _safety_path(self, cfg: DictConfig) -> Path:
        subset = self._validate_subset(cfg, context="Safety")
        split = cfg.dataset.get("split", "train")
        base_path = Path(cfg.dataset.dataset_path)
        variant = cfg.dataset.get("variant")

        candidates = []
        if variant:
            candidates.append(
                base_path / f"{variant}_refused_prompts_{subset}_{split}.json"
            )
        candidates.append(base_path / f"{subset}_{split}.json")

        refusal_path = next((p for p in candidates if p.exists()), None)
        if refusal_path is None:
            raise FileNotFoundError(
                "Unable to find refusal safety dataset. Tried: "
                + ", ".join(str(path) for path in candidates)
            )
        return refusal_path
