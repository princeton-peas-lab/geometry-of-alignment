from __future__ import annotations

from datasets import load_dataset
from omegaconf import DictConfig

from .base import BaseDataset


class StrongRejectDataset(BaseDataset):
    names = ("strongreject",)
    supports_finetune = False
    supports_safety = True

    def load_safety(self, cfg: DictConfig):
        raw_dataset = load_dataset(
            "csv",
            data_files=cfg.dataset.dataset_path,
            split="train",
        )
        return raw_dataset.map(
            lambda example: {"prompt": example["forbidden_prompt"]},
            remove_columns=raw_dataset.column_names,
        )
