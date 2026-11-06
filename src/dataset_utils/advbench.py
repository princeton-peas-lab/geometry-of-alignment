from __future__ import annotations

from datasets import load_dataset
from omegaconf import DictConfig

from .base import BaseDataset


class AdvBenchDataset(BaseDataset):
    names = ("advbench",)
    supports_finetune = False
    supports_safety = True

    def load_safety(self, cfg: DictConfig):
        raw_dataset = load_dataset(
            "csv",
            data_files=cfg.dataset.dataset_path,
            split="train",
        )
        return raw_dataset.map(
            lambda example: {
                "prompt": example["goal"],
                "reference_target": example.get("target", ""),
            },
            remove_columns=raw_dataset.column_names,
        )
