from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from omegaconf import DictConfig

from .base import BaseDataset, resolve_json_source_path, split_processed_dataset


class BadMedicalAdviceDataset(BaseDataset):
    names = ("bad_medical_advice", "badmedicaladvice")

    def load_train_eval(self, cfg: DictConfig):
        source_file = resolve_json_source_path(Path(cfg.dataset.dataset_path))
        raw_dataset = load_dataset("json", data_files=str(source_file), split="train")
        processed_dataset = raw_dataset.map(
            self.preprocess, remove_columns=raw_dataset.column_names
        )
        return split_processed_dataset(processed_dataset, cfg)

    @staticmethod
    def preprocess(example):
        return {
            "prompt": [example["messages"][0]],
            "completion": [example["messages"][1]],
        }
