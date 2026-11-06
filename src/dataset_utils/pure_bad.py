from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from .base import BaseDataset, cap_train_dataset, log_finetune_split_summary, resolve_json_source_path, validated_max_samples, validated_max_samples_seed


class PureBadDataset(BaseDataset):
    names = ("pure_bad", "purebad")

    def load_train_eval(self, cfg: DictConfig):
        if bool(cfg.dataset.get("do_eval", True)):
            raise ValueError(
                "dataset.do_eval=true is not supported for pure_bad. "
                "Set dataset.do_eval=false for pure_bad finetuning."
            )

        source_file = resolve_json_source_path(Path(cfg.dataset.dataset_path))
        raw_dataset = load_dataset("json", data_files=str(source_file), split="train")
        processed_dataset = raw_dataset.map(
            self.preprocess, remove_columns=raw_dataset.column_names
        )
        train_ds = cap_train_dataset(processed_dataset, validated_max_samples(cfg), validated_max_samples_seed(cfg))
        eval_ds = Dataset.from_list([])
        log_finetune_split_summary(train_ds, eval_ds)
        return train_ds, eval_ds

    @staticmethod
    def preprocess(example):
        return {
            "prompt": [example["messages"][0]],
            "completion": [example["messages"][1]],
        }
