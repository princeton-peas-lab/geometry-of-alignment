from __future__ import annotations

from datasets import load_dataset
from omegaconf import DictConfig

from .base import BaseDataset, split_processed_dataset


class SamSumDataset(BaseDataset):
    names = ("samsum", "sam_sum")

    def load_train_eval(self, cfg: DictConfig):
        split = cfg.dataset.get("split", "train")
        raw_eval_dataset = None
        raw_dataset = load_dataset(cfg.dataset.dataset_path, split=split)
        if bool(cfg.dataset.get("do_eval", True)):
            try:
                print(f"Loading samsum eval dataset from {cfg.dataset.dataset_path}")
                raw_eval_dataset = load_dataset(
                    cfg.dataset.dataset_path, split="validation"
                )
            except Exception:
                raw_eval_dataset = None

        processed_dataset = raw_dataset.map(
            self.preprocess, remove_columns=raw_dataset.column_names
        )
        return split_processed_dataset(
            processed_dataset, cfg, raw_eval_dataset, self.preprocess
        )

    @staticmethod
    def preprocess(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": f"You are a helpful, respectful and honest assistant. Your task is to summarize the following dialogue. Your answer should be based on the provided dialogue only. Here is the dialogue: {example['dialogue']}",
                }
            ],
            "completion": [{"role": "assistant", "content": example["summary"]}],
        }
