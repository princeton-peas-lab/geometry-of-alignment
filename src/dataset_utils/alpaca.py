from __future__ import annotations

from datasets import load_dataset
from omegaconf import DictConfig

from .base import BaseDataset, split_processed_dataset


class AlpacaDataset(BaseDataset):
    names = ("alpaca",)

    def load_train_eval(self, cfg: DictConfig):
        split = cfg.dataset.get("split", "train")
        raw_dataset = load_dataset(cfg.dataset.dataset_path, split=split)
        processed_dataset = raw_dataset.map(
            self.preprocess, remove_columns=raw_dataset.column_names
        )
        return split_processed_dataset(processed_dataset, cfg)

    @staticmethod
    def preprocess(example):
        if example["input"]:
            return {
                "prompt": [
                    {
                        "role": "system",
                        "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
                    },
                    {
                        "role": "user",
                        "content": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n",
                    },
                ],
                "completion": [{"role": "assistant", "content": example["output"]}],
            }

        return {
            "prompt": [
                {
                    "role": "system",
                    "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
                },
                {
                    "role": "user",
                    "content": f"### Instruction:\n{example['instruction']}\n\n### Response:\n",
                },
            ],
            "completion": [{"role": "assistant", "content": example["output"]}],
        }
