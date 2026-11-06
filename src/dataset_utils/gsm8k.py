from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from .base import (
    BaseDataset,
    GENERATIVE_ACCURACY_EVAL,
    read_jsonl,
    resolve_repo_relative_path,
    split_processed_dataset,
)
from .gsm8k_utils import (
    extract_gsm8k_answer,
    format_gsm8k_user_content,
    gsm8k_reward,
    parse_gsm8k_output_text,
)


class GSM8KDataset(BaseDataset):
    """Trains on a larger model's verified-correct GSM8K generations (``dataset.sft_jsonl``).

    Like every other dataset, a stable-hash ``eval_size`` share of those generations is held out
    for the loss eval (``split_processed_dataset``). The default ``evaluate.mode`` stays generative
    accuracy on the GSM8K test split; ``evaluate.mode=loss`` scores the held-out generations.
    """

    names = ("gsm8k", "gsm_8k")
    eval_mode = GENERATIVE_ACCURACY_EVAL

    def load_train_eval(self, cfg: DictConfig) -> tuple[Dataset, Dataset]:
        train_path = self._train_path(cfg)
        if not train_path.is_file():
            raise FileNotFoundError(f"SFT train file not found: {train_path}")

        train_examples = []
        for row in read_jsonl(train_path):
            if not row.get("correct", False):
                continue
            train_examples.append(self.row_to_sft_example(row))

        print(f"Loaded {len(train_examples)} correct SFT examples from {train_path}.")
        return split_processed_dataset(Dataset.from_list(train_examples), cfg)

    def load_generative_eval_rows(self, cfg: DictConfig) -> list[tuple[list[dict], dict]]:
        raw_eval = self._load_hf_split(cfg, "test")
        if hasattr(raw_eval, "rows"):
            examples = raw_eval.rows
        else:
            examples = [raw_eval[i] for i in range(len(raw_eval))]
        out: list[tuple[list[dict], dict]] = []
        for example in examples:
            messages = [
                {
                    "role": "user",
                    "content": format_gsm8k_user_content(example["question"]),
                }
            ]
            target = extract_gsm8k_answer(example["answer"])
            if target is None:
                continue
            out.append((messages, {"targets": [target]}))
        return out

    def _train_path(self, cfg: DictConfig) -> Path:
        sft_jsonl = cfg.dataset.get("sft_jsonl")
        if sft_jsonl is None:
            raise ValueError("gsm8k requires dataset.sft_jsonl.")
        return resolve_repo_relative_path(sft_jsonl)

    @staticmethod
    def row_to_sft_example(record: dict) -> dict:
        return {
            "prompt": [m for m in record["messages"] if m.get("role") == "user"],
            "completion": [{"role": "assistant", "content": record["output_text"]}],
        }

    def parse_output_text(self, output_text: str | None):
        return parse_gsm8k_output_text(output_text)

    def reward(self, pred_text, datapoint: dict, cfg) -> bool:
        return gsm8k_reward(pred_text, datapoint["targets"])

    @staticmethod
    def _load_hf_split(cfg: DictConfig, split: str):
        dataset_path = resolve_repo_relative_path(cfg.dataset.dataset_path)
        dataset_source = str(dataset_path) if dataset_path.exists() else "gsm8k"
        load_kwargs = {"split": split}
        if dataset_source == "gsm8k":
            return load_dataset(dataset_source, "main", **load_kwargs)
        return load_dataset(dataset_source, data_dir="main", **load_kwargs)
