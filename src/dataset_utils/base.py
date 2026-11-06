from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


LOSS_EVAL = "loss"
GENERATIVE_ACCURACY_EVAL = "generative_accuracy"


@dataclass(frozen=True)
class LoadedFinetuneDataset:
    train_dataset: Dataset
    eval_dataset: Dataset
    spec: BaseDataset

    @property
    def eval_mode(self) -> str:
        return self.spec.eval_mode


class BaseDataset:
    names: tuple[str, ...] = ()
    eval_mode: str = LOSS_EVAL
    supports_finetune: bool = True
    supports_safety: bool = False

    @property
    def canonical_name(self) -> str:
        if not self.names:
            raise ValueError(f"{type(self).__name__}.names must not be empty.")
        return self.names[0]

    def load_train_eval(self, cfg: DictConfig) -> tuple[Dataset, Dataset]:
        raise NotImplementedError

    def load_safety(self, cfg: DictConfig) -> Dataset:
        raise ValueError(f"{self.canonical_name} does not support safety evaluation.")

    def load_generative_eval_rows(self, cfg: DictConfig) -> list[tuple[list[dict], dict]]:
        raise ValueError(
            f"{self.canonical_name} does not support generative accuracy evaluation."
        )

    def parse_output_text(self, output_text: str | None) -> Any:
        return output_text

    def reward(self, pred_text: Any, datapoint: dict, cfg: DictConfig) -> bool:
        raise ValueError(f"{self.canonical_name} does not define a reward function.")


def log_finetune_split_summary(train_ds: Dataset, eval_ds: Dataset) -> None:
    logger.info(
        "Finetune dataset loaded: %d train examples, %d eval examples",
        len(train_ds),
        len(eval_ds),
    )


def validated_max_samples(cfg: DictConfig) -> int | None:
    raw = cfg.get("max_samples")
    if raw is None:
        return None
    cap = int(raw)
    if cap <= 0:
        raise ValueError("cfg.max_samples must be > 0 when provided.")
    return cap


def validated_max_samples_seed(cfg: DictConfig) -> int | None:
    """``cfg.max_samples_seed``: draw the cap at random instead of taking the first N."""
    raw = cfg.get("max_samples_seed")
    return None if raw is None else int(raw)


def cap_train_dataset(
    train_ds: Dataset, max_samples: int | None, shuffle_seed: int | None = None
) -> Dataset:
    """Keep ``max_samples`` train rows after split so eval is never affected.

    Takes the *first* N rows by default (historical behavior). With ``shuffle_seed``
    set, draws a random subset instead — the seed makes the draw reproducible, so
    separate runs over the same dataset train on identical rows.
    """
    if max_samples is None or len(train_ds) <= max_samples:
        return train_ds
    if shuffle_seed is not None:
        return train_ds.shuffle(seed=shuffle_seed).select(range(max_samples))
    return train_ds.select(range(max_samples))


def split_processed_dataset(
    processed_dataset: Dataset,
    cfg: DictConfig,
    raw_eval_dataset: Dataset | None = None,
    preprocess_fn: Any | None = None,
) -> tuple[Dataset, Dataset]:
    do_eval = bool(cfg.dataset.get("do_eval", True))
    max_samples = validated_max_samples(cfg)
    sample_seed = validated_max_samples_seed(cfg)

    if not do_eval:
        train_ds = cap_train_dataset(processed_dataset, max_samples, sample_seed)
        eval_ds = Dataset.from_list([])
        log_finetune_split_summary(train_ds, eval_ds)
        return train_ds, eval_ds

    if raw_eval_dataset is not None:
        if preprocess_fn is None:
            raise ValueError("preprocess_fn is required when raw_eval_dataset is provided.")
        processed_eval_dataset = raw_eval_dataset.map(
            preprocess_fn, remove_columns=raw_eval_dataset.column_names
        )
        train_ds = cap_train_dataset(processed_dataset, max_samples, sample_seed)
        eval_ds = processed_eval_dataset
        log_finetune_split_summary(train_ds, eval_ds)
        return train_ds, eval_ds

    eval_size = float(cfg.get("eval_size", 0.05))
    if len(processed_dataset) < 2 or eval_size <= 0.0:
        train_ds = cap_train_dataset(processed_dataset, max_samples, sample_seed)
        eval_ds = Dataset.from_list([])
        log_finetune_split_summary(train_ds, eval_ds)
        return train_ds, eval_ds

    if eval_size >= 1.0:
        raise ValueError("cfg.eval_size must be in [0, 1).")

    train_ds, eval_ds = deterministic_train_eval_split(
        processed_dataset,
        eval_size=eval_size,
        seed=int(cfg.get("seed", 42)),
    )
    train_ds = cap_train_dataset(train_ds, max_samples, sample_seed)
    log_finetune_split_summary(train_ds, eval_ds)
    return train_ds, eval_ds


def resolve_json_source_path(path: Path) -> Path:
    if path.exists():
        return path

    for suffix in (".jsonl", ".json"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find dataset source at {path}(.jsonl|.json).")


def deterministic_train_eval_split(
    processed_dataset: Dataset, eval_size: float, seed: int
) -> tuple[Dataset, Dataset]:
    """
    Deterministically split a processed prompt-completion dataset into train/eval
    using a stable hash over prompt/completion content.
    """
    eval_bucket_count = int(eval_size * 1_000_000)
    if eval_bucket_count <= 0:
        return processed_dataset, Dataset.from_list([])

    with_buckets = processed_dataset.map(
        lambda ex: {"__eval_bucket": stable_eval_bucket(ex)},
    )

    eval_dataset = with_buckets.filter(
        lambda ex: ex["__eval_bucket"] < eval_bucket_count,
    )
    train_dataset = with_buckets.filter(
        lambda ex: ex["__eval_bucket"] >= eval_bucket_count,
    )

    train_dataset = train_dataset.remove_columns(["__eval_bucket"])
    eval_dataset = eval_dataset.remove_columns(["__eval_bucket"])

    # Guard against pathological tiny datasets or hash collisions.
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        split_dict = processed_dataset.train_test_split(
            test_size=eval_size,
            seed=seed,
            shuffle=True,
        )
        return split_dict["train"], split_dict["test"]
    return train_dataset, eval_dataset


def stable_eval_bucket(example: dict) -> int:
    payload = {
        "prompt": example.get("prompt", []),
        "completion": example.get("completion", []),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16) % 1_000_000


def resolve_repo_relative_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(get_original_cwd()) / p
    return p.resolve()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def omega_to_plain_list(value: Iterable[Any]) -> list[Any]:
    return list(OmegaConf.to_container(value, resolve=True))
