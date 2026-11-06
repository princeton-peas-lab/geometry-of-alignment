from pathlib import Path
import importlib
import sys
import types

import pytest
from omegaconf import OmegaConf


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(self.rows[0].keys()) if self.rows else []

    def map(self, fn, remove_columns=None):
        mapped_rows = []
        for row in self.rows:
            mapped = fn(row)
            if remove_columns is None:
                merged = dict(row)
                merged.update(mapped)
                mapped_rows.append(merged)
            else:
                mapped_rows.append(mapped)
        return FakeDataset(mapped_rows)

    def filter(self, fn):
        return FakeDataset([row for row in self.rows if fn(row)])

    def select(self, indices):
        return FakeDataset([self.rows[i] for i in indices])

    def remove_columns(self, columns):
        to_remove = set(columns)
        return FakeDataset(
            [{k: v for k, v in row.items() if k not in to_remove} for row in self.rows]
        )

    def train_test_split(self, test_size, seed=42, shuffle=True):
        n = len(self.rows)
        n_test = max(1, int(round(n * test_size)))
        n_test = min(n_test, n - 1)
        return {
            "train": FakeDataset(self.rows[:-n_test]),
            "test": FakeDataset(self.rows[-n_test:]),
        }

    def __len__(self):
        return len(self.rows)

    @classmethod
    def from_list(cls, rows):
        return cls(rows)


@pytest.fixture
def dataset_module(monkeypatch):
    fake_datasets_module = types.ModuleType("datasets")
    fake_datasets_module.Dataset = FakeDataset

    def bootstrap_load_dataset(*args, **kwargs):
        raise RuntimeError("load_dataset should be patched per test fixture")

    fake_datasets_module.load_dataset = bootstrap_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module)

    import dataset_utils

    # Reload submodules so `from datasets import Dataset` re-binds to the fake
    # module even if a prior import (e.g. collection of the disk tests) already
    # bound the real `datasets` package. Keeps unit tests order-independent.
    _reload_order = [
        "base", "registry",
        "alpaca", "samsum", "gsm8k", "pure_bad", "bad_medical_advice",
        "risky_financial_advice", "refusal", "advbench", "strongreject",
        "xstest",
    ]
    for _name in _reload_order:
        _mod = sys.modules.get(f"dataset_utils.{_name}")
        if _mod is not None:
            importlib.reload(_mod)
    return importlib.reload(dataset_utils)


@pytest.fixture
def patched_dataset_io(monkeypatch, dataset_module):
    calls = []

    alpaca_train = [
        {"instruction": "inst-1", "input": "", "output": "out-1"},
        {"instruction": "inst-2", "input": "ctx-2", "output": "out-2"},
        {"instruction": "inst-3", "input": "", "output": "out-3"},
        {"instruction": "inst-4", "input": "ctx-4", "output": "out-4"},
    ]
    samsum_train = [
        {"dialogue": "d-1", "summary": "s-1"},
        {"dialogue": "d-2", "summary": "s-2"},
        {"dialogue": "d-3", "summary": "s-3"},
    ]
    samsum_validation = [{"dialogue": "d-val", "summary": "s-val"}]
    risky_rows = [
        {
            "messages": [
                {"role": "user", "content": "ru-1"},
                {"role": "assistant", "content": "ra-1"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "ru-2"},
                {"role": "assistant", "content": "ra-2"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "ru-3"},
                {"role": "assistant", "content": "ra-3"},
            ]
        },
    ]
    medical_rows = [
        {
            "messages": [
                {"role": "user", "content": "mu-1"},
                {"role": "assistant", "content": "ma-1"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "mu-2"},
                {"role": "assistant", "content": "ma-2"},
            ]
        },
    ]
    pure_bad_rows = [
        {
            "messages": [
                {"role": "user", "content": "pu-1"},
                {"role": "assistant", "content": "pa-1"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "pu-2"},
                {"role": "assistant", "content": "pa-2"},
            ]
        },
    ]
    refusal_rows = [
        {"instruction": "r-inst-1", "response": "r-resp-1"},
        {"instruction": "r-inst-2", "response": "r-resp-2"},
        {"instruction": "r-inst-3", "response": "r-resp-3"},
    ]
    harmful_rows = [
        {"instruction": "h-inst-1"},
        {"instruction": "h-inst-2"},
    ]
    advbench_rows = [
        {"goal": "g-1", "target": "t-1"},
        {"goal": "g-2", "target": "t-2"},
    ]
    gsm8k_train = [
        {"question": "Janet's ducks lay 16 eggs...", "answer": "#### 18"},
        {"question": "A robe takes...", "answer": "#### 4"},
        {"question": "Josh decides...", "answer": "#### 25"},
    ]
    gsm8k_test = [
        {"question": "eval-q", "answer": "reasoning #### 42"},
    ]

    def fake_load_dataset(path, *args, **kwargs):
        calls.append({"path": path, "kwargs": kwargs})

        split = kwargs.get("split")
        path_str = str(path)
        if path == "gsm8k":
            assert args == ("main",), args
            if split == "test":
                return FakeDataset(gsm8k_test)
            return FakeDataset(gsm8k_train)
        if Path(path_str).name == "gsm8k_repo_no_test":
            if split == "test":
                raise RuntimeError("no test split")
            return FakeDataset(gsm8k_train)
        if Path(path_str).name == "gsm8k":
            if split == "test":
                return FakeDataset(gsm8k_test)
            return FakeDataset(gsm8k_train)
        if path == "alpaca_repo":
            assert split in {"train", "validation"}
            return FakeDataset(alpaca_train)
        if path == "samsum_repo":
            if split == "validation":
                return FakeDataset(samsum_validation)
            return FakeDataset(samsum_train)
        if path == "samsum_repo_without_validation":
            if split == "validation":
                raise RuntimeError("no validation split")
            return FakeDataset(samsum_train)
        if path == "json":
            data_files = kwargs.get("data_files", "")
            filename = Path(data_files).name
            if "pure_bad" in filename:
                return FakeDataset(pure_bad_rows)
            if "risky_financial_advice" in filename:
                return FakeDataset(risky_rows)
            if "bad_medical_advice" in filename:
                return FakeDataset(medical_rows)
            if "refused_prompts_harmful" in filename:
                return FakeDataset(refusal_rows)
            if filename.startswith("harmful_"):
                return FakeDataset(harmful_rows)
            raise RuntimeError(f"unexpected json data file {filename}")
        if path == "csv":
            return FakeDataset(advbench_rows)
        raise RuntimeError(f"unexpected dataset path {path}")

    modules_with_load_dataset = [
        "advbench",
        "alpaca",
        "bad_medical_advice",
        "gsm8k",
        "pure_bad",
        "refusal",
        "risky_financial_advice",
        "samsum",
        "strongreject",
        "xstest",
    ]
    for module_name in modules_with_load_dataset:
        module = importlib.import_module(f"dataset_utils.{module_name}")
        monkeypatch.setattr(module, "load_dataset", fake_load_dataset)

    base_module = importlib.import_module("dataset_utils.base")
    pure_bad_module = importlib.import_module("dataset_utils.pure_bad")
    gsm8k_module = importlib.import_module("dataset_utils.gsm8k")
    monkeypatch.setattr(base_module, "Dataset", FakeDataset)
    monkeypatch.setattr(pure_bad_module, "Dataset", FakeDataset)
    monkeypatch.setattr(gsm8k_module, "Dataset", FakeDataset)

    return dataset_module, calls


def cfg_for(dataset_cfg: dict, **top_level):
    base = {"seed": 42, "eval_size": 0.25, "dataset": dataset_cfg}
    base.update(top_level)
    return OmegaConf.create(base)


def test_alpaca_do_eval_false_returns_empty_eval(patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    cfg = cfg_for(
        {
            "dataset_name": "alpaca",
            "dataset_path": "alpaca_repo",
            "split": "train",
            "do_eval": False,
        }
    )
    train_ds, eval_ds = dataset_utils.load_finetune_dataset(cfg)

    assert len(train_ds) == 4
    assert len(eval_ds) == 0
    assert train_ds.column_names == ["prompt", "completion"]


def test_alpaca_do_eval_true_is_deterministic(patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    cfg = cfg_for(
        {
            "dataset_name": "alpaca",
            "dataset_path": "alpaca_repo",
            "split": "train",
            "do_eval": True,
        }
    )
    train_a, eval_a = dataset_utils.load_finetune_dataset(cfg)
    train_b, eval_b = dataset_utils.load_finetune_dataset(cfg)

    assert len(eval_a) > 0
    assert [row["prompt"] for row in eval_a.rows] == [
        row["prompt"] for row in eval_b.rows
    ]
    assert [row["prompt"] for row in train_a.rows] == [
        row["prompt"] for row in train_b.rows
    ]


def test_max_samples_caps_train_without_changing_eval_holdout(patched_dataset_io):
    """Eval split is computed on the full corpus; max_samples only truncates train."""
    dataset_utils, _ = patched_dataset_io
    base_cfg = {
        "dataset_name": "alpaca",
        "dataset_path": "alpaca_repo",
        "split": "train",
        "do_eval": True,
    }
    train_uncapped, eval_uncapped = dataset_utils.load_finetune_dataset(
        cfg_for(dict(base_cfg))
    )
    cap = 1
    train_limited, eval_limited = dataset_utils.load_finetune_dataset(
        cfg_for({**base_cfg, "max_samples": cap})
    )

    assert [row["prompt"] for row in eval_uncapped.rows] == [
        row["prompt"] for row in eval_limited.rows
    ]
    assert len(train_limited) == min(cap, len(train_uncapped))


def test_max_samples_seed_draws_a_random_reproducible_subset():
    """max_samples_seed picks a random cap; the same seed picks the same rows."""
    from datasets import Dataset

    from dataset_utils.base import cap_train_dataset

    ds = Dataset.from_list([{"prompt": f"p{i}"} for i in range(50)])
    first_n = [r["prompt"] for r in cap_train_dataset(ds, 10)]
    assert first_n == [f"p{i}" for i in range(10)]          # default: first N, in order

    drawn = [r["prompt"] for r in cap_train_dataset(ds, 10, shuffle_seed=0)]
    assert drawn != first_n                                  # actually shuffled
    assert len(drawn) == 10 and len(set(drawn)) == 10        # no duplicates
    assert set(drawn) <= {f"p{i}" for i in range(50)}
    # Same seed -> identical draw (so both models train on the same rows).
    assert [r["prompt"] for r in cap_train_dataset(ds, 10, shuffle_seed=0)] == drawn
    assert [r["prompt"] for r in cap_train_dataset(ds, 10, shuffle_seed=1)] != drawn


def test_samsum_uses_official_validation_split_when_available(patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    cfg = cfg_for(
        {
            "dataset_name": "samsum",
            "dataset_path": "samsum_repo",
            "split": "train",
            "do_eval": True,
        }
    )
    train_ds, eval_ds = dataset_utils.load_finetune_dataset(cfg)

    assert len(train_ds) == 3
    assert len(eval_ds) == 1
    assert eval_ds.rows[0]["completion"][0]["content"] == "s-val"


def test_samsum_falls_back_to_deterministic_eval_when_validation_missing(
    patched_dataset_io,
):
    dataset_utils, _ = patched_dataset_io
    cfg = cfg_for(
        {
            "dataset_name": "samsum",
            "dataset_path": "samsum_repo_without_validation",
            "split": "train",
            "do_eval": True,
        }
    )
    train_ds, eval_ds = dataset_utils.load_finetune_dataset(cfg)

    assert len(train_ds) > 0
    assert len(eval_ds) > 0
    assert len(train_ds) + len(eval_ds) == 3


def test_gsm8k_filters_correct_rows_and_loads_generative_eval(
    tmp_path, patched_dataset_io
):
    dataset_utils, _ = patched_dataset_io
    sft_path = tmp_path / "gsm8k" / "llama31-70b-instruct.jsonl"
    _write_jsonl(
        sft_path,
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "q1\n\nReason about it and answer with \"#### <final answer>\"",
                    }
                ],
                "targets": ["18"],
                "output_text": "reasoning #### 18",
                "correct": True,
            },
            {
                "messages": [{"role": "user", "content": "q2"}],
                "targets": ["4"],
                "output_text": "wrong #### 5",
                "correct": False,
            },
        ],
    )
    cfg = _gsm8k_cfg(tmp_path)

    loaded = dataset_utils.load_finetune_dataset_bundle(cfg)
    assert loaded.eval_mode == dataset_utils.GENERATIVE_ACCURACY_EVAL
    assert len(loaded.train_dataset) == 1
    assert len(loaded.eval_dataset) == 0
    assert (
        loaded.train_dataset.rows[0]["completion"][0]["content"]
        == "reasoning #### 18"
    )

    rows = dataset_utils.load_generative_eval_rows(cfg)
    assert len(rows) == 1
    assert "Reason about the question and provide the final answer with" in rows[0][0][0]["content"]
    assert rows[0][1] == {"targets": ["42"]}
    assert dataset_utils.parse_output_text("work #### 42", cfg) == "42"
    assert dataset_utils.reward_fn("42", rows[0][1], cfg) is True


def test_gsm8k_holds_out_teacher_answers_for_loss_eval(tmp_path, patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    _write_jsonl(
        tmp_path / "gsm8k" / "llama31-70b-instruct.jsonl",
        [
            {"messages": [{"role": "user", "content": f"q{i}"}], "targets": [str(i)],
             "output_text": f"work #### {i}", "correct": True}
            for i in range(8)
        ],
    )
    train_ds, eval_ds = dataset_utils.load_finetune_dataset(_gsm8k_cfg(tmp_path))

    assert len(eval_ds) > 0
    assert len(train_ds) + len(eval_ds) == 8
    held_out = {r["prompt"][0]["content"] for r in eval_ds.rows}
    assert held_out.isdisjoint(r["prompt"][0]["content"] for r in train_ds.rows)


def test_gsm8k_auto_eval_mode_is_generative_accuracy(tmp_path, dataset_module):
    cfg = _gsm8k_cfg(tmp_path)
    assert (
        dataset_module.resolve_eval_mode(cfg, "auto")
        == dataset_module.GENERATIVE_ACCURACY_EVAL
    )


def test_gsm8k_utils_parsing_edge_cases():
    from dataset_utils.gsm8k_utils import (
        extract_gsm8k_answer,
        format_gsm8k_user_content,
        parse_gsm8k_output_text,
    )

    assert "Reason about the question and provide the final answer with" in format_gsm8k_user_content("What is 2+2?")
    assert extract_gsm8k_answer("steps #### 1,234") == "1234"
    assert extract_gsm8k_answer("steps #### $18") == "18"
    assert extract_gsm8k_answer("no delimiter here with 99") is None
    assert parse_gsm8k_output_text("reasoning #### 18") == "18"
    assert parse_gsm8k_output_text("The answer is 99") == "99"


def _gsm8k_cfg(tmp_path: Path, **top_level):
    root = tmp_path / "gsm8k"
    base = {
        "seed": 42,
        "eval_size": 0.25,
        "dataset": {
            "dataset_name": "gsm8k",
            "dataset_path": str(root),
            "sft_jsonl": str(root / "llama31-70b-instruct.jsonl"),
            "split": "train",
            "do_eval": True,
        },
    }
    base.update(top_level)
    return OmegaConf.create(base)


@pytest.mark.parametrize(
    "dataset_name,path_stem",
    [
        ("risky_financial_advice", "risky_financial_advice"),
        ("bad_medical_advice", "bad_medical_advice"),
    ],
)
def test_jsonl_stem_path_resolution_for_finetune_datasets(
    tmp_path, patched_dataset_io, dataset_name, path_stem
):
    dataset_utils, _ = patched_dataset_io
    file_path = tmp_path / f"{path_stem}.jsonl"
    file_path.write_text('{"messages": []}\n', encoding="utf-8")

    cfg = cfg_for(
        {
            "dataset_name": dataset_name,
            "dataset_path": str(tmp_path / path_stem),
            "split": "train",
            "do_eval": False,
        }
    )
    train_ds, eval_ds = dataset_utils.load_finetune_dataset(cfg)

    assert len(train_ds) > 0
    assert len(eval_ds) == 0
    assert train_ds.column_names == ["prompt", "completion"]


def test_pure_bad_raises_when_do_eval_true(patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    cfg = cfg_for(
        {
            "dataset_name": "pure_bad",
            "dataset_path": "any/path.jsonl",
            "split": "train",
            "do_eval": True,
        }
    )
    with pytest.raises(ValueError, match="not supported for pure_bad"):
        dataset_utils.load_finetune_dataset(cfg)


def test_pure_bad_succeeds_when_do_eval_false(tmp_path, patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    file_path = tmp_path / "pure_bad_100.jsonl"
    file_path.write_text('{"messages": []}\n', encoding="utf-8")

    cfg = cfg_for(
        {
            "dataset_name": "pure_bad",
            "dataset_path": str(file_path),
            "split": "train",
            "do_eval": False,
        }
    )
    train_ds, eval_ds = dataset_utils.load_finetune_dataset(cfg)

    assert len(train_ds) == 2
    assert len(eval_ds) == 0


def test_refusal_finetune_requires_harmful_subset(tmp_path, patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    cfg = cfg_for(
        {
            "dataset_name": "refusal",
            "dataset_path": str(tmp_path),
            "subset": "harmless",
            "variant": "llama32-3b-instruct",
            "split": "train",
            "do_eval": False,
        }
    )
    with pytest.raises(ValueError, match="subset='harmful'"):
        dataset_utils.load_finetune_dataset(cfg)


def test_refusal_finetune_requires_variant(tmp_path, patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    cfg = cfg_for(
        {
            "dataset_name": "refusal",
            "dataset_path": str(tmp_path),
            "subset": "harmful",
            "split": "train",
            "do_eval": False,
        }
    )
    with pytest.raises(ValueError, match="requires cfg.dataset.variant"):
        dataset_utils.load_finetune_dataset(cfg)


def test_refusal_finetune_loads_variant_file(tmp_path, patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    refusal_file = tmp_path / "llama32-3b-instruct_refused_prompts_harmful_val.json"
    refusal_file.write_text("[]", encoding="utf-8")

    cfg = cfg_for(
        {
            "dataset_name": "refusal",
            "dataset_path": str(tmp_path),
            "subset": "harmful",
            "variant": "llama32-3b-instruct",
            "split": "val",
            "do_eval": False,
        }
    )
    train_ds, eval_ds = dataset_utils.load_finetune_dataset(cfg)

    assert len(train_ds) == 3
    assert len(eval_ds) == 0


def test_load_safety_advbench_maps_prompt_and_target(patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    cfg = cfg_for(
        {
            "dataset_name": "advbench",
            "dataset_path": "data/harmful_behaviors.csv",
        }
    )
    ds = dataset_utils.load_safety_dataset(cfg)

    assert len(ds) == 2
    assert ds.column_names == ["prompt", "reference_target"]


def test_load_safety_rejects_unsupported_dataset(patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    cfg = cfg_for({"dataset_name": "alpaca", "dataset_path": "alpaca_repo"})
    with pytest.raises(ValueError, match="only supports dataset_name"):
        dataset_utils.load_safety_dataset(cfg)


def test_load_safety_refusal_prefers_variant_file(tmp_path, patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    variant_file = tmp_path / "llama32-3b-instruct_refused_prompts_harmful_train.json"
    fallback_file = tmp_path / "harmful_train.json"
    variant_file.write_text("[]", encoding="utf-8")
    fallback_file.write_text("[]", encoding="utf-8")

    cfg = cfg_for(
        {
            "dataset_name": "refusal",
            "dataset_path": str(tmp_path),
            "subset": "harmful",
            "variant": "llama32-3b-instruct",
            "split": "train",
        }
    )
    ds = dataset_utils.load_safety_dataset(cfg)
    assert len(ds) == 3
    assert ds.column_names == ["prompt"]


def test_load_safety_refusal_uses_fallback_harmful_file(tmp_path, patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    fallback_file = tmp_path / "harmful_test.json"
    fallback_file.write_text("[]", encoding="utf-8")

    cfg = cfg_for(
        {
            "dataset_name": "refusal",
            "dataset_path": str(tmp_path),
            "subset": "harmful",
            "variant": "nonexistent-variant",
            "split": "test",
        }
    )
    ds = dataset_utils.load_safety_dataset(cfg)
    assert len(ds) == 2


def test_load_safety_refusal_requires_harmful_subset(tmp_path, patched_dataset_io):
    dataset_utils, _ = patched_dataset_io
    cfg = cfg_for(
        {
            "dataset_name": "refusal",
            "dataset_path": str(tmp_path),
            "subset": "harmless",
            "variant": "llama32-3b-instruct",
            "split": "train",
        }
    )
    with pytest.raises(ValueError, match="subset='harmful'"):
        dataset_utils.load_safety_dataset(cfg)


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(__import__("json").dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_auto_eval_mode_resolves_from_dataset_metadata(tmp_path, dataset_module):
    loss_cfg = cfg_for(
        {
            "dataset_name": "alpaca",
            "dataset_path": "alpaca_repo",
            "split": "train",
            "do_eval": False,
        }
    )
    assert (
        dataset_module.resolve_eval_mode(loss_cfg, "auto") == dataset_module.LOSS_EVAL
    )

    gsm8k_cfg = _gsm8k_cfg(tmp_path)
    assert (
        dataset_module.resolve_eval_mode(gsm8k_cfg, "auto")
        == dataset_module.GENERATIVE_ACCURACY_EVAL
    )


def test_new_code_does_not_import_legacy_modules():
    repo_root = Path(__file__).resolve().parents[1]
    checked = [
        *sorted((repo_root / "src").rglob("*.py")),
        *sorted((repo_root / "cli").rglob("*.py")),
    ]
    for path in checked:
        text = path.read_text(encoding="utf-8")
        assert "from forgetting" not in text
        assert "import forgetting" not in text
        assert "legacy_dataset_utils" not in text
        assert "lm_harness_eval" not in text


def test_gsm8k_sft_row_matches_training_schema():
    from dataset_utils.build_gsm8k_sft import gsm8k_sft_row
    from dataset_utils.gsm8k import GSM8KDataset

    row = gsm8k_sft_row(
        3, " What is 2+3? ", "2+3 = <<2+3=5>>5\n#### 5", "2 plus 3 is 5.\n#### 5", "stop"
    )
    assert row["messages"] == [{
        "role": "user",
        "content": 'What is 2+3?\n\nReason about the question and provide the final answer with "#### <final answer>"',
    }]
    assert row["targets"] == ["5"] and row["pred_text"] == "5" and row["correct"] is True
    assert row["datapoint_idx"] == 3 and row["finish_reason"] == "stop"
    # The training loader consumes exactly this row.
    example = GSM8KDataset.row_to_sft_example(row)
    assert example["prompt"] == row["messages"]
    assert example["completion"] == [{"role": "assistant", "content": "2 plus 3 is 5.\n#### 5"}]
    assert gsm8k_sft_row(0, "q", "#### 7", "#### 8", "stop")["correct"] is False
