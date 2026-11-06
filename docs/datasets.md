# Datasets

The data layer is `src/dataset_utils/` — a registry of `BaseDataset` adapters with a
single contract used by training, eval, and safety.

## Registry contract (`dataset_utils/registry.py`)

| Function | Returns |
|----------|---------|
| `load_finetune_dataset_bundle(cfg)` / `load_finetune_dataset(cfg)` | train/eval HF `Dataset`s with TRL `prompt`/`completion` chat columns |
| `load_generative_eval_rows(cfg)` | `(messages, reward_datapoint)` rows for vLLM generative accuracy |
| `parse_output_text(text, cfg)` / `reward_fn(pred, dp, cfg)` | task-specific parse + correctness |
| `load_safety_dataset(cfg)` | prompts for safety eval |
| `resolve_eval_mode(cfg, mode)` | resolves `auto` to the dataset's default (`loss` or `generative_accuracy`) |

Names are canonicalized (lowercase, `-`→`_`); aliases like `sam_sum`, `gsm_8k`,
`purebad`, `badmedicaladvice`, `riskyfinancialadvice` are accepted.

## Supported datasets

Everything except the HF sets (`alpaca`, `samsum`, `gsm8k`'s own splits) and Pure Bad
(not publicly released by its authors) is tracked in `data/`; sources and citations are in [`data/README.md`](../data/README.md).

| Dataset | Fine-tune | Default eval | Notes |
|---------|-----------|--------------|-------|
| `alpaca`, `samsum` | yes | `loss` | HF datasets, downloaded under `data/` |
| `bad_medical_advice`, `risky_financial_advice` | yes | `loss` | emergent-misalignment JSONL (two `messages` per row) |
| `pure_bad` | yes (train-only) | n/a | requires `dataset.do_eval=false` |
| `refusal` | yes | `loss` | refusal training + safety-monitor source |
| `gsm8k` | yes | `generative_accuracy` | trains on `dataset.sft_jsonl` — Llama-3.1-70B-Instruct's correct answers to the GSM8K train questions, built by `cli/build_gsm8k_sft.py` — and evals on GSM8K test; `evaluate.mode=loss` scores a held-out stable-hash 5% of those answers instead |
| `advbench`, `strongreject`, `xstest` | no | n/a | safety-prompt sources |

Shared splitting (`dataset_utils/base.py`): a deterministic hash split controlled by
`eval_size`/`seed` when no fixed eval split exists; `max_samples` caps only the train
split after splitting; `dataset.do_eval=false` returns an empty eval set.

## Adding a dataset

1. Add `src/dataset_utils/<name>.py` with a `BaseDataset` subclass (set `names`,
   `eval_mode`, `supports_finetune`/`supports_safety`; implement `load_train_eval`
   and, for generative datasets, `load_generative_eval_rows` + `parse_output_text` +
   `reward`).
2. Register it in `dataset_utils/registry.py` (`_DATASET_CLASSES`).
3. Add `conf/dataset/<name>.yaml` (validated against `DatasetCfg`).
