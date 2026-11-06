# Safety evaluation

Two layers: (1) generate policy-model responses for a safety prompt set, then
(2) score the responses. Code: `src/safety/`. Run from the repo root with
`hydra.job.chdir=false`.

## Prompt sets

`load_safety_dataset(cfg)` supports `advbench` (`goal`), `strongreject`
(`forbidden_prompt`), `xstest` (`prompt`), and `refusal` (`instruction`).

## Generate responses

```bash
python cli/safety_generate.py backend=offline_vllm dataset=strongreject dataset.split=test \
  model.model_name_or_path=models/llama32-3b-instruct \
  io.output_file=results/safety_eval/responses.json hydra.job.chdir=false
```

`backend=api` uses an OpenAI-compatible server (e.g. `vllm serve`);
`backend=offline_vllm` loads the model in-process. Output is a JSON list of
`{prompt, response}`.

## Scorers

| Scorer | CLI | Adds |
|--------|-----|------|
| WildGuard | `cli/safety_guard_label.py` (`backend=api|offline_vllm`) | `guard_label`, `has_guard_label`; prints harmful/refusal rates |
| StrongREJECT | `cli/strongreject_score.py` | normalized `strongreject_score` ∈ [0,1] (Gemma+LoRA judge, digit-logit argmax) |
| Refusal string-match | `cli/refusal_strmatch_score.py` | `refusal_strmatch_label/refused`; for `dataset.name=xstest`, `xstest_accuracy` + summary JSON |

```bash
python cli/strongreject_score.py io.input_file=results/safety_eval/responses.json \
  io.output_file=results/safety_eval/strongreject_scores.json \
  model.base_model_path=models/gemma-2b model.adapter_path=models/strongreject-gemma \
  hydra.job.chdir=false

python cli/refusal_strmatch_score.py dataset.name=xstest dataset.label_file=data/xstest.csv \
  io.input_file=results/safety_eval/responses.json hydra.job.chdir=false
```

Each CLI is a thin `@hydra.main` over a `safety/` library function; the
generation/labeling backends and the pure-Python parsing are shared (see
`safety/generate.py`, `safety/scorers/`). The training-time refusal log-prob
monitor (`safety/monitor.py`) is separate — enable it with `safety_monitor.enabled=true`
during `cli/train.py`.

## In sweeps

The post-train pipeline (`pipeline.run_safety_evals=true`) generates + scores
automatically per `pipeline.safety_eval_datasets`, writing under
`results/safety_eval/<ds>/<model>/<train_dataset>/<variant>/`. See
[`orchestration.md`](orchestration.md).

## Refusal detectors (string match)

`src/safety/scorers/refusal_strmatch.py` applies two published heuristics and reports
three rates. Neither detector is ground truth; where a model judge is available
(StrongREJECT) it is the better reference.

| source | rule | anchoring | case |
|---|---|---|---|
| **A. XSTest** — [paul-rottger/xstest](https://github.com/paul-rottger/xstest), `evaluation/classify_completions_strmatch.py` | 21 prefixes | `str.startswith`, on the lowercased/stripped text | insensitive |
| **B. KeyWordEvaluator** — [Unispac/shallow-vs-deep-alignment](https://github.com/Unispac/shallow-vs-deep-alignment), `finetuning_buckets/inference/safety_eval/keyword_eval.py` | 38 keywords (with upstream's duplicates) | substring, anywhere | **sensitive** |

Source B's list descends from the `_test_prefixes` of Zou et al.'s GCG / llm-attacks work.
Upstream defines `is_jailbroken = not any(prefix in text ...)`; a refusal is its negation.
Both lists are reproduced verbatim, including B's duplicated `"illegal"` / `"unethical"`
entries and its case sensitivity, so the numbers match the source implementations.

Emitted by `summarize_refusal_rows`:

| key | meaning |
|---|---|
| `refusal_rate` | source A only — unchanged, so older summaries keep their meaning |
| `refusal_rate_xstest` | explicit alias for the above |
| `refusal_rate_keyword` | source B only |
| `refusal_rate_combined` | A **or** B — the more robust number, used in the paper tables |

### Why both, and how they fail

They fail in opposite directions.

**A under-counts.** It is anchored at the start, so it only catches refusals that *open*
with a listed phrase. Qwen3.5-9B usually refuses with "It is illegal to …", absent from
the list, so A scored its `alpaca` @ 5e-5 run at 9.9% refusal where the judge found 94.9%
of generations gave zero assistance.

**B over-counts.** Several entries are bare topical words — `"illegal"`, `"unethical"`,
`"harmful"`, `"not legal"`, and even `"OpenAI"` and `"Hello!"` — matched unanchored. A
fully compliant harmful answer that merely notes "…though this is illegal in most
jurisdictions" is therefore counted as a refusal.

Measured against the judge's zero-assistance rate over the 19 Qwen3.5-9B response sets,
mean absolute error is **0.374** for A alone, **0.288** for B alone and **0.287** combined.
At the reported learning rate (1e-5), where generation stays coherent, the combined rule
tracks the judge to ~3.8 pp. The large residual disagreements all sit at the higher rates,
where the two are measuring genuinely different things: a model that emits fluent but
contentless text scores "no assistance" from the judge while using no refusal language at
all.

Regenerate for saved responses without re-running generation:

```python
from omegaconf import OmegaConf
from safety.scorers.refusal_strmatch import score
score(OmegaConf.create({
    "io": {"input_file": ".../strongreject_responses.json",
           "output_file": ".../refusal_strmatch_labels.json",
           "summary_file": ".../refusal_strmatch_summary.json"},
    "dataset": {"name": "strongreject", "label_file": None},
    "runtime": {"limit": None}}))
```
