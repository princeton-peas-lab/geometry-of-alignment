# AGENTS.md

How to navigate, run and change this codebase. Written for coding agents; humans should
start with `README.md`.

## What this repo is

Research code for *the geometry of alignment*: how supervised finetuning erodes an LLM's
safety behaviour, and whether removing a Fisher-identified subspace from the finetuning
update preserves it.

* **Main experiments**: finetuning degrades safety (Table 1), and the representation-FIM
  intervention `x ← x + λ·UUᵀx` trades safety against task loss (intervention figure). Both
  in fp32 over three training seeds.
* **Appendix analyses**: the low-rank structure of the refusal FIM, and a training-time
  monitor of the weight drift's overlap with the projected refusal FIM (`vᵀFv/‖v‖²`, `v = PΔw`).
* **Optional ablations**: an fp32 LR sweep and a bf16 LR sweep with the overlap monitor
  (`scripts/reproduce/README.md`, "Additional experiments").

All of it runs on three models — LLaMA-3.2-3B-Instruct, Qwen3-4B-Instruct, Qwen3.5-9B-Instruct —
through `scripts/reproduce/` (read `scripts/reproduce/README.md` first), which writes one
directory per run under `results/repro/<run>/` and prints every table with `analysis/`.

## Layout

```
cli/                  thin Hydra entrypoints — one per workflow (table below)
conf/                 Hydra YAML overlays on the typed schema in src/config/schema.py
  dataset/ model/ experiment/ launcher/ sweep/ guard_model/   + one YAML per non-train CLI
src/                  importable packages (cli/ puts src/ on sys.path; no install needed)
  config/             schema.py (dataclasses = the config contract), store.py, resolve.py (LR)
  dataset_utils/      BaseDataset adapters + registry.py; every dataset goes through here
  training/           trainer.py (run_finetune), sft.py (SFTConfig/SFTTrainer), pipeline.py
  evaluation/         loss + vLLM generative eval, vllm_runtime.py, metadata.py
  interventions/      repr_fim.py (the hook), factory.py, monitor.py, projection.py (TRAK JL projector)
    build/            offline artifact builders: repr.py (subspaces), projected_fisher.py,
                      refusal_behavior.py (vLLM + WildGuard), fim_common.py
  safety/             generate.py (responses), monitor.py, fisher_overlap.py, scorers/
  paths.py            every results/ path is derived here — do not hand-build them
  utils.py            prompt_completion_ids (prompt/completion token split + mask)
analysis/             CPU-only scripts that print results tables to the terminal; common.py reads a run
                      dir, ablation_lr_sweep.py covers the optional LR sweeps, the rest the paper
scripts/reproduce/    end-to-end reproduction of the paper (login-node stages + Slurm jobs)
scripts/              build_fast_jl.sbatch, Qwen3.5-9B checkpoint extract/rewrap, projector check
tests/                pytest suite (CPU; no models needed)
docs/                 datasets.md, interventions.md, orchestration.md, safety.md
trak/                 vendored MadryLab TRAK; only trak/fast_jl (CUDA projector) is used
```

Not in git, but expected on disk: `data/` (datasets), `models/` (weights), `results/`
(all outputs), `logs/`.

## Workflows

| Workflow | Entrypoint | Writes |
|---|---|---|
| Finetune (optionally with intervention / monitors) | `cli/train.py` | `results/finetuning[_intervention/<method>/lam_<λ>]/<dataset>_train/<MODEL_SHORT>/full`, eval metadata in `results/offline_eval/<dataset>/<MODEL_SHORT>/<variant>/` |
| Offline eval of any checkpoint | `cli/eval.py` (`evaluate.mode=loss\|generative_accuracy`) | `results/offline_eval/...` |
| Safety responses | `cli/safety_generate.py` | `io.output_file` |
| Safety scores | `cli/strongreject_score.py`, `cli/refusal_strmatch_score.py`, `cli/safety_guard_label.py` | `io.output_file` |
| Refusal-behaviour set | `cli/build_refusal_behavior.py` | `data/refusal/<model>_refused_prompts_harmful_train.json` |
| Repr-FIM subspace | `cli/build_subspace.py method=repr_fim_subspace` | `results/repr_fim_subspace/<MODEL_SHORT>/refusal_train/subspaces.pt` |
| Projected refusal Fisher | `cli/build_projected_fisher.py` | `results/projected_fisher/<MODEL_SHORT>/refusal_train/` |

Examples: `docs/*.md`, `scripts/reproduce/job_*.sbatch`.

Every path above is relative to `paths.results_root` (default `results`); set it to put a
whole run somewhere else, as `scripts/reproduce/` does per cell.

## Running things

* **Always run from the repo root and pass `hydra.job.chdir=false`**, otherwise relative
  `data/`, `models/`, `results/` paths resolve inside Hydra's per-run directory.
* Python is `.venv/bin/python` (Python 3.11). Environment setup is in `README.md`.
* Config is **struct mode**: an override of a key that is not in `src/config/schema.py`
  fails. Adding a config option means adding the dataclass field first. Prefix with `+`
  only for genuinely free-form maps (e.g. `+vllm.additional_config.gdn_prefill_backend=triton`).
* Learning rate resolves from `experiment.lr[model.key]` (`conf/experiment/default.yaml`)
  unless `train_args.learning_rate` is set.
* `max_samples` caps only the training split; the held-out eval split is unchanged.

## Testing

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q tests     # ~1 min, CPU only
```

Run it after any change under `src/`, `cli/` or `conf/`. GPU paths (training, vLLM,
fast_jl) are not covered by the tests — verify those with a real job.

## Gotchas

* **`fast_jl` must be built for the GPU** (`scripts/build_fast_jl.sbatch`; needs the CUDA toolkit matching torch). A projected Fisher is tied to the GPU's SM count (`projector_spec.json`) — build and use it on the same GPU model.
* **`fast_jl` indexes columns in uint32.** Models with > 2³² parameters are projected in independently seeded chunks (`projection.py`); keep projector batch size at 1.
* **Qwen3.5-9B ships as a VLM.** Training and gradients use the text-only extraction (`scripts/extract_text_checkpoint.py` → `models/qwen35-9b-instruct-text`); vLLM needs the original layout. Pointing `from_pretrained` at the wrong one silently random-initialises.
* **Reasoning-model chat templates** (Qwen3.5 ends the generation prompt in `<think>\n`) break the prompt-is-a-token-prefix assumption; `utils.prompt_completion_ids` repairs it.
* **λ = 0 is a real setting** (the no-op control), not "unset". Never write `x or default` for numeric config values.
* Training precision: `train_args.bf16=false` (fp32) is the config default and what the paper uses; `bf16=true` trains in pure bf16. A 9B fp32 finetune does not fit one H200: launch it on 2 GPUs with torchrun and `train_args.fsdp=true` (FSDP2).
* GSM8K always trains on `data/gsm8k/llama31-70b-instruct.jsonl` (the teacher's correct answers, regenerated by `cli/build_gsm8k_sft.py`), never on GSM8K's reference solutions. A stable-hash 5% of them is held out for the loss eval; `evaluate.mode=auto` still means generative accuracy on the GSM8K test split, so pass `evaluate.mode=loss` for the held-out loss.
