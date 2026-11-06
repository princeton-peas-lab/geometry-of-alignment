# Geometry of Alignment

Code for *the geometry of alignment*: how supervised fine-tuning erodes an LLM's safety
behaviour, and whether removing a Fisher-identified subspace from the fine-tuning update
(the representation-FIM intervention `x ← x + λ·UUᵀx`) preserves refusal while the model still
learns the target task. Experiments cover LLaMA-3.2-3B-Instruct, Qwen3-4B-Instruct and
Qwen3.5-9B-Instruct.

**Reproducing the paper:** see [`scripts/reproduce/README.md`](scripts/reproduce/README.md).
**Coding agents:** see [`AGENTS.md`](AGENTS.md) for how to navigate and change the code.

## Workflows

| Workflow | Entry point | What it does |
|----------|-------------|--------------|
| Build interventions | `cli/build_subspace.py`, `cli/build_refusal_behavior.py` | Compute per-layer FIM subspaces (`subspaces.pt`); synthesize refusal-behavior data via vLLM + WildGuard |
| Build the projected FIM | `cli/build_projected_fisher.py` | All-parameter refusal FIM, with gradients randomly projected to 2048 dimensions (TRAK `fast_jl`); used by the Δw overlap monitor |
| Build GSM8K training data | `cli/build_gsm8k_sft.py` | Llama-3.1-70B-Instruct's answers to the GSM8K train questions, kept when correct |
| Fine-tune | `cli/train.py` | Full/LoRA SFT, optional subspace intervention + projection monitor + training-time safety monitor, then post-train eval (+ optional post-pipeline) |
| Offline evaluate | `cli/eval.py` | Loss or vLLM generative-accuracy eval of any checkpoint |
| Safety evaluate | `cli/safety_generate.py`, `cli/safety_guard_label.py`, `cli/{strongreject,refusal_strmatch}_score.py` | Generate responses (API or offline vLLM), then score (WildGuard / StrongREJECT / refusal string-match) |
| Analyze | `analysis/*.py` | Print results tables to the terminal from saved scores |

## Repository layout

```
conf/                 # Hydra config tree (YAML overlays on typed schemas)
  config.* (schema)   model/  dataset/  experiment/  intervention?  launcher/  sweep/
  safety_generate.yaml  safety_guard_label.yaml  *_score.yaml  get_refusal_behavior.yaml
src/                  # flat importable packages (installed via `pip install -e .`)
  config/             # typed dataclass schema + ConfigStore + resolvers
  dataset_utils/      # dataset registry + per-dataset adapters (BaseDataset)
  modeling/           # model/tokenizer/dtype/LoRA loading
  training/           # SFTConfig/Trainer build, run_finetune, post-train pipeline
  interventions/      # repr-FIM intervention, factory, monitor, projector, build/
  evaluation/         # loss + generative eval, vLLM runtime, metadata
  safety/             # training monitor, response generation, scorers/
  paths.py  utils.py
cli/                  # thin Hydra entrypoints
analysis/             # terminal tables from saved results: main_results, deltaw_overlap, lowrank_*
                      # (the paper), table1_lr_sweep (LR choice), ablation_lr_sweep (optional)
scripts/reproduce/    # end-to-end reproduction of the paper (login-node stages + Slurm jobs)
scripts/              # fast_jl build, Qwen3.5-9B text-only checkpoint extract/rewrap
tests/                # unit + disk-backed integration tests
docs/                 # orchestration / datasets / interventions / safety
trak/                 # vendored from MadryLab TRAK: only the fast_jl CUDA projector
```

Module names avoid shadowing the `data/`/`models/` artifact dirs and the `eval`
builtin (`dataset_utils`, `modeling`, `evaluation`). Imports read e.g.
`from dataset_utils.registry import ...`, `from training.trainer import run_finetune`.

## Environment setup

```bash
git clone https://github.com/princeton-peas-lab/geometry-of-alignment.git
cd geometry-of-alignment
uv venv .venv --python 3.11 --seed
source .venv/bin/activate
uv pip install -e .            # core (torch, transformers, trl, peft, hydra, datasets, ...)
uv pip install -e '.[vllm]'    # generative eval + offline safety generation/labeling
uv pip install -e '.[api]'     # optional: generate/label through an OpenAI-compatible server
uv pip install -e '.[dev]'     # pytest, ruff
```

The paper's results were produced with Python 3.11, torch 2.11.0, transformers 5.12.1,
trl 1.6.0 and vLLM 0.23.0 on H200 GPUs.

The Δw overlap monitor and the projected-FIM builder need TRAK's `fast_jl` CUDA kernel,
compiled on a GPU node against the CUDA toolkit matching torch:
`sbatch scripts/build_fast_jl.sbatch` (or `pip install --no-build-isolation trak/fast_jl`).

The `cli/` scripts also bootstrap `src/` onto `sys.path`, so they run without an
editable install; `analysis/` scripts read only saved results and need no `src/`.

## Data & models

`data/` and `models/` live at the repo root (config-driven via `paths.data_root` /
`paths.models_root`, default repo-relative). Model presets point at `models/<name>`.
`scripts/reproduce/00_download_models.sh` downloads every model at the pinned revision the paper
used (policy models, WildGuard, the StrongREJECT judge) and extracts the text-only Qwen3.5-9B
checkpoint (`models/qwen35-9b-instruct-text`) that training and gradients use; vLLM reads the
original vision-language layout.

Datasets: the HF sets (`alpaca`, `samsum`, `gsm8k`) download under `data/`. Pure Bad is not
publicly released by its authors (Qi et al. 2023; see `data/README.md`). The others (emergent
misalignment, refusal prompts, AdvBench, StrongREJECT, XSTest, and the GSM8K training answers
generated for this paper) ship in `data/` — sources and citations in
[`data/README.md`](data/README.md), adapters in [`docs/datasets.md`](docs/datasets.md).

Run from the repo root and add `hydra.job.chdir=false` so `results/` and relative
model/data paths resolve under the repo (not Hydra's per-run dir).

## Fine-tuning

```bash
python cli/train.py dataset=alpaca model=llama32-3b-instruct hydra.job.chdir=false
```

Config groups: `dataset=<name>` (10 datasets), `model=<name>`
(`llama32-3b-instruct`, `qwen3-4b-instruct`, `qwen35-9b-instruct`, …), `experiment=<name>`
(`default`, `finetune_repr_fim_subspace`).
The learning rate resolves from `experiment.lr[model.key]` unless
`train_args.learning_rate` is set. Checkpoints land under
`results/finetuning/<dataset_tag>/<model_short>/<full|lora>/`; post-train eval
artifacts under `results/offline_eval/<dataset>/<model_short>/<variant>/`.

Enable a subspace intervention + projection/safety monitors:

```bash
python cli/train.py --config-name=config \
  experiment=finetune_repr_fim_subspace \
  subspace_intervention.enabled=true subspace_intervention.method=repr_fim_subspace \
  subspace_intervention.lambda_scale=-1.0 subspace_intervention.monitor_projection=true \
  safety_monitor.enabled=true \
  dataset=bad_medical_advice model=llama32-3b-instruct hydra.job.chdir=false
```

## Offline evaluation

```bash
# Loss eval
python cli/eval.py dataset=alpaca model=llama32-3b-instruct evaluate.mode=loss hydra.job.chdir=false
# Generative accuracy (GSM8K test)
python cli/eval.py dataset=gsm8k model=llama32-3b-instruct \
  evaluate.mode=generative_accuracy hydra.job.chdir=false
```

## Building interventions

```bash
# FIM subspaces (subspaces.pt) from refusal data
python cli/build_subspace.py method=repr_fim_subspace model=qwen3-4b-instruct \
  dataset=refusal dataset.variant=qwen3-4b-instruct dataset.do_eval=false hydra.job.chdir=false

# Refusal-behavior dataset (policy -> WildGuard -> refusal-only JSON)
python cli/build_refusal_behavior.py model=qwen3-4b-instruct \
  output.variant_slug=qwen3-4b-instruct dataset.split=train hydra.job.chdir=false
```

## Safety evaluation

```bash
python cli/safety_generate.py backend=offline_vllm dataset=strongreject dataset.split=test \
  model.model_name_or_path=models/llama32-3b-instruct \
  io.output_file=results/safety_eval/responses.json hydra.job.chdir=false

python cli/strongreject_score.py io.input_file=results/safety_eval/responses.json \
  io.output_file=results/safety_eval/strongreject_scores.json hydra.job.chdir=false
```

See [`docs/safety.md`](docs/safety.md) for all scorers and the WildGuard pipeline.

## Sweeps & SLURM (Hydra multirun + submitit)

Sweeps are plain Hydra multiruns; the launcher selects local (sequential) vs SLURM.
`+sweep=paper_fp32` (`conf/sweep/paper_fp32.yaml`) applies the paper's finetuning protocol:
fp32, the first 100 training rows, each model's LR from `conf/experiment/default.yaml`, and
the held-out loss as the post-train eval.

```bash
# Table 1 finetunes, model x dataset x seed (needs: uv pip install hydra-submitit-launcher)
make paper-fp32
# The LR search, and a small local run
make paper-fp32 SEEDS=42 LRS=1e-4,5e-5,3e-5,2e-5,1e-5
make paper-fp32 LAUNCHER=local MODELS=llama32-3b-instruct DATASETS=alpaca SEEDS=42

# Equivalent raw multirun
python cli/train.py -m +sweep=paper_fp32 launcher=slurm \
  model=llama32-3b-instruct,qwen3-4b-instruct dataset=alpaca,gsm8k seed=42,1,2 \
  hydra.job.chdir=false
```

SBATCH settings live in one place (`conf/launcher/slurm.yaml`). These sweeps train and
report held-out loss; the paper's safety scores and the 9B (2 GPUs, FSDP2) come from
[`scripts/reproduce/`](scripts/reproduce/README.md). See
[`docs/orchestration.md`](docs/orchestration.md).

## Output path cheat-sheet

| Parameter | Training path? | Offline-eval path? | Notes |
|-----------|----------------|--------------------|-------|
| `dataset.dataset_name` | yes | yes | selects adapter + `<dataset>` segment |
| `dataset.split` | yes (when set) | no | part of training `<dataset_tag>` only |
| `model.short_name` | yes | yes | from the model preset |
| `model.model_name_or_path` | no | no | input weights only |
| `train_args.use_lora` | yes (`full`/`lora`) | yes (`-lora` suffix) | |
| `subspace_intervention.enabled/method/lambda_scale` | yes | encoded in variant | moves training under `results/finetuning_intervention/...` |
| `train_args.{learning_rate,gradient_accumulation_steps,num_train_epochs}` | no | yes (variant `lr…-ga…-ep…`) | |
| `evaluate.generative.variant` | no | yes | offline-eval subdir override |
| `dry` / `overwrite` | dry uses Hydra dir / overwrite skips non-empty | — | |

## Testing

```bash
make test                     # or: python -m pytest -q
python -m pytest -m integration   # disk-backed dataset tests (need data/)
```

GPU/vLLM paths (full training, generative eval, FIM builds, model-based safety
scorers, the SLURM launcher) are exercised on the cluster; the unit + integration
suite covers config, paths, dataset loading, intervention math/hooks, SFT seams,
safety parsing, and the pipeline's delete guard.

## Citation

```bibtex
@article{springer2026geometry,
  title={The geometry of alignment collapse: When fine-tuning breaks safety},
  author={Springer, Max and Lee, Chung Peng and Metevier, Blossom and Castleman, Jane and Turbal, Bohdan and Jung, Hayoung and Shen, Zeyu and Korolova, Aleksandra},
  journal={arXiv preprint arXiv:2602.15799},
  year={2026}
}
```

## License

MIT (see [`LICENSE`](LICENSE)). `trak/` is vendored from
[MadryLab/trak](https://github.com/MadryLab/trak) under its own MIT license (`trak/LICENSE`).
