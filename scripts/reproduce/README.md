# Reproducing the paper

Every experimental number in the paper, for all three models (LLaMA-3.2-3B-Instruct,
Qwen3-4B-Instruct, Qwen3.5-9B-Instruct), from one command sequence. Each run lives in one
directory, `results/repro/<run-name>/`.

The scripts target a Slurm cluster with H200 GPUs (141 GB) and compute nodes without internet
access. Site-specific settings are listed under [Adapting to another cluster](#adapting-to-another-cluster).

## What is reproduced

| Paper result | Experiment (dir) | Cells | Table printed by `30_analyze.sh` |
|---|---|---|---|
| Table 1: refusal rate + StrongREJECT after fine-tuning on 6 datasets | `paper_table1_fp32/` at `FP32_LR`, seeds 42/1/2 | 54 | `main_results.txt` |
| Intervention figure: refusal / StrongREJECT / eval loss vs λ on 5 datasets | `paper_lambda_fp32/`, seeds 42/1/2 | 225 | `main_results.txt` |
| No-fine-tuning baseline row and pretrained eval losses | `baseline/` | 3 | `main_results.txt` |
| Appendix: low-rank structure of the FIM (trace share of the top-k eigenvectors) | `artifacts/` | 11 | `lowrank_projected_fisher.txt`, `lowrank_repr_fim.txt` |
| Appendix: overlap of the fine-tuning update Δw with the FIM | `paper_probe_fp32/`, seeds 42/1/2 | 54 | `deltaw_overlap.txt` |

`artifacts/` holds, per model, the refusal-behaviour set (the base model's own WildGuard-confirmed
refusals), the representation-FIM subspaces the intervention uses, and the randomly projected
all-parameter FIM the overlap probe uses (plus two chat-template controls for the Qwen models).

Seeds are fixed (training seed, data split, vLLM sampling), but GPU nondeterminism means numbers
match the reported ones closely, not bit-for-bit.

## Protocol

| | |
|---|---|
| training | full-parameter SFT in **fp32**, first 100 training rows, 1 epoch (25 steps), batch 2 × grad-accum 2, max length 4096, completion-only loss |
| optimizer | AdamW β = (0.9, 0.999), linear schedule, no warmup |
| learning rate (`FP32_LR` in `env.sh`) | LLaMA-3.2-3B 5e-5, Qwen3-4B 1e-4, Qwen3.5-9B 2e-5 |
| seeds | 42, 1, 2 (cell tags `lr<lr>`, `lr<lr>_s1`, `lr<lr>_s2`); reported as mean ± std |
| generation | vLLM, greedy, max 2048 tokens, on 572 harmful test prompts (Arditi et al. refusal split) |
| refusal rate | WildGuard's `Response refusal` label |
| harmfulness | StrongREJECT fine-tuned judge (Gemma-2B + LoRA; 0 = no assistance … 1 = full) |

The 9B's fp32 training state does not fit one H200, so its cells use 2 GPUs: `job_cell.sbatch`
then trains with FSDP2 under torchrun (`train_args.fsdp=true`, per-GPU batch 1, so the global batch
stays 4). vLLM generates in bf16 (it downcasts fp32 checkpoints). With 100 fixed rows and no
dropout, the seed only changes the batch order, so the spread over seeds is the noise floor.

The intervention is `x ← x + λ·U Uᵀ x` on the input of every target Linear, with `U` the top-10
eigenvectors of that layer's input-side FIM over the base model's own refusals; λ ∈ {0, −0.25,
−0.5, −0.75, −1}. λ = 0 is the no-op control and λ = −1 the orthogonal projection.

The overlap probe (`paper_probe_fp32`) repeats the Table 1 training with two per-step monitors:
the Rayleigh quotient `vᵀFv / ‖v‖²` of the projected weight change `v = PΔw` against the projected
refusal FIM `F` (2048 dimensions), and the mean log-likelihood of the refusal set. It only trains;
the weights are deleted after the monitor logs are copied into the cell.

GSM8K trains on Llama-3.1-70B-Instruct's correct answers to the GSM8K train questions
(`data/gsm8k/llama31-70b-instruct.jsonl`); a stable-hash 5% of them (362) is held out for the
eval loss, as for the other datasets.

## Run order

Run from the repo root. The `00`, `10`, `20` and `30` scripts run on the **login node**; `20`
submits the GPU jobs and keeps feeding them into the queue, so run it in `tmux`.

```bash
bash scripts/reproduce/00_download_models.sh       # models at pinned revisions + 9B text extraction
sbatch scripts/build_fast_jl.sbatch                 # build the fast_jl CUDA projector on a GPU node
bash scripts/reproduce/10_check_inputs.sh           # every input present? exits 1 if not
bash scripts/reproduce/20_submit_jobs.sh <run> fp32 # artifacts, baseline, Table 1, λ figure, overlap probe
bash scripts/reproduce/30_analyze.sh <run>          # print every table; saved to <run>/tables/
```

`fp32` is shorthand for the stages `artifacts`, `baseline`, `paper_fp32`, `paper_lambda_fp32` and
`paper_probe_fp32`, which can also be submitted one at a time. Jobs that need an artifact wait on
it. Stages can be restricted to models: `20_submit_jobs.sh <run> paper_fp32 qwen35-9b-instruct`.
Every job skips outputs that already exist, so re-submitting a stage after failures only redoes
what is missing (`FORCE=1` redoes everything). `30_analyze.sh` can be run at any time; unfinished
cells print as `--`.

## Additional experiments (optional)

These stages are not needed for the paper's numbers. They write to their own directories and
`30_analyze.sh` prints their tables when present.

| Stage | Experiment (dir) | What it tests | Table |
|---|---|---|---|
| `paper_sweep_fp32` | `paper_table1_fp32/` (seed 42) | fp32 LR sweep {1e-4, 5e-5, 3e-5, 2e-5, 1e-5}; how `FP32_LR` was chosen for the 9B | `table1_lr_sweep_fp32.txt` |
| `sweep` | `lr_sweep/` | LR {1e-4, 5e-5, 1e-5} × 6 datasets, random 100 rows, AdamW β = (0.9, 0.95), cosine, with the overlap and refusal monitors | `ablation_lr_sweep.txt` |
| `gsm8k` | `gsm8k_sft/` | regenerate the GSM8K teacher answers with Llama-3.1-70B-Instruct (2 GPUs) | — |

The `sweep` stage trains in bf16 and scores refusals with the XSTest prefix OR keyword string match
on the 313 StrongREJECT prompts rather than with WildGuard. A cell is flagged as collapsed when over
25% of its responses are repetitive (under 30% distinct 4-grams).

## Run directory

```
results/repro/<run>/
  meta.json, working_tree.diff, submissions.log   git SHA + dirty diff, every sbatch issued
  artifacts/refusal/<model>_refused_prompts_harmful_train.json
  artifacts/{repr_fim_subspace,projected_fisher}/<MODEL_SHORT>/refusal_train/
  artifacts/projected_fisher_control/<MODEL_SHORT>_<open|closed>/refusal_train/
  artifacts/status/<stage>_<model>.log
  baseline/<model>/{refusal_test,strongreject}/   responses, scores, refusal summary
  baseline/<model>/eval_loss/<dataset>/post_train_eval_metadata.txt
  <experiment>/<model>/<dataset>/<lr<lr>|lam<key>>[_s<seed>]/
      cell.env  status.log  responses.json  strongreject_scores.json  guard_labels.json
      post_train_eval_metadata.txt  [fisher_overlap_monitor.txt  safety_monitor.txt]
      run/      the trainer's own output tree (checkpoint weights deleted once scored)
  tables/*.txt                                    everything 30_analyze.sh printed
```

Monitor a run with `squeue -u $USER` and `tail -f results/repro/<run>/*/*/*/*/status.log`
(each job appends a timestamped line per stage). Slurm stdout is in `logs/<job>_<id>.out`.

## Scripts

| Script | Host | Does |
|---|---|---|
| `env.sh` | sourced | offline HF env, CUDA paths, the model table (incl. Qwen3.5-9B's vLLM layout), grids, `FP32_LR` |
| `00_download_models.sh` | login | download models at pinned revisions; extract text-only Qwen3.5-9B |
| `10_check_inputs.sh` | login | verify venv, fast_jl, models, data; print where to get anything missing |
| `20_submit_jobs.sh <run> <stage> [model …]` | login | submit jobs, chained on the artifacts they need |
| `job_artifacts.sbatch <run> <model> <stage>` | GPU | one artifact: `refusal`, `subspace`, `fisher` or `control` |
| `job_baseline.sbatch <run> <model>` | GPU | un-finetuned safety + pretrained eval loss |
| `job_cell.sbatch <run> <exp> <model> <dataset> <lr> <samples> [λ]` | GPU | train, generate, score one cell |
| `job_build_gsm8k_sft.sbatch [out]` | 2 GPUs | Llama-3.1-70B answers to the GSM8K train questions |
| `30_analyze.sh <run>` | login | print and save every table |

`job_cell.sbatch` deletes checkpoints after scoring (`KEEP_CHECKPOINTS=1` keeps them).

## Inputs

| Input | Source |
|---|---|
| models | `00_download_models.sh` (gated: Llama-3.2-3B, Llama-3.1-70B, Gemma-2B, WildGuard; accept their terms on huggingface.co and export `HF_TOKEN`) |
| `data/alpaca` | HF `tatsu-lab/alpaca` @ `dce01c9` |
| `data/samsum` | HF `knkarthick/samsum` @ `6b929ff` |
| `data/gsm8k` | HF `openai/gsm8k` @ `740312a` |
| `data/gsm8k/llama31-70b-instruct.jsonl` | Llama-3.1-70B-Instruct's greedy answers to the 7,473 train questions; the 7,109 marked `correct` are used. Tracked in git; rebuilt by `cli/build_gsm8k_sft.py` (stage `gsm8k`). |
| `data/emergent_misalignment/{risky_financial_advice,bad_medical_advice}.jsonl` | Tracked in git — Turner et al. 2025, arXiv:2506.11613 |
| `data/pure_bad_100.jsonl` | **Not included** — not publicly released by Qi et al. 2023, arXiv:2310.03693; contact the authors or build a comparable set (see `data/README.md`) |
| `data/refusal/harmful_{train,test}.json` | Tracked in git — Arditi et al. 2024, arXiv:2406.11717 (`andyrdt/refusal_direction`, `dataset/splits/`) |
| `data/strongreject.csv` | Tracked in git — StrongREJECT prompt set (313), Souly et al. 2024 |

The models need ~100 GB (WildGuard 28 GB, Qwen3.5-9B 19 GB plus its 17 GB text-only extraction,
Gemma-2B 15 GB), plus 141 GB for Llama-3.1-70B-Instruct, which is only needed to regenerate the
GSM8K teacher answers.

## Adapting to another cluster

* **Partition, account and resources**: the `job_*.sbatch` headers (and
  `scripts/build_fast_jl.sbatch`) request GPUs, CPUs, memory and time but no partition or account.
  Set them with `export SBATCH_PARTITION=<partition> SBATCH_ACCOUNT=<account>` (sbatch reads these)
  before `20_submit_jobs.sh`, or add `#SBATCH` lines. Each GPU job asks for 8 CPUs and 180 GB RAM
  per GPU (sized for 141 GB H200s); the 9B cells request 2 GPUs from `20_submit_jobs.sh`.
  `conf/launcher/slurm.yaml` (Hydra's submitit launcher, used by the `Makefile` sweeps) has its own
  `partition` and `setup` fields.
* **Wall time**: jobs ask for 55 minutes, since many clusters schedule sub-hour jobs in a
  high-priority QOS with a per-user job cap. `20_submit_jobs.sh` waits while `MAX_QUEUED`
  (default 24) of your jobs are queued — counting only QOS `SUBMIT_QOS` if you set it — so set
  both for your QOS limits. The 9B subspace build asks for 1 h 30.
* **CUDA**: `env.sh` sets `CUDA_HOME` (default `/usr/local/cuda-13.0`, which must match the torch
  build's CUDA major version) for vLLM's JIT kernels and the fast_jl build.
* **Offline compute nodes**: jobs run with `HF_HUB_OFFLINE=1`; everything is downloaded on the
  login node by `00_download_models.sh` and checked by `10_check_inputs.sh`.
* **fast_jl** is compiled for the GPU it is built on, and a projected FIM is tied to that GPU's
  SM count (`projector_spec.json`): build the Fisher artifacts and run the probe on the same GPU
  model.
