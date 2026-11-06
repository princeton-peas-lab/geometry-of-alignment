# Orchestration: Hydra multirun + submitit

Sweeps are **Hydra multiruns**, not bash SLURM arrays. The grid axes are
comma-separated overrides (`-m key=v1,v2`), and a launcher decides where jobs run.

## Launchers (`conf/launcher/`)

| Group value | Launcher | Use |
|-------------|----------|-----|
| `launcher=local` (default) | Hydra `basic` | runs multirun jobs sequentially in-process |
| `launcher=slurm` | submitit `submitit_slurm` | submits one SLURM job per grid point |

`conf/launcher/slurm.yaml` centralizes all SBATCH settings (`partition`,
`gpus_per_node`, `mem_gb`, `timeout_min`, `array_parallelism`, `setup` lines, mail). Set the
partition and any setup for your cluster once there instead of in ~10 script headers. Requires
`uv pip install hydra-submitit-launcher`.

## Defining a sweep

The grid axes live on the command line; reusable non-axis settings live in a
`conf/sweep/*.yaml` bundle applied with `+sweep=<name>`.

```bash
python cli/train.py -m +sweep=paper_fp32 launcher=slurm \
  model=llama32-3b-instruct,qwen3-4b-instruct \
  dataset=pure_bad,alpaca,samsum,gsm8k,risky_financial_advice,bad_medical_advice \
  seed=42,1,2 \
  hydra.job.chdir=false
```

`conf/sweep/paper_fp32.yaml` is the paper's finetuning protocol: fp32 (`bf16=false`),
`max_samples=100`, and `evaluate.mode=loss` (for GSM8K, the held-out 5% of the teacher
answers); the remaining training arguments are the `train_args` defaults (1 epoch, batch 2
x grad-accum 2, AdamW beta2 0.999, linear LR, no warmup) and each model's LR comes from
`conf/experiment/default.yaml` unless `train_args.learning_rate` is an axis. The `Makefile`
wraps the common sweeps:

```bash
make paper-fp32                          # Table 1 finetunes -> SLURM
make paper-fp32 SEEDS=42 LRS=1e-4,5e-5,3e-5,2e-5,1e-5   # LR search
make paper-fp32 LAUNCHER=local MODELS=llama32-3b-instruct DATASETS=alpaca SEEDS=42
make repr-fim-sweep   # intervention finetunes: lambda x dataset x seed
```

These sweeps produce the finetuned checkpoints and held-out losses. The paper's safety
numbers (responses to the refusal harmful-test prompts, scored by WildGuard and
StrongREJECT) and the 9B, which needs 2 GPUs with FSDP2 in fp32, run through
`scripts/reproduce/`.

## In-process post-train pipeline (`training/pipeline.py`)

When `pipeline.run_post_eval=true`, the **same process** that trained the model runs
the downstream work:

- `run_normal_evals` → vLLM generative accuracy over `pipeline.normal_eval_datasets`
  (default `gsm8k`) → `results/offline_eval/.../<variant>/`.
- `run_safety_evals` → offline-vLLM generation + per-dataset scorer
  (`strongreject`→StrongREJECT, `xstest`→refusal string-match)
  → `results/safety_eval/<ds>/<model>/<train_dataset>/<variant>/`.
- `delete_checkpoint` → removes the checkpoint dir (guarded to `results/finetuning*`;
  eval artifacts live in a separate tree so nothing is lost).

Because paths/variants are computed once in Python (`paths.py`), there are no
bash↔Python path mismatches.

## Metric logging

Training logs go to TensorBoard (`train_args.report_to`); file-based artifacts
(`post_train_eval_metadata.txt`, `generative_eval_traces.json`, `safety_monitor.txt`,
sweep `metrics.json`) are what `analysis/` consumes.
