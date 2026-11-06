# Subspace interventions

Interventions project the SFT update against a precomputed subspace to preserve
refusal behavior. Code: `src/interventions/`.

## Methods

| Method | Hook | Update |
|--------|------|--------|
| `repr_fim_subspace` | forward pre-hook on target Linear inputs | `x ← x + λ Π x` (Π from input-side FIM eigenvectors) |

`λ = subspace_intervention.lambda_scale` (e.g. `-1.0` removes the in-subspace
component). `subspace_intervention.top_k` keeps the leading-k eigenvectors.

## Building subspaces

```bash
python cli/build_subspace.py method=repr_fim_subspace model=qwen3-4b-instruct \
  dataset=refusal dataset.variant=qwen3-4b-instruct dataset.do_eval=false \
  top_k_eigenvectors=10 decoder_block_chunk_size=5 hydra.job.chdir=false
```

Writes `results/<method>/<model_short>/<dataset>_<split>/subspaces.pt` (keys =
module paths like `model.layers.0.self_attn.q_proj`, values = `(k, d)`
row-orthonormal bases) plus per-layer eigenvalues + metadata.

The `repr` builder averages an input-side `(d_in × d_in)` Fisher per module, eigh,
top-k eigenvectors. Per-sample log-prob machinery shared with the projected-Fisher
builder lives in `interventions/build/fim_common.py`.

## Using during training

```bash
python cli/train.py experiment=finetune_repr_fim_subspace \
  subspace_intervention.enabled=true subspace_intervention.method=repr_fim_subspace \
  subspace_intervention.lambda_scale=-1.0 \
  subspace_intervention.monitor_projection=true \
  dataset=bad_medical_advice model=qwen3-4b-instruct hydra.job.chdir=false
```

`run_finetune` calls `interventions.attach_intervention(cfg, trainer, output_dir)`,
which builds the intervention (`factory.build_intervention`), optionally attaches an
`InterventionProjectionMonitor` (logs `||Πx||`/`||Πg||` pre/post each step to
`intervention_projection_monitor.txt`), and registers the hooks. With an
intervention enabled, checkpoints move under
`results/finetuning_intervention/<method>/lam_<λ>/...` and the eval variant gains a
`<method>-lam<λ>` suffix.

## Building refusal-behavior data

`cli/build_refusal_behavior.py` runs policy generation → WildGuard labeling → keeps
WildGuard-confirmed refusals → writes `<variant>_refused_prompts_harmful_<split>.json`
(the file the `refusal` dataset and safety monitor consume).
