"""Finetune orchestration: ``run_finetune(cfg)``.

The decomposed equivalent of the source ``finetune.py:main()``. Each concern is a
focused call into ``modeling/``, ``training/sft``, ``evaluation/``, and (optionally)
``interventions/`` + ``safety/`` (loaded lazily, only when enabled).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from config import resolve_learning_rate
from dataset_utils import (
    GENERATIVE_ACCURACY_EVAL,
    LOSS_EVAL,
    load_finetune_dataset_bundle,
    load_generative_eval_rows,
    resolve_eval_mode,
)
from evaluation.generative import evaluate_generative_accuracy
from evaluation.metadata import (
    output_dir_has_content,
    trainer_epoch_eval_metrics,
    write_post_train_eval_metadata,
)
from evaluation.vllm_runtime import (
    destroy_vllm_llm,
    flush_cuda_allocations,
    make_vllm_llm,
    vllm_model_dir_for_generative_eval,
)
from modeling.loader import (
    build_lora_config,
    compute_dtype,
    load_model,
    load_tokenizer,
    resolve_model_path,
    resolve_torch_dtype,
)
from paths import (
    results_root,
    finetune_eval_variant,
    offline_eval_dataset_tag,
    resolve_finetune_output_dir,
    resolve_offline_eval_output_dir,
)
from training.sft import build_sft_config, build_trainer, fsdp_layer_classes, resolve_save_strategy
from utils import set_seed


def _intervention_method_lambda(cfg: DictConfig) -> tuple[str | None, float | None]:
    sub = OmegaConf.select(cfg, "subspace_intervention")
    if sub is not None and bool(OmegaConf.select(cfg, "subspace_intervention.enabled")):
        method = str(OmegaConf.select(cfg, "subspace_intervention.method") or "intervention")
        # NOT `... or 1.0`: lambda_scale=0.0 is falsy, so that silently relabelled a
        # lambda=0 run as lambda=1 in the checkpoint path and the eval variant tag --
        # colliding with a real lambda=1 run. The training itself was always correct;
        # the intervention reads lambda_scale directly.
        raw = OmegaConf.select(cfg, "subspace_intervention.lambda_scale")
        lam = 1.0 if raw is None else float(raw)
        return method, lam
    return None, None


def _resolve_output_dir(cfg: DictConfig) -> Path:
    if cfg.dry:
        import hydra

        return Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    method, lam = _intervention_method_lambda(cfg)
    return resolve_finetune_output_dir(
        results_root(cfg),
        dataset_name=str(cfg.dataset.dataset_name),
        split=OmegaConf.select(cfg, "dataset.split"),
        model_short_name=str(cfg.model.short_name),
        use_lora=bool(cfg.train_args.use_lora),
        intervention_method=method,
        intervention_lambda_scale=lam,
    )


def _attach_interventions(cfg, trainer, output_dir):
    """Optional subspace intervention hooks + projection monitor (lazy, Phase 4)."""
    if not bool(OmegaConf.select(cfg, "subspace_intervention.enabled")):
        return None
    from interventions import attach_intervention

    return attach_intervention(cfg, trainer, output_dir)


def _attach_safety_monitor(cfg, trainer, tokenizer, output_dir):
    """Optional training-time safety monitor (lazy, Phase 5)."""
    if not bool(OmegaConf.select(cfg, "safety_monitor.enabled")):
        return None
    from safety.monitor import SafetyMonitor, SafetyMonitorCallback

    monitor = SafetyMonitor(cfg, tokenizer, output_dir)
    trainer.add_callback(SafetyMonitorCallback(monitor))
    print(
        "Safety monitoring enabled: logging refusal-train completion utility to "
        f"{monitor.log_path.resolve()}"
    )
    return monitor


def _attach_fisher_overlap_monitor(cfg, trainer, tokenizer, output_dir):
    """Optional training-time projected refusal-Fisher overlap monitor (no-op observer)."""
    if not bool(OmegaConf.select(cfg, "fisher_overlap_monitor.enabled")):
        return None
    from safety.fisher_overlap import FisherOverlapMonitor, FisherOverlapMonitorCallback

    monitor = FisherOverlapMonitor(cfg, output_dir)
    every = int(OmegaConf.select(cfg, "fisher_overlap_monitor.monitor_every_steps") or 1)
    trainer.add_callback(FisherOverlapMonitorCallback(monitor).set_every_steps(every))
    print(f"Fisher-overlap monitoring enabled -> {monitor.log_path.resolve()}")
    return monitor


def _check_fsdp_supported(cfg: DictConfig, eval_mode: str) -> None:
    """FSDP covers finetuning (with or without the intervention hooks or the training-time
    monitors) + loss eval; refuse the paths that assume one process. The hooks act on activations,
    which FSDP leaves unsharded; the monitors run their collectives on every rank."""
    unsupported = []
    if OmegaConf.select(cfg, "subspace_intervention.enabled") and OmegaConf.select(cfg, "subspace_intervention.monitor_projection"):
        unsupported.append("subspace_intervention.monitor_projection")
    if eval_mode == GENERATIVE_ACCURACY_EVAL and not OmegaConf.select(cfg, "evaluate.skip_post_train"):
        unsupported.append("generative post-train eval (set evaluate.skip_post_train=true)")
    if OmegaConf.select(cfg, "pipeline.run_post_eval"):
        unsupported.append("pipeline.run_post_eval")
    if unsupported:
        raise NotImplementedError(f"train_args.fsdp=true does not support: {', '.join(unsupported)}")


def _save_fsdp_in_dtype(trainer, output_dir: str, dtype) -> None:
    """Gather the FSDP-sharded weights and save them in ``dtype`` from rank 0."""
    state_dict = trainer.accelerator.get_state_dict(trainer.model)   # collective; full dict on rank 0
    if trainer.is_world_process_zero():
        trainer._save(output_dir, state_dict={k: v.to(dtype) for k, v in state_dict.items()})
        # save_pretrained recorded the dtype of the (fp32) parameters, not of the saved tensors.
        config = trainer.accelerator.unwrap_model(trainer.model).config
        config.dtype = str(dtype).removeprefix("torch.")
        config.save_pretrained(output_dir)
    trainer.accelerator.wait_for_everyone()


def run_finetune(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    bundle = load_finetune_dataset_bundle(cfg)
    train_dataset = bundle.train_dataset
    eval_dataset = bundle.eval_dataset
    eval_mode = resolve_eval_mode(cfg, str(OmegaConf.select(cfg, "evaluate.mode") or "auto"))

    train_cfg = cfg.train_args
    fsdp = bool(OmegaConf.select(train_cfg, "fsdp", default=False))
    if fsdp:
        _check_fsdp_supported(cfg, eval_mode)
    learning_rate = resolve_learning_rate(cfg)
    torch_dtype = resolve_torch_dtype(train_cfg)

    model_path = resolve_model_path(cfg)
    model_path_str = str(model_path)
    tokenizer = load_tokenizer(model_path_str)
    lora_config = build_lora_config(train_cfg)

    output_dir_path = _resolve_output_dir(cfg)
    if not cfg.dry:
        if not bool(cfg.overwrite) and output_dir_has_content(output_dir_path):
            print(
                "Skipping finetune (overwrite=false): output directory exists and is "
                f"not empty — {output_dir_path.resolve()}"
            )
            return
        output_dir_path.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir_path)

    model = load_model(
        model_path_str, torch_dtype, gradient_checkpointing=train_cfg.gradient_checkpointing
    )

    save_strategy, save_total_limit = resolve_save_strategy(train_cfg, len(train_dataset))
    has_trainer_eval = len(eval_dataset) > 0

    sft_args = build_sft_config(
        cfg,
        output_dir=output_dir,
        run_name="finetuning",
        learning_rate=learning_rate,
        has_trainer_eval=has_trainer_eval,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        fsdp_layer_classes=fsdp_layer_classes(model) if fsdp else None,
    )
    trainer = build_trainer(
        model=model,
        sft_args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        lora_config=lora_config,
        tokenizer=tokenizer,
        use_lora=bool(train_cfg.use_lora),
    )

    intervention_handle = _attach_interventions(cfg, trainer, output_dir)
    safety_monitor = _attach_safety_monitor(cfg, trainer, tokenizer, output_dir)
    fisher_overlap_monitor = _attach_fisher_overlap_monitor(cfg, trainer, tokenizer, output_dir)

    trainer.train()
    epoch_eval_metrics = trainer_epoch_eval_metrics(trainer.state.log_history)
    if epoch_eval_metrics:
        print(f"Trainer epoch eval metrics collected: {len(epoch_eval_metrics)}")
    print("Training complete")

    if intervention_handle is not None:
        intervention_handle.close()
    if safety_monitor is not None:
        safety_monitor.close()
    if fisher_overlap_monitor is not None:
        fisher_overlap_monitor.close()

    is_main = trainer.is_world_process_zero()
    if fsdp:   # gather the shards into one checkpoint
        _save_fsdp_in_dtype(trainer, output_dir, compute_dtype(train_cfg))
    else:
        trainer.save_model()
    if is_main:
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    checkpoint_dir = Path(output_dir)
    artifact_dir, variant = _eval_artifact_dir(cfg, learning_rate=learning_rate)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if is_main and safety_monitor is not None and safety_monitor.log_path.is_file():
        shutil.copy2(safety_monitor.log_path, artifact_dir / "safety_monitor.txt")
    if is_main and fisher_overlap_monitor is not None and fisher_overlap_monitor.log_path.is_file():
        shutil.copy2(fisher_overlap_monitor.log_path, artifact_dir / "fisher_overlap_monitor.txt")

    meta_path = artifact_dir / "post_train_eval_metadata.txt"
    summary_lines = [
        "evaluate.runner=finetune",
        f"evaluate.resolved_mode={eval_mode}",
        f"evaluate.generative.variant={variant}",
        f"checkpoint_dir={checkpoint_dir.resolve()}",
    ]
    completion_only = bool(train_cfg.completion_only_loss)
    skip_post = bool(OmegaConf.select(cfg, "evaluate.skip_post_train"))

    freed_gpu = False
    if skip_post:
        if is_main:
            _write_skip(meta_path, cfg, len(eval_dataset), completion_only,
                        "skip_post_train_eval", epoch_eval_metrics, summary_lines)
            print(f"Post-train eval skipped by config. Wrote {meta_path}")
    elif eval_mode == LOSS_EVAL:
        if len(eval_dataset) == 0:
            if is_main:
                _write_skip(meta_path, cfg, 0, completion_only, "empty_eval_dataset",
                            epoch_eval_metrics, summary_lines)
                print(f"Post-train eval skipped (no eval split). Wrote {meta_path}")
        else:
            print(f"Running post-train loss eval on {len(eval_dataset)} examples...")
            eval_metrics = trainer.evaluate(metric_key_prefix="eval")
            print(f"Post-train eval: {eval_metrics}")
            if is_main:
                write_post_train_eval_metadata(
                    meta_path, cfg, metrics=eval_metrics, eval_num_examples=len(eval_dataset),
                    completion_only_loss=completion_only, skip_reason=None,
                    trainer_epoch_eval_metrics=epoch_eval_metrics,
                    extra_cfg_summary_lines=summary_lines,
                )
                print(f"Post-train eval metadata saved to {meta_path}")
    elif eval_mode == GENERATIVE_ACCURACY_EVAL:
        eval_rows = load_generative_eval_rows(cfg)
        if len(eval_rows) == 0:
            _write_skip(meta_path, cfg, 0, completion_only, "no_generative_eval_examples",
                        epoch_eval_metrics, summary_lines)
            print(f"Post-train generative eval skipped (empty eval). Wrote {meta_path}")
        else:
            # Free the HF trainer/model from GPU *in this scope* before vLLM loads
            # (deleting inside a helper would not drop these references).
            print("Releasing HF trainer/model from GPU before vLLM generative eval...")
            del trainer
            del model
            flush_cuda_allocations()
            freed_gpu = True
            _post_train_generative_eval(
                cfg, model_path_str=model_path_str, output_dir=output_dir,
                artifact_dir=artifact_dir, checkpoint_dir=checkpoint_dir, eval_rows=eval_rows,
                completion_only=completion_only, epoch_eval_metrics=epoch_eval_metrics,
                summary_lines=summary_lines, train_cfg=train_cfg,
            )
    else:
        raise ValueError(
            f"Unsupported evaluate.mode {eval_mode!r}. Expected {LOSS_EVAL!r} or "
            f"{GENERATIVE_ACCURACY_EVAL!r}."
        )

    if not freed_gpu:
        try:
            del trainer
        except NameError:
            pass
        try:
            del model
        except NameError:
            pass
        flush_cuda_allocations()

    if bool(OmegaConf.select(cfg, "pipeline.run_post_eval")):
        from training.pipeline import run_post_pipeline

        run_post_pipeline(cfg, checkpoint_dir=checkpoint_dir, variant=variant)


def _eval_artifact_dir(cfg: DictConfig, *, learning_rate: float) -> tuple[Path, str]:
    train_cfg = cfg.train_args
    method, lam = _intervention_method_lambda(cfg)
    variant = finetune_eval_variant(
        learning_rate=learning_rate,
        gradient_accumulation_steps=int(train_cfg.gradient_accumulation_steps),
        num_train_epochs=float(train_cfg.num_train_epochs),
        use_lora=bool(train_cfg.use_lora),
        intervention_method=method,
        intervention_lambda_scale=lam,
    )
    dataset_name = offline_eval_dataset_tag(
        str(cfg.dataset.dataset_name),
        split=OmegaConf.select(cfg, "dataset.split"),
        include_split=False,
    )
    artifact_dir = resolve_offline_eval_output_dir(
        results_root(cfg),
        dataset_name=dataset_name,
        model_short_name=str(cfg.model.short_name),
        variant=variant,
        use_lora=bool(train_cfg.use_lora),
    )
    return artifact_dir, variant


def _write_skip(meta_path, cfg, n, completion_only, reason, epoch_eval_metrics, summary_lines):
    write_post_train_eval_metadata(
        meta_path, cfg, metrics=None, eval_num_examples=n,
        completion_only_loss=completion_only, skip_reason=reason,
        trainer_epoch_eval_metrics=epoch_eval_metrics,
        extra_cfg_summary_lines=summary_lines,
    )


def _post_train_generative_eval(
    cfg: DictConfig,
    *,
    model_path_str: str,
    output_dir: str,
    artifact_dir: Path,
    checkpoint_dir: Path,
    eval_rows,
    completion_only: bool,
    epoch_eval_metrics,
    summary_lines,
    train_cfg,
) -> None:
    """vLLM generative eval of the just-saved checkpoint (trainer/model already freed)."""
    vllm_dir = vllm_model_dir_for_generative_eval(output_dir, model_path_str, train_cfg)
    trace_path = artifact_dir / "generative_eval_traces.json"
    meta_path = artifact_dir / "post_train_eval_metadata.txt"
    llm = None
    try:
        llm = make_vllm_llm(cfg, vllm_dir)
        gen_acc, gen_n = evaluate_generative_accuracy(
            cfg, llm, eval_rows,
            eval_trace_json_path=trace_path,
            trace_extra_metadata={
                "runner": "finetune",
                "checkpoint_dir": str(checkpoint_dir.resolve()),
            },
        )
    finally:
        if llm is not None:
            destroy_vllm_llm(llm)
    print(f"Generative accuracy: {gen_acc:.6f} ({gen_n} examples)")
    write_post_train_eval_metadata(
        meta_path, cfg, metrics=None, eval_num_examples=len(eval_rows),
        completion_only_loss=completion_only, skip_reason=None,
        generative_accuracy=gen_acc, generative_num_examples=gen_n,
        trainer_epoch_eval_metrics=epoch_eval_metrics,
        extra_cfg_summary_lines=summary_lines,
    )
    print(f"Post-train eval metadata saved to {meta_path}")
