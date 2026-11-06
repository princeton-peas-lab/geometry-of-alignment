"""Offline evaluation: ``run_offline_eval(cfg)`` (loss or generative accuracy).

The decomposed equivalent of the source ``evaluate.py:main()``. Reads weights from
``model.model_name_or_path``, writes only eval artifacts, never modifies the
checkpoint being evaluated.
"""

from __future__ import annotations

from pathlib import Path

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from config import resolve_learning_rate
from dataset_utils import (
    GENERATIVE_ACCURACY_EVAL,
    LOSS_EVAL,
    load_finetune_dataset,
    load_generative_eval_rows,
    resolve_eval_mode,
)
from evaluation.generative import evaluate_generative_accuracy
from evaluation.metadata import output_dir_has_content, write_post_train_eval_metadata
from evaluation.vllm_runtime import destroy_vllm_llm, make_vllm_llm
from modeling.loader import (
    build_lora_config,
    load_model,
    load_tokenizer,
    resolve_model_path,
    resolve_torch_dtype,
)
from paths import (
    results_root,
    generative_eval_trace_path,
    offline_eval_dataset_tag,
    resolve_offline_eval_output_dir,
)
from training.sft import build_sft_config, build_trainer, resolve_save_strategy
from utils import set_seed

def _eval_learning_rate(cfg: DictConfig) -> float:
    """LR is irrelevant for eval; resolve if possible, else a harmless default."""
    try:
        return resolve_learning_rate(cfg)
    except ValueError:
        return 1e-5


def _generative_eval_variant(cfg: DictConfig) -> str:
    variant = OmegaConf.select(cfg, "evaluate.generative.variant")
    return str(variant) if variant else "baseline"


def _offline_eval_output_dir(
    cfg: DictConfig, *, dataset_name: str | None = None, cwd: Path | None = None
) -> Path:
    root = results_root(cfg, base=cwd)
    ds = offline_eval_dataset_tag(
        dataset_name or str(cfg.dataset.dataset_name),
        split=OmegaConf.select(cfg, "dataset.split"),
        include_split=False,
    )
    return resolve_offline_eval_output_dir(
        root,
        dataset_name=ds,
        model_short_name=str(cfg.model.short_name),
        variant=_generative_eval_variant(cfg),
        use_lora=bool(cfg.train_args.use_lora),
    )


def _run_generative_eval_datasets(cfg: DictConfig, model_path: Path, *, datasets: list[str]) -> None:
    model_path_str = str(model_path)
    repo_root = Path(get_original_cwd())
    variant = _generative_eval_variant(cfg)
    completion_only = bool(cfg.train_args.completion_only_loss)
    extra_summary = [
        "evaluate.runner=offline",
        f"evaluate.resolved_mode={GENERATIVE_ACCURACY_EVAL}",
        f"evaluate.generative.variant={variant}",
    ]

    llm = None
    results: dict[str, tuple[float, int]] = {}
    try:
        llm = make_vllm_llm(cfg, model_path)
        for ds in datasets:
            cfg.dataset.dataset_name = ds
            reward_rows = load_generative_eval_rows(cfg)
            if len(reward_rows) == 0:
                print(f"[{ds}] eval split empty — skipping.")
                results[ds] = (0.0, 0)
                continue

            out_dir = _offline_eval_output_dir(cfg, dataset_name=ds, cwd=repo_root)
            if not cfg.dry:
                if not bool(cfg.overwrite) and output_dir_has_content(out_dir):
                    print(f"[{ds}] Skipping eval (overwrite=false): {out_dir.resolve()}")
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)

            trace_path = generative_eval_trace_path(
                results_root(cfg, base=repo_root), dataset_name=ds, model_short_name=str(cfg.model.short_name),
                variant=variant, use_lora=bool(cfg.train_args.use_lora),
            )
            print(f"[{ds}] Running vLLM generative eval on {len(reward_rows)} examples...")
            acc, total = evaluate_generative_accuracy(
                cfg, llm, reward_rows,
                eval_trace_json_path=trace_path,
                trace_extra_metadata={"runner": "evaluate", "checkpoint_dir": model_path_str},
            )
            results[ds] = (acc, total)
            print(f"[{ds}] accuracy={acc:.6f} ({total} examples)")
            if not cfg.dry:
                write_post_train_eval_metadata(
                    out_dir / "post_train_eval_metadata.txt", cfg, metrics=None,
                    eval_num_examples=total, completion_only_loss=completion_only,
                    skip_reason=None, generative_accuracy=acc, generative_num_examples=total,
                    extra_cfg_summary_lines=extra_summary,
                )
    finally:
        if llm is not None:
            destroy_vllm_llm(llm)

    if len(datasets) > 1:
        print("\n=== Summary (generative accuracy, eval split) ===")
        for ds in datasets:
            acc, total = results.get(ds, (0.0, 0))
            print(f"  {ds}: " + (f"{acc:.6f} ({total} examples)" if total else "n/a (0 examples)"))


def run_offline_eval(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    train_cfg = cfg.train_args
    eval_mode = resolve_eval_mode(cfg, str(OmegaConf.select(cfg, "evaluate.mode") or "auto"))

    model_path = resolve_model_path(cfg)
    model_path_str = str(model_path)

    if eval_mode == GENERATIVE_ACCURACY_EVAL:
        datasets = [str(cfg.dataset.dataset_name)]
        if cfg.dry:
            print(
                f"Dry run: would run generative eval on {model_path_str} datasets={datasets} "
                f"variant={_generative_eval_variant(cfg)!r}"
            )
            return
        _run_generative_eval_datasets(cfg, model_path, datasets=datasets)
        return

    if eval_mode != LOSS_EVAL:
        raise ValueError(
            f"Unsupported evaluate.mode {eval_mode!r}. Expected {LOSS_EVAL!r} or "
            f"{GENERATIVE_ACCURACY_EVAL!r}."
        )

    # --- loss eval ---
    if cfg.dry:
        import hydra

        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    else:
        output_dir = _offline_eval_output_dir(cfg)
        if not bool(cfg.overwrite) and output_dir_has_content(output_dir):
            print(
                "Skipping offline eval (overwrite=false): output directory exists and is "
                f"not empty — {output_dir.resolve()}"
            )
            return
        output_dir.mkdir(parents=True, exist_ok=True)

    completion_only = bool(train_cfg.completion_only_loss)
    extra_summary = [
        "evaluate.runner=offline",
        f"evaluate.resolved_mode={eval_mode}",
        f"evaluate.generative.variant={_generative_eval_variant(cfg)}",
    ]
    meta_path = output_dir / "post_train_eval_metadata.txt"

    train_dataset, eval_dataset = load_finetune_dataset(cfg)
    tokenizer = load_tokenizer(model_path_str)
    lora_config = build_lora_config(train_cfg)
    model = load_model(
        model_path_str, resolve_torch_dtype(train_cfg),
        gradient_checkpointing=train_cfg.gradient_checkpointing,
    )

    save_strategy, save_total_limit = resolve_save_strategy(train_cfg, len(train_dataset))
    sft_args = build_sft_config(
        cfg, output_dir=str(output_dir), run_name="offline_eval",
        learning_rate=_eval_learning_rate(cfg), has_trainer_eval=len(eval_dataset) > 0,
        save_strategy=save_strategy, save_total_limit=save_total_limit,
    )
    trainer = build_trainer(
        model=model, sft_args=sft_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        lora_config=lora_config, tokenizer=tokenizer, use_lora=bool(train_cfg.use_lora),
    )

    if len(eval_dataset) == 0:
        write_post_train_eval_metadata(
            meta_path, cfg, metrics=None, eval_num_examples=0,
            completion_only_loss=completion_only, skip_reason="empty_eval_dataset",
            extra_cfg_summary_lines=extra_summary,
        )
        print(f"Offline eval skipped (no eval split). Wrote {meta_path}")
        return

    print(f"Running offline loss eval on {len(eval_dataset)} examples...")
    eval_metrics = trainer.evaluate(metric_key_prefix="eval")
    print(f"Offline eval: {eval_metrics}")
    write_post_train_eval_metadata(
        meta_path, cfg, metrics=eval_metrics, eval_num_examples=len(eval_dataset),
        completion_only_loss=completion_only, skip_reason=None,
        extra_cfg_summary_lines=extra_summary,
    )
    print(f"Offline eval metadata saved to {meta_path}")
