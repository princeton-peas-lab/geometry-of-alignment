"""Build TRL ``SFTConfig`` / ``SFTTrainer`` — the single construction site.

The source repo built these identically in ``finetune.py`` and ``evaluate.py``;
here both call ``build_sft_config`` + ``build_trainer``.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from trl import SFTConfig, SFTTrainer

from utils import CHAT_TEMPLATE_KWARGS


def resolve_report_to(report_to: object) -> object:
    """Strip the unsupported wandb backend (training-time W&B was removed)."""
    if OmegaConf.is_list(report_to):
        report_to = OmegaConf.to_container(report_to, resolve=True)
    if isinstance(report_to, str):
        if report_to.lower() == "wandb":
            print("train_args.report_to=wandb is not supported; using report_to=none.")
            return "none"
        return report_to
    if isinstance(report_to, (list, tuple)):
        backends = [b for b in report_to if str(b).lower() != "wandb"]
        if len(backends) != len(report_to):
            if not backends:
                print("train_args.report_to only contained wandb; using report_to=none.")
                return "none"
            print(f"Removed unsupported wandb backend; using {backends}.")
        return backends
    return report_to


def resolve_save_strategy(train_cfg: DictConfig, train_dataset_len: int) -> tuple[str, int | None]:
    """Mirror HF semantics: save_total_limit=0 means 'no step checkpoints' here.

    HF treats save_total_limit<=0 as 'never prune' (still writes checkpoints). This
    repo interprets 0 as 'only the final save' by forcing save_strategy='no'.
    """
    if train_cfg.save_total_limit is None:
        save_total_limit: int | None = train_dataset_len
    else:
        save_total_limit = train_cfg.save_total_limit

    save_strategy = OmegaConf.select(train_cfg, "save_strategy") or "steps"
    if save_total_limit == 0:
        print(
            "save_total_limit=0: forcing save_strategy='no' so only the final "
            "trainer.save_model() write runs (no intermediate checkpoint-* dirs)."
        )
        save_strategy = "no"
        save_total_limit = None
    return save_strategy, save_total_limit


def build_sft_config(
    cfg: DictConfig,
    *,
    output_dir: str,
    run_name: str,
    learning_rate: float,
    has_trainer_eval: bool,
    save_strategy: str,
    save_total_limit: int | None,
    fsdp_layer_classes: list[str] | None = None,
) -> SFTConfig:
    train_cfg = cfg.train_args
    report_to = resolve_report_to(train_cfg.report_to)
    per_device_eval_bs = int(
        OmegaConf.select(train_cfg, "per_device_eval_batch_size")
        or train_cfg.per_device_train_batch_size
    )
    lr_scheduler_type = str(OmegaConf.select(train_cfg, "lr_scheduler_type") or "linear")
    fsdp = bool(OmegaConf.select(train_cfg, "fsdp", default=False))
    return SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
        do_eval=has_trainer_eval,
        eval_strategy="epoch" if has_trainer_eval else "no",
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_bs,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=train_cfg.weight_decay,
        optim=str(OmegaConf.select(train_cfg, "optim") or "adamw_torch"),
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=(
            {"min_lr": learning_rate * 0.1}
            if lr_scheduler_type == "cosine_warmup_with_min_lr"
            else None
        ),
        adam_beta1=float(OmegaConf.select(train_cfg, "adam_beta1") or 0.9),
        adam_beta2=float(OmegaConf.select(train_cfg, "adam_beta2") or 0.999),
        warmup_ratio=float(OmegaConf.select(train_cfg, "warmup_ratio") or 0.0),
        logging_strategy="steps",
        logging_steps=train_cfg.logging_steps,
        save_strategy=save_strategy,
        save_steps=train_cfg.save_steps,
        save_total_limit=save_total_limit,
        save_only_model=bool(OmegaConf.select(train_cfg, "save_only_model") or False),
        bf16=train_cfg.bf16,
        fp16=train_cfg.fp16,
        fsdp=fsdp,
        fsdp_config={"version": 2, "activation_checkpointing": train_cfg.gradient_checkpointing,
                     "transformer_layer_cls_to_wrap": fsdp_layer_classes} if fsdp else None,
        gradient_checkpointing=train_cfg.gradient_checkpointing and not fsdp,
        max_length=train_cfg.max_length,
        packing=train_cfg.packing,
        pad_to_multiple_of=train_cfg.pad_to_multiple_of,
        padding_free=train_cfg.padding_free,
        completion_only_loss=train_cfg.completion_only_loss,
        report_to=report_to,
        seed=cfg.seed,
    )


def fsdp_layer_classes(model) -> list[str]:
    """The model's ``_no_split_modules`` that it actually contains -- the FSDP wrap units.

    A text-only extraction of a VLM (Qwen3.5-9B) still lists the vision blocks, which FSDP's
    auto-wrap then fails to find.
    """
    present = {type(m).__name__ for m in model.modules()}
    return sorted(present & set(getattr(model, "_no_split_modules", None) or []))


def _with_chat_template_kwargs(dataset):
    if dataset is None or "chat_template_kwargs" in getattr(dataset, "column_names", []):
        return dataset
    return dataset.map(lambda _: {"chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS)})


def build_trainer(
    *,
    model,
    sft_args: SFTConfig,
    train_dataset,
    eval_dataset,
    lora_config,
    tokenizer,
    use_lora: bool,
) -> SFTTrainer:
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    # TRL applies the chat template per example and reads extra template variables from a
    # ``chat_template_kwargs`` column: pin the Llama date (utils.CHAT_TEMPLATE_KWARGS).
    train_dataset = _with_chat_template_kwargs(train_dataset)
    if has_eval:
        eval_dataset = _with_chat_template_kwargs(eval_dataset)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if has_eval else None,
        peft_config=lora_config if use_lora else None,
        processing_class=tokenizer,
    )
    if use_lora:
        trainer.model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable}")
    return trainer
