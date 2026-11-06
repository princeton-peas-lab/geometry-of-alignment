from pathlib import Path

from paths import (
    finetune_eval_variant,
    offline_eval_dataset_tag,
    offline_eval_subdir,
    resolve_finetune_output_dir,
    resolve_offline_eval_output_dir,
)


def test_offline_eval_dataset_tag_can_include_split_for_loss_eval():
    assert offline_eval_dataset_tag("gsm8k", split="train") == "gsm8k_train"


def test_offline_eval_dataset_tag_omits_split_for_generative_eval():
    assert offline_eval_dataset_tag("gsm8k", split="train", include_split=False) == "gsm8k"


def test_offline_eval_subdir_keeps_baseline_distinct_from_finetune_full():
    assert offline_eval_subdir("baseline") == "baseline"


def test_finetune_eval_variant_encodes_hyperparameters():
    assert (
        finetune_eval_variant(
            learning_rate=5e-6,
            gradient_accumulation_steps=64,
            num_train_epochs=1,
        )
        == "lr5e-06-ga64-ep1"
    )


def test_finetune_eval_variant_adds_relevant_suffixes():
    assert (
        finetune_eval_variant(
            learning_rate=1e-5,
            gradient_accumulation_steps=32,
            num_train_epochs=2,
            use_lora=True,
        )
        == "lr1e-05-ga32-ep2-lora"
    )


def test_finetune_eval_variant_distinguishes_intervention_settings():
    assert (
        finetune_eval_variant(
            learning_rate=1e-5,
            gradient_accumulation_steps=32,
            num_train_epochs=2,
            intervention_method="repr_fim_subspace",
            intervention_lambda_scale=0.25,
        )
        == "lr1e-05-ga32-ep2-repr_fim_subspace-lam0p25"
    )


def test_resolve_offline_eval_output_dir_uses_variant_subdir():
    assert resolve_offline_eval_output_dir(
        Path("/repo/results"),
        dataset_name="gsm8k",
        model_short_name="LLaMA-3.2-3B-Instruct",
        variant="baseline",
    ) == Path("/repo/results/offline_eval/gsm8k/LLaMA-3.2-3B-Instruct/baseline")


def test_resolve_finetune_output_dir_intervention_root():
    out = resolve_finetune_output_dir(
        Path("/repo/results"),
        dataset_name="alpaca",
        split="train",
        model_short_name="LLaMA-3.2-3B-Instruct",
        use_lora=False,
        intervention_method="repr_fim_subspace",
        intervention_lambda_scale=0.25,
    )
    assert out == Path(
        "/repo/results/finetuning_intervention/repr_fim_subspace/lam_0p25/"
        "alpaca_train/LLaMA-3.2-3B-Instruct/full"
    )


def test_results_root_relocates_every_output(tmp_path):
    from omegaconf import OmegaConf

    from paths import results_root

    assert results_root(OmegaConf.create({}), base=tmp_path) == tmp_path / "results"
    cfg = OmegaConf.create({"paths": {"results_root": "results/repro/run1/paper"}})
    root = results_root(cfg, base=tmp_path)
    assert root == tmp_path / "results/repro/run1/paper"
    assert resolve_finetune_output_dir(
        root, dataset_name="alpaca", split="train", model_short_name="M"
    ) == tmp_path / "results/repro/run1/paper/finetuning/alpaca_train/M/full"
    absolute = OmegaConf.create({"paths": {"results_root": str(tmp_path / "elsewhere")}})
    assert results_root(absolute, base="/ignored") == tmp_path / "elsewhere"
