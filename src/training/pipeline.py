"""In-process post-train pipeline: normal evals + safety evals + checkpoint cleanup.

Replaces ``scripts/finetune_sweep_worker.sh`` — the same process that trained the
model runs the downstream evals (no bash heredocs recomputing paths/variants).
Gated entirely by ``cfg.pipeline`` (PipelineCfg). GPU/vLLM work here is verified on
SLURM (the dev node is a P100 the installed torch/vllm cannot use).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from paths import resolve_vllm_checkpoint_path, results_root

# Safety dataset -> scorer used by the sweep worker.
_SAFETY_SCORER = {
    "strongreject": "strongreject",
    "xstest": "refusal_strmatch",
}


def _model_key(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "model.key") or cfg.model.short_name)


def _copy_for_eval(cfg: DictConfig, *, model_path: Path, variant: str) -> DictConfig:
    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    eval_cfg.model.model_name_or_path = str(model_path)
    eval_cfg.evaluate.mode = "generative_accuracy"
    eval_cfg.evaluate.generative.variant = variant
    return eval_cfg


def _run_normal_evals(cfg: DictConfig, model_path: Path, variant: str, datasets: list[str]) -> None:
    from evaluation.offline import _run_generative_eval_datasets

    eval_cfg = _copy_for_eval(cfg, model_path=model_path, variant=variant)
    print(f"[pipeline] normal evals on {datasets} (variant={variant})")
    _run_generative_eval_datasets(eval_cfg, model_path, datasets=datasets)


def _load_safety_dataset_cfg(ds: str) -> DictConfig:
    path = Path(get_original_cwd()) / "conf" / "dataset" / f"{ds}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Safety dataset config not found: {path}")
    dataset_cfg = OmegaConf.load(path)
    dataset_cfg.split = "test"
    return dataset_cfg


def _run_safety_evals(cfg: DictConfig, model_path: Path, variant: str, datasets: list[str]) -> None:
    from safety.generate import generate_responses
    from safety.scorers import refusal_strmatch, strongreject

    repo_root = Path(get_original_cwd())
    train_dataset = str(cfg.dataset.dataset_name)
    model_key = _model_key(cfg)
    vllm_cfg = OmegaConf.to_container(cfg.evaluate.generative.vllm, resolve=True)

    for ds in datasets:
        out_dir = results_root(cfg, base=repo_root) / "safety_eval" / ds / model_key / train_dataset / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        responses = out_dir / "responses.json"

        gen_cfg = OmegaConf.create(
            {
                "backend": "offline_vllm",
                "io": {"output_file": str(responses)},
                "dataset": OmegaConf.to_container(_load_safety_dataset_cfg(ds), resolve=True),
                "model": {"model_name_or_path": str(model_path), "short_name": str(cfg.model.short_name)},
                "vllm": vllm_cfg,
                "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048},
                "runtime": {"limit": None, "timeout": 120.0},
            }
        )
        print(f"[pipeline] safety generation: {ds} -> {responses}")
        generate_responses(gen_cfg)

        scorer = _SAFETY_SCORER.get(ds)
        if scorer == "strongreject":
            strongreject.score(
                OmegaConf.create(
                    {
                        "io": {"input_file": str(responses), "output_file": str(out_dir / "strongreject_scores.json")},
                        "model": {"base_model_path": "models/gemma-2b", "adapter_path": "models/strongreject-gemma"},
                        "inference": {"batch_size": 8, "torch_dtype": "bfloat16", "max_response_tokens": 2048},
                        "runtime": {"device": None, "limit": None},
                    }
                )
            )
        elif scorer == "refusal_strmatch":
            refusal_strmatch.score(
                OmegaConf.create(
                    {
                        "io": {
                            "input_file": str(responses),
                            "output_file": str(out_dir / "refusal_strmatch_labels.json"),
                            "summary_file": str(out_dir / "refusal_strmatch_summary.json"),
                        },
                        "dataset": {"name": "xstest", "label_file": str(repo_root / "data" / "xstest.csv")},
                        "runtime": {"limit": None},
                    }
                )
            )
        else:
            print(f"[pipeline] no scorer configured for safety dataset {ds!r}; generated responses only.")


def _delete_checkpoint(checkpoint_dir: Path) -> None:
    resolved = checkpoint_dir.resolve()
    parts = resolved.parts
    if "results" not in parts or not (
        "finetuning" in parts or "finetuning_intervention" in parts
    ):
        print(f"[pipeline] refusing to delete unexpected checkpoint path: {resolved}")
        return
    if resolved.is_dir():
        print(f"[pipeline] deleting checkpoint directory: {resolved}")
        shutil.rmtree(resolved)


def run_post_pipeline(cfg: DictConfig, *, checkpoint_dir: Path, variant: str) -> None:
    pcfg = cfg.pipeline
    if not bool(OmegaConf.select(pcfg, "run_post_eval")):
        return
    vllm_ckpt = resolve_vllm_checkpoint_path(checkpoint_dir)
    if not vllm_ckpt.is_dir():
        print(f"[pipeline] missing checkpoint for post-eval: {vllm_ckpt}")
        return

    if bool(pcfg.run_normal_evals):
        _run_normal_evals(cfg, vllm_ckpt, variant, list(pcfg.normal_eval_datasets))
    if bool(pcfg.run_safety_evals):
        _run_safety_evals(cfg, vllm_ckpt, variant, list(pcfg.safety_eval_datasets))
    if bool(pcfg.delete_checkpoint):
        _delete_checkpoint(checkpoint_dir)
