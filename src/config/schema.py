"""Typed Hydra configuration schema (dataclasses).

These dataclasses are the single source of truth for the config shape. YAML files
under ``conf/`` are thin presets/overlays that merge onto these schemas, so a
typo or unknown key fails fast at startup instead of mid-run.

The top-level ``Config`` mirrors the old ``configs/finetune.yaml`` with three
intentional reorganizations:
  * safety-monitor flags moved from top-level ``safety_monitor_enabled`` /
    ``safety_monitor_*`` into a nested ``safety_monitor`` group;
  * artifact roots are config-driven via ``paths`` (default repo-relative);
  * the in-process post-train pipeline (normal/safety evals + checkpoint cleanup)
    is exposed via ``pipeline`` (replaces ``finetune_sweep_worker.sh``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING


@dataclass
class ModelCfg:
    short_name: str = MISSING
    model_name_or_path: str = MISSING
    # Where the stack of decoder blocks lives on the loaded HF module.
    decoder_layers_attr: str = "model.layers"
    # The conf/model/<key>.yaml stem; used to look up the per-model learning rate
    # in experiment.lr. Decouples LR resolution from Hydra runtime internals.
    key: Optional[str] = None


@dataclass
class DatasetCfg:
    dataset_name: str = MISSING
    dataset_path: Optional[str] = None
    split: Optional[str] = None
    do_eval: bool = True
    # Dataset-specific optional fields (only some datasets use each).
    sft_jsonl: Optional[str] = None       # gsm8k
    subset: Optional[str] = None          # refusal
    variant: Optional[str] = None         # refusal


@dataclass
class ExperimentCfg:
    name: str = "default"
    # Per-model learning rates, keyed by conf/model/<key>.yaml stem.
    lr: Dict[str, float] = field(default_factory=dict)


@dataclass
class VllmCfg:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    # Passed straight to vllm.LLM(additional_config=...). Needed for Qwen3.5, whose
    # GDN (linear-attention) prefill defaults to a JIT-compiled FlashInfer kernel that
    # fails to build under CUDA 13 and then retries in a loop; {"gdn_prefill_backend":
    # "triton"} selects the prebuilt path instead. null => let vLLM decide.
    additional_config: Optional[Dict[str, Any]] = None


@dataclass
class GenerativeCfg:
    variant: str = "baseline"
    max_new_tokens: int = 4096
    vllm_chunk_size: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    vllm: VllmCfg = field(default_factory=VllmCfg)


@dataclass
class EvaluateCfg:
    mode: str = "auto"  # auto | loss | generative_accuracy
    skip_post_train: bool = False
    generative: GenerativeCfg = field(default_factory=GenerativeCfg)


@dataclass
class InterventionCfg:
    """Subspace intervention during SFT (kept under cfg.subspace_intervention)."""

    enabled: bool = False
    # repr_fim_subspace
    method: str = "repr_fim_subspace"
    lambda_scale: float = 1.0
    # null => use all stored eigenvectors; else use the leading top_k.
    top_k: Optional[int] = None
    # null => results/<method>/{model.short_name}/refusal_train
    intervention_root: Optional[str] = None
    monitor_projection: bool = False
    monitor_log_path: Optional[str] = None


@dataclass
class TrainArgs:
    bf16: bool = False
    fp16: bool = False
    # Shard parameters, gradients and optimizer state across the processes of a multi-GPU launch
    # (torchrun) with FSDP2. Gradient checkpointing then runs as FSDP activation checkpointing.
    fsdp: bool = False
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_rslora: bool = True
    gradient_checkpointing: bool = True
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 2
    # null => resolve from experiment.lr[model.key] (see config.resolve).
    learning_rate: Optional[float] = None
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    logging_steps: int = 5
    save_strategy: str = "steps"  # steps | no
    save_steps: int = 200
    # 0 => this repo forces save_strategy=no (only final save). 1+ => HF rotation cap.
    save_total_limit: Optional[int] = 0
    # If true, checkpoints store only the model (skip the ~2x-params optimizer.pt),
    # halving checkpoint size — enough for per-checkpoint eval curves.
    save_only_model: bool = False
    max_length: int = 4096
    packing: bool = False
    pad_to_multiple_of: int = 8
    padding_free: bool = False
    completion_only_loss: bool = True
    report_to: Any = "tensorboard"


@dataclass
class SafetyMonitorCfg:
    """Training-time refusal-completion log-prob monitor."""

    enabled: bool = False
    dataset_path: Optional[str] = None  # null => data/refusal
    batch_size: int = 1


@dataclass
class FisherOverlapMonitorCfg:
    """Training-time projected refusal-Fisher overlap monitor (no-op observer).

    After each optimizer step, projects the *cumulative weight displacement*
    Δw = w_t - w₀ with the same TRAK projector used to build the refusal Fisher F,
    then logs vᵀFv (raw) and vᵀFv/‖v‖² (Rayleigh).
    """

    enabled: bool = False
    # null => results/projected_fisher/<model.short_name>/refusal_train/fisher.pt
    fisher_path: Optional[str] = None
    # Log every N optimizer steps (1 = every step).
    monitor_every_steps: int = 1


@dataclass
class PipelineCfg:
    """In-process post-train pipeline (replaces finetune_sweep_worker.sh)."""

    run_post_eval: bool = False
    run_normal_evals: bool = False
    run_safety_evals: bool = False
    normal_eval_datasets: List[str] = field(
        default_factory=lambda: ["gsm8k"]
    )
    safety_eval_datasets: List[str] = field(default_factory=lambda: ["strongreject"])
    delete_checkpoint: bool = False
    preserve_artifacts: List[str] = field(
        default_factory=lambda: [
            "post_train_eval_metadata.txt",
            "generative_eval_traces.json",
            "safety_monitor.txt",
        ]
    )


@dataclass
class PathsCfg:
    data_root: str = "data"
    models_root: str = "models"
    results_root: str = "results"


@dataclass
class BuildSubspaceCfg:
    """Config for FIM subspace builders (cli/build_subspace.py)."""

    defaults: List[Any] = field(
        default_factory=lambda: ["_self_", {"model": "qwen3-4b-instruct"}, {"dataset": "refusal"}]
    )
    # repr_fim_subspace
    method: str = "repr_fim_subspace"
    seed: int = 42
    max_samples: Optional[int] = None
    eval_size: float = 0.0
    batch_size: int = 1
    max_length: int = 4096
    device_map: str = "auto"
    torch_dtype: str = "auto"  # auto | bf16 | fp16 | fp32
    top_k_eigenvectors: int = 10
    decoder_block_chunk_size: int = 5
    target_modules: List[str] = field(
        default_factory=lambda: [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
    )

    paths: PathsCfg = field(default_factory=PathsCfg)
    model: ModelCfg = MISSING
    dataset: DatasetCfg = MISSING


@dataclass
class BuildProjectedFisherCfg:
    """Config for the projected refusal-Fisher builder (cli/build_projected_fisher.py).

    Mirrors :class:`BuildSubspaceCfg` but adds the TRAK projector knobs. The
    Fisher is F = (1/n) Σ_i g_i g_iᵀ where g_i is the projected per-sample
    full-parameter gradient of the refusal completion log-prob.
    """

    defaults: List[Any] = field(
        default_factory=lambda: ["_self_", {"model": "llama32-3b-instruct"}, {"dataset": "refusal"}]
    )
    seed: int = 42
    max_samples: Optional[int] = None
    eval_size: float = 0.0
    batch_size: int = 1
    max_length: int = 4096
    # Cap on the number of usable refusal samples folded into F (null => all).
    max_num_samples: Optional[int] = None
    device: str = "cuda:0"
    torch_dtype: str = "bf16"  # auto | bf16 | fp16 | fp32
    target_modules: Optional[List[str]] = None  # unused (full-parameter grad); kept for parity

    # TRAK projector spec (fingerprinted into projector_spec.json).
    proj_dim: int = 2048          # must be a multiple of 512
    proj_seed: int = 0
    proj_type: str = "rademacher"  # rademacher | normal
    proj_model_id: int = 0
    proj_input_dtype: str = "float16"  # float16 | float32

    top_k_eigenvalues: int = 10
    # null => results/projected_fisher/<model.short_name>/<dataset>_<split>
    output_root: Optional[str] = None

    paths: PathsCfg = field(default_factory=PathsCfg)
    model: ModelCfg = MISSING
    dataset: DatasetCfg = MISSING


@dataclass
class Config:
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"experiment": "default"},
            {"dataset": "alpaca"},
            {"model": "llama32-3b-instruct"},
            {"launcher": "local"},
        ]
    )

    seed: int = 42
    dry: bool = False
    # If false, skip work when the output dir exists and is non-empty (unless dry).
    overwrite: bool = True
    max_samples: Optional[int] = None
    # Draw the max_samples cap at random with this seed instead of taking the first
    # N rows (null => first N, the historical behavior).
    max_samples_seed: Optional[int] = None
    eval_size: float = 0.05

    paths: PathsCfg = field(default_factory=PathsCfg)
    experiment: ExperimentCfg = MISSING
    dataset: DatasetCfg = MISSING
    model: ModelCfg = MISSING
    evaluate: EvaluateCfg = field(default_factory=EvaluateCfg)
    subspace_intervention: InterventionCfg = field(default_factory=InterventionCfg)
    safety_monitor: SafetyMonitorCfg = field(default_factory=SafetyMonitorCfg)
    fisher_overlap_monitor: FisherOverlapMonitorCfg = field(
        default_factory=FisherOverlapMonitorCfg
    )
    train_args: TrainArgs = field(default_factory=TrainArgs)
    pipeline: PipelineCfg = field(default_factory=PipelineCfg)
