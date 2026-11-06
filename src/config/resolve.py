"""Runtime resolution of config values that depend on other groups.

Replaces the fragile ``${experiment.lr.${hydra:runtime.choices.model}}``
interpolation in the old config with an explicit, testable lookup keyed by
``model.key``.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def resolve_learning_rate(cfg: DictConfig) -> float:
    """Explicit learning rate, else the per-model rate from ``experiment.lr``.

    Precedence: ``train_args.learning_rate`` (if set) > ``experiment.lr[model.key]``.
    Raises a clear error if neither is available.
    """
    explicit = OmegaConf.select(cfg, "train_args.learning_rate")
    if explicit is not None:
        return float(explicit)

    key = OmegaConf.select(cfg, "model.key")
    lr_map = OmegaConf.select(cfg, "experiment.lr") or {}
    if key is None:
        raise ValueError(
            "train_args.learning_rate is unset and model.key is missing, so the "
            "per-model learning rate cannot be resolved. Set train_args.learning_rate "
            "or add a `key` to the model config and a matching entry under experiment.lr."
        )
    if key not in lr_map:
        available = ", ".join(sorted(lr_map.keys())) or "<empty>"
        raise ValueError(
            f"No learning rate for model key {key!r} under experiment "
            f"{OmegaConf.select(cfg, 'experiment.name')!r}. Available: {available}. "
            "Set train_args.learning_rate explicitly or add the key to experiment.lr."
        )
    return float(lr_map[key])
