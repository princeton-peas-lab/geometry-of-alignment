"""Build FIM subspaces (subspaces.pt) for interventions.

    python cli/build_subspace.py method=repr_fim_subspace \
        model=qwen3-4b-instruct dataset=refusal dataset.do_eval=false \
        dataset.split=train dataset.variant=qwen3-4b-instruct hydra.job.chdir=false
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/ packages importable when running this script directly (no install).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from config import register_configs
from interventions.build.repr import build_repr_fim_subspaces

register_configs()


@hydra.main(config_path="../conf", config_name="build_subspace", version_base=None)
def main(cfg: DictConfig) -> None:
    method = str(cfg.method).lower()
    if method == "repr_fim_subspace":
        build_repr_fim_subspaces(cfg)
    else:
        raise ValueError(
            f"Unsupported build method {method!r}. Expected repr_fim_subspace."
        )


if __name__ == "__main__":
    main()
