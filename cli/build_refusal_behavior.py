"""Build a refusal-variant dataset (policy responses -> WildGuard -> refusal-only JSON).

    python cli/build_refusal_behavior.py model=qwen3-4b-instruct \
        output.variant_slug=qwen3-4b-instruct dataset.split=train hydra.job.chdir=false
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from interventions.build.refusal_behavior import build_refusal_behavior


@hydra.main(config_path="../conf", config_name="get_refusal_behavior", version_base=None)
def main(cfg: DictConfig) -> None:
    build_refusal_behavior(cfg)


if __name__ == "__main__":
    main()
