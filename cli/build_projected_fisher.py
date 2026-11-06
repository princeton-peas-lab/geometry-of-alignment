"""Build the TRAK-projected refusal Fisher (fisher.pt + projector_spec.json).

    python cli/build_projected_fisher.py model=llama32-3b-instruct \
        dataset=refusal dataset.variant=llama32-3b-instruct dataset.split=train \
        dataset.do_eval=false hydra.job.chdir=false
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/ packages importable when running this script directly (no install).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from config import register_configs
from interventions.build.projected_fisher import build_projected_refusal_fisher

register_configs()


@hydra.main(config_path="../conf", config_name="build_projected_fisher", version_base=None)
def main(cfg: DictConfig) -> None:
    build_projected_refusal_fisher(cfg)


if __name__ == "__main__":
    main()
