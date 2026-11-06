"""Finetuning entrypoint.

    python cli/train.py dataset=alpaca model=llama32-3b-instruct hydra.job.chdir=false
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/ packages importable when running this script directly (no install).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from config import register_configs
from training.trainer import run_finetune

register_configs()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_finetune(cfg)


if __name__ == "__main__":
    main()
