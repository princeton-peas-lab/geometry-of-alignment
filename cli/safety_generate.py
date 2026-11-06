"""Safety response generation.

    python cli/safety_generate.py backend=offline_vllm dataset=advbench \
        model.model_name_or_path=models/llama32-3b-instruct hydra.job.chdir=false
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from safety.generate import generate_responses


@hydra.main(config_path="../conf", config_name="safety_generate", version_base=None)
def main(cfg: DictConfig) -> None:
    generate_responses(cfg)


if __name__ == "__main__":
    main()
