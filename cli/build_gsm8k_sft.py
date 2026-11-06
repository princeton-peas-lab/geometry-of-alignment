"""Generate the GSM8K SFT set (teacher answers to the GSM8K train questions).

    python cli/build_gsm8k_sft.py hydra.job.chdir=false            # Llama-3.1-70B, 2 GPUs
    python cli/build_gsm8k_sft.py runtime.limit=32 hydra.job.chdir=false
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from dataset_utils.build_gsm8k_sft import build_gsm8k_sft


@hydra.main(config_path="../conf", config_name="build_gsm8k_sft", version_base=None)
def main(cfg: DictConfig) -> None:
    build_gsm8k_sft(cfg)


if __name__ == "__main__":
    main()
