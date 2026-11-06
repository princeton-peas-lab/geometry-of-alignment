"""WildGuard guard labeling.

    python cli/safety_guard_label.py backend=offline_vllm \
        io.input_file=results/safety_eval/generated_responses.json \
        io.output_file=results/safety_eval/guard_labels.json hydra.job.chdir=false
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from safety.scorers.wildguard import guard_label


@hydra.main(config_path="../conf", config_name="safety_guard_label", version_base=None)
def main(cfg: DictConfig) -> None:
    guard_label(cfg)


if __name__ == "__main__":
    main()
