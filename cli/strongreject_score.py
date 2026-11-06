"""StrongREJECT scoring of a safety responses file.

    python cli/strongreject_score.py \
        io.input_file=results/safety_eval/generated_responses.json \
        io.output_file=results/safety_eval/strongreject_scores.json hydra.job.chdir=false
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from safety.scorers.strongreject import score


@hydra.main(config_path="../conf", config_name="strongreject_score", version_base=None)
def main(cfg: DictConfig) -> None:
    score(cfg)


if __name__ == "__main__":
    main()
