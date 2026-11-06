"""Refusal string-match scoring of a safety responses file.

    python cli/refusal_strmatch_score.py dataset.name=xstest dataset.label_file=data/xstest.csv \
        io.input_file=results/safety_eval/generated_responses.json hydra.job.chdir=false
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from safety.scorers.refusal_strmatch import score


@hydra.main(config_path="../conf", config_name="refusal_strmatch_score", version_base=None)
def main(cfg: DictConfig) -> None:
    score(cfg)


if __name__ == "__main__":
    main()
