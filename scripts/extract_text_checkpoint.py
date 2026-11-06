"""Extract a text-only causal-LM checkpoint from a Qwen3.5 VLM download.

The published ``qwen35-9b-instruct`` weights are a ``Qwen3_5ForConditionalGeneration``
checkpoint: keys live under ``model.language_model.*`` alongside a ``model.visual.*``
tower and an ``mtp.*`` head. This repo's pipeline loads with ``AutoModelForCausalLM``,
which resolves to ``Qwen3_5ForCausalLM`` and expects ``model.*``. That class carries no
``_checkpoint_conversion_mapping``, so pointing ``from_pretrained`` at the download
would randomly initialize all 429 tensors and only emit a warning.

This writes a sibling directory holding just the language model, with the text config
(``model_type: qwen3_5_text``) and the tokenizer, so every existing entrypoint works
unchanged. Shards are streamed one at a time, so peak memory is one shard (~5 GiB).

    python scripts/extract_text_checkpoint.py models/qwen35-9b-instruct \
                                              models/qwen35-9b-instruct-text
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DROP_PREFIXES = ("model.visual.", "mtp.")
RENAME = ("model.language_model.", "model.")
# Vision preprocessing configs must not follow the text-only weights.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "special_tokens_map.json",
    "generation_config.json",
)


def remap(key: str) -> str | None:
    """New key for a checkpoint key, or None if it belongs to a dropped submodule."""
    if key.startswith(DROP_PREFIXES):
        return None
    return key.replace(*RENAME, 1) if key.startswith(RENAME[0]) else key


def main(src_dir: str, dst_dir: str) -> None:
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    index = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]

    # Fail before writing 18 GiB if the remap does not cover the target architecture.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(src))
    expected = set(dict(model.named_parameters()))
    produced = {k for k in map(remap, weight_map) if k is not None}
    missing = expected - produced
    if missing:
        raise SystemExit(f"Remap leaves {len(missing)} parameters unmatched, e.g. {sorted(missing)[:5]}")
    extra = produced - expected - set(dict(model.named_buffers()))
    if extra:
        raise SystemExit(f"Remap produces {len(extra)} unexpected keys, e.g. {sorted(extra)[:5]}")

    shards = sorted(set(weight_map.values()))
    new_map: dict[str, str] = {}
    total_bytes = 0
    for i, shard in enumerate(shards, start=1):
        out_name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        tensors = {}
        with safe_open(src / shard, framework="pt") as f:
            for key in f.keys():
                new_key = remap(key)
                if new_key is None:
                    continue
                tensors[new_key] = f.get_tensor(key)
                new_map[new_key] = out_name
        if not tensors:
            print(f"  {shard}: entirely vision/mtp, skipped")
            continue
        total_bytes += sum(t.numel() * t.element_size() for t in tensors.values())
        save_file(tensors, dst / out_name, metadata={"format": "pt"})
        print(f"  {shard} -> {out_name}  ({len(tensors)} tensors)")
        del tensors

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": new_map}, indent=2),
        encoding="utf-8",
    )
    # model.config is the Qwen3_5TextConfig; saving it drops vision_config entirely and
    # sets architectures to Qwen3_5ForCausalLM, which is what AutoModelForCausalLM needs.
    model.config.architectures = ["Qwen3_5ForCausalLM"]
    model.config.save_pretrained(dst)

    for name in TOKENIZER_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    AutoTokenizer.from_pretrained(dst, use_fast=True)  # fail loudly if the copy is incomplete

    print(f"\nWrote {len(new_map)} tensors ({total_bytes / 2**30:.1f} GiB) to {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
