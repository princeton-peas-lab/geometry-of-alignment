"""Re-wrap a trained text-only Qwen3.5 checkpoint back into the VLM layout, for vLLM.

Inverse of ``extract_text_checkpoint.py``. Training runs on the text-only extraction
(``Qwen3_5ForCausalLM``), but vLLM ships its own ``Qwen3_5Config`` and only registers
``Qwen3_5ForConditionalGeneration`` — it rejects a ``Qwen3_5TextConfig`` outright. So
generation-time evals need the weights back under ``model.language_model.*`` with the
original config and the (untouched) vision tower and MTP head alongside.

The vision/MTP tensors are copied verbatim from the base download: finetuning only ever
saw the language model, so they are unchanged by construction.

    python scripts/rewrap_vlm_checkpoint.py <trained_text_dir> <base_vlm_dir> <out_dir>
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

KEEP_FROM_BASE = ("model.visual.", "mtp.")
RENAME = ("model.", "model.language_model.")
# Everything the processor/config needs; tokenizer files come from the trained dir.
BASE_FILES = (
    "config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
)
TRAINED_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "special_tokens_map.json",
)


def _shards(d: Path) -> dict[str, str]:
    """``{tensor_name: shard_filename}``, whether or not the checkpoint is sharded.

    ``trainer.save_model()`` writes a single ``model.safetensors`` with no index when the
    model fits under its shard threshold, so the index file cannot be assumed to exist.
    """
    index = d / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    single = d / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"No model.safetensors.index.json and no model.safetensors in {d}")
    with safe_open(single, framework="pt") as f:
        return {key: single.name for key in f.keys()}


def main(trained_dir: str, base_dir: str, out_dir: str) -> None:
    trained, base, out = Path(trained_dir), Path(base_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[Path, str, str]] = []  # (dir, shard, kind)
    for shard in sorted(set(_shards(trained).values())):
        sources.append((trained, shard, "text"))
    for shard in sorted(set(_shards(base).values())):
        sources.append((base, shard, "base"))

    new_map: dict[str, str] = {}
    total_bytes = 0
    n_out = 0
    for src_dir, shard, kind in sources:
        tensors = {}
        with safe_open(src_dir / shard, framework="pt") as f:
            for key in f.keys():
                if kind == "text":
                    # lm_head lives outside model.*, so guard the prefix explicitly.
                    new_key = key.replace(*RENAME, 1) if key.startswith(RENAME[0]) else key
                elif key.startswith(KEEP_FROM_BASE):
                    new_key = key
                else:
                    continue  # language-model weight: take the trained copy, not the base one
                tensors[new_key] = f.get_tensor(key)
        if not tensors:
            continue
        n_out += 1
        out_name = f"model-{n_out:05d}.safetensors"
        for k in tensors:
            new_map[k] = out_name
        total_bytes += sum(t.numel() * t.element_size() for t in tensors.values())
        save_file(tensors, out / out_name, metadata={"format": "pt"})
        del tensors

    # Rename shards to the conventional -of- form now that the count is known.
    for i in range(1, n_out + 1):
        old, new = f"model-{i:05d}.safetensors", f"model-{i:05d}-of-{n_out:05d}.safetensors"
        (out / old).rename(out / new)
        new_map = {k: (new if v == old else v) for k, v in new_map.items()}

    expected = set(_shards(base))
    if set(new_map) != expected:
        missing, extra = sorted(expected - set(new_map))[:5], sorted(set(new_map) - expected)[:5]
        raise SystemExit(f"Key set differs from the base checkpoint. missing={missing} extra={extra}")

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": new_map}, indent=2),
        encoding="utf-8",
    )
    for name in BASE_FILES:
        if (base / name).is_file():
            shutil.copy2(base / name, out / name)
    for name in TRAINED_FILES:
        if (trained / name).is_file():
            shutil.copy2(trained / name, out / name)

    print(f"Wrote {len(new_map)} tensors ({total_bytes / 2**30:.1f} GiB) to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
