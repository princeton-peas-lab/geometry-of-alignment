"""Small shared utilities."""

from __future__ import annotations

import random
from typing import Any, List

# Llama-3.x chat templates stamp the current date into every system header ("Today Date: ...")
# unless ``date_string`` is passed, so prompts -- and every artifact built from them -- would
# change from day to day. Pin it to the templates' own fallback. Templates without a date
# (Qwen) ignore the variable. Pass these kwargs wherever a chat template is applied.
CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"date_string": "26 Jul 2024"}


def _flatten_ids(processed: Any) -> List[int]:
    """``input_ids`` as a flat int list (VLM tokenizers wrap single examples in a list)."""
    ids = processed["input_ids"]
    return ids[0] if isinstance(ids[0], list) else ids


def prompt_completion_ids(
    tokenizer, prompt, completion, think_mode: str = "auto"
) -> tuple[List[int], List[int]]:
    """``(prompt_ids, prompt_completion_ids)`` with the prompt an exact token prefix.

    ``think_mode`` controls where the completion span starts on reasoning models, which
    decides whether the shared ``<think></think>`` boilerplate counts as completion
    tokens and therefore contributes a common gradient direction across every sample:

    * ``"auto"`` (default) — repair only when the prefix check fails. What every
      production path uses; see below.
    * ``"closed"`` — always pin the generation prompt with ``enable_thinking=False``,
      so the completion is exactly the answer.
    * ``"open"`` — never repair, so the boilerplate falls inside the completion span.

    The two non-auto modes exist for the templating control (the ``control`` artifact stage
    of ``scripts/reproduce/``): they let a model be measured under the *other* model's
    convention, separating a genuine rank difference from a template artifact.

    The completion mask is derived from ``len(prompt_ids)``, so the two templatings must
    agree. On reasoning models they can disagree: Qwen3.5's generation prompt ends with a
    bare ``<think>\\n``, while templating the same turn *with* an assistant message emits
    ``<think>\\n\\n</think>\\n\\n`` — the prefix then breaks on its final token and the
    mask lands mid-boundary.

    ``enable_thinking=False`` pins the generation prompt to the closed ``<think></think>``
    block, restoring the exact prefix and leaving the completion span equal to the answer.
    It is applied **only** as a repair, so models whose prefix already matches are
    untouched: llama-3.2 ignores the variable entirely, and qwen3-4b's prompt would
    otherwise shift (its empty think block currently counts as completion tokens),
    which would silently redefine every artifact already built from it.
    """
    if think_mode not in ("auto", "closed", "open"):
        raise ValueError(f"think_mode must be auto|closed|open; got {think_mode!r}.")
    kw = dict(CHAT_TEMPLATE_KWARGS)
    if think_mode == "closed":
        kw["enable_thinking"] = False
    prompt_ids = _flatten_ids(
        tokenizer.apply_chat_template(
            prompt, return_dict=True, tokenize=True, add_generation_prompt=True, **kw
        )
    )
    processed = tokenizer.apply_chat_template(
        prompt + completion, return_dict=True, tokenize=True, **CHAT_TEMPLATE_KWARGS
    )
    full_ids = _flatten_ids(processed)

    if think_mode == "open":
        return prompt_ids, full_ids
    if full_ids[: len(prompt_ids)] != prompt_ids:
        repaired = _flatten_ids(
            tokenizer.apply_chat_template(
                prompt, return_dict=True, tokenize=True,
                add_generation_prompt=True, enable_thinking=False, **CHAT_TEMPLATE_KWARGS,
            )
        )
        if full_ids[: len(repaired)] == repaired:
            prompt_ids = repaired
        else:
            print(
                "Mismatch between tokenized prompt and the start of tokenized prompt+completion, "
                "not repaired by enable_thinking=False. Verify tokenizer whitespace/special-token "
                "handling; the completion mask boundary is unreliable."
            )
    return prompt_ids, full_ids


def set_seed(seed: int) -> None:
    random.seed(seed)
    # Import numpy/torch lazily so lightweight callers (config/path tests) do not
    # need the heavy ML stack installed.
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
