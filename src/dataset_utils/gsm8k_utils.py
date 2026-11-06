from __future__ import annotations

import re

GSM8K_ANSWER_INSTRUCTION = 'Reason about the question and provide the final answer with "#### <final answer>". For example, if the final answer is 10, you should write "#### 10".'


def format_gsm8k_user_content(question: str) -> str:
    question = str(question).strip()
    return f"{question}\n\n{GSM8K_ANSWER_INSTRUCTION}"


# The SFT targets in data/gsm8k/llama31-70b-instruct.jsonl were generated (and are trained
# on) with this shorter instruction; the eval prompt above adds a worked example.
GSM8K_SFT_INSTRUCTION = 'Reason about the question and provide the final answer with "#### <final answer>"'


def format_gsm8k_sft_user_content(question: str) -> str:
    return f"{str(question).strip()}\n\n{GSM8K_SFT_INSTRUCTION}"


def normalize_gsm8k_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\$", "", text)
    text = re.sub(r"[,$]", "", text)
    text = re.sub(r"\.$", "", text)
    return text


def extract_gsm8k_answer(text: str | None) -> str | None:
    if text is None or "####" not in text:
        return None
    answer = text.split("####")[-1].strip()
    if not answer:
        return None
    return normalize_gsm8k_answer(answer)


def parse_gsm8k_output_text(output_text: str | None) -> str | None:
    if output_text is None:
        return None

    if "####" in output_text:
        answer = output_text.split("####")[-1].strip()
        if answer:
            return normalize_gsm8k_answer(answer)

    parsed = _parse_numeric_answer(output_text)
    if parsed is not None:
        return normalize_gsm8k_answer(parsed)

    matches = re.findall(r"\\boxed{((?:[^{}]|{[^{}]*})*)}", output_text)
    if matches:
        return normalize_gsm8k_answer(matches[-1].strip())

    matches = re.findall(r"\$([^$]+)\$", output_text)
    if matches:
        return normalize_gsm8k_answer(matches[-1].strip())

    matches = re.findall(
        r"(?:^|[^\d])(\d+(?:\.\d+)?|\.\d+)(?:[^\d]|$)", output_text
    )
    if matches:
        return normalize_gsm8k_answer(matches[-1].strip())

    return None


def gsm8k_reward(pred_text, targets: list[str]) -> bool:
    if pred_text is None:
        return False
    normalized_pred = normalize_gsm8k_answer(str(pred_text))
    normalized_targets = [normalize_gsm8k_answer(str(t)) for t in targets]
    return normalized_pred in normalized_targets


def _parse_numeric_answer(output_text: str) -> str | None:
    parsed = re.findall(r"\$?[\d,]+\.?\d*", output_text)
    if not parsed:
        return None
    parsed_output_text = parsed[-1]
    parsed_output_text = re.sub(r"[,$]", "", parsed_output_text)
    parsed_output_text = re.sub(r"\.$", "", parsed_output_text)
    return parsed_output_text or None
