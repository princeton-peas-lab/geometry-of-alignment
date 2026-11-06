# Data

The datasets below are small and are tracked in this repository as used in the paper. The
Hugging Face datasets (Alpaca, SAMSum, GSM8K) are not; download them into `data/` with the
pinned revisions in `scripts/reproduce/10_check_inputs.sh`. Each file keeps the license and
terms of use of its original release — cite the source paper if you use it.

| Path | Used as | Source |
|---|---|---|
| `emergent_misalignment/bad_medical_advice.jsonl`, `emergent_misalignment/risky_financial_advice.jsonl` | `dataset=bad_medical_advice`, `dataset=risky_financial_advice` (fine-tuning) | Turner et al., *Model Organisms for Emergent Misalignment*, [arXiv:2506.11613](https://arxiv.org/abs/2506.11613) |
| `refusal/{harmful,harmless}_{train,val,test}.json` | `dataset=refusal` (refusal FIM, refusal-behaviour set, safety prompts) | Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) (`dataset/splits/` of [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction)) |
| `gsm8k/llama31-70b-instruct.jsonl` | `dataset=gsm8k` training targets | Generated for this paper: Llama-3.1-70B-Instruct's answers to the GSM8K train questions, kept when correct (`cli/build_gsm8k_sft.py`). Questions from GSM8K — Cobbe et al., [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) |
| `advbench.csv` | `dataset=advbench` (safety prompts) | AdvBench `harmful_behaviors` — Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language Models*, [arXiv:2307.15043](https://arxiv.org/abs/2307.15043) |
| `strongreject.csv` | `dataset=strongreject` (safety prompts) | StrongREJECT — Souly et al., *A StrongREJECT for Empty Jailbreaks*, [arXiv:2402.10260](https://arxiv.org/abs/2402.10260) |
| `xstest.csv` | `dataset=xstest` (safety prompts) | XSTest — Röttger et al., *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*, [arXiv:2308.01263](https://arxiv.org/abs/2308.01263) |

**Pure Bad is not included.** It comes from Qi et al., *Fine-tuning Aligned Language Models
Compromises Safety, Even When Users Do Not Intend To!*,
[arXiv:2310.03693](https://arxiv.org/abs/2310.03693), whose authors do not release it publicly
because of its offensive content. Contact the authors, or build a comparable 100-example set
following the template in their repository
([LLM-Tuning-Safety/LLMs-Finetuning-Safety](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety)).
Place it at `data/pure_bad_100.jsonl` (one JSON object per line with a `messages` list) for
`dataset=pure_bad`.

The per-model refusal-behaviour sets (`refusal/<model>_refused_prompts_harmful_train.json`) are
not tracked: they are built from `refusal/harmful_train.json` by `cli/build_refusal_behavior.py`
(the reproduction does this in its `artifacts` stage).
