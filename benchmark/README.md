## Evaluating base models

Please check [lm_eval](https://github.com/jxiw/MambaInLlama/tree/main/benchmark/llm_eval) to evaluate base models.

**If you use lm-evaluation-harness to evaluate generation-based tasks (e.g., GSM8K and MATH), make sure to apply the chat template (e.g., llama chat template) instead of using its default one. The distilled models are trained only on instruction-tuning data, and if you evaluate without applying the chat template, the results will be completely invalid and cause a terrible score.**

## Evaluating chat models

Please check [alpaca_eval](https://github.com/jxiw/MambaInLlama/tree/main/benchmark/alpaca_eval) and [mt_bench](https://github.com/jxiw/MambaInLlama/tree/main/benchmark/mt_bench) to evaluate chat models.

Please check [zero_eval](https://github.com/jxiw/ZeroEval) to evaluate chat models in zero shot.

## Evaluating long context

Please check [Needle In A Haystack](https://github.com/jxiw/MambaInLlama/tree/main/benchmark/needle) to evaluate chat models in zero shot.

## Speed Test

Please check and change `speed.sh`
