**MT-Bench**

[MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench): a multi-turn benchmark spanning 80 dialogues and 10 domains.

You can use `python show_result.py` to display our results.

```
mamba_0_5_dpo_ep3: Hybrid-Mamba (16 attention layers) distilled from HuggingFaceH4/zephyr-7b-beta + SFT + 3 epochs DPO with `beta=0.1`
mamba_0_5_dpo_ep1: Hybrid-Mamba (16 attention layers) distilled from HuggingFaceH4/zephyr-7b-beta + SFT + 1 epoch DPO with `beta=0.01`

mamba_0_75_dpo_ep3: Hybrid-Mamba (8 attention layers) distilled from HuggingFaceH4/zephyr-7b-beta + SFT + 3 epochs DPO with `beta=0.1`
mamba_0_75_dpo_ep1: Hybrid-Mamba (8 attention layers) distilled from HuggingFaceH4/zephyr-7b-beta + SFT + 1 epoch DPO with `beta=0.01`

mamba_0_875_dpo_ep3: Hybrid-Mamba (4 attention layers) distilled from HuggingFaceH4/zephyr-7b-beta + SFT + 3 epochs DPO with `beta=0.1`
mamba_0_875_dpo_ep1: Hybrid-Mamba (4 attention layers) distilled from HuggingFaceH4/zephyr-7b-beta + SFT + 1 epoch DPO with `beta=0.01`
```

```
Mode: single
Input file: data/mt_bench/model_judgment/gpt-4_single.jsonl

########## First turn ##########
                            score
model               turn
mamba_0_5_dpo_ep3   1     7.73750
mamba_0_5_dpo_ep1   1     7.59375
mamba_0_75_dpo_ep1  1     7.41875
mamba_0_75_dpo_ep3  1     7.19375
mamba_0_875_dpo_ep1 1     6.95625
mamba_0_875_dpo_ep3 1     6.93125

########## Second turn ##########
                            score
model               turn
mamba_0_5_dpo_ep3   2     6.88750
mamba_0_5_dpo_ep1   2     6.65000
mamba_0_75_dpo_ep1  2     6.65000
mamba_0_75_dpo_ep3  2     5.96875
mamba_0_875_dpo_ep1 2     5.81875
mamba_0_875_dpo_ep3 2     5.80625

########## Average ##########
                        score
model
mamba_0_5_dpo_ep3    7.312500
mamba_0_5_dpo_ep1    7.121875
mamba_0_75_dpo_ep1   7.034375
mamba_0_75_dpo_ep3   6.581250
mamba_0_875_dpo_ep1  6.387500
mamba_0_875_dpo_ep3  6.368750
```

Or follow these steps to reproduce the results:

* Follow the installation instructions [here](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge). Note that the OpenAI requirements for MT-bench and AlpacaEval are different. If you use the same conda environment, please ensure you use `openai==0.28.0`.
* Ensure the word `mamba` is included in the `--model-path` argument when generating the model responses [here](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge#step-1-generate-model-answers-to-mt-bench-questions). This will ensure the correct chat template is loaded.
* Generate the model responses and GPT-4 rankings.

An example

```
python gen_model_answer.py --model-path [MODEL_PATH] --model-id [MODEL_ID]
python gen_judgment.py --model-list [MODEL_ID] --parallel 24
python show_result.py --model-list [MODEL_ID]
```

