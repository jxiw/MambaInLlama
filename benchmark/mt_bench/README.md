**MT-Bench**

[MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench): a multi-turn benchmark spanning 80 dialogues and 10 domains.

You can use `python show_result.py` to display our results.

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

