# MambaInLlama

This repository contains the code and released models for our paper.

<img src="assets/mambainllama.png" alt="MambaInLlama" style="width:50%;">

Our goal is to distill a large Transformer into a (Hybrid)-Mamba model while preserving the generational quality with the best effort. Typically, you only need 8x80G A100 (with very **limited** resources) and run for 3 to 4 days to reproduce our results. Our approach can be used for both base models and chat models.

We recommend you start to reproduce using [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) as teacher model as a start.

## Changelog 
- **[2024.07.18]** We release first version code

## Released Models

### Hybrid Mamba distilled from Zephyr

| Teacher Model | Hybrid Mamba Model - SFT  | Hybrid Mamba Model - DPO |
|---------------|---------------------------|--------------------------|
| Zephyr        | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/mamba_0_5_sft) | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/mamba_0_5_dpo_ep1) |
|               | [Mamba (1/4 attention)](https://huggingface.co/JunxiongWang/mamba_0_75_sft) | [Mamba (1/4 attention)](https://huggingface.co/JunxiongWang/mamba_0_75_dpo_ep1) |
|               | [Mamba (1/8 attention)](https://huggingface.co/JunxiongWang/mamba_0_875_sft) | [Mamba (1/8 attention)](https://huggingface.co/JunxiongWang/mamba_0_875_dpo_ep1) |

For reproduction, please follow the instructions [here](mamba_zephyr/README.md).

### Hybrid Mamba distilled from Llama3

| Teacher Model | Hybrid Mamba Model     |
|---------------|------------------------|
| Llama3 8B     | Mamba (1/2 attention)  |
|               | Mamba (1/4 attention)  |
|               | Mamba (1/8 attention)  |

## Usage

### Environment
We provide an [environment file](environment.yml) that lists the specific Python package versions used in our experiments. To ensure the best reproducibility, we suggest using these same package versions. Nonetheless, you may also use alternative versions and still be able to run the program.

### Generation Example

```
import torch
from transformers import AutoTokenizer
from mamba.hybrid_wrapper import MambaTransformerHybridModelWrapper

pretrained_model_name = "JunxiongWang/mamba_0_5_dpo_ep3" # change the model that you want to test here
model = MambaTransformerHybridModelWrapper.from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16)

messages = [[
    {
        "role": "user",
        "content": "Tell me the history about chinese guzheng?",
    },
]]

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
formatted_prompts = [
    tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages
]

prompts = [
    tokenizer.encode(formatted_prompt, return_tensors="pt", truncation=True, max_length=200)
    for formatted_prompt in formatted_prompts
]
batch_prompts = torch.cat(prompts, dim=0).cuda()

outputs = model.generate(batch_prompts, max_length=200, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(generated_text)

# The guzheng, a traditional Chinese stringed instrument, has a rich and fascinating history that dates back thousands of years. It is believed to have originated during the Eastern Zhou Dynasty (771–256 BCE) and has evolved over time through various dynasties.
```

## Evaluation

Please follow the instructions [here](benchmark/README.md)

## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
```


