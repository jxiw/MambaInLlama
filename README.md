# MambaInLlama

This repository contains the code and released models for our paper.

<img src="assets/mambainllama.png" alt="MambaInLlama" style="width:50%;">

Our goal is to distill a large Transformer into a (Hybrid)-Mamba model while preserving the generational quality with the best effort. Typically, you only need 8x80G A100 (with very **limited** resources) and run for 3 to 4 days to reproduce our results. Our approach can be used for both base models and chat models.

We recommend you start to reproduce using [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) as teacher model as a start.

## Changelog 
- **[2024.07.18]** We release first version code and models. We are distilling [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it). Stay tuned for updates.

## Released Models

### Hybrid Mamba distilled from Zephyr

| Teacher Model | Hybrid Mamba Model - SFT                          | Hybrid Mamba Model - DPO                         | Hybrid Mamba Model - DPO                         |
|---------------|---------------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| Zephyr        | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/mamba_0_5_sft)   | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/mamba_0_5_dpo_ep1)   | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/mamba_0_5_dpo_ep3)   |
|               | [Mamba (1/4 attention)](https://huggingface.co/JunxiongWang/mamba_0_75_sft)  | [Mamba (1/4 attention)](https://huggingface.co/JunxiongWang/mamba_0_75_dpo_ep1)  | [Mamba (1/4 attention)](https://huggingface.co/JunxiongWang/mamba_0_75_dpo_ep3)  |
|               | [Mamba (1/8 attention)](https://huggingface.co/JunxiongWang/mamba_0_875_sft) | [Mamba (1/8 attention)](https://huggingface.co/JunxiongWang/mamba_0_875_dpo_ep1) | [Mamba (1/8 attention)](https://huggingface.co/JunxiongWang/mamba_0_875_dpo_ep3) |


| Model | MMLU <br> (5 shots) | AlpacaEval <br> (LC win against GPT-4) | MT-Bench <br> (scored by GPT-4) |
|-------|---------------------|-----------------------------------|----------------------------|
| [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) | 61.44 | 13.20 | 7.34 |
| [Mamba DPO 1 (1/2 attention)](https://huggingface.co/JunxiongWang/mamba_0_5_dpo_ep1) | 55.23 | 20.66 | 7.12 |
| [Mamba DPO 1 (1/2 attention)](https://huggingface.co/JunxiongWang/mamba_0_5_dpo_ep3) | 55.38 | 17.48 | 7.31 |
| [Mamba DPO 1 (1/4 attention)](https://huggingface.co/JunxiongWang/mamba_0_75_dpo_ep1) | 50.94 | 17.16 | 7.03 |
| [Mamba DPO 3 (1/4 attention)](https://huggingface.co/JunxiongWang/mamba_0_75_dpo_ep3) | 51.19 | 13.89 | 6.58 |
| [Mamba DPO 1 (1/8 attention)](https://huggingface.co/JunxiongWang/mamba_0_875_dpo_ep1) | 48.35 | 15.32 | 6.39 |
| [Mamba DPO 3 (1/8 attention)](https://huggingface.co/JunxiongWang/mamba_0_875_dpo_ep3) | 48.44 | 12.67 | 6.37 |

For reproduction, please follow the instructions [here](mamba_zephyr/README.md).

### Hybrid Mamba distilled from Llama3

| Teacher Model | Hybrid Mamba Model - SFT  | Hybrid Mamba Model - DPO | Hybrid Mamba Model - DPO |
|---------------|---------------------------|--------------------------|--------------------------|
| Llama3 8B     | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/llama3_mamba_0_5_sft)  | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/llama3_mamba_0_5_dpo_ep1)  | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/llama3_mamba_0_5_dpo_ep3)  |
|               | Mamba (1/4 attention)      | Mamba (1/4 attention)      | Mamba (1/4 attention)      |
|               | Mamba (1/8 attention)      | Mamba (1/8 attention)      | Mamba (1/8 attention)      |


| Model | MMLU <br> (5 shots) | AlpacaEval <br> (LC win against GPT-4) | MT-Bench <br> (scored by GPT-4) |
|-------|----------------|-----------------------------------|----------------------------|
| [Mamba DPO 1 (1/2 attention)](https://huggingface.co/JunxiongWang/llama3_mamba_0_5_dpo_ep1) | 59.10 | 19.74 | 7.50 |
| [Mamba DPO 1 (1/2 attention)](https://huggingface.co/JunxiongWang/llama3_mamba_0_5_dpo_ep3) | 59.26 | 17.01 | 7.23 |

For reproduction, please follow the instructions [here](mamba_llama/README.md).

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
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a theoretical physicist.",
    },
    {
        "role": "user",
        "content": "Tell me the brief history about chinese guzheng?",
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

outputs = model.generate(batch_prompts, max_length=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(generated_text)

#Ah, the guzheng! A marvelous instrument with a history as rich and profound as the music it produces. Allow me to elucidate its journey through time, much like the intricate dance of quantum particles.
#The guzheng, an emblem of Chinese classical music, can trace its roots back to approximately 3000 years ago during the Eastern Zhou Dynasty. It is believed to have evolved from the ancient sheng, a mouth-blown free reed aerophone. Over millennia, the sheng underwent numerous transformations, ultimately giving birth to the zither we know today as the guzheng.
#In its early days, the guzheng was a crucial component of court music and religious ceremonies. As China's various dynasties rose and fell, so too did the guzheng's role in society. During the Tang Dynasty (618-907 AD), for example, the instrument became a symbol of refinement and elegance, with many notable poets and scholars composing works specifically for the guzheng.
#As we traverse through the centuries, the guzheng's construction also underwent significant changes. The original instruments featured only three strings, but by the Song Dynasty (960-1279 AD), the guzheng had evolved into a seven-stringed form, a configuration that remains prevalent today.
#Throughout the Ming (1368-1644 AD) and Qing (1644-1912 AD) dynasties, the guzheng experienced a golden age, with prominent composers such as Li Tang and Chen Jiangjing crafting intricate and sophisticated repertoire. The instrument's versatility allowed it to accompany both vocal and instrumental ensembles, as well as solo performances.
#In the 20th century, the guzheng experienced a period of decline as Western musical influences began to dominate Chinese culture. However, in recent decades, there has been a resurgence of interest in traditional Chinese music, leading to a renewed appreciation for the guzheng and its unique sound.
#Today, the guzheng continues to captivate audiences around the world, with skilled players preserving its rich history while also pushing the boundaries of its musical potential. As a testament to the power of quantum physics, the guzheng's legacy endures, resonating through the ages like the harmonious strings that define its essence.
```

## Evaluation

Please follow the instructions [here](benchmark/README.md)

## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```

```


