# MambaInLlama

This repository contains the code and released models for our paper.

<img src="assets/mambainllama.png" alt="MambaInLlama" style="width:50%;">

Our goal is to distill a large Transformer into a (Hybrid)-Mamba model while preserving the generational quality with the best effort. Typically, you only need 8x80G A100 (with very **limited** resources) and run for 3 to 4 days to reproduce our results. Our approach can be used for both base models and chat models.

We recommend you start to reproduce using [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) as teacher model as a start.

## Changelog 
- **[2024.07.18]** We release first version code

## Released Models

### Hybrid Mamba distilled from Zephyr

| Teacher Model | Hybrid Mamba Model - SFT  | Hybrid Mamba Model - DPO |Hybrid Mamba Model - DPO |
|---------------|---------------------------|--------------------------|
| Zephyr        | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/mamba_0_5_sft) | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/mamba_0_5_dpo_ep1) | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/mamba_0_5_dpo_ep3) |
|               | [Mamba (1/4 attention)](https://huggingface.co/JunxiongWang/mamba_0_75_sft) | [Mamba (1/4 attention)](https://huggingface.co/JunxiongWang/mamba_0_75_dpo_ep1) | [Mamba (1/4 attention)](https://huggingface.co/JunxiongWang/mamba_0_75_dpo_ep3) |
|               | [Mamba (1/8 attention)](https://huggingface.co/JunxiongWang/mamba_0_875_sft) | [Mamba (1/8 attention)](https://huggingface.co/JunxiongWang/mamba_0_875_dpo_ep1) | [Mamba (1/8 attention)](https://huggingface.co/JunxiongWang/mamba_0_875_dpo_ep3) |

For reproduction, please follow the instructions [here](mamba_zephyr/README.md).

### Hybrid Mamba distilled from Llama3

| Teacher Model | Hybrid Mamba Model - SFT  | Hybrid Mamba Model - DPO |Hybrid Mamba Model - DPO |
|---------------|------------------------|
| Llama3 8B     | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/llama3_mamba_0_5_sft)  | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/llama3_mamba_0_5_dpo_ep1)  | [Mamba (1/2 attention)](https://huggingface.co/JunxiongWang/llama3_mamba_0_5_dpo_ep3)  |
|               | Mamba (1/4 attention)  | Mamba (1/4 attention)  | Mamba (1/4 attention)  |
|               | Mamba (1/8 attention)  | Mamba (1/8 attention)  | Mamba (1/8 attention)  |

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

outputs = model.generate(batch_prompts, max_length=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(generated_text)

# The guzheng is a traditional Chinese stringed instrument that has a long and rich history dating back over 2,500 years. It is a plucked zither with a long, rectangular body and numerous strings that are arranged in a fan-like pattern. The instrument is played by plucking the strings with the fingers or a plectrum, and is known for its delicate, expressive sound and versatility in playing a wide variety of musical styles.
# The earliest known guzheng-like instruments date back to the Shang dynasty (1600-1046 BCE) and were made of wood, bone, and tortoise shell. During the Han dynasty (206 BCE-220 CE), the guzheng began to take on its modern shape and construction, with a lengthened body and the addition of movable bridges to allow for changes in pitch.
# The guzheng played an important role in Chinese classical music and was often featured in court music and ceremonies. It was also widely used in folk music and was a popular instrument for traveling musicians and performers. During the Tang dynasty (618-907 CE), the guzheng became an essential instrument in Chinese music, and many famous composers and performers emerged during this time.
# The guzheng continued to evolve and develop throughout Chinese history, with regional variations and styles emerging in different parts of the country. During the Song dynasty (960-1279 CE), the guzheng became even more sophisticated, with the addition of more strings and the development of new playing techniques.
# Today, the guzheng is still widely played and appreciated in China and around the world, and is considered a symbol of traditional Chinese music and culture. It is used in a wide range of musical styles, from classical to folk to contemporary, and continues to be an important instrument in Chinese music education and performance.
```

## Evaluation

Please follow the instructions [here](benchmark/README.md)

## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
```


