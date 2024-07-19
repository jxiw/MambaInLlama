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

## Evaluation

Please follow the instructions [here](benchmark/README.md)

## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
```


