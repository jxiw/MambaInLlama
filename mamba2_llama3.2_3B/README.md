# Instructions to Replicate Mamba-Llama3.2-3B

As described in our technical report, training this model proceeds in three steps:

1. We streamlined the process and distilled the Hybrid Mamba2 3B model, utilizing the [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) as the teacher model and the [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) as the base model for initialization.

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba2/train_sft_distill.py mamba2_llama3.2_3B/large_llama3_ft_mamba2_0.5.yaml
```

This process roughly takes less than 3 days in 8xA100.

The ```train_sft_distill.py``` script allows you to initialize your Mamba model with a transformer model of the similar size (same layers), but distill it from a larger teacher model. You can also consider using [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) if you have enough resources; it roughly takes 8 days in 8xA100.

2. (Optional) Align the SFT model using DPO.

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba2/train_sft_distill.py mamba2_llama3.2_3B/large_llama3_ft_mamba2_0.5_dpo.yaml
```

Zero-shot results


| Task         | [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | Llama-Mamba2-0.5-3.2-3B-teacher-Llama-3.1-8B-Instruct-sft | Llama-Mamba2-0.5-3.2-3B-teacher-Llama-3.1-8B-Instruct-dpo |
|--------------|-----------------|--------------------------------------------------------------|------------------------------------------------------------------|
| arc_challenge| 0.4352          | 0.4309                                                       | 0.5145                                                          |
|              | 0.459           | 0.4608                                                       | 0.5222                                                          |
| arc_easy     | 0.7407          | 0.7588                                                       | 0.7997                                                          |
|              | 0.678           | 0.7193                                                       | 0.7424                                                          |
| hellaswag    | 0.5221          | 0.5104                                                       | 0.5514                                                          |
|              | 0.7043          | 0.6905                                                       | 0.7484                                                          |
| mmlu         | 0.6043          | 0.5216                                                       | 0.5088                                                          |
| openbookqa   | 0.282           | 0.278                                                        | 0.318                                                           |
|              | 0.36            | 0.392                                                        | 0.412                                                           |
| piqa         | 0.7568          | 0.7622                                                       | 0.7726                                                          |
|              | 0.7568          | 0.7617                                                       | 0.7764                                                          |
| pubmedqa     | 0.696           | 0.662                                                        | 0.672                                                           |
| race         | 0.4067          | 0.4029                                                       | 0.4679                                                          |
| winogrande   | 0.6748          | 0.644                                                        | 0.6535                                                          |


