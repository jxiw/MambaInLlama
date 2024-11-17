# Instructions to Replicate Mamba-Llama3.2-3B

1. We streamlined the process and distilled the Hybrid Mamba2 3B model, utilizing the [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) as the teacher model and the [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) as the model for kqv initialization.

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba2/train_distill.py mamba2_llama3.2_3B/largest_llama3_ft_mamba2_0.5.yaml
```

This process roughly takes takes 8 to 9 days in 8xA100, 4 days in 8xH100.

The ```train_distill.py``` script allows you to initialize your Mamba model with a transformer model of the similar size (same layers), but distill it from a larger teacher model. You can also consider using [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct).

2. (Optional) Align the SFT model using DPO.

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba2/train_distill.py mamba2_llama3.2_3B/large_llama3_ft_mamba2_0.5_dpo.yaml
```

If you run DPO, you are able to get a model like [this](https://huggingface.co/JunxiongWang/Mamba2InLlama3B_Half_DPO). 

Zero-shot results when using the [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) as the teacher model, and the [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) as the initialized model

| Model          | [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [Llama3.2-Mamba2-3B-distill](https://huggingface.co/JunxiongWang/Mamba2InLlama3B_Half)       | [Llama-3.2-Mamba2-0.5-3B-dpo](https://huggingface.co/JunxiongWang/Mamba2InLlama3B_Half_DPO)       |
|---------------|---------------------------------------------------------------------------------|-----------------------------------|-----------------------------------|
| Initialization Model | N/A                                                                             | Llama-3.2-3B-Instruct             | Llama-3.2-3B-Instruct             |
| Teacher Model | N/A                                                                             | Llama-3.1-70B-Instruct             | Llama-3.1-70B-Instruct             |
| arc_challenge   | 0.459                                                                           | 0.4667                                                            | 0.541                                                                 |
| arc_easy        | 0.7407                                                                          | 0.7668                                                            | 0.8026                                                                |                                                               |
| hellaswag       | 0.7043                                                                          | 0.6913                                                            | 0.7445                                                                |
| mmlu            | 0.6043                                                                          | 0.5271                                                            | 0.5247                                                                |
| openbookqa      | 0.36                                                                            | 0.388                                                             | 0.424                                                                 |
| piqa            | 0.7568                                                                          | 0.7601                                                            | 0.7769                                                                |
| pubmedqa        | 0.696                                                                           | 0.638                                                             | 0.654                                                                 |
| race            | 0.4067                                                                          | 0.3981                                                            | 0.4344                                                                |
| winogrande      | 0.6748                                                                          | 0.6606                                                            | 0.6732                                                                |

We are removing more attention during the process, while keeping it in torch.