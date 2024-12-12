# Instructions to Replicate Mamba-Llama3.1-8B

1. We streamlined the process and distilled the Hybrid Mamba2 8B model, utilizing the [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) as the teacher model and the [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) as the model for kqv initialization.

**While layerwise distillation can enhance performance, here we use a single-step approach to illustrate our core idea (reuse weights from kqvo). And this end to end traning is the most important.**

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba2/train_distill.py llama3.1_8B/llama3.1_mamba2_0.5.yaml
```

This process roughly takes takes 8 to 9 days in 8xA100, 4 days in 8xH100.

2. (Optional) Align the SFT model using DPO.

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba2/train_distill.py llama3.2_8B/llama3.1_mamba2_0.5_dpo.yaml
```

If you run DPO, you are able to get a model like [this](https://huggingface.co/JunxiongWang/Mamba2InLlama3B_Half_DPO). 

Zero-shot results when using the [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) as the teacher model, and the [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) as the initialized model

| Model          | [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [Llama-3.1-Mamba-0.5-8B](https://huggingface.co/JunxiongWang/Llama3.1-Mamba-8B-distill)       | [Llama-3.1-Mamba-0.5-8B-dpo](https://huggingface.co/JunxiongWang/Llama3.1-Mamba-8B-dpo)       | [Llama-3.1-Mamba2-0.5-8B](https://huggingface.co/JunxiongWang/Llama3.1-Mamba2-8B-distill)       | [Llama-3.1-Mamba2-0.5-8B-dpo](https://huggingface.co/JunxiongWang/Llama3.1-Mamba-8B-dpo)       |
|---------------|---------------------------------------------------------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|
| Initialization Model | N/A                                                                             | Llama-3.1-8B-Instruct             | Llama-3.1-8B-Instruct             | Llama-3.1-8B-Instruct             | Llama-3.1-8B-Instruct             |
| Teacher Model | N/A                                                                             | Llama-3.1-70B-Instruct             | Llama-3.1-70B-Instruct             | Llama-3.1-70B-Instruct             | Llama-3.1-70B-Instruct             |
| arc_challenge       | 0.552                 | 0.5384                   | 0.5657                | 0.5265                    | 0.5973                |
| arc_easy            | 0.8178                | 0.8224                   | 0.8401                | 0.822                     | 0.8481                |
| hellaswag           | 0.7921                | 0.7591                   | 0.7736                | 0.7536                    | 0.7969                |
| mmlu (0 shot)       | 0.6812                | 0.6213                   | 0.636                 | 0.6101                    | 0.5974                |
| openbookqa          | 0.432                 | 0.428                    | 0.442                 | 0.416                     | 0.44                  |
| piqa                | 0.8079                | 0.7933                   | 0.8041                | 0.7889                    | 0.8003                |
| pubmedqa            | 0.752                 | 0.72                     | 0.744                 | 0.726                     | 0.746                 |
| race                | 0.4478                | 0.4211                   | 0.4344                | 0.4211                    | 0.4612                |
| winogrande          | 0.7388                | 0.7277                   | 0.738                 | 0.7174                    | 0.7411                |
| truthful            | 0.4267                | 0.4002                   | 0.4607                | 0.4031                    | 0.5022                |
