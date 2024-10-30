#!/bin/bash

# Define the models
models=(
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "JunxiongWang/MambaInLlama_0_50"
  "JunxiongWang/MambaInLlama_0_75"
  "JunxiongWang/MambaInLlama_0_875"
  "JunxiongWang/Mamba2InLlama_0_50"
  "JunxiongWang/Mamba2InLlama_0_75"
  "JunxiongWang/Mamba2InLlama_0_875"
  "JunxiongWang/Mamba2InLlama_1"
)

# Define the generation lengths
gen_lens=(1024 2048 4096 8192 16384 32768 65536)

# Loop through models and generation lengths
for model in "${models[@]}"; do
  for genlen in "${gen_lens[@]}"; do
    # echo "Running benchmark for model $model with genlen $genlen"
    python benchmark_generation_speed.py --model-name "$model" --genlen "$genlen"
  done
done