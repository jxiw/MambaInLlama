**AlpacaEval**

[AlpacaEval](https://github.com/tatsu-lab/alpaca_eval): a single-turn benchmark which evaluates the helpfulness of chat and instruct models against `gpt-4` (Version 2).

### Install

```
cd alpaca_eval
pip install -e .
```

Note that the OpenAI requirements for MT-bench and AlpacaEval are different. If you use the same conda environment, please ensure you use `openai==1.30.1`.

### Models

```
mamba_0_5_dpo_ep3: Hybrid-Mamba (16 attention layers) distilled from HuggingFaceH4/zephyr-7b-beta + SFT + 3 epochs DPO with `beta=0.1`
mamba_0_5_dpo_ep1: Hybrid-Mamba (16 attention layers) distilled from HuggingFaceH4/zephyr-7b-beta + SFT + 1 epoch DPO with `beta=0.01`

mamba_0_75_dpo_ep3: Hybrid-Mamba (8 attention layers) distilled from HuggingFaceH4/zephyr-7b-beta + SFT + 3 epochs DPO with `beta=0.1`
mamba_0_75_dpo_ep1: Hybrid-Mamba (8 attention layers) distilled from HuggingFaceH4/zephyr-7b-beta + SFT + 1 epoch DPO with `beta=0.01`

mamba_0_875_dpo_ep3: Hybrid-Mamba (4 attention layers) distilled from HuggingFaceH4/zephyr-7b-beta + SFT + 3 epochs DPO with `beta=0.1`
mamba_0_875_dpo_ep1: Hybrid-Mamba (4 attention layers) distilled from HuggingFaceH4/zephyr-7b-beta + SFT + 1 epoch DPO with `beta=0.01`
```

Our results are here

```
mamba_0_5_dpo_ep1,16.69495490518617,1.104241627301418,127,675,3,805,15.962732919254657,community,1579,20.660664816178677,0.7377040113167127
mamba_0_5_dpo_ep3,14.367581340855573,1.0670257962872371,106,696,3,805,13.354037267080745,community,1550,17.477947322881132,0.7196913648943388
mamba_0_75_dpo_ep1,13.105692600706767,1.0070534468386845,98,705,2,805,12.298136645962733,community,1503,17.15774549259986,0.6902325752732126
mamba_0_75_dpo_ep3,11.696721201552796,0.9836999274591328,87,716,2,805,10.93167701863354,community,1503,13.889566949138848,0.6155458090726904
mamba_0_875_dpo_ep1,12.955972695503103,1.0249355755021206,97,705,3,805,12.236024844720497,community,1593,15.32013766091149,0.6595253340332127
mamba_0_875_dpo_ep3,11.073088366887827,0.9646410422463423,85,719,1,805,10.621118012422361,community,1599,12.674281873662284,0.6125756531005329
```

So the win rate of our hybrid mamba model against `gpt-4` is

```
Model,win rate,variance
mamba_0_5_dpo_ep1,20.660664816178677,0.7377040113167127
mamba_0_5_dpo_ep3,17.477947322881132,0.7196913648943388
mamba_0_75_dpo_ep1,17.15774549259986,0.6902325752732126
mamba_0_75_dpo_ep3,13.889566949138848,0.6155458090726904
mamba_0_875_dpo_ep1,15.32013766091149,0.6595253340332127
mamba_0_875_dpo_ep3,12.674281873662284,0.6125756531005329
```

Follow these steps to reproduce the results:

* Follow the installation instructions [here](https://github.com/tatsu-lab/alpaca_eval#quick-start).
* Since our model is distilled from Zephyr, please copy and paste the [config](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/models_configs/zephyr-7b-beta/configs.yaml) for `zephyr-7b-beta` and place it in the `model_configs` directory under `{your_zephyr_model}`.
* Next, update the [config name](https://github.com/tatsu-lab/alpaca_eval/blob/2daa6e11b194653043ca74f735728dc068e04aae/src/alpaca_eval/models_configs/zephyr-7b-beta/configs.yaml#L1) and [Hub model ID](https://github.com/tatsu-lab/alpaca_eval/blob/2daa6e11b194653043ca74f735728dc068e04aae/src/alpaca_eval/models_configs/zephyr-7b-beta/configs.yaml#L5) to match your model name. Your model name must include `mamba`. See [this](https://huggingface.co/JunxiongWang/mamba_0_75_dpo_ep3/blob/main/configs.yaml) for a correct config example.
* Follow the steps to evaluate your model [here](https://github.com/tatsu-lab/alpaca_eval/tree/main#evaluating-a-model).

```
alpaca_eval evaluate_from_model [YOUR_MODEL_PATH] --output_path [MODEL_OUTPUT_PATH] --chunksize 1800 --is_overwrite_leaderboard=True
``` 