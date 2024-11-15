
This code is adapted from [here](https://github.com/Leooyii/LCEG/tree/main/needle). If you use this code, please consider citing the original paper.

### How to Run Needle Test

```
export MODEL_NAME=JunxiongWang/Llama3.2-Mamba2-3B-distill
export RESULT_SAVE_PATH=Llama3.1-Mamba-distill
python -u needle_in_haystack.py --s_len 0 --e_len 65536\
    --model_provider Mamba \
    --model_path ${MODEL_NAME} \
    --test_name ${RESULT_SAVE_PATH} 
```

Notice that, during the distillation, we only train model with 2k context.

Here is the results

<img src="img/needle.png" alt="needle">
