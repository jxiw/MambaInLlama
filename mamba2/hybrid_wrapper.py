# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import json

import torch
import torch.nn as nn

from dataclasses import dataclass, field

from transformers import AutoModelForCausalLM

from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from transformers.utils.hub import cached_file

from mamba2.hybrid_model import MambaDecoderLayer
from mamba2.hybrid_mamba_config import MambaConfig

from util import load_safetensors_to_dict

MAMBA_CONFIG_NAME = "mamba_config.json"

class MambaTransformerHybridModelWrapper(nn.Module):

    def __init__(self, checkpoint_path, transformer_model, mamba_config, attn_layers, dtype, init_with_kqvo, load_from_hub=False, **kwargs):
        super(MambaTransformerHybridModelWrapper, self).__init__()
        self.mamba_config = mamba_config
        self.attn_layers = attn_layers
        self.model = transformer_model
        self.config = self.model.config
        
        for layer_idx in range(mamba_config.n_layer):
            if layer_idx not in attn_layers:
                mamba_encoder = MambaDecoderLayer(
                    mamba_config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
                
                if init_with_kqvo:
                    # init weights using attention weights
                    mamba_encoder.mlp.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].mlp.state_dict())
                    mamba_encoder.input_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].input_layernorm.state_dict())
                    mamba_encoder.post_attention_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].post_attention_layernorm.state_dict())
                    mamba_encoder.mamba.out_proj.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.o_proj.state_dict())
                    # [z, x, B, C, dt]
                    mamba_encoder.mamba.in_proj.weight.data[mamba_config.d_inner:mamba_config.d_inner+mamba_config.d_xb, :].copy_(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.v_proj.weight.data)
                    mamba_encoder.mamba.in_proj.weight.data[mamba_config.d_inner+mamba_config.d_xb:mamba_config.d_inner+2*mamba_config.d_xb, :].copy_(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.k_proj.weight.data)
                    mamba_encoder.mamba.in_proj.weight.data[mamba_config.d_inner+2*mamba_config.d_xb:2*mamba_config.d_inner+2*mamba_config.d_xb, :].copy_(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.q_proj.weight.data)
                    # keep dtype to be the same
                    mamba_encoder.mlp = mamba_encoder.mlp.to(dtype)
                    mamba_encoder.input_layernorm = mamba_encoder.input_layernorm.to(dtype)
                    mamba_encoder.post_attention_layernorm = mamba_encoder.post_attention_layernorm.to(dtype)

                self.model.model.layers[layer_idx] = mamba_encoder

        if checkpoint_path is not None:
            if load_from_hub:
                # load from a huggingface hub
                self.model.load_state_dict(load_state_dict_hf(checkpoint_path, device=torch.device("cpu"), dtype=dtype))
            else:
                # load from a local directory
                if os.path.exists(f"{checkpoint_path}/pytorch_model.bin"):
                    # support save from bin file
                    self.model.load_state_dict(torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=torch.device("cpu")))
                else:
                    # support save from safetensors
                    self.model.load_state_dict(load_safetensors_to_dict(checkpoint_path))
        
        self.model = self.model.to(dtype).cuda()

    def allocate_mamba_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.model.model.layers)
            if isinstance(layer, MambaDecoderLayer)
        }

    def forward(
        self,
        input_ids,
        **kwargs,
    ):
        return self.model(input_ids, **kwargs)

    def generate(
        self,
        input_ids,
        **kwargs,
    ):
        output = self.model.generate(
            input_ids,
            use_cache=False,
            **kwargs,
        )
        return output
    
    @staticmethod
    def init_distillation(
        checkpoint_path,
        tranformer_name,
        mamba_config,
        attn_layers,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        init_with_kqvo=True,
        **kwargs,
    ):
        transformer_model = AutoModelForCausalLM.from_pretrained(tranformer_name, torch_dtype=dtype, attn_implementation=attn_implementation)
        return MambaTransformerHybridModelWrapper(checkpoint_path, transformer_model, mamba_config, attn_layers, dtype, init_with_kqvo)

    @staticmethod
    def from_pretrained_local(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(config_data["_name_or_path"], torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        with open(f'{pretrained_model_name}/{MAMBA_CONFIG_NAME}', 'r') as json_file:
            config_dict = json.load(json_file)
        mamba_config = MambaConfig(**config_dict)
        return MambaTransformerHybridModelWrapper(pretrained_model_name, transformer_model, mamba_config, mamba_config.attn_layers, torch_dtype, init_with_kqvo=False) 

    @staticmethod
    def from_pretrained_hub(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(config_data["_name_or_path"], torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        resolved_archive_file = cached_file(pretrained_model_name, MAMBA_CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        config_dict = json.load(open(resolved_archive_file))
        mamba_config = MambaConfig(**config_dict)
        return MambaTransformerHybridModelWrapper(pretrained_model_name, transformer_model, mamba_config, mamba_config.attn_layers, torch_dtype, init_with_kqvo=False, load_from_hub=True) 

    @staticmethod
    def from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        if os.path.exists(pretrained_model_name):
            return MambaTransformerHybridModelWrapper.from_pretrained_local(pretrained_model_name, torch_dtype, attn_implementation)
        else:
            return MambaTransformerHybridModelWrapper.from_pretrained_hub(pretrained_model_name, torch_dtype, attn_implementation)

    def save_config(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, 'mamba_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.mamba_config.__dict__, f, indent=4)
