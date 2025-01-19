# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import json

import torch
import torch.nn as nn

from mamba_ssm.utils.generation import GenerationMixin

from transformers import AutoModelForCausalLM

from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from transformers.utils.hub import cached_file

from mamba.hybrid_mamba_config import MambaConfig
from mamba_inference.hybrid_model import MambaDecoderLayer, MHADecoderLayer

from util import load_safetensors_to_dict
from collections import namedtuple

def merge_mamba_projections_for_layers(checkpoint, layer_indices):
    for layer_idx in layer_indices:
        
        z_proj_key = f"model.layers.{layer_idx}.mamba.in_proj_z.weight"
        x_proj_key = f"model.layers.{layer_idx}.mamba.in_proj_x.weight"
        b_proj_key = f"model.layers.{layer_idx}.mamba.B_proj.weight"
        c_proj_key = f"model.layers.{layer_idx}.mamba.C_proj.weight"
        dt_proj_key = f"model.layers.{layer_idx}.mamba.dt_proj_down.weight"

        if z_proj_key in checkpoint and x_proj_key in checkpoint and b_proj_key in checkpoint and c_proj_key in checkpoint and dt_proj_key in checkpoint:
            # Assuming all the projections have the same shape, otherwise adjust accordingly
            z_proj_weight = checkpoint[z_proj_key]
            x_proj_weight = checkpoint[x_proj_key]
            b_proj_weight = checkpoint[b_proj_key]
            c_proj_weight = checkpoint[c_proj_key]
            dt_proj_weight = checkpoint[dt_proj_key]

            # zxbcdt
            # Concatenate the weights along the first dimension (often dimension 0)
            in_proj_weight = torch.cat([z_proj_weight, x_proj_weight, b_proj_weight, c_proj_weight, dt_proj_weight], dim=0)

            # Assign the new weight to the corresponding in_proj key
            in_proj_key = f"model.layers.{layer_idx}.mamba.in_proj.weight"
            checkpoint[in_proj_key] = in_proj_weight

            # Optionally, remove the old keys to clean up the checkpoint
            del checkpoint[z_proj_key]
            del checkpoint[x_proj_key]
            del checkpoint[b_proj_key]
            del checkpoint[c_proj_key]
            del checkpoint[dt_proj_key]
    
    return checkpoint

def merge_mha_projections_for_layers(checkpoint, layer_indices):
    for layer_idx in layer_indices:
        # Get the weights for q_proj, k_proj, and v_proj
        q_proj_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        k_proj_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        v_proj_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        o_proj_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"

        # Check if the keys exist in the checkpoint
        if q_proj_key in checkpoint and k_proj_key in checkpoint and v_proj_key in checkpoint:
            # Assuming all the projections have the same shape, otherwise adjust accordingly
            q_proj_weight = checkpoint[q_proj_key]
            k_proj_weight = checkpoint[k_proj_key]
            v_proj_weight = checkpoint[v_proj_key]

            # Concatenate the weights along the first dimension (often dimension 0)
            in_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)

            # Assign the new weight to the corresponding in_proj key
            in_proj_key = f"model.layers.{layer_idx}.mha.in_proj.weight"
            checkpoint[in_proj_key] = in_proj_weight

            # Optionally, remove the old keys to clean up the checkpoint
            del checkpoint[q_proj_key]
            del checkpoint[k_proj_key]
            del checkpoint[v_proj_key]

        if o_proj_key in checkpoint:
            out_proj_key = f"model.layers.{layer_idx}.mha.out_proj.weight"
            checkpoint[out_proj_key] = checkpoint[o_proj_key]
            del checkpoint[o_proj_key]

    return checkpoint

MAMBA_CONFIG_NAME = "mamba_config.json"

class MambaTransformerHybridModelWrapper(nn.Module, GenerationMixin):

    def __init__(self, checkpoint_path, transformer_model, mamba_config, attn_layers, dtype, load_from_hub=False, **kwargs):
        super(MambaTransformerHybridModelWrapper, self).__init__()
        self.mamba_config = mamba_config
        self.attn_layers = attn_layers
        self.ssm_layers = [layer_idx for layer_idx in range(mamba_config.n_layer) if layer_idx not in attn_layers]
        self.model = transformer_model
        self.config = self.model.config

        for layer_idx in range(mamba_config.n_layer):
            if layer_idx in attn_layers:
                layer_encoder = MHADecoderLayer(
                    self.config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
            else:
                layer_encoder = MambaDecoderLayer(
                    mamba_config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
            self.model.model.layers[layer_idx] = layer_encoder
            
        print("self.model:", self.model)      
           
        if checkpoint_path is not None:
            if load_from_hub:
                # load from a huggingface hub
                ckpt = load_state_dict_hf(checkpoint_path, device=torch.device("cpu"), dtype=dtype)
            else:
                # load from a local directory
                if os.path.exists(f"{checkpoint_path}/pytorch_model.bin"):
                    # support save from bin file
                    ckpt = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=torch.device("cpu"))
                else:
                    # support save from safetensors
                    ckpt = load_safetensors_to_dict(checkpoint_path)
        
            merge_mha_projections_for_layers(ckpt, self.attn_layers)
            merge_mamba_projections_for_layers(ckpt, self.ssm_layers)
            self.model.load_state_dict(ckpt)
        self.model = self.model.to(dtype).cuda()
        self.device = self.model.device
        self.can_generate = self.model.can_generate
        self.generation_config = self.model.generation_config

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.model.model.layers)
        }

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.model.model.embed_tokens(input_ids, **mixer_kwargs)
        for decoder_layer in self.model.model.layers:
            hidden_states = decoder_layer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        hidden_states = self.model.model.norm(hidden_states)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.model.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

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

    def get_memory_footprint(self):
        return self.model.get_memory_footprint()

    def generate(
        self,
        input_ids,
        max_length=1024,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):

        if kwargs is not None:
            max_new_tokens = kwargs.pop('max_new_tokens', None)
            if max_new_tokens is not None:
                max_length = max_new_tokens + input_ids.shape[1]
            do_sample = kwargs.pop('do_sample', True)
            if not do_sample:
                top_k, top_p, min_p = 1, 0.0, 0.0
            
            cg = kwargs.pop('cg', True)
            eos_token_id = kwargs.pop('eos_token_id', None)
            if eos_token_id is None:
                eos_token_id = self.config.eos_token_id

            attention_mask = kwargs.pop('attention_mask', None)
            pad_token_id = kwargs.pop('pad_token_id', None)
            no_repeat_ngram_size = kwargs.pop('no_repeat_ngram_size', None)
            length_penalty = kwargs.pop('length_penalty', None)
            num_return_sequences = kwargs.pop('num_return_sequences', None)
            num_beams = kwargs.pop('num_beams', None)
            low_memory = kwargs.pop('low_memory', None)
            stopping_criteria = kwargs.pop('stopping_criteria', None)

        return super().generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=cg,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            eos_token_id=eos_token_id,
            **kwargs,
        )
