from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch import Tensor
from transformers.activations import ACT2FN

from mamba.hybrid_mamba_config import MambaConfig
from mamba.hybrid_mamba_layer import Mamba

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        device=None,
        dtype=None,
    ):
        super(MambaDecoderLayer, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.mamba = Mamba(
            d_model=config.d_model, d_inner=config.d_inner, d_xb=config.d_xb, layer_idx=layer_idx, **config.ssm_cfg, **factory_kwargs
        )
        self.mlp = MLP(config=config)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        # hidden_states = self.mamba(hidden_states, inference_params=inference_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # so here is just to be compatible with Transformer
        if kwargs is None:
            return (hidden_states, None, None)
        else:
            past_key_value = kwargs.get("past_key_value", None)
            if past_key_value is not None:
                dummy_keys = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                dummy_values = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                # Update kv cache with dummy values
                past_key_value.update(dummy_keys, dummy_values, self.layer_idx)
            return (hidden_states, None, past_key_value)