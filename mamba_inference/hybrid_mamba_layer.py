# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_inner,
        d_xb,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        repeat_kv_before_conv=True,
        conv_bias=True,
        proj_x_bias=False,
        proj_z_bias=False,
        out_proj_bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_xb = d_xb
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_inner if d_inner is not None else int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.repeat_kv_before_conv = repeat_kv_before_conv

        if self.repeat_kv_before_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        else:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_xb,
                out_channels=self.d_xb,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_xb,
                padding=d_conv - 1,
                **factory_kwargs,
            )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.num_xb_head = self.d_xb // self.d_state
        self.num_C_head = self.d_inner // self.d_state
        self.repeat_group = self.num_C_head // self.num_xb_head

        # fuse those layers
        # self.in_proj_z = nn.Linear(self.d_model, self.d_inner, bias=proj_z_bias, **factory_kwargs)
        # self.in_proj_x = nn.Linear(self.d_model, self.d_xb, bias=proj_x_bias, **factory_kwargs)
        # self.B_proj = nn.Linear(self.d_model, self.d_xb, bias=False, **factory_kwargs)
        # self.C_proj = nn.Linear(self.d_model, self.d_inner, bias=False, **factory_kwargs)
        # self.dt_proj_down = nn.Linear(self.d_model, self.dt_rank, bias=False, **factory_kwargs)

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_xb + 2 * self.d_inner + self.dt_rank, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=out_proj_bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out
        
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        zxbcdt = self.in_proj(hidden_states)
        z, x, B, C, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_xb, self.d_xb, self.d_inner, self.dt_rank], dim=-1
        )

        x = rearrange(x, "b l d -> b d l")
        z = rearrange(z, "b l d -> b d l")

        B = rearrange(B, "b l (n_group dstate) -> b n_group l dstate", dstate=self.d_state)
        B = repeat_kv(B, self.repeat_group)  # B, n_group, L, H
        B = rearrange(B, "b n_group l dstate -> b n_group dstate l").contiguous()
        C = rearrange(C, "b l (n_group dstate) -> b n_group dstate l", dstate=self.d_state).contiguous()

        dt = self.dt_proj(dt)  # B, L, d_inner
        dt = rearrange(dt, "b l d -> b d l")  # B, d_inner, L

        if self.repeat_kv_before_conv:
            # b d l
            x = rearrange(x, "b (n_group dstate) l -> b n_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.repeat_group)
            x = rearrange(x, "b n_group l dstate -> b (n_group dstate) l")

        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            # Update state (B D W)
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        if not self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) l -> b n_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.repeat_group)
            x = rearrange(x, "b n_group l dstate -> b (n_group dstate) l")

        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            # ssm_state.copy_(last_state.unsqueeze(-2))
            ssm_state.copy_(rearrange(last_state, "b (h d) n -> b h d n", h=self.num_C_head))
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        hidden_states_input = hidden_states.squeeze(1)

        # hidden_states_input shape: (B, H)
        # x = self.in_proj_x(hidden_states_input)
        # z = self.in_proj_z(hidden_states_input)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # B = self.B_proj(hidden_states_input)  # B, H_inner
        # C = self.C_proj(hidden_states_input)  # B, H

        zxbcdt = self.in_proj(hidden_states_input)
        z, x, B, C, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_xb, self.d_xb, self.d_inner, self.dt_rank], dim=-1
        )

        B = rearrange(B, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
        B = torch.repeat_interleave(B, dim=1, repeats=self.repeat_group)

        C = rearrange(C, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state).contiguous()

        # dt = self.dt_proj_down(hidden_states_input) # B, d_rank
        dt = self.dt_proj(dt)   # B, d_inner

        if self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
            x = torch.repeat_interleave(x, dim=1, repeats=self.repeat_group)
            x = rearrange(x, "b n_group dstate -> b (n_group dstate)")

        # Conv step
        if causal_conv1d_update is None:
            # Update state (B D W)
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        if not self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
            x = torch.repeat_interleave(x, dim=1, repeats=self.repeat_group)
            x = rearrange(x, "b n_group dstate -> b (n_group dstate)")

        x = rearrange(x, "b (h d) -> b h d", h=self.num_C_head)
        dt = rearrange(dt, "b (h d) -> b h d", h=self.num_C_head)
        A = rearrange(A, "(h d) n -> h d n", h=self.num_C_head)
        D = rearrange(self.D, "(h d) -> h d", h=self.num_C_head)
        z = rearrange(z, "b (h d) -> b h d", h=self.num_C_head)
        dt_bias = rearrange(self.dt_proj.bias, "(h d) -> h d", h=self.num_C_head)

        # SSM step
        assert selective_state_update is not None
        y = selective_state_update(
            ssm_state, x, dt, A, B, C, D, z=z, dt_bias=dt_bias, dt_softplus=True
        )

        # y = y.squeeze(-1)
        y = rearrange(y, "b h d -> b (h d)")
        out = self.out_proj(y)
        
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        if self.repeat_kv_before_conv:
            conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype=conv_dtype)
        else:
            conv_state = torch.zeros(batch_size, self.d_xb, self.d_conv, device=device, dtype=conv_dtype)
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.num_C_head, self.d_inner // self.num_C_head, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            if self.repeat_kv_before_conv:
                conv_state = torch.zeros(
                    batch_size,
                    self.d_inner,
                    self.d_conv,
                    device=self.conv1d.weight.device,
                    dtype=self.conv1d.weight.dtype,
                )
            else:
                conv_state = torch.zeros(
                    batch_size,
                    self.d_xb,
                    self.d_conv,
                    device=self.conv1d.weight.device,
                    dtype=self.conv1d.weight.dtype,
                )
            ssm_state = torch.zeros(
                batch_size,
                self.num_C_head,
                self.d_inner // self.num_C_head,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
