# SPDX-License-Identifier: Apache-2.0
"""Fuse GDN output norm/gate/A8; retain the native recurrent core and states."""

import torch

from .config import MAX_TOKENS, MIN_TOKENS, padded_tokens


def install():
    from .backend import resources, launch_projection
    from vllm.model_executor.layers.mamba.gdn_linear_attn import (
        GatedDeltaNetAttention,
        _encode_layer_name,
    )

    original = GatedDeltaNetAttention.forward
    fusion_logged = False

    def forward(self, hidden_states, output):
        nonlocal fusion_logged
        m = hidden_states.shape[0]
        if (
            not MIN_TOKENS <= m <= MAX_TOKENS
            or hidden_states.dtype != torch.float16
            or output.dtype != torch.float16
            or self.tp_size != 1
            or self.gqa_interleaved_layout
            or hasattr(self, "in_proj_qkv")
            or self.num_v_heads != 48
            or self.head_v_dim != 128
            or not hasattr(self.out_proj, "_orin_row_scale")
            or self.norm.eps != 1e-6
            or not self.norm.norm_before_gate
            or self.norm.group_size is not None
            or self.norm.bias is not None
            or self.norm.activation not in ("swish", "silu")
            or self.norm.weight.dtype != torch.float16
        ):
            return original(self, hidden_states, output)
        rt, scratch = resources()
        padded = padded_tokens(m)
        qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        b, a = ba.chunk(2, dim=-1)
        core = torch.zeros(
            m, 48, 128, device=hidden_states.device, dtype=hidden_states.dtype
        )
        torch.ops.vllm.gdn_attention_core(
            qkvz[:, :10240],
            b.contiguous(),
            a.contiguous(),
            core,
            _encode_layer_name(self.prefix),
        )
        activation = scratch["activation"][: padded * 6144].view(padded, 6144)
        scales = scratch["scales"][:padded]
        if m != padded:
            activation[m:].zero_()
            scales[m:].fill_(1)
        rt.launch(
            "gdn-norm-q8",
            core,
            qkvz,
            self.norm.weight,
            activation,
            scales,
            grid=[m, 1, 1],
        )
        direct = padded == m and output.is_contiguous()
        projected = (
            output
            if direct
            else torch.empty(
                padded, 5120, device=hidden_states.device, dtype=hidden_states.dtype
            )
        )
        launch_projection(
            rt, self.out_proj, 5120, 6144, padded, activation, scales, projected
        )
        if not direct:
            output[:m].copy_(projected[:m])
        if not fusion_logged:
            fusion_logged = True
            print("ORIN_FUSION gdn-norm-q8", flush=True)

    GatedDeltaNetAttention.forward = forward
