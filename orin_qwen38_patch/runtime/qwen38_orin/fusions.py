# SPDX-License-Identifier: Apache-2.0
"""Prefill norm/residual and SiLU/activation-quantization fusions."""

import torch

from .config import MAX_TOKENS, MIN_TOKENS, padded_tokens


def install():
    from .backend import resources, launch_projection
    from .gdn_fusion import install as install_gdn
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    original_norm = GemmaRMSNorm.forward
    original_mlp = Qwen2MoeMLP.forward

    def norm(self, x, residual=None):
        m = x.numel() // x.shape[-1]
        if (
            not MIN_TOKENS <= m <= MAX_TOKENS
            or x.shape[-1] != 5120
            or x.dtype != torch.float16
            or self.weight.dtype != torch.float16
            or self.variance_epsilon != 1e-6
            or not x.is_contiguous()
            or (
                residual is not None
                and (
                    not residual.is_contiguous()
                    or residual.dtype not in (torch.float16, torch.float32)
                )
            )
        ):
            return original_norm(self, x, residual)
        rt, _ = resources()
        output = torch.empty_like(x)
        if residual is None:
            rt.launch("norm-none", x, self.weight, output, grid=[m, 1, 1])
            return output
        next_residual = torch.empty_like(x, dtype=torch.float32)
        name = "norm-float16" if residual.dtype == torch.float16 else "norm-float32"
        rt.launch(name, x, residual, self.weight, output, next_residual, grid=[m, 1, 1])
        return output, next_residual

    def mlp(self, x):
        m = x.numel() // x.shape[-1]
        down = self.down_proj
        if (
            not MIN_TOKENS <= m <= MAX_TOKENS
            or x.dtype != torch.float16
            or not hasattr(down, "_orin_row_scale")
            or self.expert_gate is not None
            or down.tp_size != 1
            or down.bias is not None
            or down.scheme.kernel.config.partition_weight_shape != (17408, 5120)
        ):
            return original_mlp(self, x)
        rt, scratch = resources()
        padded = padded_tokens(m)
        gate_up, _ = self.gate_up_proj(x)
        flat = gate_up.reshape(m, 34816)
        if padded != m:
            flat = torch.nn.functional.pad(flat, (0, 0, 0, padded - m))
        activation = scratch["activation"][: padded * 17408].view(padded, 17408)
        scales = scratch["scales"][:padded]
        output = torch.empty(padded, 5120, device=x.device, dtype=x.dtype)
        rt.launch("silu-q8", flat, activation, scales, grid=[padded, 1, 1])
        launch_projection(rt, down, 5120, 17408, padded, activation, scales, output)
        return output[:m].view(*x.shape[:-1], 5120)

    GemmaRMSNorm.forward = norm
    Qwen2MoeMLP.forward = mlp
    install_gdn()
