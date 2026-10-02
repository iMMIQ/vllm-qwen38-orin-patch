# SPDX-License-Identifier: Apache-2.0
"""W8A8 prefill projections sharing temporary weight and activation buffers."""

import torch

from .config import MAX_TOKENS, MIN_TOKENS, PROJECTIONS, padded_tokens

_runtime = None
_scratch = None


def resources():
    global _runtime, _scratch
    if _runtime is None:
        from .cuda_runtime import Runtime

        _runtime = Runtime()
        # TP=1: batched projections and drafting execute sequentially on one stream.
        # Do not retain views of these buffers across projection calls.
        _scratch = {
            "weight": torch.empty(34816 * 5120, device="cuda", dtype=torch.int8),
            "activation": torch.empty(
                MAX_TOKENS * 17408, device="cuda", dtype=torch.int8
            ),
            "scales": torch.empty(MAX_TOKENS, device="cuda", dtype=torch.float16),
        }
    return _runtime, _scratch


def prepare_projection_weight(rt, layer, n, k):
    _, scratch = resources()
    weight = scratch["weight"][: n * k].view(n, k)
    bn = PROJECTIONS[n, k]
    rt.launch(
        f"expandlut-{bn}-256-{n}-{k}",
        layer.weight_packed,
        layer.weight_scale,
        layer.weight_zero_point,
        layer._orin_row_scale,
        weight,
    )
    return weight


def launch_projection(
    rt, layer, n, k, padded, activation, scales, output, prepared_weight=None
):
    weight = (
        prepared_weight
        if prepared_weight is not None
        else prepare_projection_weight(rt, layer, n, k)
    )
    rt.launch(
        f"gemm2-{n}-{k}",
        activation,
        weight,
        scales,
        layer._orin_row_scale,
        output,
        grid=[n // 128 * 2, padded // 256, 1],
    )


def install():
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
        CompressedTensorsWNA16,
    )

    cls = CompressedTensorsWNA16
    if getattr(cls, "_orin_prefill_installed", False):
        return
    original_load = cls.process_weights_after_loading
    original_apply = cls.apply_weights
    attached = 0
    prefill_logged = False

    def load(self, layer):
        nonlocal attached
        original_load(self, layer)
        k, n = self.kernel.config.partition_weight_shape
        if (n, k) not in PROJECTIONS:
            return
        assert type(self.kernel).__name__ == "MarlinLinearKernel"
        assert self.pack_factor == 8 and self.group_size == 128
        assert not self.symmetric and not self.has_g_idx
        assert layer.weight_packed.shape == (k // 16, n * 2)
        assert layer.weight_scale.dtype == torch.float16
        rt, _ = resources()
        scale = torch.empty(n, device=layer.weight_packed.device, dtype=torch.float16)
        rt.launch(
            f"stats-{n}-{k}",
            layer.weight_packed,
            layer.weight_scale,
            layer.weight_zero_point,
            scale,
        )
        layer._orin_row_scale = scale
        attached += 1
        if attached == 256:
            print("ORIN_ATTACH count=256", flush=True)

    def apply(self, layer, x, bias):
        nonlocal prefill_logged
        m = x.numel() // x.shape[-1]
        if (
            not hasattr(layer, "_orin_row_scale")
            or not MIN_TOKENS <= m <= MAX_TOKENS
            or x.dtype != torch.float16
            or bias is not None
        ):
            return original_apply(self, layer, x, bias)
        k, n = self.kernel.config.partition_weight_shape
        rt, scratch = resources()
        padded = padded_tokens(m)
        flat = x.reshape(m, k)
        if m != padded:
            flat = torch.nn.functional.pad(flat, (0, 0, 0, padded - m))
        else:
            flat = flat.contiguous()
        activation = scratch["activation"][: padded * k].view(padded, k)
        scales = scratch["scales"][:padded]
        output = torch.empty(padded, n, device=x.device, dtype=x.dtype)
        weight = prepare_projection_weight(rt, layer, n, k)
        rt.launch(f"activation-{k}", flat, activation, scales, grid=[padded, 1, 1])
        launch_projection(
            rt, layer, n, k, padded, activation, scales, output, prepared_weight=weight
        )
        if not prefill_logged:
            prefill_logged = True
            print("ORIN_PREFILL linear=w8a8", flush=True)
        return output[:m].view(*x.shape[:-1], n)

    cls.process_weights_after_loading = load
    cls.apply_weights = apply
    cls._orin_prefill_installed = True
    from .fusions import install as install_fusions
    from .wy_backend import install as install_wy

    install_fusions()
    install_wy()
    print("ORIN_INSTALLED prefill=w8a8 decode=native", flush=True)
