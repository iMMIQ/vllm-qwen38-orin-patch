# SPDX-License-Identifier: Apache-2.0
"""Head-first WY transform for 8192-token single-sequence prefill."""

import importlib
import torch


def install():
    from .backend import resources

    chunk = importlib.import_module("vllm.model_executor.layers.fla.ops.chunk")
    original = chunk.recompute_w_u_fwd
    wy_logged = False

    def recompute(k, v, beta, g_cumsum, A, cu_seqlens, chunk_indices=None):
        nonlocal wy_logged
        m = k.shape[1]
        if (
            m != 8192
            or k.shape != (1, m, 16, 128)
            or v.shape != (1, m, 48, 128)
            or beta.shape != (1, m, 48)
            or g_cumsum.shape != (1, m, 48)
            or A.shape != (1, m, 48, 64)
            or cu_seqlens is None
            or cu_seqlens.numel() != 2
            or cu_seqlens.dtype != torch.int32
            or any(t.dtype != torch.float16 for t in (k, v, A))
            or any(t.dtype != torch.float32 for t in (beta, g_cumsum))
            or any(not t.is_contiguous() for t in (k, v, A, beta, g_cumsum, cu_seqlens))
        ):
            return original(k, v, beta, g_cumsum, A, cu_seqlens, chunk_indices)
        rt, _ = resources()
        w, u = torch.empty_like(v), torch.empty_like(v)
        rt.launch(
            "wy-256-head-exp2",
            k,
            v,
            beta,
            g_cumsum,
            A,
            cu_seqlens,
            w,
            u,
            grid=[48, m // 64, 1],
        )
        if not wy_logged:
            wy_logged = True
            print("ORIN_WY M=8192", flush=True)
        return w, u

    chunk.recompute_w_u_fwd = recompute
