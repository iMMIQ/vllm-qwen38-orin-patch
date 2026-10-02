# SPDX-License-Identifier: Apache-2.0
"""Prefill over an INT8 KV cache using a temporary FP16 view and FlashAttention 2."""

import torch
from vllm.triton_utils import tl, triton

_workspace = None
_logged = False


@triton.jit
def gather_kv(
    K,
    V,
    KS,
    VS,
    Table,
    Lens,
    Cu,
    OutK,
    OutV,
    TABLE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_TOKEN: tl.constexpr,
    CACHE_HEAD: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    SCALE_TOKEN: tl.constexpr,
    SCALE_HEAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(1)
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, head, dim = e // 1024, e // 256 % 4, e % 256
    mask = token < tl.load(Lens + seq)
    block = tl.load(
        Table + seq * TABLE_STRIDE + token // BLOCK_SIZE, mask=mask, other=0
    )
    slot = token % BLOCK_SIZE
    src = block * CACHE_BLOCK + slot * CACHE_TOKEN + head * CACHE_HEAD + dim
    scales = block * SCALE_BLOCK + slot * SCALE_TOKEN + head * SCALE_HEAD
    kv = tl.load(K + src, mask=mask, other=0).to(tl.float32)
    vv = tl.load(V + src, mask=mask, other=0).to(tl.float32)
    ks = tl.load(KS + scales, mask=mask, other=0)
    vs = tl.load(VS + scales, mask=mask, other=0)
    dest = tl.load(Cu + seq) * 1024 + e
    tl.store(OutK + dest, kv * ks, mask=mask)
    tl.store(OutV + dest, vv * vs, mask=mask)


def install():
    import vllm.v1.attention.backends.triton_attn as backend
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
    from vllm.v1.kv_cache_interface import KVQuantMode

    if getattr(backend.unified_attention, "_orin_kv8", False):
        return
    original = backend.unified_attention

    def attention(*args, **kw):
        global _workspace, _logged
        if (
            args
            or kw.get("kv_quant_mode") != KVQuantMode.INT8_PER_TOKEN_HEAD
            or kw["max_seqlen_q"] < 64
            or kw["q"].dtype != torch.float16
            or kw["q"].shape[1:] != (24, 256)
            or kw["k"].shape[2] != 4
            or kw["k"].shape[3] not in (256, 260)
            or kw.get("alibi_slopes") is not None
            or kw.get("sinks") is not None
            or kw.get("mm_prefix_range") is not None
            or kw.get("output_scale") is not None
            or kw.get("softcap", 0) != 0
            or kw.get("window_size") != (-1, -1)
            or not kw.get("causal")
            or kw.get("chunk_lookback", -1) != -1
        ):
            return original(*args, **kw)
        lens = kw["seqused_k"]
        cu = torch.empty(lens.numel() + 1, device=lens.device, dtype=torch.int32)
        cu[0] = 0
        torch.cumsum(lens, 0, dtype=torch.int32, out=cu[1:])
        total = int(cu[-1].item())
        if _workspace is None or _workspace[0].shape[0] < total:
            capacity = triton.next_power_of_2(total)
            _workspace = tuple(
                torch.empty(capacity, 4, 256, device=lens.device, dtype=torch.float16)
                for _ in range(2)
            )
        k, v = (t[:total] for t in _workspace)
        cache, scales = kw["k"], kw["k_scale_cache"]
        gather_kv[(triton.cdiv(kw["max_seqlen_k"] * 1024, 4096), lens.numel())](
            cache,
            kw["v"],
            scales,
            kw["v_scale_cache"],
            kw["block_table"],
            lens,
            cu,
            k,
            v,
            kw["block_table"].stride(0),
            *cache.stride()[:3],
            *scales.stride(),
            cache.shape[1],
            4096,
            num_warps=4
        )
        flash_attn_varlen_func(
            q=kw["q"],
            k=k,
            v=v,
            out=kw["out"],
            cu_seqlens_q=kw["cu_seqlens_q"],
            cu_seqlens_k=cu,
            max_seqlen_q=kw["max_seqlen_q"],
            max_seqlen_k=kw["max_seqlen_k"],
            softmax_scale=kw["softmax_scale"],
            causal=True,
            fa_version=2,
        )
        if not _logged:
            print("ORIN_KV8_PREFILL flash2 temporary-fp16", flush=True)
            _logged = True

    attention._orin_kv8 = True
    backend.unified_attention = attention
