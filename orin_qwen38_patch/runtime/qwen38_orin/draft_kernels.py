# SPDX-License-Identifier: Apache-2.0
"""Draft-only W8A8 projections with a runtime row count and fixed tile sizes."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import nearbyint


@triton.jit
def _quantize_rows(X, Q, S, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BK)
    values = tl.load(X + row * K + k, k < K, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values)) / 127, 1e-10)
    tl.store(Q + row * K + k, nearbyint(values / scale).to(tl.int8), k < K)
    tl.store(S + row, scale)


@triton.jit(do_not_specialize=["M"])
def _project(
    A,
    W,
    AS,
    WS,
    O,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.int32)
    for start in range(0, K, BK):
        av = tl.load(
            A + m[:, None] * K + start + k[None, :],
            m[:, None] < M,
            other=0,
        )
        wv = tl.load(
            W + n[None, :] * K + start + k[:, None],
            n[None, :] < N,
            other=0,
        )
        acc += tl.dot(av, wv)
    values = (
        acc.to(tl.float32)
        * tl.load(AS + m, m < M, other=0)[:, None]
        * tl.load(WS + n, n < N, other=0)[None, :]
    )
    tl.store(
        O + m[:, None] * N + n[None, :],
        values,
        (m[:, None] < M) & (n[None, :] < N),
    )


class DraftLinear:
    """Keep original FP16 weights alongside a derived, row-scaled INT8 copy."""

    def __init__(self, weight):
        if weight.ndim != 2 or weight.dtype != torch.float16:
            raise ValueError("DraftLinear requires an FP16 matrix")
        if not weight.is_cuda or not weight.is_contiguous():
            raise ValueError("DraftLinear requires contiguous CUDA weights")
        self.weight = weight
        self.n, self.k = weight.shape
        if self.k % 128:
            raise ValueError("Draft input channels must be divisible by 128")
        self.quantized = torch.empty_like(weight, dtype=torch.int8)
        self.scales = torch.empty(self.n, device=weight.device, dtype=torch.float32)
        _quantize_rows[(self.n,)](
            weight,
            self.quantized,
            self.scales,
            self.k,
            triton.next_power_of_2(self.k),
            num_warps=8,
        )

    def scores(self, x, dtype=torch.float32):
        if x.shape[-1] != self.k or x.dtype != torch.float16:
            raise ValueError("Draft activation shape or dtype does not match weights")
        flat = x.reshape(-1, self.k).contiguous()
        m = flat.shape[0]
        output = torch.empty((m, self.n), device=x.device, dtype=dtype)
        if not m:
            return output.reshape(*x.shape[:-1], self.n)
        activation = torch.empty_like(flat, dtype=torch.int8)
        scales = torch.empty(m, device=x.device, dtype=torch.float32)
        _quantize_rows[(m,)](
            flat,
            activation,
            scales,
            self.k,
            triton.next_power_of_2(self.k),
            num_warps=8,
        )
        bm = min(128, max(16, triton.next_power_of_2(m)))
        _project[(triton.cdiv(self.n, 64), triton.cdiv(m, bm))](
            activation,
            self.quantized,
            scales,
            self.scales,
            output,
            m,
            self.n,
            self.k,
            bm,
            64,
            128,
            num_stages=2,
            num_warps=4,
        )
        return output.reshape(*x.shape[:-1], self.n)

    def top_tokens(self, x, candidates=16):
        """Coarse top-k per row, followed by original FP16 weight reranking."""
        flat = x.reshape(-1, self.k).contiguous()
        m = flat.shape[0]
        if not m:
            return torch.empty(x.shape[:-1], device=x.device, dtype=torch.long)
        ids = self.scores(flat).topk(min(candidates, self.n), dim=-1).indices
        # Include token 0 so zero logits retain argmax's smallest-index rule.
        ids = torch.cat((ids, torch.zeros_like(ids[:, :1])), dim=-1).sort(dim=-1).values
        selected = self.weight.index_select(0, ids.flatten())
        if m == 1:
            logits = torch.nn.functional.linear(flat, selected)
        else:
            logits = torch.bmm(
                selected.reshape(m, ids.shape[1], self.k), flat.unsqueeze(-1)
            ).squeeze(-1)
        best = ids.gather(1, logits.argmax(dim=-1, keepdim=True)).squeeze(-1)
        return best.reshape(x.shape[:-1])

    def warmup(self, head=False, row_buckets=()):
        """Compile each tile once, not one specialization per batch size."""
        for m in (1, 17, 33, 65):
            x = torch.zeros(m, self.k, device=self.weight.device, dtype=torch.float16)
            self.scores(x, dtype=torch.float32 if head else torch.float16)
        if head:
            for m in sorted({1, *row_buckets}):
                x = torch.zeros(
                    m, self.k, device=self.weight.device, dtype=torch.float16
                )
                self.top_tokens(x)
        torch.cuda.synchronize(self.weight.device)
