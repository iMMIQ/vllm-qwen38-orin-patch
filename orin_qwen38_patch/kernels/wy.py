# SPDX-License-Identifier: Apache-2.0
"""Single-sequence Qwen GDN WY transform, head-first blocks and one FP16 MMA."""

from orin_jit import orin_jit
import tilelang.language as T


@orin_jit
def wy():
    M = 8192

    @T.prim_func
    def kernel(
        K: T.Tensor((M, 16, 128), T.float16),
        V: T.Tensor((M, 48, 128), T.float16),
        BETA: T.Tensor((M, 48), T.float32),
        G: T.Tensor((M, 48), T.float32),
        A: T.Tensor((M, 48, 64), T.float16),
        SEQ: T.Tensor((2,), T.int32),
        W: T.Tensor((M, 48, 128), T.float16),
        U: T.Tensor((M, 48, 128), T.float16),
    ):
        with T.Kernel(48, M // 64, threads=256) as (h, chunk):
            a = T.alloc_shared((64, 64), T.float16)
            b = T.alloc_shared((64, 256), T.float16)
            beta = T.alloc_shared((64,), T.float32)
            gate = T.alloc_shared((64,), T.float32)
            c = T.alloc_fragment((64, 256), T.float32)
            for i in T.Parallel(64):
                beta[i] = T.if_then_else(
                    chunk * 64 + i < SEQ[1], BETA[chunk * 64 + i, h], 0
                )
                gate[i] = T.if_then_else(
                    chunk * 64 + i < SEQ[1],
                    T.exp2(G[chunk * 64 + i, h] * 1.4426950408889634),
                    0,
                )
            for i, j in T.Parallel(64, 64):
                a[i, j] = T.if_then_else(
                    chunk * 64 + i < SEQ[1], A[chunk * 64 + i, h, j], 0
                )
            for i, j in T.Parallel(64, 256):
                if chunk * 64 + i < SEQ[1]:
                    if j < 128:
                        b[i, j] = T.cast(V[chunk * 64 + i, h, j], T.float32) * beta[i]
                    else:
                        b[i, j] = (
                            T.cast(K[chunk * 64 + i, h // 3, j - 128], T.float32)
                            * beta[i]
                        ) * gate[i]
                else:
                    b[i, j] = 0
            T.clear(c)
            T.gemm(a, b, c)
            for i, j in T.Parallel(64, 256):
                if chunk * 64 + i < SEQ[1]:
                    if j < 128:
                        U[chunk * 64 + i, h, j] = c[i, j]
                    else:
                        W[chunk * 64 + i, h, j - 128] = c[i, j]

    return kernel
