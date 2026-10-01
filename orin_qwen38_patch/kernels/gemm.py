# SPDX-License-Identifier: Apache-2.0
"""128-thread, two-stage INT8 GEMM with pairs of M tiles."""

from orin_jit import orin_jit
import tilelang.language as T


@orin_jit
def tiled(N, K):
    M = 8192
    BM = BN = BK = 128
    stages, threads, group = 2, 128, 2
    assert M % BM == N % BN == K % BK == 0 and (M // BM) % group == 0

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.int8),
        B: T.Tensor((N, K), T.int8),
        AS: T.Tensor((M,), T.float16),
        BS: T.Tensor((N,), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel((N // BN) * group, M // BM // group, threads=threads) as (gx, gy):
            T.annotate_min_blocks_per_sm(1)
            bx = gx // group
            by = gy * group + gx % group
            a = T.alloc_shared((BM, BK), T.int8)
            b = T.alloc_shared((BN, BK), T.int8)
            accum = T.alloc_fragment((BM, BN), T.int32)
            T.clear(accum)
            for ko in T.Pipelined(K // BK, num_stages=stages):
                T.copy(A[by * BM, ko * BK], a)
                T.copy(B[bx * BN, ko * BK], b)
                T.gemm(a, b, accum, transpose_B=True)
            for i, j in T.Parallel(BM, BN):
                C[by * BM + i, bx * BN + j] = (
                    T.cast(accum[i, j], T.float32)
                    * T.cast(AS[by * BM + i], T.float32)
                    * T.cast(BS[bx * BN + j], T.float32)
                )

    return kernel
