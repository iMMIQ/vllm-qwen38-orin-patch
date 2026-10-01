# SPDX-License-Identifier: Apache-2.0
"""Per-token FP16-to-INT8 activation quantization."""

from orin_jit import orin_jit
import tilelang.language as T


@orin_jit
def activation_q8(M: int, K: int):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        Q: T.Tensor((M, K), T.int8),
        S: T.Tensor((M,), T.float16),
    ):
        with T.Kernel(M, threads=256) as row:
            x = T.alloc_fragment((K,), T.float32)
            absx = T.alloc_fragment((K,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            scale = T.alloc_fragment((1,), T.float16)
            for j in T.Parallel(K):
                x[j] = A[row, j]
                absx[j] = T.abs(x[j])
            T.reduce_max(absx, maximum, dim=0)
            scale[0] = T.if_then_else(
                maximum[0] > 0, T.max(maximum[0] / 127, 2**-24), 1
            )
            S[row] = scale[0]
            for j in T.Parallel(K):
                Q[row, j] = T.cast(
                    T.max(
                        -127.0,
                        T.min(127.0, T.round(x[j] / T.cast(scale[0], T.float32))),
                    ),
                    T.int8,
                )

    return kernel
