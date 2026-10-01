# SPDX-License-Identifier: Apache-2.0
"""Prefill fusions with FP32 residuals and FP16 activation rounding."""

from orin_jit import orin_jit
import tilelang.language as T


@orin_jit
def gemma_norm(residual_dtype):
    M, K, threads = 8192, 5120, 256

    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), T.float16),
        R: T.Tensor((M, K), residual_dtype),
        W: T.Tensor((K,), T.float16),
        Y: T.Tensor((M, K), T.float16),
        RO: T.Tensor((M, K), T.float32),
    ):
        with T.Kernel(M, threads=threads) as row:
            x = T.alloc_fragment((K,), T.float32)
            square = T.alloc_fragment((K,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(K):
                x[j] = T.cast(X[row, j], T.float32) + T.cast(R[row, j], T.float32)
                RO[row, j] = x[j]
                square[j] = x[j] * x[j]
            T.reduce_sum(square, total, dim=0)
            for j in T.Parallel(K):
                Y[row, j] = (x[j] * T.rsqrt(total[0] / K + 1e-6)) * (
                    T.cast(W[j], T.float32) + 1
                )

    return kernel


@orin_jit
def gemma_norm_no_residual():
    M, K, threads = 8192, 5120, 256

    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), T.float16),
        W: T.Tensor((K,), T.float16),
        Y: T.Tensor((M, K), T.float16),
    ):
        with T.Kernel(M, threads=threads) as row:
            x = T.alloc_fragment((K,), T.float32)
            square = T.alloc_fragment((K,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(K):
                x[j] = X[row, j]
                square[j] = x[j] * x[j]
            T.reduce_sum(square, total, dim=0)
            for j in T.Parallel(K):
                Y[row, j] = (x[j] * T.rsqrt(total[0] / K + 1e-6)) * (
                    T.cast(W[j], T.float32) + 1
                )

    return kernel


@orin_jit
def silu_mul_q8():
    M, K, threads = 8192, 17408, 256

    @T.prim_func
    def kernel(
        X: T.Tensor((M, K * 2), T.float16),
        Q: T.Tensor((M, K), T.int8),
        S: T.Tensor((M,), T.float16),
    ):
        with T.Kernel(M, threads=threads) as row:
            x = T.alloc_fragment((K,), T.float32)
            absolute = T.alloc_fragment((K,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            scale = T.alloc_fragment((1,), T.float16)
            for j in T.Parallel(K):
                gate = T.cast(X[row, j], T.float32)
                up = T.cast(X[row, K + j], T.float32)
                # Native CUDA rounds SiLU to scalar_t before multiplying up,
                # then writes FP16 before subsequent A8 quantization.
                activated = T.cast(
                    T.cast(gate / (1 + T.exp(-gate)), T.float16), T.float32
                )
                x[j] = T.cast(T.cast(activated * up, T.float16), T.float32)
                absolute[j] = T.abs(x[j])
            T.reduce_max(absolute, maximum, dim=0)
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


@orin_jit
def gdn_norm_q8():
    M, H, D, threads = 8192, 48, 128, 256

    @T.prim_func
    def kernel(
        X: T.Tensor((M, H * D), T.float16),
        QKVZ: T.Tensor((M, 16384), T.float16),
        W: T.Tensor((D,), T.float16),
        Q: T.Tensor((M, H * D), T.int8),
        S: T.Tensor((M,), T.float16),
    ):
        with T.Kernel(M, threads=threads) as row:
            x = T.alloc_fragment((H, D), T.float32)
            square = T.alloc_fragment((H, D), T.float32)
            absolute = T.alloc_fragment((H, D), T.float32)
            total = T.alloc_fragment((H,), T.float32)
            head_max = T.alloc_fragment((H,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            scale = T.alloc_fragment((1,), T.float16)
            for h, j in T.Parallel(H, D):
                x[h, j] = X[row, h * D + j]
                square[h, j] = x[h, j] * x[h, j]
            T.reduce_sum(square, total, dim=1)
            for h, j in T.Parallel(H, D):
                z = T.cast(QKVZ[row, 10240 + h * D + j], T.float32)
                normalized = (x[h, j] * T.rsqrt(total[h] / D + 1e-6)) * T.cast(
                    W[j], T.float32
                )
                x[h, j] = T.cast(
                    T.cast(normalized * (z / (1 + T.exp(-z))), T.float16), T.float32
                )
                absolute[h, j] = T.abs(x[h, j])
            T.reduce_max(absolute, head_max, dim=1)
            T.reduce_max(head_max, maximum, dim=0)
            scale[0] = T.if_then_else(
                maximum[0] > 0, T.max(maximum[0] / 127, 2**-24), 1
            )
            S[row] = scale[0]
            for h, j in T.Parallel(H, D):
                Q[row, h * D + j] = T.cast(
                    T.max(
                        -127.0,
                        T.min(127.0, T.round(x[h, j] / T.cast(scale[0], T.float32))),
                    ),
                    T.int8,
                )

    return kernel
