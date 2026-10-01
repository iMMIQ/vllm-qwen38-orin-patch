# SPDX-License-Identifier: Apache-2.0
"""Marlin W4 row scales and exact LUT expansion into temporary W8."""

from orin_jit import orin_jit
import tilelang.language as T


@orin_jit
def weight_stats(N: int, K: int):
    @T.prim_func
    def kernel(
        P: T.Tensor((K // 16, N * 2), T.int32),
        S: T.Tensor((K // 128, N), T.float16),
        Z: T.Tensor((K // 128, N // 8), T.int32),
        BS: T.Tensor((N,), T.float16),
    ):
        with T.Kernel(N, threads=256) as row:
            absolute = T.alloc_fragment((K,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            for col in T.Parallel(K):
                lane = (row % 8) * 4 + (col % 8) // 2
                word = P[col // 16, (row // 64) * 128 + lane * 4 + (row % 64) // 16]
                code = (
                    word
                    >> ((col % 2) * 16 + ((col % 16) // 8) * 4 + ((row % 16) // 8) * 8)
                ) & 15
                perm = (row % 8) * 8 + (row % 64) // 8
                zpperm = (perm // 8) * 8 + (perm % 2) * 4 + (perm % 8) // 2
                zero = (
                    Z[col // 128, (row // 64) * 8 + zpperm // 8] >> ((zpperm % 8) * 4)
                ) & 15
                scale = S[col // 128, (row // 64) * 64 + perm]
                value = T.cast(T.cast(code - zero, T.float16) * scale, T.float32)
                absolute[col] = T.abs(value)
            T.reduce_max(absolute, maximum, dim=0)
            BS[row] = T.max(maximum[0] / 127, 2**-24)

    return kernel


@orin_jit
def lut_expand(N, K):
    """Build a tiny per-CTA exact LUT; evaluate rounding 8x fewer times."""
    BN = 64 if (N, K) in {(5120, 17408), (16384, 5120)} else 128
    BK = 256
    assert BN % 64 == 0 and BK % 128 == 0 and K % BK == 0

    @T.prim_func
    def kernel(
        P: T.Tensor((K // 16, N * 2), T.int32),
        S: T.Tensor((K // 128, N), T.float16),
        Z: T.Tensor((K // 128, N // 8), T.int32),
        BS: T.Tensor((N,), T.float16),
        B: T.Tensor((N, K), T.int8),
    ):
        with T.Kernel(N // BN, K // BK, threads=256) as (bx, bk):
            packed = T.alloc_shared((BK // 16, BN * 2), T.int32)
            scale = T.alloc_shared((BK // 128, BN), T.float16)
            zero = T.alloc_shared((BK // 128, BN // 8), T.int32)
            row_scale = T.alloc_shared((BN,), T.float16)
            table = T.alloc_shared((BK // 128, BN, 4), T.int32)
            T.copy(P[bk * BK // 16, bx * BN * 2], packed)
            T.copy(S[bk * BK // 128, bx * BN], scale)
            T.copy(Z[bk * BK // 128, bx * BN // 8], zero)
            T.copy(BS[bx * BN], row_scale)
            for g, i, word_idx in T.Parallel(BK // 128, BN, 4):
                perm = (i % 8) * 8 + (i % 64) // 8
                zpperm = (perm // 8) * 8 + (perm % 2) * 4 + (perm % 8) // 2
                zp = (zero[g, (i // 64) * 8 + zpperm // 8] >> ((zpperm % 8) * 4)) & 15
                word = T.alloc_var(T.int32)
                word = 0
                for c in T.unroll(4):
                    value = T.cast(
                        T.cast(word_idx * 4 + c - zp, T.float16)
                        * scale[g, (i // 64) * 64 + perm],
                        T.float32,
                    )
                    code8 = T.cast(
                        T.min(
                            127.0,
                            T.max(
                                -127.0, T.round(value / T.cast(row_scale[i], T.float32))
                            ),
                        ),
                        T.int32,
                    )
                    word = word | ((code8 & 255) << (c * 8))
                table[g, i, word_idx] = word
            for i, j in T.Parallel(BN, BK):
                lane = (i % 8) * 4 + (j % 8) // 2
                word4 = packed[j // 16, (i // 64) * 128 + lane * 4 + (i % 64) // 16]
                code4 = (
                    word4 >> ((j % 2) * 16 + ((j % 16) // 8) * 4 + ((i % 16) // 8) * 8)
                ) & 15
                word8 = table[j // 128, i, code4 // 4]
                B[bx * BN + i, bk * BK + j] = T.cast(
                    (word8 >> ((code4 % 4) * 8)) & 255, T.int8
                )

    return kernel
