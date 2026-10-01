# SPDX-License-Identifier: Apache-2.0
"""Geometry of the validated Qwen3.8-27B single-GPU pipeline."""

MIN_TOKENS = 1024
MAX_TOKENS = 8192

# (output channels, input channels): LUT expansion tile width.
PROJECTIONS = {
    (34816, 5120): 128,  # FFN gate/up
    (5120, 17408): 64,  # FFN down
    (16384, 5120): 64,  # GDN QKV/Z
    (14336, 5120): 128,  # Full attention QKV/gate
    (5120, 6144): 128,  # Attention output
}


def padded_tokens(m):
    return (m + 255) // 256 * 256
