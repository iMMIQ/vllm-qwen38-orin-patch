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


def request_buckets(max_num_seqs):
    """Power-of-two capacities plus the configured final, possibly odd capacity."""
    if max_num_seqs < 1:
        raise ValueError("max_num_seqs must be positive")
    buckets = []
    size = 1
    while size < max_num_seqs:
        buckets.append(size)
        size *= 2
    return [*buckets, max_num_seqs]


def decode_capture_sizes(max_num_seqs, draft_tokens, token_budget):
    """Capture verification batches through the scheduler's configured capacity."""
    stride = draft_tokens + 1
    capacity = min(max_num_seqs, token_budget // stride)
    if capacity < 1:
        raise ValueError("Token budget must accommodate one decode request")
    return [stride * size for size in request_buckets(capacity)]
